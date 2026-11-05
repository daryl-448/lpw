"""Per-agent plan latent s = (owned-count bucket, amount-level bucket) — Phase 1 drift experiment.

Diagnosis (runs/p1_diag/drift_brief.md and the two brainstorm reports): in free-running generation the policy
re-infers each agent's slow levels (activity rate, amount level) from its own sampled events, over-weighting the
latest ones, so a fixed level becomes a random walk. s is a persistent anchor fed through the PROFILE token:
- training: s is computed from the agent's whole training-window stream (like P2's z) and dropped at random
  (P_DROP) to a learned "unknown" index, so the same network also models s-unknown;
- generation: s is either the agent's real s (ORACLE, an upper-bound diagnostic that uses the future) or drawn
  from a prior p(s | role, cohort, profile) fitted on the training split;
- NLL evaluation uses s = unknown, a proper model of the data without future information.
"""

import numpy as np
import torch

from lpw.p1.store import IN_AGENT

COUNT_EDGES = np.array([0, 1, 2, 3, 5, 10, 30, 100, 300, 1000])  # bucket i is [edge_i, edge_{i+1}); last is >= 1000
N_PLAN_N = len(COUNT_EDGES)          # 10 count buckets; index N_PLAN_N = unknown
N_PLAN_A = 6                          # 5 train-quintile amount buckets + "no owned events"; index N_PLAN_A = unknown
P_DROP = 0.2


# plan progress (lpw_plan_prog): where the agent's owned-event count so far, n, sits in its count bucket [lo, hi)
PROG_BELOW, PROG_ROOM, PROG_DONE, PROG_UNKNOWN = 0, 1, 2, 3  # n < lo | lo <= n < hi - 1 | n >= hi - 1 | s unknown
N_PROG = 4


def progress_state(plan_n: torch.Tensor, n_own: torch.Tensor) -> torch.Tensor:
    """Per row: PROG_BELOW (the plan needs more owned events), PROG_ROOM (it allows more), PROG_DONE (a further owned
    event would leave the bucket; with the real plan this only happens when no further owned event follows), or
    PROG_UNKNOWN when s is unknown. The last bucket is open-ended, so it is never DONE. plan_n is the effective
    (dropped / unknown-masked) plan of each row's agent; n_own counts owned events before the row's target, from the
    start of the stream (not the context window)."""
    edges = torch.as_tensor(np.append(COUNT_EDGES, np.iinfo(np.int64).max), device=plan_n.device)
    unknown = plan_n == N_PLAN_N
    k = plan_n.clamp(max=N_PLAN_N - 1)
    lo, hi = edges[k], edges[k + 1]
    st = torch.where(n_own < lo, PROG_BELOW, torch.where(n_own >= hi - 1, PROG_DONE, PROG_ROOM))
    return torch.where(unknown, PROG_UNKNOWN, st)


def count_bucket(n: np.ndarray) -> np.ndarray:
    return np.searchsorted(COUNT_EDGES, np.asarray(n), side="right") - 1


def plan_targets(store) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(plan_n, plan_a, amount quintile edges) for every agent, from its owned events in the window. The amount
    quintile edges come from training-split agents only."""
    A = store.n_agents
    n_own = np.zeros(A, np.int64)
    med = np.full(A, np.nan)
    for a in range(A):
        s, e = store.off[a], store.off[a + 1]
        own = store.kind[s:e] != IN_AGENT
        n_own[a] = own.sum()
        if n_own[a]:
            med[a] = float(np.median(store.amt[s:e][own]))
    tr = (store.split == "train") & np.isfinite(med)
    edges = np.quantile(med[tr], [0.2, 0.4, 0.6, 0.8])
    pa = np.where(np.isfinite(med), np.searchsorted(edges, np.nan_to_num(med), side="right"), N_PLAN_A - 1)
    return count_bucket(n_own), pa.astype(np.int64), edges


class PlanPrior:
    """p(s | role, cohort, profile): multinomial logistic regression over the joint (count, amount) bucket, fitted
    on training-split agents. Joint, because count and amount level are correlated."""

    def __init__(self, store, seed=0):
        from sklearn.linear_model import LogisticRegression
        tr = store.agents("train")
        X = self._x(store.role[tr], store.cohort[tr], np.stack([store.prof_vec(a) for a in tr]))
        y = store.plan_n[tr] * N_PLAN_A + store.plan_a[tr]
        self.clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(X, y)
        va = store.agents("val")
        Xv = self._x(store.role[va], store.cohort[va], np.stack([store.prof_vec(a) for a in va]))
        p = self.clf.predict_proba(Xv)
        yv = store.plan_n[va] * N_PLAN_A + store.plan_a[va]
        col = {c: i for i, c in enumerate(self.clf.classes_)}
        self.val_nll = float(-np.mean([np.log(max(p[i, col[c]], 1e-12)) if c in col else np.log(1e-12) for i, c in enumerate(yv)]))

    @staticmethod
    def _x(role, cohort, prof):
        return np.concatenate([np.asarray(role)[:, None], np.asarray(cohort)[:, None], prof], 1).astype(np.float64)

    def sample(self, role, cohort, prof, rng) -> tuple[np.ndarray, np.ndarray]:
        p = self.clf.predict_proba(self._x(role, cohort, prof))
        u = rng.random(len(p))[:, None]
        k = self.clf.classes_[np.minimum((p.cumsum(1) < u).sum(1), p.shape[1] - 1)]
        return k // N_PLAN_A, k % N_PLAN_A
