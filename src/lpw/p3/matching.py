"""Matched evaluation on common support (SPEC §7.5; reports/p3a_prereg.md §6).

The propensity model for role is fitted on REAL TRAIN agents only, on the confounders: age, the lifetime-count and
volume proxies of the profile, cohort (= profile missingness: every pre-existing agent has a profile), and the
activation time within W (era inside the slice). Support bounds and caliper are frozen from train:
- support: [max(min scam logit, min normal logit), min(max scam, max normal)] on train;
- caliper: 0.2 x the SD of the train logit.
Evaluation agents are matched 1:1 without replacement inside the support, exactly on cohort and then by greedy
nearest neighbour on the logit within the caliper, highest-propensity scams first. Fewer than 200 pairs means
"not testable".
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

PROP = ["p_age_days", "p_days_active", "p_days_since_active", "p_normal_sent_cnt", "p_normal_recv_cnt", "p_erc20_sent_cnt",
        "p_erc20_recv_cnt", "p_eth_sent_sum", "p_eth_recv_sum", "cohort", "act_frac"]
MIN_PAIRS = 200


class Matcher:
    def __init__(self, F_train: pl.DataFrame, y_train: np.ndarray):
        X = F_train.select(PROP).to_numpy()
        self.lr = LogisticRegression(max_iter=5000).fit(X, y_train)
        lg = self.lr.decision_function(X)
        self.caliper = float(0.2 * lg.std())
        s, n = lg[y_train == 1], lg[y_train == 0]
        self.lo, self.hi = float(max(s.min(), n.min())), float(min(s.max(), n.max()))

    def logit(self, F: pl.DataFrame) -> np.ndarray:
        return self.lr.decision_function(F.select(PROP).to_numpy())

    def match(self, F: pl.DataFrame, y: np.ndarray) -> dict:
        lg = self.logit(F)
        ins = (lg >= self.lo) & (lg <= self.hi)
        coh = F["cohort"].to_numpy()
        pairs = []
        for c in (0, 1):  # exact on cohort, then nearest neighbour on the logit within the caliper
            si = np.flatnonzero(ins & (y == 1) & (coh == c))
            ni = np.flatnonzero(ins & (y == 0) & (coh == c))
            free = np.ones(len(ni), bool)
            for s_ in si[np.argsort(-lg[si], kind="stable")]:  # hardest (highest-propensity) scams first
                d = np.abs(lg[ni] - lg[s_])
                d[~free] = np.inf
                k = int(np.argmin(d)) if len(d) else -1
                if k >= 0 and d[k] <= self.caliper:
                    pairs.append((int(s_), int(ni[k]))); free[k] = False
        pairs = np.array(pairs, np.int64).reshape(-1, 2)
        return {"pairs": pairs, "n_pairs": len(pairs), "testable": len(pairs) >= MIN_PAIRS,
                "support_scam": int((ins & (y == 1)).sum()), "support_normal": int((ins & (y == 0)).sum()), "n_scam": int((y == 1).sum()),
                "balance": self.balance(F, y, pairs)}

    @staticmethod
    def balance(F, y, pairs) -> dict:
        """standardized mean difference (scam - normal) / pooled SD of each confounder, before and after matching"""
        out = {}
        for c in PROP:
            v = F[c].to_numpy()
            sd = np.sqrt((v[y == 1].var() + v[y == 0].var()) / 2) + 1e-12
            out[c] = {"before": float((v[y == 1].mean() - v[y == 0].mean()) / sd),
                      "after": float((v[pairs[:, 0]].mean() - v[pairs[:, 1]].mean()) / sd) if len(pairs) else float("nan")}
        return out
