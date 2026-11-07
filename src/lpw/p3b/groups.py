"""Conditional group prior (reports/p3b_prereg.md §P3): a Poissonized empirical multi-group prior with singleton
completion, conditional on a saved population's role totals.

From real train, count templates (s, k) = (group size, scam count) with s >= 2. For a population with N agents and
role totals (N0, N1), draw M[s, k] ~ Poisson(N / N_train * c[s, k]) for every template, reject the whole draw if it
needs more than N0 normal or N1 scam slots (at most 1,000 proposals from one stream; exhaustion is a technical
failure), fill the rest with role-specific singletons, and place the agents uniformly within role (an independent
permutation per role). Slots are laid out canonically: templates in sorted (s, k) order, copies in order, each
group's normal slots then its scam slots, then normal singletons, then scam singletons.
"""
import numpy as np

MAX_PROPOSALS = 1000


def templates(group: np.ndarray, role: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(keys (T, 2) sorted (s, k) with s >= 2, counts (T,), N_train) from real train group ids and roles."""
    g, inv = np.unique(group, return_inverse=True)
    size = np.bincount(inv)
    scam = np.bincount(inv, weights=role).astype(np.int64)
    m = size >= 2
    keys, counts = np.unique(np.stack([size[m], scam[m]], 1), axis=0, return_counts=True)
    return keys.astype(np.int64), counts.astype(np.int64), int(len(group))


def sample(role: np.ndarray, keys: np.ndarray, counts: np.ndarray, n_train: int, seed: int) -> tuple[np.ndarray, dict]:
    """(group id per agent of `role`, info). Multi-member groups get ids 0..G-1 in slot order, then singletons."""
    role = np.asarray(role, np.int64)
    N, N1 = len(role), int(role.sum())
    N0 = N - N1
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    lam = N / n_train * counts
    s, k = keys[:, 0], keys[:, 1]
    for t in range(MAX_PROPOSALS):
        M = rng.poisson(lam)
        need0, need1 = int((M * (s - k)).sum()), int((M * k).sum())
        if need0 <= N0 and need1 <= N1:
            break
    else:
        raise RuntimeError("group prior: proposal limit exhausted (technical failure)")
    slot_role, slot_group = [], []
    gid = 0
    for (si, ki), mi in zip(keys, M):
        for _ in range(int(mi)):
            slot_role += [0] * int(si - ki) + [1] * int(ki)
            slot_group += [gid] * int(si)
            gid += 1
    n_multi = gid
    for r, n_r in ((0, N0 - need0), (1, N1 - need1)):
        slot_role += [r] * n_r
        slot_group += list(range(gid, gid + n_r))
        gid += n_r
    slot_role, slot_group = np.array(slot_role), np.array(slot_group)
    out = np.full(N, -1, np.int64)
    for r in (0, 1):
        agents = np.flatnonzero(role == r)
        perm = np.random.default_rng(np.random.SeedSequence([seed, 1, r])).permutation(agents)
        out[perm] = slot_group[slot_role == r]
    assert (out >= 0).all()
    info = {"proposals": t + 1, "n_multi_groups": n_multi, "agents_in_multi": need0 + need1,
            "singletons_normal": N0 - need0, "singletons_scam": N1 - need1, "max_group": int(np.bincount(out).max())}
    return out, info
