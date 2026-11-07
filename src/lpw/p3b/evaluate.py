"""P3b readout statistics (reports/p3b_prereg.md §P8-§P10): group folds, group-aware C2ST, grouped F1 threshold,
incremental-transfer contrasts, and half-view coupling-fidelity distances."""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from lpw.p3.utility import PRIMARY, SEED, make

MIN_VALID_DRAWS = 950
R_FLOOR = 0.020
KAPPA = 0.8


def group_folds(groups: np.ndarray, folds: int = 5, seed: int = SEED) -> np.ndarray:
    """Whole-group folds: groups largest-first (ties by a seed-0 permutation), each into the currently
    least-populated fold (ties -> lowest fold index)."""
    ug, inv, size = np.unique(groups, return_inverse=True, return_counts=True)
    perm_rank = np.empty(len(ug), np.int64)
    perm_rank[np.random.default_rng(seed).permutation(len(ug))] = np.arange(len(ug))
    order = np.lexsort((perm_rank, -size))
    load = np.zeros(folds, np.int64)
    gfold = np.empty(len(ug), np.int64)
    for g in order:
        f = int(np.argmin(load))
        gfold[g] = f
        load[f] += size[g]
    return gfold[inv]


def f1_threshold_grouped(name, X, y, groups) -> float:
    """F1-maximizing threshold over every distinct out-of-fold score of the TRAINING source, five whole-group folds
    (group_folds); ties -> the highest threshold."""
    fold = group_folds(groups)
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        oof[te] = make(name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    p, r, t = precision_recall_curve(y, oof)
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12), 0.0)
    return float(t[np.flatnonzero(f1 == f1.max())[-1]])


def c2st_grouped(X_real, g_real, X_syn, g_syn, seed: int = SEED, folds: int = 5):
    """Real-vs-synthetic OOF AUC of the primary detector. Sources balanced by seed-0 subsampling; five group-disjoint
    folds on both sides (group_folds on each side); within each training fold each domain carries total weight half
    the fold's rows. None (NOT TESTABLE) if some fold lacks a domain."""
    rng = np.random.default_rng(seed)
    n = min(len(X_real), len(X_syn))
    ri = np.sort(rng.choice(len(X_real), n, replace=False))
    si = np.sort(rng.choice(len(X_syn), n, replace=False))
    fold = np.concatenate([group_folds(g_real[ri], folds), group_folds(g_syn[si], folds)])
    dom = np.r_[np.zeros(n), np.ones(n)]
    X = np.concatenate([X_real[ri], X_syn[si]])
    for f in range(folds):
        for d in (0, 1):
            if not ((fold == f) & (dom == d)).any() or not ((fold != f) & (dom == d)).any():
                return None
    oof = np.zeros(2 * n)
    for f in range(folds):
        tr, te = fold != f, fold == f
        w = np.zeros(tr.sum())
        dtr = dom[tr]
        for d in (0, 1):
            w[dtr == d] = (tr.sum() / 2) / (dtr == d).sum()
        oof[te] = make(PRIMARY, seed).fit(X[tr], dom[tr], sample_weight=w).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(dom, oof))


def _auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def incremental(y, draws, trtr_L, trtr_LS, tstr_L: list, tstr_LS_by_arm: dict) -> dict:
    """§P9 A/B: R, T_j, D = T_1 - 0.8 R, H = T_1 - T_0 from fixed predictions, with whole-group bootstrap draws
    (index arrays). tstr_LS_by_arm: {'p3b0': [per world], 'p3b1': [...]}."""
    def stats(i):
        aL, aLS = _auc(y[i], trtr_L[i]), _auc(y[i], trtr_LS[i])
        if aL is None or aLS is None:
            return None
        R = aLS - aL
        T = {}
        for j, lst in tstr_LS_by_arm.items():
            d = []
            for sL, sLS in zip(tstr_L, lst):
                a, b = _auc(y[i], sLS[i]), _auc(y[i], sL[i])
                if a is None or b is None:
                    return None
                d.append(a - b)
            T[j] = float(np.mean(d))
        return {"R": R, "T0": T["p3b0"], "T1": T["p3b1"], "D": T["p3b1"] - KAPPA * R, "H": T["p3b1"] - T["p3b0"]}

    full = stats(np.arange(len(y)))
    bs = [b for b in (stats(i) for i in draws) if b is not None]
    out = {"point": full, "n_valid_draws": len(bs), "n_draws": len(draws)}
    if len(bs) < MIN_VALID_DRAWS:
        out.update(utility="NOT TESTABLE (bootstrap)", learning="NOT TESTABLE (bootstrap)")
        return out
    q = {k: np.array([b[k] for b in bs]) for k in full}
    out["ci95"] = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in q.items()}
    out["lower_1s95"] = {k: float(np.quantile(v, 0.05)) for k, v in q.items()}
    R, D, H = full["R"], full["D"], full["H"]
    lo = out["lower_1s95"]
    if R < R_FLOOR:
        out["utility"] = "NOT TESTABLE"
    elif D < 0:
        out["utility"] = "NOT SUPPORTED"
    else:
        out["utility"] = "SUPPORTED" if (lo["D"] >= 0 and lo["R"] > 0) else "CONSISTENT"
    out["learning"] = ("SUPPORTED" if lo["H"] > 0 else "CONSISTENT") if H > 0 else "NOT SUPPORTED"
    return out


def point_incremental(y, trtr_L, trtr_LS, tstr_L, tstr_LS_by_arm, idx=None) -> dict | None:
    """Point estimates of R, T_j, D, H on an index subset (secondary analyses)."""
    i = np.arange(len(y)) if idx is None else np.asarray(idx)
    return incremental(y[i], [], trtr_L[i], trtr_LS[i], [s[i] for s in tstr_L],
                       {k: [s[i] for s in v] for k, v in tstr_LS_by_arm.items()})["point"]
