"""P3a detectors and criticism tests (SPEC §7.4; reports/p3a_prereg.md §5-§8). Settings are frozen here.

All detectors are node-level role classifiers on lpw.p3.features sets. Training sources: real train (TRTR) or one
synthetic train world (TSTR). Evaluation: real val (development; the sealed test is opened once at the end).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

DETECTORS = ("lr", "rf", "xgb", "lgbm")
PRIMARY = "xgb"
SEED = 0


def make(name: str, seed: int = SEED):
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=8, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=8, eval_metric="logloss")
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, n_jobs=8, verbose=-1)
    raise KeyError(name)


def fit_score(name, Xtr, ytr, Xte_list):
    m = make(name).fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xte_list]


def f1_threshold(name, Xtr, ytr) -> float:
    """The score threshold maximizing F1 on 5-fold out-of-fold predictions of the TRAINING source (never the
    evaluation data), over every distinct OOF score; ties go to the highest threshold."""
    oof = np.zeros(len(ytr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr):
        oof[b] = make(name).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(ytr, oof)  # every distinct OOF score is a candidate threshold
    f = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12), 0.0)
    best = np.flatnonzero(f == f.max())
    return float(t[best[-1]])  # ties: the highest threshold


def scores(y, s, thr=None) -> dict:
    out = {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}
    if thr is not None:
        out["f1"] = float(f1_score(y, s >= thr))
    return out


class GroupBoot:
    """Whole-group bootstrap over real evaluation agents: the same B resampled group sets for every quantity."""

    def __init__(self, groups: np.ndarray, B: int = 1000, seed: int = 0):
        ug, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(ug))]
        if not len(ug):  # an empty evaluation pool: B empty draws, every statistic over them invalid
            self.idx = [np.array([], np.int64)] * B
            return
        draws = np.random.default_rng(seed).integers(0, len(ug), (B, len(ug)))
        self.idx = [np.concatenate([members[g] for g in d]) for d in draws]

    def values(self, fn) -> np.ndarray:
        """fn over every draw, keeping the valid (non-None, finite) values; len() is the valid-draw count."""
        return np.array([x for x in (fn(i) for i in self.idx) if x is not None and np.isfinite(x)], np.float64)

    def ci(self, fn) -> tuple[float, float]:
        v = self.values(fn)
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    def lower(self, fn, q=0.05) -> float:
        return float(np.quantile(self.values(fn), q))


MIN_VALID_FRAC = 0.95  # joint addendum §J4.4: >= 950 of the frozen 1000 draws must be valid two-role draws, else INVALID


def auc_or_none(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def transfer_contrast(y, tstr_scores: list[np.ndarray], trtr_score: np.ndarray, boot: GroupBoot, kappa=0.8) -> dict:
    """Excess-AUC ratio (mean_w TSTR_w - 0.5) / (TRTR - 0.5) and the paired contrast
    D = (mean_w TSTR_w - 0.5) - kappa (TRTR - 0.5), with a group-bootstrap CI over real evaluation groups (evaluation
    uncertainty only: 3 worlds cannot estimate world-to-world variability, which is reported as the spread)."""
    n_draws, min_scam = len(boot.idx), min_scam_per_draw(y, boot)
    if not 0 < y.sum() < len(y):  # a one-role or empty evaluation set (a subset): no contrast is defined
        return {"outcome": "NOT TESTABLE", "reason": "one role or empty", "n": int(len(y)), "n_scam": int(y.sum()), "D": None,
                "D_ci95": None, "D_lower_1s95": None, "ratio": None, "n_valid_draws": 0, "n_draws": n_draws, "min_scam_per_draw": min_scam}

    def D(i):
        a = [auc_or_none(y[i], s[i]) for s in tstr_scores]
        r = auc_or_none(y[i], trtr_score[i])
        return None if r is None or None in a else (np.mean(a) - 0.5) - kappa * (r - 0.5)
    ts = [roc_auc_score(y, s) for s in tstr_scores]
    tr = roc_auc_score(y, trtr_score)
    d = (np.mean(ts) - 0.5) - kappa * (tr - 0.5)
    v = boot.values(D)
    valid = len(v) >= MIN_VALID_FRAC * len(boot.idx)
    lo95 = float(np.quantile(v, 0.05)) if valid else None
    ci95 = (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if valid else None
    if tr < 0.60:  # prereg §7 gate: no inferential statistic below it
        outcome = "NOT TESTABLE"
    elif not valid:
        outcome = "INVALID"
    else:
        outcome = ("SUPPORTED" if lo95 >= 0 else "CONSISTENT") if d >= 0 else "NOT SUPPORTED"
    return {"tstr_auc_worlds": [float(x) for x in ts], "tstr_auc_mean": float(np.mean(ts)), "tstr_auc_sd": float(np.std(ts)),
            "trtr_auc": float(tr), "ratio": float((np.mean(ts) - 0.5) / (tr - 0.5)) if tr >= 0.60 else None, "D": float(d), "D_ci95": ci95,
            "D_lower_1s95": lo95, "outcome": outcome, "trtr_gate": bool(tr >= 0.60), "n_valid_draws": int(len(v)),
            "n_draws": n_draws, "min_scam_per_draw": min_scam}


def min_scam_per_draw(y, boot_or_draws) -> int | None:
    """The smallest scam count in any bootstrap draw (joint addendum §J4.4); None without draws."""
    idx = boot_or_draws.idx if isinstance(boot_or_draws, GroupBoot) else boot_or_draws
    return int(min(y[i].sum() for i in idx)) if len(idx) else None


def contrast_line(h: dict) -> str:
    """One readout line for a transfer contrast, also when its interval is absent (INVALID, NOT TESTABLE)."""
    if h.get("tstr_auc_worlds") is None:
        return f"-> {h['outcome']} ({h.get('reason')})"
    ci = "n/a" if h["D_ci95"] is None else f"[{h['D_ci95'][0]:+.3f},{h['D_ci95'][1]:+.3f}]"
    return (f"TSTR {h['tstr_auc_mean']:.3f} ± {h['tstr_auc_sd']:.3f} (worlds {np.round(h['tstr_auc_worlds'], 3).tolist()}) "
            f"vs TRTR {h['trtr_auc']:.3f}; ratio {h['ratio']}; D {h['D']:+.3f} CI95 {ci} -> {h['outcome']} (TRTR gate {h['trtr_gate']}; "
            f"valid draws {h['n_valid_draws']}/{h['n_draws']}, min scam per draw {h['min_scam_per_draw']})")


def c2st(X_real, X_syn, real_groups, seed=SEED, folds=5) -> float:
    """Real-vs-synthetic AUC of the primary detector, balanced by subsampling the larger side, 5-fold out-of-fold.
    Real agents are folded by group (no group in both training and scoring folds); synthetic agents at random, with
    the same per-fold counts as the real side."""
    rng = np.random.default_rng(seed)
    n = min(len(X_real), len(X_syn))
    ri = np.sort(rng.choice(len(X_real), n, replace=False)); si = np.sort(rng.choice(len(X_syn), n, replace=False))
    ug, ginv = np.unique(real_groups[ri], return_inverse=True)
    rfold = (rng.permutation(len(ug)) % folds)[ginv]
    # synthetic fold sizes copy the real ones, so every fold (and its complement) is exactly balanced
    sfold = rng.permutation(np.repeat(np.arange(folds), np.bincount(rfold, minlength=folds)))
    fold = np.concatenate([rfold, sfold])
    X = np.concatenate([X_real[ri], X_syn[si]]); y = np.r_[np.zeros(n), np.ones(n)]
    oof = np.zeros(2 * n)
    for f in range(folds):
        tr, te = fold != f, fold == f
        oof[te] = make(PRIMARY, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def single_auc(y, v) -> float:
    return float(roc_auc_score(y, v)) if v.std() > 0 else 0.5


def shortcut_audit(names, X_train_real, y_train_real, X_real, y_real, X_syn, y_syn, sentinels, top=20, flag=0.1) -> dict:
    """§7.4: single-feature role AUC on synthetic minus on real (evaluation split), for the top features ranked by
    |AUC - 0.5| on REAL TRAIN, plus the sentinels whatever their rank."""
    rank = sorted(range(len(names)), key=lambda i: -abs(single_auc(y_train_real, X_train_real[:, i]) - 0.5))
    sel = list(dict.fromkeys([names[i] for i in rank[:top]] + list(sentinels)))
    out = {}
    for f in sel:
        i = names.index(f)
        r, s = single_auc(y_real, X_real[:, i]), single_auc(y_syn, X_syn[:, i])
        out[f] = {"real": r, "synth": s, "delta": s - r, "flag": bool(abs(s - r) > flag), "sentinel": f in sentinels}
    return out


def matched_prevalence_ap(y, s, prevalence, draws=20, seed=SEED) -> float:
    """AP after subsampling positives (or negatives) to the target prevalence; mean over draws."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) / len(y) > prevalence:
        n_pos, n_neg = int(round(prevalence * len(neg) / (1 - prevalence))), len(neg)
    else:
        n_pos, n_neg = len(pos), int(round(len(pos) * (1 - prevalence) / prevalence))
    v = []
    for _ in range(draws):
        i = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        v.append(average_precision_score(y[i], s[i]))
    return float(np.mean(v))


def ranking_agreement(real_auc: dict, syn_auc: dict) -> float:
    k = sorted(real_auc)
    return float(spearmanr([real_auc[d] for d in k], [syn_auc[d] for d in k]).statistic)


def dcr(X_ref, y_ref, X_syn, y_syn, X_hold, y_hold, n=None, seed=SEED) -> dict:
    """Distance to closest record in REAL TRAIN, within role, on features scaled by the real-train median / IQR
    (Euclidean). Synthetic and real-holdout query sets are subsampled to the same size per role. Ratio of 5th
    percentiles (synthetic / holdout) >= 1 passes; a holdout 5th percentile of 0 makes the ratio undefined."""
    from sklearn.neighbors import NearestNeighbors
    med = np.median(X_ref, 0); iqr = np.subtract(*np.quantile(X_ref, [0.75, 0.25], 0)); iqr[iqr == 0] = 1.0
    z = lambda X: (X - med) / iqr  # noqa: E731
    rng = np.random.default_rng(seed)
    out = {}
    for r in (0, 1):
        nn = NearestNeighbors(n_neighbors=1).fit(z(X_ref[y_ref == r]))
        S, H = np.flatnonzero(y_syn == r), np.flatnonzero(y_hold == r)
        m = min(len(S), len(H)) if n is None else min(n, len(S), len(H))
        ds = nn.kneighbors(z(X_syn[rng.choice(S, m, replace=False)]))[0][:, 0]
        dh = nn.kneighbors(z(X_hold[rng.choice(H, m, replace=False)]))[0][:, 0]
        p_s, p_h = float(np.quantile(ds, 0.05)), float(np.quantile(dh, 0.05))
        out["normal" if r == 0 else "scam"] = {"n": int(m), "p5_synth": p_s, "p5_holdout": p_h,
                                               "ratio": (p_s / p_h) if p_h > 0 else None,
                                               "pass": (p_s / p_h >= 1) if p_h > 0 else None,
                                               "synth_exact_copies": int((ds == 0).sum()), "holdout_exact": int((dh == 0).sum())}
    return out


def precision_recall_k(X_real, X_syn, k=5) -> dict:
    """Improved precision / recall (Kynkaanniemi et al., 2019): precision = share of synthetic points inside some real
    point's k-NN ball (radius = distance to its k-th real neighbour); recall = the same with the roles swapped."""
    from sklearn.neighbors import NearestNeighbors

    def radii(X):
        return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, k]  # column 0 is the point itself

    def covered(Xq, Xs, rs, chunk=512):
        hit = np.zeros(len(Xq), bool)
        for a in range(0, len(Xq), chunk):  # every ball, not only the nearest centres
            d = np.sqrt(((Xq[a:a + chunk, None, :] - Xs[None]) ** 2).sum(-1))
            hit[a:a + chunk] = (d <= rs[None]).any(1)
        return float(hit.mean())

    rr, rs_ = radii(X_real), radii(X_syn)
    return {"precision": covered(X_syn, X_real, rr), "recall": covered(X_real, X_syn, rs_)}
