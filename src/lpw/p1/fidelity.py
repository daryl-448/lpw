"""Single-agent generative fidelity (SPEC §7.2): M1–M16 against real streams (IN_agent removed), with noise floors.

Noise floor: split the real agents of a group into two random halves R1, R2; floor = d(R1, R2). The synthetic
score uses the same sizes: d(S1, R2), where S1 is the synthetic streams paired with R1 (paired generation) or a
random synthetic subset of |R1| agents (unconditional). 20 bootstraps; a metric passes if mean synthetic ≤ 2 × mean floor.

Units (SPEC §7.2 table): amounts are log10 ETH (M1 also per kind: it passes only if overall, OUT and IN_exo all
pass); gaps are log1p seconds between consecutive owned events of one agent; counts are log1p (zero-event agents
count 0 distinct counterparties); spans are days; flows and balances are signed log1p ETH. M15 covers every
agent-month from the agent's origin month to the window end, inactive months as 0. Real balances are replayed from
the initial balance over owned events only, matching the generator, which never sees IN_agent inflows.
"""

import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

from lpw.p1.store import IN_AGENT, OUT, REPEAT, month_index, slog

METRICS = [
    ("M1", "amount", "w1"), ("M2", "log_gap", "w1"), ("M3", "kind", "jsd"), ("M4", "hour", "jsd"), ("M5", "dow", "jsd"),
    ("M6", "kind3", "jsd"), ("M7", "kind_hour", "jsd"), ("M8", "cls", "jsd"), ("M9", "repeat_rate", "w1"),
    ("M10", "distinct_cp", "w1"), ("M11", "n_events", "w1"), ("M12", "span_days", "w1"), ("M13", "out_frac", "w1"),
    ("M14", "hour_entropy", "w1"), ("M15", "monthly_net", "w1"), ("M16", "end_balance", "w1"),
]
EVENT_METRICS = {"amount", "amount_out", "amount_in", "log_gap", "kind", "hour", "dow", "kind3", "kind_hour", "cls"}
M1_KINDS = ("amount_out", "amount_in")
N_CAT = {"kind": 2, "hour": 24, "dow": 7, "kind3": 8, "kind_hour": 48, "cls": 3}


def real_frames(store, agents: np.ndarray):
    """Real streams of the given agents in the generator's schema, IN_agent removed."""
    rows, ag = [], []
    for i, a in enumerate(agents):
        s, e = store.off[a], store.off[a + 1]
        k = store.kind[s:e]
        m = k != IN_AGENT
        rows.append(pl.DataFrame({"agent": np.full(m.sum(), i, np.int64), "kind": k[m], "ts": store.ts[s:e][m],
                                  "cls": store.cls[s:e][m], "ent": store.ent[s:e][m], "amt": store.amt[s:e][m],
                                  "eth": store.eth[s:e][m], "fee": store.fee[s:e][m]}))
    ev = pl.concat(rows)
    ag = pl.DataFrame({"agent": np.arange(len(agents)), "role": store.role[agents], "cohort": store.cohort[agents],
                       "origin": store.origin[agents], "b0": store.b0[agents], "age_days": store.prof_raw[agents, 17]})
    return ag, ev


def features(ag: pl.DataFrame, ev: pl.DataFrame, w_start: int):
    """Per-event and per-agent quantities used by M1–M16. Generated timestamps are continuous; real ones whole seconds."""
    ev = ev.with_columns(pl.col("ts").cast(pl.Float64)).sort("agent", "ts", maintain_order=True)
    ti = pl.col("ts").floor().cast(pl.Int64)
    ev = ev.with_columns(
        hour=((ti % 86400) // 3600), dow=(((ti // 86400) - 4) % 7),
        month=pl.Series(month_index(ev["ts"].to_numpy(), w_start)),
        flow=pl.when(pl.col("kind") == OUT).then(-pl.col("eth") - pl.col("fee")).otherwise(pl.col("eth")))
    ev = ev.with_columns(
        log_gap=pl.when(pl.col("agent") == pl.col("agent").shift(1)).then((pl.col("ts") - pl.col("ts").shift(1)).log1p()),
        kind3=pl.when((pl.col("agent") == pl.col("agent").shift(2)))
        .then(pl.col("kind").shift(2).cast(pl.Int64) * 4 + pl.col("kind").shift(1).cast(pl.Int64) * 2 + pl.col("kind").cast(pl.Int64)),
        kind_hour=pl.col("kind").cast(pl.Int64) * 24 + pl.col("hour"))
    per = ev.group_by("agent").agg(
        n=pl.len(), repeat_rate=(pl.col("cls") == REPEAT).mean(), distinct=pl.col("ent").n_unique(),
        span_days=(pl.col("ts").max() - pl.col("ts").min()) / 86400, out_frac=(pl.col("kind") == OUT).mean(),
        flow=pl.col("flow").sum(), hours=pl.col("hour"))
    ent = []
    for hs in per["hours"].to_list():
        c = np.bincount(np.asarray(hs, dtype=np.int64), minlength=24) / len(hs)
        c = c[c > 0]
        ent.append(float(-(c * np.log(c)).sum()))
    per = per.drop("hours").with_columns(pl.Series("hour_entropy", ent))
    A = ag.join(per, on="agent", how="left").with_columns(pl.col("n").fill_null(0), pl.col("flow").fill_null(0.0))
    A = A.with_columns(n_events=pl.col("n").log1p(), distinct_cp=pl.col("distinct").fill_null(0).log1p(),
                       end_balance=pl.Series(slog((A["b0"] + A["flow"]).to_numpy())))
    # every agent-month from the origin month to the window end (12 months), inactive months 0
    agents = ag["agent"].to_numpy()
    m0 = month_index(ag["origin"].to_numpy(), w_start)
    pos = {int(a): i for i, a in enumerate(agents)}
    grid = np.zeros((len(agents), 12))
    np.add.at(grid, (np.array([pos[int(a)] for a in ev["agent"].to_numpy()], np.int64), ev["month"].to_numpy()), ev["flow"].to_numpy())
    elig = np.arange(12)[None] >= m0[:, None]
    ia, im = np.nonzero(elig)
    monthly = pl.DataFrame({"agent": agents[ia], "month": im, "monthly_net": slog(grid[ia, im])})
    amt = pl.col("amt").cast(pl.Float64)
    return ev.with_columns(amount=amt, amount_out=pl.when(pl.col("kind") == OUT).then(amt),
                           amount_in=pl.when(pl.col("kind") != OUT).then(amt)), A, monthly


class Sample:
    """Numpy arrays per metric with agent ids, for fast subsetting."""

    def __init__(self, ag, ev, w_start: int):
        E, A, Mo = features(ag, ev, w_start)
        self.n_agents = ag.height
        self.ev_agent = E["agent"].to_numpy()
        self.ev = {m: E[m].to_numpy() for m in EVENT_METRICS}
        self.a_agent = A["agent"].to_numpy()
        self.ag = {m: A[m].to_numpy().astype(float) for m in ("repeat_rate", "distinct_cp", "n_events", "span_days", "out_frac",
                                                              "hour_entropy", "end_balance")}
        self.has_ev = A["n"].to_numpy() > 0
        self.mo_agent = Mo["agent"].to_numpy()
        self.mo = Mo["monthly_net"].to_numpy()
        self.A = A

    def values(self, metric: str, agents: np.ndarray) -> np.ndarray:
        if metric in EVENT_METRICS:
            v = self.ev[metric][np.isin(self.ev_agent, agents)]
        elif metric == "monthly_net":
            v = self.mo[np.isin(self.mo_agent, agents)]
        else:
            sel = np.isin(self.a_agent, agents)
            if metric not in ("n_events", "end_balance", "distinct_cp"):
                sel &= self.has_ev
            v = self.ag[metric][sel]
        return v[~np.isnan(v)] if v.dtype.kind == "f" else v


def distance(kind: str, metric: str, x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x.astype(float))]
    y = y[~np.isnan(y.astype(float))]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if kind == "w1":
        return float(wasserstein_distance(x.astype(float), y.astype(float)))
    k = N_CAT[metric]
    p = np.bincount(x.astype(np.int64), minlength=k)
    q = np.bincount(y.astype(np.int64), minlength=k)
    return float(jensenshannon(p, q, base=2) ** 2)


def compare(real: Sample, synth: Sample, groups: dict[str, np.ndarray], paired: bool, n_boot=20, seed=0,
            synth_groups: dict[str, np.ndarray] | None = None) -> dict:
    """groups: name → real agent ids. Paired: synthetic agent i is the stream of real agent i.
    Unpaired: synth_groups gives the synthetic ids of each group."""
    rng = np.random.default_rng(seed)
    out = {}
    for gname, R in groups.items():
        if len(R) < 4:
            continue
        S = R if paired else synth_groups[gname]
        res = {}
        todo = [(m, kind) for _, m, kind in METRICS] + [(m, "w1") for m in M1_KINDS]
        boots = {m: ([], []) for m, _ in todo}
        for _ in range(n_boot):
            p = rng.permutation(R)
            h = len(p) // 2
            R1, R2 = p[:h], p[h:2 * h]
            S1 = R1 if paired else rng.choice(S, size=min(h, len(S)), replace=False)
            for m, kind in todo:
                r2 = real.values(m, R2)
                boots[m][0].append(distance(kind, m, real.values(m, R1), r2))
                boots[m][1].append(distance(kind, m, synth.values(m, S1), r2))

        def score(m):
            fl, sy = np.array(boots[m][0]), np.array(boots[m][1])
            f = float(np.nanmean(fl)) if np.isfinite(fl).any() else float("nan")
            s_ = float(np.nanmean(sy)) if np.isfinite(sy).any() else float("nan")
            return {"metric": m, "floor": f, "synth": s_, "ratio": s_ / f if f > 0 else float("inf"),
                    "pass": bool(np.isfinite(s_) and np.isfinite(f) and s_ <= 2 * f)}

        for mid, m, kind in METRICS:
            res[mid] = score(m)
        res["M1"]["per_kind"] = {m: score(m) for m in M1_KINDS}
        res["M1"]["pass_overall"] = res["M1"]["pass"]
        res["M1"]["pass"] = res["M1"]["pass"] and all(v["pass"] for v in res["M1"]["per_kind"].values())
        res["pass_rate"] = float(np.mean([res[mid]["pass"] for mid, _, _ in METRICS]))
        # continuous companion of pass_rate (does not flip when a metric sits at the 2x threshold): geometric mean of the
        # synth / floor ratios over M1-M16; finite ratios only, n_ratio says how many entered
        lr = np.array([np.log(res[mid]["ratio"]) for mid, _, _ in METRICS if np.isfinite(res[mid]["ratio"]) and res[mid]["ratio"] > 0])
        res["ratio_gmean"] = float(np.exp(lr.mean())) if len(lr) else float("nan")
        res["n_ratio"] = int(len(lr))
        res["n_real"] = int(len(R))
        res["n_synth"] = int(len(S))
        out[gname] = res
    return out


def drop_largest_group(groups: dict[str, np.ndarray], real: Sample, group_of_agent: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    """groups without the agents of the real group (group_of_agent[agent id]) holding the most real events: the
    event-pooled metrics are otherwise dominated by one group. Returns (groups, dropped group id)."""
    n = np.bincount(real.ev_agent, minlength=len(group_of_agent))
    ug, inv = np.unique(group_of_agent, return_inverse=True)
    big = ug[np.argmax(np.bincount(inv, n[:len(group_of_agent)]))]
    keep = np.flatnonzero(group_of_agent != big)
    return {k: np.intersect1d(v, keep) for k, v in groups.items()}, int(big)


def groups_for(ag: pl.DataFrame, age_edges: np.ndarray | None = None) -> dict[str, np.ndarray]:
    g = {}
    role = ag["role"].to_numpy()
    coh = ag["cohort"].to_numpy()
    age = ag["age_days"].to_numpy()
    ids = ag["agent"].to_numpy()
    for r, rn in ((0, "normal"), (1, "scam")):
        g[rn] = ids[role == r]
        for c, cn in ((0, "pre"), (1, "born")):
            g[f"{rn}/{cn}"] = ids[(role == r) & (coh == c)]
        if age_edges is not None:
            for t in range(3):
                sel = (role == r) & (coh == 0) & (age >= age_edges[t]) & (age < age_edges[t + 1])
                g[f"{rn}/pre/age_t{t + 1}"] = ids[sel]
    return g


def agent_table(sample: Sample, prof: np.ndarray) -> np.ndarray:
    """Per-agent summaries for role-signal parity: M9–M16 quantities plus the profile vector."""
    A = sample.A.sort("agent")
    mo = pl.DataFrame({"agent": sample.mo_agent, "m": sample.mo}).group_by("agent").agg(pl.col("m").mean())
    A = A.join(mo, on="agent", how="left")
    cols = ["repeat_rate", "distinct_cp", "n_events", "span_days", "out_frac", "hour_entropy", "m", "end_balance"]
    return np.concatenate([A.select(cols).to_numpy().astype(float), prof], 1)


def role_parity(train_X, train_y, real_X, real_y, syn_X, syn_y, seed=0) -> dict:
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=8, eval_metric="logloss")
    clf.fit(train_X, train_y)
    ar = roc_auc_score(real_y, clf.predict_proba(real_X)[:, 1])
    a_s = roc_auc_score(syn_y, clf.predict_proba(syn_X)[:, 1])
    return {"auc_real": float(ar), "auc_synth": float(a_s), "delta": float(a_s - ar), "pass": bool(abs(a_s - ar) <= 0.05)}
