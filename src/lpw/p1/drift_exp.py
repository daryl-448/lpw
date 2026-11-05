"""Drift experiment harness (2018 by default): 5-seed paired generation from a checkpoint with plan mode
none | oracle | prior, drift-specific readouts, M-pass at full length and with both sides cut to 5 events, val NLL.

usage: uv run python -m lpw.p1.drift_exp --ckpt DIR --model lpw|lpw_plan --mode none|oracle|prior --out FILE.json [--pilot data/pilot_2018]
"""
import argparse, json, time
import numpy as np, polars as pl
from pathlib import Path
from lpw.p1.store import build_store
from lpw.p1.batch import Views
from lpw.p1.train import load
from lpw.p1.evaluate import evaluate
from lpw.p1.fidelity import Sample, compare, groups_for, real_frames
from lpw.p1.run import generate_replicas, average_fidelity
from lpw.p1.plan import PlanPrior, count_bucket

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--model", default="lpw"); ap.add_argument("--mode", default="none")
ap.add_argument("--out", required=True); ap.add_argument("--pilot", default="data/pilot_2018"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--no-nll", action="store_true")
ap.add_argument("--cuts", default="5", help="comma list of k: M-pass with both sides cut to the first k events")
ap.add_argument("--save", default=None, help="write the synthetic replicas to this parquet")
a = ap.parse_args()
t_start = time.time()
pilot = Path(a.pilot); st = build_store(pilot); views = Views(st)
model = load(pilot, a.model, Path(a.ckpt), st)
va = st.agents("val"); roles = st.role[va]
rag, rev = real_frames(st, va); real = Sample(rag, rev, st.w_start)
pre_age = rag.filter(pl.col("cohort") == 0)["age_days"].to_numpy(); edges = np.quantile(pre_age, [0, 1/3, 2/3, 1]); edges[-1] += 1
groups = groups_for(rag, edges)
base = [dict(role=int(st.role[x]), cohort=int(st.cohort[x]), origin=int(st.origin[x]), prof=st.prof_vec(x), b0=float(st.b0[x]),
             age_days=float(st.prof_raw[x, 17])) for x in va]
rng = np.random.default_rng(101)
pops, plans = [], []
if a.mode == "none":
    pops = [base] * a.seeds
else:
    model.plan_inference = True
    prior = PlanPrior(st) if a.mode == "prior" else None
    for _ in range(a.seeds):
        if a.mode == "oracle":
            pn, pa_ = st.plan_n[va], st.plan_a[va]
        else:
            pn, pa_ = prior.sample(st.role[va], st.cohort[va], np.stack([c["prof"] for c in base]), rng)
        plans.append(pn)
        pops.append([{**c, "plan_n": int(n), "plan_a": int(m)} for c, n, m in zip(base, pn, pa_)])
reps, info = generate_replicas(model, pops, st, seed=7, log=lambda *x: None)


def per_agent(ev, n_agents):
    ev = ev.sort("agent", "ts")
    out = {}
    for (ag,), g in ev.group_by(["agent"], maintain_order=True):
        out[ag] = (g["ts"].to_numpy(), g["amt"].to_numpy())
    return [out.get(i, (np.zeros(0), np.zeros(0))) for i in range(n_agents)]


def readouts(streams_list, mask):
    """streams_list: list (replicas) of per-agent (ts, amt) lists. Returns drift metrics over agents in mask."""
    n = np.array([[len(s[0]) for s in S] for S in streams_list])[:, mask]
    tot = n.sum(1)
    wander, deep_amt, epd = [], [], []
    for S in streams_list:
        for i in np.nonzero(mask)[0]:
            ts, amt = S[i]
            if len(amt) >= 60:
                wander.append(amt[55:60].mean() - amt[5:10].mean())
            if len(amt) > 100:
                deep_amt.append(amt[100:1000])
            if len(ts) >= 500:
                epd.append(len(ts) / len(np.unique(np.floor(ts / 86400))))
    deep = np.concatenate(deep_amt) if deep_amt else np.zeros(0)
    return {"events_per_replica": float(tot.mean()), "events_cv": float(tot.std() / max(tot.mean(), 1e-9)) if len(tot) > 1 else 0.0,
            "P(n>=1000)": float((n >= 1000).mean()), "capped(>=10000)": int((n >= 10000).sum()),
            "survival": {d: float((n >= d).mean()) for d in (2, 5, 10, 50, 200, 1000)},
            "wander_sd": float(np.std(wander)) if wander else None, "wander_n": len(wander),
            "deep_median_amt": float(np.median(deep)) if len(deep) else None,
            "events_per_active_day_n500": float(np.median(epd)) if epd else None, "n500": len(epd)}


def cut(ev, k):
    return ev.sort("agent", "ts").with_columns(pl.int_range(pl.len()).over("agent").alias("_k")).filter(pl.col("_k") < k).drop("_k")


res = {"ckpt": a.ckpt, "model": a.model, "mode": a.mode, "gen_info": info}
real_streams = [per_agent(rev, len(va))]
syn_streams = [per_agent(e, len(va)) for _, e in reps]
for ro, nm in ((0, "normal"), (1, "scam")):
    m = roles == ro
    res[f"real_{nm}"] = readouts(real_streams, m)
    res[f"syn_{nm}"] = readouts(syn_streams, m)
    if plans:
        comp = [np.mean(np.abs(count_bucket(np.array([len(s[0]) for s in S]))[m] - p[m]) <= 1) for S, p in zip(syn_streams, plans)]
        res[f"compliance_{nm}"] = float(np.mean(comp))
fid = average_fidelity([compare(real, Sample(ag_, e_, st.w_start), groups, paired=True) for ag_, e_ in reps])
failing = lambda f: [k for k, v in f.items() if isinstance(v, dict) and "pass_freq" in v and v["pass_freq"] < 0.5]
ratios = lambda f: {k: v["ratio_mean"] for k, v in f.items() if isinstance(v, dict) and "ratio_mean" in v}
for g in ("normal", "scam"):
    res[f"mpass_{g}"] = (fid[g]["pass_rate"], fid[g]["pass_rate_sd"])
    res[f"failing_{g}"] = failing(fid[g])
    res[f"ratio_{g}"] = ratios(fid[g])
for k in (int(x) for x in a.cuts.split(",")):
    realk = Sample(rag, cut(rev, k), st.w_start)
    fidk = average_fidelity([compare(realk, Sample(ag_, cut(e_, k), st.w_start), groups, paired=True) for ag_, e_ in reps])
    for g in ("normal", "scam"):
        res[f"mpass{k}_{g}"] = (fidk[g]["pass_rate"], fidk[g]["pass_rate_sd"])
        res[f"failing{k}_{g}"] = failing(fidk[g])
        res[f"ratio{k}_{g}"] = ratios(fidk[g])
if a.save:
    from lpw.p1.run import write_replicas
    write_replicas(reps, Path(a.save))
if not a.no_nll:
    model.plan_inference = False
    e = evaluate(model, st, "val", views=views)["all"]
    res["val_nll"] = {k: e[k] for k in ("nll_full", "nll_shared", "nll_kind", "nll_time", "nll_cens", "nll_cp", "nll_amt")}
res["sec"] = round(time.time() - t_start)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: v for k, v in res.items() if k not in ("gen_info",)}, indent=1, default=float))
