"""Pre-registered pace score for the pace-shrinkage arms (Sol step-1 review round).

For each arm (saved replicas) vs real val:
- exposure-normalized time to the k-th event, tau_k = (t_k - origin) / (w_end - origin), non-reachers tau = 1;
  W1(real, synthetic) for normal plans 30-100 and 100-300 at k = 10 and 30; the primary score is the mean of these 4.
  Synthetic agents are grouped by their own plan (oracle: the real plan; prior: the sampled plan, read from the
  .plans.npz saved by drift_exp --save; required), real agents by the real plan.
- secondary: normal >= 300 and scam >= 30 W1s, under-plan rates per bucket, reach-30, medians of tau_10 / tau_30.
- endpoint: event shares in the last 1 / 7 / 30 days of the window (replica mean and per-replica values, for the
  O4 / P4 control-spread bound), same-second gap share, events per active day.
- paired whole-group bootstrap (val groups, 2000 resamples): score(control) - score(arm) and the relative reduction,
  95% intervals; the control is the first arm.
usage: scripts/p1_pace/pacescore.py PILOT oracle|prior LABEL=GEN.parquet [LABEL=GEN.parquet ...] [--out FILE] [--json FILE]"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import wasserstein_distance

from lpw.p1.atomic import write_text
from lpw.p1.fidelity import real_frames
from lpw.p1.plan import count_bucket
from lpw.p1.store import build_store

args = [x for x in sys.argv[1:] if not x.startswith("--")]
outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
jsonp = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
for x in (outp, jsonp):
    if x in args:
        args.remove(x)
pilot, mode, arms = Path(args[0]), args[1], [x.split("=", 1) for x in args[2:]]
assert mode in ("oracle", "prior") and arms and len({l for l, _ in arms}) == len(arms)
RES = {"mode": mode, "control": arms[0][0], "arm_paths": {l: str(Path(p)) for l, p in arms}, "replicas": {},
       "arms": {}, "vs_control": {}, "real_endpoint": {}}
st = build_store(pilot); va = st.agents("val"); roles = st.role[va]; W_END = float(st.w_end); DAY = 86400.0
origin = st.origin[va].astype(np.float64); expo = np.maximum(W_END - origin, 1.0)
_ag = pl.read_parquet(pilot / "agents.parquet")
gid = dict(zip(_ag["node_id"].to_list(), _ag["group_id"].to_list()))
group = np.array([gid[int(x)] for x in st.node_id[va]])
ug, ginv = np.unique(group, return_inverse=True)
real_plan = st.plan_n[va]
B = 2000
lines = []
def say(s=""):
    print(s); lines.append(s)


def per_agent_ts(ev):
    ev = ev.sort("agent", "ts"); a = ev["agent"].to_numpy(); t = ev["ts"].to_numpy().astype(np.float64)
    cuts = np.searchsorted(a, np.arange(len(va) + 1))
    return [t[cuts[i]:cuts[i + 1]] for i in range(len(va))]


def tau(streams, k):
    return np.array([min((s[k - 1] - origin[i]) / expo[i], 1.0) if len(s) >= k else 1.0 for i, s in enumerate(streams)])


def plans_for(path, R):
    if mode == "oracle":
        return np.broadcast_to(real_plan, (R, len(va)))
    npz = Path(path).with_suffix(".plans.npz")
    assert npz.exists(), f"{npz} (drift_exp --save) is required in prior mode"
    P = np.load(npz)["plan_n"]
    assert P.shape == (R, len(va)), (P.shape, R, len(va))
    return P


rev = real_frames(st, va)[1]
real_ts = per_agent_ts(rev)
real_tau = {k: tau(real_ts, k) for k in (10, 30)}
real_n = np.array([len(s) for s in real_ts])
CELLS = [("normal 30-100", 0, [6]), ("normal 100-300", 0, [7]), ("normal >=300", 0, [8, 9]), ("scam >=30", 1, [6, 7, 8, 9])]
PRIMARY = [(c, k) for c in CELLS[:2] for k in (10, 30)]

arm_data = []
for label, path in arms:
    sev = pl.read_parquet(path); reps = sorted(sev["replica"].unique().to_list())
    S = [per_agent_ts(sev.filter(pl.col("replica") == r)) for r in reps]
    P = plans_for(path, len(reps))
    RES["replicas"][label] = len(reps)
    arm_data.append(dict(label=label, S=S, P=P, tau={k: np.stack([tau(s, k) for s in S]) for k in (10, 30)},
                         n=np.stack([[len(x) for x in s] for s in S])))


def w1(arm, cell, k, gw=None):
    """W1 between real tau_k (agents with real plan in cell) and synthetic tau_k (agent-replicas with plan in cell);
    gw: per-group bootstrap multiplicities."""
    _, ro, bks = cell
    rm = (roles == ro) & np.isin(real_plan, bks)
    sm = (roles[None] == ro) & np.isin(arm["P"], bks)
    ww = np.ones(len(va)) if gw is None else gw[ginv].astype(np.float64)
    rw, sw = ww[rm], np.broadcast_to(ww, sm.shape)[sm]
    if rw.sum() == 0 or sw.sum() == 0:
        return np.nan
    return wasserstein_distance(real_tau[k][rm], arm["tau"][k][sm], rw, sw)


def score(arm, gw=None):
    return float(np.mean([w1(arm, c, k, gw) for c, k in PRIMARY]))


say(f"pace score ({mode}); primary = mean W1 of exposure-normalized tau_10 / tau_30 over normal 30-100 and 100-300")
_rng0 = np.random.default_rng(1)
_draws0 = [np.bincount(_rng0.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(B)]
for ro, rn in ((0, "normal"), (1, "scam")):  # real end-of-window shares and their whole-group bootstrap SD
    m = roles == ro
    cnt = np.array([len(real_ts[i]) for i in range(len(va))], np.float64)
    last = {d: np.array([np.sum(real_ts[i] > W_END - d * DAY) for i in range(len(va))], np.float64) for d in (1, 7, 30)}
    gs = lambda v: np.bincount(ginv[m], v[m], len(ug))
    RES["real_endpoint"][rn] = {str(d): {"share": float(last[d][m].sum() / cnt[m].sum()),
                                         "sd_boot": float(np.std([(w @ gs(last[d])) / max(w @ gs(cnt), 1) for w in _draws0]))}
                                for d in (1, 7, 30)}
for arm in arm_data:
    say(f"\n== {arm['label']}: primary score {score(arm):.4f}")
    A = RES["arms"][arm["label"]] = {"score": score(arm), "cells": {}, "endpoint": {}}
    for cell in CELLS:
        nm, ro, bks = cell
        rm = (roles == ro) & np.isin(real_plan, bks); sm = (roles[None] == ro) & np.isin(arm["P"], bks)
        lo = {6: 30, 7: 100, 8: 300, 9: 1000}
        under = np.mean([np.mean(count_bucket(arm["n"][r][sm[r]]) < arm["P"][r][sm[r]]) for r in range(len(arm["S"]))])
        A["cells"][nm] = {"w1_tau10": w1(arm, cell, 10), "w1_tau30": w1(arm, cell, 30), "under_plan": float(under),
                          "reach30": float(np.mean(arm["n"][sm] >= 30))}
        say(f"  {nm:15s} W1 tau10 {w1(arm, cell, 10):.4f} tau30 {w1(arm, cell, 30):.4f} | median tau10 real {np.median(real_tau[10][rm]):.3f} "
            f"syn {np.median(arm['tau'][10][sm]):.3f}, tau30 real {np.median(real_tau[30][rm]):.3f} syn {np.median(arm['tau'][30][sm]):.3f} | "
            f"reach30 real {np.mean(real_n[rm] >= 30):.2f} syn {np.mean(arm['n'][sm] >= 30):.2f} | under-plan {under:.2f}")
    for ro, rn in ((0, "normal"), (1, "scam")):
        m = roles == ro
        allr = np.concatenate([real_ts[i] for i in np.flatnonzero(m)])
        alls = [np.concatenate([s[i] for i in np.flatnonzero(m)]) for s in arm["S"]]
        gaps_r = np.concatenate([np.diff(real_ts[i]) for i in np.flatnonzero(m)])
        gaps_s = np.concatenate([np.diff(s[i]) for s in arm["S"] for i in np.flatnonzero(m)])
        reps_sh = {str(d): [float(np.mean(x > W_END - d * DAY)) if len(x) else float("nan") for x in alls] for d in (1, 7, 30)}
        A.setdefault("endpoint_reps", {})[rn] = reps_sh
        A["endpoint"][rn] = {d: float(np.mean(v)) for d, v in reps_sh.items()}
        A["events_" + rn] = float(np.mean([sum(len(s[i]) for i in np.flatnonzero(m)) for s in arm["S"]]))
        act = lambda ts_list: sum(len(x) for x in ts_list) / max(sum(len(np.unique(np.floor(x / DAY))) for x in ts_list if len(x)), 1)
        say(f"  {rn:6s} last 1/7/30 d share real {[round(float(np.mean(allr > W_END - d * DAY)), 4) for d in (1, 7, 30)]} "
            f"syn {[round(float(np.mean([np.mean(x > W_END - d * DAY) for x in alls])), 4) for d in (1, 7, 30)]} | "
            f"same-second real {np.mean(gaps_r == 0):.3f} syn {np.mean(gaps_s == 0):.3f} | ev/active day real {act([real_ts[i] for i in np.flatnonzero(m)]):.2f} "
            f"syn {np.mean([act([s[i] for i in np.flatnonzero(m)]) for s in arm['S']]):.2f} | events real {m.sum() and sum(len(real_ts[i]) for i in np.flatnonzero(m))} "
            f"syn {np.mean([sum(len(s[i]) for i in np.flatnonzero(m)) for s in arm['S']]):.0f}")

if len(arm_data) > 1:
    say(f"\npaired whole-group bootstrap vs {arm_data[0]['label']} ({B} resamples of {len(ug)} val groups)")
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(B)]
    base = [score(arm_data[0], gw) for gw in draws]
    for arm in arm_data[1:]:
        sc = [score(arm, gw) for gw in draws]
        d = np.array(base) - np.array(sc); rel = d / np.array(base)
        RES["vs_control"][arm["label"]] = {"improvement": score(arm_data[0]) - score(arm), "improvement_lo": float(np.percentile(d, 2.5)),
                                           "improvement_hi": float(np.percentile(d, 97.5)), "rel_reduction": 1 - score(arm) / score(arm_data[0]),
                                           "rel_lo": float(np.percentile(rel, 2.5)), "rel_hi": float(np.percentile(rel, 97.5)),
                                           "control": arm_data[0]["label"]}
        say(f"  {arm['label']}: improvement {score(arm_data[0]) - score(arm):+.4f} (95% {np.percentile(d, 2.5):+.4f}..{np.percentile(d, 97.5):+.4f}), "
            f"relative reduction {1 - score(arm) / score(arm_data[0]):.3f} (95% {np.percentile(rel, 2.5):.3f}..{np.percentile(rel, 97.5):.3f})")
if outp:
    write_text(outp, "\n".join(lines) + "\n")
if jsonp:
    write_text(jsonp, json.dumps(RES, indent=1, default=float))
