"""P3a Checkpoint 0 (reports/p3a_prereg.md §9): safety, population and fidelity of one world before any utility
readout. No detector touches labels of real val here except the P1 role-parity diagnostic.

usage: uv run python scripts/p3a/checkpoint0.py [--world 0]
Writes runs/p3a/2018/world{W}/checkpoint0.{json,txt}. Exit status 2 on a safety stop.
"""
import argparse
import json
import sys

import numpy as np
import polars as pl

from common import ROOT, group_ids, load_pop, raw_events, real, store, synth  # noqa: F401
from lpw.p1.atomic import write_text
from lpw.p1.fidelity import Sample, compare, groups_for, real_frames
from lpw.p1.plan import N_PLAN_A, N_PLAN_N
from lpw.p3.features import SETS, matrix
from lpw.p3.utility import c2st

CAP = 10000
MAX_CAPPED_SHARE = 0.001   # safety stop above 0.1% capped streams per population (prereg §9)
REAL_MAX = 4839            # longest real 2018 stream (all splits)
ap = argparse.ArgumentParser()
ap.add_argument("--world", type=int, default=0)
a = ap.parse_args()
st = store()
W = st.w_end - st.w_start
tr, va = st.agents("train"), st.agents("val")
gen = json.loads((ROOT / f"world{a.world}" / "gen.json").read_text())
res, L, stop = {"world": a.world}, [], []

# population: rates and joints against real train (the population models' fitting data)
for part in ("train", "hold"):
    conds, raw = load_pop(a.world, part)
    role = np.array([c["role"] for c in conds]); coh = np.array([c["cohort"] for c in conds])
    o = np.array([c["origin"] for c in conds]); pn = np.array([c["plan_n"] for c in conds]); pa = np.array([c["plan_a"] for c in conds])
    p = {"scam_rate": float(role.mean()), "born_given_role": [float(coh[role == r].mean()) for r in (0, 1)],
         "act_frac_q": np.quantile((o[coh == 1] - st.w_start) / W, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
         "origin_at_last_second": int((o == st.w_end - 1).sum())}
    plan = {}
    for r in (0, 1):
        for c in (0, 1):
            m = (role == r) & (coh == c); mr = (st.role[tr] == r) & (st.cohort[tr] == c)
            plan[f"{r}{c}"] = {"synth_n": np.bincount(pn[m], minlength=N_PLAN_N).tolist(), "real_n": np.bincount(st.plan_n[tr][mr], minlength=N_PLAN_N).tolist(),
                               "synth_a": np.bincount(pa[m], minlength=N_PLAN_A).tolist(), "real_a": np.bincount(st.plan_a[tr][mr], minlength=N_PLAN_A).tolist()}
    p["plan_by_role_cohort"] = plan
    res[f"population_{part}"] = p
rt = {"scam_rate": float(st.role[tr].mean()), "born_given_role": [float(st.cohort[tr][st.role[tr] == r].mean()) for r in (0, 1)],
      "act_frac_q": np.quantile((st.origin[tr][st.cohort[tr] == 1] - st.w_start) / W, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()}
res["population_real_train"] = rt
pt = res["population_train"]
L.append(f"population (synthetic train world vs real train): scam {pt['scam_rate']:.4f} vs {rt['scam_rate']:.4f}; born|role "
         f"{pt['born_given_role'][0]:.3f}/{pt['born_given_role'][1]:.3f} vs {rt['born_given_role'][0]:.3f}/{rt['born_given_role'][1]:.3f}; "
         f"born act_frac q10-90 {np.round(pt['act_frac_q'], 3).tolist()} vs {np.round(rt['act_frac_q'], 3).tolist()}; "
         f"origins at w_end-1: {pt['origin_at_last_second']}")

# profile joints: C2ST of synthetic vs real-train pre-existing agents on the profile + confounder block
rag_tr, rev_tr, Rtr = real(st, "train")
gtr = group_ids(st, tr)
rag_va, rev_va, Rva = real(st, "val")
real_va_ag, real_va_ev = real_frames(st, va)
realS = Sample(real_va_ag, real_va_ev, st.w_start)
pre_age = real_va_ag.filter(pl.col("cohort") == 0)["age_days"].to_numpy()
edges = np.quantile(pre_age, [0, 1 / 3, 2 / 3, 1]); edges[-1] += 1
groups = groups_for(real_va_ag, edges)

for arm in gen["arms"]:
    out = {}
    for part in ("train", "hold"):
        ev = raw_events(a.world, arm, part)
        conds, _ = load_pop(a.world, part)
        N = np.bincount(ev["agent"].to_numpy(), minlength=len(conds))
        burst = int(ev.with_columns(pl.col("ts").floor()).group_by("agent", "ts").len()["len"].max())
        nonfin = sum(int((~np.isfinite(ev[c].to_numpy().astype(float))).sum()) for c in ("amt", "eth", "fee", "ts"))
        ag, oev, F, info = synth(st, a.world, arm, part, enforce_cap=False)
        s = {"capped": int((N >= CAP).sum()), "gen_capped": gen["arms"][arm]["gen_info"]["capped"], "max_len": int(N.max()),
             "p999_len": float(np.quantile(N, 0.999)), "max_same_second_burst": burst, "nonfinite": nonfin,
             "dropped_no_events": info["n_dropped_no_events"], "dropped_by_role": info["dropped_by_role"],
             "events_per_agent": float(N.mean()), "capped_share": float((N >= CAP).mean()), "over_real_max": int((N > REAL_MAX).sum()),
             "capped_agents": [{"agent": int(i), "role": int(conds[i]["role"]), "cohort": int(conds[i]["cohort"]), "plan_n": int(conds[i]["plan_n"])}
                               for i in np.flatnonzero(N >= CAP)]}
        out[f"safety_{part}"] = s
        if s["capped_share"] > MAX_CAPPED_SHARE or s["nonfinite"]:
            stop.append(f"{arm}/{part}: capped {s['capped']}, nonfinite {s['nonfinite']}")
        L.append(f"{arm} {part}: cap {s['capped']} max {s['max_len']} p99.9 {s['p999_len']:.0f} burst {burst} nonfin {nonfin} "
                 f">real-max {s['over_real_max']} dropped {info['n_dropped_no_events']} {info['dropped_by_role']} events/agent {s['events_per_agent']:.2f}")
        if part == "train":
            pre = ag["cohort"].to_numpy() == 0
            rpre = rag_tr["cohort"].to_numpy() == 0
            out["profile_c2st_pre"] = c2st(matrix(Rtr, "confonly")[rpre], matrix(F, "confonly")[pre], gtr[rpre])
            L.append(f"{arm} profile+confounder C2ST (pre-existing, synthetic train world vs real train): {out['profile_c2st_pre']:.3f}")
        else:
            # P1 fidelity M1-M16 of the holdout world against real val, unpaired (amended scorer)
            conds, _ = load_pop(a.world, "hold")
            sag = pl.DataFrame({"agent": np.arange(len(conds)), "role": [c["role"] for c in conds], "cohort": [c["cohort"] for c in conds],
                                "origin": [c["origin"] for c in conds], "b0": [float(c["b0"]) for c in conds],
                                "age_days": [float(c["age_days"]) for c in conds]})
            synS = Sample(sag, ev, st.w_start)
            fid = compare(realS, synS, groups, paired=False, synth_groups=groups_for(sag, edges))
            out["fidelity"] = {g: {"pass_rate": v["pass_rate"], "ratio_gmean": v.get("ratio_gmean"), "n_real": v["n_real"],
                                   "n_synth": v["n_synth"], "fails": [m for m in v if isinstance(v[m], dict) and not v[m]["pass"]]}
                               for g, v in fid.items()}
            L.append(f"{arm} hold fidelity M-pass: " + " ".join(f"{g} {v['pass_rate']:.2f}" for g, v in out["fidelity"].items()
                                                                 if g in ("normal", "scam", "normal/born", "scam/born", "normal/pre", "scam/pre")))
            L.append(f"   fails normal {out['fidelity']['normal']['fails']} scam {out['fidelity']['scam']['fails']}")
    res[arm] = out
res["safety_stop"] = stop
L.insert(0, f"P3a checkpoint 0, world {a.world}: " + ("SAFETY STOP: " + "; ".join(stop) if stop else "safety ok"))
d = ROOT / f"world{a.world}"
write_text(d / "checkpoint0.json", json.dumps(res, indent=1, default=float))
write_text(d / "checkpoint0.txt", "\n".join(L) + "\n")
print("\n".join(L))
sys.exit(2 if stop else 0)
