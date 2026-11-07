"""Readouts of the shallow-row correction (exploratory; design and decision rules: reports/p1_shallow.md).

usage: uv run python scripts/p1_shallow/readout.py ROOT [--pilot data/pilot_2018] [--donors 3] [--boot 50]

ROOT holds fit.json (lpw.p1.shallow_fit) and gen/{arm}_{oracle,prior}.{parquet,plans.npz,json} written by
lpw.p1.drift_exp for arms B, k, kc, kct.

For each arm, normal and scam, by real owned-count band (light 1-4, mid 5-99, heavy 100+) and overall:
- M-pass and ratio_gmean in oracle mode (5 paired replicas);
- a matched real reference: donors are train agents with the same role, cohort, age decile / activation month and
  plan count bucket, over several draws;
- gap closed = (M_arm - M_B) / (M_ref - M_B), unclipped, with the raw difference.
Whole-population M-pass and gmean in prior mode, and prior retention of the oracle gain. Prior-mode safety counts.
Generated shallow behaviour of light accounts. The s-unknown val NLL of every arm (must equal B's).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from lpw.p1.atomic import write_text
from lpw.p1.fidelity import METRICS, Sample, compare, real_frames
from lpw.p1.plan import COUNT_EDGES, N_PLAN_N
from lpw.p1.store import IN_AGENT, build_store, month_index

ARMS = ("B", "k", "kc", "kct")
BANDS = (("light1-4", 1, 5), ("mid5-99", 5, 100), ("heavy100+", 100, 10 ** 9))
CAP = 10000


def bucket(n):
    return int(np.searchsorted(COUNT_EDGES, n, side="right") - 1)


def summarize(results: list[dict]) -> dict:
    out = {}
    for g in results[0]:
        mp = [r[g]["pass_rate"] for r in results]
        gm = [r[g]["ratio_gmean"] for r in results]
        out[g] = {"mpass": float(np.mean(mp)), "mpass_sd": float(np.std(mp)), "gmean": float(np.exp(np.mean(np.log(gm)))),
                  "ratios": {m: float(np.mean([r[g][m]["ratio"] for r in results])) for m, _, _ in METRICS},
                  "n_real": results[0][g]["n_real"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pilot", default="data/pilot_2018")
    ap.add_argument("--donors", type=int, default=3)
    ap.add_argument("--boot", type=int, default=50)
    a = ap.parse_args()
    root = Path(a.root)
    st = build_store(Path(a.pilot))
    va, tr = st.agents("val"), st.agents("train")
    n_own = lambda x: int((st.kind[st.off[x]:st.off[x + 1]] != IN_AGENT).sum())
    nva, ntr = np.array([n_own(x) for x in va]), np.array([n_own(x) for x in tr])
    rag, rev = real_frames(st, va)
    realS = Sample(rag, rev, st.w_start)
    ids = np.arange(len(va))
    G = {}
    for r, rn in ((0, "normal"), (1, "scam")):
        G[rn] = ids[st.role[va] == r]
        for bn, lo, hi in BANDS:
            G[f"{rn}/{bn}"] = ids[(st.role[va] == r) & (nva >= lo) & (nva < hi)]
    res = {"bands": {g: int(len(v)) for g, v in G.items()}}

    # matched real reference
    age = st.prof_raw[:, 17]
    pre_edges = np.quantile(age[tr][st.cohort[tr] == 0], np.linspace(0, 1, 11))
    pre_edges[-1] += 1
    def keys(x, n):
        c = (0, int(np.searchsorted(pre_edges, age[x], side="right"))) if st.cohort[x] == 0 else \
            (1, int(month_index(np.array([st.origin[x]]), st.w_start)[0]))
        return [(int(st.role[x]),) + c + (bucket(n),), (int(st.role[x]), int(st.cohort[x]), bucket(n)), (int(st.role[x]), bucket(n))]
    pools = [{}, {}, {}]
    for x, n in zip(tr, ntr):
        for i, k in enumerate(keys(x, n)):
            pools[i].setdefault(k, []).append(x)
    # every donor draw is scored over the arms' split-seed schedule (seed = replica index), so all share the floors
    seeds = sorted(pl.read_parquet(root / "gen" / "B_oracle.parquet", columns=["replica"])["replica"].unique().to_list())
    rng = np.random.default_rng(0)
    ref, fallback = [], 0
    for d in range(a.donors):
        pick = []
        for x, n in zip(va, nva):
            ks = keys(x, n)
            lvl = next(i for i in range(3) if ks[i] in pools[i])
            fallback += lvl > 0
            pick.append(rng.choice(pools[lvl][ks[lvl]]))
        dag, dev_ = real_frames(st, np.array(pick))
        donor = Sample(dag, dev_, st.w_start)
        ref.extend(compare(realS, donor, G, paired=True, n_boot=a.boot, seed=r) for r in seeds)
    res["reference"] = summarize(ref)
    res["reference_fallbacks_per_draw"] = fallback / a.donors

    ag_s = rag.select("agent", "role", "cohort", "origin", "b0", "age_days")
    for mode in ("oracle", "prior"):
        plans0 = None
        for arm in ARMS:
            ev = pl.read_parquet(root / "gen" / f"{arm}_{mode}.parquet")
            assert sorted(ev["replica"].unique().to_list()) == seeds, f"{arm} {mode}: replicas differ from B oracle's"
            P = dict(np.load(root / "gen" / f"{arm}_{mode}.plans.npz"))
            plans0 = plans0 or P
            assert all(np.array_equal(P[k], plans0[k]) for k in P), f"{arm} {mode}: plans differ from B's"
            reps = sorted(ev["replica"].unique().to_list())
            fids, safety = [], {"capped": [], "over3U": [], "zero_plan_nonempty": []}
            light = {"d1_out": [], "d1_in_repeat": []}
            for r in reps:
                e = ev.filter(pl.col("replica") == r).drop("replica")
                groups = G if mode == "oracle" else {g: G[g] for g in ("normal", "scam")}
                fids.append(compare(realS, Sample(ag_s, e, st.w_start), groups, paired=True, n_boot=a.boot, seed=r))
                N = np.bincount(e["agent"].to_numpy(), minlength=len(va))
                pn = P["plan_n"][r]
                fin = pn < N_PLAN_N - 1
                U = np.where(fin, COUNT_EDGES[np.minimum(pn + 1, N_PLAN_N - 1)] - 1, np.iinfo(np.int64).max)
                safety["capped"].append(int((N >= CAP).sum()))
                safety["over3U"].append(int((fin & (N > 3 * U)).sum()))
                safety["zero_plan_nonempty"].append(int(((pn == 0) & (N > 0)).sum()))
                if mode == "oracle":
                    li = G["normal/light1-4"]
                    s2 = e.filter(pl.col("agent").is_in(li.tolist())).sort("agent", "ts", maintain_order=True) \
                        .with_columns(pos=pl.int_range(pl.len()).over("agent")).filter(pl.col("pos") == 1)
                    light["d1_out"].append(float((s2["kind"] == 0).mean()) if s2.height else float("nan"))
                    si = s2.filter(pl.col("kind") == 1)
                    light["d1_in_repeat"].append(float((si["cls"] == 0).mean()) if si.height else float("nan"))
            cell = {"fidelity": summarize(fids), "safety": safety}
            if mode == "oracle":
                cell["light_generated"] = {k: float(np.nanmean(v)) for k, v in light.items()}
            j = root / "gen" / f"{arm}_{mode}.json"
            if j.exists():
                cell["val_nll_s_unknown"] = json.loads(j.read_text()).get("val_nll")
            res.setdefault(mode, {})[arm] = cell
    # real light accounts' second event, for the generated comparison
    li = G["normal/light1-4"]
    s2 = rev.filter(pl.col("agent").is_in(li.tolist())).sort("agent", "ts", maintain_order=True) \
        .with_columns(pos=pl.int_range(pl.len()).over("agent")).filter(pl.col("pos") == 1)
    res["light_real"] = {"d1_out": float((s2["kind"] == 0).mean()),
                         "d1_in_repeat": float((s2.filter(pl.col("kind") == 1)["cls"] == 0).mean())}

    # gap closed (oracle) and prior retention
    gap = {}
    for arm in ARMS[1:]:
        gap[arm] = {}
        for g in G:
            mb, ma, mr = (res["oracle"]["B"]["fidelity"][g]["mpass"], res["oracle"][arm]["fidelity"][g]["mpass"],
                          res["reference"][g]["mpass"])
            gap[arm][g] = {"diff": ma - mb, "closed": (ma - mb) / (mr - mb) if abs(mr - mb) > 1e-9 else float("nan")}
        for g in ("normal", "scam"):
            o = res["oracle"][arm]["fidelity"][g]["mpass"] - res["oracle"]["B"]["fidelity"][g]["mpass"]
            p = res["prior"][arm]["fidelity"][g]["mpass"] - res["prior"]["B"]["fidelity"][g]["mpass"]
            gap[arm][f"{g}_prior_diff"] = p
            gap[arm][f"{g}_prior_retention"] = p / o if abs(o) > 1e-9 else float("nan")
    res["gap"] = gap

    # decision rules (reports/p1_shallow.md)
    fit = json.loads((root / "fit.json").read_text())
    prim = fit["selection"]["primary"]
    dec = {"primary": prim}
    if prim == "zero":
        dec["outcome"] = "KILL: the selection keeps no correction"
    else:
        vr = fit["val"][prim]
        dec["val_light_normal_shallow_nll_delta"] = vr["normal"]["light1-4"]["delta_vs_B"]
        dec["val_known_plan_delta_per_owned"] = {rn: vr[rn]["delta_per_owned_event"] for rn in ("normal", "scam")}
        ok_t = dec["val_light_normal_shallow_nll_delta"] < 0
        ok_l = all(v <= 0.02 for v in dec["val_known_plan_delta_per_owned"].values())
        g = gap[prim]
        promote = (g["normal/light1-4"]["closed"] >= 0.25 and g["normal/mid5-99"]["diff"] >= 0 and g["normal_prior_diff"] >= 0)
        dec["checks"] = {"shallow NLL gain transfers to val (light normal)": ok_t, "known-plan val NLL change <= +0.02 per role": ok_l,
                         "light normal gap closed >= 25% (oracle)": g["normal/light1-4"]["closed"] >= 0.25,
                         "mid normal M-pass not below B (oracle)": g["normal/mid5-99"]["diff"] >= 0,
                         "prior-mode normal M-pass not below B": g["normal_prior_diff"] >= 0}
        dec["outcome"] = ("PROMOTE" if (ok_t and ok_l and promote) else "KILL" if not (ok_t and ok_l) else "STOP PROMOTION")
    res["decision"] = dec
    write_text(root / "readout.json", json.dumps(res, indent=1, default=float))

    L = [f"primary variant: {prim}; outcome: {dec['outcome']}", json.dumps(dec.get("checks", {}), indent=1)]
    for mode in ("oracle", "prior"):
        for g in (G if mode == "oracle" else ("normal", "scam")):
            line = f"{mode:6s} {g:18s} n {res['bands'][g]:5d} | " + " | ".join(
                f"{arm} {res[mode][arm]['fidelity'][g]['mpass']:.3f}/{res[mode][arm]['fidelity'][g]['gmean']:.2f}" for arm in ARMS)
            if mode == "oracle":
                line += f" | ref {res['reference'][g]['mpass']:.3f}/{res['reference'][g]['gmean']:.2f}"
            L.append(line)
    for arm in ARMS[1:]:
        L.append(f"gap closed {arm}: " + " ".join(f"{g} {v['closed']:+.2f}" for g, v in gap[arm].items() if isinstance(v, dict)))
        L.append(f"   prior: normal diff {gap[arm]['normal_prior_diff']:+.3f} retention {gap[arm]['normal_prior_retention']:+.2f}; "
                 f"scam diff {gap[arm]['scam_prior_diff']:+.3f}")
    L.append(f"light normal 2nd event real: OUT {res['light_real']['d1_out']:.3f}, IN REPEAT {res['light_real']['d1_in_repeat']:.3f}; "
             + " | ".join(f"{arm} {res['oracle'][arm]['light_generated']['d1_out']:.3f} / {res['oracle'][arm]['light_generated']['d1_in_repeat']:.3f}" for arm in ARMS))
    for arm in ARMS:
        L.append(f"safety prior {arm}: {res['prior'][arm]['safety']}; s-unknown val NLL {(res['oracle'][arm].get('val_nll_s_unknown') or {}).get('nll_full')}")
    write_text(root / "readout.txt", "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
