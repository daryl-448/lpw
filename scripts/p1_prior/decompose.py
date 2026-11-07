"""Prior-mode loss decomposition for the shallow-corrected model (reports/p1_prior.md).

usage: uv run python scripts/p1_prior/decompose.py ROOT [--base-gen runs/p1_shallow/2018/gen] [--pilot data/pilot_2018] [--boot 50]

Plan sources, all kct with generation seed 7 and 5 replicas:
- oracle and prior are read from --base-gen (the shallow run);
- prior_exp, perm and perm_a are read from ROOT/gen (lpw.p1.drift_exp).
Every source is scored on the same split seeds (seed = replica index), for normal and scam. Real owned-count bands
are scored only for the sources that keep each agent's real count bucket (oracle, perm_a): elsewhere a band's
agents generate under other count plans, so band scores mean nothing.
Plan diagnostics per role: total variation of the count / amount bucket distributions vs the real val plans, and
the share of agents whose sampled count bucket equals their real one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "p1_shallow"))
from readout import BANDS, summarize  # noqa: E402

from lpw.p1.atomic import write_text  # noqa: E402
from lpw.p1.fidelity import Sample, compare, real_frames  # noqa: E402
from lpw.p1.plan import N_PLAN_A, N_PLAN_N  # noqa: E402
from lpw.p1.store import IN_AGENT, build_store  # noqa: E402

ARM = "kct"
SOURCES = ("oracle", "prior", "prior_exp", "perm", "perm_a")
REAL_COUNT = ("oracle", "perm_a")
CONTRASTS = (("assignment (oracle - perm)", "oracle", "perm"), ("amount assignment (oracle - perm_a)", "oracle", "perm_a"),
             ("prior vs exact marginal (prior - perm)", "prior", "perm"), ("exposure prior (prior_exp - prior)", "prior_exp", "prior"))


def tv(x, y, n):
    return 0.5 * float(np.abs(np.bincount(x, minlength=n) / len(x) - np.bincount(y, minlength=n) / len(y)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base-gen", default="runs/p1_shallow/2018/gen")
    ap.add_argument("--pilot", default="data/pilot_2018")
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--tag", default="", help="output name suffix, e.g. _v2 for a rescoring that keeps the earlier files")
    a = ap.parse_args()
    root = Path(a.root)
    st = build_store(Path(a.pilot))
    va = st.agents("val")
    nva = np.array([int((st.kind[st.off[x]:st.off[x + 1]] != IN_AGENT).sum()) for x in va])
    rag, rev = real_frames(st, va)
    realS, ag_s = Sample(rag, rev, st.w_start), rag.select("agent", "role", "cohort", "origin", "b0", "age_days")
    ids = np.arange(len(va))
    G = {}
    for r, rn in ((0, "normal"), (1, "scam")):
        G[rn] = ids[st.role[va] == r]
        for bn, lo, hi in BANDS:
            G[f"{rn}/{bn}"] = ids[(st.role[va] == r) & (nva >= lo) & (nva < hi)]
    path = lambda s: (Path(a.base_gen) if s in ("oracle", "prior") else root / "gen") / f"{ARM}_{s}"
    res = {"bands": {g: int(len(v)) for g, v in G.items()}, "fidelity": {}, "plans": {}, "per_replica": {}}
    seeds = None
    for s in SOURCES:
        ev = pl.read_parquet(path(s).with_suffix(".parquet"))
        reps = sorted(ev["replica"].unique().to_list())
        seeds = seeds or reps
        assert reps == seeds, f"{s}: replicas {reps} differ from {seeds}"
        groups = G if s in REAL_COUNT else {g: G[g] for g in ("normal", "scam")}
        # one compare() per group: its bootstrap rng runs through the groups in order, so a group's floors would
        # otherwise depend on which groups precede it
        fids = []
        for r in seeds:
            syn = Sample(ag_s, ev.filter(pl.col("replica") == r).drop("replica"), st.w_start)
            fids.append({g: compare(realS, syn, {g: R}, paired=True, n_boot=a.boot, seed=r)[g] for g, R in groups.items()})
        res["fidelity"][s] = summarize(fids)
        res["per_replica"][s] = {g: [f[g]["pass_rate"] for f in fids] for g in ("normal", "scam")}
        P = dict(np.load(path(s).with_suffix(".plans.npz")))
        pd = {}
        for r, rn in ((0, "normal"), (1, "scam")):
            m = st.role[va] == r
            rn_, ra_ = st.plan_n[va][m], st.plan_a[va][m]
            pd[rn] = {"count_tv": float(np.mean([tv(p[m], rn_, N_PLAN_N + 1) for p in P["plan_n"]])),
                      "amount_tv": float(np.mean([tv(p[m], ra_, N_PLAN_A + 1) for p in P["plan_a"]])),
                      "count_exact": float(np.mean([np.mean(p[m] == rn_) for p in P["plan_n"]])),
                      "amount_exact": float(np.mean([np.mean(p[m] == ra_) for p in P["plan_a"]]))}
        res["plans"][s] = pd
    res["contrasts"] = {}
    for name, x, y in CONTRASTS:
        res["contrasts"][name] = {}
        for g in ("normal", "scam"):
            d = np.array(res["per_replica"][x][g]) - np.array(res["per_replica"][y][g])
            res["contrasts"][name][g] = {"diff": float(d.mean()), "sd": float(d.std())}
    write_text(root / f"decompose{a.tag}.json", json.dumps(res, indent=1, default=float))

    L = [f"M-pass / gmean ({ARM}, {len(seeds)} replicas, shared split seeds)"]
    for g in G:
        L.append(f"{g:18s} n {res['bands'][g]:5d} | " + " | ".join(
            f"{s} {res['fidelity'][s][g]['mpass']:.3f}±{res['fidelity'][s][g]['mpass_sd']:.3f}/{res['fidelity'][s][g]['gmean']:.2f}"
            for s in SOURCES if g in res["fidelity"][s]))
    L.append("contrasts (mean of per-replica differences ± sd):")
    for name, v in res["contrasts"].items():
        L.append(f"  {name:40s} normal {v['normal']['diff']:+.3f}±{v['normal']['sd']:.3f}  scam {v['scam']['diff']:+.3f}±{v['scam']['sd']:.3f}")
    L.append("plans vs real val (count TV / amount TV / count exact / amount exact):")
    for s in SOURCES:
        L.append(f"  {s:10s} " + "  ".join(f"{rn} {v['count_tv']:.3f}/{v['amount_tv']:.3f}/{v['count_exact']:.3f}/{v['amount_exact']:.3f}"
                                           for rn, v in res["plans"][s].items()))
    worst = {s: sorted(res["fidelity"][s]["normal"]["ratios"].items(), key=lambda kv: -kv[1])[:5] for s in SOURCES}
    L.append("normal top ratios: " + " | ".join(f"{s} " + " ".join(f"{m}:{v:.2f}" for m, v in w) for s, w in worst.items()))
    write_text(root / f"decompose{a.tag}.txt", "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
