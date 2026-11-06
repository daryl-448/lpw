"""Base qualification (reports/p1_base_qualification.md): tf_alibi vs tf_ref (B1 reported) on untouched confirmation
groups, two training seeds, both slices. Configurations and lr are frozen from the encoder screen (lr 1e-3).

  qualify.py fit SLICE NAME SEED   train + recalibrate on PILOT_conf (train split only; early stopping on val)
  qualify.py eval RUN              exact per-prefix terms on the 'conf' split -> RUN/conf_rows.npz, RUN/conf_eval.json
  qualify.py gen RUN               3 paired 'conf' replicas (seed 7): fidelity, role parity, capped -> RUN/gen_conf.json
  qualify.py report                gates per seed x slice -> ROOT/qualify.json, ROOT/qualify.txt
SLICE: 2018 (data/pilot_2018_conf) or 2021 (data/pilot_conf). ROOT = runs/p1_qualify; RUN = ROOT/SLICE/NAME_sSEED."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.atomic import write_text

PILOTS = {"2018": Path("data/pilot_2018_conf"), "2021": Path("data/pilot_conf")}
ROOT = Path("runs/p1_qualify")
ARMS, SEEDS, LR = ("tf_alibi", "tf_ref", "b1"), (20261104, 20261105), 1e-3
MARGIN, NI_MARGIN, NBOOT, GEN_REPS, SPLIT = -0.03, 0.03, 1000, 3, "conf"


def run_dir(sl, name, seed):
    return ROOT / sl / f"{name}_s{seed}"


def parse(run: Path):
    return run.parent.name, run.name.rsplit("_s", 1)[0], int(run.name.rsplit("_s", 1)[1])


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def fit(sl, name, seed):
    from lpw.p1.recal import ensure_recalibrated
    from lpw.p1.train import train
    out = run_dir(sl, name, seed)
    if not ((out / "model.pt").exists() and (out / "train.json").exists()):  # train.json = training completed
        out.mkdir(parents=True, exist_ok=True)
        info = train(PILOTS[sl], name, out, seed=int(seed), lr=LR, epochs=80, log=log)
        log(f"{sl}/{out.name}: best val {info['best_val']:.4f} at epoch {info['best_epoch']}")
    ensure_recalibrated(PILOTS[sl], name, out, log=log)


@torch.no_grad()
def eval_run(run: Path):
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.evaluate import evaluate
    from lpw.p1.store import build_store
    from lpw.p1.train import load, loader, to_dev
    sl, name, _ = parse(run)
    st = build_store(PILOTS[sl]); views = Views(st)
    m = load(PILOTS[sl], name, run, st)
    batches = pack(exact_specs(st, st.agents(SPLIT)), views)
    R = {k: [] for k in ("ktc", "ca", "own", "agent", "depth")}
    for bi, b in enumerate(loader(views, batches)):
        sp = batches[bi]
        b = to_dev(b, "cuda")
        t = m.terms(b)
        n = len(t["kind"])
        R["ktc"].append((t["kind"] + t["time"] + t["cens"]).double().cpu().numpy())
        R["ca"].append(torch.zeros(n, dtype=torch.float64, device="cuda").index_add(0, t["o_row"], (t["cp"] + t["amt"]).double()).cpu().numpy())
        R["own"].append(t["own"].cpu().numpy())
        rb, rj = b["row_b"].cpu().numpy(), b["row_j"].cpu().numpy()
        R["agent"].append(np.array([sp[i][0] for i in rb], np.int64)); R["depth"].append(np.array([sp[i][1] for i in rb], np.int64) + rj)
    R = {k: np.concatenate(v) for k, v in R.items()}
    np.savez(run / "conf_rows.npz", **R)
    ev = evaluate(m, st, SPLIT, views=views)
    full = (R["ktc"].sum() + R["ca"].sum()) / R["own"].sum()
    assert abs(full - ev["all"]["nll_full"]) < 1e-6 * max(1.0, abs(full)), (full, ev["all"]["nll_full"])
    write_text(run / "conf_eval.json", json.dumps(ev, indent=1))
    log(f"{sl}/{run.name}: exact conf full {ev['all']['nll_full']:.4f}")


def gen(run: Path):
    from lpw.p1.fidelity import Sample, agent_table, compare, groups_for, real_frames, role_parity
    from lpw.p1.run import average_fidelity, average_parity, generate_replicas, write_replicas
    from lpw.p1.store import build_store
    from lpw.p1.train import load
    sl, name, _ = parse(run)
    st = build_store(PILOTS[sl])
    agents = st.agents(SPLIT)
    rag, rev = real_frames(st, agents)
    real = Sample(rag, rev, st.w_start)
    pre_age = rag.filter(pl.col("cohort") == 0)["age_days"].to_numpy()
    age_edges = np.quantile(pre_age, [0, 1 / 3, 2 / 3, 1]); age_edges[-1] += 1
    groups = groups_for(rag, age_edges)
    tr = st.agents("train")
    tag_, tev = real_frames(st, tr)
    train_X = agent_table(Sample(tag_, tev, st.w_start), np.stack([st.prof_vec(x) for x in tr]))
    real_X = agent_table(real, np.stack([st.prof_vec(x) for x in agents]))
    conds = [dict(role=int(st.role[x]), cohort=int(st.cohort[x]), origin=int(st.origin[x]), prof=st.prof_vec(x),
                  b0=float(st.b0[x]), age_days=float(st.prof_raw[x, 17])) for x in agents]
    m = load(PILOTS[sl], name, run, st)
    t = time.time()
    reps, info = generate_replicas(m, [conds] * GEN_REPS, st, seed=7, log=log)
    write_replicas(reps, run / "gen_conf_paired.parquet")
    prof = np.stack([c["prof"] for c in conds])
    fids, pars = [], []
    for sag, sev in reps:
        syn = Sample(sag, sev, st.w_start)
        fids.append(compare(real, syn, groups, paired=True))
        pars.append(role_parity(train_X, st.role[tr], real_X, st.role[agents], agent_table(syn, prof), st.role[agents]))
    out = {"fidelity": average_fidelity(fids), "parity": average_parity(pars), "capped": int(info.get("capped", 0)),
           "events": [e.height for _, e in reps], "real_events": rev.height, "sec": round(time.time() - t)}
    write_text(run / "gen_conf.json", json.dumps(out, indent=1, default=float))
    f = out["fidelity"]
    log(f"{sl}/{run.name}: pass normal {f['normal']['pass_rate']:.3f} scam {f['scam']['pass_rate']:.3f}, "
        f"mean |parity delta| {out['parity']['abs_delta_mean']:.3f}, capped {out['capped']}")


def report():
    from lpw.p1.store import build_store
    lines, R, g = [], {}, {}

    def say(s=""):
        print(s); lines.append(s)

    rng = np.random.default_rng(0)
    for sl, pilot in PILOTS.items():
        ag = pl.read_parquet(pilot / "agents.parquet")
        gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
        st = build_store(pilot)
        R[sl] = {}
        for seed in SEEDS:
            rows = {}
            for name in ARMS:
                z = np.load(run_dir(sl, name, seed) / "conf_rows.npz")
                rows[name] = {k: z[k] for k in z.files}
            A = rows["tf_alibi"]
            for name in ARMS:
                assert np.array_equal(rows[name]["agent"], A["agent"]) and np.array_equal(rows[name]["depth"], A["depth"])
            own = A["own"].astype(np.float64)
            gid = np.array([gid_of[int(x)] for x in st.node_id[A["agent"]]])
            ug, inv = np.unique(gid, return_inverse=True)
            go = np.bincount(inv, own, len(ug))
            draws = rng.integers(0, len(ug), (NBOOT, len(ug)))
            full = {n: float((r["ktc"] + r["ca"]).sum() / own.sum()) for n, r in rows.items()}
            cell = {"full": full}
            for other in ("tf_ref", "b1"):
                diff = (A["ktc"] + A["ca"]) - (rows[other]["ktc"] + rows[other]["ca"])
                bs = np.bincount(inv, diff, len(ug))[draws].sum(1) / go[draws].sum(1)
                cell[f"alibi_minus_{other}"] = {"delta": float(diff.sum() / own.sum()), "upper95": float(np.quantile(bs, 0.95)),
                                                "by_depth": {f"{lo}-{hi}": float(diff[(A["depth"] >= lo) & (A["depth"] < hi)].sum()
                                                                                 / max(own[(A["depth"] >= lo) & (A["depth"] < hi)].sum(), 1))
                                                             for lo, hi in ((0, 4), (4, 100), (100, 256), (256, 10 ** 9))}}
            G = {n: json.loads((run_dir(sl, n, seed) / "gen_conf.json").read_text()) for n in ARMS}
            cell["gen"] = {n: {"pass_normal": G[n]["fidelity"]["normal"]["pass_rate"], "pass_scam": G[n]["fidelity"]["scam"]["pass_rate"],
                               "parity_abs_delta_mean": G[n]["parity"]["abs_delta_mean"], "capped": G[n]["capped"]} for n in ARMS}
            R[sl][str(seed)] = cell
            d, ge = cell["alibi_minus_tf_ref"], cell["gen"]
            say(f"{sl} seed {seed}: full NLL alibi {full['tf_alibi']:.4f} ref {full['tf_ref']:.4f} b1 {full['b1']:.4f}; "
                f"alibi - ref {d['delta']:+.4f} (upper95 {d['upper95']:+.4f}); alibi - b1 {cell['alibi_minus_b1']['delta']:+.4f} "
                f"(upper95 {cell['alibi_minus_b1']['upper95']:+.4f})")
            say("  alibi - ref by depth: " + ", ".join(f"{k} {v:+.3f}" for k, v in d["by_depth"].items())
                + " | alibi - b1: " + ", ".join(f"{k} {v:+.3f}" for k, v in cell["alibi_minus_b1"]["by_depth"].items()))
            for n in ARMS:
                say(f"  gen {n}: pass normal {ge[n]['pass_normal']:.3f} scam {ge[n]['pass_scam']:.3f}, mean |parity delta| "
                    f"{ge[n]['parity_abs_delta_mean']:.3f}, capped {ge[n]['capped']}")
            k = f"{sl} seed {seed}"
            g[f"L {k}: alibi - ref <= {MARGIN} and upper95 < 0"] = bool(d["delta"] <= MARGIN and d["upper95"] < 0)
            g[f"G {k}: alibi M-pass per role not below ref"] = bool(ge["tf_alibi"]["pass_normal"] >= ge["tf_ref"]["pass_normal"]
                                                                   and ge["tf_alibi"]["pass_scam"] >= ge["tf_ref"]["pass_scam"])
            g[f"G {k}: alibi mean |role-AUC delta| <= 0.05"] = bool(ge["tf_alibi"]["parity_abs_delta_mean"] <= 0.05)
            g[f"G {k}: alibi no capped streams"] = ge["tf_alibi"]["capped"] == 0
            b = cell["alibi_minus_b1"]
            cell["amended_b1_criterion_reported"] = {
                "non_inferior": bool(b["upper95"] <= NI_MARGIN),
                "generates_better": bool(ge["tf_alibi"]["pass_normal"] > ge["b1"]["pass_normal"] and ge["tf_alibi"]["pass_scam"] > ge["b1"]["pass_scam"]),
                "original_beats_b1": bool(b["delta"] < 0)}
            say(f"  reported (SPEC A1, not gated here): {cell['amended_b1_criterion_reported']}")
    R["gates"] = g
    say("\ngates:")
    for k, v in g.items():
        say(f"  {'PASS' if v else 'FAIL'} {k}")
    say(f"base qualification {'PASSES: tf_alibi qualifies as the transformer base' if all(g.values()) else 'FAILS: tf_alibi does not qualify'}")
    write_text(ROOT / "qualify.json", json.dumps(R, indent=1, default=float))
    write_text(ROOT / "qualify.txt", "\n".join(lines) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(*sys.argv[2:5])
    elif cmd == "eval":
        eval_run(Path(sys.argv[2]))
    elif cmd == "gen":
        gen(Path(sys.argv[2]))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
