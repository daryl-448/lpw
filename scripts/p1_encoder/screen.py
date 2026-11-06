"""Encoder screen (reports/p1_encoder_screen.md), 2018 slice, val only (test sealed).

Arms: b1 (2-layer GRU, the B1 comparator), tf_ref (existing 4-layer absolute-position transformer, independent layer
init), tf_alibi (ALiBi event-event recency bias, PROFILE unpenalised, no position embeddings), tf_small (2 layers,
ff 128, B1-sized). Every arm: lr in {1e-3, 3e-4}, seed 20261104, current training recipe, staged recalibration.

  screen.py fit NAME LR SEED     train + recalibrate ROOT/NAME_lrLR_sSEED (idempotent)
  screen.py eval RUN             exact val per-prefix terms -> RUN/val_rows.npz, RUN/val_eval.json (evaluate())
  screen.py select               per arm: LR with the lowest exact val full NLL / owned event; challenger = the best
                                 transformer arm -> ROOT/selection.json (written once; never overwritten)
  screen.py gen RUN              3 paired val replicas (seed 7) -> fidelity, role parity, capped
  screen.py report               gates -> ROOT/screen.json, ROOT/screen.txt
ROOT = runs/p1_enc/2018."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.atomic import write_text

PILOT, ROOT = Path("data/pilot_2018"), Path("runs/p1_enc/2018")
ARMS, TF_ARMS, LRS = ("b1", "tf_ref", "tf_alibi", "tf_small"), ("tf_ref", "tf_alibi", "tf_small"), ("1e-3", "3e-4")
SEED1, SEED2 = 20261104, 20261105
MARGIN, NBOOT, GEN_REPS = -0.03, 1000, 3
OLD = Path("runs/p1/2018/results_val.json")  # B0 / B2 references (frozen baselines, unchanged code paths)


def run_dir(name, lr, seed):
    return ROOT / f"{name}_lr{lr}_s{seed}"


def parse(run: Path):
    name, lr, seed = run.name.rsplit("_lr", 1)[0], run.name.split("_lr")[1].split("_s")[0], int(run.name.rsplit("_s", 1)[1])
    return name, lr, seed


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def fit(name, lr, seed):
    from lpw.p1.recal import ensure_recalibrated
    from lpw.p1.train import train
    out = run_dir(name, lr, seed)
    if not (out / "model.pt").exists():
        out.mkdir(parents=True, exist_ok=True)
        info = train(PILOT, name, out, seed=int(seed), lr=float(lr), epochs=80, log=log)
        log(f"{out.name}: best val {info['best_val']:.4f} at epoch {info['best_epoch']}")
    ensure_recalibrated(PILOT, name, out, log=log)
    write_text(out / "fit.done", "ok\n")


@torch.no_grad()
def eval_run(run: Path):
    """per exact val prefix: kind + time + cens, cp + amt (owned rows), owned flag, agent, depth; plus evaluate()."""
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.evaluate import evaluate
    from lpw.p1.store import build_store
    from lpw.p1.train import load, loader, to_dev
    name, _, _ = parse(run)
    st = build_store(PILOT); views = Views(st)
    m = load(PILOT, name, run, st)
    batches = pack(exact_specs(st, st.agents("val")), views)
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
    np.savez(run / "val_rows.npz", **R)
    ev = evaluate(m, st, "val", views=views)
    full = (R["ktc"].sum() + R["ca"].sum()) / R["own"].sum()
    assert abs(full - ev["all"]["nll_full"]) < 1e-6 * max(1.0, abs(full)), (full, ev["all"]["nll_full"])
    write_text(run / "val_eval.json", json.dumps(ev, indent=1))
    log(f"{run.name}: exact val full {ev['all']['nll_full']:.4f} shared {ev['all']['nll_shared']:.4f}")


def full_nll(run):
    return json.loads((run / "val_eval.json").read_text())["all"]["nll_full"]


def select():
    p = ROOT / "selection.json"
    if p.exists():
        log(f"selection already made: {p.read_text()}")
        return
    sel = {}
    for name in ARMS:
        scores = {lr: full_nll(run_dir(name, lr, SEED1)) for lr in LRS}  # KeyError / missing file -> not selectable yet
        sel[name] = {"lr": min(scores, key=scores.get), "val_full": scores}
    ch = min(TF_ARMS, key=lambda n: sel[n]["val_full"][sel[n]["lr"]])
    out = {"arms": sel, "challenger": ch, "challenger_lr": sel[ch]["lr"], "b1_lr": sel["b1"]["lr"], "reference_lr": sel["tf_ref"]["lr"]}
    write_text(p, json.dumps(out, indent=1))
    log(f"selected: {json.dumps(out)}")


def gen(run: Path):
    from lpw.p1.batch import Views
    from lpw.p1.fidelity import Sample, agent_table, compare, groups_for, real_frames, role_parity
    from lpw.p1.run import average_fidelity, average_parity, generate_replicas, write_replicas
    from lpw.p1.store import build_store
    from lpw.p1.train import load
    name, _, _ = parse(run)
    st = build_store(PILOT)
    agents = st.agents("val")
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
    m = load(PILOT, name, run, st)
    t = time.time()
    reps, info = generate_replicas(m, [conds] * GEN_REPS, st, seed=7, log=log)
    write_replicas(reps, run / "gen_val_paired.parquet")
    prof = np.stack([c["prof"] for c in conds])
    fids, pars = [], []
    for sag, sev in reps:
        syn = Sample(sag, sev, st.w_start)
        fids.append(compare(real, syn, groups, paired=True))
        pars.append(role_parity(train_X, st.role[tr], real_X, st.role[agents], agent_table(syn, prof), st.role[agents]))
    out = {"fidelity": average_fidelity(fids), "parity": average_parity(pars), "capped": int(info.get("capped", 0)),
           "events": [e.height for _, e in reps], "real_events": rev.height, "sec": round(time.time() - t)}
    write_text(run / "gen_val.json", json.dumps(out, indent=1, default=float))
    f = out["fidelity"]
    log(f"{run.name}: pass normal {f['normal']['pass_rate']:.3f} scam {f['scam']['pass_rate']:.3f}, parity delta "
        f"{out['parity']['delta']:+.3f}, capped {out['capped']}")


def rows(run):
    z = np.load(run / "val_rows.npz")
    return {k: z[k] for k in z.files}


def report():
    from lpw.p1.store import build_store
    st = build_store(PILOT)
    ag = pl.read_parquet(PILOT / "agents.parquet")
    gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    S = json.loads((ROOT / "selection.json").read_text())
    ch, clr, blr, rlr = S["challenger"], S["challenger_lr"], S["b1_lr"], S["reference_lr"]
    lines, R, g = [], {"selection": S, "seeds": {}, "baselines": {}, "generation": {}, "depth": {}, "loo_long": {}}, {}

    def say(s=""):
        print(s); lines.append(s)

    say(f"selection: challenger {ch} (lr {clr}); b1 lr {blr}; tf_ref lr {rlr}")
    for name in ARMS:
        say(f"  {name}: " + ", ".join(f"lr {lr} val full {v:.4f}" for lr, v in S["arms"][name]["val_full"].items()))
    rng = np.random.default_rng(0)
    for seed in (SEED1, SEED2):
        A, B = rows(run_dir(ch, clr, seed)), rows(run_dir("b1", blr, seed))
        assert np.array_equal(A["agent"], B["agent"]) and np.array_equal(A["depth"], B["depth"])
        dA, dB = A["ktc"] + A["ca"], B["ktc"] + B["ca"]
        diff, own = dA - dB, A["own"].astype(np.float64)
        delta = diff.sum() / own.sum()
        gid = np.array([gid_of[int(x)] for x in st.node_id[A["agent"]]])
        ug, inv = np.unique(gid, return_inverse=True)
        gd, go = np.bincount(inv, diff, len(ug)), np.bincount(inv, own, len(ug))
        draws = rng.integers(0, len(ug), (NBOOT, len(ug)))
        bs = gd[draws].sum(1) / go[draws].sum(1)
        ub = float(np.quantile(bs, 0.95))
        R["seeds"][str(seed)] = {"challenger_full": float(dA.sum() / own.sum()), "b1_full": float(dB.sum() / own.sum()),
                                 "delta": float(delta), "boot_upper95": ub}
        say(f"seed {seed}: {ch} {dA.sum() / own.sum():.4f} vs b1 {dB.sum() / own.sum():.4f}: delta {delta:+.4f}, one-sided 95% upper {ub:+.4f}")
        g[f"L seed {seed}: delta <= {MARGIN} and bootstrap upper < 0"] = bool(delta <= MARGIN and ub < 0)
        if seed == SEED1:  # diagnostics (reported, not gated)
            nev = (st.off[A["agent"] + 1] - st.off[A["agent"]]).astype(np.int64)
            for lo, hi in ((0, 1), (1, 4), (4, 30), (30, 100), (100, 256), (256, 10 ** 9)):
                mm = (A["depth"] >= lo) & (A["depth"] < hi)
                R["depth"][f"{lo}-{hi}"] = float(diff[mm].sum() / max(own[mm].sum(), 1))
            say("  delta by depth (per owned event): " + ", ".join(f"{k} {v:+.3f}" for k, v in R["depth"].items()))
            longg = np.unique(gid[nev >= 300])
            loo = {str(x): float((diff.sum() - diff[gid == x].sum()) / (own.sum() - own[gid == x].sum())) for x in longg}
            R["loo_long"] = {"groups": len(longg), "max_delta": max(loo.values()) if loo else None, "per_group": loo}
            if loo:
                say(f"  leave-one-long-group-out ({len(longg)} groups with a >= 300-event account): delta max {max(loo.values()):+.4f}, min {min(loo.values()):+.4f}")
    old = json.loads(OLD.read_text())["nll"]
    for seed in (SEED1, SEED2):
        ev = json.loads((run_dir(ch, clr, seed) / "val_eval.json").read_text())["all"]
        b0, b2 = old["b0"]["all"]["nll_full"], old["b2"]["all"]["nll_shared"]
        R["baselines"][str(seed)] = {"full": ev["nll_full"], "shared": ev["nll_shared"], "b0_full": b0, "b2_shared": b2}
        g[f"L seed {seed}: beats B0 full ({b0:.3f}) and B2 shared ({b2:.3f})"] = bool(ev["nll_full"] < b0 and ev["nll_shared"] < b2)
    G = {}
    for lab, run in (("challenger", run_dir(ch, clr, SEED1)), ("tf_ref", run_dir("tf_ref", rlr, SEED1)), ("b1", run_dir("b1", blr, SEED1))):
        G[lab] = json.loads((run / "gen_val.json").read_text())
        f = G[lab]["fidelity"]
        R["generation"][lab] = {"run": run.name, "pass_normal": f["normal"]["pass_rate"], "pass_scam": f["scam"]["pass_rate"],
                                "parity_delta": G[lab]["parity"]["delta"], "capped": G[lab]["capped"],
                                "ratios": {r: {k: v["ratio_mean"] for k, v in f[r].items() if isinstance(v, dict) and "ratio_mean" in v} for r in ("normal", "scam")}}
        say(f"generation {lab} ({run.name}): pass normal {f['normal']['pass_rate']:.3f} scam {f['scam']['pass_rate']:.3f}, "
            f"parity delta {G[lab]['parity']['delta']:+.3f}, capped {G[lab]['capped']}")
    c, rf = R["generation"]["challenger"], R["generation"]["tf_ref"]
    g["G mean M-pass per role not below tf_ref"] = bool(c["pass_normal"] >= rf["pass_normal"] and c["pass_scam"] >= rf["pass_scam"])
    g["G role-AUC |delta| <= 0.05"] = bool(abs(c["parity_delta"]) <= 0.05)
    g["G no capped streams"] = c["capped"] == 0
    R["gates"] = g
    say("\ngates:")
    for k, v in g.items():
        say(f"  {'PASS' if v else 'FAIL'} {k}")
    say(f"encoder screen {'PASSES' if all(g.values()) else 'FAILS'} (challenger {ch})")
    write_text(ROOT / "screen.json", json.dumps(R, indent=1, default=float))
    write_text(ROOT / "screen.txt", "\n".join(lines) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(*sys.argv[2:5])
    elif cmd == "eval":
        eval_run(Path(sys.argv[2]))
    elif cmd == "select":
        select()
    elif cmd == "gen":
        gen(Path(sys.argv[2]))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
