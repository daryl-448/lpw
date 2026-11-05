"""Phase 1 pipeline for one slice (SPEC §8 P1).

Usage: uv run python -m lpw.p1.run --pilot data/pilot --tag 2021 --split val
       (--split test only once, after the code is frozen; SPEC §7 split discipline)

Steps: train every model (early stop on val), exact next-event evaluation, profile generator, paired and
unconditional generation, M1–M16 with noise floors, role-signal parity, temporal check, exit criteria.

Generation is stochastic and one sample's M-pass rate moves by up to ±0.2 between seeds, so every generated
population is drawn --gen-seeds times (as replicas inside one generate call) and fidelity and parity are averaged
over the replicas; the exit criteria use the averages.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.batch import Views
from lpw.p1.evaluate import evaluate
from lpw.p1.fidelity import Sample, agent_table, compare, groups_for, real_frames, role_parity
from lpw.p1.generate import generate
from lpw.p1.profile_gen import ProfileGen
from lpw.p1.recal import ensure_recalibrated
from lpw.p1.store import build_store
from lpw.p1.train import cutoff_ts, load, train

MODELS = ["lpw", "a1", "a3", "b0", "b1", "b2"]
GEN_MODELS = ["lpw", "a1", "a3", "b0", "b1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/pilot")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--gen-seeds", type=int, default=5)
    ap.add_argument("--reuse-nll", action="store_true", help="reuse NLL and temporal results from an existing results file")
    a = ap.parse_args()
    pilot, root = Path(a.pilot), Path("runs/p1") / a.tag
    root.mkdir(parents=True, exist_ok=True)
    logf = open(root / f"run_{a.split}.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    models = a.models.split(",")
    store = build_store(pilot)
    views = Views(store)
    res = {"pilot": str(pilot), "split": a.split, "gen_seeds": a.gen_seeds, "nll": {}, "fidelity": {}, "parity": {}, "gen_info": {}}
    prev, prev_mtime = {}, 0.0
    rf = root / f"results_{a.split}.json"
    if a.reuse_nll and rf.exists():
        prev, prev_mtime = json.loads(rf.read_text()), rf.stat().st_mtime
        if prev.get("pilot") != str(pilot):
            prev = {}

    def reusable(*ckpts):
        """Only results computed after every checkpoint they depend on was written (the evaluator code itself is
        the caller's responsibility when passing --reuse-nll)."""
        return all((root / c / "model.pt").exists() and (root / c / "model.pt").stat().st_mtime < prev_mtime for c in ckpts)

    for m in models:
        if not (root / m / "model.pt").exists():
            log(f"train {m}")
            info = train(pilot, m, root / m, epochs=a.epochs, log=log)
            log(f"  {m}: best val {info['best_val']:.4f} at epoch {info['best_epoch']}")
        ensure_recalibrated(pilot, m, root / m, log=log)  # no-op once done; older checkpoints get it here
    for m in models:
        if m in prev.get("nll", {}) and reusable(m):
            res["nll"][m] = prev["nll"][m]
            log(f"eval {m}: reused full {res['nll'][m]['all']['nll_full']:.4f} shared {res['nll'][m]['all']['nll_shared']:.4f}")
            continue
        model = load(pilot, m, root / m, store)
        t = time.time()
        res["nll"][m] = evaluate(model, store, a.split, views=views)
        log(f"eval {m}: full {res['nll'][m]['all']['nll_full']:.4f} shared {res['nll'][m]['all']['nll_shared']:.4f} ({time.time() - t:.0f}s)")

    agents = store.agents(a.split)
    rag, rev = real_frames(store, agents)
    real = Sample(rag, rev, store.w_start)
    pre_age = rag.filter(pl.col("cohort") == 0)["age_days"].to_numpy()
    age_edges = np.quantile(pre_age, [0, 1 / 3, 2 / 3, 1]) if len(pre_age) else None
    if age_edges is not None:
        age_edges[-1] += 1
    groups = groups_for(rag, age_edges)
    tr = store.agents("train")
    tag_, tev = real_frames(store, tr)
    train_X = agent_table(Sample(tag_, tev, store.w_start), np.stack([store.prof_vec(x) for x in tr]))
    train_y = store.role[tr]
    real_X = agent_table(real, np.stack([store.prof_vec(x) for x in agents]))
    real_y = store.role[agents]
    conds = [dict(role=int(store.role[x]), cohort=int(store.cohort[x]), origin=int(store.origin[x]), prof=store.prof_vec(x),
                  b0=float(store.b0[x]), age_days=float(store.prof_raw[x, 17])) for x in agents]
    for m in GEN_MODELS:
        if m not in models:
            continue
        model = load(pilot, m, root / m, store)
        t = time.time()
        reps, info = generate_replicas(model, [conds] * a.gen_seeds, store, seed=7, log=log)
        write_replicas(reps, root / f"gen_{m}_{a.split}_paired.parquet")
        res["gen_info"][f"{m}/paired"] = {**info, "events": [e.height for _, e in reps], "real_events": rev.height,
                                         "sec": round(time.time() - t)}
        prof = np.stack([c["prof"] for c in conds])
        fids, pars = [], []
        for sag, sev in reps:
            syn = Sample(sag, sev, store.w_start)
            fids.append(compare(real, syn, groups, paired=True))
            pars.append(role_parity(train_X, train_y, real_X, real_y, agent_table(syn, prof), real_y))
        res["fidelity"][f"{m}/paired"] = average_fidelity(fids)
        res["parity"][f"{m}/paired"] = average_parity(pars)
        fr = res["fidelity"][f"{m}/paired"]
        log(f"gen {m}: {np.mean(res['gen_info'][f'{m}/paired']['events']):.0f} events/seed vs {rev.height} real ({time.time() - t:.0f}s); "
            f"pass normal {fr['normal']['pass_rate']:.2f}±{fr['normal']['pass_rate_sd']:.2f} scam {fr['scam']['pass_rate']:.2f}"
            f"±{fr['scam']['pass_rate_sd']:.2f}; parity delta {res['parity'][f'{m}/paired']['delta']:+.3f}")

    if "lpw" in models:
        pg = ProfileGen(store, log=log)
        rng = np.random.default_rng(11)
        roles = store.role[agents]
        upops = [pg.population(roles, rng) for _ in range(a.gen_seeds)]
        model = load(pilot, "lpw", root / "lpw", store)
        reps, info = generate_replicas(model, upops, store, seed=13, log=log)
        write_replicas(reps, root / f"gen_lpw_{a.split}_uncond.parquet")
        res["gen_info"]["lpw/uncond"] = {**info, "events": [e.height for _, e in reps]}
        fids, pars = [], []
        for (uag, uev), uconds in zip(reps, upops):
            usyn = Sample(uag, uev, store.w_start)
            fids.append(compare(real, usyn, groups, paired=False, synth_groups=groups_for(uag, age_edges)))
            pars.append(role_parity(train_X, train_y, real_X, real_y, agent_table(usyn, np.stack([c["prof"] for c in uconds])), roles))
        res["fidelity"]["lpw/uncond"] = average_fidelity(fids)
        res["parity"]["lpw/uncond"] = average_parity(pars)
        res["profile_gen"] = {"val_nll": pg.val_nll, "epochs": pg.epochs}
        fr = res["fidelity"]["lpw/uncond"]
        log(f"uncond lpw: {np.mean(res['gen_info']['lpw/uncond']['events']):.0f} events/seed; pass normal {fr['normal']['pass_rate']:.2f}"
            f"±{fr['normal']['pass_rate_sd']:.2f} scam {fr['scam']['pass_rate']:.2f}±{fr['scam']['pass_rate_sd']:.2f}")

    tdir = root / "lpw_t2"
    if "lpw" in models and not a.no_temporal and (tdir / "model.pt").exists():
        ensure_recalibrated(pilot, "lpw", tdir, cutoff_months=2, log=log)  # before the reuse check below
    if "temporal" in prev and reusable("lpw", "lpw_t2"):
        res["temporal"] = prev["temporal"]
    elif "lpw" in models and not a.no_temporal:
        cut = cutoff_ts(store, 2)
        if not (tdir / "model.pt").exists():
            log("train lpw with a 2-month temporal hold-out")
            train(pilot, "lpw", tdir, epochs=a.epochs, cutoff_months=2, log=log)
        res["temporal"] = {"cutoff": cut,
                           "lpw_full_window": evaluate(load(pilot, "lpw", root / "lpw", store), store, a.split, views=views, min_t0=cut),
                           "lpw_trained_before_cutoff": evaluate(load(pilot, "lpw", tdir, store), store, a.split, views=views, min_t0=cut)}

    res["exit"] = exit_criteria(res)
    (root / f"results_{a.split}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"exit criteria: {json.dumps(res['exit'])}")


def generate_replicas(model, pops: list[list[dict]], store, seed, log):
    """Generate several populations in one call (they share rounds, so the long tail of a few active streams is not
    paid once per population). Streams are independent, so replica r is exactly a population drawn on its own."""
    sizes = [len(p) for p in pops]
    (ag, ev), info = generate(model, [c for p in pops for c in p], store, seed=seed, log=log)
    out, lo = [], 0
    for n in sizes:
        sel = (pl.col("agent") >= lo) & (pl.col("agent") < lo + n)
        out.append((ag.filter(sel).with_columns(pl.col("agent") - lo), ev.filter(sel).with_columns(pl.col("agent") - lo)))
        lo += n
    return out, info


def write_replicas(reps, path: Path):
    pl.concat([e.with_columns(pl.lit(r, pl.Int16).alias("replica")) for r, (_, e) in enumerate(reps)]).write_parquet(path)


def average_fidelity(fids: list[dict]) -> dict:
    """Per group: mean and sd of the pass rate over replicas, each metric's pass frequency and mean synth/floor
    ratio, and the per-replica results."""
    out = {}
    for g in fids[0]:
        rates = [f[g]["pass_rate"] for f in fids]
        mets = {}
        for mid in fids[0][g]:
            if not isinstance(fids[0][g][mid], dict):
                continue
            mets[mid] = {"pass_freq": float(np.mean([f[g][mid]["pass"] for f in fids])),
                         "ratio_mean": float(np.mean([f[g][mid]["ratio"] for f in fids])),
                         "majority_pass": bool(np.mean([f[g][mid]["pass"] for f in fids]) > 0.5)}
        out[g] = {**mets, "pass_rate": float(np.mean(rates)), "pass_rate_sd": float(np.std(rates)),
                  "pass_rates": rates, "n_real": fids[0][g]["n_real"], "n_synth": fids[0][g]["n_synth"],
                  "replicas": [f[g] for f in fids]}
    return out


def average_parity(pars: list[dict]) -> dict:
    """Passes when the mean per-replica |ΔAUC| is within 0.05 (signed deltas could cancel across replicas)."""
    d = np.array([p["delta"] for p in pars])
    return {"auc_real": pars[0]["auc_real"], "auc_synth": float(np.mean([p["auc_synth"] for p in pars])),
            "delta": float(d.mean()), "delta_sd": float(d.std()), "abs_delta_mean": float(np.abs(d).mean()),
            "pass": bool(np.abs(d).mean() <= 0.05), "replicas": pars}


def exit_criteria(res) -> dict:
    n = res["nll"]
    out = {}
    if all(k in n for k in ("lpw", "b0", "b1")):
        out["beats_b0_b1_full"] = bool(n["lpw"]["all"]["nll_full"] < min(n["b0"]["all"]["nll_full"], n["b1"]["all"]["nll_full"]))
    if "b2" in n and "lpw" in n:
        out["beats_b2_shared"] = bool(n["lpw"]["all"]["nll_shared"] < n["b2"]["all"]["nll_shared"])
    f = res["fidelity"].get("lpw/paired")
    if f:
        out["m_pass_normal"] = f["normal"]["pass_rate"]
        out["m_pass_scam"] = f["scam"]["pass_rate"]
        out["m_pass_ok"] = bool(min(f["normal"]["pass_rate"], f["scam"]["pass_rate"]) >= 0.7)
    p = res["parity"].get("lpw/paired")
    if p:
        out["role_auc_ok"] = p["pass"]
    out["all"] = bool(out) and all(v for k, v in out.items() if isinstance(v, bool))
    return out


if __name__ == "__main__":
    main()
