"""Phase 1 pipeline for one slice (SPEC §8 P1).

Usage: uv run python -m lpw.p1.run --pilot data/pilot --tag 2021 --split val
       (--split test only once, after the code is frozen; SPEC §7 split discipline)

Steps: train every model (early stop on val), exact next-event evaluation, profile generator, paired and
unconditional generation, M1–M16 with noise floors, role-signal parity, temporal check, exit criteria.
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
    res = {"pilot": str(pilot), "split": a.split, "nll": {}, "fidelity": {}, "parity": {}, "gen_info": {}}

    for m in models:
        if not (root / m / "model.pt").exists():
            log(f"train {m}")
            info = train(pilot, m, root / m, epochs=a.epochs, log=log)
            log(f"  {m}: best val {info['best_val']:.4f} at epoch {info['best_epoch']}")
    for m in models:
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
        (sag, sev), info = generate(model, conds, store, log=log)
        sev.write_parquet(root / f"gen_{m}_{a.split}_paired.parquet")
        res["gen_info"][f"{m}/paired"] = {**info, "events": sev.height, "real_events": rev.height, "sec": round(time.time() - t)}
        syn = Sample(sag, sev, store.w_start)
        res["fidelity"][f"{m}/paired"] = compare(real, syn, groups, paired=True)
        res["parity"][f"{m}/paired"] = role_parity(train_X, train_y, real_X, real_y,
                                                   agent_table(syn, np.stack([c["prof"] for c in conds])), real_y)
        fr = res["fidelity"][f"{m}/paired"]
        log(f"gen {m}: {sev.height} events vs {rev.height} real ({time.time() - t:.0f}s); pass normal {fr['normal']['pass_rate']:.2f} "
            f"scam {fr['scam']['pass_rate']:.2f}; parity {res['parity'][f'{m}/paired']}")

    if "lpw" in models:
        pg = ProfileGen(store, log=log)
        rng = np.random.default_rng(11)
        roles = store.role[agents]
        uconds = pg.population(roles, rng)
        model = load(pilot, "lpw", root / "lpw", store)
        (uag, uev), info = generate(model, uconds, store, seed=13, log=log)
        uev.write_parquet(root / f"gen_lpw_{a.split}_uncond.parquet")
        res["gen_info"]["lpw/uncond"] = {**info, "events": uev.height}
        usyn = Sample(uag, uev, store.w_start)
        ug = groups_for(uag, age_edges)
        res["fidelity"]["lpw/uncond"] = compare(real, usyn, groups, paired=False, synth_groups=ug)
        res["parity"]["lpw/uncond"] = role_parity(train_X, train_y, real_X, real_y,
                                                  agent_table(usyn, np.stack([c["prof"] for c in uconds])), roles)
        res["profile_gen"] = {"val_nll": pg.val_nll, "epochs": pg.epochs}
        fr = res["fidelity"]["lpw/uncond"]
        log(f"uncond lpw: {uev.height} events; pass normal {fr['normal']['pass_rate']:.2f} scam {fr['scam']['pass_rate']:.2f}")

    if "lpw" in models and not a.no_temporal:
        tdir = root / "lpw_t2"
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
