"""Generate P3a world w (reports/p3a_prereg.md §3): a train-size and a val-size (holdout) sampled population, each
run through B and kct with the same population and the same generation seed (common random numbers).

usage: uv run python scripts/p3a/generate.py --world W [--arms B,kct] [--root runs/p3a/2018]
Writes ROOT/world{W}/{pop_train,pop_hold}.npz and ROOT/world{W}/{arm}_{train,hold}.parquet + gen.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from lpw.p1.atomic import write_text
from lpw.p1.run import generate_replicas, write_replicas
from lpw.p1.store import build_store
from lpw.p1.train import load
from lpw.p3.population import Population

PILOT = Path("data/pilot_2018")
ARMS = {"B": ("lpw_plan_prog", "runs/p1_r0/2018/lpw_plan_prog_rp"),
        "kct": ("lpw_plan_prog_shallow_kct", "runs/p1_shallow/2018/kct")}
POP_SEED = {"train": 1000, "hold": 2000}   # + world
GEN_SEED = 100                             # + world

ap = argparse.ArgumentParser()
ap.add_argument("--world", type=int, required=True)
ap.add_argument("--arms", default="B,kct")
ap.add_argument("--root", default="runs/p3a/2018")
ap.add_argument("--smoke", type=int, default=0, help="plumbing test: populations of this size (never for readouts)")
a = ap.parse_args()
st = build_store(PILOT)
pop = Population(st, Path(a.root) / "population")
out = Path(a.root) / f"world{a.world}"
out.mkdir(parents=True, exist_ok=True)
sizes = {"train": len(st.agents("train")), "hold": len(st.agents("val"))}
if a.smoke:
    sizes = {"train": a.smoke, "hold": a.smoke // 2}
pops = {}
for part in ("train", "hold"):
    conds, raw = pop.sample(sizes[part], POP_SEED[part] + a.world)
    pops[part] = conds
    np.savez(out / f"pop_{part}.npz", raw=raw, prof=np.stack([c["prof"] for c in conds]),
             **{k: np.array([c[k] for c in conds]) for k in ("role", "cohort", "origin", "b0", "age_days", "plan_n", "plan_a")})
info = {"world": a.world, "sizes": sizes, "pop_seeds": {k: v + a.world for k, v in POP_SEED.items()}, "gen_seed": GEN_SEED + a.world,
        "population": pop.meta, "arms": {}}
for arm in a.arms.split(","):
    name, ckpt = ARMS[arm]
    model = load(PILOT, name, Path(ckpt), st)
    model.plan_inference = True
    t = time.time()
    reps, gi = generate_replicas(model, [pops["train"], pops["hold"]], st, seed=GEN_SEED + a.world, log=lambda *x: None)
    for part, (_, ev) in zip(("train", "hold"), reps):
        write_replicas([(None, ev)], out / f"{arm}_{part}.parquet")
    info["arms"][arm] = {"config": name, "ckpt": ckpt, "sec": round(time.time() - t), "events": [e.height for _, e in reps],
                         "gen_info": gi}
    print(f"world {a.world} {arm}: {[e.height for _, e in reps]} events, {time.time() - t:.0f}s", flush=True)
    del model
write_text(out / "gen.json", json.dumps(info, indent=1, default=float))
