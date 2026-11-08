"""Shared loading for the P3a scripts: real observed frames and synthetic worlds."""
import os
from pathlib import Path

import numpy as np
import polars as pl

from lpw import sealed
from lpw.p1.store import build_store
from lpw.p3.features import node_features
from lpw.p3.observe import observe_real, observe_synth

PILOT = Path("data/pilot_2018")
ROOT = Path(os.environ.get("P3A_ROOT", "runs/p3a/2018"))  # override only for plumbing smoke tests


def store():
    return build_store(PILOT)


def store_for(split):
    """Guarded store with train and one evaluation split only (joint addendum §J7); test needs lpw.sealed.authorize()."""
    sealed.require(split)
    return build_store(PILOT, splits=("train", split))


def group_ids(st, agents) -> np.ndarray:
    ids = pl.Series([int(x) for x in st.node_id[agents]], dtype=pl.UInt64).implode()
    ag = pl.scan_parquet(PILOT / "agents.parquet").filter(pl.col("node_id").is_in(ids)).select("node_id", "group_id").collect()
    gid = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    return np.array([gid[int(st.node_id[a])] for a in agents])


def real(st, split, owned_only=False):
    """(observed agents, observed events, node features) of a real split. Never called with 'test' before the
    authorized sealed evaluation."""
    sealed.require(split)
    ag, ev = observe_real(st, st.agents(split), owned_only=owned_only)
    return ag, ev, node_features(ag, ev, st.w_start, st.w_end)


def load_pop(world, part):
    z = np.load(ROOT / f"world{world}" / f"pop_{part}.npz")
    keys = ("role", "cohort", "origin", "b0", "age_days", "plan_n", "plan_a")
    conds = [dict(zip(keys, v)) for v in zip(*(z[k].tolist() for k in keys))]
    for c, p in zip(conds, z["prof"]):
        c["prof"] = p
    return conds, z["raw"]


def raw_events(world, arm, part):
    return pl.read_parquet(ROOT / f"world{world}" / f"{arm}_{part}.parquet").drop("replica")


CAP = 10000
MAX_CAPPED_SHARE = 0.001  # prereg §9 safety stop


def synth(st, world, arm, part, drop_capped=False, enforce_cap=True):
    """(observed agents, observed events, node features, O info) of one synthetic population. drop_capped removes
    streams that reached the generation cap (sensitivity analysis, prereg §9). enforce_cap=False lets Checkpoint 0
    report a cap-share failure instead of aborting."""
    conds, raw = load_pop(world, part)
    ev = raw_events(world, arm, part)
    capped = np.flatnonzero(np.bincount(ev["agent"].to_numpy(), minlength=len(conds)) >= CAP)
    if enforce_cap:  # on the original population size, before any removal
        assert len(capped) <= MAX_CAPPED_SHARE * len(conds), f"world {world} {arm}/{part}: {len(capped)} capped streams (prereg §9 stop)"
    if drop_capped and len(capped):
        keep = np.setdiff1d(np.arange(len(conds)), capped)
        remap = np.full(len(conds), -1); remap[keep] = np.arange(len(keep))
        conds, raw = [conds[i] for i in keep], raw[keep]
        ev = ev.filter(~pl.col("agent").is_in(capped.tolist()))
        ev = ev.with_columns(agent=pl.Series(remap[ev["agent"].to_numpy()]))
    ag, ev, info = observe_synth(st, conds, raw, ev)
    info["n_capped"] = int(len(capped))
    return ag, ev, node_features(ag, ev, st.w_start, st.w_end), info
