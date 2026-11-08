"""Guarded train / val access for P3b development (reports/p3b_prereg.md, access rule).

The sealed test split never reaches a frame: agent rows are filtered by split inside the parquet scan, and events,
profiles and labelled-contract flags by the kept node ids, before collection (lpw.p1.store.build_store(splits=...)).
Every API here rejects 'test' unless lpw.sealed.authorize() has passed in this process (joint addendum §J7).

Real(...) adds what P3a's observation schema drops: the global counterparty id of every event (raw events.cp), its
transaction identity and the cp_is_agent flag, aligned one-to-one with the store's event order, plus Louvain group ids.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from lpw import sealed
from lpw.p1.store import Store, build_store

PILOT = Path("data/pilot_2018")
DEV_SPLITS = ("train", "val")


@dataclass
class Real:
    st: Store
    cp: np.ndarray        # (n_events,) uint64 global counterparty id, store event order
    block: np.ndarray     # (n_events,) int64
    tx: np.ndarray        # (n_events,) int64 dense transaction code (block, tx_hash), shared across the kept agents
    cp_agent: np.ndarray  # (n_events,) bool: the counterparty is an agent seed (any split; event metadata)
    group: np.ndarray     # (n_agents,) int64 Louvain group id

    def agents(self, split: str) -> np.ndarray:
        sealed.require(split)
        return self.st.agents(split)


def load(splits: tuple[str, ...] = DEV_SPLITS, pilot: Path = PILOT) -> Real:
    for s in splits:  # train / val always; test only after lpw.sealed.authorize() (joint addendum §J7)
        sealed.require(s)
    st = build_store(pilot, splits=splits)
    keep = pl.Series(st.node_id).implode()
    ev = (pl.scan_parquet(pilot / "events.parquet").filter(pl.col("node_id").is_in(keep)).collect()
          .with_columns((pl.col("dir") == "IN").cast(pl.Int8).alias("_in"))
          .sort("node_id", "ts", "block_number", "tx_hash", "_in"))
    assert ev.height == len(st.ts)
    assert (ev["ts"].to_numpy() == st.ts).all() and (ev["eth"].to_numpy() == st.eth).all(), "event alignment"
    tx = ev.select(pl.struct("block_number", "tx_hash").rank("dense").cast(pl.Int64) - 1).to_series().to_numpy()
    ag = pl.scan_parquet(pilot / "agents.parquet").filter(pl.col("node_id").is_in(keep)).select("node_id", "group_id", "split").collect()
    gid = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    assert set(ag["split"].unique().to_list()) <= set(splits)
    return Real(st, ev["cp"].to_numpy(), ev["block_number"].to_numpy(), tx, ev["cp_is_agent"].to_numpy(),
                np.array([gid[int(n)] for n in st.node_id], np.int64))


def real_pop(R: Real, split: str):
    """lpw.p3b.replay.Pop of one real development split (agents in store order within the split)."""
    from lpw.p1.store import NEW, OUT
    from lpw.p3b.replay import Pop
    st = R.st
    ag = R.agents(split)
    idx = np.concatenate([np.arange(st.off[a], st.off[a + 1]) for a in ag])
    off = np.concatenate([[0], np.cumsum(st.off[ag + 1] - st.off[ag])])
    order = np.lexsort(((st.kind[idx] != OUT).astype(np.int8), R.tx[idx], st.ts[idx]))  # (ts, block, tx_hash, OUT-first)
    rank = np.empty(len(idx), np.int64)
    rank[order] = np.arange(len(idx))
    return Pop(group=R.group[ag], role=st.role[ag], cohort=st.cohort[ag], origin=st.origin[ag], off=off,
               sec=st.ts[idx].astype(np.int64), out=st.kind[idx] == OUT, elig=(st.venue[idx] < 0) & ~R.cp_agent[idx],
               first=st.cls[idx] == NEW, ent=st.ent[idx].astype(np.int64), cp=R.cp[idx], rank=rank, w_start=st.w_start, w_end=st.w_end), idx


def clean_agents(R: Real, split: str) -> np.ndarray:
    """Boolean mask over R.agents(split): True if none of the agent's non-venue counterparties occurs among the train
    agents' non-venue counterparties. This is the P0 rule of lpw.data.audit.leakage_report (joint addendum §J4.3)."""
    st = R.st

    def cps(agents):
        idx = np.concatenate([np.arange(st.off[a], st.off[a + 1]) for a in agents])
        idx = idx[st.venue[idx] < 0]
        return idx, R.cp[idx]

    _, tr_cp = cps(R.agents("train"))
    ev_agents = R.agents(split)
    idx, cp = cps(ev_agents)
    hit = np.isin(cp, np.unique(tr_cp))
    owner = np.searchsorted(st.off, idx, side="right") - 1  # store agent of each event
    dirty = np.unique(owner[hit])
    return ~np.isin(ev_agents, dirty)
