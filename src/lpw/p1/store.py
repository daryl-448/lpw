"""Per-agent event streams for Phase 1 (SPEC §6.2, §5.4), built from a pilot slice.

One flat array per field, agents contiguous, each agent's events sorted by (ts, block_number, tx_hash, OUT first).
OUT-first matters only for self-transfers (one tx gives an OUT row and an IN_agent row with identical keys):
the owned OUT must come before its own delivery.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from lpw.p1.timelaw import dt_bin, hour_dow

OUT, IN_EXO, IN_AGENT, PROFILE, PAD = 0, 1, 2, 3, 4
REPEAT, VENUE, NEW = 0, 1, 2
PERIPHERY, TIER_VENUE, TIER_AGENT = 0, 1, 2
DUST_LOG10 = -4.0  # 1e14 wei
SUPPLY_ETH = 1.2e8  # total ETH supply: no transfer, balance or partner total can exceed it (an environment bound,
SUPPLY_LOG10 = float(np.log10(SUPPLY_ETH))  # like the dust floor; no real value comes near it)

PROFILE_FIELDS = [
    "days_active", "normal_sent_cnt", "normal_recv_cnt", "normal_failed_cnt", "normal_to_contract_cnt", "normal_to_eoa_cnt",
    "erc20_sent_cnt", "erc20_recv_cnt", "internal_out_cnt", "internal_in_cnt", "uniq_peers_cnt", "sessions_cnt",
    "eth_sent_sum", "eth_recv_sum", "tx_fee_eth_sum", "internal_out_value_eth_sum", "internal_in_value_eth_sum",
    "age_days", "days_since_active", "balance0",
]
SIGNED_FIELDS = {"balance0"}


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def profile_transform(raw: np.ndarray) -> np.ndarray:
    """Raw profile fields (n, 20) → log-scale features (signed log for the balance, log1p for the rest)."""
    out = np.empty_like(raw, dtype=np.float64)
    for i, f in enumerate(PROFILE_FIELDS):
        out[:, i] = slog(raw[:, i]) if f in SIGNED_FIELDS else np.log1p(np.clip(raw[:, i], 0, None))
    return out


def month_index(ts: np.ndarray, w_start: int) -> np.ndarray:
    m = np.floor(np.asarray(ts, dtype=np.float64)).astype(np.int64).astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    m0 = np.array([w_start], dtype="datetime64[s]").astype("datetime64[M]").astype(np.int64)[0]
    return np.clip(m - m0, 0, 11)


@dataclass
class Store:
    """Agents and their events. Agent a's events are rows off[a]:off[a+1]; its partner entries are eoff[a]:eoff[a+1]."""
    w_start: int
    w_end: int
    node_id: np.ndarray
    role: np.ndarray        # 1 = scam
    cohort: np.ndarray      # 1 = born in W
    split: np.ndarray       # 'train' / 'val' / 'test'
    origin: np.ndarray      # W start (pre-existing) or activation time (born in W)
    obs_end: np.ndarray
    activity_end: np.ndarray
    prof_raw: np.ndarray    # (A, 20) raw profile fields (zeros for born in W)
    b0: np.ndarray          # initial observed balance
    off: np.ndarray
    kind: np.ndarray
    ts: np.ndarray
    cls: np.ndarray
    ent: np.ndarray         # partner entry index within the agent, numbered by first appearance
    venue: np.ndarray       # venue index of the counterparty, -1 if not a venue
    amt: np.ndarray         # log10 ETH
    eth: np.ndarray
    fee: np.ndarray
    bal: np.ndarray         # observed balance before the event
    eoff: np.ndarray
    e_tier: np.ndarray
    e_sc: np.ndarray        # sender-capable (may appear as the source of an IN_exo)
    e_venue: np.ndarray
    venue_sc: np.ndarray    # (V,)
    venue_ids: np.ndarray
    prof_mu: np.ndarray = field(default=None)
    prof_sd: np.ndarray = field(default=None)
    plan_n: np.ndarray = field(default=None)   # per-agent plan latent (lpw.p1.plan); used only by plan models
    plan_a: np.ndarray = field(default=None)

    @property
    def n_agents(self) -> int:
        return len(self.node_id)

    @property
    def n_venues(self) -> int:
        return len(self.venue_sc)

    def n_events(self, a: int) -> int:
        return int(self.off[a + 1] - self.off[a])

    def view(self, a: int) -> dict:
        s, e = self.off[a], self.off[a + 1]
        es, ee = self.eoff[a], self.eoff[a + 1]
        v = {k: getattr(self, k)[s:e] for k in ("kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal")}
        v.update(e_tier=self.e_tier[es:ee], e_sc=self.e_sc[es:ee], e_venue=self.e_venue[es:ee])
        v.update(origin=int(self.origin[a]), obs_end=int(self.obs_end[a]), role=int(self.role[a]), cohort=int(self.cohort[a]),
                 prof=self.prof_vec(a), bal_end=self.bal_end(a))
        if self.plan_n is not None:
            v.update(plan_n=int(self.plan_n[a]), plan_a=int(self.plan_a[a]))
        return v

    def bal_end(self, a: int) -> float:
        s, e = self.off[a], self.off[a + 1]
        if e == s:
            return float(self.b0[a])
        k = self.kind[e - 1]
        return float(self.bal[e - 1] + (self.eth[e - 1] if k != OUT else -self.eth[e - 1] - self.fee[e - 1]))

    def prof_vec(self, a: int) -> np.ndarray:
        return profile_vector(self.prof_raw[a:a + 1], self.cohort[a:a + 1], self.prof_mu, self.prof_sd)[0]

    def agents(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def profile_vector(raw: np.ndarray, cohort: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Model input: standardized log profile (zeros for born in W) plus the born flag."""
    z = (profile_transform(raw) - mu) / sd
    z[cohort == 1] = 0.0
    return np.concatenate([z, cohort[:, None].astype(np.float64)], 1).astype(np.float32)


def _ts(col: pl.Expr) -> pl.Expr:
    return col.dt.epoch("s")


def build_store(pilot: Path, cutoff: int | None = None, raw: Path = Path("data/raw")) -> Store:
    """Build the store. With `cutoff`, events at or after it are dropped and observation ends there (temporal check).
    Labelled contracts (from `raw`) are not sender-capable: contracts never send native ETH in E1 (SPEC §5.1, §6.3)."""
    meta = json.loads((pilot / "meta.json").read_text())
    w_start = int(datetime.fromisoformat(meta["W_START"]).timestamp())
    w_end = int(datetime.fromisoformat(meta["W_END"]).timestamp())
    ag = pl.read_parquet(pilot / "agents.parquet").sort("node_id")
    prof = pl.read_parquet(pilot / "profiles.parquet")
    ven = pl.read_parquet(pilot / "venues.parquet").sort("cp")
    ev = pl.read_parquet(pilot / "events.parquet")
    ag = ag.join(prof.select(["node_id", *PROFILE_FIELDS]), on="node_id", how="left")

    venue_ids = ven["cp"].to_numpy()
    venue_sc = ven["sender_capable"].to_numpy().astype(bool)
    vidx = {int(c): i for i, c in enumerate(venue_ids)}
    contracts = set(pl.read_parquet(raw / "gnn_dataset/labels/targets_global.parquet", columns=["node_id", "is_contract"])
                    .filter(pl.col("is_contract") == 1)["node_id"].to_list())

    origin = np.where(ag["cohort"].to_numpy() == "born", ag.select(_ts(pl.col("activity_start_ts"))).to_series().to_numpy(), w_start)
    obs_end = ag.select(_ts(pl.col("obs_end"))).to_series().to_numpy().astype(np.int64)
    act_end = ag.select(_ts(pl.col("activity_end_ts"))).to_series().to_numpy().astype(np.int64)
    if cutoff is not None:
        ended = (obs_end > w_end) & (act_end < cutoff)  # activity truly ended before the cutoff: keep the data end
        obs_end = np.where(ended, obs_end, np.minimum(obs_end, cutoff))
        ev = ev.filter(pl.col("ts") < cutoff)

    ev = (ev.with_columns((pl.col("dir") == "IN").cast(pl.Int8).alias("_in"))
          .sort("node_id", "ts", "block_number", "tx_hash", "_in"))
    pos = {int(n): i for i, n in enumerate(ag["node_id"].to_numpy())}
    a_of = np.array([pos[int(n)] for n in ev["node_id"].to_numpy()], dtype=np.int64)
    assert np.all(np.diff(a_of) >= 0)
    counts = np.bincount(a_of, minlength=len(pos))
    off = np.concatenate([[0], np.cumsum(counts)])

    kind = ev["kind"].replace_strict({"OUT": OUT, "IN_exo": IN_EXO, "IN_agent": IN_AGENT}, return_dtype=pl.Int8).to_numpy()
    ts = ev["ts"].to_numpy().astype(np.int64)
    cp = ev["cp"].to_numpy()
    eth = ev["eth"].to_numpy()
    fee = ev["fee_eth"].to_numpy()
    cp_agent = ev["cp_is_agent"].to_numpy()
    roles = ag["is_scam"].to_numpy().astype(np.int64)
    cohort = (ag["cohort"].to_numpy() == "born").astype(np.int64)
    prof_raw = ag.select(PROFILE_FIELDS).fill_null(0).to_numpy().astype(np.float64)
    prof_raw[cohort == 1] = 0.0
    b0 = np.where(cohort == 1, 0.0, ag["balance0"].fill_null(0).to_numpy())

    n = len(ts)
    cls = np.empty(n, np.int8)
    ent = np.empty(n, np.int32)
    venue = np.full(n, -1, np.int16)
    bal = np.empty(n, np.float64)
    eoff = [0]
    e_tier, e_sc, e_venue = [], [], []
    for a in range(len(pos)):
        s, e = off[a], off[a + 1]
        table: dict[int, int] = {}
        b = b0[a]
        for i in range(s, e):
            c = int(cp[i])
            vi = vidx.get(c, -1)
            venue[i] = vi
            if c in table:
                cls[i] = REPEAT
                ent[i] = table[c]
            else:
                ent[i] = table[c] = len(table)
                cls[i] = VENUE if vi >= 0 else NEW
                tier = TIER_AGENT if cp_agent[i] else (TIER_VENUE if vi >= 0 else PERIPHERY)
                e_tier.append(tier)
                e_sc.append(bool(venue_sc[vi]) if vi >= 0 else not (cp_agent[i] or c in contracts))
                e_venue.append(vi)
            bal[i] = b
            b += -eth[i] - fee[i] if kind[i] == OUT else eth[i]
        eoff.append(eoff[-1] + len(table))
        if e > s:
            assert ts[s] >= origin[a], "event before origin"
    # the IN_exo mask must never exclude an observed source (SPEC §6.3)
    e_sc_arr = np.array(e_sc, bool)
    eoff_arr = np.array(eoff, np.int64)
    flat_e = np.repeat(eoff_arr[:-1], counts) + ent
    assert np.all(e_sc_arr[flat_e[kind == IN_EXO]]), "an IN_exo source is masked as not sender-capable"
    amt = np.log10(eth)
    assert amt.min() >= DUST_LOG10 - 1e-9

    split = ag["split"].to_numpy()
    st = Store(w_start, w_end, ag["node_id"].to_numpy(), roles, cohort, split, origin.astype(np.int64), obs_end, act_end,
               prof_raw, b0, off.astype(np.int64), kind, ts, cls, ent, venue, amt.astype(np.float32), eth, fee, bal,
               eoff_arr, np.array(e_tier, np.int8), e_sc_arr, np.array(e_venue, np.int16), venue_sc, venue_ids)
    tr = (split == "train") & (cohort == 0)
    x = profile_transform(prof_raw[tr])
    st.prof_mu, st.prof_sd = x.mean(0), x.std(0) + 1e-6
    from lpw.p1.plan import plan_targets
    st.plan_n, st.plan_a, _ = plan_targets(st)
    return st


def derived(view: dict, w_start: int) -> dict:
    """Per-event derived fields: dt from the previous token (or origin), its bin, hour, dow, window month."""
    ts = view["ts"]
    prev = np.concatenate([[view["origin"]], ts[:-1]])
    dt = ts - prev
    h, d = hour_dow(ts)
    return dict(dt=dt, dtbin=dt_bin(dt), hour=h, dow=d, month=month_index(ts, w_start))
