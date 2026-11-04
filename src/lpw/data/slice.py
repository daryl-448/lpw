"""Build the Phase 0 pilot slice (SPEC §5.3) from data/raw into data/pilot.

Order (frozen, one way):
  1. label-blind structure: hubs, agents, venues, groups
  2. split by group, stratified by role
  3. (train-only fitting happens in later phases, not here)

Usage: uv run python -m lpw.data.slice [--raw data/raw] [--out data/pilot] [--window 2021-05:2022-05]
(the window end month is exclusive)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import networkx as nx

W_START = datetime(2021, 5, 1, tzinfo=timezone.utc)
W_END = datetime(2022, 5, 1, tzinfo=timezone.utc)


def set_window(spec: str) -> None:
    """Set the module window from 'YYYY-MM:YYYY-MM' (end exclusive). Other modules read lpw.data.slice.W_START/W_END."""
    global W_START, W_END
    a, b = spec.split(":")
    W_START = datetime(int(a[:4]), int(a[5:7]), 1, tzinfo=timezone.utc)
    W_END = datetime(int(b[:4]), int(b[5:7]), 1, tzinfo=timezone.utc)


DATA_END = datetime(2025, 8, 18, tzinfo=timezone.utc)  # last edge ts in the dataset (1755475139)
HUB_MONTHLY_EVENTS = 2000
S_PUB = 50
MAX_VENUES = 1024
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 20261104
EDGE_COLS = ["src_id", "dst_id", "ts", "value_wei", "tx_fee_wei", "block_number", "tx_hash"]


def months(start: datetime, end: datetime) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) < (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def edge_files(raw: Path) -> list[Path]:
    return [raw / f"gnn_dataset/edges_by_month/month={m}/edges.parquet" for m in months(W_START, W_END)]


def load_labels(raw: Path) -> pl.DataFrame:
    lab = pl.read_parquet(raw / "gnn_dataset/labels/targets_global.parquet")
    csv = pl.read_csv(raw / "addr_labels_balanced.csv.zst").unique("address", keep="first")
    ts = lambda c: pl.col(c).str.to_datetime("%Y-%m-%d %H:%M:%S %Z", strict=False).dt.replace_time_zone("UTC").alias(c)
    csv = csv.select("address", "description", ts("activity_start_ts"), ts("activity_end_ts"))
    return lab.join(csv, on="address", how="left")


def seed_month_counts(files: list[Path], seeds: pl.Series) -> pl.DataFrame:
    """Per (seed, calendar month) count of E1 events, seed on either side."""
    parts = []
    for f in files:
        e = pl.scan_parquet(f).select("src_id", "dst_id", "ts")
        month = pl.from_epoch("ts", time_unit="s").dt.strftime("%Y-%m").alias("month")
        side = lambda c: e.filter(pl.col(c).is_in(seeds.implode())).select(pl.col(c).alias("node_id"), month)
        parts.append(pl.concat([side("src_id"), side("dst_id")]).group_by("node_id", "month").len().collect())
    return pl.concat(parts).group_by("node_id", "month").agg(pl.col("len").sum())


def agent_events(files: list[Path], agents: pl.Series) -> pl.DataFrame:
    """One row per (agent, transaction) in W: OUT rows for the sender, IN rows for the receiver."""
    parts = []
    lo, hi = int(W_START.timestamp()), int(W_END.timestamp())
    for f in files:
        e = (
            pl.scan_parquet(f)
            .select(EDGE_COLS)
            .filter(pl.col("ts").is_between(lo, hi, closed="left"))
            .filter(pl.col("src_id").is_in(agents.implode()) | pl.col("dst_id").is_in(agents.implode()))
            .with_columns(
                (pl.col("value_wei").cast(pl.Float64) / 1e18).alias("eth"),
                (pl.col("tx_fee_wei").cast(pl.Float64) / 1e18).alias("fee_eth"),
            )
            .drop("value_wei", "tx_fee_wei")
        )
        out = e.filter(pl.col("src_id").is_in(agents.implode())).select(
            pl.col("src_id").alias("node_id"), pl.col("dst_id").alias("cp"), pl.lit("OUT").alias("dir"),
            "ts", "block_number", "tx_hash", "eth", "fee_eth")
        inn = e.filter(pl.col("dst_id").is_in(agents.implode())).select(
            pl.col("dst_id").alias("node_id"), pl.col("src_id").alias("cp"), pl.lit("IN").alias("dir"),
            "ts", "block_number", "tx_hash", "eth", "fee_eth")
        parts.append(pl.concat([out, inn]).collect())
    ev = pl.concat(parts)
    ev = ev.with_columns(pl.col("cp").is_in(agents.implode()).alias("cp_is_agent"))
    ev = ev.with_columns(
        pl.when((pl.col("dir") == "IN") & pl.col("cp_is_agent")).then(pl.lit("IN_agent"))
        .when(pl.col("dir") == "IN").then(pl.lit("IN_exo"))
        .otherwise(pl.lit("OUT")).alias("kind"))
    return ev.sort("node_id", "ts", "block_number", "tx_hash")


def find_groups(ev: pl.DataFrame, venues: pl.Series) -> pl.DataFrame:
    """Groups = Louvain communities of the agent–counterparty graph after removing venues (label-blind, fixed seed).

    Plain connected components do not work: Phase 0 found one component holding 47% of agents, and it persists
    for every venue threshold above 2. The component id is kept as `component_id` for reference.
    """
    # polars unique() row order is not stable across runs, and Louvain depends on insertion order: sort everything.
    pairs = ev.filter(~pl.col("cp").is_in(venues.implode())).select("node_id", "cp").unique().sort("node_id", "cp")
    agent_ids = ev["node_id"].unique().sort()
    agent_set = set(agent_ids.to_list())
    G = nx.Graph()
    G.add_nodes_from(("a", n) for n in agent_ids.to_list())
    G.add_edges_from((("a", n), ("a", c) if c in agent_set else ("c", c)) for n, c in pairs.iter_rows())
    comms = sorted(nx.community.louvain_communities(G, seed=SEED), key=lambda cs: min(cs))
    comm = pl.DataFrame([(n[1], i) for i, cs in enumerate(comms) for n in cs if n[0] == "a"],
                        schema=[("node_id", pl.UInt64), ("group_id", pl.Int64)], orient="row")
    comp = {n[1]: i for i, cs in enumerate(sorted(nx.connected_components(G), key=lambda cs: min(cs))) for n in cs if n[0] == "a"}
    comp_df = pl.DataFrame({"node_id": list(comp), "component_id": list(comp.values())}, schema={"node_id": pl.UInt64, "component_id": pl.Int64})
    return pl.DataFrame({"node_id": agent_ids}).join(comm, on="node_id").join(comp_df, on="node_id")


DAILY_SUM = ["normal_sent_cnt", "normal_recv_cnt", "normal_failed_cnt", "normal_to_contract_cnt", "normal_to_eoa_cnt",
             "erc20_sent_cnt", "erc20_recv_cnt", "internal_out_cnt", "internal_in_cnt", "uniq_peers_cnt", "sessions_cnt"]
DAILY_ETH = ["eth_sent_sum", "eth_recv_sum", "tx_fee_eth_sum", "internal_out_value_eth_sum", "internal_in_value_eth_sum"]


def profiles(raw: Path, agents: pl.DataFrame) -> pl.DataFrame:
    """Pre-window profile from daily records strictly before W start (SPEC §5.3). Exact duplicate rows are dropped."""
    d = (pl.scan_parquet(raw / "lstm_dataset/daily_filtered.parquet")
         .filter(pl.col("node_id").is_in(agents["node_id"].implode()), pl.col("day") < W_START.date())
         .drop("address", "week", "month").unique()
         .group_by("node_id")
         .agg(pl.len().alias("days_active"), pl.col("day").max().alias("last_active_day"),
              *[pl.col(c).sum() for c in DAILY_SUM], *[pl.col(c).cast(pl.Float64).sum() for c in DAILY_ETH])
         .collect())
    p = agents.select("node_id", "cohort", "activity_start_ts").join(d, on="node_id", how="left")
    p = p.with_columns(
        ((pl.lit(W_START) - pl.col("activity_start_ts")).dt.total_seconds() / 86400).clip(lower_bound=0).alias("age_days"),
        ((pl.lit(W_START.date()) - pl.col("last_active_day")).dt.total_days()).alias("days_since_active"),
        (pl.col("eth_recv_sum") + pl.col("internal_in_value_eth_sum") - pl.col("eth_sent_sum")
         - pl.col("internal_out_value_eth_sum") - pl.col("tx_fee_eth_sum")).alias("balance0"),
        pl.col("days_active").is_null().alias("profile_missing"),
    )
    # born-in-W agents have an empty profile by definition
    fill = [c for c in p.columns if c not in ("node_id", "cohort", "activity_start_ts", "last_active_day", "days_since_active", "profile_missing")]
    born = pl.col("cohort") == "born"
    return p.with_columns([pl.when(born).then(0).otherwise(pl.col(c)).alias(c) for c in fill]).drop("activity_start_ts")


def split_groups(agents: pl.DataFrame) -> pl.DataFrame:
    """Assign whole groups to splits, greedily balancing per-role counts against the target fractions."""
    g = agents.group_by("group_id").agg(pl.col("is_scam").sum().alias("n_scam"), (1 - pl.col("is_scam")).sum().alias("n_norm")).sort("group_id")
    rng = np.random.default_rng(SEED)
    g = g.with_columns(pl.Series("r", rng.random(g.height))).sort(pl.col("n_scam") + pl.col("n_norm"), "r", descending=[True, False])
    tot = np.array([g["n_scam"].sum(), g["n_norm"].sum()], dtype=float)
    names = list(SPLIT_FRACS)
    target = np.array([[SPLIT_FRACS[s] * tot[0], SPLIT_FRACS[s] * tot[1]] for s in names])
    have = np.zeros_like(target)
    assign = []
    for ns, nn in zip(g["n_scam"].to_list(), g["n_norm"].to_list()):
        v = np.array([ns, nn], dtype=float)
        deficit = ((target - have) / np.maximum(target, 1) * v).sum(axis=1)
        k = int(np.argmax(deficit))
        have[k] += v
        assign.append(names[k])
    return agents.join(g.select("group_id", pl.Series("split", assign)), on="group_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--window", default="2021-05:2022-05")
    args = ap.parse_args()
    set_window(args.window)
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = edge_files(raw)
    lab = load_labels(raw)
    seeds = lab.filter(pl.col("activity_start_ts").is_not_null())["node_id"]

    # 1. label-blind structure
    smc = seed_month_counts(files, seeds)
    smc = smc.filter(pl.col("month").is_in(months(W_START, W_END)))
    peak = smc.group_by("node_id").agg(pl.col("len").max().alias("peak_month_events"), pl.col("len").sum().alias("w_events"))
    peak = peak.join(lab.select("node_id", "is_contract"), on="node_id")
    hubs = peak.filter((pl.col("is_contract") == 0) & (pl.col("peak_month_events") > HUB_MONTHLY_EVENTS))
    agents = peak.filter((pl.col("is_contract") == 0) & (pl.col("peak_month_events") <= HUB_MONTHLY_EVENTS))["node_id"]
    ev = agent_events(files, agents)
    agents = ev["node_id"].unique()

    deg = ev.filter(~pl.col("cp_is_agent")).group_by("cp").agg(pl.col("node_id").n_unique().alias("n_agents"))
    shared = deg.filter(pl.col("n_agents") >= S_PUB).sort("n_agents", descending=True).head(MAX_VENUES - hubs.height)
    venues = pl.concat([hubs.select(pl.col("node_id").alias("cp"), pl.lit(None, pl.UInt32).alias("n_agents"), pl.lit("hub").alias("venue_kind")),
                        shared.select("cp", pl.col("n_agents").cast(pl.UInt32), pl.lit("shared").alias("venue_kind"))]).unique("cp", keep="first")
    sender = ev.filter(pl.col("dir") == "IN").select("cp").unique().with_columns(pl.lit(True).alias("sender_capable"))
    venues = (venues.join(lab.select(pl.col("node_id").alias("cp"), "is_contract", "is_scam"), on="cp", how="left")
              .join(sender, on="cp", how="left").with_columns(pl.col("sender_capable").fill_null(False)))
    groups = find_groups(ev, venues["cp"])

    # agent table: label, cohort, observation end
    a = (groups.join(lab, on="node_id", how="left")
         .with_columns(
             pl.when(pl.col("activity_start_ts") < W_START).then(pl.lit("pre")).otherwise(pl.lit("born")).alias("cohort"),
             pl.when(pl.col("activity_end_ts") < W_END).then(pl.lit(DATA_END)).otherwise(pl.lit(W_END)).alias("obs_end"))
         .join(peak.select("node_id", "peak_month_events", "w_events"), on="node_id", how="left"))

    # 2. split by group
    a = split_groups(a)

    prof = profiles(raw, a)

    ev.write_parquet(out / "events.parquet")
    prof.write_parquet(out / "profiles.parquet")
    a.write_parquet(out / "agents.parquet")
    venues.write_parquet(out / "venues.parquet")
    hubs.join(lab.select("node_id", "is_scam"), on="node_id").write_parquet(out / "hubs.parquet")
    meta = {"W_START": W_START.isoformat(), "W_END": W_END.isoformat(), "DATA_END": DATA_END.isoformat(),
            "HUB_MONTHLY_EVENTS": HUB_MONTHLY_EVENTS, "S_PUB": S_PUB, "MAX_VENUES": MAX_VENUES, "SEED": SEED,
            "n_agents": a.height, "n_events": ev.height, "n_venues": venues.height, "n_hubs": hubs.height}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
