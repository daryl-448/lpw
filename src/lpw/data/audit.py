"""Phase 0 audit of the pilot slice (SPEC §8 P0). Writes runs/p0_audit.json.

Usage: uv run python -m lpw.data.audit [--raw data/raw] [--pilot data/pilot] [--out runs/p0_audit.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from lpw.data import slice as sl


def q(s: pl.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    s = s.drop_nulls()
    return {f"p{int(x * 100)}": (None if s.is_empty() else float(s.quantile(x))) for x in qs}


def by_role(df: pl.DataFrame, expr: pl.Expr, name: str) -> dict:
    t = df.group_by("is_scam").agg(expr.alias(name)).sort("is_scam")
    return {("scam" if r["is_scam"] else "normal"): r[name] for r in t.to_dicts()}


def per_agent_stats(ev: pl.DataFrame, a: pl.DataFrame) -> dict:
    owned = ev.filter(pl.col("kind") != "IN_agent")
    cp_n = ev.filter(~pl.col("cp_is_agent")).group_by("cp").agg(pl.col("node_id").n_unique().alias("cp_agents"))
    e = ev.join(cp_n, on="cp", how="left").with_columns(
        (pl.col("cp").cum_count().over("node_id", "cp") > 1).alias("repeat"))
    per = e.group_by("node_id").agg(
        pl.len().alias("n"), (pl.col("dir") == "OUT").mean().alias("out_frac"), pl.col("repeat").mean().alias("repeat_rate"),
        pl.col("cp").n_unique().alias("n_cp"), (pl.col("cp_agents").fill_null(0) >= 5).mean().alias("shared5_frac"),
    ).join(a.select("node_id", "is_scam", "cohort"), on="node_id")
    med = lambda c: by_role(per, pl.col(c).median(), c)
    return {
        "agents_by_role": by_role(per, pl.len(), "n"),
        "events_by_role": by_role(per, pl.col("n").sum(), "n"),
        "median_events": med("n"), "median_out_frac": med("out_frac"), "median_repeat_rate": med("repeat_rate"),
        "median_distinct_cp": med("n_cp"), "median_shared5_frac": med("shared5_frac"),
        "events_per_agent_quantiles": {"normal": q(per.filter(pl.col("is_scam") == 0)["n"]), "scam": q(per.filter(pl.col("is_scam") == 1)["n"])},
        "agents_over_255_events": by_role(per, (pl.col("n") > 255).sum(), "n"),
        "kind_counts": dict(ev.group_by("kind").len().iter_rows()),
        "agent_agent_event_frac": float(ev["cp_is_agent"].mean()),
        "self_transfer_events": int((ev["node_id"] == ev["cp"]).sum()),
        "owned_events": owned.height,
    }


def groups_report(a: pl.DataFrame) -> dict:
    g = a.group_by("group_id").agg(pl.len().alias("size"), pl.col("is_scam").mean().alias("scam_frac"))
    big = g.sort("size", descending=True).head(5)
    sizes = g["size"]
    mixed = g.filter((pl.col("scam_frac") > 0) & (pl.col("scam_frac") < 1))
    splits = a.group_by("split", "is_scam").len().sort("split", "is_scam")
    return {
        "n_groups": g.height, "singletons": int((sizes == 1).sum()), "size_quantiles": q(sizes),
        "largest_groups": big.to_dicts(), "largest_group_agent_share": float(sizes.max() / a.height),
        "mixed_groups": mixed.height, "agents_in_mixed_groups": int(mixed["size"].sum()),
        "split_counts": [r for r in splits.to_dicts()],
    }


def leakage_report(ev: pl.DataFrame, a: pl.DataFrame, venues: pl.DataFrame) -> dict:
    """Non-venue counterparties shared across splits, and test agents that share one with a train agent."""
    sp = (ev.filter(~pl.col("cp").is_in(venues["cp"].implode())).select("node_id", "cp").unique()
          .join(a.select("node_id", "split"), on="node_id"))
    cps = sp.group_by("cp").agg(pl.col("split").n_unique().alias("ns"), pl.col("node_id").n_unique().alias("na")).filter(pl.col("na") >= 2)
    train_cps = sp.filter(pl.col("split") == "train").select("cp").unique()
    out = {"shared_nonvenue_cps": cps.height, "spanning_splits": int((cps["ns"] > 1).sum())}
    for s in ["val", "test"]:
        ids = sp.filter(pl.col("split") == s).join(train_cps, on="cp")["node_id"].unique()
        n = a.filter(pl.col("split") == s)
        out[f"{s}_agents_sharing_cp_with_train"] = int(ids.len())
        out[f"{s}_clean_subset"] = by_role(n.filter(~pl.col("node_id").is_in(ids.implode())), pl.len(), "n")
    return out


def activity_end_check(ev: pl.DataFrame, a: pl.DataFrame) -> dict:
    last = ev.group_by("node_id").agg(pl.col("ts").max().alias("last_ts"), pl.col("ts").min().alias("first_ts"))
    x = a.join(last, on="node_id").with_columns(
        pl.from_epoch("last_ts", time_unit="s").dt.replace_time_zone("UTC").alias("last_dt"),
        pl.from_epoch("first_ts", time_unit="s").dt.replace_time_zone("UTC").alias("first_dt"))
    return {
        "activity_end_null": int(x["activity_end_ts"].is_null().sum()),
        "activity_end_inside_W": by_role(x, (pl.col("activity_end_ts") < sl.W_END).sum(), "n"),
        "last_edge_after_activity_end": int((x["last_dt"] > x["activity_end_ts"]).sum()),
        "first_edge_before_activity_start": int((x["first_dt"] < x["activity_start_ts"]).sum()),
        "lag_last_edge_to_activity_end_days_if_end_in_W": q(x.filter(pl.col("activity_end_ts") < sl.W_END)
                                                            .select((pl.col("activity_end_ts") - pl.col("last_dt")).dt.total_seconds() / 86400).to_series()),
    }


def ledger_check(ev: pl.DataFrame, prof: pl.DataFrame, a: pl.DataFrame) -> dict:
    """Observed balance b0 + Σin − Σ(out + fee) before each event; how often does it go negative?"""
    delta = pl.when(pl.col("dir") == "OUT").then(-(pl.col("eth") + pl.col("fee_eth"))).otherwise(pl.col("eth"))
    x = (ev.join(prof.select("node_id", "balance0"), on="node_id", how="left")
         .with_columns(delta.alias("d"))
         .with_columns((pl.col("balance0").fill_null(0) + pl.col("d").cum_sum().over("node_id")).alias("b_after")))
    x = x.join(a.select("node_id", "is_scam", "cohort"), on="node_id")
    outs = x.filter(pl.col("dir") == "OUT")
    tol = 1e-9
    per = x.group_by("node_id", "is_scam", "cohort").agg((pl.col("b_after") < -tol).any().alias("ever_neg"))
    return {
        "balance0_negative_frac_pre": float((prof.filter(pl.col("cohort") == "pre")["balance0"] < -tol).mean()),
        "balance0_quantiles_pre": q(prof.filter(pl.col("cohort") == "pre")["balance0"]),
        "out_events_leaving_negative_frac": by_role(outs, (pl.col("b_after") < -tol).mean(), "f"),
        "agents_ever_negative_frac": {f"{r['cohort']}/{'scam' if r['is_scam'] else 'normal'}": r["f"] for r in
                                      per.group_by("cohort", "is_scam").agg(pl.col("ever_neg").mean().alias("f")).to_dicts()},
    }


def daily_vs_edges(raw: Path, ev: pl.DataFrame, a: pl.DataFrame) -> dict:
    """Compare in-W native ETH received per agent: daily records vs E1 edges (≥ dust)."""
    d = (pl.scan_parquet(raw / "lstm_dataset/daily_filtered.parquet")
         .filter(pl.col("node_id").is_in(a["node_id"].implode()), pl.col("day") >= sl.W_START.date(), pl.col("day") < sl.W_END.date())
         .drop("address", "week", "month").unique()
         .group_by("node_id").agg(pl.col("eth_recv_sum").cast(pl.Float64).sum().alias("daily_recv"),
                                  pl.col("normal_recv_cnt").sum().alias("daily_recv_cnt"))
         .collect())
    e = ev.filter(pl.col("dir") == "IN").group_by("node_id").agg(pl.col("eth").sum().alias("edge_recv"), pl.len().alias("edge_recv_cnt"))
    x = a.select("node_id").join(e, on="node_id", how="left").join(d, on="node_id", how="left").fill_null(0)
    x = x.filter(pl.col("daily_recv") > 0).with_columns((pl.col("edge_recv") / pl.col("daily_recv")).alias("ratio"),
                                                       (pl.col("edge_recv_cnt") / pl.col("daily_recv_cnt").clip(lower_bound=1)).alias("cnt_ratio"))
    return {"agents_with_daily_recv": x.height, "edge_over_daily_recv_value": q(x["ratio"]), "edge_over_daily_recv_count": q(x["cnt_ratio"]),
            "agents_without_any_daily_row_in_W": int(a.height - d.height)}


def caliper_pairs(ls: np.ndarray, ln: np.ndarray, cal: float) -> int:
    """Maximum number of 1:1 pairs with |score difference| <= cal (exact for a 1-D score: sorted two-pointer greedy)."""
    ls, ln = np.sort(ls), np.sort(ln)
    i = j = pairs = 0
    while i < len(ls) and j < len(ln):
        if abs(ls[i] - ln[j]) <= cal:
            pairs += 1; i += 1; j += 1
        elif ls[i] < ln[j]:
            i += 1
        else:
            j += 1
    return pairs


def confound_report(a: pl.DataFrame, prof: pl.DataFrame) -> dict:
    x = a.select("node_id", "is_scam", "split").join(prof, on="node_id")
    y = x["is_scam"].to_numpy()
    out = {"age_only_auc_all": float(roc_auc_score(y, -x["age_days"].fill_null(0).to_numpy()))}
    pre = x.filter(pl.col("cohort") == "pre")
    out["age_only_auc_pre_cohort"] = float(roc_auc_score(pre["is_scam"].to_numpy(), -pre["age_days"].fill_null(0).to_numpy()))
    out["cohort_by_role"] = [r for r in x.group_by("cohort", "is_scam").len().sort("cohort", "is_scam").to_dicts()]
    # propensity on confounders (age, cohort, proxies, missingness), fitted on train; 1:1 NN matching on logit with caliper
    feats = ["age_days", "days_active", "days_since_active", "normal_sent_cnt", "normal_recv_cnt", "erc20_sent_cnt",
             "erc20_recv_cnt", "eth_sent_sum", "eth_recv_sum"]
    X = np.column_stack([np.log1p(np.clip(x[c].fill_null(0).to_numpy().astype(float), 0, None)) for c in feats]
                        + [(x["cohort"] == "born").to_numpy().astype(float), x["profile_missing"].fill_null(True).to_numpy().astype(float)])
    tr = (x["split"] == "train").to_numpy()
    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    logit = lr.decision_function(X)
    out["confounder_only_auc"] = {s: float(roc_auc_score(y[(x["split"] == s).to_numpy()], logit[(x["split"] == s).to_numpy()])) for s in ["train", "val", "test"]}
    cal = 0.2 * logit.std()
    for s in ["train", "val", "test"]:
        m = (x["split"] == s).to_numpy()
        ls, ln = logit[m & (y == 1)], logit[m & (y == 0)]
        lo, hi = max(ls.min(), ln.min()), min(ls.max(), ln.max())
        ls, ln = ls[(ls >= lo) & (ls <= hi)], ln[(ln >= lo) & (ln <= hi)]
        pairs = caliper_pairs(ls, ln, cal)
        out[f"matched_pairs_{s}"] = {"common_support_scam": int(len(ls)), "common_support_normal": int(len(ln)), "pairs": pairs}
    return out


def throughput() -> dict:
    import time

    import torch
    if not torch.cuda.is_available():
        return {"cuda": False}
    dev = "cuda"
    layer = torch.nn.TransformerEncoderLayer(128, 4, 512, dropout=0.1, batch_first=True, norm_first=True)
    model = torch.nn.TransformerEncoder(layer, 4, enable_nested_tensor=False).to(dev)
    head = torch.nn.Linear(128, 256).to(dev)
    params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), 1e-3)
    scaler = torch.amp.GradScaler()
    mask = torch.nn.Transformer.generate_square_subsequent_mask(256, device=dev)
    B, L = 64, 256
    x = torch.randn(B, L, 128, device=dev)
    for i in range(25):
        if i == 5:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast(dev, dtype=torch.float16):
            loss = head(model(x, mask=mask, is_causal=True)).float().pow(2).mean()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    train_tps = 20 * B * L / (time.time() - t0)
    model.eval()
    x1 = torch.randn(1, L, 128, device=dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16):
        for i in range(220):
            if i == 20:
                torch.cuda.synchronize(); t0 = time.time()
            head(model(x1, mask=mask, is_causal=True)[:, -1])
        torch.cuda.synchronize()
    lat = (time.time() - t0) / 200
    return {"cuda": True, "device": torch.cuda.get_device_name(0), "params": params,
            "train_tokens_per_s_fp16_B64_L256": round(train_tps), "batch1_L256_pass_ms": round(lat * 1000, 3),
            "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pilot", default="data/pilot")
    ap.add_argument("--out", default="runs/p0_audit.json")
    args = ap.parse_args()
    raw, pilot = Path(args.raw), Path(args.pilot)
    meta = json.loads((pilot / "meta.json").read_text())
    sl.set_window(f"{meta['W_START'][:7]}:{meta['W_END'][:7]}")
    ev = pl.read_parquet(pilot / "events.parquet")
    a = pl.read_parquet(pilot / "agents.parquet")
    prof = pl.read_parquet(pilot / "profiles.parquet")
    venues = pl.read_parquet(pilot / "venues.parquet")
    hubs = pl.read_parquet(pilot / "hubs.parquet")
    lab = pl.read_parquet(raw / "gnn_dataset/labels/targets_global.parquet")
    mt = pl.read_parquet(raw / "gnn_dataset/targets/month_targets.parquet").join(lab, on="node_id")
    w = mt.filter(pl.col("month").is_in(sl.months(sl.W_START, sl.W_END)), pl.col("is_contract") == 0).unique("node_id")
    report = {
        "meta": json.loads((pilot / "meta.json").read_text()),
        "month_targets_W_EOAs": by_role(w, pl.len(), "n"),
        "hubs": {"n": hubs.height, "by_role": by_role(hubs, pl.len(), "n"), "events_in_W": int(hubs["w_events"].sum())},
        "venues": {"n": venues.height, "by_kind": dict(venues.group_by("venue_kind").len().iter_rows()),
                   "is_contract": dict(venues.group_by("is_contract").len().iter_rows()),
                   "labelled_scam": int((venues["is_scam"] == 1).sum()),
                   "sender_capable": int(venues["sender_capable"].sum())},
        "agents": per_agent_stats(ev, a),
        "groups": groups_report(a),
        "leakage": leakage_report(ev, a, venues),
        "activity_end": activity_end_check(ev, a),
        "profiles": {"pre_cohort_profile_missing": int(prof.filter(pl.col("cohort") == "pre")["profile_missing"].sum())},
        "ledger": ledger_check(ev, prof, a),
        "daily_vs_edges": daily_vs_edges(raw, ev, a),
        "confound": confound_report(a, prof),
        "throughput": throughput(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
