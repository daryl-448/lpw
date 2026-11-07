"""Per-agent node features for P3a detectors (reports/p3a_prereg.md §4). One function for real and O(world).

Blocks (frozen):
- CONF: cohort, log age, activation time as a fraction of W, log exposure days (w_end - origin), days from origin to
  the first event;
- PROF: the 20 profile fields, log-transformed (zero for born-in-W agents; a confounder proxy block under §7.5);
- LOCAL: activity, timing, amounts, counterparties and fees from the agent's own observed events;
- BAL: the balance trajectory implied by b0 and the observed flows (defect-1 sentinels);
- VENUE: venue usage without venue identities;
- STRUCT (cross-agent sharing) is not a P3a block: an uncoupled world has none (reported on real only).
"""
import numpy as np
import polars as pl

from lpw.p1.store import OUT, PROFILE_FIELDS, profile_transform, slog

DAY = 86400.0
ZERO_ETH = 1e-4            # |balance| at or below the dust floor (1e14 wei) counts as swept to zero
MISSING = -1.0             # value of a statistic that is undefined for the agent (e.g. no OUT events)

CONF = ["cohort", "log_age", "act_frac", "log_exposure", "first_delay_days"]  # first delay: 0 for most born agents, > 0 for
# every pre-existing one, so it is a cohort / activation proxy, not behaviour
PROF = [f"p_{f}" for f in PROFILE_FIELDS]
LOCAL = ["n_events", "n_out", "n_in", "out_frac", "active_days", "span_days", "end_idle_days",
         "med_log_gap", "same_second_share", "max_second_burst", "max_day_events", "hour_entropy", "weekend_share",
         "out_amt_med", "out_amt_max", "in_amt_med", "in_amt_max", "amt_std", "out_total", "in_total",
         "distinct_cp", "repeat_share", "top_cp_share", "distinct_senders", "distinct_receivers", "fee_med"]
BAL = ["end_balance", "net_flow", "ends_near_zero", "min_balance", "neg_out_share", "sweep_out_share"]
VENUE = ["venue_share", "distinct_venues", "top_venue_share", "venue_out_share"]
BLOCKS = {"CONF": CONF, "PROF": PROF, "LOCAL": LOCAL, "BAL": BAL, "VENUE": VENUE}
SETS = {  # detector feature sets
    "full": CONF + PROF + LOCAL + BAL + VENUE,
    "noconf": LOCAL + BAL + VENUE,          # §7.5 (2): no age, era, cohort or proxy (profile) features
    "confonly": CONF + PROF,                # §7.5 conditional baseline
    "noprof": CONF + LOCAL + BAL + VENUE,   # profile-gain parity: full minus this
    "nobal": CONF + PROF + LOCAL + VENUE,   # defect-1 ablation
    "local": LOCAL + BAL + VENUE,           # C2ST primary
}


def node_features(ag: pl.DataFrame, ev: pl.DataFrame, w_start: int, w_end: int) -> pl.DataFrame:
    """One row per agent of ag (same order), columns agent + every block."""
    W = float(w_end - w_start)
    e = ev.sort("agent", "ts", maintain_order=True).join(ag.select("agent", "b0"), on="agent", how="left")
    out_ = pl.col("kind") == OUT
    e = e.with_columns(
        flow=pl.when(out_).then(-pl.col("eth") - pl.col("fee")).otherwise(pl.col("eth")),
        day=pl.col("ts") // 86400, hour=(pl.col("ts") % 86400) // 3600, dow=((pl.col("ts") // 86400) - 4) % 7,
        prev_ts=pl.col("ts").shift(1).over("agent"),
        seen=pl.col("cp").cum_count().over(["agent", "cp"]) > 1)
    e = e.with_columns(bal_after=pl.col("b0") + pl.col("flow").cum_sum().over("agent"))
    e = e.with_columns(gap=(pl.col("ts") - pl.col("prev_ts")).cast(pl.Float64))
    per = e.group_by("agent").agg(
        n=pl.len(), n_out=out_.sum(), n_in=(~out_).sum(), active=pl.col("day").n_unique(),
        first=pl.col("ts").min(), last=pl.col("ts").max(),
        med_log_gap=pl.col("gap").drop_nulls().log1p().median(), same_sec=(pl.col("gap") == 0).sum(),
        hour_entropy=pl.col("hour").value_counts().struct.field("count").map_batches(
            lambda c: pl.Series([float(-(p := c.to_numpy() / c.sum()) @ np.log(p))]), returns_scalar=True),
        weekend=(pl.col("dow") >= 5).mean(),
        out_amt_med=pl.col("amt").filter(out_).median(), out_amt_max=pl.col("amt").filter(out_).max(),
        in_amt_med=pl.col("amt").filter(~out_).median(), in_amt_max=pl.col("amt").filter(~out_).max(),
        amt_std=pl.col("amt").std(ddof=0),
        out_total=pl.col("eth").filter(out_).sum(), in_total=pl.col("eth").filter(~out_).sum(),
        distinct_cp=pl.col("cp").n_unique(), repeat_share=pl.col("seen").mean(),
        distinct_senders=pl.col("cp").filter(~out_).n_unique(), distinct_receivers=pl.col("cp").filter(out_).n_unique(),
        fee_med=pl.col("fee").filter(out_).log10().median(),
        net_flow=pl.col("flow").sum(), min_balance=pl.col("bal_after").min(),
        neg_out=(out_ & (pl.col("bal_after") < -ZERO_ETH)).sum(), sweep_out=(out_ & (pl.col("bal_after").abs() <= ZERO_ETH)).sum(),
        n_ven=(pl.col("venue") >= 0).sum(), distinct_venues=pl.col("venue").filter(pl.col("venue") >= 0).n_unique(),
        ven_out=(out_ & (pl.col("venue") >= 0)).sum())
    top_cp = e.group_by("agent", "cp").len().group_by("agent").agg(top_cp=pl.col("len").max())
    burst = e.group_by("agent", "ts").len().group_by("agent").agg(burst=pl.col("len").max())
    dmax = e.group_by("agent", "day").len().group_by("agent").agg(dmax=pl.col("len").max())
    tven = e.filter(pl.col("venue") >= 0).group_by("agent", "venue").len().group_by("agent").agg(tven=pl.col("len").max())
    A = (ag.select("agent", "cohort", "origin", "b0", *PROFILE_FIELDS)
         .join(per, on="agent", how="left").join(top_cp, on="agent", how="left").join(burst, on="agent", how="left")
         .join(dmax, on="agent", how="left").join(tven, on="agent", how="left"))
    n = A["n"].fill_null(0).to_numpy().astype(float)
    nz = np.maximum(n, 1)
    g = lambda c, fill=MISSING: A[c].cast(pl.Float64).fill_null(fill).fill_nan(fill).to_numpy()  # noqa: E731
    origin = A["origin"].to_numpy().astype(float)
    n_out, n_ven = g("n_out", 0), g("n_ven", 0)
    end_bal = A["b0"].to_numpy() + g("net_flow", 0)
    f = {
        "cohort": A["cohort"].to_numpy().astype(float), "log_age": np.log1p(np.clip(A["age_days"].to_numpy(), 0, None)),
        "act_frac": (origin - w_start) / W, "log_exposure": np.log1p((w_end - origin) / DAY),
        "n_events": np.log1p(n), "n_out": np.log1p(n_out), "n_in": np.log1p(g("n_in", 0)), "out_frac": n_out / nz,
        "active_days": np.log1p(g("active", 0)), "span_days": np.where(n > 0, (g("last", 0) - g("first", 0)) / DAY, MISSING),
        "first_delay_days": np.where(n > 0, (g("first", 0) - origin) / DAY, MISSING),
        "end_idle_days": np.where(n > 0, (w_end - g("last", 0)) / DAY, MISSING),
        "med_log_gap": g("med_log_gap"), "same_second_share": np.where(n > 1, g("same_sec", 0) / np.maximum(n - 1, 1), MISSING),
        "max_second_burst": np.log1p(g("burst", 0)), "max_day_events": np.log1p(g("dmax", 0)),
        "hour_entropy": g("hour_entropy"), "weekend_share": g("weekend"),
        "out_amt_med": g("out_amt_med", -9.0), "out_amt_max": g("out_amt_max", -9.0),
        "in_amt_med": g("in_amt_med", -9.0), "in_amt_max": g("in_amt_max", -9.0), "amt_std": g("amt_std", 0),
        "out_total": np.log1p(g("out_total", 0)), "in_total": np.log1p(g("in_total", 0)),
        "distinct_cp": np.log1p(g("distinct_cp", 0)), "repeat_share": g("repeat_share"), "top_cp_share": g("top_cp", 0) / nz,
        "distinct_senders": np.log1p(g("distinct_senders", 0)), "distinct_receivers": np.log1p(g("distinct_receivers", 0)),
        "fee_med": g("fee_med", -9.0),
        "end_balance": slog(end_bal), "net_flow": slog(g("net_flow", 0)), "ends_near_zero": (np.abs(end_bal) <= ZERO_ETH).astype(float),
        "min_balance": slog(np.where(n > 0, g("min_balance", 0), A["b0"].to_numpy())),
        "neg_out_share": np.where(n_out > 0, g("neg_out", 0) / np.maximum(n_out, 1), MISSING),
        "sweep_out_share": np.where(n_out > 0, g("sweep_out", 0) / np.maximum(n_out, 1), MISSING),
        "venue_share": n_ven / nz, "distinct_venues": np.log1p(g("distinct_venues", 0)),
        "top_venue_share": np.where(n_ven > 0, g("tven", 0) / np.maximum(n_ven, 1), MISSING),
        "venue_out_share": np.where(n_ven > 0, g("ven_out", 0) / np.maximum(n_ven, 1), MISSING),
    }
    pt = profile_transform(A.select(PROFILE_FIELDS).to_numpy().astype(float))
    f.update({p: pt[:, i] for i, p in enumerate(PROF)})
    out = pl.DataFrame({"agent": A["agent"], **{c: np.asarray(f[c], np.float64) for c in CONF + PROF + LOCAL + BAL + VENUE}})
    assert out.height == ag.height and (out["agent"].to_numpy() == ag["agent"].to_numpy()).all()
    return out


def matrix(F: pl.DataFrame, fset: str) -> np.ndarray:
    return F.select(SETS[fset]).to_numpy()
