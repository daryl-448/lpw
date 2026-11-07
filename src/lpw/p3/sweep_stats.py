"""Sweep diagnostics on observed streams (reports/p3a_prereg.md §15 A.5)."""
import polars as pl

from lpw.p1.store import OUT

ZERO = 1e-4
ROLES = (("normal", 0), ("scam", 1))


def sweep_stats(ag, ev):
    """per role: sweep-event share of OUTs, share of agents ever sweeping, terminal near-zero share, negative-OUT
    share, share of sweeps that are the agent's last OUT, share of sweeping agents with a later IN"""
    e = ev.sort("agent", "ts", maintain_order=True).join(ag.select("agent", "b0", "role"), on="agent")
    e = e.with_columns(flow=pl.when(pl.col("kind") == OUT).then(-pl.col("eth") - pl.col("fee")).otherwise(pl.col("eth")))
    e = e.with_columns(after=pl.col("b0") + pl.col("flow").cum_sum().over("agent"), i=pl.int_range(pl.len()).over("agent"))
    o = e.filter(pl.col("kind") == OUT).with_columns(sw=pl.col("after").abs() <= ZERO, neg=pl.col("after") < -ZERO)
    last_out = o.group_by("agent").agg(last_i=pl.col("i").max())
    first_sw = o.filter(pl.col("sw")).group_by("agent").agg(sw_i=pl.col("i").min())
    later_in = e.filter(pl.col("kind") != OUT).join(first_sw, on="agent").filter(pl.col("i") > pl.col("sw_i")).select("agent").unique()
    term = e.group_by("agent").agg(end=pl.col("after").last())
    A = ag.select("agent", "role", "b0").join(term, on="agent", how="left").join(first_sw, on="agent", how="left")
    A = A.with_columns(end=pl.col("end").fill_null(pl.col("b0")))
    out = {}
    for rn, r in ROLES:
        Ar, orr = A.filter(pl.col("role") == r), o.filter(pl.col("role") == r)
        sweepers = Ar.filter(pl.col("sw_i").is_not_null())
        swr = orr.filter(pl.col("sw")).join(last_out, on="agent")  # per swept OUT event
        out[rn] = {"sweep_event_share": float(orr["sw"].mean()) if orr.height else None, "ever_sweep": float(sweepers.height / max(Ar.height, 1)),
                   "terminal_zero": float((Ar["end"].abs() <= ZERO).mean()) if Ar.height else None, "neg_out_share": float(orr["neg"].mean()) if orr.height else None,
                   "sweep_is_last_out": float((swr["i"] == swr["last_i"]).mean()) if swr.height else None,
                   "sweeper_later_in": float(sweepers["agent"].is_in(later_in["agent"]).mean()) if sweepers.height else None}
    return out
