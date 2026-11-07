"""Observation operator O and the real reference, in one observed schema (SPEC §6.8; reports/p3a_prereg.md §3).

Observed events: agent, kind (OUT = 0, IN = 1), ts (int64 recorded second), eth, fee, amt (log10 ETH), cp (the
counterparty's identity within the agent: its venue index for venue counterparties, else V + its partner entry),
venue (-1 if none). Observed agents: agent, role, cohort, origin, b0 and the 20 raw profile fields.

- Real: IN_exo and IN_agent both become IN (an observer sees every incident edge). owned_only drops IN_agent
  instead (sensitivity analysis).
- Synthetic: times are floored to whole seconds (the chain records integer block times; unprojected float times
  alone separate the worlds), IN_exo becomes IN, and venue counterparties are keyed by venue (the generator opens
  a new partner entry on every VENUE draw, an observer sees one address). Agents with no recorded event are
  dropped and counted: real agents are active in W by construction.
"""
import numpy as np
import polars as pl

from lpw.p1.store import DUST_LOG10, IN_AGENT, OUT, PROFILE_FIELDS

IN = 1
EV_COLS = ("agent", "kind", "ts", "eth", "fee", "amt", "cp", "venue")


def _agents(role, cohort, origin, b0, raw) -> pl.DataFrame:
    d = {"agent": np.arange(len(role), dtype=np.int64), "role": np.asarray(role, np.int64), "cohort": np.asarray(cohort, np.int64),
         "origin": np.asarray(origin, np.int64), "b0": np.asarray(b0, np.float64)}
    d.update({f: np.asarray(raw, np.float64)[:, i] for i, f in enumerate(PROFILE_FIELDS)})
    return pl.DataFrame(d)


def _check(ag: pl.DataFrame, ev: pl.DataFrame, w_end: int):
    assert ev["kind"].is_in([OUT, IN]).all()
    assert (ev["amt"] >= DUST_LOG10 - 1e-9).all(), "dust floor violated"
    for c in ("eth", "fee", "amt"):
        assert ev[c].is_finite().all(), f"non-finite {c}"
    j = ev.join(ag.select("agent", "origin"), on="agent", how="left")
    assert j["origin"].null_count() == 0, "event of an unknown agent"
    assert (j["ts"] < w_end).all(), "event after W"
    assert (j["ts"] >= j["origin"]).all(), "event before the agent's origin"


def observe_real(store, agents: np.ndarray, owned_only: bool = False):
    """(agents, events) of real agents in the observed schema; agent i is store agent agents[i]."""
    V = store.n_venues
    rows = []
    for i, a in enumerate(agents):
        s, e = store.off[a], store.off[a + 1]
        k = store.kind[s:e]
        m = k != IN_AGENT if owned_only else np.ones(len(k), bool)
        ven = store.venue[s:e][m]
        rows.append(pl.DataFrame({"agent": np.full(m.sum(), i, np.int64), "kind": np.where(k[m] == OUT, OUT, IN).astype(np.int8),
                                  "ts": store.ts[s:e][m].astype(np.int64), "eth": store.eth[s:e][m].astype(np.float64),
                                  "fee": np.where(k[m] == OUT, store.fee[s:e][m], 0.0).astype(np.float64),
                                  "amt": store.amt[s:e][m].astype(np.float64),
                                  "cp": np.where(ven >= 0, ven, V + store.ent[s:e][m]).astype(np.int64), "venue": ven.astype(np.int64)}))
    ev = pl.concat(rows)
    ag = _agents(store.role[agents], store.cohort[agents], store.origin[agents], store.b0[agents], store.prof_raw[agents])
    _check(ag, ev, store.w_end)
    return ag, ev


def observe_synth(store, conds: list[dict], raw: np.ndarray, ev: pl.DataFrame):
    """O(world) for one uncoupled world from lpw.p1.generate. Returns (agents, events, info). Agents without events
    are dropped and the survivors renumbered 0..n-1 in their original order."""
    V = store.n_venues
    assert not (ev["kind"] == IN_AGENT).any(), "uncoupled world produced IN_agent"
    ev = ev.sort("agent", "ts", maintain_order=True)
    ev = pl.DataFrame({"agent": ev["agent"].to_numpy().astype(np.int64),
                       "kind": np.where(ev["kind"].to_numpy() == OUT, OUT, IN).astype(np.int8),
                       "ts": np.floor(ev["ts"].to_numpy()).astype(np.int64), "eth": ev["eth"].to_numpy().astype(np.float64),
                       "fee": ev["fee"].to_numpy().astype(np.float64), "amt": ev["amt"].to_numpy().astype(np.float64),
                       "cp": np.where(ev["venue"].to_numpy() >= 0, ev["venue"].to_numpy(), V + ev["ent"].to_numpy()).astype(np.int64),
                       "venue": ev["venue"].to_numpy().astype(np.int64)})
    n = len(conds)
    has = np.bincount(ev["agent"].to_numpy(), minlength=n) > 0
    keep = np.flatnonzero(has)
    new = np.full(n, -1, np.int64)
    new[keep] = np.arange(len(keep))
    ev = ev.with_columns(agent=pl.Series(new[ev["agent"].to_numpy()]))
    role = np.array([c["role"] for c in conds])
    ag = _agents(role[keep], [conds[i]["cohort"] for i in keep], [conds[i]["origin"] for i in keep],
                 [conds[i]["b0"] for i in keep], raw[keep])
    _check(ag, ev, store.w_end)
    info = {"n_sampled": n, "n_dropped_no_events": int(n - len(keep)),
            "dropped_by_role": {str(r): int(((role == r) & ~has).sum()) for r in (0, 1)}, "n_events": ev.height}
    return ag, ev, info
