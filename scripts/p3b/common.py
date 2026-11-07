"""Shared P3b loading: real development views, saved P3a worlds as replay populations, group maps and attachment
uniforms (reports/p3b_prereg.md §P2-P4)."""
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("p3a_common", Path(__file__).resolve().parents[1] / "p3a" / "common.py")
p3a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3a)  # load_pop / raw_events only: its real-data loaders (full store) are never called here

from lpw.p1.store import NEW, OUT  # noqa: E402
from lpw.p3b import groups as G  # noqa: E402
from lpw.p3b.replay import Pop  # noqa: E402

P3A = p3a.ROOT
ROOT = Path(os.environ.get("P3B_ROOT", "runs/p3b/2018"))
WORLDS = (0, 1, 2)
ARMS = ("kct", "B")
ATTACH = ("p3b0", "p3b1")
PART = {"train": 0, "hold": 1}


def group_seed(w, part):
    return 4000 + 10 * w + PART[part]


def attach_seed(w, part):
    return 5000 + 10 * w + PART[part]


def synth_pop(st, w, arm, part):
    """(Pop over the original sampled population, raw events sorted as in lpw.p3.observe.observe_synth, conds).
    Group ids are filled by assign_groups."""
    conds, _ = p3a.load_pop(w, part)
    ev = p3a.raw_events(w, arm, part).sort("agent", "ts", maintain_order=True)
    n = len(conds)
    agent = ev["agent"].to_numpy()
    counts = np.bincount(agent, minlength=n)
    off = np.concatenate([[0], np.cumsum(counts)])
    assert (agent == np.repeat(np.arange(n), counts)).all()
    venue = ev["venue"].to_numpy()
    cls = ev["cls"].to_numpy()
    elig = venue < 0
    pop = Pop(group=np.full(n, -1, np.int64), role=np.array([c["role"] for c in conds], np.int64),
              cohort=np.array([c["cohort"] for c in conds], np.int64), origin=np.array([c["origin"] for c in conds], np.int64),
              off=off, sec=np.floor(ev["ts"].to_numpy()).astype(np.int64), out=ev["kind"].to_numpy() == OUT, elig=elig,
              first=cls == NEW, ent=ev["ent"].to_numpy().astype(np.int64), cp=None, w_start=st.w_start, w_end=st.w_end)
    # eligible first touches are exactly the NEW draws; repeats follow
    e = ev.with_columns(fi=pl.col("ts").cum_count().over(["agent", "ent"]) == 1)
    fi = e["fi"].to_numpy()
    assert (fi[elig] == pop.first[elig]).all(), "NEW is not the first touch of a non-venue entry"
    return pop, ev, conds


def template_prior(R):
    """Train template histogram (frozen artefact content)."""
    tr = R.agents("train")
    return G.templates(R.group[tr], R.st.role[tr])


def assign_groups(pop, w, part, keys, counts, n_train):
    g, info = G.sample(pop.role, keys, counts, n_train, group_seed(w, part))
    pop.group = g
    return info


def uniforms(pop, w, part):
    """(gate_u, ptr_u) per eligible first touch in agent-then-local order: agent i's q-th first touch takes the q-th
    draw of SeedSequence([seed, 0, i]) (gate) and [seed, 1, i] (pointer)."""
    seed = attach_seed(w, part)
    m = pop.elig & pop.first
    agent = np.repeat(np.arange(pop.n), np.diff(pop.off))
    k = np.bincount(agent[m], minlength=pop.n)
    gu = np.concatenate([np.random.default_rng(np.random.SeedSequence([seed, 0, i])).random(k[i]) for i in range(pop.n)])
    pu = np.concatenate([np.random.default_rng(np.random.SeedSequence([seed, 1, i])).random(k[i]) for i in range(pop.n)])
    return gu, pu


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))


def sidecar(w, arm, part, att):
    S = np.load(ROOT / f"world{w}" / f"{arm}_{part}_{att}.npz")
    assert (int(S["world"]), str(S["part"]), str(S["arm"]), str(S["att"])) == (w, part, arm, att), "sidecar view context"
    return S


def synth_view(st, w, arm, part, att, drop_capped=False):
    """(observed agents, observed events with `node`, P3a node features, incidence (B, nodes)) of one attached world
    part. att=None gives the P3a zero-sharing anchor (private nodes). Observed events are row-aligned with the
    observe-sorted raw tape (observe_synth drops no events), so the sidecar's node column applies directly."""
    from lpw.p3b.graph import incidence
    from lpw.p3b.topology import S_PERIPHERY, node_key
    ag, ev, F, info = p3a.synth(st, w, arm, part, drop_capped=drop_capped)
    elig = ev["venue"].to_numpy() < 0
    if att is None:  # private node per (agent, entry)
        local = np.where(elig, ev["agent"].to_numpy() * (1 << 32) + ev["cp"].to_numpy(), 0)
    else:
        assert not drop_capped, "sidecars index the full tape"
        local = sidecar(w, arm, part, att)["node"]
        assert len(local) == ev.height and ((local >= 0) == elig).all(), "sidecar / tape misalignment"
    node = np.where(elig, node_key(S_PERIPHERY, np.where(elig, local, 0)), -1)  # typed, view-local periphery keys
    ev = ev.with_columns(pl.Series("node", node))
    B, nodes = incidence(ev["agent"].to_numpy()[elig], node[elig], ag.height)
    return ag, ev, F, (B, nodes), info


def real_view(R, split):
    """(observed agents, observed events, P3a node features, incidence (B, nodes)) of a real development split."""
    from lpw.p3.features import node_features
    from lpw.p3.observe import observe_real
    from lpw.p3b.graph import incidence
    agents = R.agents(split)
    ag, ev = observe_real(R.st, agents)
    F = node_features(ag, ev, R.st.w_start, R.st.w_end)
    idx = np.concatenate([np.arange(R.st.off[a], R.st.off[a + 1]) for a in agents])
    elig = (R.st.venue[idx] < 0) & ~R.cp_agent[idx]
    assert ev.height == len(idx)
    B, nodes = incidence(ev["agent"].to_numpy()[elig], R.cp[idx][elig], ag.height)
    return ag, ev, F, (B, nodes)


def real_transactions(R, agents) -> dict:
    """Unique transactions incident to real focal agents (lpw.p3b.topology.tx_table): u = sender, v = receiver as raw
    addresses; venue = the counterparty is a venue. Duplicate recordings (agent-agent transactions seen from both
    sides) must agree exactly."""
    from lpw.p3b.topology import tx_table
    st = R.st
    agents = np.asarray(agents)
    idx = np.concatenate([np.arange(st.off[a], st.off[a + 1]) for a in agents])
    me = np.repeat(st.node_id[agents], st.off[agents + 1] - st.off[agents]).astype(np.uint64)
    out = st.kind[idx] == OUT
    u = np.where(out, me, R.cp[idx])
    v = np.where(out, R.cp[idx], me)
    T = tx_table(R.tx[idx], u, v, st.ts[idx], st.eth[idx], st.venue[idx] >= 0)
    T["day"] = (T["ts"] // 86400).astype(np.int64)
    return T
