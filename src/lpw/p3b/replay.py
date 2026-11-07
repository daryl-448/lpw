"""Causal group-pool replay: one engine for real label extraction and synthetic identity attachment
(reports/p3b_prereg.md §P4).

A population is a set of focal agents with group ids and their complete local event tapes. Eligible events touch a
non-venue, non-agent counterparty. Identity decisions happen only at an agent's first local touch of an entry; every
touch (first or repeat) updates the group's pool statistics.

Recorded-second batch rule: events are processed in canonical order (second, agent, local index). Inside a second,
an agent's own entry map and touched set update immediately (own repeats and within-agent injection stay exact),
while pool additions and pool statistics become visible to other members only after the whole second.
The opportunity set of a first touch is C_e = pool nodes visible from earlier seconds that the actor has not touched;
C_e empty forces NEW.

- Real replay: node identity is the real global counterparty; the label is target in C_e.
- Synthetic replay: a kernel decides, from one gate uniform and one pointer uniform consumed at every eligible first
  touch, whether to attach to a node of C_e or mint a fresh node in the group's namespace.
"""
from dataclasses import dataclass, field

import numpy as np

DAY = 86400.0
N_GATE_CONT = 8
GATE_CONT = ("log_events_before", "log_partners_before", "log_choice_set", "log_pool", "log_other_members",
             "log_other_active", "log_age_days", "window_frac")
PTR_CONT = ("log_members", "log_events", "log_recency_days")
PTR_BIN = ("same_dir", "mint_out")


@dataclass
class Pop:
    """Focal population. Events are in each agent's local order, agents contiguous (off)."""
    group: np.ndarray    # (n,) group id (any integers)
    role: np.ndarray     # (n,)
    cohort: np.ndarray   # (n,) 1 = born in W
    origin: np.ndarray   # (n,) int seconds
    off: np.ndarray      # (n + 1,)
    sec: np.ndarray      # (E,) int64 recorded second
    out: np.ndarray      # (E,) bool
    elig: np.ndarray     # (E,) bool: non-venue, non-agent counterparty
    first: np.ndarray    # (E,) bool: first local touch of the entry (only meaningful where elig)
    ent: np.ndarray      # (E,) local entry id within the agent
    cp: np.ndarray | None = None  # (E,) real global counterparty (real replay only)
    rank: np.ndarray | None = None  # (E,) real stable cross-agent event order (ts, block, tx_hash, OUT-first): fixes a
    # real node's mint direction when several members first touch it in the same second (real replay only)
    w_start: int = 0
    w_end: int = 1

    @property
    def n(self):
        return len(self.group)


@dataclass
class Result:
    node: np.ndarray                     # (E,) node id of eligible events (-1 elsewhere); real: index into `cps`
    opp_event: np.ndarray                # (K,) event index of every eligible first touch
    opp_nc: np.ndarray                   # (K,) |C_e|
    opp_gate: np.ndarray                 # (K, 8) continuous gate features
    opp_bin: np.ndarray                  # (K, 3) role, born, out
    opp_attach: np.ndarray               # (K,) bool: label (real) or decision (synthetic)
    opp_cocreate: np.ndarray             # (K,) bool: real target already touched by the group in the same second
    ptr: list = field(default_factory=list)  # real positives: (opp index, (n_c, 5) features, target position)
    cps: np.ndarray | None = None        # real: node id -> global cp
    group_of_node: np.ndarray | None = None


def _canonical(pop: Pop):
    agent = np.repeat(np.arange(pop.n), np.diff(pop.off))
    local = np.arange(len(pop.sec)) - pop.off[agent]
    return agent, local


def replay(pop: Pop, kernel=None, uniforms=None, keep_ptr=True) -> Result:
    """kernel(role, gate_cont (8,), gate_bin (3,), cand () -> (n_c, 5), n_c, ug, up) -> candidate position or -1.
    uniforms: (gate_u, ptr_u), each (K_total,) indexed by the event's position among all eligible first touches in
    agent-then-local order (lpw.p3b.replay.first_touch_index). Real replay passes kernel=None (pop.cp required)."""
    real = kernel is None
    assert not real or (pop.cp is not None and pop.rank is not None)
    agent, local = _canonical(pop)
    E = len(pop.sec)
    node = np.full(E, -1, np.int64)
    ft_index = np.full(E, -1, np.int64)
    m = pop.elig & pop.first
    ft_index[m] = np.arange(m.sum())
    # agent-level counters
    partners_before = np.zeros(E, np.int64)
    for a in range(pop.n):
        s, e = pop.off[a], pop.off[a + 1]
        partners_before[s:e] = np.concatenate([[0], np.cumsum(m[s:e])[:-1]]) if e > s else []
    first_sec = np.array([pop.sec[pop.off[a]] if pop.off[a + 1] > pop.off[a] else np.iinfo(np.int64).max for a in range(pop.n)])
    span = float(pop.w_end - pop.w_start)

    recs = {k: [] for k in ("ev", "nc", "gate", "bin", "att", "co")}
    ptr = []
    cps_all, gnode_all = [], []
    n_nodes = 0

    order_groups = np.unique(pop.group)
    members_of = {}
    for a in np.argsort(pop.group, kind="stable"):
        members_of.setdefault(int(pop.group[a]), []).append(int(a))

    for g in order_groups:
        mem = members_of[int(g)]
        gsize = len(mem)
        ev_idx = np.concatenate([np.arange(pop.off[a], pop.off[a + 1]) for a in mem]) if mem else np.zeros(0, np.int64)
        ev_idx = ev_idx[pop.elig[ev_idx]]
        if len(ev_idx) == 0:
            continue
        ev_idx = ev_idx[np.lexsort((local[ev_idx], agent[ev_idx], pop.sec[ev_idx]))]
        mfs = np.sort(first_sec[mem])
        cap = int(pop.first[ev_idx].sum())  # upper bound on the group's nodes
        n_members = np.zeros(cap, np.int64)
        n_events = np.zeros(cap, np.int64)
        last = np.zeros(cap, np.float64)
        out_seen = np.zeros(cap, bool)
        in_seen = np.zeros(cap, bool)
        mint_out = np.zeros(cap, np.float64)
        mint_rank = np.zeros(cap, np.int64)
        pairs = set()     # (node, member) touched
        key = {}          # real cp -> local node
        cps = []
        entry = {}        # (agent, ent) -> local node
        touched = {a: set() for a in mem}
        st = {"n_vis": 0, "n": 0}
        pending = []      # (node, agent, sec, out) updates of the current second
        cur = None

        def commit():
            for nd, a, s_, o_ in pending:
                if (nd, a) not in pairs:
                    pairs.add((nd, a))
                    n_members[nd] += 1
                n_events[nd] += 1
                last[nd] = s_
                if o_:
                    out_seen[nd] = True
                else:
                    in_seen[nd] = True
            pending.clear()
            st["n_vis"] = st["n"]

        def mint(o_, s_, rk=0):
            nd = st["n"]
            st["n"] += 1
            last[nd] = s_
            mint_out[nd] = float(o_)
            mint_rank[nd] = rk
            return nd

        for i in ev_idx:
            s = int(pop.sec[i])
            if s != cur:
                commit()
                cur = s
            n_vis = st["n_vis"]
            a = int(agent[i])
            o = bool(pop.out[i])
            ek = (a, int(pop.ent[i]))
            if not pop.first[i]:
                nd = entry[ek]
                pending.append((nd, a, s, o))
                node[i] = nd
                continue
            tset = touched[a]
            own_vis = [t for t in tset if t < n_vis]
            nc = n_vis - len(own_vis)

            cache = []

            def cand(own_vis=own_vis, s=s, o=o, n_vis=n_vis, cache=cache):
                if cache:
                    return cache[0]
                keep = np.ones(n_vis, bool)
                keep[own_vis] = False
                idx = np.flatnonzero(keep)
                f = np.stack([np.log1p(n_members[idx].astype(np.float64)), np.log1p(n_events[idx].astype(np.float64)),
                              np.log1p((s - last[idx]) / DAY), (out_seen if o else in_seen)[idx].astype(np.float64),
                              mint_out[idx]], 1)
                cache.append((idx, f))
                return idx, f

            other_active = int(np.searchsorted(mfs, s, side="left")) - (1 if first_sec[a] < s else 0)
            gc = np.array([np.log1p(local[i]), np.log1p(partners_before[i]), np.log1p(nc), np.log1p(n_vis), np.log1p(gsize - 1),
                           np.log1p(other_active), np.log1p(max(s - int(pop.origin[a]), 0) / DAY), (s - pop.w_start) / span])
            gb = np.array([pop.role[a], pop.cohort[a], o], np.float64)
            co = False
            if real:
                c = int(pop.cp[i])
                nd = key.get(c)
                att = nd is not None and nd < n_vis
                co = nd is not None and nd >= n_vis
                if att and keep_ptr:
                    idx, f = cand()
                    ptr.append((len(recs["ev"]), f.astype(np.float32), int(np.searchsorted(idx, nd))))
                if nd is None:
                    nd = key[c] = mint(o, s, int(pop.rank[i]))
                    cps.append(c)
                elif co and pop.rank[i] < mint_rank[nd]:  # same-second co-creation: the real order fixes direction
                    mint_out[nd], mint_rank[nd] = float(o), int(pop.rank[i])
            else:
                q = ft_index[i]
                ug, up = uniforms[0][q], uniforms[1][q]
                pos = kernel(int(pop.role[a]), gc, gb, cand, nc, ug, up) if nc else -1
                att = pos >= 0
                if att:
                    idx, _ = cand()
                    nd = int(idx[pos])
                else:
                    nd = mint(o, s)
            entry[ek] = nd
            tset.add(nd)
            pending.append((nd, a, s, o))
            node[i] = nd
            recs["ev"].append(i); recs["nc"].append(nc); recs["gate"].append(gc); recs["bin"].append(gb)
            recs["att"].append(att); recs["co"].append(co)
        commit()
        sel = ev_idx
        node[sel] += n_nodes
        if real:
            cps_all.extend(cps)
        gnode_all.extend([int(g)] * st["n"])
        n_nodes += st["n"]

    K = len(recs["ev"])
    res = Result(node, np.array(recs["ev"], np.int64), np.array(recs["nc"], np.int64),
                 np.array(recs["gate"], np.float64).reshape(K, N_GATE_CONT), np.array(recs["bin"], np.float64).reshape(K, 3),
                 np.array(recs["att"], bool), np.array(recs["co"], bool), ptr,
                 np.array(cps_all, np.uint64) if real else None, np.array(gnode_all, np.int64))
    return res
