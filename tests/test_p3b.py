"""P3b unit tests on hand-built populations: causal replay, local injection, STRUCT, group prior."""
import numpy as np
import pytest

from lpw.p3b import groups as G
from lpw.p3b.graph import components, incidence, struct_features
from lpw.p3b.replay import Pop, replay
from lpw.p3b.topology import S_AGENT, S_PERIPHERY, S_VENUE, key_kind, ks_continuous, node_key, tx_table, zeta_sample


def make_pop(agents, group, role=None, cp=True, rank=None):
    """agents: list of per-agent event lists (sec, out, elig, ent, cp). rank: real stable order (default: row order)."""
    n = len(agents)
    off = np.concatenate([[0], np.cumsum([len(a) for a in agents])])
    rows = [e for a in agents for e in a]
    sec = np.array([r[0] for r in rows], np.int64)
    out = np.array([r[1] for r in rows], bool)
    elig = np.array([r[2] for r in rows], bool)
    ent = np.array([r[3] for r in rows], np.int64)
    first = np.zeros(len(rows), bool)
    for a in range(n):
        seen = set()
        for i in range(off[a], off[a + 1]):
            if ent[i] not in seen:
                first[i] = True
                seen.add(ent[i])
    return Pop(group=np.array(group), role=np.zeros(n, np.int64) if role is None else np.array(role), cohort=np.ones(n, np.int64),
               origin=np.zeros(n, np.int64), off=off, sec=sec, out=out, elig=elig, first=first, ent=ent,
               cp=np.array([r[4] for r in rows], np.uint64) if cp else None,
               rank=(np.arange(len(rows)) if rank is None else np.asarray(rank)) if cp else None, w_start=0, w_end=1000)


def test_real_labels_strictly_earlier_second():
    # agent 0 touches X at t=10; agent 1 touches X at t=10 (same second: not a positive) and agent 2 at t=11 (positive)
    p = make_pop([[(10, 1, 1, 0, 7)], [(10, 0, 1, 0, 7)], [(11, 0, 1, 0, 7)]], group=[0, 0, 0])
    r = replay(p)
    assert r.opp_attach.tolist() == [False, False, True]
    assert r.opp_cocreate.tolist() == [False, True, False]
    assert r.opp_nc.tolist() == [0, 0, 1]
    assert len(set(r.node[r.node >= 0].tolist())) == 1  # one real node
    (k, f, tgt), = r.ptr
    assert tgt == 0 and f.shape == (1, 5)
    assert f[0, 0] == pytest.approx(np.log1p(2))   # two members touched X before second 11
    assert f[0, 4] == 1.0                           # minted by an OUT


def test_other_groups_are_invisible():
    p = make_pop([[(10, 1, 1, 0, 7)], [(11, 0, 1, 0, 7)]], group=[0, 1])
    r = replay(p)
    assert not r.opp_attach.any() and (r.opp_nc == 0).all()


def always_attach_first(role, gc, gb, cand, nc, ug, up):
    return 0


def never_attach(role, gc, gb, cand, nc, ug, up):
    return -1


def test_synthetic_injection_and_repeats():
    # agent 0 mints two nodes at t=1; agent 1 later has 3 first touches and a repeat; always attaching to the first
    # candidate must never give agent 1 the same node twice
    a0 = [(1, 1, 1, 0, 0), (1, 1, 1, 1, 0)]
    a1 = [(5, 0, 1, 0, 0), (6, 0, 1, 1, 0), (7, 0, 1, 2, 0), (8, 1, 1, 0, 0)]
    p = make_pop([a0, a1], group=[0, 0], cp=False)
    K = int((p.elig & p.first).sum())
    r = replay(p, kernel=always_attach_first, uniforms=(np.zeros(K), np.zeros(K)))
    n1 = r.node[p.off[1]:p.off[2]]
    assert n1[0] != n1[1] and n1[2] not in (n1[0], n1[1])     # injective within agent
    assert set(n1[:2]) == set(r.node[:2])                      # attached to agent 0's nodes
    assert n1[3] == n1[0]                                      # the repeat follows the entry map
    assert r.opp_nc.tolist() == [0, 0, 2, 1, 0]


def test_same_second_pool_invisible_to_others_but_own_entries_exact():
    a0 = [(5, 1, 1, 0, 0)]
    a1 = [(5, 1, 1, 0, 0), (5, 1, 1, 0, 0)]   # a1 repeats its own entry within the second
    p = make_pop([a0, a1], group=[0, 0], cp=False)
    K = int((p.elig & p.first).sum())
    r = replay(p, kernel=always_attach_first, uniforms=(np.zeros(K), np.zeros(K)))
    assert (r.opp_nc == 0).all()                              # nothing visible yet
    assert r.node[1] == r.node[2] and r.node[0] != r.node[1]


def test_local_equality_relation_invariant():
    rng = np.random.default_rng(0)
    agents = []
    for a in range(6):
        ev = []
        for j in range(20):
            ent = int(rng.integers(0, 6))
            ev.append((int(rng.integers(0, 30)), int(rng.integers(0, 2)), int(ent != 5), ent, 0))  # entry 5: a venue
        ev.sort()
        agents.append(ev)
    p = make_pop(agents, group=[0, 0, 0, 1, 1, 1], cp=False)
    K = int((p.elig & p.first).sum())
    u = np.random.default_rng(1).random((2, K))
    r = replay(p, kernel=lambda role, gc, gb, cand, nc, ug, up: int(up * nc) if ug < 0.7 else -1, uniforms=(u[0], u[1]))
    for a in range(p.n):
        s, e = p.off[a], p.off[a + 1]
        m = p.elig[s:e]
        ent, nd = p.ent[s:e][m], r.node[s:e][m]
        for i in range(len(ent)):
            for j in range(len(ent)):
                assert (ent[i] == ent[j]) == (nd[i] == nd[j])
    # nodes are group-exclusive
    agent = np.repeat(np.arange(p.n), np.diff(p.off))
    m = r.node >= 0
    for nd in np.unique(r.node[m]):
        assert len(set(p.group[agent[m][r.node[m] == nd]])) == 1


def test_struct_features_hand_built():
    # agents 0,1 share node 10; agent 2 alone on node 11; agent 3 has nothing
    B, nodes = incidence(np.array([0, 1, 1, 2]), np.array([10, 10, 12, 11]), 4)
    f = struct_features(B)
    assert f["shared_any"].tolist() == [1, 1, 0, 0]
    assert f["shared_frac"].tolist() == [1.0, 0.5, 0.0, 0.0]
    assert f["other_agents_frac"].tolist() == pytest.approx([1 / 3, 1 / 3, 0, 0])
    assert f["max_cp_other_frac"].tolist() == pytest.approx([1 / 3, 1 / 3, 0, 0])
    assert f["mean_cp_other_frac"].tolist() == pytest.approx([1 / 3, 1 / 6, 0, 0])
    assert f["sharing_component_frac"].tolist() == pytest.approx([1 / 3, 1 / 3, 0, 0])
    assert components(B).tolist() == [2, 2, 1, 1]


def test_struct_uncoupled_zero():
    B, _ = incidence(np.array([0, 1, 2]), np.array([1, 2, 3]), 3)
    f = struct_features(B)
    assert all((v == 0).all() for v in f.values())


def test_group_prior_role_totals_and_determinism():
    keys = np.array([[2, 0], [3, 1], [5, 5]])
    counts = np.array([10, 5, 1])
    role = (np.arange(100) % 7 == 0).astype(int)
    g1, info = G.sample(role, keys, counts, 100, 4000)
    g2, _ = G.sample(role, keys, counts, 100, 4000)
    assert (g1 == g2).all()
    sizes = np.bincount(g1)
    assert sizes.sum() == 100 and info["agents_in_multi"] == sizes[sizes >= 2].sum()
    # every multi group's role mix is a template
    for g in np.flatnonzero(sizes >= 2):
        assert [sizes[g], role[g1 == g].sum()] in keys.tolist()


def test_real_mint_direction_follows_real_order_not_agent_id():
    # agents 0 (IN) and 1 (OUT) first touch X in the same second; the real order puts agent 1's OUT first, so the
    # node's mint direction is OUT even though agent 0 is processed first; agent 2 then sees mint_out = 1
    p = make_pop([[(10, 0, 1, 0, 7)], [(10, 1, 1, 0, 7)], [(11, 0, 1, 0, 7)]], group=[0, 0, 0], rank=[1, 0, 2])
    r = replay(p)
    (_, f, _), = r.ptr
    assert f[0, 4] == 1.0
    p2 = make_pop([[(10, 0, 1, 0, 7)], [(10, 1, 1, 0, 7)], [(11, 0, 1, 0, 7)]], group=[0, 0, 0], rank=[0, 1, 2])
    (_, f2, _), = replay(p2).ptr
    assert f2[0, 4] == 0.0


def test_tx_table_contract():
    one = dict(ts=[5, 5], eth=[1.0, 1.0], venue=[False, False])
    with pytest.raises(ValueError):          # one transaction, inconsistent endpoints
        tx_table([3, 3], [10, 11], [20, 20], **one)
    with pytest.raises(ValueError):          # one transaction, unequal values
        tx_table([3, 3], [10, 10], [20, 20], ts=[5, 5], eth=[1.0, 1.000001], venue=[False, False])
    T = tx_table([3, 3], [10, 10], [20, 20], **one)   # a genuine duplicate recording collapses
    assert T["n_tx"] == 1 and T["n_dup_rows"] == 1
    T = tx_table([3, 4], [10, 10], [20, 20], **one)   # two same-second transactions stay distinct
    assert T["n_tx"] == 2


def test_ks_continuous_left_limits():
    t = np.repeat([1, 2, 2.1, 2.2], 13).astype(float)
    a = 1 + len(t) / np.log(t).sum()
    assert ks_continuous(t, 1.0, a) == pytest.approx(0.46261, abs=1e-5)


def test_zeta_sampler_matches_discrete_law():
    from scipy.special import zeta
    al = 1.924202863
    x = zeta_sample(al, 2, 1 - np.random.default_rng(0).random(200_000))
    for v in (2, 3, 4):
        p = v ** -al / zeta(al, 2)
        assert abs((x == v).mean() - p) < 4 * np.sqrt(p * (1 - p) / len(x))
    assert x.min() == 2 and (x == np.floor(x)).all()


def test_node_key_namespaces():
    k = [node_key(t, np.array([0, 5])) for t in (S_AGENT, S_VENUE, S_PERIPHERY)]
    allk = np.concatenate(k)
    assert len(np.unique(allk)) == 6
    assert key_kind(k[2]).tolist() == [S_PERIPHERY] * 2
