"""P3a observation operator, node features, matching and utility helpers."""
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from lpw.p1.profile_gen import ProfileGen
from lpw.p1.store import IN_AGENT, IN_EXO, OUT, PROFILE_FIELDS
from lpw.p3.features import SETS, ZERO_ETH, node_features
from lpw.p3.matching import Matcher
from lpw.p3.observe import IN, observe_synth
from lpw.p3.utility import GroupBoot, c2st, matched_prevalence_ap, transfer_contrast

W0, W1 = 1514764800, 1546300800
V = 10


def _store():
    return SimpleNamespace(n_venues=V, w_start=W0, w_end=W1)


def _synth(rows):
    cols = ("agent", "kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal")
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"agent": pl.Int64, "kind": pl.Int8, "ts": pl.Float64, "cls": pl.Int8, "ent": pl.Int32,
                                "venue": pl.Int16, "amt": pl.Float32, "eth": pl.Float64, "fee": pl.Float64, "bal": pl.Float64})


def _conds(n, origin=W0):
    return [dict(role=i % 2, cohort=0, origin=origin, b0=1.0, prof=None) for i in range(n)]


def test_observe_synth_floors_times_keys_venues_and_drops_silent_agents():
    t = W0 + 100.7
    ev = _synth([(0, OUT, t, 2, 0, -1, -1.0, 0.1, 0.001, 1.0), (0, IN_EXO, t + 0.2, 1, 1, 3, -1.0, 0.1, 0.0, 0.9),
                 (0, OUT, t + 5.0, 1, 2, 3, -1.0, 0.1, 0.001, 1.0), (2, IN_EXO, t, 2, 0, -1, -2.0, 0.01, 0.0, 1.0)])
    ag, oev, info = observe_synth(_store(), _conds(3), np.zeros((3, len(PROFILE_FIELDS))), ev)
    assert info["n_dropped_no_events"] == 1 and info["dropped_by_role"] == {"0": 0, "1": 1}
    assert ag["agent"].to_list() == [0, 1] and ag["role"].to_list() == [0, 0]  # agent 2 renumbered to 1
    assert oev["ts"].dtype == pl.Int64 and oev["ts"].to_list()[:2] == [W0 + 100, W0 + 100]
    a0 = oev.filter(pl.col("agent") == 0)
    assert a0["kind"].to_list() == [OUT, IN, OUT]
    assert a0["cp"].to_list() == [V + 0, 3, 3]  # two VENUE draws at venue 3 are one observed counterparty


def test_observe_synth_rejects_agent_deliveries_and_dust():
    ev = _synth([(0, IN_AGENT, W0 + 1.0, 2, 0, -1, -1.0, 0.1, 0.0, 1.0)])
    with pytest.raises(AssertionError):
        observe_synth(_store(), _conds(1), np.zeros((1, 20)), ev)
    ev = _synth([(0, OUT, W0 + 1.0, 2, 0, -1, -4.5, 10 ** -4.5, 0.0, 1.0)])
    with pytest.raises(AssertionError):
        observe_synth(_store(), _conds(1), np.zeros((1, 20)), ev)


def _obs(events, b0=(1.0,), cohort=(0,), origin=None):
    n = len(b0)
    d = {"agent": np.arange(n), "role": np.zeros(n, np.int64), "cohort": np.array(cohort, np.int64),
         "origin": np.array(origin or [W0] * n, np.int64), "b0": np.array(b0, float)}
    d.update({f: np.zeros(n) for f in PROFILE_FIELDS})
    ev = pl.DataFrame({"agent": [e[0] for e in events], "kind": [e[1] for e in events], "ts": [e[2] for e in events],
                       "eth": [e[3] for e in events], "fee": [e[4] for e in events], "amt": [float(np.log10(e[3])) for e in events],
                       "cp": [e[5] for e in events], "venue": [e[6] for e in events]},
                      schema={"agent": pl.Int64, "kind": pl.Int8, "ts": pl.Int64, "eth": pl.Float64, "fee": pl.Float64,
                              "amt": pl.Float64, "cp": pl.Int64, "venue": pl.Int64})
    return pl.DataFrame(d), ev


def test_node_features_by_hand():
    day = 86400
    ag, ev = _obs([(0, IN, W0 + day, 2.0, 0.0, 3, 3), (0, OUT, W0 + day, 1.0, 0.01, V + 1, -1),
                   (0, OUT, W0 + 3 * day, 1.0, 0.01, V + 1, -1), (0, OUT, W0 + 3 * day + 60, 0.98, 0.0, 3, 3)])
    F = node_features(ag, ev, W0, W1).row(0, named=True)
    assert F["n_events"] == pytest.approx(np.log1p(4)) and F["out_frac"] == 0.75
    assert F["distinct_cp"] == pytest.approx(np.log1p(2)) and F["repeat_share"] == 0.5 and F["top_cp_share"] == 0.5
    assert F["same_second_share"] == pytest.approx(1 / 3) and F["max_second_burst"] == pytest.approx(np.log1p(2))
    assert F["span_days"] == pytest.approx(2 + 60 / day) and F["first_delay_days"] == pytest.approx(1.0)
    end = 1.0 + 2.0 - 1.01 - 1.01 - 0.98
    assert abs(end) <= ZERO_ETH and F["ends_near_zero"] == 1.0 and F["sweep_out_share"] == pytest.approx(1 / 3)
    assert F["venue_share"] == 0.5 and F["top_venue_share"] == 1.0 and F["venue_out_share"] == 0.5


def test_node_features_ignore_input_row_order_and_keep_silent_agents():
    rng = np.random.default_rng(0)
    evs = [(int(a), int(rng.integers(0, 2)), W0 + int(rng.integers(0, 10 ** 6)), float(rng.random() + 0.01), 0.001,
            int(rng.integers(0, 15)), -1) for a in rng.integers(0, 3, 60)]
    evs = sorted(evs, key=lambda e: (e[0], e[2]))
    ag, ev = _obs(evs, b0=(1.0, 0.0, 2.0, 5.0), cohort=(0, 1, 0, 1))
    F1 = node_features(ag, ev, W0, W1)
    F2 = node_features(ag, ev.sample(fraction=1.0, shuffle=True, seed=1).sort("agent", "ts", maintain_order=True), W0, W1)
    cols = [c for c in F1.columns if c not in ("same_second_share", "max_second_burst")]  # ties: all orders equivalent
    assert np.allclose(F1.select(cols).to_numpy(), F2.select(cols).to_numpy())
    assert F1.height == 4 and F1.row(3, named=True)["n_events"] == 0.0  # agent 3 has no events
    for s in SETS.values():
        assert np.isfinite(F1.select(s).to_numpy()).all()


def test_activation_origin_is_uniform_inside_the_partial_last_week():
    st = SimpleNamespace(w_start=W0, w_end=W1, prof_mu=np.zeros(20), prof_sd=np.ones(20))
    n_weeks = -(-(W1 - W0) // (7 * 86400))
    hist = np.zeros(n_weeks); hist[-1] = 1.0  # everybody activates in the last (partial) week
    fake = SimpleNamespace(store=st, p_born={0: 1.0, 1: 1.0}, week_hist={0: hist, 1: hist},
                           sample_raw=lambda roles, rng: np.zeros((len(roles), len(PROFILE_FIELDS))))
    conds = ProfileGen.population(fake, np.zeros(2000, np.int64), np.random.default_rng(0))
    o = np.array([c["origin"] for c in conds])
    lo = W0 + (n_weeks - 1) * 7 * 86400
    assert o.min() >= lo and o.max() < W1 and (o == W1 - 1).sum() < 5
    assert abs(np.median(o) - (lo + W1) / 2) < 0.1 * (W1 - lo)


def _F(n, rng, shift=0.0):
    y = (rng.random(n) < 0.3).astype(int)
    d = {c: rng.normal(size=n) + shift * y for c in ["p_age_days", "p_days_active", "p_days_since_active", "p_normal_sent_cnt",
                                                       "p_normal_recv_cnt", "p_erc20_sent_cnt", "p_erc20_recv_cnt",
                                                       "p_eth_sent_sum", "p_eth_recv_sum", "act_frac"]}
    d["cohort"] = (rng.random(n) < 0.5).astype(float)
    return pl.DataFrame(d), y


def test_matcher_pairs_are_exact_on_cohort_within_caliper_and_unique():
    rng = np.random.default_rng(0)
    m = Matcher(*_F(4000, rng, 0.5))
    F, y = _F(1500, rng, 0.5)
    r = m.match(F, y)
    p, lg, coh = r["pairs"], m.logit(F), F["cohort"].to_numpy()
    assert r["n_pairs"] > 200 and len(np.unique(p[:, 1])) == len(p) and len(np.unique(p[:, 0])) == len(p)
    assert (y[p[:, 0]] == 1).all() and (y[p[:, 1]] == 0).all() and (coh[p[:, 0]] == coh[p[:, 1]]).all()
    assert (np.abs(lg[p[:, 0]] - lg[p[:, 1]]) <= m.caliper).all()
    assert all(abs(v["after"]) < abs(v["before"]) for k, v in r["balance"].items() if k != "cohort")


def test_matched_prevalence_ap_of_a_random_score_is_the_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.5).astype(int)
    assert matched_prevalence_ap(y, rng.random(len(y)), 0.093) == pytest.approx(0.093, abs=0.02)


def test_transfer_contrast_outcomes():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    boot = GroupBoot(np.arange(3000) // 10, B=200)
    good = y + rng.normal(0, 0.5, len(y))
    assert transfer_contrast(y, [good, good], good, boot)["outcome"] == "SUPPORTED"
    noise = rng.random(len(y))
    r = transfer_contrast(y, [noise, noise], good, boot)
    assert r["outcome"] == "NOT SUPPORTED" and r["D_ci95"][1] < 0


def test_c2st_is_chance_on_one_distribution_and_high_on_two():
    rng = np.random.default_rng(0)
    g = np.arange(800) // 4
    assert abs(c2st(rng.normal(size=(800, 3)), rng.normal(size=(800, 3)), g) - 0.5) < 0.06
    assert c2st(rng.normal(size=(800, 3)), rng.normal(size=(800, 3)) + 2, g) > 0.9


def test_precision_recall_k_on_identical_and_disjoint_sets():
    from lpw.p3.utility import precision_recall_k
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    pr = precision_recall_k(X, rng.normal(size=(500, 4)))
    assert pr["precision"] > 0.8 and pr["recall"] > 0.8
    pr = precision_recall_k(X, rng.normal(size=(500, 4)) + 20)
    assert pr["precision"] == 0.0 and pr["recall"] == 0.0


def test_c2st_is_chance_with_unequal_real_groups():
    # Sol r1 S1: identical constant features with uneven real groups must not move the AUC away from 0.5
    rng = np.random.default_rng(0)
    g = np.r_[np.zeros(300), np.arange(1, 101)]  # one big group plus 100 singletons
    assert abs(c2st(np.zeros((400, 2)), np.zeros((400, 2)), g) - 0.5) < 1e-9


def test_precision_counts_a_far_ball_with_a_large_radius():
    # Sol r1 S2: a query covered only by a distant, wide ball
    from lpw.p3.utility import precision_recall_k
    near = np.c_[np.linspace(3.5, 3.51, 100), np.zeros(100)]  # 100 tight real balls nearer the query, none covering it
    wide = np.array([[0.0, 0.0]])  # the 101st-nearest centre; its 5th neighbour is the tight cluster: radius 3.5 > 3
    q = np.array([[3.0, 0.0]])
    pr = precision_recall_k(np.r_[near, wide], np.r_[q, q, q, q, q, q], k=5)
    assert pr["precision"] == 1.0


def test_transfer_contrast_below_the_trtr_gate_is_not_testable():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    noise = rng.random(len(y))
    r = transfer_contrast(y, [noise], noise, GroupBoot(np.arange(2000) // 10, B=50))
    assert r["outcome"] == "NOT TESTABLE" and r["ratio"] is None


def test_f1_threshold_reaches_the_best_oof_f1():
    from lpw.p3.utility import f1_threshold
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.3).astype(int)
    X = (y + rng.normal(0, 1.0, len(y)))[:, None]
    assert 0.0 < f1_threshold("lr", X, y) < 1.0


def test_hour_entropy_is_bitwise_equal_for_equal_hour_count_multisets():
    """value_counts() returns hash order; the entropy must not depend on it (reports/p3_joint_rehearsal.md R2)."""
    rng = np.random.default_rng(4)
    counts = [7, 5, 3, 3, 2, 1, 1, 1, 1]  # one hour-count multiset, laid on different hours for every agent
    n_ag, evs = 40, []
    for a in range(n_ag):
        hours = rng.permutation(24)[:len(counts)]
        for h, c in zip(hours, counts):
            evs += [(a, IN, W0 + int(h) * 3600 + 60 * k, 1.0, 0.0, 3, -1) for k in range(c)]
    evs.sort(key=lambda e: (e[0], e[2]))
    ag, ev = _obs(evs, b0=(1.0,) * n_ag, cohort=(0,) * n_ag)
    for _ in range(3):
        h = node_features(ag, ev, W0, W1)["hour_entropy"].to_numpy()
        assert (h == h[0]).all()
