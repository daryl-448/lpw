"""fidelity.compare's continuous ratio score and drop_largest_group."""
import numpy as np
import polars as pl

from lpw.p1.fidelity import Sample, compare, drop_largest_group, groups_for

W0 = 1514764800


def _frames(n_agents=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_agents):
        n = 3 + (40 if a < 4 else int(rng.integers(0, 8)))  # agents 0-3 are heavy
        ts = W0 + np.cumsum(rng.integers(0, 50000, n))
        rows.append(pl.DataFrame({"agent": np.full(n, a, np.int64), "kind": rng.integers(0, 2, n).astype(np.int8), "ts": ts,
                                  "cls": rng.integers(0, 3, n).astype(np.int8), "ent": rng.integers(0, 5, n).astype(np.int32),
                                  "amt": rng.normal(-1, 1, n).astype(np.float32), "eth": rng.random(n), "fee": rng.random(n) * 1e-3}))
    ag = pl.DataFrame({"agent": np.arange(n_agents), "role": np.arange(n_agents) % 2, "cohort": np.zeros(n_agents, np.int64),
                       "origin": np.full(n_agents, W0 - 86400), "b0": np.ones(n_agents), "age_days": rng.random(n_agents) * 100})
    return ag, pl.concat(rows)


def test_ratio_gmean_is_one_when_the_synthetic_streams_are_the_real_ones():
    ag, ev = _frames()
    real = Sample(ag, ev, W0)
    out = compare(real, Sample(ag, ev, W0), groups_for(ag), paired=True, n_boot=5)
    for g in ("normal", "scam"):
        assert out[g]["n_ratio"] > 0 and abs(out[g]["ratio_gmean"] - 1.0) < 1e-9


def test_drop_largest_group_removes_the_group_with_most_real_events():
    ag, ev = _frames()
    real = Sample(ag, ev, W0)
    gid = np.array([0, 0, 0, 0] + [1 + i // 4 for i in range(20)])  # group 0 = the heavy agents 0-3
    g2, big = drop_largest_group(groups_for(ag), real, gid)
    assert big == 0
    assert not np.isin(np.arange(4), np.concatenate(list(g2.values()))).any()
    assert len(g2["normal"]) == len(groups_for(ag)["normal"]) - 2
