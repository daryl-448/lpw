from types import SimpleNamespace

import numpy as np

from lpw.p1.batch import agent_weights


def _store():
    # agents with 0, 3, 9 and 99 events; the last is in val; agent 0 has no scored rows (origin >= obs_end)
    off = np.cumsum([0, 0, 3, 9, 99])
    return SimpleNamespace(off=off, origin=np.array([5, 0, 0, 0]), obs_end=np.array([5, 9, 9, 9]),
                           split=np.array(["train", "train", "train", "val"]))


def test_alpha_zero_is_plain_likelihood():
    w = agent_weights(_store(), 0.0)
    assert np.allclose(w, [0, 1, 1, 1])


def test_mean_training_row_weight_is_one_and_alpha_one_equalises_agents():
    st = _store()
    rows = np.diff(st.off) + 1
    for alpha in (0.5, 1.0):
        w = agent_weights(st, alpha)
        tr = np.array([False, True, True, False])
        assert np.isclose((w[tr] * rows[tr]).sum() / rows[tr].sum(), 1.0)
        assert w[0] == 0
    w = agent_weights(st, 1.0)
    assert np.allclose((w * rows)[1:], (w * rows)[1])  # each agent with rows carries the same total weight (val agent included)
