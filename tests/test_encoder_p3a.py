"""Addendum-B encoder pieces."""
import numpy as np
import polars as pl
import torch

from lpw.p3 import encoder as E


def test_raw_sequences_by_hand():
    ag = pl.DataFrame({"agent": [0, 1, 2]})
    ev = pl.DataFrame({"agent": [0, 0, 2], "kind": [1, 0, 0], "ts": [86400 * 4 + 3600, 86400 * 4 + 3700, 10],
                       "eth": [1.0, 0.5, 0.1], "fee": [0.0, 1e-9, 0.0], "amt": [0.0, -0.3, -1.0], "cp": [5, 5, 7], "venue": [-1, -1, 3]},
                      schema={"agent": pl.Int64, "kind": pl.Int8, "ts": pl.Int64, "eth": pl.Float64, "fee": pl.Float64,
                              "amt": pl.Float64, "cp": pl.Int64, "venue": pl.Int64})
    s = E.raw_sequences(ag, ev)
    assert [len(x) for x in s] == [2, 0, 1]
    a = s[0]
    assert a[0, 1] == 1 and a[1, 0] == 1 and a[0, 3] == 1 and a[1, 3] == 0      # IN then OUT; first flag
    assert np.isclose(a[1, 2], np.log1p(100)) and a[0, 2] == 0                   # gap
    assert np.isclose(a[1, 5], np.log1p(1.0)) and a[1, 11] == 1 and a[0, 11] == 0  # fee; seen counterparty
    assert np.isclose(a[0, 6], np.sin(2 * np.pi / 24)) and np.isclose(a[0, 8], 0.0)  # hour 1, Monday (dow 0)
    assert s[2][0, 10] == 1                                                       # venue


def test_nt_xent_prefers_matching_pairs_and_crop_keeps_one_event():
    z = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    assert E.nt_xent(z, z) < E.nt_xent(z, z.roll(1, 0))
    rng = np.random.default_rng(0)
    assert len(E.crop(np.zeros((1, 12)), rng)) == 1 and len(E.crop(np.zeros((10, 12)), rng)) == 8


def test_embed_is_order_invariant_and_unit_norm():
    torch.manual_seed(0)
    enc = E.Encoder().eval()
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(n, E.N_IN)).astype(np.float32) for n in (3, 1, 7, 2)]
    e1 = E.embed(enc, seqs, dev="cpu")
    e2 = E.embed(enc, seqs[::-1], dev="cpu")[::-1]
    assert np.allclose(e1, e2, atol=1e-5) and np.allclose(np.linalg.norm(e1, axis=1), 1, atol=1e-5)
