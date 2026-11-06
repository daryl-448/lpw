"""Drift screen v2 fine-tune (lpw.p1.drift_ft): the CDFs entering the shift-consistency term J, and the trainable set."""
import math

import numpy as np
import torch

from lpw.p1 import drift_ft as D
from lpw.p1.model import CONFIGS, Model, trunc_mog_logpdf
from lpw.p1.store import DUST_LOG10, SUPPLY_LOG10
from lpw.p1.timelaw import EDGES, TimeLaw


def _law(n=6, week=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    lp = torch.log_softmax(torch.randn(2 * n, 64, generator=g) * 2, -1)
    lh = torch.log_softmax(torch.randn(2 * n, 24, generator=g), -1) if week else None
    ld = torch.log_softmax(torch.randn(2 * n, 7, generator=g), -1) if week else None
    t0 = torch.randint(0, 3 * 10 ** 7, (n,), generator=g).double()
    logpk = torch.log_softmax(torch.randn(n, 2, generator=g), -1)
    return logpk, TimeLaw(lp, lh, ld, t0.repeat_interleave(2))


def test_gap_cdf_endpoints_monotone_and_matches_bin_masses():
    logpk, law = _law()
    F = D.gap_cdf(logpk, law)
    n = logpk.shape[0]
    bp = (logpk.double().exp()[:, :, None] * law.bin_probs().view(n, 2, 64)).sum(1)
    assert F.shape == (n, 64)
    assert torch.allclose(F[:, 0], bp[:, 0])                    # same-second atom
    assert torch.allclose(F[:, -1], 1 - bp[:, 63])              # finite-support end: everything but dormant
    assert (F.diff(dim=1) >= -1e-12).all()
    cum = torch.cumsum(bp, 1)
    # the midpoint of finite bin b lies between the CDF just before and just after the bin
    assert (F[:, 1:63] >= cum[:, :62] - 1e-12).all() and (F[:, 1:63] <= cum[:, 1:63] + 1e-12).all()


def test_gap_cdf_flat_week_midpoint_is_linear_in_seconds():
    logpk, law = _law(week=False)
    F = D.gap_cdf(logpk, law)
    n = logpk.shape[0]
    bp = (logpk.double().exp()[:, :, None] * law.bin_probs().view(n, 2, 64)).sum(1)
    lo, hi = EDGES[:-1], EDGES[1:]
    frac = torch.as_tensor((np.sqrt(lo * hi) - lo) / (hi - lo))
    want = torch.cumsum(bp, 1)[:, :62] + bp[:, 1:63] * frac[None]
    assert torch.allclose(F[:, 1:63], want, atol=1e-10)


def test_amt_cdf_matches_numerical_integral_of_the_truncated_mixture():
    torch.manual_seed(1)
    k = 8
    ap = torch.cat([torch.randn(3, k), torch.rand(3, k) * 6 - 2, torch.randn(3, k) * 0.5 - 0.5], 1)
    F = D.amt_cdf(ap, k)
    assert torch.allclose(F[:, 0], torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(F[:, -1], torch.ones(3, dtype=torch.float64), atol=1e-9)
    x = torch.linspace(DUST_LOG10, SUPPLY_LOG10, 200001, dtype=torch.float64)
    for i in range(3):
        dens = trunc_mog_logpdf(ap[i:i + 1].double().expand(len(x), -1), x, k).exp()
        cum = torch.cumulative_trapezoid(dens, x)
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), cum])
        idx = torch.searchsorted(x, torch.as_tensor(D.AMT_GRID, dtype=torch.float64)).clamp(max=len(x) - 1)
        assert torch.allclose(F[i], cum[idx], atol=2e-4)


def test_cdfs_are_differentiable():
    logpk, law0 = _law(n=3)
    lp = law0.log_pdt.clone().requires_grad_(True)
    law = TimeLaw(lp, None, None, law0.t0)
    D.gap_cdf(logpk, law).sum().backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all()
    ap = torch.randn(4, 24, requires_grad=True)
    D.amt_cdf(ap, 8).sum().backward()
    assert torch.isfinite(ap.grad).all() and ap.grad.abs().sum() > 0


def test_trainable_set_is_upper_two_layers_final_norm_and_time_amount_heads():
    m = Model(CONFIGS["lpw_plan_prog"], 5, 4, torch.zeros(5))
    enc, head = D.trainable(m)
    ids = {id(p) for p in enc + head}
    names = {n for n, p in m.named_parameters() if id(p) in ids}
    assert names and all(n.startswith(("enc.layers.2.", "enc.layers.3.", "ln.", "time_head.", "amt_head.")) for n in names)
    for pre in ("enc.layers.2.", "enc.layers.3.", "ln.", "time_head.", "amt_head."):
        assert any(n.startswith(pre) for n in names)
    assert not any(n.startswith(("enc.layers.0.", "enc.layers.1.", "prog_dorm", "e_plan", "kind_head", "gate", "ptr_")) for n in names)
    assert math.isclose(len(enc + head), len(names))
