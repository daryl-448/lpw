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


def test_amt_cdf_finite_with_saturated_components_and_finite_gradients():
    k = 8
    ap = torch.zeros(3, 3 * k)
    ap[:, :k] = -200.0
    ap[:, 0] = 0.0
    ap[0, k], ap[0, 2 * k] = -14.0, 0.0    # the one weighted component far below the dust floor (sigma 1)
    ap[1, k], ap[1, 2 * k] = 30.0, 2.0     # ... far above the total supply (sigma e^2)
    ap[2, k], ap[2, 2 * k] = -14.0, -4.0   # ... and below the floor with sigma e^-4: in-support mass ~ e^-150000
    ap.requires_grad_(True)
    F = D.amt_cdf(ap, k)
    assert torch.isfinite(F).all() and (F.diff(dim=1) >= -1e-12).all()
    F.sum().backward()
    assert torch.isfinite(ap.grad).all()
    x = torch.linspace(DUST_LOG10, SUPPLY_LOG10, 400001, dtype=torch.float64)
    idx = torch.searchsorted(x, torch.as_tensor(D.AMT_GRID, dtype=torch.float64)).clamp(max=len(x) - 1)
    for i in range(2):
        dens = trunc_mog_logpdf(ap[i:i + 1].detach().double().expand(len(x), -1), x, k).exp()
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumulative_trapezoid(dens, x)])
        assert torch.allclose(F[i].detach(), cum[idx], atol=2e-3)
    assert F[0, 1] > 0.8   # mass piles up at the dust floor


def test_trunc_mog_mean_matches_numerical_integral():
    from lpw.p1.model import trunc_mog_mean
    p = torch.zeros(1, 6)
    p[0, 2], p[0, 3] = -2.0, 20.0   # equal weights, means -2 and 20 (beyond the supply bound), sigma 1
    assert abs(float(trunc_mog_mean(p, 2)) - (-1.945)) < 2e-3
    torch.manual_seed(3)
    k = 8
    ap = torch.cat([torch.randn(3, k), torch.rand(3, k) * 14 - 6, torch.randn(3, k) * 0.7], 1)
    x = torch.linspace(DUST_LOG10, SUPPLY_LOG10, 400001, dtype=torch.float64)
    for i in range(3):
        dens = trunc_mog_logpdf(ap[i:i + 1].double().expand(len(x), -1), x, k).exp()
        mean = torch.trapezoid(dens * x, x) / torch.trapezoid(dens, x)
        assert abs(float(trunc_mog_mean(ap[i:i + 1], k)) - float(mean)) < 1e-3


def _screen():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p1_drift_screen", "scripts/p1_drift/screen.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


def test_selection_feasible_lowest_score_ties_to_earlier():
    S = _screen()
    steps = [0, 25, 50, 100]
    assert S.pick(steps, [1.0, 0.8, 0.8, 0.5], [True, True, True, False]) == (25, 0.8)  # infeasible best skipped; tie -> earlier
    assert S.pick(steps, [1.0, 1.2, 1.1, 1.3], [True, True, True, True]) == (0, 1.0)    # step 0 can win
    assert S.pick(steps, [1.0, 0.1, 0.1, 0.1], [True, False, False, False]) == (0, 1.0)


def test_max_run_and_mean_absolute_reduction():
    S = _screen()
    assert S.max_run(np.array([1, 1, 2, 2, 2, 3])) == 3 and S.max_run(np.zeros(0)) == 0 and S.max_run(np.array([5])) == 1
    eC, eR = np.array([1.0, -1.0, 2.0]), np.array([0.5, -0.5, 1.0])
    r = S.red(eC, eR, np.ones((10, 3)))
    assert abs(r["reduction"] - 0.5) < 1e-12 and r["lo"] == r["hi"] == 0.5
    # signed values that cancel in a pooled mean still count in the mean absolute discrepancy
    assert S.red(np.array([1.0, -1.0]), np.array([0.1, -0.1]), np.ones((5, 2)))["reduction"] > 0.8
