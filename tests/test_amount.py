import numpy as np
import torch

from lpw.p1.generate import _sample_trunc_mog
from lpw.p1.model import trunc_mog_logpdf
from lpw.p1.store import DUST_LOG10, SUPPLY_LOG10


def _params(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    logit = torch.randn(n, k, generator=g)
    mu = torch.randn(n, k, generator=g) * 4 + 2
    ls = torch.randn(n, k, generator=g).clamp(-2, 2)  # includes sigma up to e^2, far-tail components
    return torch.cat([logit, mu, ls], -1)


def test_truncated_mixture_integrates_to_one():
    k = 8
    p = _params(3, k)
    grid = torch.linspace(DUST_LOG10, SUPPLY_LOG10, 20001)
    for row in range(3):
        lp = trunc_mog_logpdf(p[row:row + 1].expand(len(grid), -1), grid, k)
        total = torch.trapezoid(torch.exp(lp.double()), grid.double())
        assert abs(total.item() - 1) < 1e-3


def test_sampling_stays_within_floor_and_supply():
    k = 8
    p = _params(20000, k, seed=1)
    a = _sample_trunc_mog(p, k, torch.Generator().manual_seed(2))
    assert a.min().item() >= DUST_LOG10 and a.max().item() <= SUPPLY_LOG10
    # matches the density: compare the sample mean with the numeric mean for one row repeated
    one = p[:1].expand(20000, -1)
    s = _sample_trunc_mog(one, k, torch.Generator().manual_seed(3)).numpy()
    grid = torch.linspace(DUST_LOG10, SUPPLY_LOG10, 20001)
    dens = torch.exp(trunc_mog_logpdf(p[:1].expand(len(grid), -1), grid, k).double())
    mean = torch.trapezoid(dens * grid.double(), grid.double()).item()
    assert abs(s.mean() - mean) < 0.05


def test_interval_mass_is_stable_in_both_tails():
    # reference values from scipy.stats.norm (logsf/logcdf differences); the old Φ(b) − Φ(a) form returned log 0
    from lpw.p1.model import log_interval_mass
    cases = [((-1.0, 2.0), -0.20016629432446256), ((3.0, 5.0), -6.607938594596893),
             ((30.0, 40.0), -454.32124395634327), ((-40.0, -30.0), -454.32124395634327),
             ((100.0, 200.0), -5005.524208694205)]
    for (a, b), ref in cases:
        got = log_interval_mass(torch.tensor([a], dtype=torch.float64), torch.tensor([b], dtype=torch.float64))
        assert abs(got.item() - ref) < 1e-9 * max(1, abs(ref))


def test_far_tail_mixture_still_integrates_to_one():
    # a single component centred well above total supply with a narrow sigma: all mass sits in a far tail
    k = 1
    p = torch.tensor([[0.0, SUPPLY_LOG10 + 3.0, -1.5]])
    grid = torch.linspace(SUPPLY_LOG10 - 0.2, SUPPLY_LOG10, 200001)
    lp = trunc_mog_logpdf(p.expand(len(grid), -1), grid, k)
    total = torch.trapezoid(torch.exp(lp.double()), grid.double())
    assert abs(total.item() - 1) < 1e-3


def test_far_tail_mass_has_finite_gradients():
    # a component far above supply next to a healthy one: likelihood and gradients must both be finite
    k = 2
    p = torch.tensor([[0.0, 0.0, 10.0, 0.5, -1.5, 0.0]], requires_grad=True)
    lp = trunc_mog_logpdf(p, torch.tensor([0.3]), k)
    lp.sum().backward()
    assert torch.isfinite(lp).all() and torch.isfinite(p.grad).all()


def test_tail_sampling_respects_the_interval():
    from lpw.p1.generate import _trunc_std_normal
    g = torch.Generator().manual_seed(3)
    for a, b in [(30.0, 40.0), (37.0, 40.0), (-40.0, -30.0), (100.0, 200.0), (-1.0, 2.0)]:
        z = _trunc_std_normal(torch.full((5000,), a, dtype=torch.float64), torch.full((5000,), b, dtype=torch.float64), g)
        assert z.min() >= a and z.max() <= b
        if a > 0:  # exponential-tail mean ≈ a + 1/a
            assert abs(z.mean().item() - (a + 1 / a)) < 0.05 / a
