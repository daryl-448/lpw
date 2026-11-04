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
