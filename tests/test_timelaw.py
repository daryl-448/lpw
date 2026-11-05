import numpy as np
import torch

from lpw.p1.timelaw import EDGES, MONDAY0, TimeLaw, dt_bin, hour_dow


def _law(n=3, seed=0, week=True):
    g = torch.Generator().manual_seed(seed)
    lp = torch.log_softmax(torch.randn(n, 64, generator=g) * 2, -1)
    lh = torch.log_softmax(torch.randn(n, 24, generator=g), -1) if week else None
    ld = torch.log_softmax(torch.randn(n, 7, generator=g), -1) if week else None
    t0 = torch.tensor([1_620_000_000 + 12_345 * i for i in range(n)], dtype=torch.int64)
    return TimeLaw(lp, lh, ld, t0)


def test_bins_and_clock():
    assert list(dt_bin(np.array([0, 1, 2, 365 * 86400 - 1, 365 * 86400]))) == [0, 1, 3, 62, 63]
    h, d = hour_dow(np.array([MONDAY0, MONDAY0 + 86400 * 3 + 3600 * 5]))
    assert list(h) == [0, 5] and list(d) == [0, 3]


def test_mass_sums_to_one_and_survival_is_consistent():
    for week in (True, False):
        law = _law(week=week)
        assert torch.allclose(law.bin_probs().sum(-1), torch.ones(3, dtype=torch.float64))
        rs = torch.tensor([0, 1, 5, 100, 10_000, 1_000_000, 30_000_000, 31_536_000, 40_000_000])
        for row in range(3):
            rep = lambda x: x[row:row + 1].repeat(len(rs), *([1] * (x.dim() - 1)))
            lw = TimeLaw(rep(law.log_pdt), torch.log(rep(_ph(law))) if week else None,
                         torch.log(rep(_pd(law))) if week else None, rep(law.t0))
            s = torch.exp(lw.log_survival(rs))
            assert s[0] == 1 and torch.all(s[1:] <= s[:-1] + 1e-12)
            assert torch.isclose(s[-1], lw.atom63[0] / lw.Z[0])
            # S(r) at a bin's lower edge = that bin's mass + all later masses + dormant atom
            r = torch.tensor([int(np.ceil(EDGES[40]))])
            one = TimeLaw(lw.log_pdt[:1], torch.log(_ph(lw)[:1]) if week else None, torch.log(_pd(lw)[:1]) if week else None, lw.t0[:1])
            # partial first bin: integrate density from r to the bin end in closed form via W
            expect = (torch.exp(one.log_pdt[0, 41]) / one.wid[40] * (one.W((one.r0 + one.hi[40])[:, None]) - one.W((one.r0 + r)[:, None]))[0, 0]
                      + one.mass[0, 41:].sum() + one.atom63[0]) / one.Z[0]
            assert torch.isclose(torch.exp(one.log_survival(r))[0], expect)


def test_density_integrates_to_one_numerically():
    law = _law(n=1, seed=3)
    # integrate f over [1 s, 365 d) by summing density at every integer second would be too long; integrate per hour piece
    lo, hi = EDGES[0], EDGES[-1]
    pts = np.unique(np.concatenate([EDGES, np.arange(3600 - (law.r0.item() % 3600), hi, 3600)]))
    pts = pts[(pts >= lo) & (pts <= hi)]
    mids = torch.tensor((pts[1:] + pts[:-1]) / 2)
    widths = torch.tensor(np.diff(pts))
    rep = lambda x: x.repeat(len(mids), *([1] * (x.dim() - 1)))
    big = TimeLaw(rep(law.log_pdt), torch.log(rep(_ph(law))), torch.log(rep(_pd(law))), rep(law.t0))
    # density at non-integer mids: evaluate via bin/week formula directly
    b = torch.bucketize(mids, torch.tensor(EDGES), right=True)
    dens = torch.exp(big.log_pdt.gather(1, b[:, None])[:, 0]) / big.wid[b - 1] * big.w_at(big.r0 + mids) / big.Z
    total = (dens * widths).sum() + law.atom0[0] / law.Z[0] + law.atom63[0] / law.Z[0]
    assert abs(total.item() - 1) < 1e-6


def _ph(law):
    # recover p_hour from w: w = 168 p_h p_d, so p_h = sum over dow of w / 168
    return law.w.view(-1, 7, 24).sum(1) / 168


def _pd(law):
    return law.w.view(-1, 7, 24).sum(2) / 168


def test_sampling_matches_bins_and_hours():
    law = _law(n=1, seed=5)
    n = 200_000
    rep = lambda x: x.repeat(n, *([1] * (x.dim() - 1)))
    big = TimeLaw(rep(law.log_pdt), torch.log(rep(_ph(law))), torch.log(rep(_pd(law))), rep(law.t0))
    dt, dormant = big.sample(torch.Generator().manual_seed(1))
    emp = np.bincount(np.where(dormant.numpy(), 63, dt_bin(dt.numpy())), minlength=64) / n
    p = law.bin_probs()[0].numpy()
    # every individual bin, including sub-second-wide ones: a draw never leaves its bin
    assert np.all(np.abs(emp - p) < 4 * np.sqrt(p * (1 - p) / n) + 1e-4)
    # hours of sampled times in long bins follow w
    sel = (~dormant) & (dt > 30 * 86400)
    assert torch.all(dt[dt > 0] >= 1)
    h, d = hour_dow((law.t0[0] + dt[sel]).numpy())
    cell = d * 24 + h
    ec = np.bincount(cell, minlength=168) / sel.sum().item()
    wc = (law.w[0] / law.w[0].sum()).numpy()
    assert np.abs(ec - wc).max() < 0.01


def test_concentrated_bin_samples_stay_in_bin():
    lp = torch.full((1, 64), -50.0)
    lp[0, 2] = 0.0  # bin 2 = [1.32 s, 1.75 s): no whole second inside
    law = TimeLaw(torch.log_softmax(lp, -1).repeat(1000, 1), None, None, torch.full((1000,), 1_620_000_000.0))
    dt, _ = law.sample(torch.Generator().manual_seed(0))
    assert np.all(dt_bin(dt.numpy()) == 2)


def test_subsecond_survival_counts_only_the_same_second_atom():
    for week in (True, False):
        law = _law(week=week)
        r = torch.full((3,), 0.5, dtype=torch.float64)
        expect = 1 - law.bin_probs()[:, 0]
        assert torch.allclose(torch.exp(law.log_survival(r)), expect, atol=1e-12)
