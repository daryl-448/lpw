import numpy as np
import torch

from lpw.p1.model import CONFIGS, Model
from lpw.p1.pace import MIN_PLAN, PaceRef, _pos_mean, bin_log_moments, n_features
from lpw.p1.plan import N_PLAN_A, N_PLAN_N
from lpw.p1.timelaw import TimeLaw

W0, W1 = 1514764800, 1546300800  # 2018-01-01, 2019-01-01 UTC
N_PROF = 4


def _ref(seed=0, gamma=0.5):
    torch.manual_seed(seed)
    ref = PaceRef(N_PROF).double()
    ref.set_window(W0, W1)
    with torch.no_grad():
        for p in (ref.kind.weight, ref.kind.bias, ref.time.weight, ref.time.bias):
            p.normal_(0, 0.5)
    ref.fitted.fill_(True)
    ref.gamma.fill_(gamma)
    return ref


def _inputs(n=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    lp = torch.log_softmax(torch.randn(2 * n, 64, generator=g) * 2, -1)
    lh = torch.log_softmax(torch.randn(2 * n, 24, generator=g), -1)
    ld = torch.log_softmax(torch.randn(2 * n, 7, generator=g), -1)
    t0 = torch.tensor(np.linspace(W0 + 1000, W1 - 86400 * 20, n)).double()
    logpk = torch.log_softmax(torch.randn(n, 2, generator=g), -1)
    x = torch.rand(n, n_features(N_PROF), generator=g, dtype=torch.float64)
    elig = torch.arange(n) % 3 != 0
    return lp, lh, ld, t0, logpk, (x, elig)


def test_tilt_preserves_atoms_and_kind_mass_and_hits_target():
    ref = _ref()
    lp, lh, ld, t0, logpk, pace = _inputs()
    new = ref.adapt(lp, lh, ld, t0, logpk, pace)
    t0r = t0.repeat_interleave(2)
    a, b = TimeLaw(lp, lh, ld, t0r), TimeLaw(new, lh, ld, t0r)
    pa, pb = a.bin_probs(), b.bin_probs()
    assert torch.allclose(pa[:, 0], pb[:, 0], rtol=1e-10, atol=1e-14) and torch.allclose(pa[:, 63], pb[:, 63], rtol=1e-10, atol=1e-14)
    assert torch.allclose(pa[:, 1:63].sum(1), pb[:, 1:63].sum(1), rtol=1e-10)
    assert torch.allclose(pb.sum(1), torch.ones(len(pb), dtype=torch.float64))
    x, elig = pace
    rows = torch.nonzero(~elig)[:, 0]
    r2 = torch.stack([2 * rows, 2 * rows + 1], 1).reshape(-1)
    assert torch.equal(new[r2], lp.double()[r2])  # bypass rows bit-identical
    idx = torch.nonzero(elig)[:, 0]
    e2 = torch.stack([2 * idx, 2 * idx + 1], 1).reshape(-1)
    la, lb = TimeLaw(lp[e2], lh[e2], ld[e2], t0[idx].repeat_interleave(2)), TimeLaw(new[e2], lh[e2], ld[e2], t0[idx].repeat_interleave(2))
    ell = bin_log_moments(la)
    mu_h, _ = _pos_mean(torch.log(la.mass), ell, la.mass.sum(-1), la.Z, logpk[idx].double())
    lpr, rl = ref.law(x[idx], t0[idx])
    mu_r, _ = _pos_mean(torch.log(rl.mass), bin_log_moments(rl), rl.mass.sum(-1), rl.Z, lpr.double())
    mu_new, _ = _pos_mean(torch.log(lb.mass), bin_log_moments(lb), lb.mass.sum(-1), lb.Z, logpk[idx].double())
    assert torch.allclose(bin_log_moments(lb), ell)  # the moments depend on the weekly weights only
    tgt = 0.5 * mu_h + 0.5 * mu_r
    ok = (tgt - mu_h).abs() > 0  # every eligible row moved toward the reference
    assert torch.allclose(mu_new, tgt, atol=1e-8), (mu_new - tgt).abs().max()
    assert ok.all()


def test_gamma_one_and_no_eligible_rows_are_exact_bypasses():
    lp, lh, ld, t0, logpk, (x, elig) = _inputs()
    assert torch.equal(_ref(gamma=1.0).adapt(lp, lh, ld, t0, logpk, (x, elig)), lp.double())
    assert torch.equal(_ref(gamma=0.5).adapt(lp, lh, ld, t0, logpk, (x, torch.zeros_like(elig))), lp.double())


def test_tilt_is_invariant_to_a_uniform_shift_of_the_finite_logits_within_a_kind():
    """the dormant residual shifts a kind's other log-probabilities by one constant; with a single kind the tilt must
    not depend on it (with two kinds it may, through the kinds' shares of positive-finite gaps)"""
    ref = _ref()
    lp, lh, ld, t0, logpk, pace = _inputs()
    logpk = torch.log(torch.tensor([[1.0, 0.0]]).expand_as(logpk))
    out = torch.log_softmax(torch.cat([lp[:, :63], lp[:, 63:] + 1.7], 1), -1)
    a, b = ref.adapt(lp, lh, ld, t0, logpk, pace), ref.adapt(out, lh, ld, t0, logpk, pace)
    assert torch.allclose((a - lp.double())[:, 1:63], (b - out.double())[:, 1:63], atol=1e-9)


def test_features_and_eligibility():
    ref = _ref()
    n = 6
    B = {"row_b": torch.arange(n), "row_t0": torch.full((n,), W0 + 86400.0 * 40, dtype=torch.float64),
         "row_nown": torch.tensor([0, 1, 12, 1, 3, 1]),
         "_plan_n_eff": torch.tensor([MIN_PLAN, MIN_PLAN, MIN_PLAN, N_PLAN_N, MIN_PLAN - 1, 9]),
         "_plan_a_eff": torch.tensor([1, 2, 3, N_PLAN_A, 0, 4]), "role": torch.tensor([0, 1, 0, 0, 1, 0]),
         "cohort": torch.tensor([1, 1, 0, 1, 0, 1]), "prof": torch.zeros(n, N_PROF),
         "origin": torch.full((n,), W0 + 86400 * 10, dtype=torch.int64)}
    x, elig = ref.features(B)
    assert x.shape == (n, n_features(N_PROF))
    # depth 0 / plan unknown / plan < 10 / state past BELOW are not eligible; plan >= 1000 at depth 1 is
    assert elig.tolist() == [False, True, False, False, False, True]
    o = 2 + N_PROF
    assert x[3, o:o + N_PLAN_N + N_PLAN_A].abs().sum() == 0  # unknown plan: zero plan one-hots
    month = x[:, -12:]
    assert (month.argmax(1) == 1).all() and (month.sum(1) == 1).all()  # 40 days after 1 Jan is February


def test_model_kind_time_unknown_rows_bit_identical_to_progress_model():
    torch.manual_seed(0)
    base = Model(CONFIGS["lpw_plan_prog"], 3, N_PROF, torch.zeros(3)).double().eval()
    pm = Model(CONFIGS["lpw_plan_prog_pace"], 3, N_PROF, torch.zeros(3)).double().eval()
    missing, unexpected = pm.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("pace.") for k in missing)
    pm.pace.load_state_dict(_ref().state_dict())
    n = 10
    hk, ho = torch.randn(n, 128, dtype=torch.float64), torch.randn(n, 128, dtype=torch.float64)
    t0 = torch.full((n,), W0 + 86400.0 * 50, dtype=torch.float64)
    prog = torch.tensor([3] * 5 + [0] * 5)
    x = torch.rand(n, n_features(N_PROF), dtype=torch.float64)
    elig = torch.tensor([False] * 5 + [True] * 5)
    _, la, _ = base.kind_time(hk, ho, t0, prog=prog)
    _, lb, _ = pm.kind_time(hk, ho, t0, prog=prog, pace=(x, elig))
    assert torch.equal(la.log_pdt[:10], lb.log_pdt[:10]) and not torch.equal(la.log_pdt[10:], lb.log_pdt[10:])


def test_refit_gradient_through_the_tilt_matches_finite_differences():
    """the root theta depends on the dormant residuals through the kinds' positive-finite shares; the gradient of a
    likelihood wrt per-kind dormant residuals must match central differences (asymmetric two-kind case)"""
    from lpw.p1.pace import row_nll
    ref = _ref(seed=2, gamma=0.5)
    n = 30
    g = torch.Generator().manual_seed(5)
    out0 = torch.randn(2 * n, 64, generator=g, dtype=torch.float64) * 1.5
    lh = torch.log_softmax(torch.randn(2 * n, 24, generator=g, dtype=torch.float64), -1)
    ld = torch.log_softmax(torch.randn(2 * n, 7, generator=g, dtype=torch.float64), -1)
    t0 = torch.tensor(np.linspace(W0 + 5000, W1 - 86400 * 30, n)).double()
    logpk = torch.log_softmax(torch.randn(n, 2, generator=g, dtype=torch.float64) * 2, -1)
    x = torch.rand(n, n_features(N_PROF), generator=g, dtype=torch.float64)
    elig = torch.ones(n, dtype=torch.bool)
    r = torch.exp(torch.rand(n, generator=g, dtype=torch.float64) * 15)
    typ = torch.where(torch.rand(n, generator=g) < 0.7, 1, 3)
    tk = (torch.rand(n, generator=g) < 0.4).long()

    def loss(res):
        lp = torch.log_softmax(out0 + torch.cat([torch.zeros(2 * n, 63, dtype=torch.float64), res.repeat(n)[:, None]], 1), -1)
        new = ref.adapt(lp, lh, ld, t0, logpk, (x, elig))
        return row_nll(logpk, TimeLaw(new, lh, ld, t0.repeat_interleave(2)), r, typ, tk).sum()

    res = torch.tensor([-2.0, 1.0], dtype=torch.float64, requires_grad=True)
    ref.reset_stats()
    gr = torch.autograd.grad(loss(res), res)[0]
    assert ref.stats["clipped"] == 0
    h = 1e-5
    with torch.no_grad():
        fd = torch.stack([(loss(res + h * e) - loss(res - h * e)) / (2 * h) for e in torch.eye(2, dtype=torch.float64)])
    assert torch.allclose(gr, fd, rtol=1e-5, atol=1e-7), (gr, fd)


def test_non_finite_reference_rows_are_bypassed_without_nan_gradients():
    ref = _ref(seed=3, gamma=0.5)
    with torch.no_grad():
        ref.time.bias[5] = float("nan")  # poisons the reference law of every row
    lp, lh, ld, t0, logpk, pace = _inputs(n=12)
    lp = lp.double().requires_grad_(True)
    ref.reset_stats()
    new = ref.adapt(lp, lh, ld, t0, logpk, pace)
    assert torch.isfinite(new).all() and ref.stats["invalid"] == int(pace[1].sum())
    (g,) = torch.autograd.grad(new.sum(), lp)
    assert torch.isfinite(g).all()


def test_non_finite_model_rows_are_untouched_and_gradients_stay_finite():
    ref = _ref(seed=3, gamma=0.5)
    lp, lh, ld, t0, logpk, pace = _inputs(n=12)
    x, elig = pace
    bad = int(torch.nonzero(elig)[0, 0])
    lh = lh.clone()
    lh[2 * bad, 5] = float("nan")  # one weekly-logit row of one eligible prefix
    lp = lp.double().requires_grad_(True)
    ref.reset_stats()
    new = ref.adapt(lp, lh, ld, t0, logpk, pace)
    assert ref.stats["invalid"] == 1
    rows = torch.tensor([2 * bad, 2 * bad + 1])
    assert torch.equal(new[rows], lp.double()[rows])
    assert torch.isfinite(new).all()
    (g,) = torch.autograd.grad(new.sum(), lp)
    assert torch.isfinite(g).all()


def test_kind_only_gradient_through_the_root_matches_finite_differences():
    """time heads frozen, kind logits trainable: theta still depends on the kinds' shares pi"""
    from lpw.p1.pace import row_nll
    ref = _ref(seed=2, gamma=0.5)
    n = 30
    g = torch.Generator().manual_seed(9)
    lp = torch.log_softmax(torch.randn(2 * n, 64, generator=g, dtype=torch.float64) * 1.5, -1)
    lh = torch.log_softmax(torch.randn(2 * n, 24, generator=g, dtype=torch.float64), -1)
    ld = torch.log_softmax(torch.randn(2 * n, 7, generator=g, dtype=torch.float64), -1)
    t0 = torch.tensor(np.linspace(W0 + 5000, W1 - 86400 * 30, n)).double()
    k0 = torch.randn(n, 2, generator=g, dtype=torch.float64)
    x = torch.rand(n, n_features(N_PROF), generator=g, dtype=torch.float64)
    elig = torch.ones(n, dtype=torch.bool)
    r = torch.exp(torch.rand(n, generator=g, dtype=torch.float64) * 15)
    typ = torch.where(torch.rand(n, generator=g) < 0.7, 1, 3)
    tk = (torch.rand(n, generator=g) < 0.4).long()

    def loss(kb):
        logpk = torch.log_softmax(k0 + torch.stack([torch.zeros(n, dtype=torch.float64), kb.expand(n)], 1), -1)
        new = ref.adapt(lp, lh, ld, t0, logpk, (x, elig))
        return row_nll(logpk, TimeLaw(new, lh, ld, t0.repeat_interleave(2)), r, typ, tk).sum()

    kb = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    ref.reset_stats()
    gr = torch.autograd.grad(loss(kb), kb)[0]
    assert ref.stats["clipped"] == 0 and ref.stats["invalid"] == 0
    h = 1e-5
    with torch.no_grad():
        fd = (loss(kb + h) - loss(kb - h)) / (2 * h)
    assert torch.allclose(gr, fd, rtol=1e-5, atol=1e-7), (gr, fd)


def test_grad_and_no_grad_forwards_are_bit_identical():
    ref = _ref(seed=4, gamma=0.5)
    lp, lh, ld, t0, logpk, pace = _inputs(n=24)
    with torch.no_grad():
        a = ref.adapt(lp, lh, ld, t0, logpk, pace)
    b = ref.adapt(lp.double().requires_grad_(True), lh, ld, t0, logpk.requires_grad_(True), pace)
    assert torch.equal(a, b.detach())


def test_non_finite_origin_rows_are_bypassed():
    ref = _ref(seed=3, gamma=0.5)
    lp, lh, ld, t0, logpk, pace = _inputs(n=12)
    x, elig = pace
    bad = int(torch.nonzero(elig)[0, 0])
    t0 = t0.clone()
    t0[bad] = float("nan")
    ref.reset_stats()
    new = ref.adapt(lp, lh, ld, t0, logpk, pace)
    rows = torch.tensor([2 * bad, 2 * bad + 1])
    assert ref.stats["invalid"] == 1 and torch.equal(new[rows], lp.double()[rows])
