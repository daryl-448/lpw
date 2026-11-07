from types import SimpleNamespace

import numpy as np
import torch

from lpw.p1.batch import collate, to_torch
from lpw.p1.model import CONFIGS, Model
from lpw.p1.plan import N_PLAN_A, N_PLAN_N
from lpw.p1.shallow import D_MAX, N_OUT, VARIANTS, Shallow
from lpw.p1.shallow_fit import cache_batch, law_coefs, row_nll, select, shallow_specs, subset
from lpw.p1.store import IN_AGENT, IN_EXO, NEW, OUT, PERIPHERY, REPEAT, TIER_VENUE, VENUE, derived
from lpw.p1.timelaw import TimeLaw

W0 = 1514764800
V = 4
N_PROF = 4


def _models(variant="kct", seed=0, scale=0.5):
    torch.manual_seed(seed)
    base = Model(CONFIGS["lpw_plan_prog"], V, N_PROF, torch.ones(V)).eval()
    sm = Model(CONFIGS[f"lpw_plan_prog_shallow_{variant}"], V, N_PROF, torch.ones(V)).eval()
    missing, unexpected = sm.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("shallow.") for k in missing)
    with torch.no_grad():
        base.prog_dorm.normal_(0, 1.0)
        sm.prog_dorm.copy_(base.prog_dorm)
        for prm in sm.shallow.parameters():
            prm.normal_(0, scale)
    return base, sm


def _view(seed, n, plan_n, role=0, in_agent_at=None):
    rng = np.random.default_rng(seed)
    ts = W0 + 1000 + np.cumsum(rng.choice([0, 0, 7, 900, 86400 * 3], n)).astype(np.int64)
    kind = rng.choice([OUT, IN_EXO], n)
    if in_agent_at is not None:
        kind[in_agent_at] = IN_AGENT
    cls, ent, venue = np.zeros(n, np.int8), np.zeros(n, np.int32), np.full(n, -1, np.int16)
    e_tier, e_sc, e_venue = [], [], []
    for i in range(n):
        unseen = [x for x in range(V) if x not in e_venue]
        u = rng.random()
        if e_tier and u < 0.4:
            cls[i], ent[i] = REPEAT, rng.integers(len(e_tier))
            venue[i] = e_venue[ent[i]]
        elif unseen and u < 0.7:
            cls[i], ent[i], venue[i] = VENUE, len(e_tier), unseen[0]
            e_tier.append(TIER_VENUE), e_sc.append(True), e_venue.append(unseen[0])
        else:
            cls[i], ent[i] = NEW, len(e_tier)
            e_tier.append(PERIPHERY), e_sc.append(True), e_venue.append(-1)
    eth = 10.0 ** rng.uniform(-2, 1, n)
    fee = np.where(kind == OUT, 1e-3, 0.0)
    bal, b = np.empty(n), 50.0
    for i in range(n):
        bal[i] = b
        b = b + eth[i] if kind[i] != OUT else b - eth[i] - fee[i]
    v = dict(ts=ts, kind=kind.astype(np.int8), cls=cls, ent=ent, venue=venue, amt=np.log10(eth).astype(np.float32), eth=eth,
             fee=fee, bal=bal, e_tier=np.array(e_tier, np.int8), e_sc=np.array(e_sc, bool), e_venue=np.array(e_venue, np.int16),
             origin=W0, obs_end=W0 + 86400 * 300, role=role, cohort=0, prof=np.zeros(N_PROF, np.float32), bal_end=b,
             plan_n=plan_n, plan_a=1)
    v.update(derived(v, W0))
    return v


def _batch():
    """windows: known plans (normal and scam), one unknown plan, one stream with an IN_agent token; every prefix."""
    views = [_view(0, 9, 4), _view(1, 3, 2, role=1), _view(2, 8, N_PLAN_N), _view(3, 7, 4, in_agent_at=2), _view(4, 1, 1)]
    for v in views:
        if v["plan_n"] == N_PLAN_N:
            v["plan_a"] = N_PLAN_A
    specs = [(0, np.arange(0, len(v["ts"]) + 1)) for v in views]
    b = to_torch(collate(views, specs, V, W0), "cpu")
    agent = np.concatenate([np.full(len(js), i) for i, (_, js) in enumerate(specs)])
    obs_end = np.array([v["obs_end"] for v in views])
    return b, agent, obs_end


def test_law_coefs_reproduce_timelaw_density_and_survival_for_any_bin_probabilities():
    torch.manual_seed(0)
    n = 12
    lh, ld = torch.log_softmax(torch.randn(n, 24), -1), torch.log_softmax(torch.randn(n, 7), -1)
    t0 = W0 + torch.rand(n, dtype=torch.float64) * 86400 * 30
    r = torch.tensor([0.0, 0.4, 1.0, 1.5, 59.0, 3600.0, 86400.0 * 2, 86400.0 * 364.9, 86400.0 * 365, 86400.0 * 400, 17.0, 1e6],
                     dtype=torch.float64)
    law0 = TimeLaw(torch.log_softmax(torch.randn(n, 64), -1), lh, ld, t0)
    a, s, bb, c = law_coefs(law0, r)
    lp = torch.log_softmax(torch.randn(n, 64) * 2, -1)
    law = TimeLaw(lp, lh, ld, t0)
    p = lp.double().exp()
    logZ = torch.log((p * a).sum(-1))
    assert torch.allclose(logZ, law.logZ, atol=1e-10)
    logf = lp.double().gather(1, bb[:, None])[:, 0] + c - logZ
    assert torch.allclose(logf, law.log_density(r), atol=1e-9)
    logS = torch.log((p * s).sum(-1).clamp(min=1e-300)) - logZ
    assert torch.allclose(logS, law.log_survival(r), atol=1e-9)


def test_cached_nll_matches_the_model_likelihood_row_by_row():
    base, sm = _models()
    b, agent, obs_end = _batch()
    base.plan_inference = sm.plan_inference = True
    C = cache_batch(base, dict(b), agent, obs_end)
    t0, t1 = base.terms(dict(b)), sm.terms(dict(b))
    oi = t0["o_row"]
    total = lambda t: (t["kind"] + t["time"] + t["cens"]).index_add(0, oi, t["cp"].double())
    # base: on censored rows and owned NEW rows (no pointer / venue term) the cache reproduces the model exactly
    new = torch.zeros_like(t0["own"]).index_fill(0, oi[b["o_cls"] == NEW], True)
    plain = t0["cen"] | new
    assert plain.sum() >= 5 and (t0["own"] & ~new).sum() >= 5
    assert torch.allclose(row_nll(None, C)[plain], total(t0)[plain], atol=1e-6)
    # with offsets: the change of every row matches the model's change (pointer / venue terms do not move)
    d_cache = row_nll(sm.shallow, C) - row_nll(None, C)
    d_model = total(t1) - total(t0)
    assert torch.allclose(d_cache, d_model, atol=2e-5)
    assert d_cache.abs().max() > 1e-2  # the offsets do act


def test_unknown_plan_and_deep_rows_are_bit_identical_to_the_base():
    base, sm = _models()
    b, agent, obs_end = _batch()
    for mode in (True, False):
        base.plan_inference = sm.plan_inference = mode
        t0, t1 = base.terms(dict(b)), sm.terms(dict(b))
        rb = b["row_b"]
        inactive = (b["plan_n"][rb] == N_PLAN_N) | (b["row_nown"] > D_MAX) | (not mode)
        for k in ("kind", "time", "cens"):
            assert torch.equal(t0[k][inactive], t1[k][inactive])
        oi = t0["o_row"]
        assert torch.equal(t0["cp"][inactive[oi]], t1["cp"][inactive[oi]])
        if mode:
            assert not torch.equal(t0["kind"][~inactive], t1["kind"][~inactive])
    # s-unknown (plan_inference False): every row identical, the whole likelihood unchanged
    assert torch.equal(t0["amt"], t1["amt"])


def test_disabled_blocks_stay_zero_and_rows_past_depth_or_unknown_plan_get_no_offset():
    sh = Shallow(VARIANTS["k"])
    with torch.no_grad():
        for prm in sh.parameters():
            prm.fill_(1.0)
    role = torch.tensor([0, 1, 0, 0])
    plan = torch.tensor([3, 3, N_PLAN_N, 5])
    nown = torch.tensor([0, D_MAX, 1, D_MAX + 1])
    pk, pc = torch.tensor([3, 0, 1, 2]), torch.tensor([3, 0, 1, 2])
    off, act = sh.offsets(role, plan, nown, pk, pc)
    assert off.shape == (4, N_OUT) and act.tolist() == [True, True, False, False]
    assert torch.all(off[:2, 0] == 3.0) and torch.all(off[:2, 1:] == 0) and torch.all(off[2:] == 0)


def test_subset_reindexes_owned_rows():
    base, _ = _models()
    b, agent, obs_end = _batch()
    base.plan_inference = True
    C = cache_batch(base, dict(b), agent, obs_end)
    mask = torch.as_tensor(agent % 2 == 0)
    assert torch.allclose(row_nll(None, subset(C, mask)), row_nll(None, C)[mask])


def test_shallow_specs_cover_prefixes_up_to_dmax_owned_events():
    kind = np.array([IN_EXO, IN_AGENT, OUT, OUT, IN_EXO, OUT, OUT, OUT, IN_EXO])
    st = SimpleNamespace(off=np.array([0, 9, 9]), kind=kind, origin=np.array([0, 5]), obs_end=np.array([10, 5]))
    specs = shallow_specs(st, [0, 1])
    assert len(specs) == 1  # agent 1 has no observation span
    a, s, js = specs[0]
    # owned before prefix k: 0,1,1,2,3,4,5,...: prefixes 0..5 have <= 4
    assert a == 0 and s == 0 and list(js) == [0, 1, 2, 3, 4, 5]


def test_select_takes_zero_when_no_variant_beats_it_by_one_se_and_largest_lambda_within_se():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 4)
    zero = rng.normal(1.0, 0.1, 200)
    sc = {"zero": zero}
    d = rng.normal(0, 1e-2, 200)
    d += -1e-4 - d.mean()  # every variant 1e-4 better on average, far inside its group-clustered SE
    for v in VARIANTS:
        for lam in (1.0, 10.0, 100.0):
            sc[(v, lam)] = zero + d
    out = select(sc, groups)
    assert out["primary"] == "zero"
    sc2 = dict(sc)
    u = np.repeat(np.tile([1e-2, -1e-2], 25), 4)  # mean-zero group-level noise: the differences get a nonzero clustered SE
    for lam, d, e in ((1.0, 0.05, u), (10.0, 0.0499, -u), (100.0, 0.02, u)):
        sc2[("k", lam)] = zero - d + e
    out2 = select(sc2, groups)
    assert out2["variant_lam"]["k"] == 10.0 and out2["primary"] == "k"


def test_every_enabled_offset_gets_a_gradient_at_zero_init():
    """regression: the kind offset was routed through a zero-offset bypass, so its gradient at init was 0."""
    base, _ = _models()
    b, agent, obs_end = _batch()
    base.plan_inference = True
    C = cache_batch(base, dict(b), agent, obs_end)
    for v, blocks in VARIANTS.items():
        sh = Shallow(blocks)
        row_nll(sh, C).sum().backward()
        g = sh.A.grad.abs().sum((0, 1, 2)) + sh.Bk.grad.abs().sum((0, 1, 2)) + sh.Bc.grad.abs().sum((0, 1, 2))
        on = sh.cols > 0
        assert torch.all(g[on] > 0) and torch.all(g[~on] == 0), v
