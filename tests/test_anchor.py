import numpy as np
import torch

from lpw.p1.anchor import corrupt_view, merge_corrupted
from lpw.p1.model import CONFIGS, Model
from lpw.p1.plan import N_PLAN_A, N_PLAN_N
from lpw.p1.store import IN_EXO, OUT, derived
from lpw.p1.timelaw import dt_bin

W0 = 1514764800
N_PROF = 4


def _models():
    torch.manual_seed(0)
    base = Model(CONFIGS["lpw_plan_prog"], 3, N_PROF, torch.zeros(3)).double().eval()
    am = Model(CONFIGS["lpw_plan_prog_anchor"], 3, N_PROF, torch.zeros(3)).double().eval()
    missing, unexpected = am.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("anchor.") for k in missing)
    am.anchor.double()
    return base, am


def _rows(n=10):
    hk, ho = torch.randn(n, 128, dtype=torch.float64), torch.randn(n, 128, dtype=torch.float64)
    t0 = torch.full((n,), W0 + 86400.0 * 50, dtype=torch.float64)
    prog = torch.tensor([0, 1, 2, 3, 0] * (n // 5))
    pn = torch.tensor([N_PLAN_N] * (n // 2) + [7] * (n - n // 2))
    pa = torch.tensor([N_PLAN_A] * (n // 2) + [2] * (n - n // 2))
    return hk, ho, t0, prog, pn, pa


def test_zero_initialised_adapters_are_identical_to_the_base():
    base, am = _models()
    hk, ho, t0, prog, pn, pa = _rows()
    _, la, _ = base.kind_time(hk, ho, t0, prog=prog)
    _, lb, _ = am.kind_time(hk, ho, t0, prog=prog, anchor=(pn, pa))
    assert torch.equal(la.log_pdt, lb.log_pdt) and torch.equal(la.logZ, lb.logZ)
    ap = torch.randn(10, 24, dtype=torch.float64)
    assert torch.equal(am.anchor.apply_amt(ap, torch.randn(10, 512, dtype=torch.float64), pn, pa), ap)


def test_unknown_plan_rows_bypass_trained_adapters_exactly():
    base, am = _models()
    with torch.no_grad():
        for prm in am.anchor.parameters():
            prm.normal_(0, 0.3)
    hk, ho, t0, prog, pn, pa = _rows()
    _, la, _ = base.kind_time(hk, ho, t0, prog=prog)
    _, lb, _ = am.kind_time(hk, ho, t0, prog=prog, anchor=(pn, pa))
    assert torch.equal(la.log_pdt[:10], lb.log_pdt[:10]) and not torch.equal(la.log_pdt[10:], lb.log_pdt[10:])
    # the dormant bin's logit is not moved by the adapter (only renormalization touches it)
    ap = torch.randn(10, 24, dtype=torch.float64)
    out = am.anchor.apply_amt(ap, torch.randn(10, 512, dtype=torch.float64), pn, pa)
    assert torch.equal(out[:5], ap[:5]) and not torch.equal(out[5:], ap[5:])


def _view():
    rng = np.random.default_rng(0)
    n = 30
    ts = W0 + np.cumsum(rng.choice([0, 0, 5, 600, 86400], n)).astype(np.int64) + 1000
    kind = rng.choice([OUT, IN_EXO], n)
    eth = 10.0 ** rng.uniform(-2, 1, n)
    fee = np.where(kind == OUT, 1e-3, 0.0)
    bal = np.empty(n)
    b = 50.0
    for i in range(n):
        bal[i] = b
        b = b + eth[i] if kind[i] != OUT else b - eth[i] - fee[i]
    v = dict(ts=ts, kind=kind, eth=eth, fee=fee, bal=bal, amt=np.log10(eth).astype(np.float32), origin=W0, bal_end=b)
    v.update(derived(v, W0))
    return v


def test_corruption_shifts_levels_and_keeps_calendar_zero_gaps_and_ledger():
    v = _view()
    w = corrupt_view(v, 0.3, 0.5)
    assert np.array_equal(w["ts"], v["ts"]) and np.array_equal(w["kind"], v["kind"])
    z = v["dt"] == 0
    assert np.all(w["dt"][z] == 0) and np.allclose(w["dt"][~z], v["dt"][~z] * 10 ** 0.3)
    assert np.array_equal(w["dtbin"], dt_bin(w["dt"]))
    assert np.allclose(w["amt"], v["amt"] + 0.5, atol=1e-6)
    nxt = np.r_[w["bal"][1:], w["bal_end"]]
    flow = np.where(w["kind"] != OUT, w["eth"], -w["eth"] - w["fee"])
    assert np.allclose(w["bal"] + flow, nxt) and np.all(w["bal"] >= 0)
    assert np.array_equal(v["amt"], _view()["amt"])  # the input view is not modified


def test_merge_takes_history_fields_and_keeps_targets_clean():
    clean = {"tok_cont": 1, "tok_dtbin": 1, "e_tin": 1, "o_led": 1, "o_amt": 1, "o_tbin": 1, "row_r": 1, "plan_n": 1}
    cor = {k: 2 for k in clean}
    m = merge_corrupted(clean, cor)
    assert all(m[k] == 2 for k in ("tok_cont", "tok_dtbin", "e_tin", "o_led"))
    assert all(m[k] == 1 for k in ("o_amt", "o_tbin", "row_r", "plan_n"))


def _cache(am, n=40, seed=1):
    from lpw.p1.anchor import build_cache  # noqa: F401  (layout reference)
    torch.manual_seed(seed)
    hk, ho, t0, prog, pn, pa = _rows(n)
    am._capture = {}
    am.kind_time(hk, ho, t0, prog=prog, anchor=(pn, pa))
    cap, am._capture = am._capture, None
    x = torch.randn(n, 512, dtype=torch.float64)
    with torch.no_grad():
        ap = am.amt_head(x)
    return {"logpk": cap["logpk"].detach(), "ho": ho, "out": cap["out"].detach().view(n, 2, -1), "t0": t0,
            "r": torch.rand(n, dtype=torch.float64) * 1e4 + 1.0, "typ": torch.ones(n, dtype=torch.int8),
            "tk": torch.randint(0, 2, (n,)), "prog": prog, "pn": pn, "pa": pa, "agent": torch.arange(n) % 5,
            "own_of_row": torch.arange(n), "x": x, "ap": ap, "amt": torch.randn(n, dtype=torch.float64) - 1.0, "opn": pn, "opa": pa}


def test_regularised_fit_keeps_step_zero_without_evaluations():
    from lpw.p1.anchor import fit_adapters_reg
    _, am = _models()
    C = _cache(am)
    init = {k: v.clone() for k, v in am.anchor.state_dict().items()}
    info = fit_adapters_reg(am, C, C, max_steps=3, eval_every=10, batch_rows=16, lr=1e-2, log=lambda *a: None)
    assert info["best_step"] == 0
    assert all(torch.equal(init[k], v) for k, v in am.anchor.state_dict().items())


def test_regularised_fit_restores_the_best_held_out_checkpoint():
    from lpw.p1.anchor import eval_loss, fit_adapters_reg
    _, am = _models()
    Cfit, Chold = _cache(am, seed=1), _cache(am, seed=2)
    info = fit_adapters_reg(am, Cfit, Chold, Ccor=_cache(am, seed=3), max_steps=40, eval_every=5, batch_rows=16, lr=3e-2,
                            log=lambda *a: None)
    hs = [h for _, _, h in info["trace"]]
    assert info["best_heldout"] == min(hs) <= info["heldout_step0"]
    assert abs(eval_loss(am, Chold) - info["best_heldout"]) < 1e-9
