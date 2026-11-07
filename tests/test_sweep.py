"""Sweep gate (P3a addendum A): eligibility, fitting, and generation invariants on the frozen kct checkpoint."""
from pathlib import Path

import numpy as np
import pytest
import torch

from lpw.p1.sweep import D, SweepGate, Sweeper, eligible, fit_gate

KCT = Path("runs/p1_shallow/2018/kct/model.pt")
POP = Path("runs/p3a/2018/population/population.json")


def test_eligibility_needs_a_balance_above_fee_plus_dust():
    b = np.array([1.0, 1.0, 1e-4, np.nan, 1.0, 1.0])
    fee = np.array([0.001, 0.99995, 0.0, 0.001, -0.1, 0.0])
    assert eligible(b, fee).tolist() == [True, False, True, False, False, True]


def test_fit_gate_recovers_a_logistic_signal_and_standardizes_on_fit_rows():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(4000, 3)) * 5 + 2, dtype=torch.float32)
    oh = torch.zeros(4000, 2); oh[:, 0] = 1
    p = torch.sigmoid(1.5 * (x[:, 0] - 2) / 5 - 1)
    y = torch.as_tensor(rng.random(4000) < p.numpy())
    g = fit_gate(x, oh, y, lam=1.0)
    assert torch.allclose(g.mu, x.mean(0)) and abs(float(g.beta[0]) - 1.5) < 0.25 and abs(float(g.beta[1])) < 0.2


class _Const(SweepGate):
    def __init__(self, p):
        super().__init__(1, 1)
        self.p = p

    def prob(self, cont, oh):
        return torch.full((cont.shape[0],), self.p, device=cont.device)


@pytest.fixture(scope="module")
def world():
    if not (KCT.exists() and POP.exists() and torch.cuda.is_available()):
        pytest.skip("needs the frozen kct checkpoint, the P3a population and CUDA")
    from lpw.p1.store import build_store
    from lpw.p1.train import load
    from lpw.p3.population import Population
    st = build_store(Path("data/pilot_2018"))
    model = load(Path("data/pilot_2018"), "lpw_plan_prog_shallow_kct", KCT.parent, st)
    model.plan_inference = True
    conds, _ = Population(st, POP.parent).sample(60, 99)
    return st, model, conds


def _gen(world, sweeper):
    from lpw.p1.generate import generate
    st, model, conds = world
    (_, ev), info = generate(model, conds, st, seed=5, max_events=400, log=lambda *x: None, sweeper=sweeper)
    return ev, info


def test_disabled_gate_is_bit_identical_to_legacy_generation(world):
    ev0, _ = _gen(world, None)
    ev1, info = _gen(world, Sweeper(None, 3000, world[0].w_end))
    assert ev0.equals(ev1) and info["sweep"]["n_sweep"] == 0 and info["sweep"]["n_eligible"] > 0


def test_forced_sweep_empties_the_account_exactly_and_keeps_the_stream_going(world):
    from lpw.p1.store import OUT
    sw = Sweeper(_Const(1.0).cuda(), 3000, world[0].w_end)
    ev, info = _gen(world, sw)
    assert info["sweep"]["n_sweep"] == info["sweep"]["n_eligible"] > 0
    o = ev.filter(ev["kind"] == OUT)
    after = (o["bal"] - o["eth"] - o["fee"]).to_numpy()
    el = eligible(o["bal"].to_numpy(), o["fee"].to_numpy())
    assert (np.abs(after[el]) <= 1e-9).all()  # exact up to float64 rounding
    assert (o["amt"].to_numpy() >= np.log10(D) - 1e-6).all()
    # a sweep does not end the stream: some agent has an event after an eligible (hence swept) OUT
    import polars as pl
    first = o.filter(pl.Series(el)).group_by("agent").agg(pl.col("ts").min().alias("t_sw"))
    later = ev.join(first, on="agent").filter(pl.col("ts") > pl.col("t_sw"))
    assert later.height > 0


def test_sweep_stats_count_sweep_events_not_agents():
    # Sol r1 S1: one agent sweeps twice; only the second sweep is its last OUT -> 0.5; a later IN after the first sweep
    import polars as pl
    from lpw.p3.sweep_stats import sweep_stats
    ag = pl.DataFrame({"agent": [0], "role": [0], "b0": [1.0]})
    ev = pl.DataFrame({"agent": [0, 0, 0, 0], "kind": [0, 1, 0, 1], "ts": [1, 2, 3, 4],
                       "eth": [0.99, 2.0, 1.99, 0.5], "fee": [0.01, 0.0, 0.01, 0.0]},
                      schema={"agent": pl.Int64, "kind": pl.Int8, "ts": pl.Int64, "eth": pl.Float64, "fee": pl.Float64})
    s = sweep_stats(ag, ev)["normal"]
    assert s["sweep_event_share"] == 1.0 and s["ever_sweep"] == 1.0 and s["sweep_is_last_out"] == 0.5
    assert s["sweeper_later_in"] == 1.0 and s["terminal_zero"] == 0.0
