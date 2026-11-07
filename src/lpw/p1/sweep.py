"""Balance-sweep gate: a frozen-backbone Bernoulli branch on OUT amounts (reports/p3a_prereg.md §15, addendum A).

The generator never empties an account (P1 known defect 1): its amount head is a truncated mixture of Gaussians
over log10 ETH, it cannot see the prospective fee, and no component sits on the moving target b - fee. The gate
adds that atom. On an eligible OUT it draws sweep ~ Bernoulli(p(x)); a sweep sends eth = b - fee exactly, and
otherwise the unchanged MoG draw is kept. Everything upstream stays frozen.

One feature function serves fitting (teacher-forced train rows) and generation, so the two see the same inputs:
the frozen amount-head input amt_x of the row; one-hot role, cohort, count plan, amount plan and
progress_state(plan_n, n_own + 1); and log1p(n_own), log10((b - fee) / D), fee / b, log1p(days to w_end).
"""
import numpy as np
import torch

from lpw.p1.plan import N_PLAN_A, N_PLAN_N, N_PROG, progress_state
from lpw.p1.store import SUPPLY_ETH

D = 1e-4          # the dust floor in ETH; a sweep leaves |balance| <= D
LAM = 10.0        # L2 penalty on beta (intercept unpenalized)


def eligible(b: np.ndarray, fee: np.ndarray) -> np.ndarray:
    """OUT rows on which a sweep is possible: finite, fee >= 0, and D <= b - fee <= SUPPLY."""
    b, fee = np.asarray(b, np.float64), np.asarray(fee, np.float64)
    m = b - fee
    return np.isfinite(b) & np.isfinite(fee) & (fee >= 0) & (m >= D) & (m <= SUPPLY_ETH)


def features(amt_x: torch.Tensor, role, cohort, plan_n, plan_a, n_own, b, fee, t, w_end) -> tuple[torch.Tensor, torch.Tensor]:
    """(continuous (n, dx + 4), one-hot (n, k)) gate inputs for eligible OUT rows. Inputs other than amt_x are
    array-likes of length n; b, fee and t are float64."""
    dev = amt_x.device
    f = lambda v, dt=torch.long: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)  # noqa: E731
    b, fee, t = np.asarray(b, np.float64), np.asarray(fee, np.float64), np.asarray(t, np.float64)
    pn, pa, no = f(plan_n), f(plan_a), f(n_own)
    prog = progress_state(pn, no + 1)
    oh = torch.cat([torch.nn.functional.one_hot(f(role), 2), torch.nn.functional.one_hot(f(cohort), 2),
                    torch.nn.functional.one_hot(pn, N_PLAN_N + 1)[:, :N_PLAN_N], torch.nn.functional.one_hot(pa, N_PLAN_A + 1)[:, :N_PLAN_A],
                    torch.nn.functional.one_hot(prog, N_PROG)], 1).float()
    sc = np.stack([np.log1p(np.asarray(n_own, np.float64)), np.log10((b - fee) / D), fee / b,
                   np.log1p(np.maximum(w_end - t, 0) / 86400.0)], 1)
    cont = torch.cat([amt_x.float(), torch.as_tensor(sc, dtype=torch.float32, device=dev)], 1)
    return cont, oh


class SweepGate(torch.nn.Module):
    def __init__(self, n_cont: int, n_oh: int, mu=None, sd=None):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_cont) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(n_cont) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.zeros(n_cont + n_oh))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def design(self, cont, oh):
        return torch.cat([(cont - self.mu) / self.sd, oh], 1)

    def logit(self, cont, oh):
        return self.design(cont, oh) @ self.beta + self.bias

    def prob(self, cont, oh):
        return torch.sigmoid(self.logit(cont, oh))

    @staticmethod
    def standardizer(cont: torch.Tensor):
        mu, sd = cont.mean(0), cont.std(0)
        sd = torch.where(sd > 0, sd, torch.ones_like(sd))  # constant coordinates get scale 1
        return mu, sd


def fit_gate(cont: torch.Tensor, oh: torch.Tensor, y: torch.Tensor, lam: float = LAM, max_iter: int = 500) -> SweepGate:
    """Summed BCE + lam / 2 ||beta||^2 (intercept unpenalized), deterministic L-BFGS from zero; standardization
    from the fitting rows."""
    mu, sd = SweepGate.standardizer(cont)
    g = SweepGate(cont.shape[1], oh.shape[1], mu, sd).to(cont.device)
    X = g.design(cont, oh).detach()
    yf = y.float()
    opt = torch.optim.LBFGS(g.parameters(), lr=1.0, max_iter=max_iter, tolerance_grad=1e-6, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def objective():
        z = X @ g.beta + g.bias
        return torch.nn.functional.binary_cross_entropy_with_logits(z, yf, reduction="sum") + 0.5 * lam * (g.beta ** 2).sum()

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    loss = objective()
    loss.backward()
    g.fit_info = {"objective": float(loss.detach()), "max_grad": float(max(p.grad.abs().max() for p in g.parameters())),
                  "n_rows": int(len(y)), "n_pos": int(y.sum()), "n_iter": int(opt.state[g.beta].get("n_iter", -1))}
    opt.zero_grad(set_to_none=True)
    assert np.isfinite(g.fit_info["objective"]), "non-finite gate fit"
    return g


def save(g: SweepGate, path) -> None:
    torch.save({"state": g.state_dict(), "n_cont": int(g.mu.numel()), "n_oh": int(g.beta.numel() - g.mu.numel()),
                "fit_info": getattr(g, "fit_info", None)}, path)


def load(path, dev="cuda") -> SweepGate:
    d = torch.load(path, map_location=dev, weights_only=False)
    g = SweepGate(d["n_cont"], d["n_oh"]).to(dev)
    g.load_state_dict(d["state"])
    g.fit_info = d["fit_info"]
    g.eval()
    return g


class Sweeper:
    """Generation-time branch: holds the gate and its own RNG, and records per-row diagnostics."""

    def __init__(self, gate: SweepGate | None, seed: int, w_end: int):
        self.gate, self.rng, self.w_end = gate, np.random.default_rng(seed), w_end
        self.n_out = self.n_eligible = self.n_sweep = 0

    @torch.no_grad()
    def decide(self, amt_x, conds, n_own, b, fee, t) -> np.ndarray:
        """Sweep decision for OUT rows (all arguments indexed by those rows). Draws one uniform per eligible row."""
        n = len(b)
        self.n_out += n
        out = np.zeros(n, bool)
        el = eligible(b, fee)
        self.n_eligible += int(el.sum())
        if not el.any() or self.gate is None:
            return out
        i = np.flatnonzero(el)
        c = [conds[k] for k in i]
        cont, oh = features(amt_x[torch.as_tensor(i, device=amt_x.device)], [x["role"] for x in c], [x["cohort"] for x in c],
                            [x["plan_n"] for x in c], [x["plan_a"] for x in c], np.asarray(n_own)[i], np.asarray(b)[i],
                            np.asarray(fee)[i], np.asarray(t)[i], self.w_end)
        p = self.gate.prob(cont, oh).double().cpu().numpy()
        out[i] = self.rng.random(len(i)) < p
        self.n_sweep += int(out.sum())
        return out
