"""Attachment kernels (reports/p3b_prereg.md §P5): the empirical-rate / uniform control (P3b-0) and the one learned
state-only gate + pointer (P3b-1). Both are fitted on real-train replay opportunities only (lpw.p3b.replay).

Gate (on first touches with C_e nonempty; label = target in C_e): ridge logistic on
  [z(8 continuous), role, born, out, role * z(8), role * born, role * out] + an unpenalized intercept,
  summed BCE + LAM / 2 ||beta||^2.
Pointer (on gate positives, over all of C_e): conditional logit on
  [z(log_members, log_events, log_recency_days), same_dir, mint_out] and their role interactions, no intercept,
  summed choice NLL + LAM / 2 ||theta||^2.
z = standardization by fitting-row arithmetic mean and population SD (ddof 0), zero SD -> 1. The pointer's continuous
coordinates are standardized over every alternative of every fitting choice set, each alternative weighted once.
Deterministic L-BFGS from zero on CPU float64 (lr 1, history 50, strong Wolfe, max 500 iterations, gradient tolerance
1e-6, change tolerance 1e-12).
"""
import numpy as np
import torch

LAM = 10.0
MAX_ITER = 500


def _std(X: np.ndarray, w: np.ndarray | None = None):
    mu = X.mean(0) if w is None else (X * w[:, None]).sum(0) / w.sum()
    sd = X.std(0) if w is None else np.sqrt(((X - mu) ** 2 * w[:, None]).sum(0) / w.sum())
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def gate_design(gc: np.ndarray, gb: np.ndarray, mu, sd) -> np.ndarray:
    z = (np.atleast_2d(gc) - mu) / sd
    gb = np.atleast_2d(gb)
    r = gb[:, :1]
    return np.hstack([z, gb, r * z, r * gb[:, 1:3]])


def ptr_design(f: np.ndarray, role: float, mu, sd) -> np.ndarray:
    z = np.hstack([(f[:, :3] - mu) / sd, f[:, 3:5]])
    return np.hstack([z, role * z])


def _lbfgs(params, objective):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=MAX_ITER, tolerance_grad=1e-6, tolerance_change=1e-12, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    loss = objective()
    loss.backward()
    info = {"objective": float(loss.detach()), "max_grad": float(max(p.grad.abs().max() for p in params)),
            "n_iter": int(opt.state[params[0]].get("n_iter", -1))}
    assert np.isfinite(info["objective"]), "non-finite fit"
    return info


def fit_gate(gc: np.ndarray, gb: np.ndarray, y: np.ndarray) -> dict:
    mu, sd = _std(gc)
    X = torch.as_tensor(gate_design(gc, gb, mu, sd), dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    info = _lbfgs([beta, bias], lambda: torch.nn.functional.binary_cross_entropy_with_logits(X @ beta + bias, yt, reduction="sum")
                  + 0.5 * LAM * (beta ** 2).sum())
    return {"mu": mu, "sd": sd, "beta": beta.detach().numpy(), "bias": float(bias.detach()), "fit_info": {**info, "n_rows": int(len(y)),
            "n_pos": int(y.sum())}}


def gate_prob(g: dict, gc, gb) -> np.ndarray:
    z = gate_design(gc, gb, g["mu"], g["sd"]) @ g["beta"] + g["bias"]
    return 1.0 / (1.0 + np.exp(-z))


def fit_pointer(sets: list[tuple[np.ndarray, int, float]]) -> dict:
    """sets: (features (n_c, 5), target position, role) per gate positive."""
    F = np.concatenate([f for f, _, _ in sets]).astype(np.float64)
    mu, sd = _std(F[:, :3])
    n = np.array([len(f) for f, _, _ in sets])
    seg = np.repeat(np.arange(len(sets)), n)
    role = np.repeat(np.array([r for _, _, r in sets], np.float64), n)
    X = np.hstack([(F[:, :3] - mu) / sd, F[:, 3:5]])
    X = torch.as_tensor(np.hstack([X, role[:, None] * X]), dtype=torch.float64)
    del F
    tgt = torch.as_tensor(np.concatenate([[0], np.cumsum(n)[:-1]]) + np.array([t for _, t, _ in sets]))
    segt = torch.as_tensor(seg)
    S = len(sets)
    theta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)

    def objective():
        s = X @ theta
        mx = torch.full((S,), -torch.inf, dtype=torch.float64).scatter_reduce(0, segt, s.detach(), "amax")
        lse = torch.log(torch.zeros(S, dtype=torch.float64).index_add(0, segt, torch.exp(s - mx[segt]))) + mx
        return (lse - s[tgt]).sum() + 0.5 * LAM * (theta ** 2).sum()

    info = _lbfgs([theta], objective)
    return {"mu": mu, "sd": sd, "theta": theta.detach().numpy(), "fit_info": {**info, "n_sets": S, "n_alternatives": int(n.sum())}}


def ptr_logp(p: dict, f: np.ndarray, role: float) -> np.ndarray:
    s = ptr_design(f.astype(np.float64), role, p["mu"], p["sd"]) @ p["theta"]
    s = s - s.max()
    return s - np.log(np.exp(s).sum())


def control_kernel(p0: dict):
    """P3b-0: attach with p0[role] when C_e is nonempty; uniform pointer over C_e in creation order."""
    def k(role, gc, gb, cand, nc, ug, up):
        return min(int(up * nc), nc - 1) if ug < p0[role] else -1
    return k


def learned_kernel(gate: dict, ptr: dict):
    """P3b-1: Bernoulli(gate prob) with the gate uniform; inverse-CDF draw from the pointer softmax with the pointer
    uniform (candidates in creation order)."""
    def k(role, gc, gb, cand, nc, ug, up):
        if ug >= gate_prob(gate, gc, gb)[0]:
            return -1
        _, f = cand()
        c = np.cumsum(np.exp(ptr_logp(ptr, f, float(role))))
        return int(min(np.searchsorted(c, up * c[-1], side="right"), nc - 1))
    return k
