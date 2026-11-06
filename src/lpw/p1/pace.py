"""History-free pace reference and mean shrinkage of the time law (Phase 1; Sol pace rounds, drift_results.md).

Diagnosis: on its own generated prefixes LPW forecasts slower next gaps than on the same agent's real prefix (same
real plan, same owned depth), and its forecast follows the history's pace with elasticity ~0.4 (depth 3-9) to ~0.9
(deep), so a small per-step bias compounds in free run: heavy plans fall far short of their counts even with the
real plan. A dormant-logit refit cannot change gap shape.

Fix tested here, without training on generated histories:
- PaceRef: a linear, history-free conditional law fitted on real TRAINING prefixes with the real plan. Inputs are
  only public / plan quantities (no past gaps, amounts, balances, entities, hidden states): role, cohort, the
  standardized public profile, the count and amount plan (one-hot), owned depth, progress state, activation
  exposure, remaining window time and the current calendar month. Outputs use the model's layout: 2 kind logits and,
  per kind, 64 bin + 24 hour + 7 weekday logits, scored through the same normalized TimeLaw, with the owned-event
  density and both censoring terms, uniform prefix weights and an L2 penalty on the mean objective.
- Mean shrinkage: on eligible rows (known plan >= MIN_PLAN, owned depth >= 1, state BELOW) the model's positive-finite
  gap law is tilted, m'_{k,b} ∝ m_{k,b} exp(theta * l_{k,b}), normalized within each kind so that each kind's
  positive-finite mass, the same-second atom and the dormant atom keep their probabilities, with the scalar theta
  solved so that the kind-marginal positive-finite mean log gap equals gamma * mu_model + (1 - gamma) * mu_ref.
  l_{k,b} is the weekly-weighted within-bin mean of ln(gap) (4 geometric sub-intervals per bin). gamma = 1 is an exact
  bypass. theta is found by bisection; its gradient (it depends on the model's law, e.g. on the dormant residuals
  through the kinds' shares of positive-finite gaps) comes from the implicit-function theorem, zero on clipped rows. Every other row (UNKNOWN plan, other states, plans < MIN_PLAN, depth 0) takes a hard bypass: bit-identical.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lpw.p1.plan import N_PLAN_A, N_PLAN_N, PROG_BELOW, progress_state
from lpw.p1.timelaw import EDGES, YEAR, TimeLaw

MIN_PLAN = 5      # count bucket index 5 = [10, 30): pace shrinkage only for plans of >= 10 owned events
N_MONTH = 12
THETA_MAX = 8.0   # bound on the tilt (per nat of log gap); rows whose target lies outside are clipped and counted
K_SUB = 4         # geometric sub-intervals per finite bin for the weekly-weighted log moments
_SUB = np.geomspace(1.0, float(YEAR), 62 * K_SUB + 1)
_A, _B = _SUB[:-1], _SUB[1:]
_LM = (_B * np.log(_B) - _A * np.log(_A)) / (_B - _A) - 1.0  # mean of ln x, uniform in seconds, per sub-interval


def n_features(n_prof: int) -> int:
    return 2 + n_prof + N_PLAN_N + N_PLAN_A + 1 + 3 + 3 + N_MONTH


class PaceRef(nn.Module):
    def __init__(self, n_prof: int):
        super().__init__()
        f = n_features(n_prof)
        self.kind = nn.Linear(f, 2, dtype=torch.float64)  # float64: the fit is run to a tight gradient tolerance
        self.time = nn.Linear(f, 2 * 95, dtype=torch.float64)
        for lin in (self.kind, self.time):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("gamma", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("w_start", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("w_end", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("month_edges", torch.zeros(N_MONTH + 1, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))
        self.reset_stats()

    def set_window(self, w_start: int, w_end: int):
        m0 = np.array([w_start], dtype="datetime64[s]").astype("datetime64[M]")[0]
        edges = (m0 + np.arange(N_MONTH + 1)).astype("datetime64[s]").astype(np.int64).astype(np.float64)
        self.w_start.fill_(float(w_start)), self.w_end.fill_(float(w_end))
        self.month_edges.copy_(torch.as_tensor(edges))

    def reset_stats(self):
        self.stats = {"rows": 0, "eligible": 0, "clipped": 0, "invalid": 0, "abs_theta_sum": 0.0, "shift_sum": 0.0}

    # ---- features ------------------------------------------------------------------------------------------
    def features(self, B) -> tuple[torch.Tensor, torch.Tensor]:
        """(x float64 (N, F), eligible bool (N,)) for the rows of batch B. Uses the effective plan from encode()
        (B["_plan_n_eff"], B["_plan_a_eff"]); UNKNOWN plans give zero plan / state one-hots and are never eligible."""
        assert float(self.w_end) > 0, "PaceRef window not set"
        rb, t0, n_own = B["row_b"], B["row_t0"].double(), B["row_nown"]
        pn, pa = B["_plan_n_eff"][rb], B["_plan_a_eff"][rb]
        known = pn < N_PLAN_N
        prog = progress_state(pn, n_own)
        oh = lambda v, k: torch.nn.functional.one_hot(v.clamp(0, k), k + 1)[:, :k].double()
        u = torch.log1p((self.w_end - B["origin"][rb].double()).clamp(min=0) / 86400.0) / math.log(366.0)
        rem = torch.log1p((self.w_end - t0).clamp(min=0) / 86400.0) / math.log(366.0)
        dep = torch.log1p(n_own.double()) / math.log(1001.0)
        month = torch.bucketize(t0, self.month_edges[1:N_MONTH], right=True)
        st = torch.where(known, prog, torch.full_like(prog, 3))
        x = torch.cat([B["role"][rb].double()[:, None], B["cohort"][rb].double()[:, None], B["prof"][rb].double(),
                       oh(torch.where(known, pn, N_PLAN_N), N_PLAN_N), oh(torch.where(known, pa, N_PLAN_A), N_PLAN_A),
                       dep[:, None], oh(st, 3), torch.stack([u, u * u, rem], 1), oh(month, N_MONTH)], 1)
        elig = known & (prog == PROG_BELOW) & (pn >= MIN_PLAN) & (n_own >= 1)
        return x, elig

    # ---- reference law -------------------------------------------------------------------------------------
    def law(self, x, t0) -> tuple[torch.Tensor, TimeLaw]:
        x = x.double()
        logpk = torch.log_softmax(self.kind(x), -1)
        out = self.time(x).reshape(2 * len(x), 95)
        lp, lh, ld = torch.log_softmax(out[:, :64], -1), torch.log_softmax(out[:, 64:88], -1), torch.log_softmax(out[:, 88:95], -1)
        return logpk, TimeLaw(lp, lh, ld, t0.repeat_interleave(2))

    # ---- shrinkage -----------------------------------------------------------------------------------------
    def adapt(self, lp, lh, ld, t0, logpk, pace) -> torch.Tensor:
        """lp, lh, ld: the model's (2N, ·) log-probabilities (rows 2i + kind); returns float64 log_pdt for TimeLaw,
        tilted on eligible rows, bit-identical (lp.double()) elsewhere."""
        x, elig = pace
        lp64 = lp.double()
        n = len(t0)
        self.stats["rows"] += n
        g = float(self.gamma)
        if g == 1.0 or not bool(elig.any()):
            return lp64
        assert bool(self.fitted), "PaceRef not fitted"
        idx = torch.nonzero(elig)[:, 0]
        r2 = torch.stack([2 * idx, 2 * idx + 1], 1).reshape(-1)
        t0e = t0[idx]
        base = TimeLaw(lp64[r2], lh[r2], ld[r2], t0e.repeat_interleave(2))
        ell = bin_log_moments(base)                                       # (2n, 62)
        logm = torch.log(base.mass.clamp(min=1e-300))
        Mb, Zb = base.mass.sum(-1), base.Z
        pk = logpk[idx].double()
        with torch.no_grad():
            lpr, ref = self.law(x[idx], t0e)
            mu_r, _ = _pos_mean(torch.log(ref.mass.clamp(min=1e-300)), bin_log_moments(ref), ref.mass.sum(-1), ref.Z, lpr.double())
            mu_h0, pi0 = _pos_mean(logm, ell, Mb, Zb, pk)
            target0 = g * mu_h0 + (1.0 - g) * mu_r
            valid = torch.isfinite(target0) & torch.isfinite(mu_h0) & torch.isfinite(pi0).all(-1)
            target0 = torch.where(valid, target0, torch.zeros_like(target0))
            theta0, clipped = _solve_theta(logm, ell, torch.where(valid[:, None], pi0, 0.5), target0)
            valid = valid & torch.isfinite(theta0)
            theta0 = torch.where(valid, theta0, torch.zeros_like(theta0))  # invalid rows: no tilt
            interior = valid & ~clipped
            # dF/dtheta at the root, F = tilted mean - target: sum_k pi_k Var_tilted_k(ell)
            q = torch.softmax(logm + theta0.repeat_interleave(2)[:, None] * ell, -1)
            var = (q * ell * ell).sum(-1) - (q * ell).sum(-1) ** 2
            f_th = (pi0 * var.view(-1, 2)).sum(-1)
            self.stats["eligible"] += len(idx)
            self.stats["clipped"] += int((valid & clipped).sum())
            self.stats["invalid"] += int((~valid).sum())
            self.stats["abs_theta_sum"] += float(theta0.abs().sum())
            self.stats["shift_sum"] += float((target0 - mu_h0).sum())
        theta = theta0
        sel = interior & (f_th > 1e-12)
        if torch.is_grad_enabled() and logm.requires_grad and bool(sel.any()):
            # implicit-function gradient of the root (theta depends on the model through mu_h and the kinds' shares
            # pi of positive-finite gaps): value theta0, derivative -dF/dphi / dF/dtheta. Computed on interior rows
            # only, so clipped / invalid rows get a zero derivative and cannot leak NaN gradients.
            j = torch.nonzero(sel)[:, 0]
            j2 = torch.stack([2 * j, 2 * j + 1], 1).reshape(-1)
            mu_h, pi = _pos_mean(logm[j2], ell[j2], Mb[j2], Zb[j2], pk[j])
            F = _tilted_mean(logm[j2], ell[j2], pi, theta0[j]) - (g * mu_h + (1.0 - g) * mu_r[j])
            theta = theta0.clone()
            theta[j] = theta0[j] - (F - F.detach()) / f_th[j]
        th = theta.repeat_interleave(2)[:, None]
        logM = torch.logsumexp(logm, -1, keepdim=True)
        c = logM - torch.logsumexp(logm + th * ell, -1, keepdim=True)     # keeps each kind's positive-finite mass
        out = lp64.clone()
        out[r2, 1:63] = lp64[r2, 1:63] + th * ell + c
        return out


def bin_log_moments(law: TimeLaw) -> torch.Tensor:
    """(n, 62): weekly-weighted mean of ln(gap seconds) within each positive finite bin, at the law's origin."""
    sub = torch.as_tensor(_SUB, device=law.dev, dtype=torch.float64)
    W = law.W(law.r0[:, None] + sub[None])
    wm = (W[:, 1:] - W[:, :-1]).clamp(min=0).view(-1, 62, K_SUB)
    lm = torch.as_tensor(_LM, device=law.dev, dtype=torch.float64).view(62, K_SUB)
    tot = wm.sum(-1)
    return torch.where(tot > 0, (wm * lm).sum(-1) / tot.clamp(min=1e-300), lm.mean(-1).expand_as(tot))


def _pos_mean(logm, ell, mass_sum, Z, logpk):
    """kind-marginal mean ln(gap) of the positive-finite part, and the kind weights pi (n, 2) of that part.
    mass_sum, Z: (2n,) positive-finite mass and normalizer of each kind's law."""
    n = logpk.shape[0]
    q = torch.softmax(logm, -1)
    mu_k = (q * ell).sum(-1).view(n, 2)
    M = mass_sum.view(n, 2) / Z.view(n, 2)
    w = logpk.exp() * M
    pi = w / w.sum(-1, keepdim=True).clamp(min=1e-300)
    return (pi * mu_k).sum(-1), pi


def _tilted_mean(logm, ell, pi, th):
    """sum_k pi_k E_k[ell] under the tilt m exp(th ell), each kind normalized separately."""
    q = torch.softmax(logm + th.repeat_interleave(2)[:, None] * ell, -1)
    return (pi * (q * ell).sum(-1).view(-1, 2)).sum(-1)


def _solve_theta(logm, ell, pi, target, iters=60):
    """bisection for theta in [-THETA_MAX, THETA_MAX] with sum_k pi_k E_tilted_k[ell] = target (monotone in theta)."""
    n = pi.shape[0]
    mean = lambda th: _tilted_mean(logm, ell, pi, th)

    lo = torch.full((n,), -THETA_MAX, dtype=torch.float64, device=pi.device)
    hi = -lo
    f_lo, f_hi = mean(lo), mean(hi)
    clipped = (target < f_lo) | (target > f_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = mean(mid) < target
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    th = 0.5 * (lo + hi)
    th = torch.where(target < f_lo, -THETA_MAX, torch.where(target > f_hi, THETA_MAX, th))
    return th, clipped


# ---- fitting ---------------------------------------------------------------------------------------------------
def row_nll(logpk, law, r, typ, tk) -> torch.Tensor:
    """per-row kind + time + censoring NLL (Model._terms): owned rows (typ 1) score kind and density, censored rows
    (typ 2 / 3) the kind-mixture survival."""
    n = len(r)
    r2 = r.repeat_interleave(2)
    logf = law.log_density(r2).view(n, 2)
    logS = law.log_survival(r2).view(n, 2)
    own, cen = typ == 1, (typ == 2) | (typ == 3)
    l = torch.where(own, -logpk.gather(1, tk[:, None])[:, 0].double() - logf.gather(1, tk[:, None])[:, 0], 0.0)
    return l + torch.where(cen, -torch.logsumexp(logpk.double() + logS, -1), 0.0)


@torch.no_grad()
def pace_rows(model, store, split, dev="cuda") -> dict:
    """Pace features and targets for every exact prefix of the split, with the real plan (no encoder needed)."""
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.train import loader, to_dev
    views = Views(store)
    C = {k: [] for k in ("x", "elig", "t0", "r", "typ", "tk", "role")}
    for b in loader(views, pack(exact_specs(store, store.agents(split)), views)):
        b = to_dev(b, dev)
        b["_plan_n_eff"], b["_plan_a_eff"] = b["plan_n"], b["plan_a"]
        x, e = model.pace.features(b)
        C["x"].append(x), C["elig"].append(e), C["t0"].append(b["row_t0"]), C["r"].append(b["row_r"])
        C["typ"].append(b["row_type"]), C["tk"].append(b["row_tkind"].clamp(min=0)), C["role"].append(b["role"][b["row_b"]])
    return {k: torch.cat(v) for k, v in C.items()}


@torch.no_grad()
def _init_biases(ref, C):
    """start the reference from the training marginals (kind, and per kind the bin / hour / weekday of observed owned
    events; the dormant bin from end-censored rows) so L-BFGS starts near the intercept-only optimum."""
    from lpw.p1.timelaw import dt_bin, hour_dow
    typ, tk = C["typ"].cpu().numpy(), C["tk"].cpu().numpy()
    own = typ == 1
    t = (C["t0"] + C["r"]).cpu().numpy()[own]
    b, (h, d) = dt_bin(C["r"].cpu().numpy()[own]), hour_dow(t)
    k = tk[own]
    ref.kind.bias.copy_(torch.as_tensor(np.log(np.bincount(k, minlength=2) + 1.0)))
    bias = np.zeros((2, 95))
    for kk in (0, 1):
        s = k == kk
        cb = np.bincount(b[s], minlength=64).astype(np.float64)
        cb[63] = (typ == 3).sum() * s.mean() if own.any() else 1.0
        bias[kk, :64] = np.log(cb + 1.0)
        bias[kk, 64:88] = np.log(np.bincount(h[s], minlength=24) + 1.0)
        bias[kk, 88:95] = np.log(np.bincount(d[s], minlength=7) + 1.0)
    ref.time.bias.copy_(torch.as_tensor(bias.reshape(-1), dtype=ref.time.bias.dtype))


def fit_reference(model, store, dev="cuda", l2=1e-3, chunk=16384, outer=100, inner=25, gtol=1e-6, log=print) -> dict:
    """Fit model.pace (kind + time heads, float64) by full-batch L-BFGS on the training prefixes, everything else
    untouched. Converged = max |gradient| of the mean objective <= gtol (float64 sums make 1e-6 attainable); a round
    without any objective change restarts L-BFGS (at most 3 times). Raises if it has not converged."""
    ref = model.pace
    ref.set_window(store.w_start, store.w_end)
    C = pace_rows(model, store, "train", dev)
    n = len(C["t0"])
    _init_biases(ref, C)
    params = [ref.kind.weight, ref.kind.bias, ref.time.weight, ref.time.bias]
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    mk_opt = lambda: torch.optim.LBFGS(params, lr=1.0, max_iter=inner, history_size=20, line_search_fn="strong_wolfe",
                                       tolerance_grad=1e-12, tolerance_change=1e-16)
    opt = mk_opt()

    def objective(backward=True):
        if backward:
            opt.zero_grad(set_to_none=False)
        tot = 0.0
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            logpk, law = ref.law(C["x"][sl], C["t0"][sl])
            l = row_nll(logpk, law, C["r"][sl], C["typ"][sl], C["tk"][sl]).sum() / n
            if backward:
                l.backward()
            tot += l.item()
        pen = l2 * (ref.kind.weight.square().sum() + ref.time.weight.square().sum())
        if backward:
            pen.backward()
        return tot + pen.item()

    trace, prev, converged, restarts = [], None, False, 0
    try:
        start = objective(False)
        for it in range(outer):
            opt.step(objective)
            f = objective(True)  # objective and gradient at the current point
            gmax = max(float(p.grad.abs().max()) for p in params)
            trace.append((it, f, gmax))
            log(f"  pace reference L-BFGS round {it}: objective {f:.10f}, max |grad| {gmax:.2e}")
            if gmax <= gtol:
                converged = True
                break
            if prev is not None and f == prev:
                if restarts == 3:
                    break
                restarts += 1
                opt = mk_opt()
                log(f"  pace reference: no progress, L-BFGS restart {restarts}")
            prev = f
    finally:
        for p, q in req.items():
            p.requires_grad_(q)
    if not converged:
        raise RuntimeError(f"pace reference did not converge: {trace[-3:]}")
    ref.fitted.fill_(True)
    with torch.no_grad():
        nll = sum(row_nll(*ref.law(C["x"][i:i + chunk], C["t0"][i:i + chunk]), C["r"][i:i + chunk], C["typ"][i:i + chunk],
                          C["tk"][i:i + chunk]).sum().item() for i in range(0, n, chunk)) / n
    return {"rows": n, "eligible_rows": int(C["elig"].sum()), "l2": l2, "gtol": gtol, "objective_start": start,
            "objective": trace[-1][1], "max_grad": trace[-1][2], "restarts": restarts, "train_nll": nll, "rounds": len(trace),
            "trace": trace}


@torch.no_grad()
def known_plan_nll(model, store, split="val", dev="cuda") -> dict:
    """Full conditional NLL per prefix with the real plan (oracle-conditioned; not the reportable s-unknown NLL),
    overall and per role, plus the kind + time + censoring part."""
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.train import loader, to_dev
    views = Views(store)
    was = model.plan_inference
    model.eval()
    model.plan_inference = True
    acc = {r: [0.0, 0.0, 0] for r in ("all", "normal", "scam")}
    try:
        for b in loader(views, pack(exact_specs(store, store.agents(split)), views)):
            b = to_dev(b, dev)
            t = model.terms(b)
            kt = (t["kind"] + t["time"] + t["cens"]).double()
            full = kt.index_add(0, t["o_row"], (t["cp"] + t["amt"]).double())
            role = b["role"][b["row_b"]]
            for nm, m in (("all", torch.ones_like(role, dtype=torch.bool)), ("normal", role == 0), ("scam", role == 1)):
                acc[nm][0] += float(full[m].sum()); acc[nm][1] += float(kt[m].sum()); acc[nm][2] += int(m.sum())
    finally:
        model.plan_inference = was
    return {nm: {"full": a[0] / a[2], "kind_time_cens": a[1] / a[2], "rows": a[2]} for nm, a in acc.items()}


@torch.no_grad()
def unknown_terms(model, store, split="val", dev="cuda", max_batches=None) -> dict:
    """per-row s-unknown terms on the split (all batches unless max_batches; bit-identity checks):
    {"terms": concatenated kind / time / cens / cp / amt terms, "rows": prefixes covered, "batches", "complete"}."""
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.train import loader, to_dev
    views = Views(store)
    model.eval()
    was = model.plan_inference
    model.plan_inference = False
    out, rows, nb = [], 0, 0
    batches = pack(exact_specs(store, store.agents(split)), views)
    try:
        for i, b in enumerate(loader(views, batches)):
            if max_batches is not None and i >= max_batches:
                break
            t = model.terms(to_dev(b, dev))
            out += [t["kind"], t["time"], t["cens"], t["cp"].double(), t["amt"].double()]
            rows += len(t["kind"]); nb += 1
    finally:
        model.plan_inference = was
    return {"terms": torch.cat(out), "rows": rows, "batches": nb, "complete": nb == len(batches)}


def main():
    """Build the pace arms from a recalibrated lpw_plan_prog checkpoint: fit the reference once, then for each
    gamma refit the known-state progress residuals (recal.refit_progress) with the shrinkage active."""
    import argparse
    from lpw.p1.model import CONFIGS, Model
    from lpw.p1.recal import refit_progress
    from lpw.p1.store import build_store
    from lpw.p1.train import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/pilot_2018")
    ap.add_argument("--base", required=True, help="recalibrated lpw_plan_prog checkpoint dir")
    ap.add_argument("--root", required=True, help="arms are written to ROOT/lpw_pace_g{gamma}")
    ap.add_argument("--gammas", default="1,0.75,0.5")
    ap.add_argument("--ref-only", action="store_true", help="fit and cache the reference, build no arm")
    a = ap.parse_args()
    pilot, base_dir, root = Path(a.pilot), Path(a.base), Path(a.root)
    dev = "cuda"
    st = build_store(pilot)
    base = load(pilot, "lpw_plan_prog", base_dir, st)
    sd = torch.load(base_dir / "model.pt", map_location=dev)
    name = "lpw_plan_prog_pace"
    model = Model(CONFIGS[name], st.n_venues, st.prof_vec(0).shape[0], torch.as_tensor(st.venue_sc)).to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("pace.") for k in missing), (missing, unexpected)
    model.eval()
    ref_path = root / "pace_ref.pt"
    if ref_path.exists():
        model.pace.load_state_dict(torch.load(ref_path, map_location=dev))
        fit = json.loads((root / "pace_ref.json").read_text())
    else:
        fit = fit_reference(model, st, dev)
        torch.save(model.pace.state_dict(), ref_path)
        (root / "pace_ref.json").write_text(json.dumps(fit, indent=1))
    print(f"reference: train NLL {fit['train_nll']:.4f} (kind + time + cens), {fit['rounds']} L-BFGS rounds")
    if a.ref_only:
        return
    u0 = unknown_terms(base, st)
    assert u0["complete"], "s-unknown check must cover the whole val split"
    kn_base = known_plan_nll(base, st)
    for g in (float(x) for x in a.gammas.split(",")):
        out = root / f"lpw_pace_g{g:.2f}"
        out.mkdir(parents=True, exist_ok=True)
        m = Model(CONFIGS[name], st.n_venues, st.prof_vec(0).shape[0], torch.as_tensor(st.venue_sc)).to(dev)
        m.load_state_dict(model.state_dict())
        m.pace.gamma.fill_(g)
        m.eval()
        info = refit_progress(m, st, dev=dev)
        u1 = unknown_terms(m, st)
        assert u1["complete"] and u1["rows"] == u0["rows"] and torch.equal(u0["terms"], u1["terms"]), "s-unknown terms changed"
        kn = known_plan_nll(m, st)
        torch.save(m.state_dict(), out / "model.pt")
        shutil.copy(base_dir / "train.json", out / "train.json")
        rec = {"base": str(base_dir), "gamma": g, "reference": {k: v for k, v in fit.items() if k != "trace"}, "refit_progress": info,
               "unknown_bit_identical": True, "unknown_check": {"split": "val", "prefixes": u0["rows"], "batches": u0["batches"],
                                                         "terms": int(u0["terms"].numel()), "complete": True}, "known_plan_nll_base": kn_base, "known_plan_nll": kn,
               "known_plan_nll_delta": {r: kn[r]["full"] - kn_base[r]["full"] for r in kn}}
        (out / "recal.json").write_text(json.dumps({"note": "staged pace arm; see pace.json"}, indent=1))
        (out / "pace.json").write_text(json.dumps(rec, indent=1))
        print(f"gamma {g}: known-plan val NLL delta {rec['known_plan_nll_delta']}, residuals {info['prog_dorm_after']}")


if __name__ == "__main__":
    main()
