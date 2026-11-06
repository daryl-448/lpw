"""Post-training stop recalibration (Phase 1 diagnosis, runs/p1_diag/runaway/).

The probability that an agent makes no further owned event is a tiny share of the training loss, so early stopping
on total validation NLL halts before it converges: LPW under-predicted stopping on its own training prefixes
(2018 normal depth 2: 0.235 vs 0.277 observed) although the encoder carries the information (refitting the heads on
frozen features calibrates it on held-out validation). This stage refits only the time head's dormant-atom output
row (d + 1 parameters) on the training split, everything else frozen, with every training prefix weighted equally.
Plan-progress models (cfg.progress) then refit the known-state rows of prog_dorm (3 x 2) with that row frozen, with
the real s, bounded by PROG_CLAMP; the UNKNOWN row has no rows here (s is always known) and stays as trained, so the
s-unknown predictions are those of the plain recalibration.

B2 is skipped: it has no censoring term and is not used for generation.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lpw.p1.batch import Views, agent_weights, exact_specs, pack
from lpw.p1.plan import PROG_UNKNOWN

DORMANT = 63
PROG_LR = 1e-2  # the progress residuals sit near 0 after training (AdamW decay, small share of the loss)
PROG_CLAMP = 8.0  # |residual| bound: with the real s the BELOW / DONE optima are -inf / +inf


@torch.no_grad()
def _cache(model, store, views, dev):
    from lpw.p1.train import loader, to_dev
    keys = ("hk", "ho", "t0", "r", "typ", "tk", "w", "pg") + (("px", "pe") if model.cfg.pace else ())
    C = {k: [] for k in keys}
    for b in loader(views, pack(exact_specs(store, store.agents("train")), views)):
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=b["tok_kind"].is_cuda):
            Hk, Ho = model.encode(b)
        rb, rj = b["row_b"], b["row_j"]
        C["hk"].append(Hk[rb, rj].half())
        C["ho"].append(Ho[rb, rj].half())
        C["t0"].append(b["row_t0"])
        C["r"].append(b["row_r"])
        C["typ"].append(b["row_type"].to(torch.int8))
        C["tk"].append(b["row_tkind"].clamp(min=0))
        C["pg"].append(model.row_progress(b) if model.cfg.progress else torch.zeros(len(rb), device=rb.device, dtype=torch.int64))
        C["w"].append(b["agent_w"][rb].double() if "agent_w" in b else torch.ones(len(rb), device=rb.device, dtype=torch.float64))
        if model.cfg.pace:
            x, e = model.pace.features(b)
            C["px"].append(x), C["pe"].append(e)
    return {k: torch.cat(v) for k, v in C.items()}


def _kind_time_loss(model, C, idx):
    """Per-row kind + time + censoring NLL, the same terms as Model._terms, averaged over rows with the training
    row weights (uniform unless the model was trained with agent weights)."""
    t0, r, typ, tk = C["t0"][idx], C["r"][idx], C["typ"][idx], C["tk"][idx]
    ex = {}
    if model.cfg.progress:
        ex["prog"] = C["pg"][idx]
    if model.cfg.pace:
        ex["pace"] = (C["px"][idx], C["pe"][idx])
    logpk, law, _ = model.kind_time(C["hk"][idx].float(), C["ho"][idx].float(), t0, **ex)
    n = len(idx)
    r2 = r.repeat_interleave(2)
    logf = law.log_density(r2).view(n, 2)
    logS = law.log_survival(r2).view(n, 2)
    own, cen = typ == 1, (typ == 2) | (typ == 3)
    l = torch.where(own, -logpk.gather(1, tk[:, None])[:, 0].double() - logf.gather(1, tk[:, None])[:, 0], 0.0)
    l = l + torch.where(cen, -torch.logsumexp(logpk.double() + logS, -1), 0.0)
    return (C["w"][idx] * l).mean()  # unbiased for the weighted objective: the weights average 1 over training rows


@torch.no_grad()
def _full_loss(model, C, dev, chunk=16384):
    n = len(C["t0"])
    tot = 0.0
    for i in range(0, n, chunk):
        idx = torch.arange(i, min(i + chunk, n), device=dev)
        tot += _kind_time_loss(model, C, idx).item() * len(idx)
    return tot / C["w"].sum().item()


def clamp_known_progress(prog_dorm: torch.Tensor) -> None:
    """Bound the known-state progress residuals in place; the UNKNOWN row is never touched."""
    with torch.no_grad():
        prog_dorm[:PROG_UNKNOWN].clamp_(-PROG_CLAMP, PROG_CLAMP)


def _fit(model, C, dev, params, lr, steps, batch_rows, seed, mask=None, after_step=None, log_every=500):
    """Adam on the cached rows; mask(param) zeroes the gradient of the elements that must not move. Returns the
    (step, minibatch loss) trace."""
    opt = torch.optim.Adam(params, lr=lr)  # fresh, no decay: elements with zero gradient stay exactly as they were
    g = torch.Generator(device=dev).manual_seed(seed)
    n, trace = len(C["t0"]), []
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_rows, n),), device=dev, generator=g)
        loss = _kind_time_loss(model, C, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if mask:
            mask()
        opt.step()
        if after_step:
            after_step()
        if step % log_every == 0 or step == steps - 1:
            trace.append((step, loss.item()))
    return trace


def recalibrate_stop(model, store, views=None, dev="cuda", steps=4000, lr=1e-3, batch_rows=8192, seed=0, log=print,
                     agent_alpha: float = 0.0, prog_lr: float = PROG_LR) -> dict:
    """Refit the dormant-atom output row of model.time_head on the training split, in place, with the training row
    weights (agent_alpha, as in train). A progress model then refits its known-state residuals with that row frozen
    (a joint fit could trade the shared row against the residuals and so shift the s-unknown predictions; staged,
    those are exactly those of the first stage). Returns a summary."""
    if model.cfg.b2:
        return {"skipped": "b2"}
    assert not model.cfg.pace, "pace arms refit only the progress residuals (refit_progress); the shared row stays"
    was_inference, req = model.plan_inference, {p: p.requires_grad for p in model.parameters()}
    model.eval()
    try:
        model.plan_inference = model.cfg.plan  # plan models generate with s known, so recalibrate with the real s
        views = views or Views(store, agent_weights(store, agent_alpha) if agent_alpha else None)
        C = _cache(model, store, views, dev)
        model.plan_inference = False
        for p in model.parameters():
            p.requires_grad_(False)
        last = model.time_head[2]
        state0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        before = _full_loss(model, C, dev)
        n = len(C["t0"])
        keep = torch.zeros(last.out_features, 1, device=last.weight.device)
        keep[DORMANT] = 1.0

        def mask_row():
            last.weight.grad.mul_(keep)
            last.bias.grad.mul_(keep[:, 0])

        last.weight.requires_grad_(True)
        last.bias.requires_grad_(True)
        trace = _fit(model, C, dev, [last.weight, last.bias], lr, steps, batch_rows, seed, mask=mask_row)
        last.weight.requires_grad_(False)
        last.bias.requires_grad_(False)
        mid = _full_loss(model, C, dev)
        info = {"rows": n, "steps": steps, "agent_alpha": agent_alpha, "train_loss_before": before, "train_loss_after": mid,
                "trace": trace}
        log(f"  stop recalibration: kind+time+cens train NLL {before:.4f} -> {mid:.4f} over {n} prefixes")
        if model.cfg.progress:
            info |= _fit_progress(model, C, dev, steps, batch_rows, seed + 1, prog_lr, mid, log)
        sd = model.state_dict()
        for k, v in state0.items():  # nothing outside the refit rows moved
            if k == "time_head.2.weight":
                assert torch.equal(torch.cat([sd[k][:DORMANT], sd[k][DORMANT + 1:]]), torch.cat([v[:DORMANT], v[DORMANT + 1:]]))
            elif k == "time_head.2.bias":
                assert torch.equal(torch.cat([sd[k][:DORMANT], sd[k][DORMANT + 1:]]), torch.cat([v[:DORMANT], v[DORMANT + 1:]]))
            elif k == "prog_dorm":
                assert torch.equal(sd[k][PROG_UNKNOWN], v[PROG_UNKNOWN])
            else:
                assert torch.equal(sd[k], v), k
        info["rows_changed"] = int(((sd["time_head.2.weight"] - state0["time_head.2.weight"]).abs().sum(1) > 0).sum())
        return info
    finally:
        model.plan_inference = was_inference
        for p, q in req.items():
            p.requires_grad_(q)


def _fit_progress(model, C, dev, steps, batch_rows, seed, prog_lr, before, log) -> dict:
    """Refit the known-state rows of prog_dorm on the cache, everything else frozen, bounded by PROG_CLAMP."""
    assert not (C["pg"] == PROG_UNKNOWN).any()
    p0 = model.prog_dorm.detach().clone()
    model.prog_dorm.requires_grad_(True)
    known = torch.zeros_like(model.prog_dorm)
    known[:PROG_UNKNOWN] = 1.0
    ptrace = []
    try:
        ptrace_loss = _fit(model, C, dev, [model.prog_dorm], prog_lr, steps, batch_rows, seed,
                           mask=lambda: model.prog_dorm.grad.mul_(known),
                           after_step=lambda: (clamp_known_progress(model.prog_dorm),
                                               ptrace.append(model.prog_dorm.detach()[:PROG_UNKNOWN].clone())))
    finally:
        model.prog_dorm.requires_grad_(False)
    after = _full_loss(model, C, dev)
    pd = model.prog_dorm.detach()
    log(f"  progress residuals [OUT, IN_exo] by state: {np.round(pd.cpu().numpy(), 2).tolist()}; "
        f"train NLL {before:.4f} -> {after:.4f}")
    return {"prog_lr": prog_lr, "prog_clamp": PROG_CLAMP, "prog_train_loss_after": after, "prog_trace": ptrace_loss,
            "prog_dorm_before": p0.tolist(), "prog_dorm_after": pd.tolist(),
            "prog_dorm_at_step": {s: ptrace[s].tolist() for s in (499, 999, 1999, steps - 1) if s < len(ptrace)},
            "prog_at_bound": int((pd[:PROG_UNKNOWN].abs() >= PROG_CLAMP).sum())}


def refit_progress(model, store, dev="cuda", steps=4000, batch_rows=8192, seed=0, prog_lr: float = PROG_LR, log=print) -> dict:
    """Refit only the known-state progress residuals of a progress model on the training split with the real s
    (the second stage of recalibrate_stop, same seed / steps / lr / bound), in place; asserts nothing else moved.
    Used for pace arms, whose shrinkage changes the finite gap law the residuals were fitted against."""
    assert model.cfg.progress
    was_inference, req = model.plan_inference, {p: p.requires_grad for p in model.parameters()}
    model.eval()
    try:
        model.plan_inference = True
        C = _cache(model, store, Views(store), dev)
        model.plan_inference = False
        for p in model.parameters():
            p.requires_grad_(False)
        state0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if model.cfg.pace:
            model.pace.reset_stats()
        before = _full_loss(model, C, dev)
        info = {"rows": len(C["t0"]), "steps": steps, "train_loss_before": before}
        if model.cfg.pace:
            info["eligible_rows"] = int(C["pe"].sum())
        info |= _fit_progress(model, C, dev, steps, batch_rows, seed + 1, prog_lr, before, log)
        if model.cfg.pace:
            model.pace.reset_stats()
            _full_loss(model, C, dev)
            info["pace_stats_train"] = dict(model.pace.stats)
            model.pace.reset_stats()
        sd = model.state_dict()
        for k, v in state0.items():
            if k == "prog_dorm":
                assert torch.equal(sd[k][PROG_UNKNOWN], v[PROG_UNKNOWN])
            else:
                assert torch.equal(sd[k], v), k
        return info
    finally:
        model.plan_inference = was_inference
        for p, q in req.items():
            p.requires_grad_(q)


def ensure_recalibrated(pilot: Path, model_name: str, out: Path, cutoff_months=0, dev="cuda", log=print) -> dict:
    """Recalibrate a trained checkpoint once: the raw weights are kept as model_raw.pt, model.pt is overwritten and
    recal.json records the summary. Uses the same (cutoff) store the model was trained on."""
    from lpw.p1.store import build_store
    from lpw.p1.train import cutoff_ts, load
    marker = out / "recal.json"
    if marker.exists():
        return json.loads(marker.read_text())
    store = build_store(pilot)
    if cutoff_months:
        store = build_store(pilot, cutoff=cutoff_ts(store, cutoff_months))
    model = load(pilot, model_name, out, store, dev=dev)
    if not (out / "model_raw.pt").exists():
        torch.save(model.state_dict(), out / "model_raw.pt")
    log(f"recalibrate {model_name} ({out})")
    tj = out / "train.json"
    alpha = json.loads(tj.read_text()).get("agent_alpha", 0.0) if tj.exists() else 0.0
    info = recalibrate_stop(model, store, dev=dev, log=log, agent_alpha=alpha)
    torch.save(model.state_dict(), out / "model.pt")
    marker.write_text(json.dumps(info, indent=1))
    return info
