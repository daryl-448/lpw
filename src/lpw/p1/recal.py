"""Post-training stop recalibration (Phase 1 diagnosis, runs/p1_diag/runaway/).

The probability that an agent makes no further owned event is a tiny share of the training loss, so early stopping
on total validation NLL halts before it converges: LPW under-predicted stopping on its own training prefixes
(2018 normal depth 2: 0.235 vs 0.277 observed) although the encoder carries the information (refitting the heads on
frozen features calibrates it on held-out validation). This stage refits only the time head's dormant-atom output
row (d + 1 parameters) on the training split, everything else frozen, with every training prefix weighted equally.

B2 is skipped: it has no censoring term and is not used for generation.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lpw.p1.batch import Views, exact_specs, pack

DORMANT = 63


@torch.no_grad()
def _cache(model, store, views, dev):
    from lpw.p1.train import loader, to_dev
    keys = ("hk", "ho", "t0", "r", "typ", "tk")
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
    return {k: torch.cat(v) for k, v in C.items()}


def _kind_time_loss(model, C, idx):
    """Per-row kind + time + censoring NLL, the same terms as Model._terms, averaged uniformly over rows."""
    t0, r, typ, tk = C["t0"][idx], C["r"][idx], C["typ"][idx], C["tk"][idx]
    logpk, law, _ = model.kind_time(C["hk"][idx].float(), C["ho"][idx].float(), t0)
    n = len(idx)
    r2 = r.repeat_interleave(2)
    logf = law.log_density(r2).view(n, 2)
    logS = law.log_survival(r2).view(n, 2)
    own, cen = typ == 1, (typ == 2) | (typ == 3)
    l = torch.where(own, -logpk.gather(1, tk[:, None])[:, 0].double() - logf.gather(1, tk[:, None])[:, 0], 0.0)
    l = l + torch.where(cen, -torch.logsumexp(logpk.double() + logS, -1), 0.0)
    return l.mean()


@torch.no_grad()
def _full_loss(model, C, dev, chunk=16384):
    n = len(C["t0"])
    tot = 0.0
    for i in range(0, n, chunk):
        idx = torch.arange(i, min(i + chunk, n), device=dev)
        tot += _kind_time_loss(model, C, idx).item() * len(idx)
    return tot / n


def recalibrate_stop(model, store, views=None, dev="cuda", steps=4000, lr=1e-3, batch_rows=8192, seed=0, log=print) -> dict:
    """Refit the dormant-atom output row of model.time_head on the training split, in place. Returns a summary."""
    if model.cfg.b2:
        return {"skipped": "b2"}
    model.eval()
    model.plan_inference = model.cfg.plan  # plan models generate with s known, so recalibrate with the real (train) s
    views = views or Views(store)
    C = _cache(model, store, views, dev)
    model.plan_inference = False
    last = model.time_head[2]
    params = [last.weight, last.bias]
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    keep = torch.zeros(last.out_features, 1, device=last.weight.device)
    keep[DORMANT] = 1.0
    opt = torch.optim.Adam(params, lr=lr)  # rows other than the dormant one get zero gradient, so Adam leaves them
    before = _full_loss(model, C, dev)
    w0 = (last.weight.detach().clone(), last.bias.detach().clone())
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(C["t0"])
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_rows, n),), device=dev, generator=g)
        loss = _kind_time_loss(model, C, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        last.weight.grad.mul_(keep)
        last.bias.grad.mul_(keep[:, 0])
        opt.step()
    after = _full_loss(model, C, dev)
    moved = int(((last.weight.detach() - w0[0]).abs().sum(1) > 0).sum())
    assert torch.equal(last.weight.detach()[:DORMANT], w0[0][:DORMANT]) and torch.equal(last.bias.detach()[DORMANT + 1:], w0[1][DORMANT + 1:])
    for p, q in req.items():
        p.requires_grad_(q)
    info = {"rows": n, "steps": steps, "train_loss_before": before, "train_loss_after": after, "rows_changed": moved}
    log(f"  stop recalibration: kind+time+cens train NLL {before:.4f} -> {after:.4f} over {n} prefixes")
    return info


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
    info = recalibrate_stop(model, store, dev=dev, log=log)
    torch.save(model.state_dict(), out / "model.pt")
    marker.write_text(json.dumps(info, indent=1))
    return info
