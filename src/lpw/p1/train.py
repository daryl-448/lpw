"""Train one Phase 1 model on a slice, early-stopping on validation NLL (SPEC §6.4).

Usage: uv run python -m lpw.p1.train --pilot data/pilot --model lpw --out runs/p1/2021/lpw [--cutoff-months 2]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lpw.p1.batch import Views, agent_weights, collate_specs, pack, train_specs, val_specs
from lpw.p1.model import CONFIGS, Model
from lpw.p1.store import build_store


class BatchDS(Dataset):
    def __init__(self, views, batches):
        self.views, self.batches = views, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return collate_specs(self.views, self.batches[i])


def loader(views, batches, workers=4):
    return DataLoader(BatchDS(views, batches), batch_size=None, shuffle=False, num_workers=workers, pin_memory=True,
                      prefetch_factor=4 if workers else None)


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def val_loss(model, views, batches, dev) -> tuple[float, float]:
    """(selection loss, plain per-prefix NLL). They differ only with agent weights in views: the selection loss is
    then the agent-weighted mean, the same objective as training."""
    model.eval()
    tot, n, tot_w, n_w = 0.0, 0, 0.0, 0.0
    for b in loader(views, batches):
        b = to_dev(b, dev)
        loss, t = model.loss(b)
        if model.cfg.b2:
            w = len(b["o_row"])
            tot += loss.item() * w
            n += w
            continue
        rows = len(b["row_b"])
        tot += (t["kind"].sum() + t["time"].sum() + t["cens"].sum() + t["cp"].sum() + t["amt"].sum()).item()
        n += rows
        tot_w += loss.item() * rows
        n_w += b["agent_w"][b["row_b"]].sum().item() if "agent_w" in b else rows
    plain = tot / max(n, 1)
    return (plain if model.cfg.b2 else tot_w / max(n_w, 1e-12)), plain


def cutoff_ts(store, months: int) -> int:
    from datetime import datetime, timezone
    end = datetime.fromtimestamp(store.w_end, timezone.utc)
    m = end.year * 12 + end.month - 1 - months
    return int(datetime(m // 12, m % 12 + 1, 1, tzinfo=timezone.utc).timestamp())


def train(pilot: Path, model_name: str, out: Path, seed=20261104, epochs=80, patience=6, lr=1e-3, cutoff_months=0,
          dev="cuda", log=print, init: Path | None = None, warm: int = 300, agent_alpha: float = 0.0) -> dict:
    """init: fine-tune from this checkpoint (missing parameters, e.g. a new plan embedding, keep their init).
    agent_alpha: weight each prefix by (rows of its agent)^-agent_alpha (batch.agent_weights) in the training loss,
    the early-stopping loss and the stop recalibration; 0 = the plain likelihood."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    store = build_store(pilot)
    cutoff = None
    if cutoff_months:
        cutoff = cutoff_ts(store, cutoff_months)
        store = build_store(pilot, cutoff=cutoff)
    cfg = CONFIGS[model_name]
    assert not (agent_alpha and cfg.b2), "agent weighting is not defined for B2"
    assert not cfg.pace, "pace models are built post hoc from a trained checkpoint (python -m lpw.p1.pace)"
    views = Views(store, agent_weights(store, agent_alpha) if agent_alpha else None)
    model = Model(cfg, store.n_venues, store.prof_vec(0).shape[0], torch.as_tensor(store.venue_sc)).to(dev)
    if init is not None:
        missing, unexpected = model.load_state_dict(torch.load(init, map_location=dev), strict=False)
        assert not unexpected, unexpected
        log(f"fine-tune from {init}; new parameters: {missing}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # constant-LR Adam stalls short of the optimum (Phase 1 diagnosis: 2018 LPW gained 0.16 val nats from a 1e-4
    # fine-tune and its stopping probability was under-fit), so the LR decays on validation plateaus
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler()
    vb = pack(val_specs(store, store.agents("val")), views)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("recal.json", "model_raw.pt"):  # a new checkpoint invalidates an earlier recalibration
        (out / stale).unlink(missing_ok=True)
    best, best_ep, hist, step = float("inf"), -1, [], 0
    ref_rows = None  # constant per-step normaliser, fixed from the first epoch's packing
    for ep in range(epochs):
        model.train()
        t = time.time()
        tb = pack(train_specs(store, store.agents("train"), rng), views, rng=rng)
        if ref_rows is None:
            ref_rows = float(np.mean([sum(len(js) for _, _, js in bt) for bt in tb]))
        tl, tn = 0.0, 0
        for b in loader(views, tb):
            b = to_dev(b, dev)
            if step < warm:
                for g in opt.param_groups:
                    g["lr"] = lr * (step + 1) / warm
            loss, _ = model.loss(b)
            # model.loss is a per-batch mean; batches hold very different row counts and packing groups prefixes
            # by full-stream cost, so a plain mean weights prefixes by 1/rows(batch), which correlates with
            # whether the stream is about to end (it biased the stopping probability). Sum / constant instead.
            scale = (len(b["o_row"]) if cfg.b2 else len(b["row_b"])) / ref_rows
            loss = loss * scale
            if "agent_w" in b:  # the batch's total weight replaces its row count in the clip threshold
                scale = b["agent_w"][b["row_b"]].sum().item() / ref_rows
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # the clip threshold scales with the batch too: a fixed threshold renormalises every clipped step back
            # to the old batch-mean weighting (almost every step clips early on)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 * scale)
            scaler.step(opt)
            scaler.update()
            step += 1
            tl += loss.item()
            tn += 1
        vl, vplain = val_loss(model, views, vb, dev)
        vknown = None
        if cfg.plan:  # selection uses s unknown (no future information); also track the known-s branch generation uses
            model.plan_inference = True
            vknown = val_loss(model, views, vb, dev)[1]
            model.plan_inference = False
        if step >= warm:
            sched.step(vl)
        hist.append({"epoch": ep, "train": tl / max(tn, 1), "val": vl, "val_plain": vplain, "val_known_s": vknown, "lr": opt.param_groups[0]["lr"],
                     "sec": round(time.time() - t, 1), "steps": step})
        log(f"[{model_name}] ep {ep} train {tl / max(tn, 1):.4f} val {vl:.4f} plain {vplain:.4f} ({time.time() - t:.0f}s)")
        if vl < best - 1e-4:
            best, best_ep = vl, ep
            torch.save(model.state_dict(), out / "model.pt")
        elif ep - best_ep >= patience:
            break
    info = {"model": model_name, "pilot": str(pilot), "seed": seed, "best_val": best, "best_epoch": best_ep, "ref_rows": ref_rows,
            "init": str(init) if init else None, "lr": lr, "agent_alpha": agent_alpha,
            "cutoff": cutoff, "params": sum(p.numel() for p in model.parameters()), "history": hist}
    (out / "train.json").write_text(json.dumps(info, indent=1))
    from lpw.p1.recal import ensure_recalibrated
    ensure_recalibrated(pilot, model_name, out, cutoff_months=cutoff_months, dev=dev, log=log)
    return info


def load(pilot: Path, model_name: str, out: Path, store=None, dev="cuda"):
    store = store or build_store(pilot)
    model = Model(CONFIGS[model_name], store.n_venues, store.prof_vec(0).shape[0], torch.as_tensor(store.venue_sc)).to(dev)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/pilot")
    ap.add_argument("--model", default="lpw", choices=list(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--cutoff-months", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--init", default=None, help="fine-tune from this checkpoint")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--agent-alpha", type=float, default=0.0, help="prefix weight (rows of agent)^-alpha; 0 = plain")
    a = ap.parse_args()
    train(Path(a.pilot), a.model, Path(a.out), seed=a.seed, epochs=a.epochs, cutoff_months=a.cutoff_months,
          init=Path(a.init) if a.init else None, lr=a.lr, warm=a.warm, agent_alpha=a.agent_alpha)


if __name__ == "__main__":
    main()
