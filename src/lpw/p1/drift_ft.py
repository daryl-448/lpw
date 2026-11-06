"""Partial-backbone drift fine-tune (Phase 1 drift screen v2, reports/p1_drift_screen.md; Sol design round after the
base qualification).

Starting point: a full-train lpw_plan_prog checkpoint (B). Trainable: the upper two encoder layers, the final norm,
time_head and amt_head; everything else (embeddings, lower layers, kind / counterparty heads, progress residuals) is
frozen. AdamW, weight decay 0.1, lr 1e-5 (encoder / norm) and 1e-4 (heads), constant, at most 1200 updates;
snapshots at SNAPS (step 0 = B). No recalibration afterwards.

Arms (identical clean batches, plan-dropout draws and optimiser schedule):
  C  clean likelihood only (train.py's loss: plain per-prefix likelihood with censoring and plan dropout, scaled by
     rows / the base's ref_rows, clip 1.0 x scale);
  R  clean + w_J * J, with w_J fixed once, before any update, by the pre-registered rule
     w_J = KAPPA * median ||grad clean|| / median ||grad J|| over CAL_BATCHES fixed (clean batch, auxiliary batch) pairs
     at step 0, trainable parameters only (calibrate_wj). Sol's design had w_J = 0.05; measured at step 0 that makes the
     J gradient 5e-5 to 4e-4 of the clean one (runs/p1_diag/drift_v2/magnitude.py), so R would be C.
J (shift consistency): on auxiliary prefixes of the fine-tune agents (agent drawn uniformly, then a prefix length
k in [1, n] uniformly; agent-balanced, independent of the clean batches), the student on a corrupted copy of the
prefix (anchor.corrupt_view: prefix-wide gap shift 10^{-0.3, 0, +0.3}, log-amount shift {-0.5, 0, +0.5}, probabilities
0.25 / 0.5 / 0.25, independently) is pulled towards the frozen B on the clean prefix. Both use the real (known) plan
and deterministic (eval-mode) forwards, so at step 0 J is the shift response of B itself (zero on unshifted rows). J = 1/2 mean over rows of the mean squared
difference of the next-owned-event gap CDF (kind mixture) at GAP_POINTS + 1/2 mean over owned-target rows of the mean
squared difference of the truncated amount CDF at AMT_GRID; the time and amount terms are averaged separately first.
The amount comparison conditions both forwards on the same real next event (kind, time, counterparty class / target).

usage: python -m lpw.p1.drift_ft --base DIR --out ROOT --arm C|R [--pilot data/pilot_2018]"""
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from lpw.p1.anchor import corrupt_view, draw_shifts, merge_corrupted, split_groups
from lpw.p1.batch import CTX, Views, collate, collate_specs, pack, train_specs
from lpw.p1.plan import N_PLAN_N
from lpw.p1.store import DUST_LOG10, SUPPLY_LOG10
from lpw.p1.timelaw import EDGES

SNAPS = (0, 25, 50, 100, 200, 400, 800, 1200)
KAPPA = 0.25  # target ratio of the J gradient norm to the clean one at step 0
CAL_BATCHES = 16
CAL_SEED = 777
LR_ENC, LR_HEAD, WD = 1e-5, 1e-4, 0.1
AUX_ROWS = 128
HOLDOUT = 0.2  # share of training groups withheld from fine-tuning (anchor.split_groups, seed 0)
AMT_GRID = np.linspace(DUST_LOG10, SUPPLY_LOG10, 64)
GAP_MID = np.sqrt(EDGES[:-1] * EDGES[1:])  # 62 finite-bin log midpoints; the gap CDF is also taken at 0 (the
# same-second atom) and at the finite-support end (365 d), where 1 - F is the dormant mass: 64 points


def trainable(model):
    """(encoder / norm parameters, head parameters) the fine-tune moves."""
    enc = [p for lay in model.enc.layers[2:] for p in lay.parameters()] + list(model.ln.parameters())
    head = list(model.time_head.parameters()) + list(model.amt_head.parameters())
    return enc, head


def gap_cdf(logpk, law):
    """(n, 64) CDF of the next owned event's gap (mixture over the two kinds; law rows 2i + kind) at 0, the 62
    finite-bin log midpoints and 365 d. Float64, differentiable."""
    n = logpk.shape[0]
    dev = logpk.device
    mid = torch.as_tensor(GAP_MID, dtype=torch.float64, device=dev)
    pdt = torch.exp(law.log_pdt[:, 1:63])
    part = pdt / law.wid * (law.W(law.r0[:, None] + mid[None]) - law.W(law.r0[:, None] + law.lo[None]))  # (2n, 62)
    before = torch.cumsum(law.mass, 1) - law.mass  # finite mass of the bins left of each bin
    F = torch.cat([law.atom0[:, None], law.atom0[:, None] + before + part, (law.Z - law.atom63)[:, None]], 1) / law.Z[:, None]
    return (logpk.double().exp()[:, :, None] * F.view(n, 2, 64)).sum(1)


def amt_cdf(ap, k):
    """(n, 64) CDF of the truncated amount mixture at AMT_GRID. Float64, differentiable."""
    logit, mu, ls = ap.double().split(k, -1)
    sig = torch.exp(ls.clamp(-4, 2))
    w = torch.softmax(logit, -1)
    nd = torch.special.ndtr
    g = torch.as_tensor(AMT_GRID, dtype=torch.float64, device=ap.device)
    lo = nd((DUST_LOG10 - mu) / sig)
    den = (w * (nd((SUPPLY_LOG10 - mu) / sig) - lo)).sum(-1)
    num = (w[:, None] * (nd((g[None, :, None] - mu[:, None]) / sig[:, None]) - lo[:, None])).sum(-1)
    return num / den[:, None]


def forecast_cdfs(model, b):
    """(gap CDF (N, 64), amount CDF of the owned-target rows (No, 64), o_row) of a batch, known plan, eval forward."""
    assert not model.training and model.plan_inference
    model._capture = {}
    try:
        t = model.terms(b)
        cap = model._capture
    finally:
        model._capture = None
    assert bool((b["_plan_n_eff"] < N_PLAN_N).all()), "consistency rows need known plans"
    return gap_cdf(cap["logpk_live"], cap["law_live"]), amt_cdf(cap["amt_ap_live"], model.cfg.k_mix), t["o_row"]


def consistency(student, teacher, clean, cor):
    """J and its two parts. clean / cor: the clean batch and the merged corrupted batch (same rows)."""
    with torch.no_grad():
        Gt, At, ot = forecast_cdfs(teacher, clean)
    Gs, As, os_ = forecast_cdfs(student, cor)
    assert torch.equal(ot, os_)
    jt = ((Gs - Gt) ** 2).mean(1).mean()
    ja = ((As - At) ** 2).mean(1).mean() if len(os_) else torch.zeros((), dtype=torch.float64, device=Gs.device)
    return 0.5 * jt + 0.5 * ja, jt.item(), ja.item()


class AuxDS(torch.utils.data.Dataset):
    """auxiliary batch i: AUX_ROWS single-row exact specs (agent uniform over `agents`, prefix length uniform in
    [1, n]) as (clean batch, merged corrupted batch); draws seeded by (seed, i)."""

    def __init__(self, views, agents, n, seed, rows=AUX_ROWS):
        st = views.store
        ne = np.array([st.n_events(int(a)) for a in agents])
        ok = (ne >= 1) & (st.origin[agents] < st.obs_end[agents])
        self.views, self.agents, self.ne, self.n, self.seed, self.rows = views, np.asarray(agents)[ok], ne[ok], n, seed, rows

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        pick = rng.integers(0, len(self.agents), self.rows)
        ks = rng.integers(1, self.ne[pick] + 1)
        specs = [(int(self.agents[p]), max(0, int(k) - CTX), np.array([int(k) - max(0, int(k) - CTX)])) for p, k in zip(pick, ks)]
        clean = collate_specs(self.views, specs)
        dts, das = draw_shifts(rng, len(specs))
        st = self.views.store
        cv = [corrupt_view(self.views(a), float(t), float(m)) for (a, _, _), t, m in zip(specs, dts, das)]
        cor = merge_corrupted(clean, collate(cv, [(s, js) for _, s, js in specs], st.n_venues, st.w_start))
        tt = lambda d: {k: torch.from_numpy(np.ascontiguousarray(x)) for k, x in d.items()}
        return tt(clean), tt(cor)


def _loader(ds, workers=4):
    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, pin_memory=True,
                                       prefetch_factor=4 if workers else None, persistent_workers=False)


def _eval_mode(m):
    was = (m.training, m.plan_inference)
    m.eval()
    m.plan_inference = True
    return was


def _restore(m, was):
    m.train(was[0])
    m.plan_inference = was[1]


@torch.no_grad()
def eval_j(student, teacher, batches, dev):
    from lpw.p1.train import to_dev
    was = _eval_mode(student)
    try:
        out = [consistency(student, teacher, to_dev(c, dev), to_dev(r, dev)) for c, r in batches]
    finally:
        _restore(student, was)
    return {"J": float(np.mean([float(o[0]) for o in out])), "J_time": float(np.mean([o[1] for o in out])),
            "J_amt": float(np.mean([o[2] for o in out]))}


def calibrate_wj(model, teacher, views, agents, ref_rows, dev, kappa=KAPPA, n=CAL_BATCHES, seed=CAL_SEED) -> dict:
    """w_J = kappa * median ||grad of the scaled clean loss|| / median ||grad J|| over n fixed (clean batch, auxiliary
    batch) pairs, trainable parameters only, at the current (step 0) weights. The torch RNG state is restored, so the
    calibration does not shift the clean plan-dropout / dropout draws (C and R see the same ones)."""
    from lpw.p1.train import loader, to_dev
    P = [p for p in model.parameters() if p.requires_grad]

    def gnorm(loss):
        g = torch.autograd.grad(loss, P, allow_unused=True)
        return float(torch.sqrt(sum((x.double() ** 2).sum() for x in g if x is not None)))

    rng = np.random.default_rng(seed)
    tb = pack(train_specs(views.store, agents, rng), views, rng=rng)[:n]
    aux = AuxDS(views, agents, n, seed=seed)
    gc, gj, jv = [], [], []
    was = (model.training, model.plan_inference)
    devs = [torch.device(dev).index or 0] if torch.device(dev).type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        for i, b in enumerate(loader(views, tb, 0)):
            b = to_dev(b, dev)
            model.train()
            model.plan_inference = False
            loss, _ = model.loss(b)
            gc.append(gnorm(loss * (len(b["row_b"]) / ref_rows)))
            c, r = aux[i]
            model.eval()
            model.plan_inference = True
            J, _, _ = consistency(model, teacher, to_dev(c, dev), to_dev(r, dev))
            gj.append(gnorm(J.float()))
            jv.append(J.item())
    _restore(model, was)
    w = kappa * float(np.median(gc)) / float(np.median(gj))
    return {"w_j": w, "kappa": kappa, "n": n, "seed": seed, "grad_clean": gc, "grad_J": gj, "J": jv}


def finetune(pilot: Path, base: Path, out: Path, arm: str, steps=SNAPS[-1], kappa=KAPPA, seed=20261104, dev="cuda", log=print, lr_enc=LR_ENC, lr_head=LR_HEAD, holdout=HOLDOUT,
             workers=4, store=None) -> dict:
    from lpw.p1.atomic import atomic_path, write_text
    from lpw.p1.store import build_store
    from lpw.p1.train import load, loader, to_dev
    assert arm in ("C", "R")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    st = store or build_store(pilot)
    binfo = json.loads((base / "train.json").read_text())
    assert binfo["model"] == "lpw_plan_prog", binfo["model"]
    ref_rows = float(binfo["ref_rows"])
    fit, hold = split_groups(st, pilot, holdout) if holdout > 0 else (st.agents("train"), np.zeros(0, np.int64))
    views = Views(st)
    model = load(pilot, "lpw_plan_prog", base, st, dev)
    teacher = copy.deepcopy(model).eval()
    teacher.plan_inference = True
    for p in teacher.parameters():
        p.requires_grad_(False)
    enc, head = trainable(model)
    ids = {id(p) for p in enc + head}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    frozen0 = {k: v.detach().clone() for k, v in model.named_parameters() if id(v) not in ids}
    opt = torch.optim.AdamW([{"params": enc, "lr": lr_enc}, {"params": head, "lr": lr_head}], weight_decay=WD)
    scaler = torch.amp.GradScaler()
    jeval = [AuxDS(views, fit, 8, seed=10_000)[i] for i in range(8)]  # fixed consistency monitor, both arms
    aux = iter(_loader(AuxDS(views, fit, steps + 1, seed=seed), workers)) if arm == "R" else None
    out.mkdir(parents=True, exist_ok=True)
    cal = calibrate_wj(model, teacher, views, fit, ref_rows, dev, kappa) if arm == "R" else None
    w_j = cal["w_j"] if cal else 0.0
    if cal:
        log(f"[R] w_J = {w_j:.4g} (kappa {kappa}; median |grad clean| {np.median(cal['grad_clean']):.3e}, |grad J| {np.median(cal['grad_J']):.3e})")
    info = {"arm": arm, "base": str(base), "pilot": str(pilot), "seed": seed, "w_j": w_j, "calibration": cal,
            "lr_enc": lr_enc, "lr_head": lr_head, "wd": WD, "steps": steps, "snaps": list(SNAPS), "aux_rows": AUX_ROWS,
            "ref_rows": ref_rows, "holdout": holdout, "fit_agents": int(len(fit)), "holdout_agent_ids": [int(x) for x in hold],
            "trainable": int(sum(p.numel() for p in enc + head)), "trace": []}

    def snap(s):
        d = out / f"step{s:04d}"
        d.mkdir(parents=True, exist_ok=True)
        with atomic_path(d / "model.pt") as t:
            torch.save(model.state_dict(), t)
        shutil.copy(base / "train.json", d / "train.json")
        j = eval_j(model, teacher, [(to_dev(c, dev), to_dev(r, dev)) for c, r in jeval], dev)
        info["trace"].append({"step": s, **j})
        log(f"[{arm}] snapshot {s}: monitor J {j['J']:.3e} (time {j['J_time']:.3e}, amount {j['J_amt']:.3e})")

    snap(0)
    step, t0 = 0, time.time()
    run = {"clean": 0.0, "J": 0.0, "n": 0}
    while step < steps:
        model.train()
        tb = pack(train_specs(st, fit, rng), views, rng=rng)
        for b in loader(views, tb, workers):
            if step >= steps:
                break
            b = to_dev(b, dev)
            model.train()
            model.plan_inference = False
            loss, _ = model.loss(b)
            scale = len(b["row_b"]) / ref_rows
            tot = loss * scale
            if arm == "R":
                c, r = next(aux)
                was = _eval_mode(model)
                try:
                    J, _, _ = consistency(model, teacher, to_dev(c, dev), to_dev(r, dev))
                finally:
                    _restore(model, was)
                tot = tot + w_j * J.float()
                run["J"] += J.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(tot).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc + head, 1.0 * scale)
            scaler.step(opt)
            scaler.update()
            step += 1
            run["clean"] += loss.item()
            run["n"] += 1
            if step % 25 == 0:
                log(f"[{arm}] step {step}: clean {run['clean'] / run['n']:.4f}" + (f" J {run['J'] / run['n']:.3e}" if arm == "R" else "")
                    + f" ({time.time() - t0:.0f}s)")
                run = {"clean": 0.0, "J": 0.0, "n": 0}
            if step in SNAPS:
                snap(step)
    for k, v in model.named_parameters():
        if k in frozen0:
            assert torch.equal(v.detach(), frozen0[k]), f"frozen parameter {k} moved"
    info["sec"] = round(time.time() - t0)
    write_text(out / "ft.json", json.dumps(info, indent=1))
    return info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/pilot_2018")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True, help="snapshots go to OUT/{arm}/stepNNNN")
    ap.add_argument("--arm", required=True, choices=("C", "R"))
    ap.add_argument("--steps", type=int, default=SNAPS[-1])
    ap.add_argument("--lr-enc", type=float, default=LR_ENC)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--holdout", type=float, default=HOLDOUT, help="0 = fine-tune on every training group")
    a = ap.parse_args()
    finetune(Path(a.pilot), Path(a.base), Path(a.out) / a.arm, a.arm, steps=a.steps, lr_enc=a.lr_enc, lr_head=a.lr_head, holdout=a.holdout)


if __name__ == "__main__":
    main()
