"""Plan-anchored heads (Phase 1; Sol anchor design round, runs/p1_diag/drift_results.md).

Diagnosis: every head is calibrated on real histories and the sampler is faithful, but in free-running generation heavy
accounts drift jointly to slower pace and larger amounts as depth grows; the plan s reaches the heads only through the
PROFILE token and the model follows its (drifting) history. On real data history and plan always agree, so plain
training never shows the heads what to do when they disagree.

Anchor: small residual adapters on the time logits (bins 0..62 and the weekly logits; the dormant bin stays with the
progress residuals) and on the amount mixture parameters, taking each head's own input plus explicit plan embeddings.
They are zero-initialised, and rows whose plan is unknown bypass them exactly, so the s-unknown branch (the reportable
NLL) is bit-identical to the base checkpoint.

Robust training ("R"): next to each clean batch, a corrupted view of the same prefixes with prefix-wide level shifts of
the history: positive gaps x 10^dt (dt in {-0.3, 0, +0.3}), log10 amounts + da (da in {-0.5, 0, +0.5}), probabilities
0.25 / 0.5 / 0.25, independently. Calendar timestamps, zero gaps, kinds and counterparties are kept; amount shifts scale
amounts, fees and balances together (the ledger identity holds; partner totals follow from the shifted amounts).
Targets stay the real next events (their own time / amount features come from the clean view). Objective
L_clean + 0.25 L_corrupted on the time / censoring and amount terms. "C" is the matched clean-only fit.

Screen (frozen backbone): cache the heads' inputs once (clean and corrupted), fit only the adapters on a fixed subset of
training groups. usage: python -m lpw.p1.anchor --base DIR --out ROOT [--arms C,R] [--frac 0.3]"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lpw.p1.plan import N_PLAN_A, N_PLAN_N

DT_SHIFT = (-0.30, 0.0, 0.30)
DA_SHIFT = (-0.50, 0.0, 0.50)
P_SHIFT = (0.25, 0.50, 0.25)
W_COR = 0.25
N_TIME_ADAPT = 63 + 24 + 7   # time logits the adapter may move: bins 0..62 and the weekly pieces (not the dormant bin)


class Anchor(nn.Module):
    def __init__(self, d: int, k: int, da: int = 32, h: int = 128):
        super().__init__()
        self.e_n = nn.Embedding(N_PLAN_N, da)
        self.e_a = nn.Embedding(N_PLAN_A, da)
        self.e_k = nn.Embedding(2, da)
        self.time = nn.Sequential(nn.Linear(d + 3 * da, h), nn.GELU(), nn.Linear(h, N_TIME_ADAPT))
        self.amt = nn.Sequential(nn.Linear(4 * d + 2 * da, h), nn.GELU(), nn.Linear(h, 3 * k))
        for m in (self.time[-1], self.amt[-1]):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)

    def _plan(self, pn, pa):
        known = (pn < N_PLAN_N) & (pa < N_PLAN_A)
        e = torch.cat([self.e_n(pn.clamp(max=N_PLAN_N - 1)), self.e_a(pa.clamp(max=N_PLAN_A - 1))], -1)
        return known, e

    def apply_time(self, out, ho, pn, pa):
        """out: (2N, 95) time-head output (rows 2i + kind); ho: (N, d) row states; pn, pa: (N,) row plans."""
        known, e = self._plan(pn, pa)
        if not bool(known.any()):
            return out
        n = ho.shape[0]
        x = torch.cat([ho.float()[:, None].expand(n, 2, -1), self.e_k.weight[None].expand(n, 2, -1), e[:, None].expand(n, 2, -1)], -1)
        dl = self.time(x).reshape(2 * n, N_TIME_ADAPT).to(out.dtype)
        full = torch.cat([dl[:, :63], torch.zeros_like(dl[:, :1]), dl[:, 63:]], 1)
        return torch.where(known.repeat_interleave(2)[:, None], out + full, out)

    def apply_amt(self, ap, x, pn, pa):
        """ap: (No, 3k) amount-mixture parameters; x: (No, 4d) amount-head input; pn, pa: (No,) plans."""
        known, e = self._plan(pn, pa)
        if not bool(known.any()):
            return ap
        dl = self.amt(torch.cat([x.float(), e], -1)).to(ap.dtype)
        return torch.where(known[:, None], ap + dl, ap)


# ---- corruption ----------------------------------------------------------------------------------------------------
def corrupt_view(v: dict, dt_shift: float, da_shift: float) -> dict:
    """Prefix-wide level shift of a view's history (see module doc). Returns a new dict; v is not modified."""
    from lpw.p1.timelaw import dt_bin
    w = dict(v)
    if dt_shift:
        dt = np.asarray(v["dt"], np.float64)
        dt = np.where(dt > 0, dt * 10.0 ** dt_shift, dt)
        w["dt"], w["dtbin"] = dt, dt_bin(dt)
    if da_shift:
        f = 10.0 ** da_shift
        w["amt"] = (np.asarray(v["amt"], np.float64) + da_shift).astype(np.float32)
        w["eth"] = np.asarray(v["eth"], np.float64) * f
        w["fee"] = np.asarray(v["fee"], np.float64) * f
        w["bal"] = np.asarray(v["bal"], np.float64) * f
        w["bal_end"] = float(v["bal_end"]) * f
    return w


def draw_shifts(rng, n):
    return rng.choice(DT_SHIFT, n, p=P_SHIFT), rng.choice(DA_SHIFT, n, p=P_SHIFT)


HISTORY_KEYS = ("tok_", "e_")  # plus o_led: everything derived from the prefix (not from the target event)


def merge_corrupted(clean: dict, cor: dict) -> dict:
    """clean batch with the history-derived fields of the corrupted batch (same specs, so rows align)."""
    out = dict(clean)
    for k, x in cor.items():
        if k.startswith(HISTORY_KEYS) or k == "o_led":
            out[k] = x
    return out


class CorruptDS(torch.utils.data.Dataset):
    """batches of exact specs; corrupt=True gives the merged corrupted batch (shifts seeded by (seed, batch index))."""

    def __init__(self, views, batches, corrupt: bool, seed: int = 0):
        self.views, self.batches, self.corrupt, self.seed = views, batches, corrupt, seed

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        from lpw.p1.batch import collate, collate_specs
        sp = self.batches[i]
        clean = collate_specs(self.views, sp)
        if not self.corrupt:
            return clean
        dts, das = draw_shifts(np.random.default_rng((self.seed, i)), len(sp))
        st = self.views.store
        cv = [corrupt_view(self.views(a), float(t), float(m)) for (a, _, _), t, m in zip(sp, dts, das)]
        cor = collate(cv, [(s, js) for _, s, js in sp], st.n_venues, st.w_start)
        b = merge_corrupted(clean, cor)
        b["shift_dt"], b["shift_da"] = dts.astype(np.float32), das.astype(np.float32)
        return b


# ---- caching and fitting -------------------------------------------------------------------------------------------
@torch.no_grad()
def build_cache(model, store, agents, corrupt: bool, dev="cuda", seed=0, log=print) -> dict:
    """Head inputs and base outputs on every exact prefix of `agents` (real plans), clean or corrupted."""
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.train import to_dev
    views = Views(store)
    batches = pack(exact_specs(store, agents), views)
    dl = torch.utils.data.DataLoader(CorruptDS(views, batches, corrupt, seed), batch_size=None, shuffle=False, num_workers=4,
                                     pin_memory=True, prefetch_factor=4)
    was = model.plan_inference
    model.eval()
    model.plan_inference = True
    C = {k: [] for k in ("logpk", "ho", "out", "t0", "r", "typ", "tk", "prog", "pn", "pa", "own", "x", "ap", "amt", "opn", "opa")}
    nrow = 0
    try:
        for i, b in enumerate(dl):
            b = to_dev(b, dev)
            model._capture = {}
            t = model.terms(b)
            cap = model._capture
            rb = b["row_b"]
            n = len(rb)
            pn, pa = b["_plan_n_eff"][rb], b["_plan_a_eff"][rb]
            C["logpk"].append(cap["logpk"].float()); C["ho"].append(cap["ho"].half()); C["out"].append(cap["out"].float().view(-1, 2, cap["out"].shape[-1]))  # (N, 2 kinds, 95)
            C["t0"].append(b["row_t0"]); C["r"].append(b["row_r"]); C["typ"].append(b["row_type"].to(torch.int8))
            C["tk"].append(b["row_tkind"].clamp(min=0)); C["prog"].append(model.row_progress(b)); C["pn"].append(pn); C["pa"].append(pa)
            oi = t["o_row"]
            C["own"].append(oi + nrow); C["x"].append(cap["amt_x"].half()); C["ap"].append(cap["amt_ap"].float())
            C["amt"].append(b["o_amt"].float()); C["opn"].append(pn[oi]); C["opa"].append(pa[oi])
            nrow += n
            if i % 100 == 0:
                log(f"  cache ({'corrupted' if corrupt else 'clean'}) batch {i}/{len(batches)}")
    finally:
        model._capture = None
        model.plan_inference = was
    C = {k: torch.cat(v) for k, v in C.items()}
    own_of_row = torch.full((nrow,), -1, dtype=torch.int64, device=dev)
    own_of_row[C["own"]] = torch.arange(len(C["own"]), device=dev)
    C["own_of_row"] = own_of_row
    return C


def rows_loss(model, C, idx):
    """mean per-row time + censoring (+ kind, constant) + amount NLL of rows idx under the current adapters."""
    from lpw.p1.model import trunc_mog_logpdf
    n = len(idx)
    pn, pa = C["pn"][idx], C["pa"][idx]
    out = model.anchor.apply_time(C["out"][idx].reshape(2 * n, -1), C["ho"][idx], pn, pa)
    logpk, law, _ = model.time_law(C["logpk"][idx], out, C["t0"][idx], prog=C["prog"][idx])
    r, typ, tk = C["r"][idx], C["typ"][idx], C["tk"][idx]
    r2 = r.repeat_interleave(2)
    logf = law.log_density(r2).view(n, 2)
    logS = law.log_survival(r2).view(n, 2)
    own, cen = typ == 1, (typ == 2) | (typ == 3)
    l = torch.where(own, -logpk.gather(1, tk[:, None])[:, 0].double() - logf.gather(1, tk[:, None])[:, 0], 0.0)
    l = l + torch.where(cen, -torch.logsumexp(logpk.double() + logS, -1), 0.0)
    oi = C["own_of_row"][idx]
    sel = oi >= 0
    o = oi[sel]
    ap = model.anchor.apply_amt(C["ap"][o], C["x"][o], C["opn"][o], C["opa"][o])
    la = -trunc_mog_logpdf(ap, C["amt"][o], model.cfg.k_mix).double()
    return (l.sum() + la.sum()) / n


def eval_loss(model, C, chunk=16384):
    with torch.no_grad():
        n = len(C["t0"])
        return sum(rows_loss(model, C, torch.arange(i, min(i + chunk, n), device=C["t0"].device)).item() * min(chunk, n - i)
                   for i in range(0, n, chunk)) / n


def fit_adapters(model, Cc, Ccor=None, steps=1000, batch_rows=4096, lr=3e-4, w_cor=W_COR, seed=0, log=print) -> dict:
    """Adam on the adapters only; every other parameter frozen and untouched."""
    params = list(model.anchor.parameters())
    before = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("anchor.")}
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator(device=Cc["t0"].device).manual_seed(seed)
    trace = []
    try:
        for s in range(steps):
            ic = torch.randint(0, len(Cc["t0"]), (batch_rows,), generator=g, device=g.device)
            loss = rows_loss(model, Cc, ic)
            if Ccor is not None:
                ir = torch.randint(0, len(Ccor["t0"]), (batch_rows,), generator=g, device=g.device)
                loss = loss + w_cor * rows_loss(model, Ccor, ir)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if s % 100 == 0 or s == steps - 1:
                trace.append((s, float(loss)))
                log(f"  step {s}: loss {float(loss):.4f}")
    finally:
        for p, f in req.items():
            p.requires_grad_(f)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "a non-adapter parameter moved"
    return {"steps": steps, "batch_rows": batch_rows, "lr": lr, "w_cor": w_cor if Ccor is not None else 0.0, "seed": seed, "trace": trace}


def train_subset(store, pilot, frac: float, seed: int = 0) -> np.ndarray:
    """training agents of a fixed random subset of training groups."""
    import polars as pl
    tr = store.agents("train")
    ag = pl.read_parquet(Path(pilot) / "agents.parquet")
    gid = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    g = np.array([gid[int(x)] for x in store.node_id[tr]])
    ug = np.unique(g)
    keep = np.random.default_rng(seed).choice(ug, int(round(frac * len(ug))), replace=False)
    return tr[np.isin(g, keep)]


def load_anchor_model(pilot, base_dir, store, dev="cuda"):
    """the base lpw_plan_prog checkpoint inside an anchor model (adapters zero = identical to the base)."""
    from lpw.p1.model import CONFIGS, Model
    m = Model(CONFIGS["lpw_plan_prog_anchor"], store.n_venues, store.prof_vec(0).shape[0], torch.as_tensor(store.venue_sc)).to(dev)
    sd = torch.load(Path(base_dir) / "model.pt", map_location=dev)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("anchor.") for k in missing), (missing, unexpected)
    m.eval()
    return m


def main():
    import argparse
    from lpw.p1.atomic import atomic_path, write_text
    from lpw.p1.store import build_store
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/pilot_2018")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True, help="arms go to OUT/{arm}")
    ap.add_argument("--arms", default="C,R")
    ap.add_argument("--frac", type=float, default=0.3, help="share of training groups (screen)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-rows", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    pilot, base, root = Path(a.pilot), Path(a.base), Path(a.out)
    st = build_store(pilot)
    agents = train_subset(st, pilot, a.frac) if a.frac < 1 else st.agents("train")
    m0 = load_anchor_model(pilot, base, st)
    print(f"subset: {len(agents)} training agents ({a.frac:.0%} of groups)")
    Cc = build_cache(m0, st, agents, corrupt=False)
    arms = a.arms.split(",")
    Ccor = build_cache(m0, st, agents, corrupt=True) if "R" in arms else None
    print(f"cache: {len(Cc['t0'])} rows, {len(Cc['own'])} owned")
    base_loss = {"clean": eval_loss(m0, Cc), **({"corrupted": eval_loss(m0, Ccor)} if Ccor is not None else {})}
    for arm in arms:
        m = load_anchor_model(pilot, base, st)
        info = fit_adapters(m, Cc, Ccor if arm == "R" else None, steps=a.steps, batch_rows=a.batch_rows, lr=a.lr)
        info.update(arm=arm, base=str(base), frac=a.frac, agents=int(len(agents)), rows=int(len(Cc["t0"])),
                    loss_base=base_loss, loss_after={"clean": eval_loss(m, Cc), **({"corrupted": eval_loss(m, Ccor)} if Ccor is not None else {})})
        out = root / arm
        out.mkdir(parents=True, exist_ok=True)
        with atomic_path(out / "model.pt") as t:
            torch.save(m.state_dict(), t)
        shutil.copy(base / "train.json", out / "train.json")
        write_text(out / "anchor.json", json.dumps(info, indent=1))
        print(f"{arm}: subset loss clean {base_loss['clean']:.4f} -> {info['loss_after']['clean']:.4f}"
              + (f", corrupted {base_loss['corrupted']:.4f} -> {info['loss_after']['corrupted']:.4f}" if Ccor is not None else ""))


if __name__ == "__main__":
    main()
