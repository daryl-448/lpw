"""Fit the shallow-row correction (lpw.p1.shallow) on a frozen base checkpoint, select it on group-held-out folds
of the training groups, and write one checkpoint per variant plus teacher-forced readouts.

**Cache.** Every row the offsets can touch (real plan, owned depth <= D_MAX) is cached once from the frozen base:
- the kind log-probs and the 64 time-bin logits after the progress residual;
- the raw counterparty-gate logits and their legality for owned rows;
- the targets.

The offsets change only the time-bin logits, never the hour / weekday law. So the normalizer, the observed density and
the survival of each row are linear in the bin probabilities, with coefficients precomputed once from the base TimeLaw
(law_coefs). The correction's NLL (kind + time + censoring + counterparty gate) is recomputed from the cache with the
model's own offset functions (lpw.p1.shallow.apply_*, lpw.p1.model.masked_gate_logp). Pointer, venue and amount terms
do not depend on the offsets.

**Selection** (design: runs/p1_diag/after_v2/sol_r2/sol_r2.md §4):
- 3 folds of training groups (seed 0);
- each variant (k, kc, kct) x L2 penalty {1, 10, 100} on the summed NLL is fitted on 2 folds and scored on the third;
- score = mean over held-out agents of the agent's mean shallow-row NLL.
Per variant, the largest penalty within one whole-group SE of the variant's best is chosen. The primary variant is
the first of (zero, k, kc, kct) whose score is within one SE of the overall best. "zero" means no correction is
selected. Every variant is refitted on all training groups at its penalty.

usage: uv run python -m lpw.p1.shallow_fit --base runs/p1_r0/2018/lpw_plan_prog_rp --pilot data/pilot_2018 --out runs/p1_shallow/2018
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.atomic import atomic_path, write_text
from lpw.p1.batch import CTX, Views, pack
from lpw.p1.model import CONFIGS, Model, masked_gate_logp
from lpw.p1.shallow import D_MAX, VARIANTS, Shallow, apply_gate, apply_kind, apply_time
from lpw.p1.store import IN_AGENT, build_store
from lpw.p1.timelaw import TimeLaw
from lpw.p1.train import load, loader, to_dev

BASE_MODEL = "lpw_plan_prog"
LAMBDAS = (1.0, 10.0, 100.0)
N_FOLDS = 3
ORDER = ("zero", "k", "kc", "kct")
BANDS = (("light1-4", 1, 5), ("mid5-99", 5, 100), ("heavy100+", 100, 10 ** 9))
OWNED_KEYS = ("g", "ent_any", "ven_any", "cls")


def shallow_specs(store, agents) -> list[tuple[int, int, np.ndarray]]:
    """Every prefix k with at most D_MAX owned events before it, each with the simulator's context: positions of the
    s = 0 window for k <= CTX, else the window s = k - CTX scored at its last position (as batch.exact_specs)."""
    specs = []
    for a in agents:
        if store.origin[a] >= store.obs_end[a]:
            continue
        s0, e0 = store.off[a], store.off[a + 1]
        n_own = np.concatenate([[0], np.cumsum(store.kind[s0:e0] != IN_AGENT)])
        ks = np.flatnonzero(n_own <= D_MAX)  # n_own is monotone: prefixes 0..kmax
        specs.append((int(a), 0, ks[ks <= CTX]))
        for k in ks[ks > CTX]:
            specs.append((int(a), int(k) - CTX, np.array([CTX])))
    return specs


def law_coefs(law: TimeLaw, r: torch.Tensor) -> tuple:
    """For rows of a TimeLaw and gaps r (float64, one per row): coefficients a, s (n, 64), the bin b of r and a
    log-density term c (n,), such that for every bin-probability vector p with the same hour / weekday law,
      Z = p . a,   log f(r) = log p_b + c - log Z,   S(r) = (p . s) / Z
    reproduce TimeLaw.log_density / log_survival."""
    n = law.r0.shape[0]
    f64 = torch.float64
    w0 = law.w_at(law.r0)
    integ = law.W(law.r0[:, None] + law.hi[None]) - law.W(law.r0[:, None] + law.lo[None])
    a = torch.cat([w0[:, None], integ / law.wid[None], torch.ones(n, 1, dtype=f64, device=law.dev)], 1)
    r = r.to(f64)
    b = law._bin(r)
    fin = (b >= 1) & (b <= 62)
    m = torch.clamp(b, 1, 62) - 1
    x = law.r0 + r
    c = torch.where(b == 0, torch.log(w0), torch.where(fin, -torch.log(law.wid[m]) + torch.log(law.w_at(x)), torch.zeros_like(r)))
    idx = torch.arange(64, device=law.dev)[None]
    bf = (m + 1)[:, None]
    part = (law.W((law.r0 + law.hi[m])[:, None])[:, 0] - law.W(x[:, None])[:, 0]) / law.wid[m]
    s = torch.where(idx > bf, a, torch.zeros_like(a))
    s = s.scatter(1, bf, part[:, None])
    s = torch.where((b >= 63)[:, None], torch.where(idx == 63, a, torch.zeros_like(a)), s)
    s = torch.where(((r > 0) & (r < 1))[:, None], torch.where(idx >= 1, a, torch.zeros_like(a)), s)
    s = torch.where((r <= 0)[:, None], a, s)
    return a, s, b, c


@torch.no_grad()
def cache_batch(model, b: dict, agent: np.ndarray, obs_end: np.ndarray) -> dict:
    """The cache rows of one collated batch (on its device), from the uncorrected model. agent: the agent id of every
    row; obs_end: per agent id. The model's plan mode is used as set (build_cache sets the real plan)."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=b["tok_kind"].is_cuda):
        Hk, Ho = model.encode(b)
    Hk, Ho = Hk.float(), Ho.float()
    rb, rj, t0 = b["row_b"], b["row_j"], b["row_t0"]
    N = len(rb)
    assert len(agent) == N
    ex = model.row_extras(b)
    assert "shallow" not in ex, "cache from the uncorrected base"
    with torch.autocast("cuda", enabled=False):
        logpk, law, out = model.kind_time(Hk[rb, rj], Ho[rb, rj], t0, **ex)
        hor = (torch.as_tensor(obs_end[agent], dtype=torch.float64, device=t0.device) - t0.double()).clamp(min=0)
        a_, s_r, b_r, c_r = law_coefs(law, b["row_r"].double().repeat_interleave(2))
        _, s_h, _, _ = law_coefs(law, hor.repeat_interleave(2))
        tk = b["row_tkind"].clamp(min=0)
        oi = b["o_row"]
        ko = tk[oi]
        tt = model.time_emb(b["o_tbin"], b["o_tmonth"], b["o_tcont"])
        ctx = model.cp_context(b, rb[oi], Ho[rb, rj][oi], ko, tt)
        ent_ok, ven_ok = model.masks(b, ko)
        g = model.gate(ctx).float()
    return {"agent": torch.as_tensor(agent, device=t0.device), "role": b["role"][rb], "plan": b["_plan_n_eff"][rb],
            "nown": b["row_nown"], "pk": b["tok_kind"][rb, rj], "pc": b["tok_cls"][rb, rj], "typ": b["row_type"], "tk": tk,
            "logpk": logpk.float(), "bins": out[:, :64].float().view(N, 2, 64), "a": a_.view(N, 2, 64),
            "s_r": s_r.view(N, 2, 64), "s_h": s_h.view(N, 2, 64), "b_r": b_r.view(N, 2)[:, 0], "c_r": c_r.view(N, 2),
            "own_idx": oi, "g": g, "ent_any": ent_ok.any(-1), "ven_any": ven_ok.any(-1), "cls": b["o_cls"]}


def build_cache(model, store, agents, views, dev="cuda") -> dict:
    """Frozen-base quantities of every shallow row of the agents, with the REAL plan (model.plan_inference = True).
    Row tensors are on the CPU; owned-row tensors (OWNED_KEYS) are indexed by own_idx into the rows. Per-kind tensors
    have shape (N, 2, ...)."""
    model.eval()
    was = model.plan_inference
    model.plan_inference = True
    parts, base = [], 0
    batches = pack(shallow_specs(store, agents), views)
    for specs, b in zip(batches, loader(views, batches)):
        agent = np.concatenate([np.full(len(js), a) for a, _, js in specs])
        c = cache_batch(model, to_dev(b, dev), agent, store.obs_end)
        c["own_idx"] = c["own_idx"] + base
        base += len(c["typ"])
        parts.append({k: v.cpu() for k, v in c.items()})
    model.plan_inference = was
    return {k: torch.cat([p_[k] for p_ in parts]) for k in parts[0]}


def to(C: dict, dev) -> dict:
    return {k: v.to(dev) for k, v in C.items()}


def subset(C: dict, rows: torch.Tensor) -> dict:
    """the cache restricted to rows (bool mask), owned-row tensors re-indexed."""
    keep_o = rows[C["own_idx"]]
    new_idx = torch.cumsum(rows.long(), 0) - 1
    out = {k: v[rows] for k, v in C.items() if k != "own_idx" and k not in OWNED_KEYS}
    out.update({k: C[k][keep_o] for k in OWNED_KEYS})
    out["own_idx"] = new_idx[C["own_idx"][keep_o]]
    return out


def row_nll(sh: Shallow | None, C: dict, detail: bool = False):
    """Per-row NLL (float64) of the terms the offsets touch: kind + time (owned rows), censoring (censored rows) and the
    counterparty gate (owned rows). sh = None scores the base. With detail, also returns P(an owned event of each kind
    before the horizon) (N, 2) and the gate log-probs of the owned rows."""
    N = len(C["typ"])
    off, act = sh.offsets(C["role"], C["plan"], C["nown"], C["pk"], C["pc"]) if sh is not None else (None, None)
    logpk, bins = C["logpk"], C["bins"].reshape(2 * N, 64)
    if off is not None:
        logpk, bins = apply_kind(logpk, off, act), apply_time(bins, off)
    lp = torch.log_softmax(bins, -1).double().view(N, 2, 64)  # float32 log_softmax, as Model.time_law
    p = lp.exp()
    logZ = torch.log((p * C["a"]).sum(-1))  # (N, 2)
    b = C["b_r"][:, None, None].expand(N, 2, 1)
    logf = lp.gather(2, b)[..., 0] + C["c_r"] - logZ
    logS = torch.log((p * C["s_r"]).sum(-1).clamp(min=1e-300)) - logZ
    own, cen = C["typ"] == 1, (C["typ"] == 2) | (C["typ"] == 3)
    tk = C["tk"][:, None]
    z = torch.zeros(N, dtype=torch.float64, device=logpk.device)
    nll = torch.where(own, -logpk.gather(1, tk)[:, 0].double() - logf.gather(1, tk)[:, 0], z)
    nll = nll + torch.where(cen, -torch.logsumexp(logpk.double() + logS, -1), z)
    oi = C["own_idx"]
    g = C["g"] if off is None else apply_gate(C["g"], off[oi], C["tk"][oi])
    lg = masked_gate_logp(g, C["ent_any"], C["ven_any"])
    nll = nll.index_add(0, oi, -lg.gather(1, C["cls"][:, None])[:, 0].double())
    if not detail:
        return nll
    Sh = ((p * C["s_h"]).sum(-1) / logZ.exp()).clamp(0, 1)
    occ = logpk.double().exp() * (1 - Sh)
    return nll, occ, lg


def agent_mean(nll: torch.Tensor, agent: torch.Tensor):
    """(agent ids, mean shallow-row NLL of each agent)."""
    ua, inv = torch.unique(agent, return_inverse=True)
    s = torch.zeros(len(ua), dtype=torch.float64, device=nll.device).index_add(0, inv, nll)
    n = torch.zeros(len(ua), dtype=torch.float64, device=nll.device).index_add(0, inv, torch.ones_like(nll))
    return ua, s / n


def fit(C: dict, blocks, lam: float, dev="cuda", max_iter=500) -> Shallow:
    """Penalized MLE from zero: sum of row NLL + lam / 2 x sum of squared enabled offsets, by L-BFGS."""
    sh = Shallow(blocks).to(dev)
    opt = torch.optim.LBFGS(sh.parameters(), lr=1.0, max_iter=max_iter, tolerance_grad=1e-6, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def objective():
        return row_nll(sh, C).sum() + 0.5 * lam * sh.penalty().double()

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    loss = objective()
    loss.backward()
    gmax = max(float((p.grad * sh.cols).abs().max()) for p in sh.parameters())
    n_iter = opt.state[next(iter(sh.parameters()))].get("n_iter", -1)
    sh.fit_info = {"objective": float(loss), "max_grad": gmax, "lam": lam, "blocks": list(blocks),
                   "n_rows": int(len(C["typ"])), "n_iter": int(n_iter)}
    opt.zero_grad(set_to_none=True)
    assert np.isfinite(sh.fit_info["objective"]) and np.isfinite(gmax), "non-finite fit"
    return sh


def cluster_se(d: np.ndarray, groups: np.ndarray) -> float:
    """SE of mean(d) with whole groups as clusters."""
    _, inv = np.unique(groups, return_inverse=True)
    gs = np.bincount(inv, d - d.mean())
    return float(np.sqrt((gs ** 2).sum()) / len(d))


def select(scores: dict, groups: np.ndarray) -> dict:
    """scores: config -> per-agent held-out score (same agent order); configs are "zero" and (variant, lam).
    Returns the per-variant penalty and the primary variant (the 1-SE rules in the module docstring)."""
    out = {"variant_lam": {}, "se_rule": {}}
    for v in VARIANTS:
        cand = [(v, lam) for lam in LAMBDAS]
        best = min(cand, key=lambda c: scores[c].mean())
        ok = [c for c in cand if scores[c].mean() - scores[best].mean() <= cluster_se(scores[c] - scores[best], groups)]
        out["variant_lam"][v] = max(ok, key=lambda c: c[1])[1]
    pick = {"zero": scores["zero"], **{v: scores[(v, out["variant_lam"][v])] for v in VARIANTS}}
    best = min(pick, key=lambda k: pick[k].mean())
    for k in ORDER:
        out["se_rule"][k] = {"score": float(pick[k].mean()), "minus_best": float(pick[k].mean() - pick[best].mean()),
                             "se": cluster_se(pick[k] - pick[best], groups)}
    out["best"] = best
    out["primary"] = next(k for k in ORDER if out["se_rule"][k]["minus_best"] <= out["se_rule"][k]["se"])
    return out


def readouts(C: dict, shs: dict, store, agents_split) -> dict:
    """Teacher-forced readouts on one split's cache for the base ("B") and each fitted variant:
    - agent-mean shallow-row NLL by role x band and its change vs B;
    - the summed change per owned event of the role (= the known-plan NLL change; deeper rows are unchanged);
    - per role x band x depth 0-3, on rows followed by an owned event (n, model, observed):
      - P(OUT | an owned event before the horizon);
      - P(REPEAT | next kind)."""
    agent = C["agent"].cpu().numpy()
    n_own = {a: int((store.kind[store.off[a]:store.off[a + 1]] != IN_AGENT).sum()) for a in np.unique(agent)}
    N_agent = np.array([n_own[a] for a in agent])
    role = C["role"].cpu().numpy()
    owned_events = {r: sum(int((store.kind[store.off[a]:store.off[a + 1]] != IN_AGENT).sum())
                           for a in agents_split if store.role[a] == r and store.origin[a] < store.obs_end[a]) for r in (0, 1)}
    own = (C["typ"] == 1).cpu().numpy()
    tk, nown, cls = C["tk"].cpu().numpy(), C["nown"].cpu().numpy(), C["cls"].cpu().numpy()
    oi = C["own_idx"].cpu().numpy()
    res, base_nll = {}, None
    for name, sh in [("B", None)] + list(shs.items()):
        with torch.no_grad():
            nll, occ, lg = row_nll(sh, C, detail=True)
        nll = nll.cpu().numpy()
        base_nll = nll if base_nll is None else base_nll
        p_out = (occ[:, 0] / occ.sum(1).clamp(min=1e-300)).cpu().numpy()
        p_rep = lg[:, 0].exp().cpu().numpy()
        r_ = {}
        for ro, rn in ((0, "normal"), (1, "scam")):
            d = {"delta_per_owned_event": float((nll - base_nll)[role == ro].sum() / max(owned_events[ro], 1))}
            for bn, lo, hi in BANDS:
                sel = (role == ro) & (N_agent >= lo) & (N_agent < hi)
                if not sel.any():
                    continue
                ua, inv = np.unique(agent[sel], return_inverse=True)
                am, am0 = np.bincount(inv, nll[sel]) / np.bincount(inv), np.bincount(inv, base_nll[sel]) / np.bincount(inv)
                cell = {"agents": int(len(ua)), "agent_mean_nll": float(am.mean()), "delta_vs_B": float((am - am0).mean())}
                for dep in range(4):
                    s = sel & own & (nown == dep)
                    if s.sum() >= 20:
                        cell[f"d{dep}_out"] = (int(s.sum()), float(p_out[s].mean()), float((tk[s] == 0).mean()))
                    for kk, kn in ((0, "out"), (1, "in")):
                        so = np.flatnonzero(s[oi] & (tk[oi] == kk))
                        if len(so) >= 20:
                            cell[f"d{dep}_{kn}_repeat"] = (int(len(so)), float(p_rep[so].mean()), float((cls[so] == 0).mean()))
                d[bn] = cell
            r_[rn] = d
        res[name] = r_
    return res


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def with_offsets(pilot, base_state: dict, sh: Shallow, variant: str, store) -> Model:
    """the base state plus fitted offsets, loaded strictly into the shallow config of the variant."""
    name = f"lpw_plan_prog_shallow_{variant}"
    m = Model(CONFIGS[name], store.n_venues, store.prof_vec(0).shape[0], torch.as_tensor(store.venue_sc))
    state = dict(base_state)
    state.update({f"shallow.{k}": t.detach().cpu() for k, t in sh.state_dict().items()})
    m.load_state_dict(state, strict=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/p1_r0/2018/lpw_plan_prog_rp")
    ap.add_argument("--pilot", default="data/pilot_2018")
    ap.add_argument("--out", default="runs/p1_shallow/2018")
    ap.add_argument("--limit-agents", type=int, default=0, help="smoke runs only: this many train / val agents")
    a = ap.parse_args()
    t_start = time.time()
    out, pilot, base = Path(a.out), Path(a.pilot), Path(a.base)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *x: print(f"[{time.time() - t_start:6.0f}s]", *x, flush=True)
    st = build_store(pilot)
    views = Views(st)
    model = load(pilot, BASE_MODEL, base, st)
    tr, va = st.agents("train"), st.agents("val")
    if a.limit_agents:
        rng = np.random.default_rng(0)
        tr, va = np.sort(rng.choice(tr, a.limit_agents, replace=False)), np.sort(rng.choice(va, a.limit_agents, replace=False))
    ag = pl.read_parquet(pilot / "agents.parquet")
    gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    gid = np.array([gid_of[int(x)] for x in st.node_id])
    Ctr = to(build_cache(model, st, tr, views), "cuda")
    Cva = to(build_cache(model, st, va, views), "cuda")
    log(f"cache: train {len(Ctr['typ'])} rows / {len(torch.unique(Ctr['agent']))} agents, val {len(Cva['typ'])} rows")

    # folds of the training groups that have shallow rows
    agent_tr = Ctr["agent"].cpu().numpy()
    ug = np.unique(gid[np.unique(agent_tr)])
    np.random.default_rng(0).shuffle(ug)
    fold_of_group = {g: i % N_FOLDS for i, g in enumerate(ug)}
    fold_row = torch.as_tensor(np.array([fold_of_group[gid[x]] for x in agent_tr]), device="cuda")
    configs = ["zero"] + [(v, lam) for v in VARIANTS for lam in LAMBDAS]
    scores = {c: [] for c in configs}
    held_agents, cv_fits = [], []
    for f in range(N_FOLDS):
        Cfit, Cho = subset(Ctr, fold_row != f), subset(Ctr, fold_row == f)
        with torch.no_grad():
            ua, s0 = agent_mean(row_nll(None, Cho), Cho["agent"])
        held_agents.append(ua.cpu().numpy())
        scores["zero"].append(s0.cpu().numpy())
        for v in VARIANTS:
            for lam in LAMBDAS:
                sh = fit(Cfit, VARIANTS[v], lam)
                with torch.no_grad():
                    _, s = agent_mean(row_nll(sh, Cho), Cho["agent"])
                scores[(v, lam)].append(s.cpu().numpy())
                cv_fits.append({"fold": f, "variant": v, **sh.fit_info, "heldout": float(s.mean()), "heldout_zero": float(s0.mean())})
                log(f"fold {f} {v} lam {lam:g}: held-out {float(s.mean()):.5f} (zero {float(s0.mean()):.5f}), "
                    f"iters {sh.fit_info['n_iter']}, max grad {sh.fit_info['max_grad']:.1e}")
    held = np.concatenate(held_agents)
    scores = {c: np.concatenate(v) for c, v in scores.items()}
    sel = select(scores, gid[held])
    log("selection:", json.dumps(sel))

    base_state = torch.load(base / "model.pt", map_location="cpu")
    finals, ckpts = {}, {}
    for v in VARIANTS:
        sh = fit(Ctr, VARIANTS[v], sel["variant_lam"][v])
        finals[v] = sh
        m2 = with_offsets(pilot, base_state, sh, v, st)
        d = out / v
        d.mkdir(exist_ok=True)
        with atomic_path(d / "model.pt") as t:
            torch.save(m2.state_dict(), t)
        info = {"model": m2.cfg.name, "base": str(base), "base_sha256": sha256(base / "model.pt"), "fit": sh.fit_info}
        write_text(d / "shallow.json", json.dumps(info, indent=1))
        ckpts[v] = {"dir": str(d), "model": m2.cfg.name, "sha256": sha256(d / "model.pt"), **sh.fit_info}
        log(f"final {v} lam {sh.fit_info['lam']:g}: objective {sh.fit_info['objective']:.2f}, iters {sh.fit_info['n_iter']}, "
            f"max grad {sh.fit_info['max_grad']:.1e}")
    res = {"base": str(base), "base_sha256": sha256(base / "model.pt"), "pilot": str(pilot), "limit_agents": a.limit_agents,
           "cv_scores": {("zero" if c == "zero" else f"{c[0]}@{c[1]:g}"): float(s.mean()) for c, s in scores.items()},
           "cv_fits": cv_fits, "selection": sel, "checkpoints": ckpts,
           "val": readouts(Cva, finals, st, va), "train": readouts(Ctr, finals, st, tr), "sec": round(time.time() - t_start)}
    write_text(out / "fit.json", json.dumps(res, indent=1))
    log(f"primary variant: {sel['primary']}; wrote {out / 'fit.json'}")


if __name__ == "__main__":
    main()
