"""Drift screen v2 (reports/p1_drift_screen.md): arms B (untouched lpw_plan_prog base), C (clean partial-backbone
fine-tune) and R (clean + shift consistency J), fitted by `python -m lpw.p1.drift_ft` into ROOT/{C,R}/stepNNNN.

  eval ARM STEP  likelihood rows (exact prefixes, known-plan AND unknown-plan full NLL, on val and on the reported-only
                 retention subset = the 20% training-group mask anchor.split_groups seed 0) and shift forecasts on
                 the real val items, for one checkpoint -> ROOT/eval/ARM_STEP.npz, fingerprinted by the checkpoint's
                 sha256 (B = arm "B", step 0; C / R step 0 are B and reuse its file)
  select         val likelihood budget, C by unknown-plan val NLL, R by shift ratio, both among feasible snapshots
                 -> ROOT/select.json
  gen ARM        for the selected checkpoint of ARM (B, C, R): oracle pilot generation (G, D) and prior-mode
                 full-val-population generation (safety, fidelity) -> ROOT/gen/
  report         mechanism (S, D), pilot generation (G), safety (P), fidelity (F) and the gates -> ROOT/screen.{json,txt}

usage: screen.py PILOT BASE_DIR ROOT CMD [ARM [STEP]]"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import wasserstein_distance as w1

from lpw.p1.anchor import corrupt_view, merge_corrupted
from lpw.p1.atomic import atomic_path, write_text
from lpw.p1.batch import CTX, Views, collate, exact_specs, pack
from lpw.p1.drift_ft import SNAPS
from lpw.p1.fidelity import Sample, compare, drop_largest_group, groups_for, real_frames
from lpw.p1.generate import generate
from lpw.p1.drift_ft import file_sha
from lpw.p1.model import trunc_mog_mean
from lpw.p1.plan import COUNT_EDGES, N_PLAN_N, PlanPrior
from lpw.p1.store import IN_AGENT, OUT, PERIPHERY, TIER_VENUE, VENUE, build_store, derived
from lpw.p1.timelaw import EDGES
from lpw.p1.train import load, loader, to_dev

NBOOT, LMAX, S_RED, D_RED, G_RED = 1000, 0.02, 0.25, 0.25, 0.15
FID_NORMAL, FID_SCAM, FID_BOOT, RUN_X = 0.10, 1.05, 200, 10
RETENTION, COVER_TOL = 0.2, 0.05
FID_GROUPS = ("normal", "scam")
PILOT_REPS, PRIOR_REPS, PILOT_CAP, PRIOR_CAP, GEN_SEED, PLAN_SEED = 3, 3, 150, 10_000, 7, 101
DEPTHS, DD = (3, 9, 29, 99), (9, 29, 99)
SHIFTS = {"dt+": (0.3, 0.0), "dt-": (-0.3, 0.0), "da+": (0.0, 0.5), "da-": (0.0, -0.5)}
PLAN300 = int(np.searchsorted(COUNT_EDGES, 300))  # count bucket index of >= 300 owned events
EV = ("kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal")
GEN_RECIPE = {"pilot_reps": PILOT_REPS, "prior_reps": PRIOR_REPS, "pilot_cap": PILOT_CAP, "prior_cap": PRIOR_CAP, "gen_seed": GEN_SEED,
              "plan_seed": PLAN_SEED, "population": "val", "version": 2}
EVAL_VERSION = 2


def arr_sha(*arrays) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()
X = None  # bin log-midpoints on the GPU (forecast)


class Ctx:
    def __init__(self, pilot, base, root):
        self.pilot, self.base, self.root = Path(pilot), Path(base), Path(root)
        self.st = st = build_store(self.pilot)
        self.views = Views(st)
        self.va = st.agents("val")
        ag = pl.read_parquet(self.pilot / "agents.parquet")
        gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
        self.gid = np.array([gid_of[int(x)] for x in st.node_id])  # per store agent
        from lpw.p1.anchor import split_groups
        self.ret = split_groups(st, self.pilot, RETENTION)[1]  # reported-only retention subset (in B's and C/R's training data)

    def eval_pop_sha(self):
        if not hasattr(self, "_eval_pop"):
            it = shift_items(self)
            self._eval_pop = arr_sha(self.va, self.ret, np.array(it, np.int64).reshape(-1, 3))
        return self._eval_pop

    def gen_pop_sha(self):
        return arr_sha(self.va, pilot_agents(self))

    def ckpt(self, arm, step):
        return self.base if arm == "B" or step == 0 else self.root / arm / f"step{step:04d}"

    def model(self, arm, step):
        m = load(self.pilot, "lpw_plan_prog", self.ckpt(arm, step), self.st)
        m.plan_inference = True
        return m


def boot_weights(ctx, agents, seed):
    """NBOOT x len(agents) multiplicities from whole-group resampling (store agent indices)."""
    g = ctx.gid[agents]
    ug, inv = np.unique(g, return_inverse=True)
    draws = np.random.default_rng(seed).integers(0, len(ug), (NBOOT, len(ug)))
    cnt = np.zeros((NBOOT, len(ug)), np.int64)
    np.add.at(cnt, (np.arange(NBOOT)[:, None], draws), 1)
    return cnt[:, inv]


# ---- likelihood and forecasts ----------------------------------------------------------------------------------------
@torch.no_grad()
def rows_nll(ctx, m, agents, known):
    """exact full NLL per prefix (known or unknown plan) and the prefix's store agent."""
    m.plan_inference = known
    batches = pack(exact_specs(ctx.st, agents), ctx.views)
    full, who = [], []
    for bi, b in enumerate(loader(ctx.views, batches)):
        sp = batches[bi]
        b = to_dev(b, "cuda")
        t = m.terms(b)
        kt = (t["kind"] + t["time"] + t["cens"]).double()
        full.append(kt.index_add(0, t["o_row"], (t["cp"] + t["amt"]).double()).cpu().numpy())
        who.append(np.array([sp[j][0] for j in b["row_b"].cpu().numpy()], np.int64))
    m.plan_inference = True
    return np.concatenate(full), np.concatenate(who)


@torch.no_grad()
def forecast(ctx, m, vs, ks, shift=(0.0, 0.0)):
    """per (view, prefix length k), known plan: (Elog10 positive-finite gap of the next owned event, mean of the
    truncated log10-amount distribution at the real next event if it is owned, else nan). As
    scripts/p1_anchor/rescreen.py, except the amount forecast is the truncated mixture's mean (model.trunc_mog_mean)
    rather than the clamped untruncated mean."""
    global X
    if X is None:
        X = torch.as_tensor(np.log10(np.sqrt(EDGES[:-1] * EDGES[1:])), device="cuda")
    st = ctx.st
    m.plan_inference = True
    el, am = [], []
    for c0 in range(0, len(vs), 128):
        cv, ck = vs[c0:c0 + 128], ks[c0:c0 + 128]
        specs = [(max(0, k - CTX), np.array([k - max(0, k - CTX)])) for k in ck]
        b = collate(cv, specs, st.n_venues, st.w_start)
        if shift != (0.0, 0.0):
            b = merge_corrupted(b, collate([corrupt_view(v, *shift) for v in cv], specs, st.n_venues, st.w_start))
        b = to_dev({k: torch.from_numpy(x) for k, x in b.items()}, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            Hk, Ho = m.encode(b)
        rb, rj, t0 = b["row_b"], b["row_j"], b["row_t0"]
        logpk, law, _ = m.kind_time(Hk.float()[rb, rj], Ho.float()[rb, rj], t0, **m.row_extras(b))
        n = len(rb)
        bp = (logpk.double().exp()[:, :, None] * law.bin_probs().view(n, 2, 64)).sum(1)
        q = bp[:, 1:63] / bp[:, 1:63].sum(1, keepdim=True)
        el.append((q * X).sum(1).cpu().numpy())
        m._capture = {}
        t = m.terms(b)
        ap = m._capture["amt_ap_final"]
        m._capture = None
        mean = np.full(n, np.nan)
        mean[t["o_row"].cpu().numpy()] = trunc_mog_mean(ap, m.cfg.k_mix).cpu().numpy()
        am.append(mean)
    return np.concatenate(el), np.concatenate(am)


def depth_k(v, d):
    """prefix length ending at the d-th owned event, if the next event exists and is owned; else None."""
    c = np.cumsum(v["kind"] != IN_AGENT)
    if len(c) == 0 or c[-1] < d:
        return None
    k = int(np.searchsorted(c, d)) + 1
    return k if k < len(v["ts"]) and v["kind"][k] != IN_AGENT else None


def shift_items(ctx):
    """real val prefixes of agents with plan >= 10 owned events at depths DEPTHS: (store agent, depth, k)."""
    items = []
    for a in ctx.va:
        if ctx.st.plan_n[a] < 5:
            continue
        v = ctx.views(int(a))
        for d in DEPTHS:
            k = depth_k(v, d)
            if k is not None:
                items.append((int(a), d, k))
    return items


def cmd_eval(ctx, arm, step):
    m = ctx.model(arm, step)
    out = {"sha": np.array(file_sha(ctx.ckpt(arm, step) / "model.pt")), "arm": np.array(arm), "step": np.array(step)}
    for sname, agents in (("retention", ctx.ret), ("val", ctx.va)):
        for mode in ("known", "unknown"):
            nll, who = rows_nll(ctx, m, agents, mode == "known")
            out[f"nll_{sname}_{mode}"] = nll
            out[f"who_{sname}"] = who
    items = shift_items(ctx)
    vs, ks = [ctx.views(a) for a, _, _ in items], [k for _, _, k in items]
    F = {s: forecast(ctx, m, vs, ks, sh) for s, sh in SHIFTS.items()}
    out["e_tt"] = (F["dt+"][0] - F["dt-"][0]) / 0.6
    out["e_at"] = (F["dt+"][1] - F["dt-"][1]) / 0.6
    out["e_ta"] = (F["da+"][0] - F["da-"][0]) / 1.0
    out["e_aa"] = (F["da+"][1] - F["da-"][1]) / 1.0
    out["item_agent"] = np.array([a for a, _, _ in items], np.int64)
    out["item_depth"] = np.array([d for _, d, _ in items], np.int64)
    for k in ("e_tt", "e_at", "e_ta", "e_aa"):  # every item's next event exists and is owned: all forecasts must be finite
        assert np.isfinite(out[k]).all(), f"non-finite {k} forecast at {arm} step {step}"
    out["pop_sha"] = np.array(ctx.eval_pop_sha())
    out["version"] = np.array(EVAL_VERSION)
    (ctx.root / "eval").mkdir(parents=True, exist_ok=True)
    with atomic_path(ctx.root / "eval" / f"{arm}_{step:04d}.npz") as t, open(t, "wb") as f:
        np.savez(f, **out)
    print(f"eval {arm} step {step}: val unknown {out['nll_val_unknown'].mean():.4f} known {out['nll_val_known'].mean():.4f}, "
          f"retention unknown {out['nll_retention_unknown'].mean():.4f}")


def load_eval(ctx, arm, step):
    E = dict(np.load(ctx.root / "eval" / f"{'B' if step == 0 else arm}_{step:04d}.npz"))
    assert str(E["sha"]) == file_sha(ctx.ckpt(arm, step) / "model.pt"), f"eval of {arm} step {step} is not of the current checkpoint"
    assert int(E["version"]) == EVAL_VERSION and str(E["pop_sha"]) == ctx.eval_pop_sha(), f"eval of {arm} step {step}: other population / recipe"
    for sname, agents in (("retention", ctx.ret), ("val", ctx.va)):  # every prefix of the registered population, once
        n_exp = sum(ctx.st.n_events(int(a)) + 1 for a in agents if ctx.st.origin[a] < ctx.st.obs_end[a])
        assert len(E[f"who_{sname}"]) == n_exp and np.array_equal(np.unique(E[f"who_{sname}"]),
                                                                   np.unique([a for a in agents if ctx.st.origin[a] < ctx.st.obs_end[a]]))
    return E


# ---- selection -------------------------------------------------------------------------------------------------------
def cells(ctx, who):
    st = ctx.st
    return {"all": np.ones(len(who), bool), "normal": st.role[who] == 0, "scam": st.role[who] == 1, "plan>=300": st.plan_n[who] >= PLAN300}


def budget(ctx, E, EB, sname="val", boot=False):
    """{sname_mode: {cell: arm - B}} per-prefix NLL deltas (feasible when every val one is <= LMAX); boot=True adds
    whole-group bootstrap 95% intervals and the cell's agent / group counts (reported, not gated)."""
    d, ci = {}, {}
    assert np.array_equal(E[f"who_{sname}"], EB[f"who_{sname}"])
    who = EB[f"who_{sname}"]
    cs = cells(ctx, who)
    for mode in ("known", "unknown"):
        a, b = E[f"nll_{sname}_{mode}"], EB[f"nll_{sname}_{mode}"]
        d[f"{sname}_{mode}"] = {c: float(a[mm].mean() - b[mm].mean()) for c, mm in cs.items()}
        if boot:
            for c, mm in cs.items():
                g = ctx.gid[who[mm]]
                ug, inv = np.unique(g, return_inverse=True)
                gs, gn = np.bincount(inv, a[mm] - b[mm], len(ug)), np.bincount(inv, minlength=len(ug)).astype(float)
                dr = np.random.default_rng(4).integers(0, len(ug), (NBOOT, len(ug)))
                bs = gs[dr].sum(1) / gn[dr].sum(1)
                ci[f"{sname}_{mode}/{c}"] = {"lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                                              "agents": int(len(np.unique(who[mm]))), "groups": int(len(ug)), "prefixes": int(mm.sum())}
    return (d, ci) if boot else d


def pick(steps, scores, feasible):
    """(step, score) of the lowest score among feasible snapshots, ties to the earlier step; steps ascending, and the
    first (step 0 = B) must be feasible."""
    assert list(steps) == sorted(steps) and feasible[0], "step 0 must be first and feasible"
    best = None
    for s, sc, f in zip(steps, scores, feasible):
        if f and (best is None or sc < best[1]):
            best = (s, sc)
    return best


RECIPE_KEYS_DIFFER = ("arm", "w_j", "calibration", "trace", "sec", "snap_sha", "skipped_steps")


def mean_abs_normal(ctx, E, k):
    nm = ctx.st.role[E["item_agent"]] == 0
    assert np.isfinite(E[k][nm]).all(), f"non-finite {k} forecast"
    return float(np.abs(E[k][nm]).mean())


def cmd_select(ctx):
    EB = load_eval(ctx, "B", 0)
    sB = {k: mean_abs_normal(ctx, EB, k) for k in ("e_tt", "e_aa")}
    base_sd = torch.load(ctx.base / "model.pt", map_location="cpu")
    ft = {arm: json.loads((ctx.root / arm / "ft.json").read_text()) for arm in ("C", "R")}
    rc = {arm: {k: v for k, v in f.items() if k not in RECIPE_KEYS_DIFFER} for arm, f in ft.items()}
    assert rc["C"] == rc["R"], "C and R recipes differ beyond J"
    assert ft["C"]["w_j"] == 0.0 and ft["R"]["w_j"] > 0 and ft["R"]["calibration"] is not None
    res = {"B_shift": sB, "arms": {}}
    for arm in ("C", "R"):
        sd0 = torch.load(ctx.root / arm / "step0000" / "model.pt", map_location="cpu")
        assert sd0.keys() == base_sd.keys() and all(torch.equal(sd0[k], base_sd[k]) for k in sd0), f"{arm} step 0 is not B"
        f = ft[arm]
        assert Path(f["base"]).resolve() == ctx.base.resolve() and Path(f["pilot"]).resolve() == ctx.pilot.resolve()
        assert f["holdout"] == 0 and not f["holdout_agent_ids"], "the registered recipe adapts every training group"
        assert sorted(int(x) for x in f["snap_sha"]) == list(SNAPS)
        rows = []
        for s in SNAPS:
            assert f["snap_sha"][str(s)] == file_sha(ctx.ckpt(arm, s) / "model.pt") or s == 0
            E = load_eval(ctx, arm, s)
            d, ci = budget(ctx, E, EB, "val", boot=True)
            ret = budget(ctx, E, EB, "retention")
            feas = all(v <= LMAX for c in d.values() for v in c.values())
            sh = {k: mean_abs_normal(ctx, E, k) for k in ("e_tt", "e_aa")}
            ratio = max(sh["e_tt"] / sB["e_tt"], sh["e_aa"] / sB["e_aa"])
            val_u = float(E["nll_val_unknown"].mean())
            score = val_u if arm == "C" else ratio
            if not np.isfinite(score):
                raise RuntimeError(f"{arm} step {s}: non-finite selection score")
            rows.append({"step": s, "feasible": feas, "budget": d, "budget_ci": ci, "retention": ret, "shift": sh,
                         "shift_ratio": ratio, "val_unknown": val_u, "val_known": float(E["nll_val_known"].mean()), "score": score})
        best = pick([r_["step"] for r_ in rows], [r_["score"] for r_ in rows], [r_["feasible"] for r_ in rows])
        res["arms"][arm] = {"selected": best[0], "score": best[1], "snapshots": rows, "w_j": f["w_j"],
                            "skipped_steps": f["skipped_steps"], "sha": f["snap_sha"][str(best[0])] if best[0] else file_sha(ctx.base / "model.pt")}
    write_text(ctx.root / "select.json", json.dumps(res, indent=1))
    for arm, r in res["arms"].items():
        print(f"select {arm}: step {r['selected']} (score {r['score']:.4f}); skipped updates {r['skipped_steps']}")
        for row in r["snapshots"]:
            worst = max(v for c in row["budget"].values() for v in c.values())
            print(f"  step {row['step']:4d}: feasible {row['feasible']} (worst val delta {worst:+.4f}), val unknown {row['val_unknown']:.4f}, "
                  f"retention unknown {row['retention']['retention_unknown']['all']:+.4f}, "
                  f"shift e_tt {row['shift']['e_tt']:.4f} e_aa {row['shift']['e_aa']:.4f} ratio {row['shift_ratio']:.3f}")


def selected(ctx):
    sel = json.loads((ctx.root / "select.json").read_text())
    steps = {"B": 0, "C": sel["arms"]["C"]["selected"], "R": sel["arms"]["R"]["selected"]}
    for arm in ("C", "R"):
        assert sel["arms"][arm]["sha"] == file_sha(ctx.ckpt(arm, steps[arm]) / "model.pt"), f"{arm}: selected checkpoint changed"
    return steps, sel


def check_gen(ctx, arm, step):
    g = json.loads((ctx.root / "gen" / f"{arm}_gen.json").read_text())
    assert g["step"] == step and g["sha"] == file_sha(ctx.ckpt(arm, step) / "model.pt"), f"{arm}: generation is not of the selected checkpoint"
    assert g["recipe"] == GEN_RECIPE and g["pop_sha"] == ctx.gen_pop_sha(), f"{arm}: generation recipe / population differs"


# ---- generation ------------------------------------------------------------------------------------------------------
def pilot_agents(ctx):
    return ctx.va[np.flatnonzero(ctx.st.plan_n[ctx.va] >= 6)]


def conds_of(st, agents, plan_n, plan_a):
    return [dict(role=int(st.role[x]), cohort=int(st.cohort[x]), origin=int(st.origin[x]), prof=st.prof_vec(x), b0=float(st.b0[x]),
                 age_days=float(st.prof_raw[x, 17]), plan_n=int(n), plan_a=int(a)) for x, n, a in zip(agents, plan_n, plan_a)]


def prior_plans(ctx):
    """PRIOR_REPS joint plan draws for the val population (as lpw.p1.drift_exp prior mode); identical for every arm."""
    st, va = ctx.st, ctx.va
    prior, rng = PlanPrior(st), np.random.default_rng(PLAN_SEED)
    prof = np.stack([st.prof_vec(x) for x in va])
    return [prior.sample(st.role[va], st.cohort[va], prof, rng) for _ in range(PRIOR_REPS)]


def cmd_gen(ctx, arm):
    from lpw.p1.run import generate_replicas, write_replicas
    steps = {"B": 0} if arm == "B" else selected(ctx)[0]  # B needs no selection (measured before the R run)
    m = ctx.model(arm, steps[arm])
    st = ctx.st
    gd = ctx.root / "gen"
    gd.mkdir(parents=True, exist_ok=True)
    sa = pilot_agents(ctx)
    conds = conds_of(st, sa, st.plan_n[sa], st.plan_a[sa])
    (_, ev), gi = generate(m, conds * PILOT_REPS, st, seed=GEN_SEED, max_events=PILOT_CAP, log=lambda *x: None)
    with atomic_path(gd / f"{arm}_pilot.parquet") as t:
        ev.write_parquet(t)
    plans = prior_plans(ctx)
    pops = [conds_of(st, ctx.va, pn, pa) for pn, pa in plans]
    reps, info = generate_replicas(m, pops, st, seed=GEN_SEED, log=lambda *x: None)
    write_replicas(reps, gd / f"{arm}_prior.parquet")
    with atomic_path(gd / f"{arm}_prior.plans.npz") as t, open(t, "wb") as f:
        np.savez(f, plan_n=np.stack([p[0] for p in plans]), plan_a=np.stack([p[1] for p in plans]))
    write_text(gd / f"{arm}_gen.json", json.dumps({"step": steps[arm], "sha": file_sha(ctx.ckpt(arm, steps[arm]) / "model.pt"), "recipe": GEN_RECIPE,
                                                  "pop_sha": ctx.gen_pop_sha(), "pilot_capped": gi["capped"], "prior_capped": info["capped"]}, indent=1))
    print(f"gen {arm} (step {steps[arm]}): pilot capped {gi['capped']}, prior capped {info['capped']}")


# ---- report ----------------------------------------------------------------------------------------------------------
def syn_view(st, a, v):
    """generated stream of store agent a as a Views-compatible dict (as scripts/p1_anchor/rescreen.py)."""
    ne = int(v["ent"].max()) + 1 if len(v["ent"]) else 0
    tier, sc, ven = np.zeros(ne, np.int8), np.ones(ne, bool), np.full(ne, -1, np.int16)
    seen = set()
    for c, e, vi in zip(v["cls"], v["ent"], v["venue"]):
        if e in seen:
            continue
        seen.add(e)
        if c == VENUE:
            tier[e], sc[e], ven[e] = TIER_VENUE, bool(st.venue_sc[vi]), vi
        else:
            tier[e], sc[e], ven[e] = PERIPHERY, True, -1
    v.update(e_tier=tier, e_sc=sc, e_venue=ven, origin=int(st.origin[a]), obs_end=int(st.w_end), role=int(st.role[a]),
             cohort=int(st.cohort[a]), prof=st.prof_vec(a), plan_n=int(st.plan_n[a]), plan_a=int(st.plan_a[a]))
    n = len(v["ts"])
    v["bal_end"] = float(v["bal"][-1] + (v["eth"][-1] if v["kind"][-1] != OUT else -v["eth"][-1] - v["fee"][-1])) if n else float(st.b0[a])
    v.update(derived(v, st.w_start))
    return v


def split_streams(ev, n):
    ev = ev.sort("agent", "ts", maintain_order=True)
    a_col = ev["agent"].to_numpy()
    cuts = np.searchsorted(a_col, np.arange(n + 1))
    cols = {f: ev[f].to_numpy() for f in EV}
    return [{f: cols[f][cuts[i]:cuts[i + 1]] for f in EV} for i in range(n)]


def max_run(ts):
    """longest run of equal consecutive timestamps."""
    if len(ts) == 0:
        return 0
    br = np.flatnonzero(np.diff(ts) != 0)
    return int(np.diff(np.concatenate([[-1], br, [len(ts) - 1]])).max())


def real_train_max_run(st):
    best = 0
    for a in st.agents("train"):
        s, e = st.off[a], st.off[a + 1]
        k = st.kind[s:e]
        best = max(best, max_run(st.ts[s:e][k != IN_AGENT]))
    return best


def red(eC, eR, W):
    """mean-absolute reduction 1 - mean|e_R| / mean|e_C| and its whole-group bootstrap interval (W: NBOOT x n)."""
    aC, aR = np.abs(eC), np.abs(eR)
    rb = 1 - (W * aR).sum(1) / (W * aC).sum(1)
    return {"C": float(aC.mean()), "R": float(aR.mean()), "reduction": float(1 - aR.mean() / aC.mean()),
            "lo": float(np.quantile(rb, 0.025)), "hi": float(np.quantile(rb, 0.975))}


def cmd_report(ctx):
    st, va = ctx.st, ctx.va
    steps, sel = selected(ctx)
    W_END = float(st.w_end)
    RES = {"selected": steps, "w_j": sel["arms"]["R"]["w_j"], "shift": {}, "discrepancy": {}, "pilot": {}, "safety": {}, "fidelity": {}}
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"selected: C step {steps['C']}, R step {steps['R']} (w_J {RES['w_j']:.4g})")
    for arm in ("C", "R"):
        r = next(x for x in sel["arms"][arm]["snapshots"] if x["step"] == steps[arm])
        say(f"  {arm}: budget " + "; ".join(f"{k} " + " ".join(f"{c} {v:+.4f}" for c, v in d.items()) for k, d in r["budget"].items()))

    # (S) shift sensitivity, normal items, mean absolute
    E = {a: load_eval(ctx, a, s) for a, s in steps.items()}
    ia = E["B"]["item_agent"]
    assert all(np.array_equal(E[a]["item_agent"], ia) for a in E)
    nm = st.role[ia] == 0
    ua = np.unique(ia[nm])
    pos = {a: i for i, a in enumerate(ua)}
    Wa = boot_weights(ctx, ua, 1)
    for k in ("e_tt", "e_aa", "e_ta", "e_at"):
        RES["shift"][k] = {a: float(np.nanmean(np.abs(E[a][k][nm]))) for a in E}
    for k in ("e_tt", "e_aa"):
        ok = nm & np.isfinite(E["C"][k]) & np.isfinite(E["R"][k])
        RES["shift"][f"R_vs_C_{k}"] = r = red(E["C"][k][ok], E["R"][k][ok], Wa[:, [pos[a] for a in ia[ok]]])
        say(f"(S) mean |{k}| normal: B {RES['shift'][k]['B']:.4f} C {r['C']:.4f} R {r['R']:.4f}; R vs C reduction "
            f"{r['reduction']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

    # (G) pilot generation and (D) rollout discrepancy, oracle plans
    sa = pilot_agents(ctx)
    n = len(sa)
    role_s, plan_s = st.role[sa], st.plan_n[sa]
    origin = st.origin[sa].astype(np.float64)
    expo = np.maximum(W_END - origin, 1.0)
    rev = real_frames(st, sa)[1].sort("agent", "ts", maintain_order=True)
    _a = rev["agent"].to_numpy()
    _c = np.searchsorted(_a, np.arange(n + 1))
    _t, _m = rev["ts"].to_numpy().astype(np.float64), rev["amt"].to_numpy().astype(np.float64)
    real = [(_t[_c[i]:_c[i + 1]], _m[_c[i]:_c[i + 1]]) for i in range(n)]
    heavy = (role_s == 0) & (plan_s >= 7)
    CELLS = [(0, [6]), (0, [7])]

    def tau(S, k):
        return np.array([min((s[0][k - 1] - origin[i % n]) / expo[i % n], 1.0) if len(s[0]) >= k else 1.0 for i, s in enumerate(S)])

    RT = {k: tau(real, k) for k in (10, 30)}
    RSEG = [s[1][30:100] for s in real]

    def pilot_scores(ST, SSEG, w):
        wr = np.tile(w, PILOT_REPS).astype(np.float64)
        cs = []
        for ro, bks in CELLS:
            rm = (role_s == ro) & np.isin(plan_s, bks) & (w > 0)
            sm = np.tile(rm, PILOT_REPS)
            for k in (10, 30):
                cs.append(float(w1(RT[k][rm], ST[k][sm], w[rm], wr[sm])))
        xr, wa_r, xs, wa_s = [], [], [], []
        for j in np.flatnonzero(heavy & (w > 0)):
            if len(RSEG[j]):
                xr.append(RSEG[j]); wa_r.append(np.full(len(RSEG[j]), w[j] / len(RSEG[j])))
            reps = [SSEG[r * n + j] for r in range(PILOT_REPS) if len(SSEG[r * n + j])]
            for s in reps:
                xs.append(s); wa_s.append(np.full(len(s), w[j] / (len(s) * len(reps))))
        aw = float(w1(np.concatenate(xr), np.concatenate(xs), np.concatenate(wa_r), np.concatenate(wa_s))) if xs else float("nan")
        return float(np.mean(cs)), aw, cs

    Wg = boot_weights(ctx, sa, 2)
    real_items = [(j, d, depth_k(ctx.views(int(sa[j])), d)) for j in np.flatnonzero(heavy) for d in DD]
    real_items = [x for x in real_items if x[2] is not None]
    GENS, BOOT, DIS = {}, {}, {}
    for arm in ("B", "C", "R"):
        check_gen(ctx, arm, steps[arm])
    for arm in ("B", "C", "R"):
        raw = split_streams(pl.read_parquet(ctx.root / "gen" / f"{arm}_pilot.parquet"), n * PILOT_REPS)
        S = [(r["ts"].astype(np.float64), r["amt"].astype(np.float64)) for r in raw]
        ST = {k: tau(S, k) for k in (10, 30)}
        SSEG = [s[1][30:100] for s in S]
        pace, aw, cs = pilot_scores(ST, SSEG, np.ones(n, np.int64))
        hv = np.flatnonzero(np.tile(heavy, PILOT_REPS))
        reach = float(np.mean([len(S[i][0]) >= 100 for i in hv]))
        RES["pilot"][arm] = {"pace_score": pace, "cells": cs, "amount_w1": aw, "reach100": reach,
                             "reach100_real": float(np.mean([len(real[j][0]) >= 100 for j in np.flatnonzero(heavy)]))}
        BOOT[arm] = np.array([pilot_scores(ST, SSEG, Wg[b])[:2] for b in range(NBOOT)])
        say(f"(G) {arm}: pace {pace:.4f}, agent-weighted amount W1 {aw:.4f}, reach-100 {reach:.3f} (real {RES['pilot'][arm]['reach100_real']:.3f})")
        # (D): forecasts on the arm's own generated prefixes vs on the real prefix at the same owned-event depth
        m = ctx.model(arm, steps[arm])
        fr = forecast(ctx, m, [ctx.views(int(sa[j])) for j, _, _ in real_items], [k for _, _, k in real_items])
        assert np.isfinite(fr[0]).all() and np.isfinite(fr[1]).all(), f"{arm}: non-finite real-prefix forecast"
        sv, sk, sidx = [], [], []
        for p, (j, d, _) in enumerate(real_items):
            for r in range(PILOT_REPS):
                v = raw[r * n + j]
                if len(v["ts"]) == 0:
                    continue
                vv = syn_view(st, int(sa[j]), dict(v))
                k = depth_k(vv, d)
                if k is not None:
                    sv.append(vv); sk.append(k); sidx.append(p)
        fs = forecast(ctx, m, sv, sk) if sv else (np.zeros(0), np.zeros(0))
        assert np.isfinite(fs[0]).all() and np.isfinite(fs[1]).all(), f"{arm}: non-finite generated-prefix forecast"  # NaN = not reached only
        sidx = np.array(sidx, np.int64)
        out = {"reached": np.bincount(sidx, minlength=len(real_items))}  # replicas reaching the pair's depth
        for q, key in ((0, "D_t"), (1, "D_a")):
            num, cnt = np.zeros(len(real_items)), np.zeros(len(real_items))
            ok = np.isfinite(fs[q])
            np.add.at(num, sidx[ok], fs[q][ok]); np.add.at(cnt, sidx[ok], 1)
            has = (cnt > 0) & np.isfinite(fr[q])
            out[key] = np.where(has, num / np.maximum(cnt, 1) - fr[q], np.nan)
        DIS[arm] = out
        del m
        torch.cuda.empty_cache()
    for i, k in enumerate(("pace_score", "amount_w1")):
        c, r = RES["pilot"]["C"][k], RES["pilot"]["R"][k]
        diff = BOOT["C"][:, i] - BOOT["R"][:, i]
        RES["pilot"][f"R_vs_C_{k}"] = g = {"reduction": float(1 - r / c), "diff_lo": float(np.quantile(diff, 0.025)),
                                           "diff_hi": float(np.quantile(diff, 0.975))}
        say(f"  R vs C {k}: {c:.4f} -> {r:.4f}, reduction {g['reduction']:.3f}, C - R 95% [{g['diff_lo']:.4f}, {g['diff_hi']:.4f}]")
    # D on the common support: pairs with a value in every arm (B, C, R); each pair's discrepancy is the mean over the
    # replicas that reached the depth; per-depth coverage (share of real pairs with at least one reaching replica) is
    # reported and guarded (R not more than COVER_TOL below C or B at any depth)
    pa_ = np.array([sa[j] for j, _, _ in real_items])
    dep = np.array([d for _, d, _ in real_items])
    common = {key: np.all([np.isfinite(DIS[a][key]) for a in DIS], 0) for key in ("D_t", "D_a")}
    for arm in DIS:
        RES["discrepancy"][arm] = {"coverage": {str(d): float((DIS[arm]["reached"][dep == d] > 0).mean()) for d in DD},
                                   "replicas_reached_mean": float(DIS[arm]["reached"].mean() / PILOT_REPS)}
        for key in ("D_t", "D_a"):
            x, c = DIS[arm][key], common[key]
            RES["discrepancy"][arm][key] = {"mean_abs_common": float(np.abs(x[c]).mean()), "pairs_common": int(c.sum()),
                                            "mean_abs_own": float(np.nanmean(np.abs(x))), "pairs_own": int(np.isfinite(x).sum()),
                                            **{f"mean_abs_common_d{d}": float(np.abs(x[c & (dep == d)]).mean()) if (c & (dep == d)).any() else float("nan") for d in DD}}
    RES["discrepancy"]["real_pairs"] = len(real_items)
    ud = np.unique(pa_)
    Wd = boot_weights(ctx, ud, 3)
    posd = {a: i for i, a in enumerate(ud)}
    for key in ("D_t", "D_a"):
        c = common[key]
        RES["discrepancy"][f"R_vs_C_{key}"] = r = red(DIS["C"][key][c], DIS["R"][key][c], Wd[:, [posd[a] for a in pa_[c]]])
        r["pairs_common"] = int(c.sum())
        say(f"(D) mean |{key}| on {int(c.sum())} common pairs of {len(real_items)}: B {RES['discrepancy']['B'][key]['mean_abs_common']:.4f} "
            f"C {r['C']:.4f} R {r['R']:.4f}; R vs C reduction {r['reduction']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    say("    coverage by depth " + " | ".join(f"{a} " + " ".join(f"d{d} {RES['discrepancy'][a]['coverage'][str(d)]:.2f}" for d in DD) for a in DIS))

    # (P) safety and (F) fidelity on the prior-mode val population
    rmax = real_train_max_run(st)
    nmax = int(max(int((st.kind[st.off[a]:st.off[a + 1]] != IN_AGENT).sum()) for a in st.agents("train")))  # longest real-train stream
    RES["safety"]["real_train_max_run"] = rmax
    rag, rev_all = real_frames(st, va)
    realS = Sample(rag, rev_all, st.w_start)
    pre_age = rag.filter(pl.col("cohort") == 0)["age_days"].to_numpy()
    edges = np.quantile(pre_age, [0, 1 / 3, 2 / 3, 1])
    edges[-1] += 1
    groups = groups_for(rag, edges)
    groups_drop, big = drop_largest_group(groups, realS, ctx.gid[va])
    RES["fidelity"]["dropped_group"] = big
    plans0, EVS = None, {}
    for arm in ("B", "C", "R"):
        P = dict(np.load(ctx.root / "gen" / f"{arm}_prior.plans.npz"))
        if plans0 is None:
            plans0 = P
        assert all(np.array_equal(P[k], plans0[k]) for k in P), "arms used different prior plans"
        ev = pl.read_parquet(ctx.root / "gen" / f"{arm}_prior.parquet")
        EVS[arm] = ev
        sf = {"capped": [], "capped_finite_plan": [], "capped_open_plan": [], "over3U": [], "zero_plan_nonempty": [], "runs_over": [], "max_run": [], "over_real_max": [],
              "overU": [], "open_over_real_max_len": []}
        for r in range(PRIOR_REPS):
            streams = split_streams(ev.filter(pl.col("replica") == r).drop("replica"), len(va))
            N = np.array([len(s_["ts"]) for s_ in streams])
            pn = P["plan_n"][r]
            fin = pn < N_PLAN_N - 1
            U = np.where(fin, COUNT_EDGES[np.minimum(pn + 1, N_PLAN_N - 1)] - 1, np.iinfo(np.int64).max)
            runs = np.array([max_run(s_["ts"]) for s_ in streams])
            sf["capped"].append(int((N >= PRIOR_CAP).sum()))
            sf["capped_finite_plan"].append(int((fin & (N >= PRIOR_CAP)).sum()))
            sf["capped_open_plan"].append(int(((~fin) & (N >= PRIOR_CAP)).sum()))
            sf["over3U"].append(int((fin & (N > 3 * U)).sum()))
            sf["zero_plan_nonempty"].append(int(((pn == 0) & (N > 0)).sum()))
            sf["runs_over"].append(int((runs > RUN_X * rmax).sum()))
            sf["max_run"].append(int(runs.max()))
            sf["over_real_max"].append(int((runs > rmax).sum()))
            sf["overU"].append(int((fin & (N > U)).sum()))
            sf["open_over_real_max_len"].append(int(((~fin) & (N > nmax)).sum()))
        RES["safety"][arm] = sf
        say(f"(P) {arm}: capped {sf['capped']} (finite plan {sf['capped_finite_plan']}, open {sf['capped_open_plan']}), finite-plan N > 3U {sf['over3U']} (zero-plan nonempty {sf['zero_plan_nonempty']}), "
            f"runs > {RUN_X} x {rmax} {sf['runs_over']}; reported: max run {sf['max_run']}, runs > {rmax} {sf['over_real_max']}, "
            f"N > U {sf['overU']}, open-bucket N > {nmax} {sf['open_over_real_max_len']}")
    fgroups = {g: groups[g] for g in FID_GROUPS}
    fgroups_drop = {g: groups_drop[g] for g in FID_GROUPS}
    ag_s = rag.select("agent", "role", "cohort", "origin", "b0", "age_days")
    for arm in ("B", "C", "R"):
        fid = {"all": [], "drop_largest": []}
        for r in range(PRIOR_REPS):
            syn = Sample(ag_s, EVS[arm].filter(pl.col("replica") == r).drop("replica"), st.w_start)
            fid["all"].append(compare(realS, syn, fgroups, paired=True, n_boot=FID_BOOT, seed=r))
            fid["drop_largest"].append(compare(realS, syn, fgroups_drop, paired=True, n_boot=FID_BOOT, seed=r))
        RES["fidelity"][arm] = {}
        for view, fl in fid.items():
            for g in FID_GROUPS:
                rg = np.array([f[g]["ratio_gmean"] for f in fl])
                RES["fidelity"][arm][f"{view}/{g}"] = {
                    "ratio_gmean": float(np.exp(np.mean(np.log(rg)))) if np.all(np.isfinite(rg) & (rg > 0)) else float("nan"),
                    "per_replica": rg.tolist(), "n_ratio": [f[g]["n_ratio"] for f in fl], "mpass": float(np.mean([f[g]["pass_rate"] for f in fl])),
                    "mpass_per_replica": [f[g]["pass_rate"] for f in fl],
                    "ratios": {mid: float(np.mean([f[g][mid]["ratio"] for f in fl])) for mid in fl[0][g] if isinstance(fl[0][g][mid], dict)},
                    "M1_per_kind": {m: float(np.mean([f[g]["M1"]["per_kind"][m]["ratio"] for f in fl])) for m in fl[0][g]["M1"]["per_kind"]}}
        say(f"(F) {arm}: " + " | ".join(f"{k} gmean {v['ratio_gmean']:.3f} M-pass {v['mpass']:.3f} n_ratio {v['n_ratio']}"
                                        for k, v in RES["fidelity"][arm].items() if isinstance(v, dict)))

    # gates
    g = {}
    sh = RES["shift"]
    for k in ("e_tt", "e_aa"):
        r = sh[f"R_vs_C_{k}"]
        g[f"S mean |{k}| R vs C -{S_RED:.0%}, lower bound > 0"] = r["reduction"] >= S_RED and r["lo"] > 0
        g[f"S mean |{k}| R <= B"] = sh[k]["R"] <= sh[k]["B"]
    di = RES["discrepancy"]
    for key in ("D_t", "D_a"):
        r = di[f"R_vs_C_{key}"]
        g[f"D mean |{key}| R vs C -{D_RED:.0%}, lower bound > 0"] = r["reduction"] >= D_RED and r["lo"] > 0
        g[f"D mean |{key}| R <= B (common pairs)"] = di["R"][key]["mean_abs_common"] <= di["B"][key]["mean_abs_common"]
    g[f"D coverage R >= C - {COVER_TOL} and >= B - {COVER_TOL} at every depth"] = all(
        di["R"]["coverage"][str(d)] >= max(di["C"]["coverage"][str(d)], di["B"]["coverage"][str(d)]) - COVER_TOL for d in DD)
    pi = RES["pilot"]
    for k in ("pace_score", "amount_w1"):
        r = pi[f"R_vs_C_{k}"]
        g[f"G {k} R vs C -{G_RED:.0%}, paired lower bound > 0"] = r["reduction"] >= G_RED and r["diff_lo"] > 0
        g[f"G {k} R <= B"] = pi["R"][k] <= pi["B"][k]
    g["G reach-100 R >= C and >= B"] = pi["R"]["reach100"] >= max(pi["C"]["reach100"], pi["B"]["reach100"])
    sfR = RES["safety"]["R"]
    g["P R capped streams (pooled over replicas) <= B and <= C"] = sum(sfR["capped"]) <= min(sum(RES["safety"]["B"]["capped"]), sum(RES["safety"]["C"]["capped"]))
    g["P R zero nonempty zero-plan streams"] = sum(sfR["zero_plan_nonempty"]) == 0
    g["P R zero finite-plan streams with N > 3U"] = sum(sfR["over3U"]) == 0
    g[f"P R zero same-second runs > {RUN_X} x real-train max"] = sum(sfR["runs_over"]) == 0
    fi = RES["fidelity"]
    for view in ("all", "drop_largest"):
        n_, s_ = f"{view}/normal", f"{view}/scam"
        g[f"F {n_} R <= (1 - {FID_NORMAL}) C and <= B"] = (fi["R"][n_]["ratio_gmean"] <= (1 - FID_NORMAL) * fi["C"][n_]["ratio_gmean"]
                                                         and fi["R"][n_]["ratio_gmean"] <= fi["B"][n_]["ratio_gmean"])
        g[f"F {s_} R <= {FID_SCAM} x B and C"] = fi["R"][s_]["ratio_gmean"] <= FID_SCAM * min(fi["B"][s_]["ratio_gmean"], fi["C"][s_]["ratio_gmean"])
    g["F every ratio_gmean finite"] = all(np.isfinite(v["ratio_gmean"]) for a in ("B", "C", "R") for v in fi[a].values() if isinstance(v, dict))
    g["F n_ratio == 16 everywhere"] = all(x == 16 for a in ("B", "C", "R") for k, v in fi[a].items() if isinstance(v, dict) for x in v["n_ratio"])
    RES["gates"] = g
    say("\ngates:")
    for k, v in g.items():
        say(f"  {'PASS' if v else 'FAIL'} {k}")
    say(f"drift screen v2 {'PASSES' if all(g.values()) else 'FAILS'}")
    write_text(ctx.root / "screen.json", json.dumps(RES, indent=1, default=float))
    write_text(ctx.root / "screen.txt", "\n".join(lines) + "\n")


if __name__ == "__main__":
    pilot, base, root, cmd = sys.argv[1:5]
    ctx = Ctx(pilot, base, root)
    if cmd == "eval":
        cmd_eval(ctx, sys.argv[5], int(sys.argv[6]))
    elif cmd == "select":
        cmd_select(ctx)
    elif cmd == "gen":
        cmd_gen(ctx, sys.argv[5])
    elif cmd == "report":
        cmd_report(ctx)
    else:
        raise SystemExit(__doc__)
