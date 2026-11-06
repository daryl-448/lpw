"""Regularised anchor re-screen (reports/p1_anchor_rescreen.md): gates for the adapter arms C (clean) and R (robust),
fitted by `python -m lpw.p1.anchor --reg` (all training groups except a fixed 20% internal holdout, AdamW, best
held-out checkpoint with step 0 allowed), against each other and the untouched base B.

(L) likelihood, for C AND R: s-unknown val terms bit-identical to B on every exact prefix; exact known-plan NLL per
    prefix (unweighted), arm - B <= 0.02 overall / normal / scam / pooled plan >= 300, on the internal held-out groups
    and on val.
(S) shift response on real val prefixes (as in the first screen): R reduces mean |e_tt| and mean |e_aa| (normal, depths
    3 / 9 / 29 / 99 pooled) by >= 25% vs C, whole-group bootstrap 2.5% quantile of the reduction > 0.
(D) rollout discrepancy, normal val agents with plan >= 100, depths 9 / 29 / 99: per agent, the arm's forecast
    (Elog10 next own gap; amount forecast mean at the next owned event) on its own generated prefixes (replica mean)
    minus the same forecast on the real prefix at the same owned-event depth; D = mean over agent-depth pairs.
    R reduces |D_t| and |D_a| by >= 25% vs C, bootstrap 2.5% quantile of the reduction > 0.
(G) pilot generation (val agents with plan >= 30, oracle plans, 3 replicas, seed 7, cap 150): pace score (as in the
    first screen) and AGENT-weighted amount W1 (normal plans >= 100, events 31-100) each reduced by >= 15% vs C with
    the paired whole-group bootstrap 2.5% quantile of (C - R) > 0; reach-100 (normal plans >= 100) not lower than C.
Bootstraps: 1000 resamples of val groups (agents.parquet group_id) with replacement, seed 0, paired across arms.
usage: rescreen.py PILOT BASE_DIR ROOT   (ROOT holds C/ and R/; writes ROOT/rescreen.json, ROOT/rescreen.txt)"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import wasserstein_distance as w1

from lpw.p1.anchor import corrupt_view, merge_corrupted
from lpw.p1.atomic import write_text
from lpw.p1.batch import CTX, Views, collate, exact_specs, pack
from lpw.p1.fidelity import real_frames
from lpw.p1.generate import generate
from lpw.p1.model import mog_mean
from lpw.p1.pace import unknown_terms
from lpw.p1.store import IN_AGENT, OUT, PERIPHERY, TIER_VENUE, VENUE, build_store, derived
from lpw.p1.timelaw import EDGES
from lpw.p1.train import load, loader, to_dev

NBOOT, LMAX, S_RED, D_RED, G_RED = 1000, 0.02, 0.25, 0.25, 0.15
pilot, base_dir, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
st = build_store(pilot); va = st.agents("val"); W_END = float(st.w_end)
views = Views(st)
ag = pl.read_parquet(pilot / "agents.parquet")
gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
gid = np.array([gid_of[int(x)] for x in st.node_id])  # per store agent
models = {"B": load(pilot, "lpw_plan_prog", base_dir, st)}
info = {}
for arm in ("C", "R"):
    models[arm] = load(pilot, "lpw_plan_prog_anchor", root / arm, st)
    info[arm] = json.loads((root / arm / "anchor.json").read_text())
    assert Path(info[arm]["base"]).resolve() == base_dir.resolve(), (arm, info[arm]["base"])
assert info["C"]["holdout_agent_ids"] == info["R"]["holdout_agent_ids"], "C and R used different internal holdouts"
hold = np.array(info["C"]["holdout_agent_ids"], np.int64)
assert np.isin(hold, st.agents("train")).all() and not np.isin(gid[hold], gid[np.setdiff1d(st.agents("train"), hold)]).any()
for m in models.values():
    m.plan_inference = True
RES = {"fit": {a: {k: info[a][k] for k in ("best_step", "heldout_step0", "best_heldout", "wd", "lr", "max_steps")} for a in info},
       "likelihood": {}, "shift": {}, "discrepancy": {}, "pilot": {}, "gates": {}}
lines = []
rng = np.random.default_rng(0)


def say(s=""):
    print(s); lines.append(s)


def boot_weights(agents):
    """NBOOT x len(agents) multiplicities from whole-group resampling (store agent indices)."""
    g = gid[agents]
    ug, inv = np.unique(g, return_inverse=True)
    draws = rng.integers(0, len(ug), (NBOOT, len(ug)))
    cnt = np.zeros((NBOOT, len(ug)), np.int64)
    np.add.at(cnt, (np.arange(NBOOT)[:, None], draws), 1)
    return cnt[:, inv]


for a in info:
    say(f"fit {a}: best step {info[a]['best_step']} of {info[a]['max_steps']}, held-out {info[a]['heldout_step0']:.4f} -> {info[a]['best_heldout']:.4f}")


# ---- (L) likelihood ----
@torch.no_grad()
def rows_nll(m, agents):
    """exact known-plan full NLL per prefix (as lpw.p1.pace.known_plan_nll) and the prefix's store agent."""
    batches = pack(exact_specs(st, agents), views)
    full, who = [], []
    for bi, b in enumerate(loader(views, batches)):
        sp = batches[bi]
        b = to_dev(b, "cuda")
        t = m.terms(b)
        kt = (t["kind"] + t["time"] + t["cens"]).double()
        full.append(kt.index_add(0, t["o_row"], (t["cp"] + t["amt"]).double()).cpu().numpy())
        who.append(np.array([sp[j][0] for j in b["row_b"].cpu().numpy()], np.int64))
    return np.concatenate(full), np.concatenate(who)


ub = unknown_terms(models["B"], st)
for arm in ("C", "R"):
    u = unknown_terms(models[arm], st)
    same = bool(u["complete"] and ub["complete"] and u["rows"] == ub["rows"] and torch.equal(u["terms"], ub["terms"]))
    RES["likelihood"][arm] = {"unknown_identical": same, "unknown_prefixes": u["rows"]}
    say(f"(L) {arm}: s-unknown val terms identical to B: {same} ({u['rows']} prefixes)")
for sname, agents in (("holdout", hold), ("val", va)):
    nll = {k: rows_nll(m, agents) for k, m in models.items()}
    who = nll["B"][1]
    assert all(np.array_equal(who, nll[k][1]) for k in nll)
    cells = {"all": np.ones(len(who), bool), "normal": st.role[who] == 0, "scam": st.role[who] == 1, "plan>=300": st.plan_n[who] >= 8}
    for arm in ("C", "R"):
        d = {c: float(nll[arm][0][mm].mean() - nll["B"][0][mm].mean()) for c, mm in cells.items()}
        RES["likelihood"][arm][sname] = {"delta_vs_B": d, "rows": {c: int(mm.sum()) for c, mm in cells.items()},
                                         "B": {c: float(nll["B"][0][mm].mean()) for c, mm in cells.items()}}
        say(f"(L) {arm} {sname}: known-plan NLL vs B " + ", ".join(f"{c} {v:+.4f}" for c, v in d.items()))

# ---- forecasts ----
X = torch.as_tensor(np.log10(np.sqrt(EDGES[:-1] * EDGES[1:])), device="cuda")


@torch.no_grad()
def forecast(m, vs, ks, shift=(0.0, 0.0)):
    """per (view, prefix length k): (Elog10 positive-finite gap of the next owned event, amount forecast mean at the
    real next event if it is owned, else nan)."""
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
        mean[t["o_row"].cpu().numpy()] = mog_mean(ap.float(), m.cfg.k_mix).cpu().numpy()
        am.append(mean)
    return np.concatenate(el), np.concatenate(am)


def depth_k(v, d):
    """prefix length ending at the d-th owned event, if the next event exists and is owned; else None."""
    c = np.cumsum(v["kind"] != IN_AGENT)
    if len(c) == 0 or c[-1] < d:
        return None
    k = int(np.searchsorted(c, d)) + 1
    return k if k < len(v["ts"]) and v["kind"][k] != IN_AGENT else None


def red_stats(eC, eR, w, W):
    """point and bootstrap reduction 1 - |mean_w e_R| / |mean_w e_C| (w: point weights; W: NBOOT x n weights)."""
    mC, mR = abs((w * eC).sum() / w.sum()), abs((w * eR).sum() / w.sum())
    bC = np.abs((W * eC).sum(1) / W.sum(1)); bR = np.abs((W * eR).sum(1) / W.sum(1))
    rb = 1 - bR / bC
    return {"C": float(mC), "R": float(mR), "reduction": float(1 - mR / mC), "lo": float(np.quantile(rb, 0.025)),
            "hi": float(np.quantile(rb, 0.975))}


# ---- (S) shift response ----
SHIFTS = {"dt+": (0.3, 0.0), "dt-": (-0.3, 0.0), "da+": (0.0, 0.5), "da-": (0.0, -0.5)}
DEPTHS = (3, 9, 29, 99)
items = []  # (store agent, depth, k)
for a in va:
    if st.plan_n[a] < 5:
        continue
    v = views(int(a))
    for d in DEPTHS:
        k = depth_k(v, d)
        if k is not None:
            items.append((int(a), d, k))
it_a = np.array([a for a, _, _ in items]); it_d = np.array([d for _, d, _ in items])
it_v = [views(a) for a, _, _ in items]; it_k = [k for _, _, k in items]
say(f"\n(S) shift response on {len(items)} real val prefixes (plan >= 10; normal {int((st.role[it_a] == 0).sum())})")
E = {}
for name, m in models.items():
    F = {s: forecast(m, it_v, it_k, sh) for s, sh in SHIFTS.items()}
    E[name] = {"e_tt": (F["dt+"][0] - F["dt-"][0]) / 0.6, "e_at": (F["dt+"][1] - F["dt-"][1]) / 0.6,
               "e_ta": (F["da+"][0] - F["da-"][0]) / 1.0, "e_aa": (F["da+"][1] - F["da-"][1]) / 1.0}
    R = {}
    for ro, rn in ((0, "normal"), (1, "scam")):
        for d in DEPTHS + ("all",):
            mm = (st.role[it_a] == ro) & ((it_d == d) if d != "all" else True)
            R[f"{rn}_{d}"] = {k: float(np.nanmean(x[mm])) for k, x in E[name].items()} | {"n": int(mm.sum())}
    RES["shift"][name] = R
    say(f"  {name} normal " + " | ".join(f"d {d}: " + " ".join(f"{k} {R[f'normal_{d}'][k]:+.3f}" for k in ("e_tt", "e_aa", "e_ta", "e_at"))
                                         for d in DEPTHS + ("all",)))
nm = st.role[it_a] == 0
ua = np.unique(it_a[nm]); Wa = boot_weights(ua); pos = {a: i for i, a in enumerate(ua)}
for k in ("e_tt", "e_aa"):
    ok = nm & np.isfinite(E["C"][k]) & np.isfinite(E["R"][k])
    col = np.array([pos[a] for a in it_a[ok]])
    RES["shift"][f"R_vs_C_{k}"] = r = red_stats(E["C"][k][ok], E["R"][k][ok], np.ones(ok.sum()), Wa[:, col])
    say(f"  R vs C mean {k}: {r['C']:.4f} -> {r['R']:.4f}, reduction {r['reduction']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

# ---- (G) pilot generation ----
sub = np.flatnonzero(st.plan_n[va] >= 6)
REPS = 3
sa = va[sub]
conds = [dict(role=int(st.role[x]), cohort=int(st.cohort[x]), origin=int(st.origin[x]), prof=st.prof_vec(x), b0=float(st.b0[x]),
              plan_n=int(st.plan_n[x]), plan_a=int(st.plan_a[x])) for x in sa]
EV = ("kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal")
role_s, plan_s = st.role[sa], st.plan_n[sa]
origin = st.origin[sa].astype(np.float64); expo = np.maximum(W_END - origin, 1.0)
rev = real_frames(st, sa)[1].sort("agent", "ts", maintain_order=True)  # IN_agent removed, as generated streams
_a = rev["agent"].to_numpy(); _c = np.searchsorted(_a, np.arange(len(sa) + 1))
_t, _m = rev["ts"].to_numpy().astype(np.float64), rev["amt"].to_numpy().astype(np.float64)
real = [(_t[_c[i]:_c[i + 1]], _m[_c[i]:_c[i + 1]]) for i in range(len(sa))]
CELLS = [(0, [6]), (0, [7])]
heavy_r = (role_s == 0) & (plan_s >= 7)


def syn_view(a, v):
    """generated stream of store agent a as a Views-compatible dict (as runs/p1_diag/runaway/burstcheck.py)."""
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


def tau(S, k, rep):
    return np.array([min((s[0][k - 1] - origin[i % len(sub)]) / expo[i % len(sub)], 1.0) if len(s[0]) >= k else 1.0
                     for i, s in enumerate(S)])


def segs(S):
    return [s[1][30:100] for s in S]


def pilot_scores(S, w):
    """(pace score, agent-weighted amount W1) with per-agent multiplicities w (len(sub)); S = REPS x len(sub) streams."""
    wr = np.tile(w, REPS).astype(np.float64)
    cells = []
    for ro, bks in CELLS:
        rm = (role_s == ro) & np.isin(plan_s, bks) & (w > 0); sm = np.tile(rm, REPS)
        for k in (10, 30):
            cells.append(float(w1(RT[k][rm], ST[k][sm], w[rm], wr[sm])))
    xr, wa_r, xs, wa_s = [], [], [], []
    for j in np.flatnonzero(heavy_r & (w > 0)):
        if len(RSEG[j]):
            xr.append(RSEG[j]); wa_r.append(np.full(len(RSEG[j]), w[j] / len(RSEG[j])))
        reps = [SSEG[r * len(sub) + j] for r in range(REPS) if len(SSEG[r * len(sub) + j])]
        for s in reps:
            xs.append(s); wa_s.append(np.full(len(s), w[j] / (len(s) * len(reps))))
    aw = float(w1(np.concatenate(xr), np.concatenate(xs), np.concatenate(wa_r), np.concatenate(wa_s))) if xs else float("nan")
    return float(np.mean(cells)), aw, cells


RT = {k: tau(real, k, 1) for k in (10, 30)}
RSEG = segs(real)
Wg = boot_weights(sa)
GEN, BOOT = {}, {}
for name, m in models.items():
    (_, ev), ginfo = generate(m, conds * REPS, st, seed=7, max_events=150, log=lambda *x: None)
    ev = ev.sort("agent", "ts", maintain_order=True)
    a_col = ev["agent"].to_numpy(); cuts = np.searchsorted(a_col, np.arange(len(conds) * REPS + 1))
    cols = {f: ev[f].to_numpy() for f in EV}
    raw = [{f: cols[f][cuts[i]:cuts[i + 1]] for f in EV} for i in range(len(conds) * REPS)]
    S = [(r["ts"].astype(np.float64), r["amt"].astype(np.float64)) for r in raw]
    GEN[name] = (raw, S)
    ST = {k: tau(S, k, REPS) for k in (10, 30)}
    SSEG = segs(S)
    pace, aw, cells = pilot_scores(S, np.ones(len(sub), np.int64))
    hv = np.flatnonzero(np.tile(heavy_r, REPS))
    reach = float(np.mean([len(S[i][0]) >= 100 for i in hv]))
    rreach = float(np.mean([len(real[j][0]) >= 100 for j in np.flatnonzero(heavy_r)]))
    med = {f"{lo}-{hi}": float(np.median(np.concatenate([S[i][1][lo - 1:hi] for i in hv] or [np.zeros(0)])))
           for lo, hi in ((1, 10), (31, 100), (101, 150))}
    RES["pilot"][name] = {"pace_score": pace, "cells": cells, "amount_w1": aw, "reach100": reach, "reach100_real": rreach,
                          "amount_median_heavy": med, "capped": ginfo["capped"]}
    if name in ("C", "R"):
        BOOT[name] = np.array([pilot_scores(S, Wg[b])[:2] for b in range(NBOOT)])
    say(f"(G) {name}: pace score {pace:.4f} (cells {np.round(cells, 4).tolist()}), agent-weighted amount W1 {aw:.4f}, "
        f"reach-100 {reach:.3f} (real {rreach:.3f}), heavy amount medians {med}")
for i, k in enumerate(("pace_score", "amount_w1")):
    c, r = RES["pilot"]["C"][k], RES["pilot"]["R"][k]
    diff = BOOT["C"][:, i] - BOOT["R"][:, i]
    RES["pilot"][f"R_vs_C_{k}"] = g = {"reduction": float(1 - r / c), "diff_lo": float(np.quantile(diff, 0.025)),
                                       "diff_hi": float(np.quantile(diff, 0.975))}
    say(f"  R vs C {k}: {c:.4f} -> {r:.4f}, reduction {g['reduction']:.3f}, C - R 95% [{g['diff_lo']:.4f}, {g['diff_hi']:.4f}]")

# ---- (D) rollout discrepancy ----
DD = (9, 29, 99)
hj = np.flatnonzero(heavy_r)
real_items = [(j, d, depth_k(views(int(sa[j])), d)) for j in hj for d in DD]
real_items = [x for x in real_items if x[2] is not None]
say(f"\n(D) rollout discrepancy: normal plan >= 100, depths {DD}; {len(real_items)} real agent-depth prefixes")
DIS = {}
for name in ("B", "C", "R"):
    m = models[name]
    raw, _ = GEN[name]
    fr = forecast(m, [views(int(sa[j])) for j, _, _ in real_items], [k for _, _, k in real_items])
    sv, sk, sidx = [], [], []
    for p, (j, d, _) in enumerate(real_items):
        for r in range(REPS):
            v = raw[r * len(sub) + j]
            if len(v["ts"]) == 0:
                continue
            vv = syn_view(int(sa[j]), dict(v))
            k = depth_k(vv, d)
            if k is not None:
                sv.append(vv); sk.append(k); sidx.append(p)
    fs = forecast(m, sv, sk) if sv else (np.zeros(0), np.zeros(0))
    sidx = np.array(sidx, np.int64)
    out = {}
    for q, key in ((0, "D_t"), (1, "D_a")):
        num = np.zeros(len(real_items)); cnt = np.zeros(len(real_items))
        ok = np.isfinite(fs[q])
        np.add.at(num, sidx[ok], fs[q][ok]); np.add.at(cnt, sidx[ok], 1)
        has = (cnt > 0) & np.isfinite(fr[q])
        out[key] = np.where(has, num / np.maximum(cnt, 1) - fr[q], np.nan)
    DIS[name] = out
    RES["discrepancy"][name] = {key: {str(d): float(np.nanmean(x[np.array([dd == d for _, dd, _ in real_items])])) for d in DD}
                                | {"all": float(np.nanmean(x)), "pairs": int(np.isfinite(x).sum())} for key, x in out.items()}
    say(f"  {name}: " + " | ".join(f"{key} " + " ".join(f"d{d} {RES['discrepancy'][name][key][str(d)]:+.3f}" for d in DD)
                                   + f" all {RES['discrepancy'][name][key]['all']:+.3f} ({RES['discrepancy'][name][key]['pairs']} pairs)"
                                   for key in ("D_t", "D_a")))
pa_ = np.array([sa[j] for j, _, _ in real_items])
ua = np.unique(pa_); Wd = boot_weights(ua); pos = {a: i for i, a in enumerate(ua)}
for key in ("D_t", "D_a"):
    # pairs where both arms have a value; agents absent from a resample get weight 0
    ok = np.isfinite(DIS["C"][key]) & np.isfinite(DIS["R"][key])
    col = np.array([pos[a] for a in pa_[ok]])
    RES["discrepancy"][f"R_vs_C_{key}"] = r = red_stats(DIS["C"][key][ok], DIS["R"][key][ok], np.ones(ok.sum()), Wd[:, col])
    say(f"  R vs C |{key}| (pairs in both arms, {int(ok.sum())}): {r['C']:.4f} -> {r['R']:.4f}, reduction {r['reduction']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

# ---- gates ----
g = {}
for arm in ("C", "R"):
    L = RES["likelihood"][arm]
    g[f"L {arm} s-unknown identical"] = L["unknown_identical"]
    for sname in ("holdout", "val"):
        g[f"L {arm} {sname} known-plan NLL (all / normal / scam / plan>=300 <= +{LMAX})"] = all(v <= LMAX for v in L[sname]["delta_vs_B"].values())
for k in ("e_tt", "e_aa"):
    r = RES["shift"][f"R_vs_C_{k}"]
    g[f"S {k} -{S_RED:.0%}, lower bound > 0"] = r["reduction"] >= S_RED and r["lo"] > 0
for k in ("D_t", "D_a"):
    r = RES["discrepancy"][f"R_vs_C_{k}"]
    g[f"D |{k}| -{D_RED:.0%}, lower bound > 0"] = r["reduction"] >= D_RED and r["lo"] > 0
for k in ("pace_score", "amount_w1"):
    r = RES["pilot"][f"R_vs_C_{k}"]
    g[f"G {k} -{G_RED:.0%}, paired lower bound > 0"] = r["reduction"] >= G_RED and r["diff_lo"] > 0
g["G reach-100 not lower"] = RES["pilot"]["R"]["reach100"] >= RES["pilot"]["C"]["reach100"]
RES["gates"] = g
say("\ngates:")
for k, v in g.items():
    say(f"  {'PASS' if v else 'FAIL'} {k}")
say(f"re-screen {'PASSES' if all(g.values()) else 'FAILS'}")
write_text(root / "rescreen.json", json.dumps(RES, indent=1, default=float))
write_text(root / "rescreen.txt", "\n".join(lines) + "\n")
