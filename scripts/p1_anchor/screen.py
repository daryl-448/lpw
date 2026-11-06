"""Anchor screen (reports/p1_anchor_screen.md): gates for the frozen-backbone adapter arms C (clean) and R (robust)
against each other and the untouched base B.

(1) likelihood: exact full-val known-plan NLL per prefix (arm - B <= 0.02 overall / normal / scam) and the s-unknown
    val terms bit-identical to B on every exact prefix.
(2) shift response on REAL val prefixes (plan >= 10, the real next event owned), normal, after d = 3 / 9 / 29 / 99 owned
    events: prefix-wide history shifts (gaps x 10^+-0.3, log10 amounts +-0.5; lpw.p1.anchor.corrupt_view, targets clean)
    -> e_tt = dElog10(gap forecast) / dlog10(gap shift), e_aa = d(amount forecast mean) / d(amount shift), and the cross
    terms e_ta, e_at. Gate: R reduces mean e_tt and mean e_aa (pooled depths) by >= 25% vs C.
(3) pilot generation: val agents with plan >= 30 (both roles), oracle plans, 3 replicas, generation seed 7, cap 150 events.
    pace score = mean W1 of exposure-normalized tau_10 / tau_30 over normal plans 30-100 and 100-300 (non-reachers 1);
    amount W1 = normal plans >= 100, events 31-100, each agent(-replica) weighted equally; reach-100 = share of normal
    plan >= 100 agent-replicas reaching 100 events. Gates: R vs C pace score -15% or better, amount W1 -15% or better,
    reach-100 not lower.
usage: screen.py PILOT BASE_DIR ROOT [--skip-l]   (ROOT holds C/ and R/; writes ROOT/screen.json, ROOT/screen.txt)"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import wasserstein_distance as w1

from lpw.p1.anchor import corrupt_view, merge_corrupted
from lpw.p1.atomic import write_text
from lpw.p1.batch import CTX, Views, collate
from lpw.p1.fidelity import real_frames
from lpw.p1.generate import generate
from lpw.p1.model import mog_mean
from lpw.p1.pace import known_plan_nll, unknown_terms
from lpw.p1.store import IN_AGENT, build_store
from lpw.p1.timelaw import EDGES
from lpw.p1.train import load, to_dev

args = [x for x in sys.argv[1:] if not x.startswith("--")]
pilot, base_dir, root = Path(args[0]), Path(args[1]), Path(args[2])
st = build_store(pilot); va = st.agents("val"); W_END = float(st.w_end)
views = Views(st)
models = {"B": load(pilot, "lpw_plan_prog", base_dir, st)}
for arm in ("C", "R"):
    models[arm] = load(pilot, "lpw_plan_prog_anchor", root / arm, st)
for m in models.values():
    m.plan_inference = True
RES = {"likelihood": {}, "shift": {}, "pilot": {}, "gates": {}}
lines = []
def say(s=""):
    print(s); lines.append(s)


# ---- (1) likelihood ----
if "--skip-l" not in sys.argv:
    kb = known_plan_nll(models["B"], st)
    ub = unknown_terms(models["B"], st)
    RES["likelihood"]["B"] = {"known": kb}
    for arm in ("C", "R"):
        k = known_plan_nll(models[arm], st)
        u = unknown_terms(models[arm], st)
        same = bool(u["complete"] and ub["complete"] and u["rows"] == ub["rows"] and torch.equal(u["terms"], ub["terms"]))
        d = {r: k[r]["full"] - kb[r]["full"] for r in k}
        RES["likelihood"][arm] = {"known": k, "delta_vs_B": d, "unknown_identical": same, "unknown_prefixes": u["rows"]}
        say(f"(1) {arm}: known-plan val NLL vs B " + ", ".join(f"{r} {v:+.4f}" for r, v in d.items()) + f"; s-unknown identical {same} ({u['rows']} prefixes)")

# ---- (2) shift response ----
X = torch.as_tensor(np.log10(np.sqrt(EDGES[:-1] * EDGES[1:])), device="cuda")
SHIFTS = {"base": (0.0, 0.0), "dt+": (0.3, 0.0), "dt-": (-0.3, 0.0), "da+": (0.0, 0.5), "da-": (0.0, -0.5)}
DEPTHS = (3, 9, 29, 99)
items = []  # (val index, depth, agent, k)
for i, a in enumerate(va):
    if st.plan_n[a] < 5:
        continue
    v = views(int(a))
    c = np.cumsum(v["kind"] != IN_AGENT)
    for d in DEPTHS:
        if c[-1:].size == 0 or c[-1] < d:
            continue
        k = int(np.searchsorted(c, d)) + 1  # prefix = first k events, ending at the d-th owned event
        if k < len(v["ts"]) and v["kind"][k] != IN_AGENT:
            items.append((i, d, int(a), k))


@torch.no_grad()
def forecast(m, shift):
    """per item: (Elog10 positive-finite gap of the next owned event, amount forecast mean at the real next event)."""
    el, am = [], []
    for c0 in range(0, len(items), 128):
        ch = items[c0:c0 + 128]
        vs = [views(a) for _, _, a, _ in ch]
        specs = [(max(0, k - CTX), np.array([k - max(0, k - CTX)])) for _, _, _, k in ch]
        clean = collate(vs, specs, st.n_venues, st.w_start)
        b = clean
        if shift != (0.0, 0.0):
            cor = collate([corrupt_view(v, *shift) for v in vs], specs, st.n_venues, st.w_start)
            b = merge_corrupted(clean, cor)
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


role_it = np.array([st.role[a] for _, _, a, _ in items]); dep_it = np.array([d for _, d, _, _ in items])
say(f"\n(2) shift response on {len(items)} real val prefixes (plan >= 10; normal {int((role_it == 0).sum())})")
for name, m in models.items():
    F = {s: forecast(m, sh) for s, sh in SHIFTS.items()}
    e = {"e_tt": (F["dt+"][0] - F["dt-"][0]) / 0.6, "e_at": (F["dt+"][1] - F["dt-"][1]) / 0.6,
         "e_ta": (F["da+"][0] - F["da-"][0]) / 1.0, "e_aa": (F["da+"][1] - F["da-"][1]) / 1.0}
    R = {}
    for ro, rn in ((0, "normal"), (1, "scam")):
        for d in DEPTHS + ("all",):
            mm = (role_it == ro) & ((dep_it == d) if d != "all" else np.ones(len(items), bool))
            R[f"{rn}_{d}"] = {k: float(np.nanmean(v[mm])) for k, v in e.items()} | {"n": int(mm.sum())}
    RES["shift"][name] = R
    for ro in ("normal", "scam"):
        say(f"  {name} {ro:6s} " + " | ".join(f"d {d}: " + " ".join(f"{k} {R[f'{ro}_{d}'][k]:+.3f}" for k in ("e_tt", "e_aa", "e_ta", "e_at"))
                                           for d in DEPTHS + ("all",)))

# ---- (3) pilot generation ----
sub = np.flatnonzero(st.plan_n[va] >= 6)
REPS = 3
conds = [dict(role=int(st.role[x]), cohort=int(st.cohort[x]), origin=int(st.origin[x]), prof=st.prof_vec(x), b0=float(st.b0[x]),
              plan_n=int(st.plan_n[x]), plan_a=int(st.plan_a[x])) for x in va[sub]]
rev = real_frames(st, va[sub])[1].sort("agent", "ts")
def streams(ev, n):
    a = ev["agent"].to_numpy(); t = ev["ts"].to_numpy().astype(np.float64); am = ev["amt"].to_numpy().astype(np.float64)
    cuts = np.searchsorted(a, np.arange(n + 1))
    return [(t[cuts[i]:cuts[i + 1]], am[cuts[i]:cuts[i + 1]]) for i in range(n)]
real = streams(rev, len(sub))
origin = st.origin[va[sub]].astype(np.float64); expo = np.maximum(W_END - origin, 1.0)
role_s, plan_s = st.role[va[sub]], st.plan_n[va[sub]]
tau = lambda S, k: np.array([min((s[0][k - 1] - origin[i % len(sub)]) / expo[i % len(sub)], 1.0) if len(s[0]) >= k else 1.0 for i, s in enumerate(S)])
rt = {k: tau(real, k) for k in (10, 30)}
CELLS = [(0, [6]), (0, [7])]


def amount_w1(S, rep):
    """normal plans >= 100, events 31-100, each stream weighted equally."""
    def collect(SS, reps):
        x, w = [], []
        for i, s in enumerate(SS):
            j = i % len(sub)
            if role_s[j] != 0 or plan_s[j] < 7:
                continue
            seg = s[1][30:100]
            if len(seg):
                x.append(seg); w.append(np.full(len(seg), 1.0 / len(seg)))
        return (np.concatenate(x), np.concatenate(w)) if x else (np.zeros(0), np.zeros(0))
    xr, wr = collect(real, 1); xs, ws = collect(S, rep)
    return float(w1(xr, xs, wr, ws)) if len(xs) else float("nan")


for name, m in models.items():
    (ag, ev), info = generate(m, conds * REPS, st, seed=7, max_events=150, log=lambda *x: None)
    S = streams(ev.sort("agent", "ts"), len(conds) * REPS)
    stau = {k: tau(S, k) for k in (10, 30)}
    rep_role, rep_plan = np.tile(role_s, REPS), np.tile(plan_s, REPS)
    cells = []
    for ro, bks in CELLS:
        rm = (role_s == ro) & np.isin(plan_s, bks); sm = (rep_role == ro) & np.isin(rep_plan, bks)
        for k in (10, 30):
            cells.append(float(w1(rt[k][rm], stau[k][sm])))
    heavy = (rep_role == 0) & (rep_plan >= 7)
    reach = float(np.mean([len(S[i][0]) >= 100 for i in np.flatnonzero(heavy)]))
    rreach = float(np.mean([len(real[j][0]) >= 100 for j in np.flatnonzero((role_s == 0) & (plan_s >= 7))]))
    med = {f"{lo}-{hi}": float(np.median(np.concatenate([S[i][1][lo - 1:hi] for i in np.flatnonzero(heavy)] or [np.zeros(0)])))
           for lo, hi in ((1, 10), (31, 100), (101, 150))}
    RES["pilot"][name] = {"pace_score": float(np.mean(cells)), "cells": cells, "amount_w1": amount_w1(S, REPS), "reach100": reach,
                          "reach100_real": rreach, "amount_median_heavy": med, "capped": info["capped"]}
    say(f"(3) {name}: pace score {np.mean(cells):.4f} (cells {np.round(cells, 4).tolist()}), amount W1 (normal >= 100, events 31-100) "
        f"{RES['pilot'][name]['amount_w1']:.4f}, reach-100 {reach:.3f} (real {rreach:.3f}), heavy amount medians {med}")
rmed = {f"{lo}-{hi}": float(np.median(np.concatenate([real[j][1][lo - 1:hi] for j in np.flatnonzero((role_s == 0) & (plan_s >= 7))])))
        for lo, hi in ((1, 10), (31, 100), (101, 150))}
say(f"    real heavy amount medians {rmed}")

# ---- gates (R vs C) ----
P, Sh = RES["pilot"], RES["shift"]
g = {}
if RES["likelihood"]:
    L = RES["likelihood"]["R"]
    g["L known-plan NLL (R - B <= 0.02)"] = all(v <= 0.02 for k, v in L["delta_vs_B"].items() if k in ("all", "normal", "scam"))
    g["L s-unknown identical"] = L["unknown_identical"]
red = lambda a, b: 1 - a / b if b else float("nan")
g["S e_tt -25%"] = red(abs(Sh["R"]["normal_all"]["e_tt"]), abs(Sh["C"]["normal_all"]["e_tt"])) >= 0.25
g["S e_aa -25%"] = red(abs(Sh["R"]["normal_all"]["e_aa"]), abs(Sh["C"]["normal_all"]["e_aa"])) >= 0.25
g["G pace score -15%"] = red(P["R"]["pace_score"], P["C"]["pace_score"]) >= 0.15
g["G amount W1 -15%"] = red(P["R"]["amount_w1"], P["C"]["amount_w1"]) >= 0.15
g["G reach-100 not lower"] = P["R"]["reach100"] >= P["C"]["reach100"]
RES["gates"] = g
say("\ngates (R vs C): " + "; ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in g.items()))
say(f"screen {'PASSES' if all(g.values()) else 'FAILS'}")
write_text(root / "screen.json", json.dumps(RES, indent=1, default=float))
write_text(root / "screen.txt", "\n".join(lines) + "\n")
