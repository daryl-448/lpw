"""Rollout feedback vs static pace error (Sol B-lite design round, step 1; fixes from the step-1 review).

Scores the model's next-gap forecast on REAL val prefixes and on its own GENERATED prefixes (saved oracle replicas:
same agents, same real plan), at matched owned depth d, and measures how strongly the forecast follows the history's
pace (feature interventions on real prefixes at a fixed final timestamp, plan and owned count).

Forecast summaries (kind-marginal, normalized weekly law at the prefix time): Elog = mean log10 positive-finite gap
using bin centres (a proxy, not an exact expectation); P(>1d) = P(gap > 1 day | positive finite), from the survival
function; same-second mass; dormant mass; P(no owned event before w_end); Elog_neutral = the same proxy from the raw
bin probabilities (removes the weekly weighting of the forecast, NOT calendar information entering the encoder).

(1) paired real vs generated: every comparison uses (agent, replica) pairs where both prefixes exist (deep rows are
    survivor-selected); 95% intervals by resampling whole val groups (1000 draws).
(2) history sensitivity on real prefixes (plan >= 10, d in 3/9/29/99, one fixed row set):
    cal_lam: calendar-frozen stretch, timestamps (hence hour / weekday / month and partner recency) kept, only the
             gap features dt / dtbin of every earlier event (incl. origin -> first event) scaled by lam in {2/3, 3/2};
             elasticity e = dElog / dlog10(lam) on the same rows for both lam;
    lam0.67: the time-consistent stretch t_last - lam (t_last - t_i) (moves calendar features too);
    recent: the second-to-last event moved earlier by half the gap before it (same span), where that lengthens the
            latest gap by >= 0.05 log10.
    --compare CTRL_CKPT: paired change of the calendar-frozen elasticity vs a control checkpoint's saved rows.
(3) --swap: timestamps swapped between real and generated prefixes (whole non-time bundle vs timing).
(4) --ablate: depth-1 single-mark swaps (kind; amount; counterparty class + venue / tier) into the real prefix.

usage: scripts/p1_pace/feedback.py CKPT MODEL PILOT ORACLE_GEN_PARQUET|- [OUT.txt] [--sens-only] [--swap] [--ablate] [--fresh]
                   [--compare CTRL_CKPT]"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.atomic import atomic_path, write_text
from lpw.p1.batch import CTX, Views, collate, to_torch
from lpw.p1.store import IN_AGENT, NEW, OUT, PERIPHERY, TIER_VENUE, VENUE, build_store, derived
from lpw.p1.timelaw import EDGES, TimeLaw, dt_bin
from lpw.p1.train import load

flags = {x for x in sys.argv[1:] if x.startswith("--")}
pos = [x for x in sys.argv[1:] if not x.startswith("--")]
ctrl = None
if "--compare" in flags:
    ctrl = Path(sys.argv[sys.argv.index("--compare") + 1]); pos.remove(str(ctrl)) if str(ctrl) in pos else None
ck, mname, pilot, gpath = Path(pos[0]), pos[1], Path(pos[2]), pos[3]
outp = pos[4] if len(pos) > 4 else None
st = build_store(pilot); va = st.agents("val"); W_END = float(st.w_end)
m = load(pilot, mname, ck, st); m.plan_inference = True
views = Views(st)
DEPTHS = [0, 1, 3, 9, 29, 99]
X = np.log10(np.sqrt(EDGES[:-1] * EDGES[1:]))  # bin centres, log10 seconds, bins 1..62
EV = ("kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal")
PGROUP = {"10-30": [5], "30-100": [6], "100-300": [7], ">=300": [8, 9]}
_ag = pl.read_parquet(pilot / "agents.parquet")
_gid = dict(zip(_ag["node_id"].to_list(), _ag["group_id"].to_list()))
group = np.array([_gid[int(x)] for x in st.node_id[va]])
RNG = np.random.default_rng(0)


def fingerprint(*paths):
    h = hashlib.sha1()
    for p in paths:
        p = Path(p)
        if p.exists():
            s = p.stat(); h.update(f"{p}:{s.st_size}:{s.st_mtime_ns}".encode())
    return h.hexdigest()[:10]


def gboot(x, agent_idx, n=1000):
    """mean of x and a 95% interval from resampling whole val groups (x rows belong to agents agent_idx)."""
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    g = group[np.asarray(agent_idx)]
    ug, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, x, len(ug)), np.bincount(inv, minlength=len(ug)).astype(np.float64)
    draws = np.stack([np.bincount(RNG.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(n)])
    means = (draws @ s) / np.maximum(draws @ c, 1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def cut(v, k, ts=None):
    """View of the first k events (optionally with replaced timestamps), derived fields recomputed, the ledger
    balance at the end of the prefix."""
    w = {f: np.asarray(v[f][:k]) for f in EV}
    if ts is not None:
        w["ts"] = ts
    for f in ("e_tier", "e_sc", "e_venue", "origin", "obs_end", "role", "cohort", "prof", "plan_n", "plan_a"):
        w[f] = v[f]
    if k:
        kk = w["kind"][-1]
        w["bal_end"] = float(w["bal"][-1] + (w["eth"][-1] if kk != OUT else -w["eth"][-1] - w["fee"][-1]))
    else:
        w["bal_end"] = float(v["bal"][0]) if len(v["bal"]) else float(v["bal_end"])
    w.update(derived(w, st.w_start))
    return w


def syn_views(ev: pl.DataFrame, i: int) -> dict:
    a = va[i]
    g = ev.filter(pl.col("agent") == i).sort("ts", maintain_order=True)
    v = {f: g[f].to_numpy() for f in EV}
    ne = int(v["ent"].max()) + 1 if len(g) else 0
    tier, sc, ven = np.zeros(ne, np.int8), np.ones(ne, bool), np.full(ne, -1, np.int16)
    seen = set()
    for c, e, vi in zip(v["cls"], v["ent"], v["venue"]):
        if e in seen:
            continue
        seen.add(e)
        if c == VENUE:
            tier[e], sc[e], ven[e] = TIER_VENUE, bool(st.venue_sc[vi]), vi
        else:
            assert c == NEW, "first occurrence of an entity must create it"
            tier[e], sc[e], ven[e] = PERIPHERY, True, -1
    v.update(e_tier=tier, e_sc=sc, e_venue=ven, origin=int(st.origin[a]), obs_end=st.w_end, role=int(st.role[a]),
             cohort=int(st.cohort[a]), prof=st.prof_vec(a), plan_n=int(st.plan_n[a]), plan_a=int(st.plan_a[a]))
    v["bal_end"] = float(v["bal"][-1] + (v["eth"][-1] if v["kind"][-1] != OUT else -v["eth"][-1] - v["fee"][-1])) if len(g) else float(st.b0[a])
    return v


@torch.no_grad()
def forecast(vs: list[dict]) -> dict:
    """Forecast summaries for the prefix 'all events of v' of each view."""
    out = {k: [] for k in ("elog", "p1d", "same", "dorm", "pstop", "elog_neutral")}
    for c in range(0, len(vs), 192):
        ch = vs[c:c + 192]
        specs = [(max(0, len(v["ts"]) - CTX), np.array([len(v["ts"]) - max(0, len(v["ts"]) - CTX)])) for v in ch]
        B = to_torch(collate(ch, specs, st.n_venues, st.w_start, sample=True), "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            Hk, Ho = m.encode(B)
        rb, rj, t0 = B["row_b"], B["row_j"], B["row_t0"]
        logpk, law, _ = m.kind_time(Hk.float()[rb, rj], Ho.float()[rb, rj], t0, **m.row_extras(B))
        n = len(rb); pk = logpk.double().exp()
        bpk = law.bin_probs().view(n, 2, 64)
        bp = (pk[:, :, None] * bpk).sum(1)
        # raw bin probabilities normalized per kind (an adapted log_pdt is an unnormalized potential)
        raw = (pk[:, :, None] * TimeLaw(law.log_pdt, None, None, law.t0).bin_probs().view(n, 2, 64)).sum(1)
        q = bp[:, 1:63] / bp[:, 1:63].sum(1, keepdim=True)
        qr = raw[:, 1:63] / raw[:, 1:63].sum(1, keepdim=True)
        xb = torch.as_tensor(X, device=q.device)
        S1 = law.log_survival(torch.full((2 * n,), 86400.0, device=t0.device, dtype=torch.float64)).exp().view(n, 2)
        pos_ = (pk * (1 - bpk[:, :, 0] - bpk[:, :, 63])).sum(1)
        p1d = (pk * (S1 - bpk[:, :, 63])).sum(1) / pos_.clamp(min=1e-300)
        logS = law.log_survival((W_END - t0).clamp(min=0).repeat_interleave(2)).view(n, 2)
        vals = dict(elog=(q * xb).sum(1), p1d=p1d, same=bp[:, 0], dorm=bp[:, 63],
                    pstop=torch.exp(torch.logsumexp(logpk.double() + logS, -1)), elog_neutral=(qr * xb).sum(1))
        for k, x in vals.items():
            out[k].append(x.cpu().numpy())
    return {k: np.concatenate(x) if x else np.zeros(0) for k, x in out.items()}


def own_pos(kind):
    """position k (events before) right after the d-th owned event, for each d; 0 for d = 0."""
    c = np.cumsum(kind != IN_AGENT)
    return {d: (0 if d == 0 else (int(np.searchsorted(c, d)) + 1 if c[-1:].size and c[-1] >= d else None)) for d in DEPTHS}


lines = []
def say(s=""):
    print(s); lines.append(s)


def save():
    if outp:
        write_text(outp, "\n".join(lines) + "\n")


role, plan = st.role[va], st.plan_n[va]
real_all = {i: views(va[i]) for i in range(len(va))}
no_inag = np.array([not (real_all[i]["kind"] == IN_AGENT).any() for i in range(len(va))])
sev = None if gpath == "-" else pl.read_parquet(gpath)
reps = sorted(sev["replica"].unique().to_list()) if sev is not None else []

# ---- (1) matched real vs generated prefixes ----
if "--sens-only" not in flags:
    CACHE = ck / f"feedback_rows_{Path(gpath).stem}_{fingerprint(ck / 'model.pt', gpath)}.parquet"
    if CACHE.exists() and "--fresh" not in flags:
        R = pl.read_parquet(str(CACHE))
    else:
        recs, fv = [], []
        for i in range(len(va)):
            if plan[i] < 5:  # pace matters for >= 10-event plans; keep the run small
                continue
            v = real_all[i]
            for d, k in own_pos(v["kind"]).items():
                if k is None:
                    continue
                w = cut(v, k)
                fv.append(w); recs.append((i, d, "real", -1, ((w["ts"][-1] if k else w["origin"]) - w["origin"]) / 86400))
        for r in reps:
            ev = sev.filter(pl.col("replica") == r)
            for i in range(len(va)):
                if plan[i] < 5:
                    continue
                v = syn_views(ev, i)
                for d in DEPTHS:
                    if d > len(v["ts"]):
                        continue
                    w = cut(v, d)
                    fv.append(w); recs.append((i, d, "syn", r, ((w["ts"][-1] if d else w["origin"]) - w["origin"]) / 86400))
        F = forecast(fv)
        R = pl.DataFrame({"i": [x[0] for x in recs], "d": [x[1] for x in recs], "src": [x[2] for x in recs], "rep": [x[3] for x in recs],
                          "elapsed": [x[4] for x in recs], **{k: v for k, v in F.items()}})
        R = R.with_columns(pl.Series("role", role[R["i"].to_numpy()]), pl.Series("plan", plan[R["i"].to_numpy()]),
                           pl.Series("noin", no_inag[R["i"].to_numpy()]))
        with atomic_path(CACHE) as t:
            R.write_parquet(str(t))
    say(f"(1) paired forecasts, real vs generated prefix of the same agent and depth ({ck}; {gpath}); plan >= 10")
    say("    pairs = (agent, replica) with both prefixes; reach = generated pairs / (agents x replicas); means over pairs;")
    say("    dElog = syn - real with a 95% whole-group bootstrap interval")
    cols = ("elog", "elog_neutral", "p1d", "same", "dorm", "pstop")
    for ro, rn in ((0, "normal"), (1, "scam")):
        for gname, bks in PGROUP.items():
            say(f"  {rn} plan {gname}")
            say("     d  pairs reach  elapsed r/s   dElog [95%]             " + "  ".join(f"{c} r/s" for c in cols))
            nag = ((role == ro) & np.isin(plan, bks)).sum()
            for d in DEPTHS:
                base = R.filter((pl.col("role") == ro) & pl.col("plan").is_in(bks) & (pl.col("d") == d))
                re_, sy = base.filter(pl.col("src") == "real"), base.filter(pl.col("src") == "syn")
                if len(re_) < 5 or len(sy) < 5:
                    continue
                pr = sy.join(re_.select("i", *[pl.col(c).alias(c + "_r") for c in cols + ("elapsed",)]), on="i")
                dd = gboot((pr["elog"] - pr["elog_r"]).to_numpy(), pr["i"].to_numpy())
                say(f"    {d:2d}  {len(pr):5d} {len(sy)/(nag*len(reps)):.2f}  {pr['elapsed_r'].median():6.1f}/{pr['elapsed'].median():6.1f}  {fmt(dd)}  "
                    + "  ".join(f"{pr[c + '_r'].mean():.3f}/{pr[c].mean():.3f}" for c in cols))
        b0 = R.filter((pl.col("role") == ro) & (pl.col("plan") >= 6) & pl.col("noin"))
        if len(b0):
            say(f"  {rn} plan >= 30, agents without IN_agent tokens: d, pairs, Elog r/s, dElog [95%]")
            for d in DEPTHS:
                b = b0.filter(pl.col("d") == d); re_, sy = b.filter(pl.col("src") == "real"), b.filter(pl.col("src") == "syn")
                if len(re_) < 5 or len(sy) < 5:
                    continue
                pr = sy.join(re_.select("i", pl.col("elog").alias("elog_r")), on="i")
                say(f"    {d:2d}  {len(pr):5d}  {pr['elog_r'].mean():.3f}/{pr['elog'].mean():.3f}  {fmt(gboot((pr['elog'] - pr['elog_r']).to_numpy(), pr['i'].to_numpy()))}")
    save()

# ---- (2) history sensitivity on real prefixes ----
say("\n(2) history sensitivity on real prefixes (plan >= 10): elasticity e = dElog / dlog10(lam); calendar-frozen rows identical for both lam")
pv, meta = [], []
for i in range(len(va)):
    if plan[i] < 5:
        continue
    v = real_all[i]
    for d, k in own_pos(v["kind"]).items():
        if k is None or d < 3:
            continue
        w0 = cut(v, k)
        pv.append(w0); meta.append((i, d, "base", 0.0))
        for lam in (2 / 3, 1.5):
            w = dict(w0); w["dt"] = w0["dt"] * lam; w["dtbin"] = dt_bin(w["dt"])
            pv.append(w); meta.append((i, d, f"cal_lam{lam:.2f}", np.log10(lam)))
        ts = np.asarray(v["ts"][:k], np.float64); tl = ts[-1]
        pv.append(cut(v, k, tl - (2 / 3) * (tl - ts))); meta.append((i, d, "lam0.67", np.log10(2 / 3)))
        if k >= 3 and ts[-2] > ts[-3] and ts[-1] > ts[-2]:
            t3 = ts.copy(); delta = 0.5 * (ts[-2] - ts[-3]); g1 = ts[-1] - ts[-2]
            if np.log10((g1 + delta) / g1) >= 0.05:  # a perturbation large enough for a stable ratio
                t3[-2] -= delta
                pv.append(cut(v, k, t3)); meta.append((i, d, "recent", np.log10((g1 + delta) / g1)))
P = forecast(pv)
M = pl.DataFrame({"i": [x[0] for x in meta], "d": [x[1] for x in meta], "kind": [x[2] for x in meta], "dx": [x[3] for x in meta],
                  "elog": P["elog"], "elog_neutral": P["elog_neutral"]})
M = M.join(M.filter(pl.col("kind") == "base").select("i", "d", pl.col("elog").alias("e0"), pl.col("elog_neutral").alias("n0")), on=["i", "d"])
M = M.with_columns(pl.Series("role", role[M["i"].to_numpy()]), pl.Series("plan", plan[M["i"].to_numpy()]),
                   ((pl.col("elog") - pl.col("e0")) / pl.col("dx")).alias("e"))
with atomic_path(ck / "feedback_sens.parquet") as t:
    M.write_parquet(str(t))
for ro, rn in ((0, "normal"), (1, "scam")):
    say(f"  {rn}: d   cal_lam 2/3          cal_lam 3/2          lam 2/3 (time-consistent)   recent-gap response   (rows)")
    for d in DEPTHS[2:]:
        b = M.filter((pl.col("role") == ro) & (pl.col("d") == d))
        cells = []
        for kk in ("cal_lam0.67", "cal_lam1.50", "lam0.67", "recent"):
            x = b.filter(pl.col("kind") == kk)
            cells.append(fmt(gboot(x["e"].to_numpy(), x["i"].to_numpy())) if len(x) >= 5 else "   -   ")
        say(f"     {d:2d}  " + "   ".join(cells) + f"   ({len(b.filter(pl.col('kind') == 'base'))})")
cal = M.filter(pl.col("kind").str.starts_with("cal_") & pl.col("d").is_in([3, 9]) & (pl.col("role") == 0))
prim = gboot(cal["e"].to_numpy(), cal["i"].to_numpy())
SENS = {"mean_e": prim[0], "lo": prim[1], "hi": prim[2], "rows": len(cal)}
say(f"  primary mechanism metric (normal, d 3 and 9, calendar-frozen, both lam): mean e {fmt(prim)}")
if ctrl is not None:
    C0 = pl.read_parquet(str(ctrl / "feedback_sens.parquet"))
    j = cal.join(C0.select("i", "d", "kind", pl.col("e").alias("e_ctrl")), on=["i", "d", "kind"])
    assert len(j) == len(cal), "control rows differ"
    rel = 1 - j["e"].mean() / j["e_ctrl"].mean()
    pdiff = gboot((j["e"] - j["e_ctrl"]).to_numpy(), j["i"].to_numpy())
    SENS |= {"control": str(ctrl), "control_mean_e": float(j["e_ctrl"].mean()), "rel_reduction": float(rel), "diff": pdiff}
    say(f"  vs control {ctrl}: mean e {j['e'].mean():+.3f} vs {j['e_ctrl'].mean():+.3f}, reduction {rel:.3f}; "
        f"paired difference {fmt(pdiff)}")
import json as _json
write_text(ck / "feedback_sens.json", _json.dumps(SENS, indent=1))
save()

# ---- (3) timing vs marks ----
if "--swap" in flags:
    say("\n(3) timing vs marks (agents without IN_agent tokens, plan >= 30): the time effect is (real marks + syn times) - real,")
    say("    the mark effect (syn marks + real times) - real; marks = the whole non-time bundle (kind, counterparty, amount, balances)")
    sv, smeta = [], []
    for r in reps:
        ev = sev.filter(pl.col("replica") == r)
        for i in np.flatnonzero(no_inag & (plan >= 6)):
            v, s = real_all[i], syn_views(ev, i)
            for d in (1, 3, 9, 29):
                if d > len(s["ts"]) or d > len(v["ts"]):
                    continue
                rt, stt = np.asarray(v["ts"][:d], np.float64), np.asarray(s["ts"][:d], np.float64)
                sv += [cut(v, d), cut(v, d, stt), cut(s, d, rt), cut(s, d)]
                smeta += [(i, d, r, x) for x in ("real", "real_marks_syn_times", "syn_marks_real_times", "syn")]
    S = forecast(sv)
    D = pl.DataFrame({"i": [x[0] for x in smeta], "d": [x[1] for x in smeta], "rep": [x[2] for x in smeta], "kind": [x[3] for x in smeta],
                      "elog": S["elog"]}).with_columns(pl.Series("role", role[np.array([x[0] for x in smeta])]))
    W = D.pivot(on="kind", index=["i", "d", "rep", "role"], values="elog")
    for ro, rn in ((0, "normal"), (1, "scam")):
        say(f"  {rn}: d   pairs  syn-real [95%]          time effect [95%]        mark effect [95%]")
        for d in (1, 3, 9, 29):
            w = W.filter((pl.col("role") == ro) & (pl.col("d") == d))
            if len(w) < 5:
                continue
            ii = w["i"].to_numpy()
            say(f"     {d:2d}  {len(w):5d}  {fmt(gboot((w['syn'] - w['real']).to_numpy(), ii))}  {fmt(gboot((w['real_marks_syn_times'] - w['real']).to_numpy(), ii))}  "
                f"{fmt(gboot((w['syn_marks_real_times'] - w['real']).to_numpy(), ii))}")
    save()

# ---- (4) depth-1 single-mark ablations ----
if "--ablate" in flags:
    say("\n(4) depth-1 single-mark swaps into the real prefix (agents without IN_agent tokens, plan >= 30): change of Elog vs real")
    av, ameta = [], []
    for r in reps:
        ev = sev.filter(pl.col("replica") == r)
        for i in np.flatnonzero(no_inag & (plan >= 6)):
            v, s = real_all[i], syn_views(ev, i)
            if len(v["ts"]) < 1 or len(s["ts"]) < 1:
                continue
            base = cut(v, 1)
            av.append(base); ameta.append((i, r, "real"))
            w = dict(base); w["kind"] = s["kind"][:1].copy(); av.append(cut(w, 1)); ameta.append((i, r, "kind"))
            w = dict(base); w["amt"], w["eth"] = s["amt"][:1].copy(), s["eth"][:1].copy(); av.append(cut(w, 1)); ameta.append((i, r, "amount"))
            w = dict(base)
            w["cls"], w["venue"], w["ent"] = s["cls"][:1].copy(), s["venue"][:1].copy(), np.zeros(1, s["ent"].dtype)
            w["e_tier"], w["e_sc"], w["e_venue"] = s["e_tier"][:1], s["e_sc"][:1], s["e_venue"][:1]
            av.append(cut(w, 1)); ameta.append((i, r, "counterparty"))
            av.append(cut(s, 1, np.asarray(v["ts"][:1], np.float64))); ameta.append((i, r, "all marks"))
    A = forecast(av)
    D = pl.DataFrame({"i": [x[0] for x in ameta], "rep": [x[1] for x in ameta], "kind": [x[2] for x in ameta], "elog": A["elog"]})
    D = D.with_columns(pl.Series("role", role[D["i"].to_numpy()]))
    W = D.pivot(on="kind", index=["i", "rep", "role"], values="elog")
    for ro, rn in ((0, "normal"), (1, "scam")):
        w = W.filter(pl.col("role") == ro)
        if len(w) < 5:
            continue
        ii = w["i"].to_numpy()
        say(f"  {rn} (pairs {len(w)}): " + "  ".join(f"{k} {fmt(gboot((w[k] - w['real']).to_numpy(), ii))}" for k in ("kind", "amount", "counterparty", "all marks")))
        same = lambda c: np.mean(w[c].to_numpy() == w["real"].to_numpy())
        say(f"     share of pairs where the swapped mark is identical to the real one (no change): kind {same('kind'):.2f} amount {same('amount'):.2f} counterparty {same('counterparty'):.2f}")
    save()
