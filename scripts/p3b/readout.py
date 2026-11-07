"""P3b frozen development readout on real VAL (reports/p3b_prereg.md §P7-§P10). One run; every outcome is kept.

Sections: (1) utility: TRTR / TSTR on L, L+S, S, full, full+S (+ confonly, noprof+S for XGB), the primary R/T/D/H
contrasts with the whole-group bootstrap, secondary analyses (STRUCT-only retention, full-based contrasts, matched
pairs, giant-group fixed-score and graph-removal sensitivities), F1, difficulty and profile-gain parity;
(2) C2ST (group-aware) and the shortcut audit; (3) coupling fidelity: the eight-item core family on 20 half-view
draws, full-view descriptive distances and cohort / age-tercile marginals; (4) SPEC §7.3 daily topology and Clauset
fits; (5) coupling diagnostics.

Frozen sampling choices (prereg §P8, implementation detail fixed here):
- real half views, seed s: perm = default_rng(s).permutation(N_val); A = perm[:n], B = perm[n:2n];
- synthetic n-subset of a hold view, seed s, world w: default_rng(SeedSequence([s, w])).choice(N_hold, n, replace=False),
  shared by every attachment arm of that (policy arm, world);
- role subsampling inside a (s, w, policy arm) triple: default_rng(SeedSequence([s, w, arm_index, role])) draws, in
  the order A, B, synthetic, shared by the attachment arms;
- daily selection: default_rng(SeedSequence([6000 + s, w, day])) draws m_d agents from A, then B, then the synthetic
  subset (shared by the attachment arms and the anchor).
Writes runs/p3b/2018/readout.{json,txt} and readout_predictions.npz, readout_samples.npz.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lpw.p1.store import OUT  # noqa: E402
from lpw.p3.features import SETS, matrix  # noqa: E402
from lpw.p3.matching import MIN_PAIRS, Matcher  # noqa: E402
from lpw.p3.utility import (DETECTORS, PRIMARY, GroupBoot, fit_score, matched_prevalence_ap, scores, shortcut_audit,  # noqa: E402
                            transfer_contrast)
from lpw.p3b import evaluate as E  # noqa: E402
from lpw.p3b.access import load, real_pop  # noqa: E402
from lpw.p3b.graph import CORE, STRUCT, core_measurements, struct_features, w1  # noqa: E402
from lpw.p3b.replay import replay  # noqa: E402
from lpw.p3b.topology import DAILY, S_AGENT, S_PERIPHERY, S_VENUE, clauset, clauset_inputs, daily_metrics, node_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--worlds", default="0,1,2")
ap.add_argument("--tag", default="")
ap.add_argument("--skip", default="", help="comma list of sections to skip (plumbing smoke tests only)")
ap.add_argument("--plumb", action="store_true",
                help="plumbing smoke test with NO val access: loads train only; pseudo-val = every 5th train group by id")
a = ap.parse_args()
if a.plumb:
    assert a.tag, "plumbing runs must be tagged"
WORLDS = [int(w) for w in a.worlds.split(",")]
SKIP = set(filter(None, a.skip.split(",")))
SENTINELS = ("end_balance", "net_flow", "ends_near_zero", "sweep_out_share", "min_balance")
ATT_ALL = ("p3b0", "p3b1", None)  # None = the P3a zero-sharing anchor
t0 = time.time()
L, res, PRED = [], {"worlds": WORLDS, "boot": a.boot, "seeds": a.seeds}, {}


def log(s):
    print(s, flush=True)
    L.append(s)


def stack(f):
    return np.stack([f[c] for c in STRUCT], 1)


def fsets(F, S):
    Lm, full = matrix(F, "local"), matrix(F, "full")
    return {"L": Lm, "L+S": np.hstack([Lm, S]), "S": S, "full": full, "full+S": np.hstack([full, S]),
            "confonly": matrix(F, "confonly"), "noprof+S": np.hstack([matrix(F, "noprof"), S])}


ALL4 = ("L", "L+S", "S", "full", "full+S")
XGB_ONLY = ("confonly", "noprof+S")
S_DEP = ("L+S", "S", "full+S", "noprof+S")

# ------------------------------------------------------------------ real
R = load(("train",)) if a.plumb else load()
if a.plumb:
    _tr = R.st.agents("train")
    _pv = (R.group[_tr] % 5) == 0
    _SPL = {"train": _tr[~_pv], "val": _tr[_pv]}
    R.agents = lambda split: _SPL[split]  # instance override: real_view / real_pop / real_transactions see the pseudo split
st = R.st
Atr, Ava = R.agents("train"), R.agents("val")
agT, evT, FT, (BT, nT) = C.real_view(R, "train")
agV, evV, FV, (BV, nV) = C.real_view(R, "val")
ytr, yva = st.role[Atr], st.role[Ava]
gtr, gva = R.group[Atr], R.group[Ava]
prev_va = float(yva.mean())
ST, SV = stack(struct_features(BT)), stack(struct_features(BV))
XT, XV = fsets(FT, ST), fsets(FV, SV)
ug, uc = np.unique(gva, return_counts=True)
giant = int(ug[np.lexsort((ug, -uc))[0]])
keep = np.flatnonzero(gva != giant)
SV_rm = stack(struct_features(BV[keep]))
XV_rm = fsets(FV[pl.Series(keep)], SV_rm)
boot = GroupBoot(gva, B=a.boot)
MATCH = Matcher(FT, ytr).match(FV, yva)  # the saved P3a rule; reproduces P3a's 348 val pairs on the guarded store
if not a.plumb:
    assert MATCH["n_pairs"] == 348, MATCH["n_pairs"]
res["matching"] = {k: v for k, v in MATCH.items() if k != "pairs"}
res["real"] = {"n_train": int(len(ytr)), "n_val": int(len(yva)), "prev_val": prev_va, "giant_group": giant,
               "giant_size": int((gva == giant).sum()), "giant_scam": int(yva[gva == giant].sum())}
log(f"real: train {len(ytr)}, val {len(yva)}; giant val group {giant} ({(gva == giant).sum()} agents, {yva[gva == giant].sum()} scam)")

# ------------------------------------------------------------------ synthetic bases
def synth_base(w, arm, part):
    ag, ev, F, info = C.p3a.synth(st, w, arm, part)
    raw = C.p3a.raw_events(w, arm, part)
    conds, _ = C.p3a.load_pop(w, part)
    kept = np.flatnonzero(np.bincount(raw["agent"].to_numpy(), minlength=len(conds)) > 0)
    assert len(kept) == ag.height
    grp = np.load(C.ROOT / f"world{w}" / f"groups_{part}.npz")["group"][kept]
    return {"ag": ag, "ev": ev, "F": F, "info": info, "group": grp, "y": ag["role"].to_numpy()}


def synth_incidence(base, w, arm, part, att):
    from lpw.p3b.graph import incidence
    ev = base["ev"]
    elig = ev["venue"].to_numpy() < 0
    if att is None:
        local = np.where(elig, ev["agent"].to_numpy() * (1 << 32) + ev["cp"].to_numpy(), 0)
    else:
        local = C.sidecar(w, arm, part, att)["node"]
        assert len(local) == ev.height and ((local >= 0) == elig).all()
    node = np.where(elig, node_key(S_PERIPHERY, np.where(elig, local, 0)), -1)
    B, nodes = incidence(ev["agent"].to_numpy()[elig], node[elig], base["ag"].height)
    return B, nodes, node


SYN = {}
for arm in C.ARMS:
    for w in WORLDS:
        for part in ("train", "hold"):
            b = synth_base(w, arm, part)
            b["inc"] = {att: synth_incidence(b, w, arm, part, att) for att in ATT_ALL}
            b["S"] = {att: stack(struct_features(b["inc"][att][0])) for att in ATT_ALL}
            SYN[(arm, w, part)] = b
log(f"loaded synthetic views ({time.time() - t0:.0f}s)")

# ------------------------------------------------------------------ (1) utility
if "utility" not in SKIP:
    trtr = {}
    for fs in ALL4 + XGB_ONLY:
        for d in (DETECTORS if fs in ALL4 else (PRIMARY,)):
            trtr[(fs, d)] = fit_score(d, XT[fs], ytr, [XV[fs], XV_rm[fs]])
    real_ap_mp = {d: matched_prevalence_ap(yva, trtr[("full+S", d)][0], prev_va) for d in DETECTORS}
    real_pgain = {m: scores(yva, trtr[("full+S", PRIMARY)][0])[m] - scores(yva, trtr[("noprof+S", PRIMARY)][0])[m] for m in ("auc", "ap")}
    thr_real = E.f1_threshold_grouped(PRIMARY, XT["full+S"], ytr, gtr)
    res["trtr"] = {f"{fs}/{d}": scores(yva, s[0]) for (fs, d), s in trtr.items()}
    res["trtr"][f"full+S/{PRIMARY}"]["f1"] = scores(yva, trtr[("full+S", PRIMARY)][0], thr_real)["f1"]
    log("TRTR AUC: " + " ".join(f"{k} {v['auc']:.4f}" for k, v in res["trtr"].items() if k.endswith(PRIMARY)))
    for (fs, d), s in trtr.items():
        PRED[f"trtr/{fs}/{d}"] = s[0]
    names = SETS["full"] + STRUCT
    sentinels = tuple(SENTINELS) + tuple(STRUCT)
    U = {}
    for arm in C.ARMS:
        tstr = {}   # (fs, d, att) -> list over worlds of (val, val_rm, hold) scores
        per_world = {}
        for w in WORLDS:
            bS, bH = SYN[(arm, w, "train")], SYN[(arm, w, "hold")]
            yS, yH = bS["y"], bH["y"]
            wr = {"O_info": {"train": bS["info"], "hold": bH["info"]}, "att": {}}
            for att in ("p3b0", "p3b1"):
                XS, XH = fsets(bS["F"], bS["S"][att]), fsets(bH["F"], bH["S"][att])
                for fs in ALL4 + XGB_ONLY:
                    if fs not in S_DEP and att == "p3b1":   # attachment-invariant sets: shared with p3b0
                        for d in (DETECTORS if fs in ALL4 else (PRIMARY,)):
                            tstr.setdefault((fs, d, att), []).append(tstr[(fs, d, "p3b0")][-1])
                        continue
                    for d in (DETECTORS if fs in ALL4 else (PRIMARY,)):
                        tstr.setdefault((fs, d, att), []).append(fit_score(d, XS[fs], yS, [XV[fs], XV_rm[fs], XH[fs]]))
                g = lambda fs, d="xgb": tstr[(fs, d, att)][-1]  # noqa: E731
                thr = E.f1_threshold_grouped(PRIMARY, XS["full+S"], yS, bS["group"])
                ar = {"tstr": {f"{fs}/{d}": scores(yva, tstr[(fs, d, att)][-1][0]) for fs in ALL4 + XGB_ONLY
                               for d in (DETECTORS if fs in ALL4 else (PRIMARY,))}}
                ar["tstr"][f"full+S/{PRIMARY}"]["f1"] = scores(yva, g("full+S")[0], thr)["f1"]
                ar["difficulty_parity"] = {}
                for d in DETECTORS:
                    sa = matched_prevalence_ap(yH, tstr[("full+S", d, att)][-1][2], prev_va)
                    ar["difficulty_parity"][d] = {"synth_ap": sa, "real_ap": real_ap_mp[d], "delta": sa - real_ap_mp[d]}
                sp = {m: scores(yH, g("full+S")[2])[m] - scores(yH, g("noprof+S")[2])[m] for m in ("auc", "ap")}
                ar["profile_gain"] = {m: {"synth": sp[m], "real": real_pgain[m]} for m in ("auc", "ap")}
                cs = {}
                for fs in ("S", "L+S", "full+S"):
                    v = {}
                    for r, rn in ((0, "normal"), (1, "scam")):
                        v[rn] = E.c2st_grouped(XV[fs][yva == r], gva[yva == r], XH[fs][yH == r], bH["group"][yH == r])
                    ok = None not in v.values()
                    cs[fs] = {**v, "gap": v["scam"] - v["normal"] if ok else None,
                              "pass": bool(ok and v["normal"] <= 0.70 and v["scam"] <= 0.70 and v["scam"] - v["normal"] <= 0.05) if ok else None}
                ar["c2st"] = cs
                ar["shortcut"] = shortcut_audit(names, XT["full+S"], ytr, XV["full+S"], yva, XS["full+S"], yS, sentinels)
                wr["att"][att] = ar
                log(f"{arm} w{w} {att}: TSTR XGB L {ar['tstr']['L/xgb']['auc']:.4f} L+S {ar['tstr']['L+S/xgb']['auc']:.4f} "
                    f"S {ar['tstr']['S/xgb']['auc']:.4f} full+S {ar['tstr']['full+S/xgb']['auc']:.4f} | C2ST S "
                    f"{cs['S']['normal']}/{cs['S']['scam']} L+S {cs['L+S']['normal']}/{cs['L+S']['scam']} ({time.time() - t0:.0f}s)")
            per_world[w] = wr
        for k, v in tstr.items():
            for w, s in zip(WORLDS, v):
                PRED[f"tstr/{arm}/{k[2]}/{k[0]}/{k[1]}/w{w}"] = s[0]
        A = {"worlds": per_world}
        # primary and detector sensitivity: L-based incremental contrasts
        A["incremental_L"] = {d: E.incremental(yva, boot.idx, trtr[("L", d)][0], trtr[("L+S", d)][0], [s[0] for s in tstr[("L", d, "p3b0")]],
                                               {att: [s[0] for s in tstr[("L+S", d, att)]] for att in ("p3b0", "p3b1")}) for d in DETECTORS}
        A["incremental_full"] = {d: E.incremental(yva, boot.idx, trtr[("full", d)][0], trtr[("full+S", d)][0],
                                                  [s[0] for s in tstr[("full", d, "p3b0")]],
                                                  {att: [s[0] for s in tstr[("full+S", d, att)]] for att in ("p3b0", "p3b1")}) for d in DETECTORS}
        A["struct_only"] = {att: transfer_contrast(yva, [s[0] for s in tstr[("S", PRIMARY, att)]], trtr[("S", PRIMARY)][0], boot) for att in ("p3b0", "p3b1")}
        # matched pairs (saved P3a rule), full-split STRUCT
        M = MATCH
        pidx = np.concatenate([M["pairs"][:, 0], M["pairs"][:, 1]])
        mt = {"n_pairs": int(M["n_pairs"]), "testable": bool(M["n_pairs"] >= MIN_PAIRS)}
        if mt["testable"]:
            mt["L"] = E.point_incremental(yva, trtr[("L", PRIMARY)][0], trtr[("L+S", PRIMARY)][0], [s[0] for s in tstr[("L", PRIMARY, "p3b0")]],
                                          {att: [s[0] for s in tstr[("L+S", PRIMARY, att)]] for att in ("p3b0", "p3b1")}, pidx)
            mt["full"] = E.point_incremental(yva, trtr[("full", PRIMARY)][0], trtr[("full+S", PRIMARY)][0], [s[0] for s in tstr[("full", PRIMARY, "p3b0")]],
                                             {att: [s[0] for s in tstr[("full+S", PRIMARY, att)]] for att in ("p3b0", "p3b1")}, pidx)
            au = lambda s: scores(yva[pidx], s[pidx])["auc"]  # noqa: E731
            mt["behavioural_gain"] = {"trtr": au(trtr[("full+S", PRIMARY)][0]) - au(trtr[("confonly", PRIMARY)][0]),
                                      **{f"tstr_{att}": float(np.mean([au(s[0]) for s in tstr[("full+S", PRIMARY, att)]]) -
                                                              np.mean([au(s[0]) for s in tstr[("confonly", PRIMARY, att)]])) for att in ("p3b0", "p3b1")}}
        A["matched"] = mt
        # giant group: fixed scores without it; graph removal (recomputed val STRUCT, same models)
        A["giant_fixed"] = E.point_incremental(yva, trtr[("L", PRIMARY)][0], trtr[("L+S", PRIMARY)][0], [s[0] for s in tstr[("L", PRIMARY, "p3b0")]],
                                               {att: [s[0] for s in tstr[("L+S", PRIMARY, att)]] for att in ("p3b0", "p3b1")}, keep)
        A["giant_graph_removed"] = E.point_incremental(yva[keep], trtr[("L", PRIMARY)][1], trtr[("L+S", PRIMARY)][1],
                                                       [s[1] for s in tstr[("L", PRIMARY, "p3b0")]],
                                                       {att: [s[1] for s in tstr[("L+S", PRIMARY, att)]] for att in ("p3b0", "p3b1")})
        # arm-level means over worlds
        ws = list(per_world.values())
        agg = {}
        for att in ("p3b0", "p3b1"):
            m = lambda f: float(np.mean([f(x["att"][att]) for x in ws]))  # noqa: E731
            ag_ = {"c2st": {}, "difficulty_parity": {}, "profile_gain": {}, "shortcut": {}}
            for fs in ("S", "L+S", "full+S"):
                vals = [x["att"][att]["c2st"][fs] for x in ws]
                if any(v["normal"] is None or v["scam"] is None for v in vals):
                    ag_["c2st"][fs] = {"pass": None, "status": "NOT TESTABLE"}
                    continue
                nm, sm = np.mean([v["normal"] for v in vals]), np.mean([v["scam"] for v in vals])
                ag_["c2st"][fs] = {"normal": float(nm), "scam": float(sm), "gap": float(sm - nm), "pass": bool(nm <= 0.70 and sm <= 0.70 and sm - nm <= 0.05)}
            for d in DETECTORS:
                sa = m(lambda x: x["difficulty_parity"][d]["synth_ap"])
                ag_["difficulty_parity"][d] = {"synth_ap": sa, "real_ap": real_ap_mp[d], "delta": sa - real_ap_mp[d], "pass": bool(abs(sa - real_ap_mp[d]) <= 0.05)}
            for mm in ("auc", "ap"):
                gsy = m(lambda x: x["profile_gain"][mm]["synth"])
                ag_["profile_gain"][mm] = {"synth": gsy, "real": real_pgain[mm], "pass": bool(np.sign(gsy) == np.sign(real_pgain[mm]) and abs(gsy - real_pgain[mm]) <= 0.05)}
            for f in ws[0]["att"][att]["shortcut"]:
                dl = m(lambda x: x["shortcut"][f]["delta"])
                ag_["shortcut"][f] = {"real": ws[0]["att"][att]["shortcut"][f]["real"], "delta": dl, "flag": bool(abs(dl) > 0.1),
                                      "sentinel": ws[0]["att"][att]["shortcut"][f]["sentinel"]}
            ag_["tstr_mean"] = {k: m(lambda x: x["tstr"][k]["auc"]) for k in ws[0]["att"][att]["tstr"]}
            ag_["f1_mean"] = m(lambda x: x["tstr"][f"full+S/{PRIMARY}"]["f1"])
            agg[att] = ag_
        A["aggregate"] = agg
        U[arm] = A
        p = A["incremental_L"][PRIMARY]
        log(f"{arm} PRIMARY (XGB, L): R {p['point']['R']:+.4f} T0 {p['point']['T0']:+.4f} T1 {p['point']['T1']:+.4f} D {p['point']['D']:+.4f} "
            f"H {p['point']['H']:+.4f} | utility {p['utility']} | learning {p['learning']} | valid draws {p['n_valid_draws']}")
        for att in ("p3b0", "p3b1"):
            so = A["struct_only"][att]
            log(f"{arm} {att} STRUCT-only: TSTR {so['tstr_auc_mean']:.4f} TRTR {so['trtr_auc']:.4f} D {so['D']:+.4f} {so['outcome']}; "
                f"C2ST {json.dumps(agg[att]['c2st'])}; flags {[f for f, v in agg[att]['shortcut'].items() if v['flag']]}")
    res["utility"] = U
    res["f1_threshold_real"] = thr_real

# ------------------------------------------------------------------ (3) coupling fidelity: core family
N_val = len(yva)
n = min(N_val // 2, min(SYN[(arm, w, "hold")]["ag"].height // 2 for arm in C.ARMS for w in WORLDS))
res["core_n"] = n


def half_views(s):
    perm = np.random.default_rng(s).permutation(N_val)
    return np.sort(perm[:n]), np.sort(perm[n:2 * n])


def measure(B, nodes, rows):
    Bs = B[rows]
    d = np.asarray(Bs.sum(0)).ravel()
    Bs = Bs[:, np.flatnonzero(d > 0)]
    return core_measurements(Bs, nodes[np.flatnonzero(d > 0)])


RECOVERED = {}


def gstats(g):
    _, sz = np.unique(g, return_counts=True)
    return _size_stats(sz)


def sizes_stats(agent_size):
    """gstats from per-agent community sizes (each community of size k contributes k agents of size k)."""
    k, n_k = np.unique(agent_size, return_counts=True)
    assert (n_k % k == 0).all()
    return _size_stats(np.repeat(k, n_k // k))


def _size_stats(sz):
    agent_sz = np.repeat(sz, sz)
    H = float(((sz / sz.sum()) ** 2).sum())
    return {"n": int(sz.sum()), "n_groups": int(len(sz)), "max": int(sz.max()), "max_share": float(sz.max() / sz.sum()), "H": H, "eff_groups": 1 / H,
            "groupw_q": np.quantile(sz, [0.5, 0.9, 0.99]).tolist(), "agentw_q": np.quantile(agent_sz, [0.5, 0.75, 0.9, 0.99]).tolist()}


ROLE_ITEMS = ("shared_frac", "other_agents_frac", "sharing_component_frac")
samples = {}
if "core" not in SKIP:
    acc = {}  # (arm, att) -> item -> list over (w, s) of (d, f)
    real_cache = {}
    for s in range(a.seeds):
        A_, B_ = half_views(s)
        samples[f"real/s{s}"] = np.stack([A_, B_])
        mA, mB = measure(BV, nV, A_), measure(BV, nV, B_)
        for ai, arm in enumerate(C.ARMS):
            for w in WORLDS:
                bH = SYN[(arm, w, "hold")]
                sub = np.sort(np.random.default_rng(np.random.SeedSequence([s, w])).choice(bH["ag"].height, n, replace=False))
                samples[f"{arm}/w{w}/s{s}"] = sub
                yA, yB, yS = yva[A_], yva[B_], bH["y"][sub]
                ridx = {}
                for r in (0, 1):
                    m_r = min((yA == r).sum(), (yB == r).sum(), (yS == r).sum())
                    rg = np.random.default_rng(np.random.SeedSequence([s, w, ai, r]))
                    ridx[r] = (m_r, [np.sort(rg.choice(np.flatnonzero(yy == r), m_r, replace=False)) if m_r else np.zeros(0, int)
                                     for yy in (yA, yB, yS)])
                for att in ("p3b0", "p3b1"):
                    Bh, nh, _ = bH["inc"][att]
                    mS = measure(Bh, nh, sub)
                    for it, r in [(f"{k}/{rn}", r) for k in ROLE_ITEMS for r, rn in ((0, "normal"), (1, "scam"))]:
                        k = it.split("/")[0]
                        m_r, (iA, iB, iS) = ridx[r]
                        dv = (w1(mS[k][iS], mA[k][iA]), w1(mA[k][iA], mB[k][iB])) if m_r >= 20 else (float("nan"), float("nan"))
                        acc.setdefault((arm, att), {}).setdefault(it, []).append((w, s) + dv)
                    for k in ("log1p_node_degree", "log1p_community_agents"):
                        acc.setdefault((arm, att), {}).setdefault(k, []).append((w, s, w1(mS[k], mA[k]), w1(mA[k], mB[k])))
        log(f"core seed {s} done ({time.time() - t0:.0f}s)")
    core = {}
    for (arm, att), items in acc.items():
        out = {"items": {}}
        verdict = "PASS"
        for it in CORE:
            rows = np.array(items[it], dtype=np.float64)
            if np.isnan(rows[:, 2:]).any():
                out["items"][it] = {"estimable": False}
                verdict = "INCOMPLETE"
                continue
            per_w = {int(w): (float(rows[rows[:, 0] == w, 2].mean()), float(rows[rows[:, 0] == w, 3].mean())) for w in WORLDS}
            dm, fm = float(np.mean([v[0] for v in per_w.values()])), float(np.mean([v[1] for v in per_w.values()]))
            ok = dm <= 2 * fm if fm > 0 else dm == 0
            out["items"][it] = {"d": dm, "f": fm, "ratio": dm / fm if fm > 0 else None, "pass": bool(ok),
                                "per_world": {w: {"d": v[0], "f": v[1], "pass": bool(v[0] <= 2 * v[1] if v[1] > 0 else v[0] == 0)} for w, v in per_w.items()}}
            if not ok and verdict == "PASS":
                verdict = "FAIL"
        out["verdict"] = "CORE " + verdict
        core[f"{arm}/{att}"] = out
        log(f"core {arm}/{att}: {out['verdict']}; " + "; ".join(f"{it} d {v['d']:.4f} f {v['f']:.4f}" for it, v in out["items"].items() if v.get("d") is not None))
    for arm in C.ARMS:
        core[f"{arm}/p3b1_minus_p3b0"] = {it: core[f"{arm}/p3b1"]["items"][it].get("d", np.nan) - core[f"{arm}/p3b0"]["items"][it].get("d", np.nan) for it in CORE}
    res["core"] = core
    # full-view descriptive distances and marginals
    tr_pre = (FT["cohort"].to_numpy() == 0)
    cuts = np.quantile(FT["log_age"].to_numpy()[tr_pre], [1 / 3, 2 / 3])

    STRATA = {"age_t1": lambda b: b == 0, "age_t2": lambda b: b == 1, "age_t3": lambda b: b == 2, "pre": lambda b: b <= 2,
              "born": lambda b: b == 3}

    def bins(F):
        coh, la = F["cohort"].to_numpy(), F["log_age"].to_numpy()
        return np.where(coh == 1, 3, np.searchsorted(cuts, la, side="right"))  # 0..2 terciles, 3 = born

    mV = measure(BV, nV, np.arange(N_val))
    bV = bins(FV)
    fullv = {}
    for arm in C.ARMS:
        for w in WORLDS:
            bH = SYN[(arm, w, "hold")]
            for att in ATT_ALL:
                Bh, nh, _ = bH["inc"][att]
                mS = measure(Bh, nh, np.arange(bH["ag"].height))
                yS, bS_ = bH["y"], bins(bH["F"])
                rec = {}
                for k in ROLE_ITEMS:
                    for r, rn in ((0, "normal"), (1, "scam")):
                        rec[f"{k}/{rn}"] = w1(mS[k][yS == r], mV[k][yva == r])
                        for bn, f in STRATA.items():
                            rec[f"{k}/{rn}/{bn}"] = w1(mS[k][(yS == r) & f(bS_)], mV[k][(yva == r) & f(bV)])
                for k in ("log1p_node_degree", "log1p_community_agents"):
                    rec[k] = w1(mS[k], mV[k])
                for bn, f in STRATA.items():  # item 8 is agent-level: same marginals, roles pooled as in the item
                    rec[f"log1p_community_agents/{bn}"] = w1(mS["log1p_community_agents"][f(bS_)], mV["log1p_community_agents"][f(bV)])
                fullv[f"{arm}/w{w}/{att or 'anchor'}"] = rec
                RECOVERED[f"{arm}/w{w}/{att or 'anchor'}"] = sizes_stats(np.expm1(mS["log1p_community_agents"]).round().astype(np.int64))
    RECOVERED["real_val"] = sizes_stats(np.expm1(mV["log1p_community_agents"]).round().astype(np.int64))
    fullv["age_tercile_cuts_log_age"] = cuts.tolist()
    res["core_full_view"] = fullv
    log(f"full-view core distances done ({time.time() - t0:.0f}s)")

# ------------------------------------------------------------------ (4) daily topology and Clauset
def synth_tx(bH, att):
    """Synthetic transaction table of a hold view: one row per event (distinct transactions)."""
    ev = bH["ev"]
    _, _, node = bH["inc"][att]
    ag_ = ev["agent"].to_numpy()
    me = node_key(S_AGENT, ag_)
    ven = ev["venue"].to_numpy()
    cp = np.where(ven >= 0, node_key(S_VENUE, np.maximum(ven, 0)), node)
    out = ev["kind"].to_numpy() == OUT
    return {"u": np.where(out, me, cp), "v": np.where(out, cp, me), "day": ev["ts"].to_numpy() // 86400, "eth": ev["eth"].to_numpy(),
            "venue": ven >= 0, "agent": ag_}


def real_tx_view(rows):
    T = C.real_transactions(R, Ava[rows])
    ids = st.node_id[Ava[rows]].astype(np.uint64)
    return T, ids


def by_day(T, focal_of_row):
    """day -> (row indices, focal agents active that day)."""
    order = np.argsort(T["day"], kind="stable")
    days, start = np.unique(T["day"][order], return_index=True)
    out = {}
    for i, d in enumerate(days):
        rr = order[start[i]: start[i + 1] if i + 1 < len(start) else len(order)]
        out[int(d)] = (rr, np.unique(np.concatenate(focal_of_row(rr))))
    return out


_SIDES = {}


def real_sides(s):
    """Per real half view of seed s: (transactions, focal-sender mask, focal-receiver mask, day -> (rows, active))."""
    if s not in _SIDES:
        out = {}
        for nm, rows in zip(("A", "B"), half_views(s)):
            T, ids = real_tx_view(rows)
            isu, isv = np.isin(T["u"], ids), np.isin(T["v"], ids)
            bd = by_day(T, lambda rr, T=T, isu=isu, isv=isv: [T["u"][rr][isu[rr]], T["v"][rr][isv[rr]]])
            out[nm] = (T, isu, isv, bd)
        _SIDES[s] = out
    return _SIDES[s]


if "daily" not in SKIP:
    daily = {}
    md_store = {}
    for arm in C.ARMS:
        for w in WORLDS:
            bH = SYN[(arm, w, "hold")]
            Tsyn = {att: synth_tx(bH, att) for att in ATT_ALL}
            for s in range(a.seeds):
                sub = np.sort(np.random.default_rng(np.random.SeedSequence([s, w])).choice(bH["ag"].height, n, replace=False))
                sides = real_sides(s)
                subset = np.zeros(bH["ag"].height, bool)
                subset[sub] = True
                T0s = Tsyn["p3b0"]
                rows_sub = np.flatnonzero(subset[T0s["agent"]])
                bd_syn = by_day({"day": T0s["day"][rows_sub]}, lambda rr: [T0s["agent"][rows_sub][rr]])
                bd_syn = {d: (rows_sub[rr], act) for d, (rr, act) in bd_syn.items()}
                days = sorted(set(sides["A"][3]) | set(sides["B"][3]) | set(bd_syn))
                for d in days:
                    act = [sides["A"][3].get(d, (None, np.zeros(0)))[1], sides["B"][3].get(d, (None, np.zeros(0)))[1], bd_syn.get(d, (None, np.zeros(0)))[1]]
                    m_d = int(min(len(x) for x in act))
                    md_store.setdefault(f"{arm}/w{w}/s{s}", []).append((d, m_d))
                    if m_d == 0:
                        continue
                    rg = np.random.default_rng(np.random.SeedSequence([6000 + s, w, d]))
                    sel = [rg.choice(x, m_d, replace=False) for x in act]
                    for venue in (True, False):
                        vt = "ven" if venue else "noven"
                        met = {}
                        for k, nm in enumerate(("A", "B")):
                            T, isu, isv, bd = sides[nm]
                            rr = bd[d][0]
                            m = np.isin(T["u"][rr], sel[k]) | np.isin(T["v"][rr], sel[k])
                            if not venue:
                                m &= ~T["venue"][rr]
                            met[nm] = daily_metrics(T["u"][rr][m].astype(np.int64), T["v"][rr][m].astype(np.int64)) if m.any() else {k_: float("nan") for k_ in DAILY}
                        rr = bd_syn[d][0]
                        selm = np.isin(T0s["agent"][rr], sel[2])
                        for att in ATT_ALL:
                            Ts = Tsyn[att]
                            m = selm & (True if venue else ~Ts["venue"][rr])
                            ms = daily_metrics(Ts["u"][rr][m], Ts["v"][rr][m]) if m.any() else {k_: float("nan") for k_ in DAILY}
                            for k_ in DAILY:
                                daily.setdefault((arm, att or "anchor", vt, k_), []).append((w, s, d, ms[k_], met["A"][k_], met["B"][k_]))
            log(f"daily {arm} w{w} done ({time.time() - t0:.0f}s)")
    dres = {}
    for (arm, att, vt, k_), rows in daily.items():
        x = np.array(rows, dtype=np.float64)
        samples[f"daily/{arm}/{att}/{vt}/{k_}"] = x  # columns: world, seed, day, synth, A, B (NaN = undefined)
        defined = np.isfinite(x[:, 3:6]).all(1)
        share = float(defined.mean())
        disc = float(np.abs(x[defined, 3] - x[defined, 4]).mean()) if defined.any() else None
        scale = float(np.abs(x[defined, 5] - x[defined, 4]).mean()) if defined.any() else None
        dres[f"{arm}/{att}/{vt}/{k_}"] = {"defined_share": share, "discrepancy": disc, "ab_scale": scale,
                                         "ratio": disc / scale if disc is not None and scale else None,
                                         "within_2x": bool(share >= 0.8 and disc is not None and scale is not None and disc <= 2 * scale) if share >= 0.8 else None}
    res["daily"] = dres
    res["daily_md"] = {k: {"n_days": len(v), "excluded_days": [int(d) for d, m in v if m == 0], "md_median": float(np.median([m for _, m in v]))}
                       for k, v in md_store.items()}
    for k, v in md_store.items():
        samples[f"daily_md/{k}"] = np.array(v, np.int64)  # (day, m_d), every common day incl. m_d = 0
    for arm in C.ARMS:
        for att in ("p3b0", "p3b1", "anchor"):
            log(f"daily {arm}/{att}/ven: " + " ".join(f"{k_} {dres[f'{arm}/{att}/ven/{k_}']['ratio']:.2f}" if dres[f'{arm}/{att}/ven/{k_}']['ratio'] is not None
                                                     else f"{k_} n/a" for k_ in DAILY))


def clauset_view(u, v, eth):
    x = clauset_inputs(u, v, eth)
    return {nm: clauset(x[nm], disc) for nm, disc in (("unique_degree", True), ("tx_degree", True), ("strength", False), ("amount", False))}


if "clauset" not in SKIP:
    cl = {}
    Tv = C.real_transactions(R, Ava)
    cl["real_val"] = clauset_view(Tv["u"].astype(np.int64), Tv["v"].astype(np.int64), Tv["eth"])
    for arm in C.ARMS:
        for w in WORLDS:
            bH = SYN[(arm, w, "hold")]
            for att in ATT_ALL:
                T = synth_tx(bH, att)
                cl[f"{arm}/w{w}/{att or 'anchor'}"] = clauset_view(T["u"], T["v"], T["eth"])
        log(f"clauset {arm} done ({time.time() - t0:.0f}s)")
    res["clauset"] = cl
    log("clauset real val: " + "; ".join(f"{k} a {v.get('alpha', float('nan')):.3f} xmin {v.get('xmin')} gof {v.get('gof_p')}" for k, v in cl["real_val"].items()))

# ------------------------------------------------------------------ (5) coupling diagnostics
diag = {}


diag["groups_real_val"] = gstats(gva)
diag["recovered_communities"] = RECOVERED  # Louvain on each full view (empty if the core section was skipped)
for arm in C.ARMS:
    for w in WORLDS:
        for part in ("train", "hold"):
            G = np.load(C.ROOT / f"world{w}" / f"groups_{part}.npz")
            diag[f"groups_assigned/w{w}/{part}"] = gstats(G["group"])
            diag[f"groups_retained/{arm}/w{w}/{part}"] = gstats(SYN[(arm, w, part)]["group"])
Tv = C.real_transactions(R, Ava)
vids = st.node_id[Ava].astype(np.uint64)
both = np.isin(Tv["u"], vids) & np.isin(Tv["v"], vids)
idx = np.concatenate([np.arange(st.off[x], st.off[x + 1]) for x in Ava])
diag["agent_agent_internal_tx_share_val"] = float(both.mean())
diag["agent_cp_event_share_val"] = float(R.cp_agent[idx].mean())
coo = BV.tocoo()
dV = np.asarray(BV.sum(0)).ravel()
gmin = np.full(BV.shape[1], np.iinfo(np.int64).max)
gmax = np.full(BV.shape[1], np.iinfo(np.int64).min)
np.minimum.at(gmin, coo.col, gva[coo.row])
np.maximum.at(gmax, coo.col, gva[coo.row])
sh = dV >= 2
diag["val_cross_group_share_of_shared_incidence"] = float(((gmin != gmax) & sh)[coo.col].sum() / sh[coo.col].sum())
pv, _ = real_pop(R, "val")
rv = replay(pv, keep_ptr=False)
diag["val_cocreate_share"] = float(rv.opp_cocreate.mean())
diag["val_first_touch_nonempty"] = float((rv.opp_nc > 0).mean())
res["diagnostics"] = diag
log(f"diagnostics: {json.dumps({k: v for k, v in diag.items() if not isinstance(v, dict)})}")

# ------------------------------------------------------------------ outputs
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# matching, folds and bootstrap recipe (all deterministic; stored so they can be checked without recomputation)
res["matched_pairs_node_ids"] = [[int(st.node_id[Ava[i]]), int(st.node_id[Ava[j]])] for i, j in MATCH["pairs"]]
samples["folds/real_train_f1"] = E.group_folds(gtr)
samples["folds/real_val_c2st_groups"] = gva
for (arm, w, part), b in SYN.items():
    samples[f"folds/{arm}/w{w}/{part}_f1"] = E.group_folds(b["group"])
res["bootstrap"] = {"recipe": "lpw.p3.utility.GroupBoot(real val Louvain group ids, B, seed 0): B draws of len(groups) group ids with "
                              "replacement; fixed predictions re-scored", "B": a.boot,
                    "draws_sha256": hashlib.sha256(np.concatenate(boot.idx).astype(np.int64).tobytes()).hexdigest()}
res["c2st_recipe"] = "lpw.p3b.evaluate.c2st_grouped per role: seed-0 balanced subsample, group_folds on each side, domain weights"
res["feature_order"] = {k: SETS[k] for k in ("local", "full", "confonly", "noprof")} | {"STRUCT": STRUCT}
INPUTS = sorted(str(p) for p in list(Path("data/pilot_2018").glob("*")) + [Path("data/raw/gnn_dataset/labels/targets_global.parquet")] + list(C.ROOT.glob("prior/*")) + list(C.ROOT.glob("kernel/*"))
                + [q for w in WORLDS for q in (C.ROOT / f"world{w}").glob("*.npz")]
                + [q for w in WORLDS for q in (C.p3a.ROOT / f"world{w}").glob("*") if q.suffix in (".parquet", ".npz")])
res["sha256_inputs"] = {p: sha(p) for p in INPUTS}
import importlib.metadata as md  # noqa: E402
import platform  # noqa: E402
res["environment"] = {"python": platform.python_version(), "uv.lock": sha("uv.lock"), "pyproject.toml": sha("pyproject.toml"),
                      **{k: md.version(k) for k in ("numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "polars", "networkx", "powerlaw")}}
CODE = ["src/lpw/p3b/access.py", "src/lpw/p3b/replay.py", "src/lpw/p3b/groups.py", "src/lpw/p3b/kernel.py", "src/lpw/p3b/graph.py",
        "src/lpw/p3b/topology.py", "src/lpw/p3b/evaluate.py", "scripts/p3b/common.py", "scripts/p3b/readout.py", "src/lpw/p3/utility.py",
        "src/lpw/p3/features.py", "src/lpw/p3/observe.py", "src/lpw/p3/matching.py", "src/lpw/p1/store.py",
        "scripts/p3a/common.py"]
res["sha256"] = {c: sha(c) for c in CODE}
res["sec"] = round(time.time() - t0)
outs = [C.ROOT / f"readout_predictions{a.tag}.npz", C.ROOT / f"readout_samples{a.tag}.npz", C.ROOT / f"readout{a.tag}.json",
        C.ROOT / f"readout{a.tag}.txt"]
np.savez_compressed(outs[0], **PRED)
np.savez_compressed(outs[1], **samples)
C.save_json(outs[2], res)
outs[3].write_text("\n".join(L) + "\n")
C.save_json(C.ROOT / f"readout_manifest{a.tag}.json", {str(p): sha(p) for p in outs})
log(f"done {res['sec']}s")
