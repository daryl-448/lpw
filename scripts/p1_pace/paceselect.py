"""Apply the pre-registered gates of reports/p1_pace_prereg.md to the pace arms and select gamma.

Reads, under ROOT (runs/p1_r0/2018):
  lpw_pace_g{g}/pace.json            known-plan NLL, s-unknown bit identity (full val split)
  lpw_pace_g{g}/feedback_sens.json   calendar-frozen elasticity (arms: --compare ROOT/lpw_pace_g1.00)
  drift_pace_g{g}_{mode}.json        drift_exp output (survival, tail, caps, M-pass, compliance)
  pacescore_pace_{mode}.json         pacescore.py --json output, labels g1.00 / g0.75 / g0.50, control first
Every gate fails on a non-finite input. An arm is rejected at its first failing gate; it is pending while any file it
needs is missing (prior gates are needed only once L and O pass). The selection walks the arms from the largest
gamma down and stays pending while a larger gamma is pending, so a smaller gamma is never selected early.
usage: scripts/p1_pace/paceselect.py ROOT"""
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
CTRL, ARMS = "g1.00", ["g0.75", "g0.50"]  # ARMS in preference order
EP_DAYS, EP_REL_FLOOR = ("1", "7", "30"), 0.10
ROLES = ("normal", "scam")


class Missing(Exception):
    pass


def J(p, what=None):
    if not p.exists():
        raise Missing(what or str(p))
    return json.loads(p.read_text())


def fin(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def gamma_of(label):
    return float(label[1:])


def pace_json(g):
    pj = J(root / f"lpw_pace_{g}" / "pace.json")
    assert abs(pj["gamma"] - gamma_of(g)) < 1e-9, (g, pj["gamma"])
    return pj


def drift(g, mode):
    d = J(root / f"drift_pace_{g}_{mode}.json")
    assert Path(d["ckpt"]).name == f"lpw_pace_{g}" and d["mode"] == mode and d["model"] == "lpw_plan_prog_pace", (g, mode, d["ckpt"])
    assert d.get("save") and Path(d["save"]).name == f"gen_pace_{g}_{mode}.parquet", (g, mode, d.get("save"))
    return d


def score_file(mode):
    ps = J(root / f"pacescore_pace_{mode}.json")
    assert ps["mode"] == mode and ps["control"] == CTRL and list(ps["arms"])[0] == CTRL, f"{mode} pacescore: control must be {CTRL}"
    for lab, path in ps["arm_paths"].items():
        assert Path(path).name == f"gen_pace_{lab}_{mode}.parquet", (mode, lab, path)
    for lab, v in ps["vs_control"].items():
        assert v["control"] == CTRL, (mode, lab, v["control"])
    return ps


def arm_scores(ps, g, mode):
    if g not in ps["arms"] or g not in ps["vs_control"]:
        raise Missing(f"{g} in pacescore_pace_{mode}.json")
    d = drift(g, mode)
    assert Path(ps["arm_paths"][g]).resolve() == Path(d["save"]).resolve(), (g, mode, ps["arm_paths"][g], d["save"])
    return ps["vs_control"][g], d


def endpoint_gate(tag, ps, g):
    """O4 / P4: per role and last-d-days window, with A / C / Rl the arm / control / real shares, s_A / s_C the SDs of
    the per-replica shares, SD_R the whole-group bootstrap SD of the real share, H = max(C, Rl):
    (a) A <= H + 3 SD_R (catastrophe guard) and (b) A <= H + max(3 sqrt(s_A^2 / R_A + s_C^2 / R_C), 0.10 H)."""
    bad = []
    for rn in ROLES:
        for dd in EP_DAYS:
            real = ps["real_endpoint"][rn][dd]
            A, C = ps["arms"][g]["endpoint"][rn][dd], ps["arms"][CTRL]["endpoint"][rn][dd]
            ra, rc = ps["arms"][g]["endpoint_reps"][rn][dd], ps["arms"][CTRL]["endpoint_reps"][rn][dd]
            if not (fin(A, C, real["share"], real["sd_boot"], *ra, *rc) and len(ra) >= 2 and len(rc) >= 2):
                bad.append(f"{rn} last {dd} d: non-finite or < 2 replicas")
                continue
            H = max(C, real["share"])
            se = math.sqrt(np.var(ra, ddof=1) / len(ra) + np.var(rc, ddof=1) / len(rc))
            lim_a, lim_b = H + 3 * real["sd_boot"], H + max(3 * se, EP_REL_FLOOR * H)
            if A > min(lim_a, lim_b):
                bad.append(f"{rn} last {dd} d {A:.4f} > {min(lim_a, lim_b):.4f} (guard {lim_a:.4f}, spread {lim_b:.4f})")
    return (f"{tag}4 endpoint", not bad, "; ".join(bad) or "all shares within limits")


def common_gates(tag, g, mode, ps, d, c):
    out = []
    cap = d["gen_info"].get("capped", float("nan"))
    tail, real_tail, ctrl_tail = d["syn_normal"]["P(n>=1000)"], d["real_normal"]["P(n>=1000)"], c["syn_normal"]["P(n>=1000)"]
    ok = fin(cap, tail, real_tail, ctrl_tail)
    lim = max(2 * real_tail, 1.5 * ctrl_tail) if ok else float("nan")
    out.append((f"{tag}3 caps / tail", ok and cap == 0 and tail <= lim, f"capped {cap} (0), P(n>=1000) {tail:.4f} (<= {lim:.4f})"))
    out.append(endpoint_gate(tag, ps, g))
    mp = [(rn, d[f"mpass_{rn}"][0], c[f"mpass_{rn}"][0]) for rn in ROLES]
    ok5 = all(fin(a, b) and a >= b - 0.10 for _, a, b in mp)
    det = ", ".join(f"{rn} {a:.3f} vs control {b:.3f}" for rn, a, b in mp) + " (>= control - 0.10)"
    if mode == "prior":
        ca, cb = d["compliance_normal"], c["compliance_normal"]
        ok5 = ok5 and fin(ca, cb) and ca >= cb - 0.05
        det += f"; compliance {ca:.3f} vs control {cb:.3f} (>= control - 0.05)"
    out.append((f"{tag}5 M-pass{' / compliance' if mode == 'prior' else ''}", ok5, det))
    return out


def likelihood_gates(g, pj_c):
    pj = pace_json(g)
    uc, ucc = pj["unknown_check"], pj_c["unknown_check"]
    ok1 = bool(pj["unknown_bit_identical"]) and uc["complete"] and uc["split"] == "val" and uc["prefixes"] == ucc["prefixes"] > 0
    assert pj["base"] == pj_c["base"], (pj["base"], pj_c["base"])
    res = [("L1 s-unknown identity", ok1, f"s-unknown val terms bit-identical to the base on {uc['prefixes']} prefixes ({uc['batches']} batches)")]
    dl = {r: pj["known_plan_nll"][r]["full"] - pj_c["known_plan_nll"][r]["full"] for r in ("all", "normal", "scam")}
    res.append(("L2 known-plan NLL", all(fin(v) and v <= 0.02 for v in dl.values()), ", ".join(f"{r} {v:+.4f}" for r, v in dl.items()) + " (<= 0.02)"))
    return res


def oracle_gates(g):
    ps = score_file("oracle")
    sc, d = arm_scores(ps, g, "oracle")
    c = drift(CTRL, "oracle")
    res = [("O1 pace score", fin(sc["rel_reduction"], sc["improvement_lo"]) and sc["rel_reduction"] >= 0.25 and sc["improvement_lo"] > 0,
            f"rel reduction {sc['rel_reduction']:.3f} (>= 0.25), improvement 95% lo {sc['improvement_lo']:+.4f} (> 0)")]
    fs = J(root / f"lpw_pace_{g}" / "feedback_sens.json")
    if "control" not in fs:
        raise Missing(f"lpw_pace_{g}/feedback_sens.json from feedback.py --compare {root / ('lpw_pace_' + CTRL)}")
    assert Path(fs["control"]).resolve() == (root / f"lpw_pace_{CTRL}").resolve(), fs["control"]
    rr = fs["rel_reduction"]
    res.append(("O2 mechanism", fin(rr, fs["mean_e"], fs["control_mean_e"]) and fs["control_mean_e"] > 0 and rr >= 0.20,
                f"elasticity {fs['mean_e']:+.3f} vs control {fs['control_mean_e']:+.3f}, reduction {rr:.3f} (>= 0.20)"))
    return res + common_gates("O", g, "oracle", ps, d, c)


def prior_gates(g, oracle_rel):
    ps = score_file("prior")
    sc, d = arm_scores(ps, g, "prior")
    c = drift(CTRL, "prior")
    s2, s5 = d["syn_normal"]["survival"]["2"], d["syn_normal"]["survival"]["5"]
    res = [("P1 survival", fin(s2, s5) and s2 >= 0.767 and s5 >= 0.401, f"P(n>=2) {s2:.3f} (>= 0.767), P(n>=5) {s5:.3f} (>= 0.401)")]
    need = 2 / 3 * oracle_rel
    res.append(("P2 gain retention", fin(need, sc["rel_reduction"], sc["improvement_lo"]) and sc["rel_reduction"] >= need and sc["improvement_lo"] > 0,
                f"rel reduction {sc['rel_reduction']:.3f} (>= {need:.3f}), improvement 95% lo {sc['improvement_lo']:+.4f} (> 0)"))
    return res + common_gates("P", g, "prior", ps, d, c)


def evaluate(g, pj_c):
    """-> (status in pass / rejected / pending, gate results, what is missing)"""
    res = []
    try:
        for stage in (lambda: likelihood_gates(g, pj_c), lambda: oracle_gates(g)):
            res += stage()
            if not all(ok for _, ok, _ in res):
                return "rejected", res, None
        o_rel = score_file("oracle")["vs_control"][g]["rel_reduction"]
        res += prior_gates(g, o_rel)
    except Missing as e:
        return "pending", res, str(e)
    return ("pass" if all(ok for _, ok, _ in res) else "rejected"), res, None


try:
    pj_c = pace_json(CTRL)
except Missing as e:
    sys.exit(f"control pace.json missing: {e}")
status = {}
for g in ARMS:
    st, res, miss = evaluate(g, pj_c)
    status[g] = st
    print(f"\n== {g}")
    for name, ok, det in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
    print(f"  -> {st}" + (f" (needs {miss})" if miss else ""))
sel = None
for g in ARMS:
    if status[g] == "pass":
        sel = g
        break
    if status[g] == "pending":
        sel = "pending"
        print(f"\nselection pending: {g} is pending and is preferred over every smaller gamma")
        break
if sel != "pending":
    print(f"\nselected: {sel or 'none (pace shrinkage rejected)'}")
