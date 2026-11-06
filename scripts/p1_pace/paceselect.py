"""Apply the pre-registered gates of reports/p1_pace_prereg.md to the pace arms and select gamma.

Reads, under ROOT (runs/p1_r0/2018):
  lpw_pace_g{g}/pace.json            known-plan NLL, s-unknown bit identity (full val split)
  lpw_pace_g{g}/feedback_sens.json   calendar-frozen elasticity (arms: --compare ROOT/lpw_pace_g1.00)
  drift_pace_g{g}_{mode}.json        drift_exp output (survival, tail, caps, M-pass, compliance)
  pacescore_pace_{mode}.json         pacescore.py --json output, labels g1.00 / g0.75 / g0.50, control first
Each gate is evaluated on its own: PASS, FAIL, or MISSING (an input file or field is absent or unreadable); every
gate fails on a non-finite input. An arm is rejected if any gate of a stage fails (even when other inputs of the
stage are missing), pending if none fails but one is missing, and only an arm passing every L and O gate gets its
prior (P) gates evaluated. The selection walks the arms from the largest gamma down and stays pending while a larger
gamma is pending, so a smaller gamma is never selected early. Provenance or protocol mismatches (wrong control,
checkpoint, generation file, replica count) are errors.
usage: scripts/p1_pace/paceselect.py ROOT"""
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
CTRL, ARMS = "g1.00", ["g0.75", "g0.50"]  # ARMS in preference order
REPS = 5                                    # replicas per generation (prereg: fixed)
EP_DAYS, EP_REL_FLOOR = ("1", "7", "30"), 0.10
ROLES = ("normal", "scam")


class Missing(Exception):
    pass


class Doc(dict):
    """a loaded JSON object whose absent fields raise Missing with file / field context."""

    def __init__(self, d, src):
        super().__init__(d)
        self.src = src

    def __getitem__(self, k):
        if k not in self:
            raise Missing(f"{self.src}: field {k!r}")
        return _wrap(super().__getitem__(k), f"{self.src}.{k}")


class DocList(list):
    def __init__(self, v, src):
        super().__init__(v)
        self.src = src

    def __getitem__(self, i):
        try:
            v = super().__getitem__(i)
        except IndexError:
            raise Missing(f"{self.src}[{i}]") from None
        return _wrap(v, f"{self.src}[{i}]")


def _wrap(v, src):
    if isinstance(v, dict):
        return Doc(v, src)
    if isinstance(v, list):
        return DocList(v, src)
    return v


_cache = {}


def J(p):
    if p not in _cache:
        if not p.exists():
            raise Missing(str(p))
        try:
            _cache[p] = Doc(json.loads(p.read_text()), p.name)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise Missing(f"{p} (unreadable / partially written: {type(e).__name__})") from None
    return _cache[p]


def fin(*xs):
    """every value a real (non-bool) finite number"""
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def count(x):
    """x if it is a non-bool int, else None"""
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def pace_json(g):
    pj = J(root / f"lpw_pace_{g}" / "pace.json")
    assert abs(pj["gamma"] - float(g[1:])) < 1e-9, (g, pj["gamma"])
    return pj


def drift(g, mode):
    d = J(root / f"drift_pace_{g}_{mode}.json")
    assert Path(d["ckpt"]).name == f"lpw_pace_{g}" and d["mode"] == mode and d["model"] == "lpw_plan_prog_pace", (g, mode, d["ckpt"])
    assert d["save"] and Path(d["save"]).name == f"gen_pace_{g}_{mode}.parquet", (g, mode, d["save"])
    assert count(d["replicas"]) == REPS, (g, mode, d["replicas"])
    return d


def score_file(mode):
    """file-level checks only; per-arm checks are in scored() and g_endpoint(), so one arm's partial record never
    blocks another gate."""
    ps = J(root / f"pacescore_pace_{mode}.json")
    labs = list(ps["arms"])
    if not labs:
        raise Missing(f"pacescore_pace_{mode}.json: no arms")
    assert ps["mode"] == mode and ps["control"] == CTRL and labs[0] == CTRL, f"{mode} pacescore: control must be {CTRL}"
    assert set(ps["arm_paths"]) == set(labs) and set(ps["replicas"]) == set(labs), f"{mode} pacescore: incomplete label maps"
    return ps


def scored(g, mode):
    """the score file with g and the control present, each with five replicas and its own generation (matching its
    drift_exp --save path), and g's comparison against the control."""
    ps = score_file(mode)
    for lab in (CTRL, g):
        if lab not in ps["arms"]:
            raise Missing(f"{lab} in pacescore_pace_{mode}.json")
        assert Path(ps["arm_paths"][lab]).name == f"gen_pace_{lab}_{mode}.parquet", (mode, lab, ps["arm_paths"][lab])
        assert count(ps["replicas"][lab]) == REPS, (mode, lab, ps["replicas"][lab])
        assert Path(ps["arm_paths"][lab]).resolve() == Path(drift(lab, mode)["save"]).resolve(), (lab, mode)
    if g not in ps["vs_control"]:
        raise Missing(f"vs_control {g} in pacescore_pace_{mode}.json")
    assert ps["vs_control"][g]["control"] == CTRL, (mode, g)
    return ps


# ---- gates: each returns (passed, detail) ----------------------------------------------------------------------
def g_L1(g):
    pj, pc = pace_json(g), pace_json(CTRL)
    assert pj["base"] == pc["base"], (pj["base"], pc["base"])
    uc, ucc = pj["unknown_check"], pc["unknown_check"]
    n, nc = count(uc["prefixes"]), count(ucc["prefixes"])
    ok = pj["unknown_bit_identical"] is True and uc["complete"] is True and uc["split"] == "val" and n is not None and n == nc and n > 0
    return ok, f"s-unknown val terms bit-identical to the base on {uc['prefixes']} prefixes ({uc['batches']} batches)"


def g_L2(g):
    pj, pc = pace_json(g), pace_json(CTRL)
    pairs = {r: (pj["known_plan_nll"][r]["full"], pc["known_plan_nll"][r]["full"]) for r in ("all", "normal", "scam")}
    if not all(fin(a, b) for a, b in pairs.values()):
        return False, f"non-finite input {pairs}"
    dl = {r: a - b for r, (a, b) in pairs.items()}
    return all(v <= 0.02 for v in dl.values()), ", ".join(f"{r} {v:+.4f}" for r, v in dl.items()) + " (<= 0.02)"


def g_O1(g):
    sc = scored(g, "oracle")["vs_control"][g]
    rr, lo = sc["rel_reduction"], sc["improvement_lo"]
    return fin(rr, lo) and rr >= 0.25 and lo > 0, f"rel reduction {rr:.3f} (>= 0.25), improvement 95% lo {lo:+.4f} (> 0)"


def g_O2(g):
    fs = J(root / f"lpw_pace_{g}" / "feedback_sens.json")
    if "control" not in fs:
        raise Missing(f"lpw_pace_{g}/feedback_sens.json from feedback.py --compare {root / ('lpw_pace_' + CTRL)}")
    assert Path(fs["control"]).resolve() == (root / f"lpw_pace_{CTRL}").resolve(), fs["control"]
    rr, e, ec = fs["rel_reduction"], fs["mean_e"], fs["control_mean_e"]
    return fin(rr, e, ec) and ec > 0 and rr >= 0.20, f"elasticity {e:+.3f} vs control {ec:+.3f}, reduction {rr:.3f} (>= 0.20)"


def g_caps(g, mode):
    d, c = drift(g, mode), drift(CTRL, mode)
    cap, tail = d["gen_info"]["capped"], d["syn_normal"]["P(n>=1000)"]
    real_tail, ctrl_tail = d["real_normal"]["P(n>=1000)"], c["syn_normal"]["P(n>=1000)"]
    if not fin(cap, tail, real_tail, ctrl_tail):
        return False, "non-finite input"
    lim = max(2 * real_tail, 1.5 * ctrl_tail)
    return cap == 0 and tail <= lim, f"capped {cap} (0), P(n>=1000) {tail:.4f} (<= {lim:.4f})"


def g_endpoint(g, mode):
    """O4 / P4: per role and last-d-days window, with A / C / Rl the arm / control / real shares, s_A / s_C the SDs
    (ddof 1) of the five per-replica shares, SD_R the whole-group bootstrap SD of the real share, H = max(C, Rl):
    (a) A <= H + 3 SD_R (catastrophe guard) and (b) A <= H + max(3 sqrt(s_A^2 / 5 + s_C^2 / 5), 0.10 H)."""
    ps = scored(g, mode)
    bad = []
    for rn in ROLES:
        for dd in EP_DAYS:
            real = ps["real_endpoint"][rn][dd]
            A, C = ps["arms"][g]["endpoint"][rn][dd], ps["arms"][CTRL]["endpoint"][rn][dd]
            ra, rc = list(ps["arms"][g]["endpoint_reps"][rn][dd]), list(ps["arms"][CTRL]["endpoint_reps"][rn][dd])
            assert len(ra) == REPS and len(rc) == REPS, (mode, g, rn, dd, len(ra), len(rc))
            if not fin(A, C, real["share"], real["sd_boot"], *ra, *rc):
                bad.append(f"{rn} last {dd} d: non-finite input")
                continue
            H = max(C, real["share"])
            se = math.sqrt((np.var(ra, ddof=1) + np.var(rc, ddof=1)) / REPS)
            lim_a, lim_b = H + 3 * real["sd_boot"], H + max(3 * se, EP_REL_FLOOR * H)
            if A > min(lim_a, lim_b):
                bad.append(f"{rn} last {dd} d {A:.4f} > {min(lim_a, lim_b):.4f} (guard {lim_a:.4f}, spread {lim_b:.4f})")
    return not bad, "; ".join(bad) or "all shares within limits"


def g_mpass(g, mode):
    d, c = drift(g, mode), drift(CTRL, mode)
    mp = [(rn, d[f"mpass_{rn}"][0], c[f"mpass_{rn}"][0]) for rn in ROLES]
    ok = all(fin(a, b) and a >= b - 0.10 for _, a, b in mp)
    det = ", ".join(f"{rn} {a:.3f} vs control {b:.3f}" for rn, a, b in mp) + " (>= control - 0.10)"
    if mode == "prior":
        ca, cb = d["compliance_normal"], c["compliance_normal"]
        ok = ok and fin(ca, cb) and ca >= cb - 0.05
        det += f"; compliance {ca:.3f} vs control {cb:.3f} (>= control - 0.05)"
    return ok, det


def g_P1(g):
    s = drift(g, "prior")["syn_normal"]["survival"]
    s2, s5 = s["2"], s["5"]
    return fin(s2, s5) and s2 >= 0.767 and s5 >= 0.401, f"P(n>=2) {s2:.3f} (>= 0.767), P(n>=5) {s5:.3f} (>= 0.401)"


def g_P2(g):
    o_rr = scored(g, "oracle")["vs_control"][g]["rel_reduction"]
    sc = scored(g, "prior")["vs_control"][g]
    rr, lo = sc["rel_reduction"], sc["improvement_lo"]
    if not fin(o_rr, rr, lo):
        return False, "non-finite input"
    need = 2 / 3 * o_rr
    return rr >= need and lo > 0, f"rel reduction {rr:.3f} (>= {need:.3f}), improvement 95% lo {lo:+.4f} (> 0)"


STAGES = [
    [("L1 s-unknown identity", g_L1), ("L2 known-plan NLL", g_L2),
     ("O1 pace score", g_O1), ("O2 mechanism", g_O2),
     ("O3 caps / tail", lambda g: g_caps(g, "oracle")), ("O4 endpoint", lambda g: g_endpoint(g, "oracle")),
     ("O5 M-pass", lambda g: g_mpass(g, "oracle"))],
    [("P1 survival", g_P1), ("P2 gain retention", g_P2),
     ("P3 caps / tail", lambda g: g_caps(g, "prior")), ("P4 endpoint", lambda g: g_endpoint(g, "prior")),
     ("P5 M-pass / compliance", lambda g: g_mpass(g, "prior"))],
]


def evaluate(g):
    """-> (status pass / rejected / pending, [(gate, PASS / FAIL / MISSING, detail)])"""
    res = []
    for stage in STAGES:
        part = []
        for name, fn in stage:
            try:
                ok, det = fn(g)
                part.append((name, "PASS" if ok else "FAIL", det))
            except Missing as e:
                part.append((name, "MISSING", str(e)))
        res += part
        if any(s == "FAIL" for _, s, _ in part):
            return "rejected", res
        if any(s == "MISSING" for _, s, _ in part):
            return "pending", res
    return "pass", res


if __name__ == "__main__":
    status = {}
    for g in ARMS:
        st, res = evaluate(g)
        status[g] = st
        print(f"\n== {g}")
        for name, s, det in res:
            print(f"  {s:7s} {name}: {det}")
        print(f"  -> {st}")
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
