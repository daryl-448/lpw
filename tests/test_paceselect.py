"""Fixture tests for the frozen selection rule scripts/p1_pace/paceselect.py (pending order, control checks, NaN)."""
import copy
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p1_pace" / "paceselect.py"
LABELS = ("g1.00", "g0.75", "g0.50")
ROLES, DAYS = ("normal", "scam"), ("1", "7", "30")
EP = {"1": 0.004, "7": 0.03, "30": 0.11}


def _score(root, mode, labels):
    reps = lambda v: [v - 0.002, v - 0.001, v, v + 0.001, v + 0.002]
    return {"mode": mode, "control": "g1.00", "arm_paths": {l: str(root / f"gen_pace_{l}_{mode}.parquet") for l in labels},
            "replicas": {l: 5 for l in labels},
            "arms": {l: {"score": 0.1, "endpoint": {r: dict(EP) for r in ROLES},
                         "endpoint_reps": {r: {d: reps(EP[d]) for d in DAYS} for r in ROLES}} for l in labels},
            "vs_control": {l: {"rel_reduction": 0.4, "improvement_lo": 0.01, "control": "g1.00"} for l in labels[1:]},
            "real_endpoint": {r: {d: {"share": EP[d], "sd_boot": 0.01} for d in DAYS} for r in ROLES}}


def _drift(root, g, mode):
    return {"ckpt": str(root / f"lpw_pace_{g}"), "model": "lpw_plan_prog_pace", "mode": mode,
            "save": str(root / f"gen_pace_{g}_{mode}.parquet"), "replicas": 5, "gen_info": {"capped": 0},
            "syn_normal": {"P(n>=1000)": 0.001, "survival": {"2": 0.8, "5": 0.45}}, "real_normal": {"P(n>=1000)": 0.001},
            "mpass_normal": [0.7], "mpass_scam": [0.7], "compliance_normal": 0.97}


def build(root, prior_labels=LABELS):
    """every file of a run in which every arm passes; prior files only for prior_labels."""
    for g in LABELS:
        (root / f"lpw_pace_{g}").mkdir(parents=True, exist_ok=True)
        pj = {"base": "base", "gamma": float(g[1:]), "unknown_bit_identical": True,
              "unknown_check": {"split": "val", "prefixes": 63360, "batches": 376, "complete": True},
              "known_plan_nll": {r: {"full": 1.0} for r in ("all", "normal", "scam")}}
        (root / f"lpw_pace_{g}" / "pace.json").write_text(json.dumps(pj))
        (root / f"drift_pace_{g}_oracle.json").write_text(json.dumps(_drift(root, g, "oracle")))
        if g in prior_labels:
            (root / f"drift_pace_{g}_prior.json").write_text(json.dumps(_drift(root, g, "prior")))
        fs = {"mean_e": 0.4, "lo": 0.3, "hi": 0.5, "rows": 100}
        if g != "g1.00":
            fs |= {"mean_e": 0.2, "control": str(root / "lpw_pace_g1.00"), "control_mean_e": 0.4, "rel_reduction": 0.5}
        (root / f"lpw_pace_{g}" / "feedback_sens.json").write_text(json.dumps(fs))
    (root / "pacescore_pace_oracle.json").write_text(json.dumps(_score(root, "oracle", LABELS)))
    if prior_labels:
        (root / "pacescore_pace_prior.json").write_text(json.dumps(_score(root, "prior", prior_labels)))


def edit(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def run(root):
    return subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)


def test_all_pass_selects_the_largest_gamma(tmp_path):
    build(tmp_path)
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "selected: g0.75" in r.stdout


def test_smaller_gamma_is_not_selected_while_a_larger_one_is_pending(tmp_path):
    build(tmp_path, prior_labels=("g1.00", "g0.50"))
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "selection pending: g0.75" in r.stdout and "selected:" not in r.stdout
    g75, g50 = r.stdout.split("== g0.50")
    assert "-> pending (needs" in g75 and "-> pass" in g50


def test_rejected_larger_gamma_falls_through(tmp_path):
    build(tmp_path, prior_labels=("g1.00", "g0.50"))
    edit(tmp_path / "pacescore_pace_oracle.json", lambda d: d["vs_control"]["g0.75"].update(rel_reduction=0.1))
    r = run(tmp_path)
    assert "FAIL  O1 pace score" in r.stdout and "selected: g0.50" in r.stdout


def test_non_finite_endpoint_fails_closed(tmp_path):
    build(tmp_path)
    def nan(d):
        d["arms"]["g0.75"]["endpoint_reps"]["scam"]["7"][2] = float("nan")
    edit(tmp_path / "pacescore_pace_oracle.json", nan)
    r = run(tmp_path)
    assert "FAIL  O4 endpoint" in r.stdout and "non-finite" in r.stdout


def test_control_spread_bound_binds_before_the_guard(tmp_path):
    build(tmp_path)
    # +0.02 on the 30-day share: inside the guard (0.11 + 3 x 0.01) but far outside the replica spread
    def up(d):
        a = d["arms"]["g0.75"]
        a["endpoint"]["normal"]["30"] += 0.02
        a["endpoint_reps"]["normal"]["30"] = [v + 0.02 for v in a["endpoint_reps"]["normal"]["30"]]
    edit(tmp_path / "pacescore_pace_oracle.json", up)
    r = run(tmp_path)
    assert "FAIL  O4 endpoint" in r.stdout and "selected: g0.50" in r.stdout


def test_wrong_controls_are_errors(tmp_path):
    for target, fn in [
        ("pacescore_pace_prior.json", lambda d: d.update(control="g0.75")),
        ("pacescore_pace_prior.json", lambda d: d["vs_control"]["g0.75"].update(control="other")),
        ("lpw_pace_g0.75/feedback_sens.json", lambda d: d.update(control="runs/elsewhere/lpw_pace_g1.00")),
        ("drift_pace_g1.00_oracle.json", lambda d: d.update(ckpt="runs/p1_r0/2018/lpw_plan_prog_rp")),
    ]:
        root = tmp_path / str(abs(hash(target + repr(fn))))
        root.mkdir()
        build(root)
        edit(root / target, fn)
        r = run(root)
        assert r.returncode != 0 and "AssertionError" in r.stderr, (target, r.stdout, r.stderr)
