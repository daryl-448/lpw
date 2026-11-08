"""Rehearsal reconciliation (reports/p3_joint_addendum.md §J7): compare a val run of scripts/p3_joint/run.py against
the committed val outputs.

usage: uv run python scripts/p3_joint/compare_val.py runs/p3_joint/2018/rehearsal_val [--out FILE.json]
       uv run python scripts/p3_joint/compare_val.py RUN --against EARLIER_RUN   (reproducibility: everything strict)

Classes of comparison:
- strict: XGB and every non-detector quantity, |delta| <= 1e-9, strings identical (P3a / P3b readouts, P3b predictions);
- detector: any value that depends on LR / RF / LGBM; differences are listed for the ledger (detector rule, §J7);
- P1: counts identical, every other field within 1e-4, all three panels, per-row references;
- encoder: re-embedded real val within 1e-4 of the saved embeddings, coverage P / R within 0.01.
Keys that are new in the joint job (not present at val) are exempt and listed.

With --against, two runs of the job are compared instead: every number in every output (detectors included) within
1e-9, strings identical, timings excepted (reports/p3_joint_rehearsal.md, the reproducibility criterion).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

STRICT, P1_TOL, EMB_TOL, COV_TOL = 1e-9, 1e-4, 1e-4, 0.01
NON_XGB = ("lr", "rf", "lgbm")
EXEMPT = {"sec", "eval_split", "eval_groups", "clean", "clean_matched", "giant_fixed", "n_valid_draws", "n_draws",
          "sha256", "sha256_inputs", "environment"}


def detector_dependent(path: tuple) -> bool:
    for p in path:
        ps = str(p)
        if ps in NON_XGB or any(ps.endswith(f"/{d}") or ps.startswith(f"{d}/") or f"/{d}/" in ps for d in NON_XGB):
            return True
        if ps == "ranking_spearman":  # a rank correlation over all four detectors
            return True
    return False


def walk(new, old, path=(), out=None):
    """Collect (path, kind, detail) over the old structure; keys only in new are 'new'."""
    out = out if out is not None else []
    if isinstance(old, dict):
        if not isinstance(new, dict):
            out.append((path, "type", f"{type(new).__name__} vs dict"))
            return out
        for k in old:
            if k in EXEMPT:
                continue
            if k not in new:
                out.append((path + (k,), "missing", None))
            else:
                walk(new[k], old[k], path + (k,), out)
        for k in new:
            if k not in old:
                out.append((path + (k,), "new", None))
    elif isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            out.append((path, "length", f"{len(new) if isinstance(new, list) else type(new).__name__} vs {len(old)}"))
            return out
        for i, (a, b) in enumerate(zip(new, old)):
            walk(a, b, path + (i,), out)
    elif isinstance(old, bool) or isinstance(new, bool) or old is None or new is None or isinstance(old, str):
        if new != old:
            out.append((path, "value", f"{new!r} vs {old!r}"))
    else:
        d = abs(float(new) - float(old))
        if not (d <= STRICT or (np.isnan(float(new)) and np.isnan(float(old)))):
            out.append((path, "num", d))
    return out


def readout_report(new_p: Path, old_p: Path) -> dict:
    new, old = json.loads(new_p.read_text()), json.loads(old_p.read_text())
    diffs = walk(new, old)
    strict = [(p, k, d) for p, k, d in diffs if k != "new" and not detector_dependent(p)]
    det = [(p, k, d) for p, k, d in diffs if k != "new" and detector_dependent(p)]
    newk = sorted({"/".join(map(str, p)) for p, k, _ in diffs if k == "new"})
    fmt = lambda xs: [{"path": "/".join(map(str, p)), "kind": k, "detail": d} for p, k, d in xs]  # noqa: E731
    num = [d for _, k, d in det if k == "num"]
    return {"strict_ok": not strict, "strict_failures": fmt(strict)[:200], "n_strict_failures": len(strict),
            "detector_differences": len(det), "detector_max_abs": max(num) if num else 0.0,
            "detector_examples": fmt(sorted(det, key=lambda x: -(x[2] if x[1] == "num" else 0)))[:40], "new_keys": newk}


def preds_report(new_p: Path, old_p: Path) -> dict:
    zn, zo = np.load(new_p), np.load(old_p)
    strict, det, missing = [], [], sorted(set(zo.files) - set(zn.files))
    for k in zo.files:
        if k not in zn.files:
            continue
        a, b = zn[k], zo[k]
        d = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
        if d > STRICT:
            (det if any(f"/{x}" in k or k.endswith(x) for x in NON_XGB) else strict).append({"key": k, "max_abs": d})
    return {"strict_ok": not strict and not missing, "strict_failures": strict, "missing": missing,
            "detector_differences": sorted(det, key=lambda x: -x["max_abs"]), "n_keys": len(zo.files)}


P1_REFS = {"kct": ("runs/p3_joint/2018/p1_ref/B_val_eval.json", None), "B": ("runs/p3_joint/2018/p1_ref/B_val_eval.json", None),
           "B1_s20261104": ("runs/p1_enc/2018/b1_lr1e-3_s20261104/val_eval.json", None),
           "B1_s20261105": ("runs/p1_enc/2018/b1_lr1e-3_s20261105/val_eval.json", None),
           "B0": ("runs/p1/2018/results_val.json", "b0"), "B2": ("runs/p1/2018/results_val.json", "b2")}
COUNTS = ("prefixes", "owned", "censored")


def p1_compare(run: Path) -> dict:
    tab = json.loads((run / "p1_nll.json").read_text())["rows"]
    fails, worst = [], 0.0
    for row, (ref_p, key) in P1_REFS.items():
        ref = json.loads(Path(ref_p).read_text())
        ref = ref["nll"][key] if key else ref
        for g in ("all", "normal", "scam"):
            for f, v in ref[g].items():
                nv = tab[row][g].get(f, "MISSING")
                if nv is None:  # N/A by design (B2 counterparty / full)
                    continue
                if nv == "MISSING":
                    fails.append(f"{row}/{g}/{f} missing")
                elif f in COUNTS:
                    if nv != v:
                        fails.append(f"{row}/{g}/{f} {nv} vs {v}")
                else:
                    d = abs(nv - v)
                    worst = max(worst, d)
                    if d > P1_TOL:
                        fails.append(f"{row}/{g}/{f} |d| {d:.2e}")
    return {"ok": not fails, "failures": fails, "max_abs": worst}


def enc_compare(run: Path) -> dict:
    z = np.load("runs/p3a/2018/encoder/embeddings.npz")
    e = np.load(run / "enc_real_eval.npz")
    same_shape = e["emb"].shape == z["emb_real_val"].shape and (e["role"] == z["role_real_val"]).all()
    emb_d = float(np.max(np.abs(e["emb"] - z["emb_real_val"]))) if same_shape else float("inf")
    new = json.loads((run / "enc_coverage.json").read_text())
    old = json.loads(Path("runs/p3a/2018/encoder/coverage.json").read_text())
    cov_d = max(abs(new["pr"][r][k][m] - old["pr"][r][k][m]) for r in ("normal", "scam") for k in old["pr"][r] for m in ("precision", "recall"))
    return {"ok": same_shape and emb_d <= EMB_TOL and cov_d <= COV_TOL and new["n_r"] == old["n_r"], "emb_max_abs": emb_d,
            "coverage_max_abs": cov_d, "n_r": [new["n_r"], old["n_r"]]}


def against(run: Path, ref: Path) -> dict:
    """Run-to-run reproducibility: every JSON and npz output of the job, strict, timings excepted."""
    global EXEMPT
    saved, EXEMPT = EXEMPT, {"sec"}
    try:
        rep = {}
        for f in sorted(p.relative_to(ref) for p in ref.rglob("*.json") if not p.name.endswith("manifest.json") and p.name != "job.json"):  # integrity manifests carry
            # run-specific paths and hash timing-bearing files; the payloads they hash are compared themselves
            if not (run / f).exists():
                rep[f.as_posix()] = {"ok": False, "missing": True}
                continue
            d = [(p, k, x) for p, k, x in walk(json.loads((run / f).read_text()), json.loads((ref / f).read_text())) if k != "new"]
            rep[f.as_posix()] = {"ok": not d, "n": len(d), "examples": [{"path": "/".join(map(str, p)), "kind": k, "detail": x}
                                                                          for p, k, x in d[:20]]}
        for f in sorted(p.relative_to(ref) for p in ref.rglob("*.npz")):
            zn, zo = np.load(run / f), np.load(ref / f)
            bad, ulp = [], 0.0  # arrays within STRICT, as for the archive (RF sums its trees in thread order: 2.2e-16)
            for k in zo.files:
                if k not in zn.files or zn[k].shape != zo[k].shape:
                    bad.append(k)
                elif not np.array_equal(zn[k], zo[k], equal_nan=zn[k].dtype.kind == "f"):
                    d = float(np.nanmax(np.abs(zn[k].astype(float) - zo[k].astype(float))))
                    ulp = max(ulp, d)
                    if not d <= STRICT:
                        bad.append(k)
            rep[f.as_posix()] = {"ok": not bad and set(zn.files) == set(zo.files), "bad": bad[:20], "max_abs_nonbitwise": ulp}
        return rep
    finally:
        EXEMPT = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--against", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    if a.against:
        rep = against(run, Path(a.against))
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
        for k, v in rep.items():
            print(f"{k}: {'OK' if v['ok'] else 'FAIL'} {v.get('n', v.get('bad', ''))}")
        sys.exit(0 if all(v["ok"] for v in rep.values()) else 1)
    rep = {}
    if (run / "p1_nll.json").exists():
        rep["p1"] = p1_compare(run)
    if (run / "enc_coverage.json").exists():
        rep["enc"] = enc_compare(run)
    if (run / "p3a" / "readout.json").exists():
        rep["p3a"] = readout_report(run / "p3a" / "readout.json", Path("runs/p3a/2018/readout.json"))
    if (run / "p3b" / "readout.json").exists():
        rep["p3b"] = readout_report(run / "p3b" / "readout.json", Path("runs/p3b/2018/readout.json"))
        rep["p3b_predictions"] = preds_report(run / "p3b" / "readout_predictions.npz", Path("runs/p3b/2018/readout_predictions.npz"))
    if (run / "clean_info.json").exists():
        ci = json.loads((run / "clean_info.json").read_text())
        p0 = json.loads(Path("runs/p0_audit_2018.json").read_text())["leakage"]
        rep["clean"] = {"ok": (ci["n_excluded"], ci["clean_normal"], ci["clean_scam"]) ==
                        (p0["val_agents_sharing_cp_with_train"], p0["val_clean_subset"]["normal"], p0["val_clean_subset"]["scam"]), **ci}
    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt)
    for k, v in rep.items():
        ok = v.get("ok", v.get("strict_ok"))
        print(f"{k}: {'OK' if ok else 'FAIL'} " + json.dumps({x: y for x, y in v.items() if x in ("max_abs", "emb_max_abs", "coverage_max_abs",
                                                                                                  "n_strict_failures", "detector_differences",
                                                                                                  "detector_max_abs", "n_keys")}, default=str))
    sys.exit(0 if all(v.get("ok", v.get("strict_ok")) for v in rep.values()) else 1)


if __name__ == "__main__":
    main()
