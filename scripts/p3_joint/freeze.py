"""Write the frozen manifest of the joint sealed-test job (reports/p3_joint_addendum.md §J2, §J8.1).

usage: uv run python scripts/p3_joint/freeze.py --rehearsal runs/p3_joint/2018/rehearsal_val_<final> --ledger FILE
Writes reports/p3_joint_manifest.json. The manifest itself is committed in the freeze commit; lpw.sealed.check()
verifies every sha256 in it before the test split can be loaded.
"""
import argparse
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

from lpw import sealed

WORLDS, ARMS, PARTS, ATT = (0, 1, 2), ("kct", "B"), ("train", "hold"), ("p3b0", "p3b1")
P1 = ("runs/p1_shallow/2018/kct", "runs/p1_r0/2018/lpw_plan_prog_rp", "runs/p1/2018/b0", "runs/p1_enc/2018/b1_lr1e-3_s20261104",
      "runs/p1_enc/2018/b1_lr1e-3_s20261105", "runs/p1/2018/b2")


def files(a) -> list[Path]:
    code = [Path(p) for p in subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "src/lpw", "scripts/p3a", "scripts/p3b", "scripts/p3_joint"],
                                           capture_output=True, text=True, check=True).stdout.split() if p.endswith(".py")]
    out = code + [Path("pyproject.toml"), Path("uv.lock"), Path("reports/p3_joint_addendum.md")]
    out += sorted(Path("data/pilot_2018").glob("*")) + [Path("data/raw/gnn_dataset/labels/targets_global.parquet")]
    for w in WORLDS:  # P3a worlds: populations, raw events, generation record
        d = Path(f"runs/p3a/2018/world{w}")
        out += [d / f"pop_{p}.npz" for p in PARTS] + [d / f"{arm}_{p}.parquet" for arm in ARMS for p in PARTS] + [d / "gen.json"]
        e = Path(f"runs/p3b/2018/world{w}")  # P3b group maps and attachment sidecars
        out += [e / f"groups_{p}.npz" for p in PARTS] + [e / f"{arm}_{p}_{att}.npz" for arm in ARMS for p in PARTS for att in ATT]
    out += [Path("runs/p3a/2018/encoder") / f for f in ("encoder.pt", "embeddings.npz", "encoder.json")]
    out += sorted(Path("runs/p3b/2018/prior").glob("*")) + sorted(Path("runs/p3b/2018/kernel").glob("*"))
    for d in P1:  # frozen P1 instruments
        out += [Path(d) / f for f in ("model.pt", "train.json", "recal.json", "shallow.json") if (Path(d) / f).exists()]
    # the development references the rehearsal reproduced, and the rehearsal itself
    out += [Path(p) for p in ("runs/p3a/2018/readout.json", "runs/p3b/2018/readout.json", "runs/p3b/2018/readout_predictions.npz",
                              "runs/p3a/2018/encoder/coverage.json", "runs/p3_joint/2018/p1_ref/B_val_eval.json",
                              "runs/p1_enc/2018/b1_lr1e-3_s20261104/val_eval.json", "runs/p1_enc/2018/b1_lr1e-3_s20261105/val_eval.json",
                              "runs/p1/2018/results_val.json", "runs/p0_audit_2018.json")]
    out += [Path(a.rehearsal) / "output_manifest.json", Path(a.ledger)]
    missing = [p for p in out if not p.exists()]
    assert not missing, f"missing frozen inputs: {missing}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", required=True)
    ap.add_argument("--ledger", required=True)
    a = ap.parse_args()
    fs = files(a)
    import torch
    man = {"addendum": "reports/p3_joint_addendum.md", "rehearsal": a.rehearsal, "ledger": a.ledger,
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                           **{k: md.version(k) for k in ("numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "polars")}},
           "sha256": {p.as_posix(): sealed.sha256(p) for p in fs}}
    sealed.MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {sealed.MANIFEST}: {len(fs)} files")


if __name__ == "__main__":
    main()
