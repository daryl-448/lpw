"""Joint sealed-test job (reports/p3_joint_addendum.md §J5-§J8): one process computes, for one evaluation split,
  clean  the clean-subset node ids (P0 rule, §J4.3);
  p1     the descriptive P1 likelihood table of six frozen instruments (§J5.3);
  enc    supplementary encoder-space mode coverage with the frozen encoder (§J5.1);
  p3a    the P3a readout (scripts/p3a/readout.py, §J5.1);
  p3b    the P3b readout (scripts/p3b/readout.py, §J5.2).

usage:
  uv run python scripts/p3_joint/run.py --split val --out runs/p3_joint/2018/rehearsal_val    (rehearsal, §J7)
  uv run python scripts/p3_joint/run.py --split test                                         (the one opening, §J8)

Test mode first passes lpw.sealed.authorize(). In both modes everything is written to a staging directory and published
by one atomic rename. Stage output (stdout and stderr) goes to <stage>.log inside the staging directory; the console
shows progress lines only, with no metrics.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lpw import sealed

PILOT = Path("data/pilot_2018")
SCRIPTS = Path(__file__).resolve().parents[1]
P1_ROWS = (  # (row, config, checkpoint dir); joint addendum §J5.3
    ("kct", "lpw_plan_prog_shallow_kct", "runs/p1_shallow/2018/kct"),
    ("B", "lpw_plan_prog", "runs/p1_r0/2018/lpw_plan_prog_rp"),
    ("B0", "b0", "runs/p1/2018/b0"),
    ("B1_s20261104", "b1", "runs/p1_enc/2018/b1_lr1e-3_s20261104"),
    ("B1_s20261105", "b1", "runs/p1_enc/2018/b1_lr1e-3_s20261105"),
    ("B2", "b2", "runs/p1/2018/b2"),
)
P1_NA = {"B2": ("nll_cp", "nll_full", "nll_full_uncensored")}  # B2 has no counterparty head
P1_CAPTION = ("Held-out descriptive likelihood of frozen instruments; P1 remains unqualified under A2. "
              "No P1 exit decision is reopened.")
ENC = Path("runs/p3a/2018/encoder")


def p3a_common():
    spec = importlib.util.spec_from_file_location("p3a_common_joint", SCRIPTS / "p3a" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ stages
def stage_clean(split, out):
    from lpw.p3b.access import clean_agents, load
    R = load(("train", split))
    m = clean_agents(R, split)
    ag = R.agents(split)
    ids = sorted(int(x) for x in R.st.node_id[ag][m])
    (out / "clean_agents.json").write_text(json.dumps(ids))
    y = R.st.role[ag]
    info = {"n_eval": int(len(ag)), "n_excluded": int((~m).sum()), "clean_normal": int((m & (y == 0)).sum()),
            "clean_scam": int((m & (y == 1)).sum())}
    (out / "clean_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def p1_eval_rows(split, st=None):
    """{row: evaluate() panels} for the six frozen instruments, unknown-plan likelihood, identical exact prefixes."""
    import torch

    from lpw.p1.batch import Views
    from lpw.p1.evaluate import evaluate
    from lpw.p1.store import build_store
    from lpw.p1.train import load
    sealed.require(split)
    st = st or build_store(PILOT, splits=("train", split))
    views = Views(st)
    rows = {}
    for name, cfg, run in P1_ROWS:
        m = load(PILOT, cfg, Path(run), st)
        m.plan_inference = False
        t = time.time()
        rows[name] = evaluate(m, st, split, views=views)
        print(f"p1 {name}: {time.time() - t:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    return rows


def same(x, y, tol) -> bool:
    """Nested equality: ints and keys exact, floats within tol (GPU reductions are not bitwise deterministic)."""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k], tol) for k in x)
    if isinstance(x, float) or isinstance(y, float):
        return abs(x - y) <= tol
    return x == y


def stage_p1(split, out):
    rows = p1_eval_rows(split)
    ref = rows["B"]
    for name, r in rows.items():  # identical prefix sets (counts per panel) for every row
        for g in ("all", "normal", "scam"):
            for k in ("prefixes", "owned", "censored"):
                assert r[g][k] == ref[g][k], (name, g, k)
    assert same(rows["kct"], rows["B"], 1e-6), "kct must equal B under the unknown-plan likelihood (§J5.3)"
    for name, keys in P1_NA.items():
        for g in rows[name].values():
            for k in keys:
                g[k] = None
    hashes = {name: sealed.sha256(Path(run) / "model.pt") for name, _, run in P1_ROWS}
    (out / "p1_nll.json").write_text(json.dumps({"caption": P1_CAPTION, "split": split, "plan_inference": False,
                                                 "kct_equals_B": True, "rows": rows, "model_sha256": hashes}, indent=1))


def stage_enc(split, out):
    import torch

    from lpw.p3 import encoder as E
    from lpw.p3.utility import precision_recall_k
    meta = json.loads((ENC / "encoder.json").read_text())
    for f in ("encoder.pt", "embeddings.npz"):  # the frozen artefacts (code is re-frozen in the joint manifest)
        assert sealed.sha256(ENC / f) == meta["sha256"][f], f"{f} changed since the encoder freeze"
    ck = torch.load(ENC / "encoder.pt", map_location="cuda")
    enc = E.Encoder().cuda()
    enc.load_state_dict(ck["state"])
    norm = E.Normalizer.__new__(E.Normalizer)
    norm.mu, norm.sd = np.array(ck["normalizer"]["mu"]), np.array(ck["normalizer"]["sd"])
    C3 = p3a_common()
    st = C3.store_for(split)
    ag, ev, _ = C3.real(st, split)
    emb = E.embed(enc, norm(E.raw_sequences(ag, ev)))
    role = ag["role"].to_numpy()
    np.savez(out / "enc_real_eval.npz", emb=emb, role=role)
    z = np.load(ENC / "embeddings.npz")
    syn = [f"{arm}_w{w}" for arm in ("kct", "B") for w in (0, 1, 2)]
    bad = {"real_eval": int((~np.isfinite(emb).all(1)).sum()), **{k: int((~np.isfinite(z[f"emb_{k}"]).all(1)).sum()) for k in syn}}
    if any(bad.values()):  # encoder_score.py's INVALID exit
        (out / "enc_coverage.json").write_text(json.dumps({"status": "INVALID", "nonfinite_rows": bad}, indent=1))
        return
    res = {"status": "OK", "split": split, "n_r": {}, "ids": {}, "pr": {}}
    pools_emb = {"real_eval": emb, **{k: z[f"emb_{k}"] for k in syn}}
    pools_role = {"real_eval": role, **{k: z[f"role_{k}"] for k in syn}}
    for rn, r in (("normal", 0), ("scam", 1)):  # scripts/p3a/encoder_score.py's recipe, real val -> the evaluation split
        pools = {k: np.flatnonzero(pools_role[k] == r) for k in pools_emb}
        n = min(len(v) for v in pools.values())
        res["n_r"][rn] = int(n)
        if n <= 5:
            res["pr"][rn] = "not estimable"
            continue
        pick = {k: np.sort(np.random.default_rng(0).choice(v, n, replace=False)) for k, v in pools.items()}
        res["ids"][rn] = {k: v.tolist() for k, v in pick.items()}
        Rm = pools_emb["real_eval"][pick["real_eval"]]
        res["pr"][rn] = {k: precision_recall_k(Rm, pools_emb[k][pick[k]], k=5) for k in syn}
    (out / "enc_coverage.json").write_text(json.dumps(res, indent=1, default=float))


def run_script(path: Path, argv: list[str]):
    """Run a readout script in this process (so lpw.sealed's authorization applies), with its own directory first on
    sys.path and no stale 'common' module from another script directory."""
    sys.modules.pop("common", None)
    sys.path.insert(0, str(path.parent))
    old = sys.argv
    sys.argv = [str(path), *argv]
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old
        sys.path.remove(str(path.parent))
        sys.modules.pop("common", None)


def stage_p3a(split, out, a):
    d = out / "p3a"
    d.mkdir()
    run_script(SCRIPTS / "p3a" / "readout.py", ["--split", split, "--out", str(d), "--clean", str(out / "clean_agents.json"),
                                               "--worlds", a.worlds, "--arms", "kct,B", "--boot", str(a.boot)])


def stage_p3b(split, out, a):
    d = out / "p3b"
    d.mkdir()
    run_script(SCRIPTS / "p3b" / "readout.py", ["--split", split, "--out", str(d), "--worlds", a.worlds, "--boot", str(a.boot),
                                               "--seeds", str(a.seeds)])


STAGES = ("clean", "p1", "enc", "p3a", "p3b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--out", default=None, help="val only; test always publishes runs/p3_joint/2018/test")
    ap.add_argument("--worlds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--stages", default=",".join(STAGES), help="val only: a subset of stages, for development")
    a = ap.parse_args()
    stages = a.stages.split(",")
    if a.split == "test":
        frozen = (a.worlds, a.boot, a.seeds, stages) == ("0,1,2", 1000, 20, list(STAGES))
        if a.out is not None or not frozen:
            raise SystemExit("test mode runs the frozen settings only, publishing runs/p3_joint/2018/test")
        if overrides := [k for k in ("P3A_ROOT", "P3B_ROOT") if k in os.environ]:  # the frozen artifact roots only
            raise SystemExit(f"test mode refuses artifact-root overrides: {overrides}")
        out = sealed.TEST_OUT
        authorized = sealed.authorize(out=out)
    else:
        if a.out is None:
            raise SystemExit("--out is required for --split val")
        out, authorized = Path(a.out), None
    staging = sealed.staging_dir(out)
    if out.exists() or staging.exists():
        raise SystemExit(f"{out} or its staging directory exists; refusing to overwrite")
    staging.mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    console = sys.__stderr__
    timing, t0 = {}, time.time()
    for name in stages:
        console.write(f"{time.strftime('%H:%M:%S')} stage {name} ...\n")
        console.flush()
        t = time.time()
        with open(staging / f"{name}.log", "w") as lg, contextlib.redirect_stdout(lg), contextlib.redirect_stderr(lg):
            {"clean": lambda: stage_clean(a.split, staging), "p1": lambda: stage_p1(a.split, staging),
             "enc": lambda: stage_enc(a.split, staging), "p3a": lambda: stage_p3a(a.split, staging, a),
             "p3b": lambda: stage_p3b(a.split, staging, a)}[name]()
        timing[name] = round(time.time() - t)
        console.write(f"{time.strftime('%H:%M:%S')} stage {name} done ({timing[name]}s)\n")
        console.flush()
    import importlib.metadata as md
    job = {"addendum": "reports/p3_joint_addendum.md", "split": a.split, "head": head, "dirty_tree": dirty,
           "authorized_commit": authorized, "argv": sys.argv, "stages": stages, "timing_sec": timing,
           "total_sec": round(time.time() - t0),
           "environment": {"python": platform.python_version(), "uv.lock": sealed.sha256(Path("uv.lock")),
                           **{k: md.version(k) for k in ("numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "polars", "torch")}},
           "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (staging / "job.json").write_text(json.dumps(job, indent=1))
    sealed.publish(staging, out)
    console.write(f"published {out} ({job['total_sec']}s)\n")


if __name__ == "__main__":
    main()
