"""P1 freeze readiness audit (SPEC A2; reports/p1_exit.md). Saved 2018 val streams only: no training, no generation.

usage: uv run python scripts/p1_freeze/audit.py [--gen runs/p1_shallow/2018/gen] [--out runs/p1_freeze/2018] [--boot 200]

For B and kct, oracle and prior mode, all saved replicas:
- **safety:** capped streams (>= 10000), stream-length tail (max, 99.9th percentile), the longest equal-timestamp
  burst, non-finite numeric fields, and plan violations (finite plan with N > 3U; zero plan with N > 0);
- **role parity (SPEC §7.2):** an XGBoost role detector fitted on REAL train agents (lpw.p1.fidelity.role_parity),
  AUC on real val vs on the synthetic val streams, with and without the profile columns. Whole-group bootstrap of
  the delta: the same resampled groups on both sides;
- **single-feature shortcut gaps (§7.4 shortcut audit, on P1's behaviour summaries):** per feature,
  AUC(role | feature) on synthetic minus on real. |delta| > 0.1 is flagged.

Writes the freeze manifest (commit, sha256 of checkpoints and scorer code, seeds, scope) to OUT/manifest.json.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from lpw.p1.atomic import write_text
from lpw.p1.fidelity import Sample, agent_table, real_frames, role_parity
from lpw.p1.plan import COUNT_EDGES, N_PLAN_N
from lpw.p1.store import build_store

PILOT = Path("data/pilot_2018")
ARMS = ("B", "kct")
MODES = ("oracle", "prior")
CAP = 10000
FEATS = ["repeat_rate", "distinct_cp", "n_events", "span_days", "out_frac", "hour_entropy", "monthly_net", "end_balance"]
N_BEH = len(FEATS)
HASHED = ["runs/p1_r0/2018/lpw_plan_prog_rp/model.pt", "runs/p1_shallow/2018/kct/model.pt",
          "runs/p1_shallow/2018/kct/shallow.json", "runs/p1_shallow/2018/fit.json",
          "src/lpw/p1/fidelity.py", "src/lpw/p1/plan.py", "src/lpw/p1/generate.py", "src/lpw/p1/model.py",
          "src/lpw/p1/shallow.py", "src/lpw/p1/drift_exp.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="runs/p1_shallow/2018/gen")
    ap.add_argument("--out", default="runs/p1_freeze/2018")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    gen, out = Path(a.gen), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = build_store(PILOT)
    va, tr = st.agents("val"), st.agents("train")
    ag = pl.read_parquet(PILOT / "agents.parquet")
    gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    ug, ginv = np.unique([gid_of[int(st.node_id[x])] for x in va], return_inverse=True)
    rag, rev = real_frames(st, va)
    real = Sample(rag, rev, st.w_start)
    tag_, tev = real_frames(st, tr)
    prof_va, prof_tr = np.stack([st.prof_vec(x) for x in va]), np.stack([st.prof_vec(x) for x in tr])
    train_X = agent_table(Sample(tag_, tev, st.w_start), prof_tr)
    real_X = agent_table(real, prof_va)
    y_tr, y = st.role[tr], st.role[va]
    ag_s = rag.select("agent", "role", "cohort", "origin", "b0", "age_days")
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(ug), (a.boot, len(ug)))
    members = [np.nonzero(ginv == g)[0] for g in range(len(ug))]

    def boot_delta(score_r, score_s):
        ds = []
        for d in draws:
            idx = np.concatenate([members[g] for g in d])
            if len(np.unique(y[idx])) < 2:
                continue
            ds.append(roc_auc_score(y[idx], score_s[idx]) - roc_auc_score(y[idx], score_r[idx]))
        return float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))

    from xgboost import XGBClassifier
    det = {}
    for name, cols in (("with_profile", slice(None)), ("behaviour_only", slice(0, N_BEH))):
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            random_state=0, n_jobs=8, eval_metric="logloss").fit(train_X[:, cols], y_tr)
        det[name] = (clf, cols, clf.predict_proba(real_X[:, cols])[:, 1])
    single_real = {f: float(roc_auc_score(y, np.nan_to_num(real_X[:, i]))) for i, f in enumerate(FEATS)}

    res = {"safety": {}, "parity": {}, "shortcuts": {}, "single_real": single_real}
    L = []
    for arm in ARMS:
        for mode in MODES:
            key = f"{arm}_{mode}"
            ev = pl.read_parquet(gen / f"{key}.parquet")
            P = dict(np.load(gen / f"{key}.plans.npz"))
            saf, par, sc = [], {n: [] for n in det}, {f: [] for f in FEATS}
            for r in sorted(ev["replica"].unique().to_list()):
                e = ev.filter(pl.col("replica") == r).drop("replica")
                N = np.bincount(e["agent"].to_numpy(), minlength=len(va))
                pn = P["plan_n"][r]
                fin = pn < N_PLAN_N - 1
                U = np.where(fin, COUNT_EDGES[np.minimum(pn + 1, N_PLAN_N - 1)] - 1, np.iinfo(np.int64).max)
                burst = e.group_by("agent", "ts").len()["len"].max()
                nonfin = sum(int((~np.isfinite(e[c].to_numpy().astype(float))).sum()) for c in ("amt", "eth", "fee"))
                saf.append({"capped": int((N >= CAP).sum()), "max_len": int(N.max()), "p999_len": float(np.quantile(N, 0.999)),
                            "max_same_ts_burst": int(burst), "nonfinite": nonfin,
                            "over3U": int((fin & (N > 3 * U)).sum()), "zero_plan_nonempty": int(((pn == 0) & (N > 0)).sum())})
                syn = Sample(ag_s, e, st.w_start)
                syn_X = agent_table(syn, prof_va)
                for n, (clf, cols, sr) in det.items():
                    ss = clf.predict_proba(syn_X[:, cols])[:, 1]
                    lo, hi = boot_delta(sr, ss)
                    ar, as_ = roc_auc_score(y, sr), roc_auc_score(y, ss)
                    par[n].append({"auc_real": float(ar), "auc_synth": float(as_), "delta": float(as_ - ar), "ci95": [lo, hi]})
                for i, f in enumerate(FEATS):
                    sc[f].append(float(roc_auc_score(y, np.nan_to_num(syn_X[:, i]))) - single_real[f])
            if arm == "kct" and mode == "prior":  # consistency with the library's role_parity on one replica
                e0 = ev.filter(pl.col("replica") == 0).drop("replica")
                rp = role_parity(train_X, y_tr, real_X, y, agent_table(Sample(ag_s, e0, st.w_start), prof_va), y)
                assert abs(rp["delta"] - par["with_profile"][0]["delta"]) < 1e-9, (rp, par["with_profile"][0])
            res["safety"][key], res["parity"][key] = saf, par
            res["shortcuts"][key] = {f: {"mean": float(np.mean(v)), "per_replica": v, "flag": bool(abs(np.mean(v)) > 0.1)} for f, v in sc.items()}
            L.append(f"{key}:")
            L.append("  safety per replica: " + " | ".join(
                f"cap {s['capped']} max {s['max_len']} p99.9 {s['p999_len']:.0f} burst {s['max_same_ts_burst']} nonfin {s['nonfinite']} "
                f">3U {s['over3U']} zero-plan {s['zero_plan_nonempty']}" for s in saf))
            for n, v in par.items():
                d = np.array([x["delta"] for x in v])
                L.append(f"  role-AUC {n:14s} real {v[0]['auc_real']:.3f} synth " + " ".join(f"{x['auc_synth']:.3f}" for x in v)
                         + f" | delta mean {d.mean():+.3f}, |delta| <= 0.05 in {int((np.abs(d) <= 0.05).sum())}/{len(d)} replicas; "
                         + "group CI95 " + " ".join(f"[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" for x in v))
            L.append("  single-feature AUC gap (synth - real): " + " ".join(
                f"{f} {v['mean']:+.3f}{'!' if v['flag'] else ''}" for f, v in res["shortcuts"][key].items()))
    L.insert(0, "real val single-feature role AUC: " + " ".join(f"{f} {v:.3f}" for f, v in single_real.items()))
    write_text(out / "audit.json", json.dumps(res, indent=1))
    write_text(out / "audit.txt", "\n".join(L) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    manifest = {"spec": "SPEC.md amendment A2", "commit": commit, "dirty_tracked_files": bool(dirty), "scope": "2018 slice; P3 feasibility",
                "primary": {"name": "kct", "config": "lpw_plan_prog_shallow_kct", "ckpt": "runs/p1_shallow/2018/kct",
                            "base_ckpt": "runs/p1_r0/2018/lpw_plan_prog_rp"},
                "control": {"name": "B", "config": "lpw_plan_prog", "ckpt": "runs/p1_r0/2018/lpw_plan_prog_rp"},
                "plan_prior": "lpw.p1.plan.PlanPrior(store), default (no exposure inputs), fitted on train; plans drawn with rng 101",
                "generation": "lpw.p1.drift_exp seed 7, max_events 10000; P1 diagnostics used 5 replicas",
                "scorer": "lpw.p1.fidelity.compare with per-group rngs (amended 2026-11-07)",
                "sha256": {p: sha(p) for p in HASHED}}
    write_text(out / "manifest.json", json.dumps(manifest, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
