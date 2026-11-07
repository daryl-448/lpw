"""Supplementary encoder-space mode coverage (reports/p3a_prereg.md §16 B.2). Run after encoder_fit.py.

usage: uv run python scripts/p3a/encoder_score.py
Writes runs/p3a/2018/encoder/coverage.{json,txt}.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from common import ROOT
from lpw.p1.atomic import write_text
from lpw.p3.utility import precision_recall_k

d = ROOT / "encoder"
meta = json.loads((d / "encoder.json").read_text())
for f, h in meta["sha256"].items():  # artefacts and every recipe / scoring code file
    p = d / f if f in ("encoder.pt", "embeddings.npz") else Path(f)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h, f"{f} changed since freeze"
z = np.load(d / "embeddings.npz")
bad = {k[4:]: int((~np.isfinite(z[k]).all(1)).sum()) for k in z.files if k.startswith("emb_")}
if any(bad.values()):  # an invalid representation is reported, never filtered into a coverage result
    write_text(d / "coverage.json", json.dumps({"status": "INVALID", "nonfinite_rows": bad}, indent=1))
    raise SystemExit(f"INVALID: non-finite embeddings {bad}")
SYN = [f"{arm}_w{w}" for arm in ("kct", "B") for w in (0, 1, 2)]
res, L = {"n_r": {}, "ids": {}, "pr": {}}, []
for rn, r in (("normal", 0), ("scam", 1)):
    pools = {k: np.flatnonzero(z[f"role_{k}"] == r) for k in ["real_val"] + SYN}
    n = min(len(v) for v in pools.values())
    res["n_r"][rn] = int(n)
    if n <= 5:
        res["pr"][rn] = "not estimable"
        continue
    pick = {k: np.sort(np.random.default_rng(0).choice(v, n, replace=False)) for k, v in pools.items()}
    res["ids"][rn] = {k: v.tolist() for k, v in pick.items()}
    R = z["emb_real_val"][pick["real_val"]]
    res["pr"][rn] = {k: precision_recall_k(R, z[f"emb_{k}"][pick[k]], k=5) for k in SYN}
for rn, v in res["pr"].items():
    if isinstance(v, str):
        L.append(f"{rn}: {v} (n_r {res['n_r'][rn]})")
        continue
    for arm in ("kct", "B"):
        P = [v[f"{arm}_w{w}"]["precision"] for w in (0, 1, 2)]; Rc = [v[f"{arm}_w{w}"]["recall"] for w in (0, 1, 2)]
        res.setdefault("summary", {}).setdefault(rn, {})[arm] = {"precision_mean": float(np.mean(P)), "precision_sd": float(np.std(P)),
                                                                "recall_mean": float(np.mean(Rc)), "recall_sd": float(np.std(Rc))}
        L.append(f"{rn} {arm} (n_r {res['n_r'][rn]}): P {np.round(P, 3).tolist()} mean {np.mean(P):.3f}; R {np.round(Rc, 3).tolist()} mean {np.mean(Rc):.3f}")
write_text(d / "coverage.json", json.dumps(res, indent=1, default=float))
write_text(d / "coverage.txt", "\n".join(L) + "\n")
print("\n".join(L))
