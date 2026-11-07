"""Fit the addendum-B encoder on real TRAIN and embed every coverage source (reports/p3a_prereg.md §16).

usage: uv run python scripts/p3a/encoder_fit.py [--throughput N]   (N > 0: time N training batches, write nothing)
Writes runs/p3a/2018/encoder/{encoder.pt, embeddings.npz, encoder.json} (refuses to overwrite). No val scoring here:
val and the synthetic holdouts are only embedded, and the embedding hashes are frozen before encoder_score.py runs.
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT, real, store, synth
from lpw.p1.atomic import write_text
from lpw.p3 import encoder as E

WORLDS, ARMS = (0, 1, 2), ("kct", "B")
CODE = ("src/lpw/p3/encoder.py", "src/lpw/p3/utility.py", "src/lpw/p3/observe.py", "scripts/p3a/common.py",
        "scripts/p3a/encoder_fit.py", "scripts/p3a/encoder_score.py")  # recipe and scoring code, verified before scoring
ap = argparse.ArgumentParser()
ap.add_argument("--throughput", type=int, default=0)
a = ap.parse_args()
out = ROOT / "encoder"
assert not (out / "encoder.pt").exists(), "the encoder is frozen: refusing to refit"
t0 = time.time()
st = store()
ag_tr, ev_tr, _ = real(st, "train")
raw_tr = E.raw_sequences(ag_tr, ev_tr)
norm = E.Normalizer(raw_tr)
seq_tr = norm(raw_tr)
if a.throughput:
    torch.manual_seed(0); rng = np.random.default_rng(0)
    enc = E.Encoder().cuda(); opt = torch.optim.AdamW(enc.parameters(), lr=E.LR, weight_decay=E.WD)
    order = rng.permutation(len(seq_tr)); t1 = time.time()
    for b in range(a.throughput):
        idx = order[b * E.BATCH:(b + 1) * E.BATCH]
        loss = E.nt_xent(enc([torch.as_tensor(E.crop(seq_tr[i], rng), device="cuda") for i in idx]),
                         enc([torch.as_tensor(E.crop(seq_tr[i], rng), device="cuda") for i in idx]))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    per = (time.time() - t1) / a.throughput
    print(f"{per:.3f}s/batch -> {per * len(seq_tr) / E.BATCH * E.EPOCHS / 60:.1f} min for {E.EPOCHS} epochs")
    raise SystemExit
out.mkdir(parents=True, exist_ok=True)
enc, losses = E.train(seq_tr)
t_fit = time.time() - t0
buf = io.BytesIO()
torch.save({"state": enc.state_dict(), "normalizer": norm.state(), "n_in": E.N_IN}, buf)
(out / "encoder.pt").write_bytes(buf.getvalue())
emb = {"real_train": E.embed(enc, seq_tr)}
role = {"real_train": ag_tr["role"].to_numpy()}
ag_va, ev_va, _ = real(st, "val")
emb["real_val"], role["real_val"] = E.embed(enc, norm(E.raw_sequences(ag_va, ev_va))), ag_va["role"].to_numpy()
for w in WORLDS:
    for arm in ARMS:
        ag, ev, _, _ = synth(st, w, arm, "hold")
        k = f"{arm}_w{w}"
        emb[k], role[k] = E.embed(enc, norm(E.raw_sequences(ag, ev))), ag["role"].to_numpy()
np.savez(out / "embeddings.npz", **{f"emb_{k}": v for k, v in emb.items()}, **{f"role_{k}": v for k, v in role.items()})
Z = emb["real_train"]
checks = {"losses": losses, "loss_finite": bool(np.isfinite(losses).all()),
          "embeddings_finite": {k: bool(np.isfinite(v).all()) for k, v in emb.items()},
          "train_coord_sd": Z.std(0).tolist(), "train_coord_sd_min": float(Z.std(0).min()),
          "train_mean_cosine_random_pairs": float(np.mean(np.sum(Z[np.random.default_rng(0).integers(0, len(Z), 5000)]
                                                                * Z[np.random.default_rng(1).integers(0, len(Z), 5000)], 1)))}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
meta = {"addendum": "reports/p3a_prereg.md §16", "fit_sec": round(t_fit), "total_sec": round(time.time() - t0), "checks": checks,
        "sha256": {"encoder.pt": sha(out / "encoder.pt"), "embeddings.npz": sha(out / "embeddings.npz"),
                   **{c: sha(Path(c)) for c in CODE}},
        "counts": {k: int(len(v)) for k, v in emb.items()}}
write_text(out / "encoder.json", json.dumps(meta, indent=1, default=float))
print(json.dumps({k: meta[k] for k in ("fit_sec", "total_sec", "counts")}), "\nloss", np.round(losses, 4).tolist(),
      "\ncoord sd min", round(checks["train_coord_sd_min"], 4), "mean cos", round(checks["train_mean_cosine_random_pairs"], 4))
