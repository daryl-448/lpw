"""Fit and freeze the P3b attachment kernel on real train (reports/p3b_prereg.md §P3, §P5).

1. Real-train replay (lpw.p3b.replay) -> opportunities, labels, pointer choice sets.
2. Diagnostics on 3 whole-group folds (seed 0, prefreeze A6 rule): gate BCE / Brier vs the p0 control (fitted on the
   fitting folds), calibration deciles by role, pointer NLL vs uniform. Reported only.
3. Final fit once on all train; p0, templates, gate, pointer saved with hashes to runs/p3b/2018/{prior,kernel}/.
Refuses to overwrite.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lpw.p3b import kernel as K  # noqa: E402
from lpw.p3b.access import load, real_pop  # noqa: E402
from lpw.p3b.replay import replay  # noqa: E402

KDIR, PDIR = C.ROOT / "kernel", C.ROOT / "prior"
for d in (KDIR, PDIR):
    assert not d.exists(), f"{d} exists: refusing to overwrite a frozen artefact"
t0 = time.time()
R = load()
pop, _ = real_pop(R, "train")
rr = replay(pop)
nz = rr.opp_nc > 0
role = rr.opp_bin[:, 0]
y = rr.opp_attach
agent_of = np.repeat(np.arange(pop.n), np.diff(pop.off))
grp = pop.group[agent_of[rr.opp_event]]
ug = np.unique(pop.group)
fold_of_group = dict(zip(np.random.default_rng(0).permutation(ug).tolist(), np.arange(len(ug)) % 3))
fold = np.array([fold_of_group[int(g)] for g in grp])
pos_opp = np.array([k for k, _, _ in rr.ptr])
ptr_sets = [(f, t, float(role[k])) for k, f, t in rr.ptr]
pfold = fold[pos_opp]
L = []


def log(s):
    print(s, flush=True)
    L.append(s)


def bce(p, yy):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())


diag = {"folds": []}
for k in range(3):
    fi, te = nz & (fold != k), nz & (fold == k)
    g = K.fit_gate(rr.opp_gate[fi], rr.opp_bin[fi], y[fi].astype(np.float64))
    p0k = {r: float(y[fi & (role == r)].mean()) for r in (0, 1)}
    pg = K.gate_prob(g, rr.opp_gate[te], rr.opp_bin[te])
    pc = np.where(role[te] == 1, p0k[1], p0k[0])
    yt = y[te].astype(np.float64)
    tr_sets = [s for s, f in zip(ptr_sets, pfold) if f != k]
    te_sets = [s for s, f in zip(ptr_sets, pfold) if f == k]
    p = K.fit_pointer(tr_sets)
    nll = float(np.mean([-K.ptr_logp(p, f, r)[t] for f, t, r in te_sets]))
    unif = float(np.mean([np.log(len(f)) for f, _, _ in te_sets]))
    row = {"fold": k, "n_te": int(te.sum()), "gate_bce": bce(pg, yt), "control_bce": bce(pc, yt),
           "gate_brier": float(((pg - yt) ** 2).mean()), "control_brier": float(((pc - yt) ** 2).mean()),
           "ptr_nll": nll, "uniform_nll": unif, "n_te_sets": len(te_sets)}
    for r in (0, 1):
        m = role[te] == r
        q = np.quantile(pg[m], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q[1:-1], pg[m], side="right"), 0, 9)
        row[f"calib_role{r}"] = [[float(pg[m][b == j].mean()), float(yt[m][b == j].mean()), int((b == j).sum())] for j in range(10) if (b == j).any()]
    diag["folds"].append(row)
    log(f"fold {k}: gate BCE {row['gate_bce']:.4f} vs control {row['control_bce']:.4f}; Brier {row['gate_brier']:.4f} vs "
        f"{row['control_brier']:.4f}; pointer NLL {nll:.3f} vs uniform {unif:.3f} ({len(te_sets)} sets)")
for key in ("gate_bce", "control_bce", "ptr_nll", "uniform_nll"):
    diag[f"mean_{key}"] = float(np.mean([r[key] for r in diag["folds"]]))

# final fit on all train
p0 = {r: float(y[nz & (role == r)].mean()) for r in (0, 1)}
gate = K.fit_gate(rr.opp_gate[nz], rr.opp_bin[nz], y[nz].astype(np.float64))
ptr = K.fit_pointer(ptr_sets)
log(f"final gate {gate['fit_info']}; pointer {ptr['fit_info']}; p0 {p0}")
keys, counts, n_train = C.template_prior(R)
PDIR.mkdir(parents=True)
KDIR.mkdir(parents=True)
np.savez(PDIR / "templates.npz", keys=keys, counts=counts, n_train=n_train)
(KDIR / "p0.json").write_text(json.dumps({str(k): v for k, v in p0.items()}))
with open(KDIR / "kernel.pkl", "wb") as fh:
    pickle.dump({"gate": gate, "ptr": ptr}, fh)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


CODE = ["src/lpw/p3b/access.py", "src/lpw/p3b/replay.py", "src/lpw/p3b/groups.py", "src/lpw/p3b/kernel.py",
        "scripts/p3b/common.py", "scripts/p3b/fit.py", "src/lpw/p1/store.py"]
meta = {"diag": diag, "p0": p0, "gate_fit": gate["fit_info"], "ptr_fit": ptr["fit_info"],
        "opportunities": {"first_touches": int(len(y)), "nonempty": int(nz.sum()), "positives": int(y.sum()), "cocreate": int(rr.opp_cocreate.sum())},
        "sha256": {"templates.npz": sha(PDIR / "templates.npz"), "p0.json": sha(KDIR / "p0.json"), "kernel.pkl": sha(KDIR / "kernel.pkl"),
                   **{c: sha(c) for c in CODE}},
        "gate_beta": gate["beta"].tolist(), "gate_bias": gate["bias"], "ptr_theta": ptr["theta"].tolist(), "sec": round(time.time() - t0)}
C.save_json(KDIR / "fit.json", meta)
(KDIR / "fit.txt").write_text("\n".join(L) + "\n")
log(f"done {meta['sec']}s")
