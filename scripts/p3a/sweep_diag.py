"""Fixed attribution diagnostics of the sweep repair (reports/p3a_prereg.md §15 A.5). Run with the readout, after
acceptance is locked. None of these numbers is a tuning target.

usage: uv run python scripts/p3a/sweep_diag.py [--worlds 10,11,12] [--arms kct_sweep,kct]
Writes runs/p3a/2018/sweep/diag.{json,txt}.
"""
import argparse
import json

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import ROOT, group_ids, real, store, synth
from lpw.p1.atomic import write_text
from lpw.p3.features import BAL, LOCAL, VENUE
from lpw.p3.sweep_stats import sweep_stats
from lpw.p3.utility import c2st

ROLES = (("normal", 0), ("scam", 1))
ap = argparse.ArgumentParser()
ap.add_argument("--worlds", default="10,11,12")
ap.add_argument("--arms", default="kct_sweep,kct")
a = ap.parse_args()
acc = ROOT / "sweep" / "acceptance.json"
assert acc.exists(), "lock acceptance first (sweep_accept.py)"
assert json.loads(acc.read_text())["decision"] != "SAFETY STOP", "safety stop: no readout (prereg §15 A.5)"
worlds, arms = [int(w) for w in a.worlds.split(",")], a.arms.split(",")
st = store()
va = st.agents("val"); g = group_ids(st, va)
rag, rev, Rva = real(st, "val"); y = rag["role"].to_numpy()
res = {"real_val": sweep_stats(rag, rev), "arms": {}}
local = LOCAL + BAL + VENUE
L = ["real val sweep stats: " + json.dumps(res["real_val"])]
for arm in arms:
    R = {}
    for w in worlds:
        ag, ev, F, _ = synth(st, w, arm, "hold")
        yH = ag["role"].to_numpy()
        d = {"sweep_stats": sweep_stats(ag, ev), "c2st": {}, "seam_single": {}}
        for name, cols in (("bal_only", BAL), ("local_venue_no_bal", LOCAL + VENUE)):
            d["c2st"][name] = {rn: c2st(Rva.select(cols).to_numpy()[y == r], F.select(cols).to_numpy()[yH == r], g[y == r]) for rn, r in ROLES}
        for rn, r in ROLES:
            lab = np.r_[np.zeros((y == r).sum()), np.ones((yH == r).sum())]
            d["seam_single"][rn] = {f: float(roc_auc_score(lab, np.r_[Rva[f].to_numpy()[y == r], F[f].to_numpy()[yH == r]])) for f in local}
        R[w] = d
    ws = list(R.values())
    agg = {"c2st": {n: {rn: float(np.mean([x["c2st"][n][rn] for x in ws])) for rn, _ in ROLES} for n in ("bal_only", "local_venue_no_bal")},
           "sweep_stats": {rn: {k: float(np.mean([x["sweep_stats"][rn][k] for x in ws if x["sweep_stats"][rn][k] is not None]))
                                for k in ws[0]["sweep_stats"][rn]} for rn, _ in ROLES},
           "seam_single": {rn: {f: float(np.mean([x["seam_single"][rn][f] for x in ws])) for f in local} for rn, _ in ROLES}}
    res["arms"][arm] = {"worlds": R, "aggregate": agg}
    L.append(f"{arm}: C2ST BAL-only {agg['c2st']['bal_only']['normal']:.3f}/{agg['c2st']['bal_only']['scam']:.3f}, "
             f"LOCAL+VENUE no BAL {agg['c2st']['local_venue_no_bal']['normal']:.3f}/{agg['c2st']['local_venue_no_bal']['scam']:.3f}")
    L.append(f"{arm} sweep stats: " + json.dumps({rn: {k: round(v, 3) for k, v in s.items()} for rn, s in agg["sweep_stats"].items()}))
    for rn, _ in ROLES:
        top = sorted(agg["seam_single"][rn].items(), key=lambda kv: -abs(kv[1] - 0.5))[:8]
        L.append(f"{arm} {rn} single-feature seam |AUC-0.5| top: " + " ".join(f"{f} {abs(v - 0.5):.3f}" for f, v in top))
write_text(ROOT / "sweep" / "diag.json", json.dumps(res, indent=1, default=float))
write_text(ROOT / "sweep" / "diag.txt", "\n".join(L) + "\n")
print("\n".join(L))
