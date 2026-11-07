"""Access check: the guarded train / val store reproduces P3a's real-side results exactly (TRTR AUCs, matched pairs).

Re-exposes only already-reported P3a val numbers; computes nothing new on val."""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts/p3a")
from common import ROOT  # noqa: E402

from lpw.p3.features import matrix, node_features  # noqa: E402
from lpw.p3.matching import Matcher  # noqa: E402
from lpw.p3.observe import observe_real  # noqa: E402
from lpw.p3.utility import fit_score, single_auc  # noqa: E402
from lpw.p3b.access import load  # noqa: E402

R = load()
st = R.st
F = {}
for sp in ("train", "val"):
    ag, ev = observe_real(st, R.agents(sp))
    F[sp] = node_features(ag, ev, st.w_start, st.w_end)
ytr, yva = st.role[R.agents("train")], st.role[R.agents("val")]
ref = json.loads((ROOT / "readout.json").read_text())
out = {}
for fs in ("full", "noconf", "confonly"):
    (s,) = fit_score("xgb", matrix(F["train"], fs), ytr, [matrix(F["val"], fs)])
    a = single_auc(yva, s)
    out[fs] = (a, ref["trtr"][f"{fs}/xgb"]["auc"])
M = Matcher(F["train"], ytr).match(F["val"], yva)
out["pairs"] = (M["n_pairs"], ref["matching"]["n_pairs"])
for k, (a, b) in out.items():
    print(k, a, b, "OK" if a == b else "MISMATCH")
assert all(a == b for a, b in out.values()), "filtered store does not reproduce P3a"
print("agents", {sp: len(R.agents(sp)) for sp in ("train", "val")}, "events", len(st.ts))
