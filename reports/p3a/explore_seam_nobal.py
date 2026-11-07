"""EXPLORATORY (prereg §11; not a gate): C2ST per role on LOCAL+VENUE (no BAL) and on BAL alone, world 0 and 1."""
import sys
sys.path.insert(0, "scripts/p3a")
import numpy as np
from common import group_ids, real, store, synth
from lpw.p3.features import BAL, LOCAL, VENUE
from lpw.p3.utility import c2st
st = store(); va = st.agents("val"); g = group_ids(st, va)
_, _, Rva = real(st, "val"); y = st.role[va]
for arm in ("kct", "B"):
    for w in (0, 1):
        ag, _, F, _ = synth(st, w, arm, "hold"); yH = ag["role"].to_numpy()
        out = []
        for name, cols in (("local+venue", LOCAL + VENUE), ("bal", BAL), ("all-local", LOCAL + BAL + VENUE)):
            v = [c2st(Rva.select(cols).to_numpy()[y == r], F.select(cols).to_numpy()[yH == r], g[y == r]) for r in (0, 1)]
            out.append(f"{name} {v[0]:.3f}/{v[1]:.3f}")
        print(arm, "world", w, "C2ST normal/scam:", " | ".join(out))
