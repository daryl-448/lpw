"""P3b pre-freeze, label-blind (Sol consult §10 B3 + work estimates): transaction identity and deduplication on the
real train / val views, daily-topology undefined-value frequencies on val halves, and Clauset fits and cost on the
full **train** view (cost checks are train-only; the v0 run of this script used val and the earlier implementation:
prefreeze_topology_v0.*, kept as a preliminary record).

No roles are read. Output: runs/p3b/2018/prefreeze_topology.{json,txt}.
"""
import json
import time
from pathlib import Path

import numpy as np

from lpw.p3b.access import load
from lpw.p3b.topology import DAILY, clauset, daily_metrics

OUTD = Path("runs/p3b/2018")
DAY = 86400
R = load()
st = R.st
res, L = {}, []


def log(s):
    print(s, flush=True)
    L.append(s)


import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def tx_table(agents):
    return C.real_transactions(R, agents)


for sp in ("train", "val"):
    T = tx_table(R.agents(sp))
    res[f"tx_{sp}"] = {k: T[k] for k in ("n_rows", "n_tx", "n_dup_rows")}
    log(f"{sp}: incident rows {T['n_rows']}, unique transactions {T['n_tx']}, duplicate recordings removed {T['n_dup_rows']}")

# daily topology on val halves (venues included / excluded)
va = R.agents("val")
t = time.time()
und = {f"{k}/{vv}": [] for k in DAILY for vv in ("ven", "noven")}
n_days_nonempty = []
for s in range(3):
    p = np.random.default_rng(s).permutation(len(va))
    half = np.sort(va[p[: len(va) // 2]])
    T = tx_table(half)
    days = np.unique(T["day"])
    n_days_nonempty.append(len(days))
    for vv in ("ven", "noven"):
        cnt = {k: 0 for k in DAILY}
        for d in days:
            m = (T["day"] == d) & ((not (vv == "noven")) | ~T["venue"])
            met = daily_metrics(T["u"][m], T["v"][m])
            for k in DAILY:
                cnt[k] += not np.isfinite(met[k])
        for k in DAILY:
            und[f"{k}/{vv}"].append(cnt[k] / len(days))
res["daily_undefined_share"] = {k: float(np.mean(v)) for k, v in und.items()}
res["daily_nonempty_days"] = n_days_nonempty
res["daily_sec_per_view_both_variants"] = (time.time() - t) / 3
log(f"daily undefined share (val halves): {res['daily_undefined_share']}; nonempty days {n_days_nonempty}; "
    f"{res['daily_sec_per_view_both_variants']:.1f}s per half view")

# Clauset fits and cost on the full train view (100 bootstrap refits)
T = tx_table(R.agents("train"))
nodes, inv = np.unique(np.concatenate([T["u"], T["v"]]), return_inverse=True)
ui, vi = inv[: len(T["u"])], inv[len(T["u"]):]
m = ui != vi
pairs = np.unique(np.stack([np.minimum(ui[m], vi[m]), np.maximum(ui[m], vi[m])], 1), axis=0)
deg_u = np.bincount(pairs.ravel(), minlength=len(nodes))
deg_tx = np.bincount(np.concatenate([ui[m], vi[m]]), minlength=len(nodes))
strength = np.bincount(np.concatenate([ui, vi]), weights=np.concatenate([T["eth"], T["eth"]]), minlength=len(nodes))
cl = {}
for nm, x, disc in (("unique_degree", deg_u, True), ("tx_degree", deg_tx, True), ("strength", strength, False), ("amount", T["eth"], False)):
    t = time.time()
    r = clauset(x, disc, n_boot=100)
    r["sec_100boot"] = time.time() - t
    cl[nm] = r
    log(f"clauset {nm}: {r}")
res["clauset_train_full_100boot"] = cl
(OUTD / "prefreeze_topology.json").write_text(json.dumps(res, indent=1, default=float))
(OUTD / "prefreeze_topology.txt").write_text("\n".join(L) + "\n")
