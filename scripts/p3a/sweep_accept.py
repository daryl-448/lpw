"""Sweep-repair acceptance (reports/p3a_prereg.md §15 A.4). Run after checkpoint0.py on worlds 10-12 and BEFORE any
utility / C2ST readout. Writes runs/p3a/2018/sweep/acceptance.json once (refuses to overwrite).

usage: uv run python scripts/p3a/sweep_accept.py [--worlds 10,11,12]
"""
import argparse
import json

import numpy as np

from common import ROOT, store, synth
from lpw.p1.atomic import write_text

R = {"normal": 0.13345, "scam": 0.40725}   # frozen real-train terminal near-zero shares (addendum A.4)
ROLES = (("normal", 0), ("scam", 1))
ap = argparse.ArgumentParser()
ap.add_argument("--worlds", default="10,11,12")
a = ap.parse_args()
worlds = [int(w) for w in a.worlds.split(",")]
out = ROOT / "sweep" / "acceptance.json"
assert not out.exists(), "acceptance is locked"
st = store()
res = {"worlds": worlds, "R_real_train": R, "per_world": {}}
safety_ok, complete = True, True
Z = {"kct_sweep": {r: [] for r, _ in ROLES}, "kct": {r: [] for r, _ in ROLES}}
dM = {r: [] for r, _ in ROLES}
for w in worlds:
    cp_path = ROOT / f"world{w}" / "checkpoint0.json"
    if not cp_path.exists():
        complete = False
        continue
    cp = json.loads(cp_path.read_text())
    pw = {"safety_stop": cp["safety_stop"]}
    safety_ok &= not cp["safety_stop"]
    for arm in ("kct_sweep", "kct"):
        if arm not in cp:
            complete = False
            continue
        ag, _, F, info = synth(st, w, arm, "hold", enforce_cap=False)
        y, z = ag["role"].to_numpy(), F["ends_near_zero"].to_numpy()
        pw[arm] = {"terminal_zero": {rn: float(z[y == r].mean()) for rn, r in ROLES},
                   "m_pass": {rn: cp[arm]["fidelity"][rn]["pass_rate"] for rn, _ in ROLES}, "dropped": info["dropped_by_role"]}
        for rn, _ in ROLES:
            Z[arm][rn].append(pw[arm]["terminal_zero"][rn])
    if "kct_sweep" in pw and "kct" in pw:
        for rn, _ in ROLES:
            dM[rn].append(pw["kct_sweep"]["m_pass"][rn] - pw["kct"]["m_pass"][rn])
    res["per_world"][w] = pw
complete &= all(len(v) == len(worlds) for v in dM.values())
mech = {rn: {"mean_Z": float(np.mean(Z["kct_sweep"][rn])) if Z["kct_sweep"][rn] else None, "band": [0.5 * R[rn], 1.5 * R[rn]],
             "kct_mean_Z": float(np.mean(Z["kct"][rn])) if Z["kct"][rn] else None} for rn, _ in ROLES}
for rn in mech:
    z = mech[rn]["mean_Z"]
    mech[rn]["pass"] = bool(z is not None and 0.5 * R[rn] <= z <= 1.5 * R[rn])
fid = {rn: {"mean_dM": float(np.mean(dM[rn])) if dM[rn] else None} for rn, _ in ROLES}
for rn in fid:
    fid[rn]["pass"] = bool(fid[rn]["mean_dM"] is not None and fid[rn]["mean_dM"] >= -0.10)
res.update(mechanism=mech, fidelity=fid, safety_ok=bool(safety_ok), complete=bool(complete))
if not safety_ok:
    res["decision"] = "SAFETY STOP"
elif not complete:
    res["decision"] = "INCONCLUSIVE"
else:
    res["decision"] = "ACCEPTED" if all(v["pass"] for v in mech.values()) and all(v["pass"] for v in fid.values()) else "REJECTED"
(ROOT / "sweep").mkdir(parents=True, exist_ok=True)
write_text(out, json.dumps(res, indent=1, default=float))
print(f"decision: {res['decision']}")
for rn in mech:
    print(f"  {rn}: terminal near-zero kct_sweep {mech[rn]['mean_Z']} (band {np.round(mech[rn]['band'], 4).tolist()}, kct {mech[rn]['kct_mean_Z']}) "
          f"pass {mech[rn]['pass']}; M-pass diff {fid[rn]['mean_dM']} pass {fid[rn]['pass']}")
print(f"  safety ok {safety_ok}, complete {complete}")
