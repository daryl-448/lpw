"""Attach shared identities to saved P3a worlds (reports/p3b_prereg.md §P2-P4, §P6).

For each world, part (train, hold), policy arm (kct, B) and attachment arm (p3b0 control, p3b1 learned): sample the
group map (shared across arms), draw the attachment uniforms, replay, and write
runs/p3b/2018/world{w}/groups_{part}.npz and {arm}_{part}_{att}.npz (node per event of the observe-sorted raw tape,
-1 for non-eligible events; per-opportunity size, decision, role and direction). The frozen kernel is verified
against fit.json before use. Refuses to overwrite.

Usage: python scripts/p3b/attach.py --worlds 0 [1 2]
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lpw.p3b.access import load  # noqa: E402
from lpw.p3b.kernel import control_kernel, learned_kernel  # noqa: E402
from lpw.p3b.replay import replay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, nargs="+", required=True)
a = ap.parse_args()
KDIR, PDIR = C.ROOT / "kernel", C.ROOT / "prior"
meta = json.loads((KDIR / "fit.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


for f, d in (("templates.npz", PDIR), ("p0.json", KDIR), ("kernel.pkl", KDIR)):
    assert sha(d / f) == meta["sha256"][f], f"{f} does not match the frozen fit"
T = np.load(PDIR / "templates.npz")
keys, counts, n_train = T["keys"], T["counts"], int(T["n_train"])
p0 = {int(k): v for k, v in json.loads((KDIR / "p0.json").read_text()).items()}
with open(KDIR / "kernel.pkl", "rb") as fh:
    kp = pickle.load(fh)
KERNELS = {"p3b0": control_kernel(p0), "p3b1": learned_kernel(kp["gate"], kp["ptr"])}
st = load().st  # only w_start / w_end are used
for w in a.worlds:
    out = C.ROOT / f"world{w}"
    out.mkdir(parents=True, exist_ok=True)
    for part in ("train", "hold"):
        gpath = out / f"groups_{part}.npz"
        assert not gpath.exists(), f"{gpath} exists"
        ginfo = None
        for arm in C.ARMS:
            pop, ev, conds = C.synth_pop(st, w, arm, part)
            info = C.assign_groups(pop, w, part, keys, counts, n_train)
            if ginfo is None:
                ginfo, group = info, pop.group.copy()
                np.savez(gpath, group=group, **{k: v for k, v in info.items()})
            assert (pop.group == group).all()
            u = C.uniforms(pop, w, part)
            for att, kern in KERNELS.items():
                path = out / f"{arm}_{part}_{att}.npz"
                assert not path.exists(), f"{path} exists"
                t = time.time()
                r = replay(pop, kern, u)
                np.savez(path, world=w, part=part, arm=arm, att=att,  # view context: node ids are local to this view
                         node=r.node, group_of_node=r.group_of_node, opp_event=r.opp_event, opp_nc=r.opp_nc,
                         opp_attach=r.opp_attach, opp_role=r.opp_bin[:, 0].astype(np.int8), opp_out=r.opp_bin[:, 2].astype(np.int8))
                nz = r.opp_nc > 0
                rates = [float(r.opp_attach[r.opp_bin[:, 0] == k].mean()) for k in (0, 1)]
                print(f"world {w} {part} {arm} {att}: {time.time() - t:.1f}s, first touches {len(nz)}, nonempty {nz.mean():.3f}, "
                      f"attach normal / scam {rates[0]:.4f} / {rates[1]:.4f}, groups {info['n_multi_groups']} multi, max {info['max_group']}",
                      flush=True)
