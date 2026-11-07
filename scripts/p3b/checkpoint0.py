"""P3b Checkpoint 0 (reports/p3b_prereg.md §P11): world 0, train + hold, kct and B, P3b-0 then P3b-1. Invariants and
mechanism summaries only: no role utility, no C2ST, no comparison with real val. Exit 2 on a hard failure.

Hard invariants: cap / retention as in P3a; sidecar alignment; within-agent injection (the local equality relation
of attached nodes equals that of local entries); group-exclusive nodes; every attachment has an earlier-second witness
from another member of the same group and no attachment uses a node minted at or after its own second; same-seed
reproducibility; finite STRUCT in [0, 1].
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lpw.p3b.access import load  # noqa: E402
from lpw.p3b.graph import STRUCT, struct_features  # noqa: E402
from lpw.p3b.topology import S_PERIPHERY, key_kind  # noqa: E402

W = 0
st = load().st
res, L, fails = {}, [], []


def log(s):
    print(s, flush=True)
    L.append(s)


def check(ok, msg):
    if not ok:
        fails.append(msg)
        log(f"FAIL: {msg}")


for part in ("train", "hold"):
    G = np.load(C.ROOT / f"world{W}" / f"groups_{part}.npz")
    group = G["group"]
    res[f"groups/{part}"] = {k: int(G[k]) for k in G.files if k != "group"}
    log(f"groups {part}: {res[f'groups/{part}']}")
    for arm in C.ARMS:
        pop, raw, conds = C.synth_pop(st, W, arm, part)
        agent = np.repeat(np.arange(pop.n), np.diff(pop.off))
        for att in C.ATTACH:
            S = C.sidecar(W, arm, part, att)
            node, gnode = S["node"], S["group_of_node"]
            tag = f"{arm}/{part}/{att}"
            m = pop.elig
            check(((node >= 0) == m).all(), f"{tag}: eligible mask")
            # injection: (agent, ent) <-> (agent, node) one-to-one
            ae = np.unique(np.stack([agent[m], pop.ent[m]], 1), axis=0)
            an = np.unique(np.stack([agent[m], node[m]], 1), axis=0)
            aen = np.unique(np.stack([agent[m], pop.ent[m], node[m]], 1), axis=0)
            check(len(ae) == len(an) == len(aen), f"{tag}: within-agent injection")
            check((gnode[node[m]] == group[agent[m]]).all(), f"{tag}: group-exclusive nodes")
            # witnesses
            ev_i = np.flatnonzero(m)
            order = np.lexsort((ev_i, pop.sec[ev_i], node[ev_i]))
            nd_sorted, sec_sorted, ag_sorted = node[ev_i][order], pop.sec[ev_i][order], agent[ev_i][order]
            first_of_node = {}
            for nd, s, a in zip(nd_sorted, sec_sorted, ag_sorted):
                first_of_node.setdefault(int(nd), []).append((int(s), int(a)))
            att_ev = S["opp_event"][S["opp_attach"]]
            bad_w = 0
            for i in att_ev:
                s, a, nd = int(pop.sec[i]), int(agent[i]), int(node[i])
                if not any(s2 < s and a2 != a for s2, a2 in first_of_node[nd]):
                    bad_w += 1
                if first_of_node[nd][0][0] >= s:
                    bad_w += 1
            check(bad_w == 0, f"{tag}: {bad_w} attachments without an earlier-second witness")
            # reproducibility (one arm per part)
            if att == "p3b1" and arm == "kct":
                import pickle
                from lpw.p3b.kernel import learned_kernel
                from lpw.p3b.replay import replay
                with open(C.ROOT / "kernel" / "kernel.pkl", "rb") as fh:
                    kp = pickle.load(fh)
                pop.group = group
                r2 = replay(pop, learned_kernel(kp["gate"], kp["ptr"]), C.uniforms(pop, W, part))
                check((r2.node == node).all(), f"{tag}: not reproducible")
            # mechanism summary
            nc, at, ro, ou = S["opp_nc"], S["opp_attach"], S["opp_role"], S["opp_out"]
            summ = {"first_touches": int(len(nc)), "nonempty": float((nc > 0).mean())}
            for r in (0, 1):
                for d, nm in ((1, "OUT"), (0, "IN")):
                    mm = (ro == r) & (ou == d)
                    summ[f"role{r}/{nm}"] = {"n": int(mm.sum()), "nonempty": float((nc[mm] > 0).mean()) if mm.any() else None,
                                             "attach": float(at[mm].mean()) if mm.any() else None,
                                             "attach_given_nonempty": float(at[mm & (nc > 0)].mean()) if (mm & (nc > 0)).any() else None}
            summ["nc_q_nonempty"] = np.quantile(nc[nc > 0], [0.5, 0.9, 1]).tolist() if (nc > 0).any() else None
            # observed view, STRUCT
            ag, ev, F, (B, nodes), info = C.synth_view(st, W, arm, part, att)
            check((key_kind(nodes) == S_PERIPHERY).all(), f"{tag}: incidence nodes outside the periphery namespace")
            f = struct_features(B)
            ok = all(np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in f.values())
            check(ok, f"{tag}: STRUCT out of range")
            summ["struct_mean"] = {c: float(f[c].mean()) for c in STRUCT}
            summ["retained"], summ["dropped"], summ["capped"] = int(ag.height), int(pop.n - ag.height), int(info["n_capped"])
            check(info["n_capped"] <= 0.001 * pop.n, f"{tag}: cap share")
            res[tag] = summ
            log(f"{tag}: first touches {summ['first_touches']} nonempty {summ['nonempty']:.3f}; "
                + "; ".join(f"{k} n {v['n']} att {v['attach']:.4f} | nonempty {v['attach_given_nonempty']}" for k, v in summ.items()
                            if k.startswith("role") and v["n"]) + f"; STRUCT {np.round(list(summ['struct_mean'].values()), 4).tolist()}")
        # anchor: zero sharing
        ag, ev, F, (B, nodes), info = C.synth_view(st, W, arm, part, None)
        f = struct_features(B)
        check(all((v == 0).all() for v in f.values()), f"{arm}/{part}: anchor not zero-sharing")
res["fails"] = fails
C.save_json(C.ROOT / f"world{W}" / "checkpoint0.json", res)
(C.ROOT / f"world{W}" / "checkpoint0.txt").write_text("\n".join(L) + "\n")
log("CHECKPOINT 0: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(2 if fails else 0)
