"""Fit the balance-sweep gate on real TRAIN (reports/p3a_prereg.md §15 A.2). No val, no test.

usage: uv run python scripts/p3a/sweep_fit.py [--out runs/p3a/2018/sweep]
Writes OUT/gate.pt, OUT/fit.json (fit info, 3-fold whole-group OOF diagnostics, calibration by role).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.atomic import write_text
from lpw.p1.batch import Views, exact_specs, pack
from lpw.p1.store import IN_AGENT, OUT, build_store
from lpw.p1.sweep import D, LAM, eligible, features, fit_gate, save
from lpw.p1.train import load, loader, to_dev

PILOT = Path("data/pilot_2018")
KCT = ("lpw_plan_prog_shallow_kct", "runs/p1_shallow/2018/kct")


@torch.no_grad()
def build_cache(model, st, agents, dev="cuda"):
    """Every eligible real OUT of the agents, teacher-forced with generation-matched exact windows and the real
    plan: gate inputs, label, agent."""
    model.eval()
    model.plan_inference = True
    views = Views(st)
    specs = exact_specs(st, agents)
    batches = pack(specs, views)
    C = {k: [] for k in ("cont", "oh", "y", "agent", "role")}
    n_out = n_el = 0
    for sp, b in zip(batches, loader(views, batches)):
        # owned target rows, in collate order: (agent, event index k)
        rows = []
        for a, s, js in sp:
            n, kind = st.n_events(a), st.kind[st.off[a]:st.off[a + 1]]
            for j in js:
                k = s + int(j)
                if k < n and kind[k] != IN_AGENT:
                    rows.append((a, k))
        b = to_dev(b, dev)
        model._capture = {}
        model.terms(b)
        x = model._capture.get("amt_x")
        model._capture = None
        if x is None:
            assert not rows
            continue
        assert x.shape[0] == len(rows) == len(b["o_row"]), (x.shape, len(rows))
        a_ = np.array([r[0] for r in rows]); k_ = np.array([r[1] for r in rows])
        e = st.off[a_] + k_
        tk = b["row_tkind"][b["o_row"]].cpu().numpy()
        assert (tk == st.kind[e]).all(), "row / event mismatch"
        out = st.kind[e] == OUT
        n_out += int(out.sum())
        bal, eth, fee = st.bal[e].astype(np.float64), st.eth[e].astype(np.float64), st.fee[e].astype(np.float64)
        el = out & eligible(bal, fee)
        n_el += int(el.sum())
        if not el.any():
            continue
        i = np.flatnonzero(el)
        nown = np.array([int((st.kind[st.off[a]:st.off[a] + k] != IN_AGENT).sum()) for a, k in zip(a_[i], k_[i])])
        cont, oh = features(x[torch.as_tensor(i, device=x.device)], st.role[a_[i]], st.cohort[a_[i]], st.plan_n[a_[i]], st.plan_a[a_[i]],
                            nown, bal[i], fee[i], st.ts[e[i]].astype(np.float64), st.w_end)
        C["cont"].append(cont.cpu()); C["oh"].append(oh.cpu())
        C["y"].append(torch.as_tensor(np.abs(bal[i] - eth[i] - fee[i]) <= D))
        C["agent"].append(torch.as_tensor(a_[i])); C["role"].append(torch.as_tensor(st.role[a_[i]]))
    C = {k: torch.cat(v) for k, v in C.items()}
    C["n_out"], C["n_eligible"] = n_out, n_el
    return C


def bce(g, cont, oh, y):
    with torch.no_grad():
        return float(torch.nn.functional.binary_cross_entropy_with_logits(g.logit(cont, oh), y.float(), reduction="sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/p3a/2018/sweep")
    ap.add_argument("--dry", type=int, default=0, help="plumbing check on the first N train agents; writes nothing")
    a = ap.parse_args()
    out = Path(a.out)
    real_dest = Path("runs/p3a/2018/sweep").resolve()
    assert not (out / "gate.pt").exists(), "a gate exists here: refusing to refit or overwrite"
    assert not (a.dry and out.resolve() == real_dest), "dry runs never write to the real gate destination"
    if not a.dry:
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = build_store(PILOT)
    tr = st.agents("train")[: a.dry] if a.dry else st.agents("train")
    model = load(PILOT, KCT[0], Path(KCT[1]), st)
    C = build_cache(model, st, tr)
    dev = "cuda"
    cont, oh, y = C["cont"].to(dev), C["oh"].to(dev), C["y"].to(dev)
    role = C["role"].numpy()
    ag = pl.read_parquet(PILOT / "agents.parquet")
    gid = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    grp = np.array([gid[int(st.node_id[x])] for x in C["agent"].numpy()])
    ug = np.unique(grp)
    fold_of = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % 3))
    fold = np.array([fold_of[g] for g in grp])
    oof = np.zeros(len(y))
    diag = {"folds": []}
    for f in range(3):
        trm, tem = torch.as_tensor(fold != f, device=dev), torch.as_tensor(fold == f, device=dev)
        g = fit_gate(cont[trm], oh[trm], y[trm])
        with torch.no_grad():
            oof[fold == f] = g.prob(cont[tem], oh[tem]).double().cpu().numpy()
        base = float(y[trm].float().mean())
        yt = y[tem].double().cpu().numpy()
        diag["folds"].append({"fold": f, "n": int(tem.sum()), "bce_per_row": bce(g, cont[tem], oh[tem], y[tem]) / int(tem.sum()),
                              "bce_per_row_intercept_only": float(-np.mean(yt * np.log(base) + (1 - yt) * np.log(1 - base))),
                              "fit_info": g.fit_info})
    yn = y.cpu().numpy().astype(float)
    diag["calibration"] = {}
    for r, rn in ((0, "normal"), (1, "scam")):
        m = role == r
        q = np.quantile(oof[m], np.linspace(0, 1, 11)); q[-1] += 1e-12
        bins = np.clip(np.searchsorted(q, oof[m], side="right") - 1, 0, 9)
        diag["calibration"][rn] = {"n": int(m.sum()), "rate": float(yn[m].mean()), "mean_p": float(oof[m].mean()),
                                   "deciles": [{"mean_p": float(oof[m][bins == d].mean()), "rate": float(yn[m][bins == d].mean()),
                                                "n": int((bins == d).sum())} for d in range(10)]}
    gate = fit_gate(cont, oh, y)
    if a.dry:
        out.mkdir(parents=True, exist_ok=True)  # a plumbing-smoke root (checked above), never the real destination
        save(gate, out / "gate.pt")
        print(f"DRY: rows {len(y)} pos {int(y.sum())} dims {cont.shape[1]}+{oh.shape[1]} fit {gate.fit_info} {time.time() - t0:.0f}s")
        return
    save(gate, out / "gate.pt")
    res = {"lam": LAM, "n_out_train": C["n_out"], "n_eligible": C["n_eligible"], "n_pos": int(y.sum()),
           "pos_rate_by_role": {rn: float(yn[role == r].mean()) for r, rn in ((0, "normal"), (1, "scam"))},
           "n_cont": int(cont.shape[1]), "n_onehot": int(oh.shape[1]), "final_fit": gate.fit_info, "oof": diag,
           "gate_sha256": hashlib.sha256((out / "gate.pt").read_bytes()).hexdigest(), "sec": round(time.time() - t0)}
    write_text(out / "fit.json", json.dumps(res, indent=1, default=float))
    f = diag["folds"]
    print(f"train OUTs {C['n_out']}, eligible {C['n_eligible']}, sweeps {int(y.sum())}; pos rate normal "
          f"{res['pos_rate_by_role']['normal']:.4f} scam {res['pos_rate_by_role']['scam']:.4f}")
    print("OOF BCE/row " + " ".join(f"fold{x['fold']} {x['bce_per_row']:.4f} (intercept {x['bce_per_row_intercept_only']:.4f})" for x in f))
    for rn, c in diag["calibration"].items():
        print(f"{rn}: rate {c['rate']:.4f} mean OOF p {c['mean_p']:.4f}; deciles p/rate " +
              " ".join(f"{d['mean_p']:.3f}/{d['rate']:.3f}" for d in c["deciles"]))
    print(f"final fit {gate.fit_info}; {res['sec']}s")


if __name__ == "__main__":
    main()
