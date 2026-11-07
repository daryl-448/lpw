"""B1 likelihood-gap audit on 2018 val (development data; reports/p1_audit.md).

The question: how far is the current base's unknown-plan likelihood from the tuned B1 GRU baseline, under matched
conventions, with whole-group uncertainty? (SPEC A1 non-inferiority: upper95 of model - b1 <= +0.03 per owned event.)

  uv run python scripts/p1_audit/b1gap.py eval   exact per-prefix val terms of the runs/p1_r0 models -> OUT/<name>_val_rows.npz
  uv run python scripts/p1_audit/b1gap.py report paired whole-group bootstrap vs both B1 seeds -> OUT/b1gap.{json,txt}

The rows have the encoder screen's format and prefix set (scripts/p1_encoder/screen.py eval_run):
- ktc = kind + time + cens;
- ca = cp + amt on owned rows;
- own, agent, depth.
The encoder-screen runs already have theirs (runs/p1_enc/2018/*/val_rows.npz). Plan models are evaluated with
plan_inference off: the unknown-plan likelihood, which kct shares with its base.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from lpw.p1.atomic import write_text

PILOT = Path("data/pilot_2018")
OUT = Path("runs/p1_audit/2018")
ENC = Path("runs/p1_enc/2018")
NBOOT, NI_MARGIN = 1000, 0.03
# name -> (run dir, model config); the current base first
EVAL = {"base_plan_prog_rp": ("runs/p1_r0/2018/lpw_plan_prog_rp", "lpw_plan_prog"),
        "lpw_s20261104": ("runs/p1_r0/2018/lpw", "lpw"),
        "lpw_s1": ("runs/p1_r0/2018/lpw_s1", "lpw"),
        "lpw_s2": ("runs/p1_r0/2018/lpw_s2", "lpw")}
REUSE = {"tf_ref_s20261104": ENC / "tf_ref_lr1e-3_s20261104", "tf_alibi_s20261104": ENC / "tf_alibi_lr1e-3_s20261104",
         "tf_alibi_s20261105": ENC / "tf_alibi_lr1e-3_s20261105"}
B1 = {"b1_s20261104": ENC / "b1_lr1e-3_s20261104", "b1_s20261105": ENC / "b1_lr1e-3_s20261105"}
DEPTHS = ((0, 1), (1, 4), (4, 30), (30, 100), (100, 256), (256, 10 ** 9))


@torch.no_grad()
def eval_all():
    from lpw.p1.batch import Views, exact_specs, pack
    from lpw.p1.evaluate import evaluate
    from lpw.p1.store import build_store
    from lpw.p1.train import load, loader, to_dev
    st = build_store(PILOT); views = Views(st)
    batches = pack(exact_specs(st, st.agents("val")), views)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (run, cfg) in EVAL.items():
        if (OUT / f"{name}_val_rows.npz").exists():
            continue
        m = load(PILOT, cfg, Path(run), st)
        m.plan_inference = False
        R = {k: [] for k in ("ktc", "ca", "own", "agent", "depth")}
        for bi, b in enumerate(loader(views, batches)):
            sp = batches[bi]
            b = to_dev(b, "cuda")
            t = m.terms(b)
            n = len(t["kind"])
            R["ktc"].append((t["kind"] + t["time"] + t["cens"]).double().cpu().numpy())
            R["ca"].append(torch.zeros(n, dtype=torch.float64, device="cuda").index_add(0, t["o_row"], (t["cp"] + t["amt"]).double()).cpu().numpy())
            R["own"].append(t["own"].cpu().numpy())
            rb, rj = b["row_b"].cpu().numpy(), b["row_j"].cpu().numpy()
            R["agent"].append(np.array([sp[i][0] for i in rb], np.int64)); R["depth"].append(np.array([sp[i][1] for i in rb], np.int64) + rj)
        R = {k: np.concatenate(v) for k, v in R.items()}
        ev = evaluate(m, st, "val", views=views)["all"]
        full = (R["ktc"].sum() + R["ca"].sum()) / R["own"].sum()
        assert abs(full - ev["nll_full"]) < 1e-6 * max(1.0, abs(full)), (name, full, ev["nll_full"])
        np.savez(OUT / f"{name}_val_rows.npz", **R)
        write_text(OUT / f"{name}_val_eval.json", json.dumps(ev, indent=1))
        print(f"{name}: exact val full {ev['nll_full']:.4f}", flush=True)


def recipe(run: Path) -> dict:
    t = json.loads((run / "train.json").read_text())
    r = json.loads((run / "recal.json").read_text()) if (run / "recal.json").exists() else {}
    return {"model": t["model"], "pilot": t["pilot"], "seed": t["seed"], "lr": t["lr"], "init": t.get("init"),
            "agent_alpha": t.get("agent_alpha"), "cutoff": t.get("cutoff"), "best_epoch": t["best_epoch"],
            "train_val_metric": t["best_val"], "params": t["params"], "recal_steps": r.get("steps"), "recal_rows": r.get("rows")}


def report():
    from lpw.p1.store import build_store
    st = build_store(PILOT)
    ag = pl.read_parquet(PILOT / "agents.parquet")
    gid_of = dict(zip(ag["node_id"].to_list(), ag["group_id"].to_list()))
    paths = {**{n: (OUT / f"{n}_val_rows.npz", Path(r)) for n, (r, _) in EVAL.items()},
             **{n: (d / "val_rows.npz", d) for n, d in {**REUSE, **B1}.items()}}
    rows = {n: dict(np.load(p)) for n, (p, _) in paths.items()}
    ref = rows["b1_s20261104"]
    for n, r in rows.items():
        assert np.array_equal(r["agent"], ref["agent"]) and np.array_equal(r["depth"], ref["depth"]), f"{n}: prefix set differs"
        assert np.array_equal(r["own"], ref["own"]), f"{n}: owned flags differ"
    own = ref["own"].astype(np.float64)
    gid = np.array([gid_of[int(x)] for x in st.node_id[ref["agent"]]])
    ug, inv = np.unique(gid, return_inverse=True)
    go = np.bincount(inv, own, len(ug))
    draws = np.random.default_rng(0).integers(0, len(ug), (NBOOT, len(ug)))
    res = {"n_prefixes": int(len(own)), "n_owned": int(own.sum()), "n_groups": int(len(ug)),
           "recipe": {n: recipe(d) for n, (_, d) in paths.items()},
           "full": {n: float((r["ktc"] + r["ca"]).sum() / own.sum()) for n, r in rows.items()}, "vs_b1": {}}
    for n, r in rows.items():
        if n in B1:
            continue
        res["vs_b1"][n] = {}
        for b in B1:
            parts = {"full": (r["ktc"] + r["ca"]) - (rows[b]["ktc"] + rows[b]["ca"]), "ktc": r["ktc"] - rows[b]["ktc"],
                     "ca": r["ca"] - rows[b]["ca"]}
            c = {}
            for k, diff in parts.items():
                bs = np.bincount(inv, diff, len(ug))[draws].sum(1) / go[draws].sum(1)
                c[k] = {"delta": float(diff.sum() / own.sum()), "upper95": float(np.quantile(bs, 0.95))}
            d = parts["full"]
            c["by_depth"] = {f"{lo}-{hi}": float(d[(ref["depth"] >= lo) & (ref["depth"] < hi)].sum() / own.sum())
                             for lo, hi in DEPTHS}  # contribution to the per-owned-event delta (sums to it)
            c["non_inferior"] = bool(c["full"]["upper95"] <= NI_MARGIN)
            res["vs_b1"][n][b] = c
    write_text(OUT / "b1gap.json", json.dumps(res, indent=1, default=float))

    L = [f"2018 val: {res['n_prefixes']} prefixes, {res['n_owned']} owned events, {res['n_groups']} groups; "
         f"paired whole-group bootstrap ({NBOOT}, seed 0); non-inferiority: upper95 <= +{NI_MARGIN}"]
    L.append("recipes:")
    for n, rc in res["recipe"].items():
        L.append(f"  {n:20s} " + " ".join(f"{k}={v}" for k, v in rc.items()))
    L.append("full NLL per owned event: " + ", ".join(f"{n} {v:.4f}" for n, v in res["full"].items()))
    for n, v in res["vs_b1"].items():
        for b, c in v.items():
            L.append(f"{n:20s} - {b}: full {c['full']['delta']:+.4f} (upper95 {c['full']['upper95']:+.4f}) "
                     f"[kind/time/cens {c['ktc']['delta']:+.4f}, cp/amt {c['ca']['delta']:+.4f}] "
                     f"{'NON-INFERIOR' if c['non_inferior'] else 'not non-inferior'}")
            L.append("    by depth (contribution): " + ", ".join(f"{k} {x:+.4f}" for k, x in c["by_depth"].items()))
    write_text(OUT / "b1gap.txt", "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"eval": eval_all, "report": report}[sys.argv[1]]()
