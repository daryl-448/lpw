"""Exact next-event evaluation (SPEC §7.1): every prefix of every agent in a split, each with the simulator's context.

All models are scored on identical prefixes. NLLs are in nats per owned event (time terms are densities per second).
"full" = kind + time + censored + counterparty + amount. "shared" = kind + time + censored + amount, where the
amount term is marginalized over the model's own counterparty choice (B2 never conditions on a counterparty).
With min_t0 (temporal check), prefixes are kept when their target (event, delivery or end of observation) is at or
after the cutoff, so boundary prefixes that started before the cutoff are included.
"""

from collections import defaultdict

import torch

from lpw.p1.batch import Views, exact_specs, pack
from lpw.p1.train import loader, to_dev

FIELDS = ("kind", "time", "cens", "cp", "amt")


@torch.no_grad()
def evaluate(model, store, split: str, dev="cuda", views=None, min_t0: int | None = None) -> dict:
    """With min_t0, only prefixes whose target time is at or after min_t0 are scored (temporal check)."""
    model.eval()
    views = views or Views(store)
    batches = pack(exact_specs(store, store.agents(split)), views)
    S = {g: defaultdict(float) for g in ("all", "normal", "scam")}
    for b in loader(views, batches):
        b = to_dev(b, dev)
        t = model.terms(b, metrics=True)
        role = b["role"][b["row_b"]]
        keep = torch.ones_like(role, dtype=torch.bool) if min_t0 is None else (b["row_t0"] + b["row_r"]) >= min_t0
        oi = t["o_row"]
        okeep = keep[oi]
        orole = role[oi]
        for g, sel_r, sel_o in (("all", keep, okeep), ("normal", keep & (role == 0), okeep & (orole == 0)),
                                ("scam", keep & (role == 1), okeep & (orole == 1))):
            s = S[g]
            s["rows"] += sel_r.sum().item()
            s["owned"] += (sel_r & t["own"]).sum().item()
            s["censored"] += (sel_r & t["cen"]).sum().item()
            for f in ("kind", "time", "cens"):
                s[f] += t[f][sel_r].sum().item()
            for f in ("cp", "amt"):
                s[f] += t[f][sel_o].sum().item()
            s["amt_marg"] += t["amt_marg"][sel_o].sum().item() if "amt_marg" in t else t["amt"][sel_o].sum().item()
            s["kind_ok"] += t["kind_ok"][sel_o].sum().item()
            s["bin_ok"] += t["bin_ok"][sel_o].sum().item()
            s["logdt_err"] += t["logdt_err"][sel_o].sum().item()
            s["amt_err"] += t["amt_err"][sel_o].sum().item()
            if "cls_ok" in t:
                s["cls_ok"] += t["cls_ok"][sel_o].sum().item()
                if t["rep_hits"] is not None:
                    rsel = (t["rep_mask"] & sel_o)[t["rep_mask"]]
                    s["rep"] += rsel.sum().item()
                    s["hits1"] += t["rep_hits"][1][rsel].sum().item()
                    s["hits5"] += t["rep_hits"][5][rsel].sum().item()
    return {g: summarize(s) for g, s in S.items()}


def summarize(s: dict) -> dict:
    n = max(s["owned"], 1)
    out = {"prefixes": int(s["rows"]), "owned": int(s["owned"]), "censored": int(s["censored"])}
    for f in FIELDS:
        out[f"nll_{f}"] = s[f] / n
    out["nll_full"] = sum(s[f] for f in FIELDS) / n
    out["nll_shared"] = sum(s[f] for f in ("kind", "time", "cens", "amt_marg")) / n
    out["nll_amt_marginal"] = s["amt_marg"] / n
    out["nll_full_uncensored"] = sum(s[f] for f in ("kind", "time", "cp", "amt")) / n
    out["kind_error"] = 1 - s["kind_ok"] / n
    out["dtbin_acc"] = s["bin_ok"] / n
    out["logdt_mae"] = s["logdt_err"] / n
    out["amt_mae"] = s["amt_err"] / n
    if "cls_ok" in s:
        out["cls_acc"] = s["cls_ok"] / n
        out["repeat_hits1"] = s["hits1"] / max(s["rep"], 1)
        out["repeat_hits5"] = s["hits5"] / max(s["rep"], 1)
    return out
