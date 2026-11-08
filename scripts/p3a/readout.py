"""P3a frozen tabular readout (reports/p3a_prereg.md §5-§8). Development evaluation on real VAL.

usage: uv run python scripts/p3a/readout.py [--worlds 0,1,2] [--arms kct,B] [--boot 1000] [--tag ""]
                                            [--split val|test] [--out DIR] [--clean FILE]
Writes OUT/readout{tag}.{json,txt} (OUT defaults to runs/p3a/2018). The store is guarded: train plus the evaluation
split only; --split test needs lpw.sealed.authorize() in this process (reports/p3_joint_addendum.md §J7). With
--clean (a JSON list of the evaluation split's clean node ids, §J4.3), the clean-subset analyses are added; the
evaluation-group summary and the giant-group influence table (§J4.5) are always added.
"""
import argparse
import json
import time

from pathlib import Path

import numpy as np
import polars as pl

from common import ROOT, group_ids, real, store_for, synth
from lpw.p1.atomic import write_text
from lpw.p3.features import SETS, matrix
from lpw.p3.matching import Matcher
from lpw.p3.utility import (DETECTORS, PRIMARY, GroupBoot, c2st, contrast_line, dcr, f1_threshold, fit_score, matched_prevalence_ap,
                            precision_recall_k, ranking_agreement, scores, shortcut_audit, single_auc, transfer_contrast)

FSETS = ("full", "noconf", "confonly", "noprof", "nobal")
SENTINELS = ("end_balance", "net_flow", "ends_near_zero", "sweep_out_share", "min_balance")
ap = argparse.ArgumentParser()
ap.add_argument("--worlds", default="0,1,2")
ap.add_argument("--arms", default="kct,B")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--tag", default="")
ap.add_argument("--split", default="val", choices=("val", "test"))
ap.add_argument("--out", default=str(ROOT))
ap.add_argument("--clean", default=None)
a = ap.parse_args()
OUT = Path(a.out)
worlds, arms = [int(w) for w in a.worlds.split(",")], a.arms.split(",")
t0 = time.time()
st = store_for(a.split)
va = st.agents(a.split)  # the evaluation split (variable names keep the development "va")
gva = group_ids(st, va)
_, _, Rtr = real(st, "train"); ytr = st.role[st.agents("train")]
_, _, Rva = real(st, a.split); yva = st.role[va]
prev_va = float(yva.mean())
boot = GroupBoot(gva, B=a.boot)
matcher = Matcher(Rtr, ytr)
M = matcher.match(Rva, yva)
pidx = np.concatenate([M["pairs"][:, 0], M["pairs"][:, 1]])
MT = bool(M["testable"])  # fewer than 200 pairs: every matched analysis is NOT TESTABLE (joint addendum §J4.2)
mscores = lambda s: scores(yva[pidx], s[pidx]) if MT else None  # noqa: E731
res = {"eval_split": a.split, "worlds": worlds, "arms": arms, "matching": {k: v for k, v in M.items() if k != "pairs"}, "trtr": {}, "arms_res": {}}
# evaluation groups and the giant group (largest by size, then id; joint addendum §J4.5)
_ug, _uc = np.unique(gva, return_counts=True)
giant = int(_ug[np.lexsort((_ug, -_uc))[0]])
keepG = np.flatnonzero(gva != giant)
bootG = GroupBoot(gva[keepG], B=a.boot)
res["eval_groups"] = {"n_agents": int(len(gva)), "n_groups": int(len(_ug)), "giant_group": giant, "giant_size": int((gva == giant).sum()),
                      "giant_scam": int(yva[gva == giant].sum()), "inverse_size_herfindahl": float(1 / ((_uc / _uc.sum()) ** 2).sum())}
CLEAN = None
if a.clean:
    _ids = np.array(json.loads(Path(a.clean).read_text()), np.uint64)
    CLEAN = np.flatnonzero(np.isin(st.node_id[va].astype(np.uint64), _ids))
    bootC = GroupBoot(gva[CLEAN], B=a.boot)
    MC = matcher.match(Rva[pl.Series(CLEAN)], yva[CLEAN]) if len(CLEAN) else {"pairs": np.zeros((0, 2), np.int64), "n_pairs": 0, "testable": False}  # frozen matcher rerun in the clean pool, no refit
    pidxC = CLEAN[np.concatenate([MC["pairs"][:, 0], MC["pairs"][:, 1]])] if MC["n_pairs"] else np.array([], np.int64)
    res["clean"] = {"n_agents": int(len(CLEAN)), "n_scam": int(yva[CLEAN].sum()), "n_groups": int(len(np.unique(gva[CLEAN]))),
                    "matching": {k: v for k, v in MC.items() if k != "pairs"}}
write_text(OUT / f"matched_pairs_{a.split}{a.tag}.json", json.dumps({"store_agent_pairs": [[int(va[i]), int(va[j])] for i, j in M["pairs"]]}))
L = [f"matched val: {M['n_pairs']} pairs (testable {M['testable']}); support scam {M['support_scam']}/{M['n_scam']}; "
     f"max |SMD| after {max(abs(v['after']) for v in M['balance'].values()):.3f}" + ("" if MT else " -> matched analyses NOT TESTABLE")]


# TRTR: real train -> real val
Xr = {fs: matrix(Rtr, fs) for fs in FSETS}
trtr = {(fs, d): fit_score(d, Xr[fs], ytr, [matrix(Rva, fs)]) for fs in FSETS for d in DETECTORS}
thr_real = f1_threshold(PRIMARY, Xr["full"], ytr)
for (fs, d), (s,) in trtr.items():
    res["trtr"][f"{fs}/{d}"] = {**scores(yva, s, thr_real if (fs, d) == ("full", PRIMARY) else None),
                                "matched": mscores(s)}
real_rank = {d: res["trtr"][f"full/{d}"]["auc"] for d in DETECTORS}
real_ap_mp = {d: matched_prevalence_ap(yva, trtr[("full", d)][0], prev_va) for d in DETECTORS}
real_pgain = {m: res["trtr"][f"full/{PRIMARY}"][m] - res["trtr"][f"noprof/{PRIMARY}"][m] for m in ("auc", "ap")}
L.append("TRTR (real train -> val) AUC: " + " ".join(f"{fs}/{d} {res['trtr'][f'{fs}/{d}']['auc']:.3f}" for fs in FSETS for d in DETECTORS))
L.append(f"TRTR primary full: AUC {res['trtr'][f'full/{PRIMARY}']['auc']:.3f} AP {res['trtr'][f'full/{PRIMARY}']['ap']:.3f} "
         f"F1 {res['trtr'][f'full/{PRIMARY}']['f1']:.3f}" + (f"; matched AUC full {res['trtr'][f'full/{PRIMARY}']['matched']['auc']:.3f} "
                                                              f"confonly {res['trtr'][f'confonly/{PRIMARY}']['matched']['auc']:.3f}" if MT else ""))

# owned-only sensitivity (real side observed without IN_agent), primary detector, full set
_, _, Rtr_o = real(st, "train", owned_only=True)
_, _, Rva_o = real(st, a.split, owned_only=True)
(s_o,) = fit_score(PRIMARY, matrix(Rtr_o, "full"), ytr, [matrix(Rva_o, "full")])
res["trtr_owned_only"] = scores(yva, s_o)

names = SETS["full"]
for arm in arms:
    R = {"worlds": {}}
    tstr_by = {k: [] for k in [(fs, d) for fs in FSETS for d in DETECTORS]}
    tstr_owned = []
    for w in worlds:
        agS, _, FS, infS = synth(st, w, arm, "train")
        agH, _, FH, infH = synth(st, w, arm, "hold")
        yS, yH = agS["role"].to_numpy(), agH["role"].to_numpy()
        Xs = {fs: matrix(FS, fs) for fs in FSETS}
        out = {}
        for fs in FSETS:
            for d in DETECTORS:
                head = (fs, d) == ("full", PRIMARY)
                sc = fit_score(d, Xs[fs], yS, [matrix(Rva, fs), matrix(FH, fs)] + ([matrix(Rva_o, fs)] if head else []))
                sv, sh = sc[0], sc[1]
                tstr_by[(fs, d)].append(sv)
                if head:
                    tstr_owned.append(sc[2])
                out[f"{fs}/{d}"] = {"tstr": scores(yva, sv), "tstr_matched": mscores(sv),
                                    "sts": scores(yH, sh), "sts_ap_matched_prev": matched_prevalence_ap(yH, sh, prev_va)}
        thr = f1_threshold(PRIMARY, Xs["full"], yS)
        out[f"full/{PRIMARY}"]["tstr"] = scores(yva, tstr_by[("full", PRIMARY)][-1], thr)
        syn_rank = {d: out[f"full/{d}"]["sts"]["auc"] for d in DETECTORS}
        syn_pgain = {m: out[f"full/{PRIMARY}"]["sts"][m] - out[f"noprof/{PRIMARY}"]["sts"][m] for m in ("auc", "ap")}
        wr = {"O_info": {"train": infS, "hold": infH}, "detectors": out,
              "difficulty_parity": {d: {"synth_ap": out[f"full/{d}"]["sts_ap_matched_prev"], "real_ap": real_ap_mp[d],
                                        "delta": out[f"full/{d}"]["sts_ap_matched_prev"] - real_ap_mp[d],
                                        "pass": bool(abs(out[f"full/{d}"]["sts_ap_matched_prev"] - real_ap_mp[d]) <= 0.05)} for d in DETECTORS},
              "ranking_spearman": ranking_agreement(real_rank, syn_rank),
              "profile_gain": {m: {"synth": syn_pgain[m], "real": real_pgain[m], "same_sign": bool(np.sign(syn_pgain[m]) == np.sign(real_pgain[m])),
                                   "abs_diff": abs(syn_pgain[m] - real_pgain[m]),
                                   "pass": bool(np.sign(syn_pgain[m]) == np.sign(real_pgain[m]) and abs(syn_pgain[m] - real_pgain[m]) <= 0.05)}
                               for m in ("auc", "ap")}}
        # seam (C2ST), per role: real val vs the independent holdout world
        cs = {}
        for fs in ("local", "full"):
            v = {rn: c2st(matrix(Rva, fs)[yva == r], matrix(FH, fs)[yH == r], gva[yva == r]) for r, rn in ((0, "normal"), (1, "scam"))}
            cs[fs] = {**v, "gap": v["scam"] - v["normal"], "pass": bool(v["normal"] <= 0.70 and v["scam"] <= 0.70 and v["scam"] - v["normal"] <= 0.05)}
        wr["c2st"] = cs
        # per-feature seam diagnostic (exploratory): single-feature real-vs-synthetic AUC on the holdout world
        wr["seam_single"] = sorted(((f, abs(single_auc(np.r_[np.zeros(len(Rva)), np.ones(len(FH))],
                                                       np.r_[Rva[f].to_numpy(), FH[f].to_numpy()]) - 0.5)) for f in SETS["local"]),
                                   key=lambda x: -x[1])[:10]
        wr["shortcut"] = shortcut_audit(names, matrix(Rtr, "full"), ytr, matrix(Rva, "full"), yva, Xs["full"], yS, SENTINELS)
        mu, sd = matrix(Rtr, "full").mean(0), matrix(Rtr, "full").std(0) + 1e-9
        wr["mode_coverage"] = {rn: precision_recall_k((matrix(Rva, "full")[yva == r] - mu) / sd, (matrix(FH, "full")[yH == r] - mu) / sd)
                               for r, rn in ((0, "normal"), (1, "scam"))}
        wr["dcr"] = dcr(matrix(Rtr, "full"), ytr, matrix(FH, "full"), yH, matrix(Rva, "full"), yva)
        R["worlds"][w] = wr
        L.append(f"{arm} world {w}: TSTR AUC full/{PRIMARY} {out[f'full/{PRIMARY}']['tstr']['auc']:.3f} | C2ST local "
                 f"{cs['local']['normal']:.3f}/{cs['local']['scam']:.3f} full {cs['full']['normal']:.3f}/{cs['full']['scam']:.3f} | "
                 f"dropped {infS['n_dropped_no_events']}+{infH['n_dropped_no_events']} | {time.time() - t0:.0f}s")
    # arm-level results: the mean over worlds, with the prereg §8 gates applied to the means
    ws = list(R["worlds"].values())
    mean = lambda f: float(np.mean([f(x) for x in ws]))  # noqa: E731
    agg = {"tstr": {k: {m: mean(lambda x: x["detectors"][k]["tstr"][m]) for m in ws[0]["detectors"][k]["tstr"]} for k in ws[0]["detectors"]},
           "tstr_matched": {k: {m: mean(lambda x: x["detectors"][k]["tstr_matched"][m]) for m in ("auc", "ap")} for k in ws[0]["detectors"]}
           if MT else None,
           "ranking_spearman": mean(lambda x: x["ranking_spearman"])}
    agg["difficulty_parity"] = {}
    for d in DETECTORS:
        sa = mean(lambda x: x["difficulty_parity"][d]["synth_ap"])
        agg["difficulty_parity"][d] = {"synth_ap": sa, "real_ap": real_ap_mp[d], "delta": sa - real_ap_mp[d], "pass": bool(abs(sa - real_ap_mp[d]) <= 0.05)}
    agg["profile_gain"] = {}
    for m in ("auc", "ap"):
        g = mean(lambda x: x["profile_gain"][m]["synth"])
        agg["profile_gain"][m] = {"synth": g, "real": real_pgain[m], "pass": bool(np.sign(g) == np.sign(real_pgain[m]) and abs(g - real_pgain[m]) <= 0.05)}
    agg["c2st"] = {}
    for fs in ("local", "full"):
        nm, sm = mean(lambda x: x["c2st"][fs]["normal"]), mean(lambda x: x["c2st"][fs]["scam"])
        agg["c2st"][fs] = {"normal": nm, "scam": sm, "gap": sm - nm, "pass": bool(nm <= 0.70 and sm <= 0.70 and sm - nm <= 0.05)}
    agg["shortcut"] = {}
    for f in ws[0]["shortcut"]:
        dl = mean(lambda x: x["shortcut"][f]["delta"])
        agg["shortcut"][f] = {"real": ws[0]["shortcut"][f]["real"], "delta": dl, "flag": bool(abs(dl) > 0.1), "sentinel": ws[0]["shortcut"][f]["sentinel"]}
    agg["dcr"] = {}
    for r in ("normal", "scam"):
        rs = [x["dcr"][r]["ratio"] for x in ws]
        agg["dcr"][r] = {"ratio": None if None in rs else float(np.mean(rs)), "pass": None if None in rs else bool(np.mean(rs) >= 1),
                         "synth_exact_copies": int(sum(x["dcr"][r]["synth_exact_copies"] for x in ws))}
    agg["mode_coverage"] = {r: {m: mean(lambda x: x["mode_coverage"][r][m]) for m in ("precision", "recall")} for r in ("normal", "scam")}
    R["aggregate"] = agg
    # transfer contrast per detector x set, over worlds; headline: full / PRIMARY
    R["transfer"] = {f"{fs}/{d}": transfer_contrast(yva, tstr_by[(fs, d)], trtr[(fs, d)][0], boot) for fs in FSETS for d in DETECTORS}
    R["transfer_matched"] = {f"{fs}/{PRIMARY}": {"tstr_auc_mean": float(np.mean([scores(yva[pidx], s[pidx])["auc"] for s in tstr_by[(fs, PRIMARY)]])),
                                                 "trtr_auc": res["trtr"][f"{fs}/{PRIMARY}"]["matched"]["auc"]} for fs in FSETS} if MT else None
    for k in ("tstr", "trtr"):
        g = R["transfer_matched"]
        R[f"behavioural_gain_matched_{k}"] = g[f"full/{PRIMARY}"][f"{k}_auc_mean" if k == "tstr" else "trtr_auc"] - \
            g[f"confonly/{PRIMARY}"][f"{k}_auc_mean" if k == "tstr" else "trtr_auc"] if MT else None
    R["transfer_drop_capped"] = None
    if any(R["worlds"][w]["O_info"]["train"]["n_capped"] for w in worlds):  # prereg §9 sensitivity: capped streams removed
        v = []
        for w in worlds:
            agC, _, FC, _ = synth(st, w, arm, "train", drop_capped=True)
            v.append(fit_score(PRIMARY, matrix(FC, "full"), agC["role"].to_numpy(), [matrix(Rva, "full")])[0])
        R["transfer_drop_capped"] = transfer_contrast(yva, v, trtr[("full", PRIMARY)][0], boot)
    R["giant_fixed"] = {f"{fs}/{PRIMARY}": transfer_contrast(yva[keepG], [s[keepG] for s in tstr_by[(fs, PRIMARY)]], trtr[(fs, PRIMARY)][0][keepG], bootG)
                        for fs in ("full", "noconf")}  # nominal influence table (§J4.5)
    if CLEAN is not None:  # clean subset, fixed scores (§J4.3); nominal
        R["clean"] = {f"{fs}/{PRIMARY}": transfer_contrast(yva[CLEAN], [s[CLEAN] for s in tstr_by[(fs, PRIMARY)]], trtr[(fs, PRIMARY)][0][CLEAN], bootC)
                      for fs in ("full", "noconf")}
        cm = {"n_pairs": int(MC["n_pairs"]), "testable": bool(MC["testable"])}
        if MC["testable"]:  # clean + matched: a separately labelled point-estimate estimand
            au = lambda s: scores(yva[pidxC], s[pidxC])["auc"]  # noqa: E731
            for fs in ("full", "confonly"):
                cm[f"{fs}/{PRIMARY}"] = {"tstr_auc_mean": float(np.mean([au(s) for s in tstr_by[(fs, PRIMARY)]])), "trtr_auc": au(trtr[(fs, PRIMARY)][0])}
            cm["behavioural_gain_tstr"] = cm[f"full/{PRIMARY}"]["tstr_auc_mean"] - cm[f"confonly/{PRIMARY}"]["tstr_auc_mean"]
            cm["behavioural_gain_trtr"] = cm[f"full/{PRIMARY}"]["trtr_auc"] - cm[f"confonly/{PRIMARY}"]["trtr_auc"]
        R["clean_matched"] = cm
    R["transfer_owned_only"] = {"tstr_auc_worlds": [scores(yva, s)["auc"] for s in tstr_owned], "trtr_auc": res["trtr_owned_only"]["auc"]}
    h = R["transfer"][f"full/{PRIMARY}"]
    L.append(f"{arm} HEADLINE full/{PRIMARY}: " + contrast_line(h))
    L.append(f"{arm} transfer ratio by set/detector: " + " ".join(f"{k} {v['ratio'] if v['ratio'] is None else round(v['ratio'], 2)}" for k, v in R["transfer"].items()))
    L.append(f"{arm} matched: " + (" ".join(f"{k} TSTR {v['tstr_auc_mean']:.3f} TRTR {v['trtr_auc']:.3f}" for k, v in R["transfer_matched"].items())
                                   + f"; behavioural gain TSTR {R['behavioural_gain_matched_tstr']:+.3f} TRTR {R['behavioural_gain_matched_trtr']:+.3f}"
                                   if MT else "NOT TESTABLE"))
    L.append(f"{arm} ARM-LEVEL (mean over worlds): TSTR full/{PRIMARY} AUC {agg['tstr'][f'full/{PRIMARY}']['auc']:.3f} AP {agg['tstr'][f'full/{PRIMARY}']['ap']:.3f} "
             f"F1 {agg['tstr'][f'full/{PRIMARY}']['f1']:.3f} | difficulty parity {PRIMARY} {agg['difficulty_parity'][PRIMARY]['delta']:+.3f} "
             f"pass {agg['difficulty_parity'][PRIMARY]['pass']} | profile gain AUC {agg['profile_gain']['auc']['synth']:+.3f} vs {real_pgain['auc']:+.3f} "
             f"pass {agg['profile_gain']['auc']['pass']} | C2ST local {agg['c2st']['local']['normal']:.3f}/{agg['c2st']['local']['scam']:.3f} "
             f"pass {agg['c2st']['local']['pass']} (full {agg['c2st']['full']['normal']:.3f}/{agg['c2st']['full']['scam']:.3f}) | "
             f"shortcut flags {[f for f, v in agg['shortcut'].items() if v['flag']]} | DCR {agg['dcr']['normal']['ratio']}/{agg['dcr']['scam']['ratio']} "
             f"| ranking rho {agg['ranking_spearman']:+.2f}")
    L.append(f"{arm} difficulty parity ({PRIMARY}): synth AP " + " ".join(f"{x['difficulty_parity'][PRIMARY]['synth_ap']:.3f}" for x in ws)
             + f" vs real {real_ap_mp[PRIMARY]:.3f}; ranking rho " + " ".join(f"{x['ranking_spearman']:+.2f}" for x in ws)
             + "; profile gain AUC synth " + " ".join(f"{x['profile_gain']['auc']['synth']:+.3f}" for x in ws) + f" vs real {real_pgain['auc']:+.3f}")
    L.append(f"{arm} shortcut flags (world 0): " + " ".join(f"{f} {v['delta']:+.3f}" for f, v in R["worlds"][worlds[0]]["shortcut"].items() if v["flag"]))
    L.append(f"{arm} DCR: " + " ".join(f"w{w} " + " ".join(f"{r} {v['ratio']:.2f}" if v["ratio"] is not None else f"{r} undefined"
                                                         for r, v in x["dcr"].items()) for w, x in R["worlds"].items()))
    L.append(f"{arm} mode coverage P/R: " + " ".join(f"w{w} " + " ".join(f"{r} {v['precision']:.2f}/{v['recall']:.2f}" for r, v in x["mode_coverage"].items())
                                                    for w, x in R["worlds"].items()))
    L.append(f"{arm} owned-only: TSTR {np.round(R['transfer_owned_only']['tstr_auc_worlds'], 3).tolist()} TRTR {R['transfer_owned_only']['trtr_auc']:.3f}")
    res["arms_res"][arm] = R
res["sec"] = round(time.time() - t0)
write_text(OUT / f"readout{a.tag}.json", json.dumps(res, indent=1, default=float))
write_text(OUT / f"readout{a.tag}.txt", "\n".join(L) + "\n")
print("\n".join(L))
