"""P3b pre-freeze measurements (Sol consult §10). Feasibility facts and model limitations, before the protocol is frozen.

Roles are used on real train only. Val is read label-blind (no role is joined to any val quantity). The test split is
never loaded (lpw.p3b.access). No synthetic attachment, no val utility, no C2ST.

A. Train, roles allowed:
  A1 group templates (s, k), singleton support by role, the largest templates; the conditional prior (lpw.p3b.groups)
     simulated on the six P3a world-0..2 train / hold role totals with 200 non-frozen seeds (90000+): proposals,
     singleton remainder, largest group, agent-weighted size quantiles vs train.
  A2 first touches, empty-pool shares, unconditional and conditional (C_e nonempty) attachment rates, by role,
     direction, cohort and group-size bin (2-4, 5-16, 17-100, >100); the frozen p0[r] values.
  A3 choice-set sizes, total pointer alternatives, same-second co-creation share.
  A4 within- vs cross-group sharing of eligible nodes with >= 2 focal agents: by node, by incidence and by
     transaction; the largest sharing component with all nodes vs with only single-group nodes.
  A5 co-activation: mean within-group SD of first-event day in multi-member groups vs a uniform-within-role
     placement null (20 permutations of agents within role over the same slots).
  A6 (pre-specified here, before running) train-only role utility of the fixed STRUCT block: 3 whole-group folds
     (groups shuffled with seed 0, assigned round-robin), STRUCT computed in each fold's own graph (held-out fold
     alone; the two fitting folds together), XGB from lpw.p3.utility (seed 0) on P3a `local` (L) and L + STRUCT,
     held-out AUC per fold and R_oof = mean(AUC(L+S) - AUC(L)); also STRUCT-only AUC. No feature, model or threshold
     is selected from it. It only warns whether the utility verdict may be NOT TESTABLE.
B. Label-blind train / val structure:
  B1 group-size distributions (group- and agent-weighted), largest share, H = sum (n_g / N)^2 and 1 / H; daily active
     focal counts; events and eligible partners per agent.
  B2 STRUCT means and recovered-community sizes on the full train view, on uniform label-blind train thinnings to
     3,741 and 1,870 agents (5 seeds), and on the val view and its disjoint halves (5 seeds).
Output: runs/p3b/2018/prefreeze.{json,txt}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/p3a")
from lpw.p3.features import SETS, matrix, node_features  # noqa: E402
from lpw.p3.observe import observe_real  # noqa: E402
from lpw.p3.utility import fit_score, single_auc  # noqa: E402
from lpw.p3b import groups as G  # noqa: E402
from lpw.p3b.access import load, real_pop  # noqa: E402
from lpw.p3b.graph import STRUCT, incidence, louvain_sizes, struct_features  # noqa: E402
from lpw.p3b.replay import replay  # noqa: E402

OUT = Path("runs/p3b/2018")
OUT.mkdir(parents=True, exist_ok=True)
DAY = 86400
t0 = time.time()
R = load()
st = R.st
res, L = {}, []


def log(s):
    print(s, flush=True)
    L.append(s)


def view(pop, sel):
    """Incidence of the focal subset `sel` (indices into pop agents), renumbered 0..len(sel)-1."""
    sel = np.asarray(sel)
    pos = np.full(pop.n, -1)
    pos[sel] = np.arange(len(sel))
    agent = np.repeat(np.arange(pop.n), np.diff(pop.off))
    m = pop.elig & (pos[agent] >= 0)
    return incidence(pos[agent[m]], pop.cp[m], len(sel))


ptr, pva = real_pop(R, "train")[0], real_pop(R, "val")[0]
ytr = ptr.role

# ---------------- A1 templates and prior
keys, counts, n_train = G.templates(ptr.group, ytr)
gsz = np.bincount(np.unique(ptr.group, return_inverse=True)[1])
gsc = np.bincount(np.unique(ptr.group, return_inverse=True)[1], weights=ytr)
sing = gsz == 1
res["A1"] = {"n_templates": int(len(keys)), "n_multi_groups": int(counts.sum()),
             "singletons_normal": int((sing & (gsc == 0)).sum()), "singletons_scam": int((sing & (gsc == 1)).sum()),
             "largest_templates": [[int(s), int(k), int(c)] for (s, k), c in sorted(zip(map(tuple, keys), counts), key=lambda x: -x[0][0])[:8]]}
log(f"A1 templates {len(keys)}, multi groups {counts.sum()}, singletons normal/scam {res['A1']['singletons_normal']}/{res['A1']['singletons_scam']}")
log(f"   largest (s, k, count): {res['A1']['largest_templates']}")
tr_agent_sizes = gsz[np.unique(ptr.group, return_inverse=True)[1]]
qs = [0.5, 0.75, 0.9, 0.99]
log(f"   train agent-weighted size quantiles {qs}: {np.quantile(tr_agent_sizes, qs).tolist()}, max share {gsz.max() / len(ytr):.4f}")
sims = {}
for w in (0, 1, 2):
    for part in ("train", "hold"):
        role = np.load(f"runs/p3a/2018/world{w}/pop_{part}.npz")["role"]
        props, sing_r, mx, q = [], [], [], []
        for s in range(200):
            g, info = G.sample(role, keys, counts, n_train, 90000 + s)
            props.append(info["proposals"]); sing_r.append(info["singletons_normal"] + info["singletons_scam"]); mx.append(info["max_group"])
            sz = np.bincount(g)[g]
            q.append(np.quantile(sz, qs))
        sims[f"w{w}/{part}"] = {"N": int(len(role)), "N1": int(role.sum()), "mean_proposals": float(np.mean(props)),
                                "max_proposals": int(np.max(props)), "singleton_share": float(np.mean(sing_r) / len(role)),
                                "max_group_q": np.quantile(mx, [0.1, 0.5, 0.9]).tolist(), "p_max_ge_500": float(np.mean(np.array(mx) >= 500)),
                                "agent_size_q": np.mean(q, 0).tolist()}
        log(f"   prior {w}/{part}: N {len(role)} proposals mean {np.mean(props):.2f} max {np.max(props)}; singleton share "
            f"{sims[f'w{w}/{part}']['singleton_share']:.3f} (train {sing.sum() / len(ytr):.3f}); max group q10/50/90 "
            f"{sims[f'w{w}/{part}']['max_group_q']}; P(max >= 500) {sims[f'w{w}/{part}']['p_max_ge_500']:.2f}; agent size q {np.round(np.mean(q, 0), 1).tolist()}")
res["A1"]["prior_sims"] = sims

# ---------------- A2/A3 replay
rr = replay(ptr)
nz = rr.opp_nc > 0
rb, born, out = rr.opp_bin[:, 0], rr.opp_bin[:, 1], rr.opp_bin[:, 2]
gs_opp = gsz[np.unique(ptr.group, return_inverse=True)[1]][np.repeat(np.arange(ptr.n), np.diff(ptr.off))[rr.opp_event]]
p0 = {}
A2 = {}
for r in (0, 1):
    m = rb == r
    p0[r] = float(rr.opp_attach[m & nz].sum() / (m & nz).sum())
    A2[f"role{r}"] = {"first_touches": int(m.sum()), "nonempty_share": float(nz[m].mean()), "uncond_rate": float(rr.opp_attach[m].mean()),
                      "p0": p0[r], "n_opportunity": int((m & nz).sum()), "n_attach": int(rr.opp_attach[m].sum())}
    log(f"A2 role {r}: first touches {m.sum()}, nonempty {nz[m].mean():.3f}, uncond {rr.opp_attach[m].mean():.4f}, p0 {p0[r]:.5f}")
    for name, cond in (("OUT", out == 1), ("IN", out == 0), ("born", born == 1), ("pre", born == 0),
                       ("g2-4", (gs_opp >= 2) & (gs_opp <= 4)), ("g5-16", (gs_opp >= 5) & (gs_opp <= 16)),
                       ("g17-100", (gs_opp >= 17) & (gs_opp <= 100)), ("g>100", gs_opp > 100)):
        mm = m & cond
        A2[f"role{r}"][name] = {"n": int(mm.sum()), "nonempty": float(nz[mm].mean()) if mm.any() else None,
                                "cond_rate": float(rr.opp_attach[mm & nz].mean()) if (mm & nz).any() else None}
        log(f"   {name}: n {mm.sum()} nonempty {A2[f'role{r}'][name]['nonempty']} cond {A2[f'role{r}'][name]['cond_rate']}")
res["A2"] = A2
res["p0"] = p0
pos_nc = np.array([len(f) for _, f, _ in rr.ptr])
res["A3"] = {"nonempty_nc_q": np.quantile(rr.opp_nc[nz], [0.5, 0.9, 0.99, 1]).tolist(), "positives": int(len(pos_nc)),
             "positive_nc_q": np.quantile(pos_nc, [0.5, 0.9, 0.99, 1]).tolist(), "total_alternatives": int(pos_nc.sum()),
             "cocreate_share": float(rr.opp_cocreate.mean()), "cocreate_n": int(rr.opp_cocreate.sum()),
             "positives_in_groups_gt100": float((gs_opp[rr.opp_attach] > 100).mean())}
log(f"A3 {res['A3']}")

# ---------------- A4 cross-group sharing (train view)
agent_of = np.repeat(np.arange(ptr.n), np.diff(ptr.off))
m = ptr.elig
B, nodes = incidence(agent_of[m], ptr.cp[m], ptr.n)
coo = B.tocoo()
d = np.asarray(B.sum(0)).ravel()
gmin = np.full(B.shape[1], np.iinfo(np.int64).max)
gmax = np.full(B.shape[1], np.iinfo(np.int64).min)
np.minimum.at(gmin, coo.col, ptr.group[coo.row])
np.maximum.at(gmax, coo.col, ptr.group[coo.row])
shared = d >= 2
cross = shared & (gmin != gmax)
inc_shared = shared[coo.col]
inc_cross = cross[coo.col]
ev_col = np.searchsorted(nodes, ptr.cp[m])
from lpw.p3b.graph import components  # noqa: E402
K_all = components(B)
B_within = B[:, np.flatnonzero(~cross)]
K_within = components(B_within)
res["A4"] = {"shared_nodes": int(shared.sum()), "cross_share_nodes": float(cross.sum() / shared.sum()),
             "cross_share_incidence": float(inc_cross.sum() / inc_shared.sum()),
             "cross_share_events": float(cross[ev_col].sum() / shared[ev_col].sum()),
             "largest_component_all": int(K_all.max()), "largest_component_within_only": int(K_within.max()),
             "agents_in_components_ge2_all": int((K_all >= 2).sum()), "agents_in_components_ge2_within": int((K_within >= 2).sum())}
log(f"A4 {res['A4']}")

# ---------------- A5 co-activation
first_day = np.array([ptr.sec[ptr.off[a]] // DAY for a in range(ptr.n)], np.float64)
inv = np.unique(ptr.group, return_inverse=True)[1]
multi = gsz[inv] >= 2


def within_sd(gidx):
    sd = []
    for g in np.unique(gidx[multi]):
        x = first_day[gidx == g]
        sd.append(x.std())
    return float(np.mean(sd))


real_sd = within_sd(inv)
null = []
for s in range(20):
    perm = inv.copy()
    for r in (0, 1):
        idx = np.flatnonzero(ytr == r)
        perm[idx] = inv[np.random.default_rng(np.random.SeedSequence([7000 + s, r])).permutation(idx)]
    # multi mask follows the slots, not the agents
    multi_slots = gsz[perm] >= 2
    sd = [first_day[perm == g].std() for g in np.unique(perm[multi_slots])]
    null.append(float(np.mean(sd)))
res["A5"] = {"within_group_sd_days": real_sd, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null))}
log(f"A5 within-group first-day SD real {real_sd:.1f} vs uniform-within-role null {np.mean(null):.1f} ± {np.std(null):.1f}")

# ---------------- A6 train-only OOF utility (pre-specified above)
ag_tr = R.agents("train")
oa, oe = observe_real(st, ag_tr)
Ftr = node_features(oa, oe, st.w_start, st.w_end)
XL = matrix(Ftr, "local")
ug = np.unique(ptr.group)
fold_of_group = dict(zip(np.random.default_rng(0).permutation(ug).tolist(), np.arange(len(ug)) % 3))
fold = np.array([fold_of_group[int(g)] for g in ptr.group])
A6 = []
for k in range(3):
    te, trn = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
    S = {}
    for nm, sel in (("te", te), ("tr", trn)):
        Bv, _ = view(ptr, sel)
        f = struct_features(Bv)
        S[nm] = np.stack([f[c] for c in STRUCT], 1)
    (sL,) = fit_score("xgb", XL[trn], ytr[trn], [XL[te]])
    (sLS,) = fit_score("xgb", np.hstack([XL[trn], S["tr"]]), ytr[trn], [np.hstack([XL[te], S["te"]])])
    (sS,) = fit_score("xgb", S["tr"], ytr[trn], [S["te"]])
    a = {"auc_L": single_auc(ytr[te], sL), "auc_LS": single_auc(ytr[te], sLS), "auc_S": single_auc(ytr[te], sS), "n_te": int(len(te))}
    a["R"] = a["auc_LS"] - a["auc_L"]
    A6.append(a)
    log(f"A6 fold {k}: n {len(te)} AUC L {a['auc_L']:.4f} L+S {a['auc_LS']:.4f} S {a['auc_S']:.4f} R {a['R']:+.4f}")
res["A6"] = {"folds": A6, "R_oof": float(np.mean([a["R"] for a in A6]))}
log(f"A6 R_oof {res['A6']['R_oof']:+.4f}")

# ---------------- B1 label-blind structure
B1 = {}
for nm, pop in (("train", ptr), ("val", pva)):
    inv_, sz = np.unique(pop.group, return_inverse=True)[1], None
    sz = np.bincount(inv_)
    N = pop.n
    H = float(((sz / N) ** 2).sum())
    days = pop.sec // DAY
    agent = np.repeat(np.arange(N), np.diff(pop.off))
    act = np.unique(np.stack([days, agent], 1), axis=0)
    per_day = np.bincount(act[:, 0] - act[:, 0].min())
    ev_per = np.diff(pop.off)
    part = np.bincount(agent[pop.elig & pop.first], minlength=N)
    B1[nm] = {"N": int(N), "group_size_q_groupw": np.quantile(sz, qs).tolist(), "group_size_q_agentw": np.quantile(sz[inv_], qs).tolist(),
              "max_share": float(sz.max() / N), "H": H, "eff_groups": 1 / H, "daily_active_q": np.quantile(per_day, [0.1, 0.5, 0.9]).tolist(),
              "events_q": np.quantile(ev_per, qs).tolist(), "elig_partners_q": np.quantile(part, qs).tolist()}
    log(f"B1 {nm}: {B1[nm]}")
res["B1"] = B1

# ---------------- B2 thinning
B2 = {}


def summarize(pop, sel, louv=True):
    Bv, nd = view(pop, sel)
    f = struct_features(Bv)
    out = {c: float(f[c].mean()) for c in STRUCT}
    if louv:
        ls = louvain_sizes(Bv, nd)
        out["louvain_agentw_q90"] = float(np.quantile(ls, 0.9))
        out["louvain_max"] = int(ls.max())
    out["node_deg_mean"] = float(np.asarray(Bv.sum(0)).mean()) if Bv.shape[1] else 0.0
    return out


tl = time.time()
B2["train_full"] = summarize(ptr, np.arange(ptr.n))
log(f"B2 train full ({time.time() - tl:.0f}s): {B2['train_full']}")
for n_sub in (3741, 1870):
    rows = [summarize(ptr, np.sort(np.random.default_rng(s).choice(ptr.n, n_sub, replace=False))) for s in range(5)]
    B2[f"train_{n_sub}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    log(f"B2 train thinned {n_sub}: {B2[f'train_{n_sub}']}")
B2["val_full"] = summarize(pva, np.arange(pva.n))
log(f"B2 val full: {B2['val_full']}")
rows = []
for s in range(5):
    p = np.random.default_rng(s).permutation(pva.n)
    for h in (p[: pva.n // 2], p[pva.n // 2: 2 * (pva.n // 2)]):
        rows.append(summarize(pva, np.sort(h)))
B2["val_halves"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
log(f"B2 val halves: {B2['val_halves']}")
res["B2"] = B2
res["sec"] = round(time.time() - t0)
(OUT / "prefreeze.json").write_text(json.dumps(res, indent=1))
(OUT / "prefreeze.txt").write_text("\n".join(L) + "\n")
log(f"done {res['sec']}s")
