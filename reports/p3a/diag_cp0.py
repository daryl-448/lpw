"""Checkpoint 0 diagnosis (prereg §9: one written diagnosis, no generator change). Train-split and holdout-world
data, plus the P1 fidelity scorer on val; no detector of role is fitted."""
import sys
sys.path.insert(0, "scripts/p3a")
import numpy as np, polars as pl
from common import load_pop, raw_events, real, store, group_ids
from lpw.p1.store import PROFILE_FIELDS
from lpw.p1.fidelity import Sample, compare, groups_for, real_frames
from lpw.p3.features import PROF, CONF
from sklearn.metrics import roc_auc_score
st = store(); tr = st.agents("train")
conds, raw = load_pop(0, "train")
coh = np.array([c["cohort"] for c in conds]); pre = coh == 0
rraw = st.prof_raw[tr][st.cohort[tr] == 0]
print("1) integrality of profile fields, pre-existing (share of non-integer values): synth vs real")
for i, f in enumerate(PROFILE_FIELDS):
    s, r = raw[pre, i], rraw[:, i]
    print(f"   {f:28s} synth {np.mean(s != np.round(s)):.3f} real {np.mean(r != np.round(r)):.3f}")
print("2) single-feature real-vs-synthetic AUC (|AUC-0.5|), pre-existing, confonly block")
_, _, Rtr = real(st, "train")
sys_ag = None
from common import synth
ag, _, F, _ = synth(st, 0, "B", "train")
Rp, Fp = Rtr.filter(pl.col("cohort") == 0), F.filter(pl.col("cohort") == 0)
lab = np.r_[np.zeros(Rp.height), np.ones(Fp.height)]
v = sorted(((c, abs(roc_auc_score(lab, np.r_[Rp[c].to_numpy(), Fp[c].to_numpy()]) - 0.5)) for c in CONF + PROF), key=lambda x: -x[1])
print("   " + " ".join(f"{c} {d:.3f}" for c, d in v[:10]))
print("3) kct hold fidelity with long streams (> real max 4839) removed")
va = st.agents("val")
rag, rev = real_frames(st, va); realS = Sample(rag, rev, st.w_start)
pa = rag.filter(pl.col("cohort") == 0)["age_days"].to_numpy(); e = np.quantile(pa, [0, 1/3, 2/3, 1]); e[-1] += 1
groups = groups_for(rag, e)
hc, _ = load_pop(0, "hold")
sag = pl.DataFrame({"agent": np.arange(len(hc)), "role": [c["role"] for c in hc], "cohort": [c["cohort"] for c in hc],
                    "origin": [c["origin"] for c in hc], "b0": [float(c["b0"]) for c in hc], "age_days": [float(c["age_days"]) for c in hc]})
for arm in ("B", "kct"):
    ev = raw_events(0, arm, "hold")
    N = np.bincount(ev["agent"].to_numpy(), minlength=len(hc))
    for lim in (10 ** 9, 4839, 1000):
        keep = np.flatnonzero(N <= lim)
        sub = sag.filter(pl.col("agent").is_in(keep.tolist()))
        sev = ev.filter(pl.col("agent").is_in(keep.tolist()))
        f = compare(realS, Sample(sub, sev, st.w_start), groups, paired=False, synth_groups=groups_for(sub, e))
        print(f"   {arm} streams <= {lim}: removed {len(hc) - len(keep)}, events {sev.height}; normal {f['normal']['pass_rate']:.2f} "
              f"normal/pre {f['normal/pre']['pass_rate']:.2f} scam {f['scam']['pass_rate']:.2f}; normal fails "
              f"{[m for m in f['normal'] if isinstance(f['normal'][m], dict) and not f['normal'][m]['pass']]}")
    top = np.argsort(-N)[:5]
    print(f"   {arm} top streams: " + " ".join(f"{N[i]}(role {hc[i]['role']} coh {hc[i]['cohort']} plan {hc[i]['plan_n']})" for i in top),
          f"; share of normal events in top 5: {N[top].sum() / N[[c['role'] == 0 for c in hc]].sum():.2f}")
print("4) profile arithmetic, pre-existing (share violating): synth vs real train")
P = {f: i for i, f in enumerate(PROFILE_FIELDS)}
def checks(x):
    g = lambda f: x[:, P[f]]
    return {"to_contract+to_eoa > sent+1": g("normal_to_contract_cnt") + g("normal_to_eoa_cnt") > g("normal_sent_cnt") + 1,
            "uniq_peers > all tx counts": g("uniq_peers_cnt") > g("normal_sent_cnt") + g("normal_recv_cnt") + g("erc20_sent_cnt") + g("erc20_recv_cnt") + g("internal_out_cnt") + g("internal_in_cnt"),
            "days_active > age+1": g("days_active") > g("age_days") + 1,
            "days_since_active > age+1": g("days_since_active") > g("age_days") + 1,
            "sessions > days_active*24": g("sessions_cnt") > 24 * np.maximum(g("days_active"), 1),
            "eth_sent>0 but sent_cnt=0": (g("eth_sent_sum") > 0) & (g("normal_sent_cnt") == 0),
            "fee>0 but sent_cnt=0": (g("tx_fee_eth_sum") > 0) & (g("normal_sent_cnt") == 0),
            "sent_cnt>0 but fee=0": (g("normal_sent_cnt") > 0) & (g("tx_fee_eth_sum") == 0)}
cs, cr = checks(raw[pre]), checks(rraw)
for k in cs:
    print(f"   {k:28s} synth {cs[k].mean():.3f} real {cr[k].mean():.3f}")
