"""Standalone single-agent generation (SPEC §7.2): each agent runs the policy alone from its origin to the window end.

Every round, each active agent draws its next owned event from its current prefix (the PROFILE token plus the
latest 255 events, the same context as training): kind, then the exact time law, then counterparty (REPEAT pointer,
unseen VENUE, or NEW), then a truncated-mixture amount. A dormant draw, or a time at or after the window end, ends
the stream. Fees for OUT events are drawn from the real fees of the same calendar day (an environment input).
No IN_agent tokens exist in standalone generation.
"""

import math

import numpy as np
import polars as pl
import torch

from lpw.p1.batch import CTX, collate, time_cont, to_torch
from lpw.p1.model import component_mass, phi
from lpw.p1.store import DUST_LOG10, NEW, OUT, PERIPHERY, REPEAT, SUPPLY_LOG10, TIER_VENUE, VENUE, month_index
from lpw.p1.timelaw import dt_bin, hour_dow

FIELDS = ("kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal", "dt", "dtbin", "hour", "dow", "month")
DTYPES = dict(kind=np.int8, ts=np.float64, cls=np.int8, ent=np.int32, venue=np.int16, amt=np.float32, eth=np.float64,
              fee=np.float64, bal=np.float64, dt=np.float64, dtbin=np.int64, hour=np.int64, dow=np.int64, month=np.int64)


class Stream:
    def __init__(self, cond: dict, w_end: int):
        self.cond = cond
        self.n = 0
        self.cap = 16
        self.a = {f: np.zeros(self.cap, DTYPES[f]) for f in FIELDS}
        self.e_tier, self.e_sc, self.e_venue = [], [], []
        self.b = float(cond["b0"])
        self.w_end = w_end
        self.seen_venues: set[int] = set()

    def view(self) -> dict:
        v = {f: self.a[f][: self.n] for f in FIELDS}
        v.update(e_tier=np.array(self.e_tier, np.int8), e_sc=np.array(self.e_sc, bool), e_venue=np.array(self.e_venue, np.int16),
                 origin=int(self.cond["origin"]), obs_end=self.w_end, role=int(self.cond["role"]),
                 cohort=int(self.cond["cohort"]), prof=self.cond["prof"], bal_end=self.b)
        return v

    def append(self, rec: dict):
        if self.n == self.cap:
            self.cap *= 2
            for f in FIELDS:
                self.a[f] = np.resize(self.a[f], self.cap)
        for f in FIELDS:
            self.a[f][self.n] = rec[f]
        self.n += 1


class FeePool:
    def __init__(self, store):
        tr = np.isin(np.repeat(np.arange(store.n_agents), np.diff(store.off)), store.agents("train")) & (store.kind == OUT)
        day = store.ts[tr] // 86400
        fee = store.fee[tr]
        o = np.argsort(day, kind="stable")
        self.day, self.fee = day[o], fee[o]
        self.u, self.start, self.count = np.unique(self.day, return_index=True, return_counts=True)

    def sample(self, ts: np.ndarray, rng) -> np.ndarray:
        d = np.floor(np.asarray(ts, np.float64) / 86400).astype(np.int64)
        i = np.searchsorted(self.u, d)
        ok = (i < len(self.u)) & (self.u[np.clip(i, 0, len(self.u) - 1)] == d)
        i = np.clip(i, 0, len(self.u) - 1)
        pick = np.where(ok, self.start[i] + (rng.random(len(d)) * self.count[i]).astype(np.int64), rng.integers(0, len(self.fee), len(d)))
        return self.fee[pick]


@torch.no_grad()
def generate(model, conds: list[dict], store, dev="cuda", seed=7, max_events=10000, tok_budget=65536, log=print):
    """conds: dicts with role, cohort, origin, prof (model input vector), b0. Returns (agents_df, events_df, info)."""
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    fees = FeePool(store)
    streams = [Stream(c, store.w_end) for c in conds]
    active = [i for i, c in enumerate(conds) if c["origin"] < store.w_end]
    done_reason = {}
    V = store.n_venues
    venue_sc = torch.as_tensor(store.venue_sc, device=dev)
    rounds = 0
    while active:
        rounds += 1
        nxt = []
        # chunk by token budget
        lens = [min(streams[i].n, CTX) + 1 for i in active]
        order = np.argsort(lens)
        chunks, cur, lmax = [], [], 0
        for oi in order:
            l = lens[oi]
            if cur and (len(cur) + 1) * max(lmax, l) > tok_budget:
                chunks.append(cur)
                cur, lmax = [], 0
            cur.append(active[oi])
            lmax = max(lmax, l)
        if cur:
            chunks.append(cur)
        for ch in chunks:
            views = [streams[i].view() for i in ch]
            specs = [(max(0, v_n - CTX), np.array([v_n - max(0, v_n - CTX)])) for v_n in (streams[i].n for i in ch)]
            B = to_torch(collate(views, specs, V, store.w_start, sample=True), dev)
            res = _step(model, B, g, venue_sc, store, rng, fees, [streams[i] for i in ch])
            for i, keep in zip(ch, res):
                if keep and streams[i].n < max_events:
                    nxt.append(i)
                elif keep:
                    done_reason[i] = "cap"
        active = nxt
    info = {"rounds": rounds, "capped": sum(1 for r in done_reason.values() if r == "cap")}
    return _frames(streams, conds), info


def _step(model, B, g, venue_sc, store, rng, fees, streams) -> list[bool]:
    with torch.autocast("cuda", dtype=torch.float16):
        Hk, Ho = model.encode(B)
    Hk, Ho = Hk.float(), Ho.float()
    rb, rj, t0 = B["row_b"], B["row_j"], B["row_t0"]
    n = len(rb)
    hk, ho = Hk[rb, rj], Ho[rb, rj]
    logpk, law, _ = model.kind_time(hk, ho, t0)
    kappa = torch.multinomial(torch.exp(logpk), 1, generator=g)[:, 0]
    dt2, dorm2 = law.sample(g)
    sel = torch.arange(n, device=rb.device) * 2 + kappa
    dt, dormant = dt2[sel], dorm2[sel]
    t = t0 + dt
    go = (~dormant) & (t < store.w_end)
    keep = go.cpu().numpy()
    if not keep.any():
        return [False] * n
    idx = torch.nonzero(go)[:, 0]
    t_np, dt_np = t[idx].cpu().numpy(), dt[idx].cpu().numpy()
    ko = kappa[idx]
    tbin = torch.as_tensor(dt_bin(dt_np), device=rb.device)
    tmon = torch.as_tensor(month_index(t_np, store.w_start), device=rb.device)
    tc = torch.as_tensor(time_cont(t_np, dt_np), device=rb.device)
    tt = model.time_emb(tbin, tmon, tc)
    # owned rows are one per window, in window order, so e_* row i = window i
    E = {k: v[idx] for k, v in B.items() if k.startswith("e_")}
    Bsub = dict(E)
    Bsub["role"] = B["role"]
    ctx = model.cp_context(B, rb[idx], ho[idx], ko, tt)
    ent_ok, ven_ok = model.masks(Bsub, ko)
    lg = model.gate_logp(ctx, ent_ok, ven_ok)
    cls = torch.multinomial(torch.exp(lg), 1, generator=g)[:, 0]
    ent = torch.full_like(cls, -1)
    ven = torch.full_like(cls, -1)
    cpvec = torch.zeros_like(ctx)
    rep = torch.nonzero(cls == REPEAT)[:, 0]
    if len(rep):
        lptr, ph = model.pointer_logp(Bsub, Ho, rep, rb[idx][rep], rj[idx][rep], ctx[rep], t[idx][rep], ent_ok[rep])
        e = torch.multinomial(torch.exp(lptr), 1, generator=g)[:, 0]
        ent[rep] = e
        cpvec[rep] = model.ent_feat(ph[torch.arange(len(rep)), e])
    vv = torch.nonzero(cls == VENUE)[:, 0]
    if len(vv):
        lv = model.venue_logp(ctx[vv], ven_ok[vv])
        v = torch.multinomial(torch.exp(lv), 1, generator=g)[:, 0]
        ven[vv] = v
        cpvec[vv] = model.e_venue(v)
    led = B["o_led"][idx]
    ap = model.amount_params(B, rb[idx], ctx, cls, cpvec, led)
    amt = _sample_trunc_mog(ap, model.cfg.k_mix, g).cpu().numpy()
    k_np, c_np, e_np, v_np = ko.cpu().numpy(), cls.cpu().numpy(), ent.cpu().numpy(), ven.cpu().numpy()
    fee = fees.sample(t_np, rng)
    h, dw = hour_dow(t_np)
    mon = month_index(t_np, store.w_start)
    bins = dt_bin(dt_np)
    vsc = store.venue_sc
    for r, wi in enumerate(idx.cpu().numpy()):
        s = streams[wi]
        c = int(c_np[r])
        if c == REPEAT:
            en = int(e_np[r])
            vi = s.e_venue[en]
        elif c == VENUE:
            vi = int(v_np[r])
            en = len(s.e_tier)
            s.e_tier.append(TIER_VENUE), s.e_sc.append(bool(vsc[vi])), s.e_venue.append(vi)
        else:
            vi = -1
            en = len(s.e_tier)
            s.e_tier.append(PERIPHERY), s.e_sc.append(True), s.e_venue.append(-1)
        kk = int(k_np[r])
        eth = 10.0 ** float(amt[r])
        f = float(fee[r]) if kk == OUT else 0.0
        s.append(dict(kind=kk, ts=float(t_np[r]), cls=c, ent=en, venue=vi, amt=float(amt[r]), eth=eth, fee=f, bal=s.b,
                      dt=float(dt_np[r]), dtbin=int(bins[r]), hour=int(h[r]), dow=int(dw[r]), month=int(mon[r])))
        s.b += -eth - f if kk == OUT else eth
    return list(keep)


def _sample_trunc_mog(params, k, g):
    logit, mu, ls = params.double().split(k, -1)
    sig = torch.exp(ls.clamp(-4, 2))
    # component weights renormalized by each component's mass inside [floor, supply], then an exact truncated draw
    lw = torch.log_softmax(logit, -1) + component_mass(mu, sig)
    m = torch.multinomial(torch.softmax(lw, -1), 1, generator=g)[:, 0]
    mu_m, sig_m = mu.gather(1, m[:, None])[:, 0], sig.gather(1, m[:, None])[:, 0]
    return mu_m + sig_m * _trunc_std_normal((DUST_LOG10 - mu_m) / sig_m, (SUPPLY_LOG10 - mu_m) / sig_m, g)


def _trunc_std_normal(a, b, g):
    """Draw of Z ~ N(0,1) restricted to [a, b], by inverse CDF on the side of the mean the interval lies on (so tail
    intervals keep their precision). Intervals more than 30σ out, where Φ nears underflow, use the exponential tail
    limit z = edge + E / edge with E ~ Exp(1) truncated to the interval, accurate to O(1/edge²)."""
    u = torch.rand(a.shape, dtype=torch.float64, device=a.device, generator=g)
    ln = torch.special.log_ndtr
    # right tail: Z = -Φ^-1(p) with p uniform on [Φ(-b), Φ(-a)]; left tail and middle: Z = Φ^-1(p), p on [Φ(a), Φ(b)]
    flip = a > 0
    lo = torch.where(flip, ln(-b), ln(a))
    hi = torch.where(flip, ln(-a), ln(b))
    lp = hi + torch.log1p(-u * (1 - torch.exp(torch.clamp(lo - hi, max=0.0))))  # log(P_hi − u·(P_hi − P_lo)), P = Φ
    z = torch.special.ndtri(torch.exp(lp).clamp(min=1e-300, max=1 - 1e-16))
    z = torch.where(flip, -z, z)
    edge = torch.where(flip, a, -b).clamp(min=1.0)  # distance of the near endpoint from the mean
    width = (b - a).clamp(max=1e300)
    far = torch.where(flip, a, -b) > 30
    u2 = torch.rand(a.shape, dtype=torch.float64, device=a.device, generator=g)
    e = -torch.log1p(-u2 * -torch.expm1(-edge * width))  # Exp(1) truncated to [0, edge·width]
    z = torch.where(far, torch.where(flip, a + e / edge, b - e / edge), z)
    return torch.minimum(torch.maximum(z, a), b)


def _frames(streams, conds):
    rows = []
    for i, s in enumerate(streams):
        v = s.view()
        rows.append(pl.DataFrame({"agent": np.full(s.n, i, np.int64), **{f: v[f] for f in ("kind", "ts", "cls", "ent", "venue", "amt", "eth", "fee", "bal")}}))
    ev = pl.concat(rows) if rows else pl.DataFrame()
    ag = pl.DataFrame({"agent": np.arange(len(conds)), "role": [c["role"] for c in conds], "cohort": [c["cohort"] for c in conds],
                       "origin": [c["origin"] for c in conds], "b0": [float(c["b0"]) for c in conds],
                       "age_days": [float(c.get("age_days", 0.0)) for c in conds]})
    return ag, ev
