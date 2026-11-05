"""Windows and collation (SPEC §6.4).

A window with start s holds PROFILE + events s+1..s+255 (1-based); window position j > 0 is event s+j.
The prefix scored at position j is k = s + j events; its target is event k+1 (0-based index k), or the end of
observation when k = n. Prefix k's partner table and ledger include every event before it, even ones evicted
from the window.

Tiling:
  train: the s = 0 window scores prefixes 0..min(n, 255). Longer streams add windows s = off + 128 m with a random
         off in [1, 128], each scoring only positions >= 128 (so every scored prefix sees >= 127 events of context).
  val:   the same tiling with off = 128 (cheap early-stopping metric).
  exact: the s = 0 window, plus one window per prefix k > 255 with s = k - 255, scored at its last position.
         Every prefix then sees exactly the context the simulator uses. All reported NLLs use this.
"""

import numpy as np
import torch

from lpw.p1.store import IN_AGENT, OUT, PAD, PROFILE, REPEAT, SUPPLY_ETH, derived
from lpw.p1.timelaw import dt_bin, hour_dow

CTX = 255
HALF = 128
N_TOK_CONT = 8
N_TIME_CONT = 5


class Views:
    """Cached per-agent views of a store, with derived fields."""

    def __init__(self, store):
        self.store = store
        self.cache: dict[int, dict] = {}
        nonrep = (store.cls != REPEAT).astype(np.int64)
        self.first_cum = np.concatenate([[0], np.cumsum(nonrep)])

    def __call__(self, a: int) -> dict:
        v = self.cache.get(a)
        if v is None:
            v = self.store.view(a)
            v.update(derived(v, self.store.w_start))
            self.cache[a] = v
        return v

    def n_entries_before(self, a: int, m: int) -> int:
        o = self.store.off[a]
        return int(self.first_cum[o + m] - self.first_cum[o])


def train_specs(store, agents, rng) -> list[tuple[int, int, np.ndarray]]:
    return _tile(store, agents, lambda: int(rng.integers(1, HALF + 1)))


def val_specs(store, agents) -> list[tuple[int, int, np.ndarray]]:
    return _tile(store, agents, lambda: HALF)


def _tile(store, agents, offset):
    specs = []
    for a in agents:
        n = store.n_events(a)
        if store.origin[a] >= store.obs_end[a]:
            continue
        specs.append((a, 0, np.arange(0, min(n, CTX) + 1)))
        if n > CTX:
            s = offset()
            while s + HALF <= n:
                specs.append((a, s, np.arange(HALF, min(n - s, CTX) + 1)))
                s += HALF
    return specs


def exact_specs(store, agents) -> list[tuple[int, int, np.ndarray]]:
    specs = []
    for a in agents:
        n = store.n_events(a)
        if store.origin[a] >= store.obs_end[a]:
            continue
        specs.append((a, 0, np.arange(0, min(n, CTX) + 1)))
        for k in range(CTX + 1, n + 1):
            specs.append((a, k - CTX, np.array([CTX])))
    return specs


def pack(specs, views: Views, tok_budget=8192, ent_budget=1_500_000, rng=None) -> list[list]:
    """Group specs into batches under a token budget (B × L_max) and a pointer budget (rows × P_max)."""
    st = views.store
    cost = []
    for sp in specs:
        a, s, js = sp
        ne = min(st.n_events(a) - s, CTX)
        cost.append((views.n_entries_before(a, s + ne + (1 if s + ne < st.n_events(a) else 0)), 1 + ne, len(js)))
    order = sorted(range(len(specs)), key=lambda i: (cost[i][0], cost[i][1]))
    batches, cur, lmax, pmax, rows = [], [], 0, 0, 0
    for i in order:
        p, l, r = cost[i]
        nl, np_, nr = max(lmax, l), max(pmax, p, 1), rows + r
        if cur and ((len(cur) + 1) * nl > tok_budget or nr * np_ > ent_budget):
            batches.append(cur)
            cur, nl, np_, nr = [], l, max(p, 1), r
        cur.append(specs[i])
        lmax, pmax, rows = nl, np_, nr
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def _entry_state(v: dict, s: int, ks: np.ndarray, P: int):
    """State of every partner entry at each prefix k in ks (events [0, k)): count, total in, total out,
    last timestamp, and window position of the last occurrence (-1 if it is not in the window)."""
    ent, kind, eth, ts = v["ent"], v["kind"], v["eth"], v["ts"]
    R = len(ks)
    cnt = np.zeros((R, P), np.float32)
    # totals stay float64: generated heavy-tail amounts can exceed the float32 range
    tin = np.zeros((R, P), np.float64)
    tout = np.zeros((R, P), np.float64)
    last = np.full((R, P), -1, np.int64)
    if P == 0 or R == 0:
        return cnt, tin, tout, np.zeros((R, P), np.float64), last
    w_in = np.where(kind != OUT, eth, 0.0)
    w_out = np.where(kind == OUT, eth, 0.0)
    if R <= 2:
        for r, k in enumerate(ks):
            idx = ent[:k]
            cnt[r] = np.bincount(idx, minlength=P)[:P]
            tin[r] = np.bincount(idx, weights=w_in[:k], minlength=P)[:P]
            tout[r] = np.bincount(idx, weights=w_out[:k], minlength=P)[:P]
            np.maximum.at(last[r], idx, np.arange(k))
    else:
        idx = ent[:s]
        c0 = np.bincount(idx, minlength=P)[:P].astype(np.float32)
        i0 = np.bincount(idx, weights=w_in[:s], minlength=P)[:P]
        o0 = np.bincount(idx, weights=w_out[:s], minlength=P)[:P]
        l0 = np.full(P, -1, np.int64)
        np.maximum.at(l0, idx, np.arange(s))
        m = int(ks.max()) - s  # in-window tokens needed
        if m > 0:
            we = ent[s:s + m]
            oh = np.zeros((m, P), np.float32)
            oh[np.arange(m), we] = 1.0
            cc = np.cumsum(oh, 0)
            ci = np.cumsum(oh * w_in[s:s + m, None], 0)
            co = np.cumsum(oh * w_out[s:s + m, None], 0)
            lw = np.maximum.accumulate(np.where(oh > 0, (s + np.arange(m))[:, None], -1), 0)
        for r, k in enumerate(ks):
            j = k - s
            if j > 0:
                cnt[r] = c0 + cc[j - 1]
                tin[r] = i0 + ci[j - 1]
                tout[r] = o0 + co[j - 1]
                last[r] = np.maximum(l0, lw[j - 1])
            else:
                cnt[r], tin[r], tout[r], last[r] = c0, i0, o0, l0
    last_ts = np.where(last >= 0, ts[np.clip(last, 0, None)], 0).astype(np.float64)
    wpos = np.where(last >= s, last - s + 1, -1)
    return cnt, tin, tout, last_ts, wpos


def _led(b):
    b = np.clip(np.asarray(b, np.float64), -SUPPLY_ETH, SUPPLY_ETH)
    return np.stack([np.log1p(np.abs(b)) / 5.0, np.sign(b)], -1)


def time_cont(ts: np.ndarray, dt: np.ndarray) -> np.ndarray:
    h, d = hour_dow(ts)
    return np.stack([np.log1p(np.maximum(dt, 0)) / 10.0, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], -1).astype(np.float32)


def collate(views: list[dict], specs: list[tuple[int, np.ndarray]], n_venues: int, w_start: int, sample: bool = False) -> dict:
    """views[i] is the agent view of window i; specs[i] = (s, scored positions). With sample=True, rows have no target
    (generation): each scored position yields a row whose entry state is computed for drawing the next event."""
    from lpw.p1.store import month_index

    B = len(views)
    lens = [1 + min(len(v["ts"]) - s, CTX) for v, (s, _) in zip(views, specs)]
    L = max(lens)
    tok = {k: np.full((B, L), fill, np.int64) for k, fill in
           [("kind", PAD), ("cls", 3), ("venue", n_venues), ("tier", 3), ("dtbin", 64), ("month", 12)]}
    tok["kind"][:, 0] = PROFILE
    cont = np.zeros((B, L, N_TOK_CONT), np.float32)
    role = np.zeros(B, np.int64)
    cohort = np.zeros(B, np.int64)
    prof = np.zeros((B, len(views[0]["prof"])), np.float32)
    ocont = np.zeros((B, 6), np.float32)
    rows = {k: [] for k in ("b", "j", "type", "t0", "r", "tkind")}
    orow = {k: [] for k in ("row", "cls", "venue", "ent", "amt", "tcont", "tbin", "tmonth", "thour", "tdow", "led")}
    ent_parts = []
    pmax = 1
    for b, (v, (s, js)) in enumerate(zip(views, specs)):
        n = len(v["ts"])
        ne = lens[b] - 1
        sl = slice(s, s + ne)
        tok["kind"][b, 1:1 + ne] = v["kind"][sl]
        tok["cls"][b, 1:1 + ne] = v["cls"][sl]
        tok["venue"][b, 1:1 + ne] = np.where(v["venue"][sl] >= 0, v["venue"][sl], n_venues)
        tok["tier"][b, 1:1 + ne] = v["e_tier"][v["ent"][sl]] if ne else []
        tok["dtbin"][b, 1:1 + ne] = v["dtbin"][sl]
        tok["month"][b, 1:1 + ne] = v["month"][sl]
        if ne:
            cont[b, 1:1 + ne, :5] = time_cont(v["ts"][sl], v["dt"][sl])
            cont[b, 1:1 + ne, 5] = v["amt"][sl] / 3.0
            cont[b, 1:1 + ne, 6:8] = _led(v["bal"][sl])
        role[b], cohort[b], prof[b] = v["role"], v["cohort"], v["prof"]
        o = np.array([v["origin"]])
        ocont[b, :5] = time_cont(o, np.zeros(1))[0]
        ocont[b, 0] = month_index(o, w_start)[0] / 11.0
        ocont[b, 5] = np.log1p(s) / 5.0
        # entries seen before the last scored prefix (every REPEAT target is among them)
        P = int(v["ent"][: s + ne].max()) + 1 if s + ne > 0 else 0
        own_ks = []
        for j in js:
            k = s + int(j)
            t0 = float(v["ts"][k - 1]) if k >= 1 else float(v["origin"])
            if sample:
                typ, tgt_t, tk = 0, t0, -1
            elif k < n:
                kk = int(v["kind"][k])
                typ, tgt_t, tk = (2, float(v["ts"][k]), -1) if kk == IN_AGENT else (1, float(v["ts"][k]), kk)
            else:
                typ, tgt_t, tk = 3, float(v["obs_end"]), -1
            ridx = len(rows["b"])
            for key, val in zip(("b", "j", "type", "t0", "r", "tkind"), (b, int(j), typ, t0, tgt_t - t0, tk)):
                rows[key].append(val)
            if typ in (0, 1):
                own_ks.append(k)
                orow["row"].append(ridx)
                if typ == 1:
                    orow["cls"].append(int(v["cls"][k]))
                    orow["venue"].append(int(v["venue"][k]))
                    orow["ent"].append(int(v["ent"][k]))
                    orow["amt"].append(float(v["amt"][k]))
                    orow["tcont"].append(time_cont(v["ts"][k:k + 1], v["dt"][k:k + 1])[0])
                    orow["tbin"].append(int(v["dtbin"][k]))
                    orow["tmonth"].append(int(v["month"][k]))
                    orow["thour"].append(int(v["hour"][k]))
                    orow["tdow"].append(int(v["dow"][k]))
                    orow["led"].append(_led(v["bal"][k]))
                else:
                    orow["led"].append(_led(v["bal_end"]))
        if own_ks:
            ks = np.array(own_ks)
            st = _entry_state(v, s, ks, P)
            ent_parts.append((st, v["e_tier"][:P], v["e_sc"][:P], v["e_venue"][:P]))
            pmax = max(pmax, P)
    No = len(orow["row"])
    E = {"count": np.zeros((No, pmax), np.float32), "tin": np.zeros((No, pmax), np.float64),
         "tout": np.zeros((No, pmax), np.float64), "last": np.zeros((No, pmax), np.float64),
         "wpos": np.full((No, pmax), -1, np.int64), "tier": np.full((No, pmax), 3, np.int64),
         "sc": np.zeros((No, pmax), bool), "venue": np.full((No, pmax), -1, np.int64)}
    r0 = 0
    for (cnt, tin, tout, last, wpos), tier, sc, ven in ent_parts:
        R, P = cnt.shape
        E["count"][r0:r0 + R, :P] = cnt
        E["tin"][r0:r0 + R, :P] = tin
        E["tout"][r0:r0 + R, :P] = tout
        E["last"][r0:r0 + R, :P] = last
        E["wpos"][r0:r0 + R, :P] = wpos
        E["tier"][r0:r0 + R, :P] = tier
        E["sc"][r0:r0 + R, :P] = sc
        E["venue"][r0:r0 + R, :P] = ven
        r0 += R
    out = {f"tok_{k}": x for k, x in tok.items()}
    from lpw.p1.plan import N_PLAN_A, N_PLAN_N
    plan_n = np.array([v.get("plan_n", N_PLAN_N) for v in views], np.int64)  # absent = unknown
    plan_a = np.array([v.get("plan_a", N_PLAN_A) for v in views], np.int64)
    out.update(tok_cont=cont, role=role, cohort=cohort, prof=prof, ocont=ocont, plan_n=plan_n, plan_a=plan_a)
    out.update({f"row_{k}": np.array(x, np.float64 if k in ("t0", "r") else np.int64) for k, x in rows.items()})
    out["o_row"] = np.array(orow["row"], np.int64)
    out["o_led"] = np.array(orow["led"], np.float32).reshape(-1, 2)
    if not sample:
        out.update(o_cls=np.array(orow["cls"], np.int64), o_venue=np.array(orow["venue"], np.int64),
                   o_ent=np.array(orow["ent"], np.int64), o_amt=np.array(orow["amt"], np.float32),
                   o_tcont=np.array(orow["tcont"], np.float32).reshape(-1, N_TIME_CONT),
                   o_tbin=np.array(orow["tbin"], np.int64), o_tmonth=np.array(orow["tmonth"], np.int64),
                   o_thour=np.array(orow["thour"], np.int64), o_tdow=np.array(orow["tdow"], np.int64))
    out.update({f"e_{k}": x for k, x in E.items()})
    return out


def to_torch(batch: dict, dev) -> dict:
    return {k: torch.from_numpy(v).to(dev, non_blocking=True) for k, v in batch.items()}


def collate_specs(views: Views, specs) -> dict:
    st = views.store
    return collate([views(a) for a, _, _ in specs], [(s, js) for _, s, js in specs], st.n_venues, st.w_start)
