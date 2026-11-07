"""Policy model and baselines (SPEC §6.3, §7.1).

  lpw  transformer, product time law with weekly weight, REPEAT/VENUE/NEW counterparty head, 8-Gaussian amount
  a1   lpw with w ≡ 1 (Δt only)
  a3   lpw with a single-Gaussian amount
  b0   per-role marginals with Markov-1 on kind (kind head sees role and previous token kind; all other heads
       see role and target kind only; uniform pointer)
  b1   GRU encoder with the lpw heads
  b2   Banksformer-adapted: Δt / hour / dow heads trained independently by cross-entropy on uncensored prefixes,
       no counterparty head, single-Gaussian amount. Scored through the same normalized law as everyone else.

Row types: 1 owned target, 2 censored by an IN_agent delivery, 3 censored at observation end, 0 sampling (no target).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpw.p1.batch import CTX, N_TIME_CONT, N_TOK_CONT
from lpw.p1.store import DUST_LOG10, OUT, PAD, REPEAT, SUPPLY_ETH, SUPPLY_LOG10, VENUE
from lpw.p1.timelaw import EDGES, TimeLaw

N_PHI = 18
DK = 64


@dataclass
class Cfg:
    name: str = "lpw"
    enc: str = "tf"        # tf | gru | marg
    week: bool = True      # False = A1
    k_mix: int = 8         # 1 = A3
    b2: bool = False
    d: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 512
    dropout: float = 0.1
    pos: str = "abs"       # abs = learned absolute window positions | alibi = per-head linear recency bias on event-event attention
    indep_init: bool = False  # re-initialise each transformer layer independently (nn.TransformerEncoder deep-copies one layer)
    plan: bool = False     # per-agent plan latent through the PROFILE token (lpw.p1.plan)
    progress: bool = False  # plan progress state -> learned residual on the dormant-atom logit (needs plan)
    pace: bool = False     # history-free pace reference + mean shrinkage of the time law (lpw.p1.pace; post-hoc only)
    anchor: bool = False   # plan-conditioned residual adapters on the time and amount heads (lpw.p1.anchor)
    shallow: str = ""      # shallow-row correction offsets, a key of lpw.p1.shallow.VARIANTS ("" = none)


CONFIGS = {
    "lpw": Cfg("lpw"),
    "a1": Cfg("a1", week=False),
    "a3": Cfg("a3", k_mix=1),
    "b0": Cfg("b0", enc="marg"),
    "b1": Cfg("b1", enc="gru"),
    "b2": Cfg("b2", k_mix=1, b2=True),
    "lpw_plan": Cfg("lpw_plan", plan=True),
    "lpw_plan_prog": Cfg("lpw_plan_prog", plan=True, progress=True),
    "lpw_plan_prog_pace": Cfg("lpw_plan_prog_pace", plan=True, progress=True, pace=True),
    "lpw_plan_prog_anchor": Cfg("lpw_plan_prog_anchor", plan=True, progress=True, anchor=True),
    # shallow-row correction on the plan-progress base (runs/p1_diag/after_v2; lpw.p1.shallow): kind / +cp / +time
    "lpw_plan_prog_shallow_k": Cfg("lpw_plan_prog_shallow_k", plan=True, progress=True, shallow="k"),
    "lpw_plan_prog_shallow_kc": Cfg("lpw_plan_prog_shallow_kc", plan=True, progress=True, shallow="kc"),
    "lpw_plan_prog_shallow_kct": Cfg("lpw_plan_prog_shallow_kct", plan=True, progress=True, shallow="kct"),
    # encoder screen (reports/p1_encoder_screen.md)
    "tf_ref": Cfg("tf_ref", indep_init=True),
    "tf_alibi": Cfg("tf_alibi", pos="alibi", indep_init=True),
    "tf_small": Cfg("tf_small", layers=2, ff=128, indep_init=True),
}


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def phi(count, rec, tin, tout, tier, in_ctx):
    """Partner-entry features (…, N_PHI): log count, log recency (+ RBF expansion), log totals, tier, in-context flag."""
    lc = torch.log1p(count)
    lr = torch.log1p(torch.clamp(rec, min=0).float())
    centers = torch.linspace(0, 17.5, 10, device=count.device)
    rbf = torch.exp(-(((lr[..., None] - centers) / 1.75) ** 2))
    t1 = F.one_hot(tier.clamp(0, 3), 4)[..., :3].float()
    tin, tout = tin.double().clamp(max=SUPPLY_ETH), tout.double().clamp(max=SUPPLY_ETH)
    return torch.cat([lc[..., None] / 3, lr[..., None] / 17, rbf, (torch.log1p(tin) / 5).float()[..., None],
                      (torch.log1p(tout) / 5).float()[..., None], t1, in_ctx[..., None].float()], -1)


def log_interval_mass(a, b):
    """log(Φ(b) − Φ(a)) for standardized bounds a < b that are not arbitrarily close (amount intervals are at least
    ~1.6σ wide). An interval right of the mean is reflected to [−b, −a] first, so the single expression
    log Φ(hi) + log1p(−exp(log Φ(lo) − log Φ(hi))) never cancels and no unselected branch can poison gradients."""
    ln = torch.special.log_ndtr
    flip = a > 0
    lo, hi = torch.where(flip, -b, a), torch.where(flip, -a, b)
    return ln(hi) + torch.log1p(-torch.exp(torch.clamp(ln(lo) - ln(hi), max=0.0)))


def component_mass(mu, sig):
    """log of each Gaussian component's mass inside [dust floor, total supply] (float64)."""
    mu, sig = mu.double(), sig.double()
    return log_interval_mass((DUST_LOG10 - mu) / sig, (SUPPLY_LOG10 - mu) / sig)


def trunc_mog_logpdf(params, a, k):
    """Mixture of Gaussians on log10 ETH truncated to [dust floor, total supply], renormalized."""
    logit, mu, ls = params.split(k, -1)
    sig = torch.exp(ls.clamp(-4, 2))
    lw = torch.log_softmax(logit, -1)
    lp = lw - 0.5 * ((a[:, None] - mu) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)
    tail = torch.logsumexp(lw.double() + component_mass(mu, sig), -1)
    return torch.logsumexp(lp, -1) - tail.float()


def gauss_logpdf(params, a):
    _, mu, ls = params.split(1, -1)
    sig = torch.exp(ls.clamp(-4, 2))[:, 0]
    return -0.5 * ((a - mu[:, 0]) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)


def trunc_mog_mean(params, k):
    """mean of the truncated mixture on [dust floor, total supply] (float64): each component's truncated-normal mean
    mu + sigma (phi(alpha) - phi(beta)) / Z, weighted by w Z / sum w Z; ratios in log space, so deep-tail components
    stay finite (their mean tends to the nearer bound)."""
    logit, mu, ls = params.double().split(k, -1)
    sig = torch.exp(ls.clamp(-4, 2))
    al, be = (DUST_LOG10 - mu) / sig, (SUPPLY_LOG10 - mu) / sig
    lz = log_interval_mass(al, be)
    lphi = lambda z: -0.5 * z * z - 0.5 * math.log(2 * math.pi)
    m = mu + sig * (torch.exp(lphi(al) - lz) - torch.exp(lphi(be) - lz))
    lw = torch.log_softmax(logit, -1) + lz
    return (torch.softmax(lw, -1) * m.clamp(DUST_LOG10, SUPPLY_LOG10)).sum(-1)


def masked_gate_logp(g, ent_any, ven_any):
    """log-probs over [REPEAT, VENUE, NEW] from raw gate logits (n, 3): REPEAT needs a valid partner entry, VENUE a
    valid unseen venue."""
    neg = torch.full_like(g[:, 0], -1e9)
    g = torch.stack([torch.where(ent_any, g[:, 0], neg), torch.where(ven_any, g[:, 1], neg), g[:, 2]], -1)
    return torch.log_softmax(g, -1)


def mog_mean(params, k):
    logit, mu, _ = params.split(k, -1)
    return (torch.softmax(logit, -1) * mu).sum(-1).clamp(DUST_LOG10, SUPPLY_LOG10)


class Model(nn.Module):
    def __init__(self, cfg: Cfg, n_venues: int, n_prof: int, venue_sc: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        V = n_venues
        self.V = V
        self.register_buffer("venue_sc", venue_sc.bool())
        self.e_kind = nn.Embedding(5, d)
        self.e_cls = nn.Embedding(4, d)
        self.e_venue = nn.Embedding(V + 1, d)
        self.e_tier = nn.Embedding(4, d)
        self.e_dtbin = nn.Embedding(65, d)
        self.e_month = nn.Embedding(13, d)
        self.tok_cont = nn.Linear(N_TOK_CONT, d)
        self.e_role = nn.Embedding(2, d)
        self.e_cohort = nn.Embedding(2, d)
        self.prof_mlp = mlp(n_prof + 6, d, d)
        if cfg.pos == "abs":
            self.pos = nn.Embedding(CTX + 1, d)
        if cfg.enc == "tf":
            layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            if cfg.indep_init:
                for lay in self.enc.layers:
                    lay.self_attn.out_proj.reset_parameters()  # _reset_parameters() only zeroes its bias
                    lay.self_attn._reset_parameters()
                    for mod in (lay.linear1, lay.linear2, lay.norm1, lay.norm2):
                        mod.reset_parameters()
            if cfg.pos == "alibi":
                h = cfg.heads
                self.register_buffer("alibi_slopes", torch.tensor([2.0 ** (-8.0 * (i + 1) / h) for i in range(h)]), persistent=False)
        elif cfg.enc == "gru":
            self.enc = nn.GRU(d, d, 2, batch_first=True, dropout=cfg.dropout)
            self.ln = nn.LayerNorm(d)
        else:
            self.m_kind = nn.Embedding(2 * 5, d)
            self.m_role = nn.Embedding(2, d)
        if cfg.plan:
            from lpw.p1.plan import N_PLAN_A, N_PLAN_N
            self.e_plan_n = nn.Embedding(N_PLAN_N + 1, d)  # last index = unknown
            self.e_plan_a = nn.Embedding(N_PLAN_A + 1, d)
            nn.init.zeros_(self.e_plan_n.weight)  # starts as a no-op on a fine-tuned checkpoint
            nn.init.zeros_(self.e_plan_a.weight)
        if cfg.progress:
            # Phase 1 diagnosis (runs/p1_diag/drift_results.md): with s through PROFILE only, the plan barely moves
            # stopping (switching a real count plan "exactly 1" -> "2" changed depth-1 P(stop) by 0.018). This adds a
            # learned shift to the dormant ("no further owned event") logit of each kind, per progress state.
            assert cfg.plan, "progress needs the plan latent"
            from lpw.p1.plan import N_PROG
            self.prog_dorm = nn.Parameter(torch.zeros(N_PROG, 2))
        if cfg.pace:
            assert cfg.progress, "pace shrinkage is built on the progress model"
            from lpw.p1.pace import PaceRef
            self.pace = PaceRef(n_prof)
        if cfg.anchor:
            assert cfg.plan and not cfg.pace and not cfg.b2, "anchor adapters sit on the plan model's heads"
            from lpw.p1.anchor import Anchor
            self.anchor = Anchor(d, cfg.k_mix)
        if cfg.shallow:
            assert cfg.plan and cfg.progress and not cfg.pace and not cfg.anchor and not cfg.b2 and cfg.enc == "tf"
            from lpw.p1.shallow import VARIANTS, Shallow
            self.shallow = Shallow(VARIANTS[cfg.shallow])
        self._capture = None  # dict -> head inputs / base outputs are recorded here (lpw.p1.anchor caching)
        # plan latent at inference: False = unknown (NLL evaluation; no future information), True = use B["plan_*"]
        self.plan_inference = False
        self.kind_head = mlp(d, d, 2)
        self.e_tkind = nn.Embedding(2, d)
        self.time_head = mlp(2 * d, d, 64 + 24 + 7)
        self.e_tbin = nn.Embedding(64, d)
        self.e_tmonth = nn.Embedding(12, d)
        self.t_cont = nn.Linear(N_TIME_CONT, d)
        k = cfg.k_mix
        if cfg.b2:
            self.amt_head = mlp(3 * d, d, 3 * k)
        else:
            self.cp_ctx = mlp(3 * d, d, d)
            self.gate = nn.Linear(d, 3)
            self.ven_q = nn.Linear(d, d)
            self.ven_out = nn.Embedding(V, d)
            self.ven_bias = nn.Parameter(torch.zeros(V))
            self.ptr_q = nn.Linear(d, DK)
            self.ptr_k = nn.Linear(d, DK)
            self.ptr_oov = nn.Parameter(torch.zeros(DK))
            self.ptr_f = nn.Linear(d, N_PHI)
            self.ent_feat = nn.Linear(N_PHI, d)
            self.led = nn.Linear(2, d)
            self.amt_head = mlp(4 * d, d, 3 * k)

    # ---- encoder -------------------------------------------------------------------------------------------
    def encode(self, B):
        if self.cfg.enc == "marg":
            hk = self.m_kind(B["role"][:, None] * 5 + B["tok_kind"])
            ho = self.m_role(B["role"])[:, None].expand_as(hk)
            return hk, ho
        x = (self.e_kind(B["tok_kind"]) + self.e_cls(B["tok_cls"]) + self.e_venue(B["tok_venue"]) + self.e_tier(B["tok_tier"])
             + self.e_dtbin(B["tok_dtbin"]) + self.e_month(B["tok_month"]) + self.tok_cont(B["tok_cont"]))
        p = self.e_role(B["role"]) + self.e_cohort(B["cohort"]) + self.prof_mlp(torch.cat([B["prof"], B["ocont"]], -1))
        if self.cfg.plan:
            from lpw.p1.plan import N_PLAN_A, N_PLAN_N, P_DROP
            pn, pa = B["plan_n"], B["plan_a"]
            if self.training:  # drop s to "unknown" at random so the same network also models s-unknown
                drop = torch.rand(pn.shape, device=pn.device) < P_DROP
                pn, pa = torch.where(drop, N_PLAN_N, pn), torch.where(drop, N_PLAN_A, pa)
            elif not self.plan_inference:
                pn, pa = torch.full_like(pn, N_PLAN_N), torch.full_like(pa, N_PLAN_A)
            p = p + self.e_plan_n(pn) + self.e_plan_a(pa)
            B["_plan_n_eff"], B["_plan_a_eff"] = pn, pa  # the effective plan (same dropout draw) for row_progress / pace
        x = torch.cat([p[:, None], x[:, 1:]], 1)
        L = x.shape[1]
        if self.cfg.pos == "alibi" and self.cfg.enc == "tf":
            h = self.ln(self.alibi_enc(x))
        elif self.cfg.enc == "tf":
            x = x + self.pos(torch.arange(L, device=x.device))[None]
            mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
            h = self.ln(self.enc(x, mask=mask, is_causal=True))
        else:
            x = x + self.pos(torch.arange(L, device=x.device))[None]
            h = self.ln(self.enc(x)[0])
        return h, h

    def alibi_enc(self, x):
        """the layers' pre-norm slow path, called directly: nn.TransformerEncoder's fused eval / no-grad fast path
        treats a float mask as boolean, which would turn the recency penalties into hard exclusions."""
        mask = self.alibi_mask(x.shape[0], x.shape[1], x.device, x.dtype)
        for lay in self.enc.layers:
            x = x + lay._sa_block(lay.norm1(x), mask, None, is_causal=False)
            x = x + lay._ff_block(lay.norm2(x))
        return x

    def alibi_mask(self, B, L, dev, dtype):
        """(B * heads, L, L) additive mask: causal, -slope_h x (i - j) between event tokens, no penalty on attention to
        the PROFILE token (position 0), so profile / plan conditioning stays globally accessible. No position
        embeddings are added in this mode."""
        i = torch.arange(L, device=dev)
        dist = (i[:, None] - i[None, :]).to(dtype)
        bias = -self.alibi_slopes.to(dtype)[:, None, None] * dist[None]
        bias[:, :, 0] = 0.0
        bias = bias.masked_fill((i[None, :] > i[:, None])[None], float("-inf"))
        return bias.repeat(B, 1, 1)

    # ---- heads ---------------------------------------------------------------------------------------------
    def row_progress(self, B):
        """Plan progress state of every row (plan.progress_state), from the plan encode() used for this batch."""
        from lpw.p1.plan import progress_state
        return progress_state(B["_plan_n_eff"][B["row_b"]], B["row_nown"])

    def row_extras(self, B) -> dict:
        """Row inputs kind_time needs beyond the hidden states: prog (progress models), pace (pace models)."""
        out = {}
        if self.cfg.progress:
            out["prog"] = self.row_progress(B)
        if self.cfg.pace:
            out["pace"] = self.pace.features(B)
        if self.cfg.anchor:
            out["anchor"] = (B["_plan_n_eff"][B["row_b"]], B["_plan_a_eff"][B["row_b"]])
        if self.cfg.shallow:
            out["shallow"] = self.shallow.row_offsets(B)
        return out

    def kind_time(self, hk, ho, t0, prog=None, pace=None, anchor=None, shallow=None):
        """Kind log-probs (N, 2) and the time law for both kinds (rows 2i + κ). prog: row_progress, required for
        progress models; pace: pace.features (x, eligible), required for pace models; anchor: the row plans
        (plan_n, plan_a), required for anchor models; shallow: the rows' (offsets (N, shallow.N_OUT), active (N,)),
        required for shallow models (see row_extras)."""
        logpk = torch.log_softmax(self.kind_head(hk), -1)
        n = hk.shape[0]
        inp = torch.cat([ho[:, None].expand(n, 2, -1), self.e_tkind.weight[None].expand(n, 2, -1)], -1)
        out = self.time_head(inp).reshape(2 * n, -1)
        if self._capture is not None:
            self._capture.update(logpk=logpk.detach(), ho=ho.detach(), out=out.detach())
        if self.cfg.anchor:
            assert anchor is not None, "anchor model: pass **model.row_extras(B)"
            out = self.anchor.apply_time(out, ho, *anchor)
        if self.cfg.shallow:
            assert shallow is not None, "shallow model: pass **model.row_extras(B)"
            from lpw.p1.shallow import apply_kind, apply_time
            logpk, out = apply_kind(logpk, *shallow), apply_time(out, shallow[0])
        return self.time_law(logpk, out, t0, prog, pace)

    def time_law(self, logpk, out, t0, prog=None, pace=None):
        """the time law from the time-head output (2N, 95): progress residual on the dormant logit, normalization,
        pace shrinkage. Returns (logpk, law, out)."""
        n = logpk.shape[0]
        if self.cfg.progress:
            assert prog is not None, "progress model: pass prog=model.row_progress(B)"
            res = self.prog_dorm[prog].reshape(2 * n).to(out.dtype)
            out = torch.cat([out[:, :63], out[:, 63:64] + res[:, None], out[:, 64:]], 1)
        lp = torch.log_softmax(out[:, :64], -1)
        lh = torch.log_softmax(out[:, 64:88], -1) if self.cfg.week else None
        ld = torch.log_softmax(out[:, 88:95], -1) if self.cfg.week else None
        t0r = t0.repeat_interleave(2)
        if self.cfg.pace:
            assert pace is not None, "pace model: pass **model.row_extras(B)"
            lp = self.pace.adapt(lp, lh, ld, t0, logpk, pace)
        law = TimeLaw(lp, lh, ld, t0r)
        return logpk, law, out

    def time_emb(self, tbin, tmonth, tcont):
        return self.e_tbin(tbin) + self.e_tmonth(tmonth) + self.t_cont(tcont)

    def cp_context(self, B, rows_b, ho, ko, tt):
        if self.cfg.enc == "marg":
            return self.cp_ctx(torch.cat([self.m_role(B["role"][rows_b]), self.e_tkind(ko), torch.zeros_like(tt)], -1))
        return self.cp_ctx(torch.cat([ho, self.e_tkind(ko), tt], -1))

    def masks(self, B, ko):
        """Valid REPEAT entries (No, P) and valid unseen venues (No, V) for the owned rows (e_* order), target kind ko."""
        cnt = B["e_count"]
        out_k = (ko == OUT)[:, None]
        ent_ok = (cnt > 0) & (out_k | ((B["e_tier"] != 2) & B["e_sc"]))
        ev = B["e_venue"]
        seen = torch.zeros(len(ko), self.V + 1, dtype=torch.bool, device=cnt.device)
        seen.scatter_(1, torch.where((cnt > 0) & (ev >= 0), ev, self.V), True)
        ven_ok = ~seen[:, :self.V] & (out_k | self.venue_sc[None])
        return ent_ok, ven_ok

    def gate_logp(self, ctx, ent_ok, ven_ok, shallow=None, ko=None):
        """log-probs over [REPEAT, VENUE, NEW] with the legality masks; shallow: the owned rows' shallow offsets (with
        their next kinds ko), for shallow models."""
        g = self.gate(ctx)
        if shallow is not None:
            from lpw.p1.shallow import apply_gate
            g = apply_gate(g, shallow, ko)
        return masked_gate_logp(g, ent_ok.any(-1), ven_ok.any(-1))

    def pointer_logp(self, B, H, oi, rb, rj, ctx, t_tgt, ent_ok):
        """Log-probs over partner entries for owned rows oi (indices into e_*; window rb, position rj), shape (n, P)."""
        rec = t_tgt[:, None] - B["e_last"][oi]
        ph = phi(B["e_count"][oi], rec, B["e_tin"][oi], B["e_tout"][oi], B["e_tier"][oi], B["e_wpos"][oi] >= 0)
        if self.cfg.enc == "marg":
            logits = torch.zeros(ph.shape[:2], device=ph.device)
        else:
            q = self.ptr_q(ctx)
            kh = self.ptr_k(H)
            Bn, L, _ = kh.shape
            qf = torch.zeros(Bn, L, DK, device=q.device, dtype=q.dtype)
            qf[rb, rj] = q
            s = torch.bmm(qf, kh.transpose(1, 2))[rb, rj]  # (n, L)
            wpos = B["e_wpos"][oi]
            sh = s.gather(1, wpos.clamp(min=0))
            soov = (q * self.ptr_oov).sum(-1, keepdim=True)
            logits = torch.where(wpos >= 0, sh, soov) / math.sqrt(DK) + (self.ptr_f(ctx)[:, None] * ph).sum(-1)
        logits = logits.masked_fill(~ent_ok, -1e9)
        return torch.log_softmax(logits, -1), ph

    def venue_logp(self, ctx, ven_ok):
        v = self.ven_q(ctx) @ self.ven_out.weight.T + self.ven_bias
        return torch.log_softmax(v.masked_fill(~ven_ok, -1e9), -1)

    def amount_params(self, B, rows_b, ctx, cls, cpvec, led, ho=None, ko=None, tt=None):
        if self.cfg.b2:
            return self.amt_head(torch.cat([ho, self.e_tkind(ko), tt], -1))
        if self.cfg.enc == "marg":
            cpvec, led_e = torch.zeros_like(ctx), torch.zeros_like(ctx)
        else:
            led_e = self.led(led)
        x = torch.cat([ctx, self.e_cls(cls), cpvec, led_e], -1)
        ap = self.amt_head(x)
        if self._capture is not None:
            self._capture.update(amt_x=x.detach(), amt_ap=ap.detach())
        if self.cfg.anchor:
            ap = self.anchor.apply_amt(ap, x, B["_plan_n_eff"][rows_b], B["_plan_a_eff"][rows_b])
        if self._capture is not None:
            self._capture["amt_ap_final"] = ap.detach()
            self._capture["amt_ap_live"] = ap  # with its graph (lpw.p1.drift_ft consistency term)
        return ap

    # ---- likelihood ----------------------------------------------------------------------------------------
    def terms(self, B, metrics=False) -> dict:
        with torch.autocast("cuda", dtype=torch.float16, enabled=B["tok_kind"].is_cuda):
            Hk, Ho = self.encode(B)
        Hk, Ho = Hk.float(), Ho.float()
        with torch.autocast("cuda", enabled=False):
            return self._terms(B, Hk, Ho, metrics)

    def _terms(self, B, Hk, Ho, metrics):
        rb, rj, typ, t0, r = B["row_b"], B["row_j"], B["row_type"], B["row_t0"], B["row_r"]
        N = len(rb)
        hk, ho = Hk[rb, rj], Ho[rb, rj]
        ex = self.row_extras(B)
        logpk, law, tout = self.kind_time(hk, ho, t0, **ex)
        if self._capture is not None:
            self._capture.update(logpk_live=logpk, law_live=law)  # with their graph (lpw.p1.drift_ft)
        own = typ == 1
        cen = (typ == 2) | (typ == 3)
        tk = B["row_tkind"].clamp(min=0)
        r2 = r.repeat_interleave(2)
        logf = law.log_density(r2).view(N, 2)
        logS = law.log_survival(r2).view(N, 2)
        z = torch.zeros(N, device=rb.device, dtype=torch.float64)
        res = {
            "kind": torch.where(own, -logpk.gather(1, tk[:, None])[:, 0].double(), z),
            "time": torch.where(own, -logf.gather(1, tk[:, None])[:, 0], z),
            "cens": torch.where(cen, -torch.logsumexp(logpk.double() + logS, -1), z),
            "own": own, "cen": cen,
        }
        oi = B["o_row"]
        ko = tk[oi]
        tt = self.time_emb(B["o_tbin"], B["o_tmonth"], B["o_tcont"])
        cls = B["o_cls"]
        amt = B["o_amt"]
        k = self.cfg.k_mix
        cp = torch.zeros(len(oi), device=rb.device)
        if self.cfg.b2:
            ap = self.amount_params(B, rb[oi], None, cls, None, None, ho[oi], ko, tt)
            amt_ll = trunc_mog_logpdf(ap, amt, 1)
            # independent cross-entropy training objective (uncensored prefixes only)
            lt = tout.view(N, 2, -1)[oi, ko]
            tbin, hour, dow = B["o_tbin"], B["o_thour"], B["o_tdow"]
            res["b2_train"] = (F.cross_entropy(logpk[oi], ko, reduction="none") + F.cross_entropy(lt[:, :64], tbin, reduction="none")
                               + F.cross_entropy(lt[:, 64:88], hour, reduction="none") + F.cross_entropy(lt[:, 88:95], dow, reduction="none")
                               - gauss_logpdf(ap, amt))
        else:
            ctx = self.cp_context(B, rb[oi], ho[oi], ko, tt)
            ent_ok, ven_ok = self.masks(B, ko)
            lg = self.gate_logp(ctx, ent_ok, ven_ok, ex["shallow"][0][oi] if "shallow" in ex else None, ko)
            cp = -lg.gather(1, cls[:, None])[:, 0]
            cpvec = torch.zeros_like(ctx)
            rep = cls == REPEAT
            ven = cls == VENUE
            hits = None
            if rep.any():
                ri = torch.nonzero(rep)[:, 0]
                lptr, ph = self.pointer_logp(B, Ho, ri, rb[oi][ri], rj[oi][ri], ctx[ri], (t0 + r)[oi][ri], ent_ok[ri])
                tgt = B["o_ent"][ri]
                assert torch.all(ent_ok[ri].gather(1, tgt[:, None])), "REPEAT target masked"
                cp = cp.index_add(0, ri, -lptr.gather(1, tgt[:, None])[:, 0])
                cpvec = cpvec.index_copy(0, ri, self.ent_feat(ph[torch.arange(len(ri)), tgt]))
                if metrics:
                    # tie-aware expected Hits@k: a target tied with t others gets credit (k - better) / (t + 1)
                    lt_ = lptr.gather(1, tgt[:, None])
                    better = (lptr > lt_).sum(-1).double()
                    ties = (lptr == lt_).sum(-1).double()
                    hits = {kk: ((kk - better) / ties).clamp(0, 1) for kk in (1, 5)}
            if ven.any():
                vi = torch.nonzero(ven)[:, 0]
                lv = self.venue_logp(ctx[vi], ven_ok[vi])
                vt = B["o_venue"][vi]
                assert torch.all(ven_ok[vi].gather(1, vt[:, None])), "VENUE target masked"
                cp = cp.index_add(0, vi, -lv.gather(1, vt[:, None])[:, 0])
                cpvec = cpvec.index_copy(0, vi, self.e_venue(vt))
            ap = self.amount_params(B, rb[oi], ctx, cls, cpvec, B["o_led"])
            amt_ll = trunc_mog_logpdf(ap, amt, k)
        res["cp"] = cp
        res["amt"] = -amt_ll
        res["o_row"] = oi
        if metrics:
            res["kind_ok"] = (logpk.argmax(-1) == tk)[oi]
            bp = law.bin_probs().view(N, 2, 64)[oi, ko]
            res["bin_ok"] = bp.argmax(-1) == B["o_tbin"]
            e = torch.as_tensor(EDGES, device=rb.device, dtype=torch.float64)
            mids = torch.cat([torch.zeros(1, device=rb.device, dtype=torch.float64), torch.log1p(torch.sqrt(e[:-1] * e[1:]))])
            pf = bp[:, :63]
            res["logdt_err"] = ((pf * mids).sum(-1) / pf.sum(-1).clamp(min=1e-30) - torch.log1p(r[oi].double())).abs()
            res["amt_err"] = (mog_mean(ap, k) - amt).abs()
            if not self.cfg.b2:
                res["cls_ok"] = lg.argmax(-1) == cls
                res["rep_hits"] = hits
                res["rep_mask"] = rep
                res["amt_marg"] = -self.amount_marginal_ll(B, Ho, rb[oi], rj[oi], ctx, (t0 + r)[oi], ent_ok, ven_ok, lg, amt)
        return res

    @torch.no_grad()
    def amount_marginal_ll(self, B, H, rb, rj, ctx, t_tgt, ent_ok, ven_ok, lg, amt, cand_budget=4_000_000):
        """log p(amount | h, kind, time) = log Σ_cp p(cp | h, kind, time) p(amount | ..., cp), summed over every
        candidate counterparty (each valid partner entry, each valid unseen venue, NEW). Used for the B2 comparison,
        whose amount head never sees a counterparty. The amount head's first layer is linear in its concatenated
        input, so per-candidate terms are added to a shared per-row part before the nonlinearity."""
        k, d = self.cfg.k_mix, self.cfg.d
        marg = self.cfg.enc == "marg"
        lin1, act, lin2 = self.amt_head[0], self.amt_head[1], self.amt_head[2]
        Wc, Wk, Wp, Wl = lin1.weight[:, :d], lin1.weight[:, d:2 * d], lin1.weight[:, 2 * d:3 * d], lin1.weight[:, 3 * d:]
        base = ctx @ Wc.T + lin1.bias
        if not marg:
            base = base + self.led(B["o_led"]) @ Wl.T
        kterm = self.e_cls.weight[:3] @ Wk.T  # (3, d): REPEAT, VENUE, NEW

        def ll(pre, a):  # pre (n, c, d) → (n, c)
            n, c, _ = pre.shape
            prm = lin2(act(pre)).reshape(n * c, 3 * k)
            return trunc_mog_logpdf(prm, a[:, None].expand(n, c).reshape(-1), k).view(n, c)

        No = len(amt)
        ll_new = ll((base + kterm[2])[:, None], amt)[:, 0]
        lv = self.venue_logp(ctx, ven_ok)
        vterm = torch.zeros(self.V, d, device=ctx.device) if marg else self.e_venue.weight[: self.V] @ Wp.T
        ll_ven = torch.empty(No, device=ctx.device)
        step = max(1, cand_budget // max(self.V, 1) // 4)
        for i in range(0, No, step):
            sl = slice(i, i + step)
            ll_ven[sl] = torch.logsumexp(lv[sl] + ll((base[sl] + kterm[1])[:, None] + vterm[None], amt[sl]), -1)
        ll_rep = torch.full((No,), -1e9, device=ctx.device)
        P = ent_ok.shape[1]
        if P > 0:
            ri = torch.arange(No, device=ctx.device)
            step = max(1, cand_budget // P // 4)
            for i in range(0, No, step):
                sl = ri[i:i + step]
                lptr, ph = self.pointer_logp(B, H, sl, rb[sl], rj[sl], ctx[sl], t_tgt[sl], ent_ok[sl])
                pterm = torch.zeros(len(sl), P, d, device=ctx.device) if marg else self.ent_feat(ph) @ Wp.T
                ll_rep[sl] = torch.logsumexp(lptr + ll((base[sl] + kterm[0])[:, None] + pterm, amt[sl]), -1)
        return torch.logsumexp(torch.stack([lg[:, 0] + ll_rep, lg[:, 1] + ll_ven, lg[:, 2] + ll_new], -1), -1)

    def loss(self, B) -> tuple[torch.Tensor, dict]:
        t = self.terms(B)
        n = len(B["row_b"])
        if self.cfg.b2:
            return t["b2_train"].sum() / max(len(B["o_row"]), 1), t
        if "agent_w" in B:  # agent-balanced weighting (batch.agent_weights); still divided by the row count
            w = B["agent_w"].double()[B["row_b"]]
            tot = (w * (t["kind"] + t["time"] + t["cens"])).sum() + (w[t["o_row"]] * (t["cp"] + t["amt"])).sum()
        else:
            tot = t["kind"].sum() + t["time"].sum() + t["cens"].sum() + t["cp"].sum() + t["amt"].sum()
        return (tot / n).float(), t
