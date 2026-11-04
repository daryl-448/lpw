"""Profile generator p(profile | role, cohort) and population sampler (SPEC §6.3, §6.7, §7.2).

Pre-existing agents: an autoregressive GRU over the 20 profile fields, each quantile-binned (≤ 32 bins, train
split only) in log space, conditioned on role. A bin decodes to its atom (when most of its train values share one
value, e.g. zeros) or to a uniform draw between the bin's train min and max; real rows are never copied.
Born-in-W agents have an empty profile. Cohort and activation week come from train-split rates per role.
"""

import numpy as np
import torch
import torch.nn as nn

from lpw.p1.store import PROFILE_FIELDS, SIGNED_FIELDS, profile_transform, profile_vector

NB = 32
COUNT_FIELDS = {f for f in PROFILE_FIELDS if f.endswith("_cnt") or f in ("days_active", "days_since_active")}


def inverse_transform(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    for i, f in enumerate(PROFILE_FIELDS):
        out[:, i] = np.sign(z[:, i]) * np.expm1(np.abs(z[:, i])) if f in SIGNED_FIELDS else np.expm1(z[:, i])
        if f in COUNT_FIELDS:
            out[:, i] = np.round(out[:, i])
    return out


class Binner:
    def __init__(self, z: np.ndarray):
        self.edges, self.lo, self.hi, self.atom, self.atom_p = [], [], [], [], []
        for i in range(z.shape[1]):
            e = np.unique(np.quantile(z[:, i], np.linspace(0, 1, NB + 1)))
            if len(e) < 2:
                e = np.array([e[0], e[0] + 1e-9])
            self.edges.append(e)
            b = self._bin(z[:, i], e)
            lo, hi, at, ap = [], [], [], []
            for k in range(len(e) - 1):
                v = z[b == k, i]
                if len(v) == 0:
                    lo.append(e[k]), hi.append(e[k + 1]), at.append(e[k]), ap.append(0.0)
                    continue
                u, c = np.unique(v, return_counts=True)
                lo.append(v.min()), hi.append(v.max()), at.append(u[c.argmax()]), ap.append(c.max() / len(v) if c.max() / len(v) > 0.5 else 0.0)
            self.lo.append(np.array(lo)), self.hi.append(np.array(hi)), self.atom.append(np.array(at)), self.atom_p.append(np.array(ap))
        self.nb = np.array([len(e) - 1 for e in self.edges])

    @staticmethod
    def _bin(x, e):
        return np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)

    def encode(self, z):
        return np.stack([self._bin(z[:, i], self.edges[i]) for i in range(z.shape[1])], 1)

    def decode(self, b, rng):
        z = np.empty(b.shape)
        for i in range(b.shape[1]):
            k = b[:, i]
            uni = self.lo[i][k] + rng.random(len(k)) * (self.hi[i][k] - self.lo[i][k])
            z[:, i] = np.where(rng.random(len(k)) < self.atom_p[i][k], self.atom[i][k], uni)
        return z


class ARNet(nn.Module):
    def __init__(self, nf, d=64):
        super().__init__()
        self.nf = nf
        self.e_role = nn.Embedding(2, d)
        self.e_bin = nn.Embedding(nf * NB + 1, d)
        self.e_field = nn.Embedding(nf, d)
        self.gru = nn.GRU(d, d, 1, batch_first=True)
        self.out = nn.Linear(2 * d, NB)
        self.drop = nn.Dropout(0.1)

    def logits(self, role, bins):
        n = len(role)
        tok = torch.cat([torch.full((n, 1), self.nf * NB, device=role.device),
                         bins[:, :-1] + torch.arange(self.nf - 1, device=role.device)[None] * NB], 1)
        x = self.e_bin(tok) + self.e_role(role)[:, None]
        h, _ = self.gru(self.drop(x))
        f = self.e_field(torch.arange(self.nf, device=role.device))[None].expand(n, -1, -1)
        return self.out(torch.cat([h, f], -1))


class ProfileGen:
    def __init__(self, store, dev="cuda", seed=0, epochs=300, patience=20, log=print):
        torch.manual_seed(seed)
        self.dev = dev
        self.store = store
        tr = (store.split == "train") & (store.cohort == 0)
        va = (store.split == "val") & (store.cohort == 0)
        ztr = profile_transform(store.prof_raw[tr])
        self.binner = Binner(ztr)
        self.mask = torch.tensor(np.arange(NB)[None] < self.binner.nb[:, None], device=dev)  # (nf, NB)
        nf = len(PROFILE_FIELDS)
        self.net = ARNet(nf).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 3e-3)
        btr = torch.as_tensor(self.binner.encode(ztr), device=dev)
        rtr = torch.as_tensor(store.role[tr], device=dev)
        bva = torch.as_tensor(self.binner.encode(profile_transform(store.prof_raw[va])), device=dev)
        rva = torch.as_tensor(store.role[va], device=dev)
        best, best_ep, best_state = float("inf"), 0, None
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(rtr), device=dev)
            for i in range(0, len(perm), 256):
                j = perm[i:i + 256]
                loss = self._nll(rtr[j], btr[j]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = self._nll(rva, bva).mean().item()
            if vl < best - 1e-4:
                best, best_ep = vl, ep
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            elif ep - best_ep >= patience:
                break
        self.net.load_state_dict(best_state)
        self.val_nll = best
        self.epochs = best_ep + 1
        # population rates from train
        self.p_born = {r: float(store.cohort[(store.split == "train") & (store.role == r)].mean()) for r in (0, 1)}
        self.week_hist = {}
        for r in (0, 1):
            sel = (store.split == "train") & (store.role == r) & (store.cohort == 1)
            w = (store.origin[sel] - store.w_start) // (7 * 86400)
            self.week_hist[r] = np.bincount(w, minlength=int((store.w_end - store.w_start) // (7 * 86400)) + 1).astype(float)
        log(f"profile generator: val NLL {best:.3f} nats/agent, {best_ep + 1} epochs")

    def _nll(self, role, bins):
        lg = self.net.logits(role, bins).masked_fill(~self.mask[None], -1e9)
        return -torch.log_softmax(lg, -1).gather(2, bins[..., None])[..., 0].sum(1)

    @torch.no_grad()
    def sample_raw(self, roles: np.ndarray, rng) -> np.ndarray:
        n = len(roles)
        role = torch.as_tensor(roles, device=self.dev)
        bins = torch.zeros(n, len(PROFILE_FIELDS), dtype=torch.long, device=self.dev)
        g = torch.Generator(device=self.dev).manual_seed(int(rng.integers(1 << 31)))
        for f in range(len(PROFILE_FIELDS)):
            lg = self.net.logits(role, bins)[:, f].masked_fill(~self.mask[f][None], -1e9)
            bins[:, f] = torch.multinomial(torch.softmax(lg, -1), 1, generator=g)[:, 0]
        return inverse_transform(self.binner.decode(bins.cpu().numpy(), rng))

    def population(self, roles: np.ndarray, rng) -> list[dict]:
        """Unconditional synthetic agents: cohort, origin, profile and initial balance for each role."""
        st = self.store
        born = rng.random(len(roles)) < np.array([self.p_born[int(r)] for r in roles])
        raw = self.sample_raw(roles, rng)
        raw[born] = 0.0
        origin = np.full(len(roles), st.w_start, np.int64)
        for i in np.flatnonzero(born):
            h = self.week_hist[int(roles[i])]
            wk = rng.choice(len(h), p=h / h.sum())
            origin[i] = min(st.w_start + wk * 7 * 86400 + int(rng.random() * 7 * 86400), st.w_end - 1)
        vec = profile_vector(raw, born.astype(np.int64), st.prof_mu, st.prof_sd)
        bi = PROFILE_FIELDS.index("balance0")
        ai = PROFILE_FIELDS.index("age_days")
        return [dict(role=int(roles[i]), cohort=int(born[i]), origin=int(origin[i]), prof=vec[i], b0=float(raw[i, bi]),
                     age_days=float(raw[i, ai])) for i in range(len(roles))]
