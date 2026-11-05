"""Normalized product time law (SPEC §6.3), used identically for training, censoring and sampling.

Bins: 0 = same second (atom at t0), 1..62 log-spaced over [1 s, 365 d), 63 = dormant (atom, no time).
Weekly weight w(c) = 168 · p_hour(h(c)) · p_dow(d(c)) over the 168 hour cells of the week (mean 1), or w ≡ 1 (A1).
All arithmetic is float64: absolute offsets reach 3e7 s and the smallest bins are 0.3 s wide.
"""

import numpy as np
import torch

N_BINS = 64
YEAR = 365 * 86400
EDGES = np.geomspace(1.0, float(YEAR), 63)  # finite bin b (1..62) is [EDGES[b-1], EDGES[b])
WEEK = 604800
MONDAY0 = 345600  # 1970-01-05 00:00 UTC, a Monday; week cells start here
CELL_HOUR = np.arange(168) % 24
CELL_DOW = np.arange(168) // 24  # Monday = 0


def dt_bin(dt: np.ndarray) -> np.ndarray:
    """Bin index of integer second gaps: 0 for dt == 0, 63 for dt >= 365 d."""
    dt = np.asarray(dt)
    return np.where(dt <= 0, 0, np.searchsorted(EDGES, dt, side="right")).astype(np.int64)


def hour_dow(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts = np.floor(np.asarray(ts, dtype=np.float64)).astype(np.int64)
    return (ts % 86400) // 3600, ((ts // 86400) - 4) % 7


class TimeLaw:
    """One law per row. Inputs are log-probabilities (any float dtype) and the origin t0 (seconds; generated
    streams keep continuous times, real ones are whole seconds)."""

    def __init__(self, log_pdt: torch.Tensor, log_ph: torch.Tensor | None, log_pd: torch.Tensor | None, t0: torch.Tensor):
        dev = log_pdt.device
        f64 = torch.float64
        self.dev = dev
        self.log_pdt = log_pdt.to(f64)
        n = log_pdt.shape[0]
        if log_ph is None:
            self.w = torch.ones(n, 168, dtype=f64, device=dev)
        else:
            ch = torch.as_tensor(CELL_HOUR, device=dev)
            cd = torch.as_tensor(CELL_DOW, device=dev)
            self.w = 168.0 * torch.exp(log_ph.to(f64)[:, ch] + log_pd.to(f64)[:, cd])
        self.cum = torch.cat([torch.zeros(n, 1, dtype=f64, device=dev), torch.cumsum(self.w * 3600.0, dim=1)], dim=1)
        self.t0 = t0.to(f64)
        self.r0 = torch.remainder(self.t0 - MONDAY0, float(WEEK))  # seconds since the start of t0's week
        e = torch.as_tensor(EDGES, dtype=f64, device=dev)
        self.lo, self.hi = e[:-1], e[1:]
        self.wid = self.hi - self.lo
        pdt = torch.exp(self.log_pdt)
        integ = self.W(self.r0[:, None] + self.hi[None]) - self.W(self.r0[:, None] + self.lo[None])  # (n, 62)
        self.mass = pdt[:, 1:63] / self.wid * integ
        self.atom0 = pdt[:, 0] * self.w_at(self.r0)
        self.atom63 = pdt[:, 63]
        self.Z = self.atom0 + self.mass.sum(-1) + self.atom63
        self.logZ = torch.log(self.Z)
        # tail[:, m] = sum of finite-bin masses with index >= m (index m = bin m+1); tail[:, 62] = 0
        self.tail = torch.flip(torch.cumsum(torch.flip(self.mass, [1]), 1), [1])
        self.tail = torch.cat([self.tail, torch.zeros(n, 1, dtype=f64, device=dev)], 1)

    def W(self, x: torch.Tensor) -> torch.Tensor:
        """Integral of w from the start of t0's week to offset x (seconds, x >= 0). Shape (n, k)."""
        nweek = torch.floor(x / WEEK)
        rem = x - nweek * WEEK
        cell = torch.clamp(torch.floor(rem / 3600.0), 0, 167).long()
        return nweek * WEEK + self.cum.gather(1, cell) + (rem - cell * 3600.0) * self.w.gather(1, cell)

    def w_at(self, x: torch.Tensor) -> torch.Tensor:
        """w at week offset x (n,)."""
        cell = torch.clamp(torch.floor((x % WEEK) / 3600.0), 0, 167).long()
        return self.w.gather(1, cell[:, None])[:, 0]

    def _bin(self, dt: torch.Tensor) -> torch.Tensor:
        b = torch.bucketize(dt.to(torch.float64), torch.as_tensor(EDGES, dtype=torch.float64, device=self.dev), right=True)
        return torch.where(dt <= 0, torch.zeros_like(b), b)

    def log_density(self, dt: torch.Tensor) -> torch.Tensor:
        """log f(t0 + dt) for an observed owned event: an atom mass for bins 0 and 63, a density per second otherwise."""
        b = self._bin(dt)
        lp = self.log_pdt.gather(1, b[:, None])[:, 0]
        x = self.r0 + dt.to(torch.float64)
        fin = (b >= 1) & (b <= 62)
        wid = self.wid[torch.clamp(b - 1, 0, 61)]
        out = torch.where(b == 0, lp + torch.log(self.w_at(self.r0)), lp)
        out = torch.where(fin, lp - torch.log(wid) + torch.log(self.w_at(x)), out)
        return out - self.logZ

    def log_survival(self, r: torch.Tensor) -> torch.Tensor:
        """log P(T >= t0 + r): no owned event before t0 + r. S(0) = 1; past 365 d only the dormant atom remains."""
        b = self._bin(r)
        bf = torch.clamp(b, 1, 62)
        m = bf - 1
        pdt = torch.exp(self.log_pdt.gather(1, bf[:, None])[:, 0])
        hi = self.hi[m]
        x = self.r0 + r.to(torch.float64)
        partial = pdt / self.wid[m] * (self.W((self.r0 + hi)[:, None])[:, 0] - self.W(x[:, None])[:, 0])
        surv = partial + self.tail.gather(1, bf[:, None])[:, 0] + self.atom63
        out = torch.log(torch.clamp(surv, min=1e-300)) - self.logZ
        out = torch.where(b >= 63, torch.log(self.atom63) - self.logZ, out)
        # 0 < r < 1 s: only the same-second atom lies before t0 + r (bin 1 starts at 1 s)
        out = torch.where((r > 0) & (r < 1), torch.log(torch.clamp(self.Z - self.atom0, min=1e-300)) - self.logZ, out)
        return torch.where(r <= 0, torch.zeros_like(out), out)

    def bin_probs(self) -> torch.Tensor:
        """Probability of each of the 64 bins (atoms and finite-bin masses), shape (n, 64)."""
        return torch.cat([self.atom0[:, None], self.mass, self.atom63[:, None]], 1) / self.Z[:, None]

    def sample(self, gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact sample. Returns (dt float64 seconds, continuous; 0 for the same-second atom, dormant bool)."""
        p = self.bin_probs()
        b = torch.multinomial(p.to(torch.float64), 1, generator=gen)[:, 0]
        dormant = b == 63
        bf = torch.clamp(b, 1, 62)
        m = bf - 1
        start = self.r0 + self.lo[m]
        wstart = self.W(start[:, None])[:, 0]
        wend = self.W((self.r0 + self.hi[m])[:, None])[:, 0]
        u = torch.rand(b.shape[0], dtype=torch.float64, device=self.dev, generator=gen)
        v = wstart + u * (wend - wstart)
        nweek = torch.floor(v / WEEK)
        rem = v - nweek * WEEK
        c = torch.clamp(torch.searchsorted(self.cum, rem[:, None], right=True)[:, 0] - 1, 0, 167)
        wc = self.w.gather(1, c[:, None])[:, 0]
        within = torch.where(wc > 0, (rem - self.cum.gather(1, c[:, None])[:, 0]) / torch.clamp(wc, min=1e-300), 0.0)
        x = nweek * WEEK + c * 3600.0 + torch.clamp(within, 0, 3600.0)
        end = self.r0 + self.hi[m]
        x = torch.minimum(torch.maximum(x, start), torch.nextafter(end, start))
        dt = x - self.r0
        # floating error must not move a draw out of its bin
        dt = torch.minimum(torch.maximum(dt, self.lo[m]), torch.nextafter(self.hi[m], self.lo[m]))
        dt = torch.where(b == 0, torch.zeros_like(dt), dt)
        return dt, dormant
