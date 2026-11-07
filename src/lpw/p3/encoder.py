"""Evaluation-only sequence encoder for supplementary mode coverage (reports/p3a_prereg.md §16, addendum B).

A role-blind one-layer GRU over complete observed event sequences (P3a O schema), with a 32-d L2-normalized output
trained by CoLES-style crop contrast on real train only. Everything here is fixed by the addendum: no search.
"""
import math

import numpy as np
import polars as pl
import torch
import torch.nn.functional as Fn
from torch.nn.utils.rnn import pack_padded_sequence

from lpw.p1.store import OUT

N_IN = 12            # out, in, gap, first, amt, fee, sin/cos hour, sin/cos weekday, venue, seen
HIDDEN, DIM = 64, 32
TAU, CROP, DROPOUT = 0.1, 0.8, 0.1
BATCH, EPOCHS, LR, WD, SEED = 64, 20, 1e-3, 1e-4, 0
SCALARS = (2, 4, 5)  # gap, amt, fee columns: the standardized log scalars


def raw_sequences(ag: pl.DataFrame, ev: pl.DataFrame) -> list[np.ndarray]:
    """One (n_i, N_IN) float32 array per agent of ag (same order), before standardization. Agents without events
    get a (0, N_IN) array."""
    e = ev.sort("agent", "ts", maintain_order=True).with_columns(
        gap=(pl.col("ts") - pl.col("ts").shift(1).over("agent")).cast(pl.Float64),
        seen=pl.col("cp").cum_count().over(["agent", "cp"]) > 1)
    out = (e["kind"] == OUT).to_numpy().astype(np.float32)
    gap = e["gap"].to_numpy()
    first = np.isnan(gap)
    ts = e["ts"].to_numpy()
    h, d = (ts % 86400) // 3600, ((ts // 86400) - 4) % 7
    X = np.stack([out, 1 - out, np.log1p(np.where(first, 0.0, gap)), first.astype(np.float64), e["amt"].to_numpy(),
                  np.log1p(e["fee"].to_numpy() / 1e-9), np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                  np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7), (e["venue"].to_numpy() >= 0).astype(np.float64),
                  e["seen"].to_numpy().astype(np.float64)], 1).astype(np.float32)
    counts = np.bincount(e["agent"].to_numpy(), minlength=ag.height)
    off = np.concatenate([[0], np.cumsum(counts)])
    assert (e["agent"].to_numpy() == np.repeat(np.arange(ag.height), counts)).all(), "agents must be 0..n-1, sorted"
    return [X[off[i]:off[i + 1]] for i in range(ag.height)]


class Normalizer:
    """Mean / SD of the three log scalars over all real-train events (zero SD -> 1)."""

    def __init__(self, seqs: list[np.ndarray]):
        X = np.concatenate(seqs)[:, SCALARS].astype(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0

    def __call__(self, seqs):
        out = []
        for x in seqs:
            x = x.copy()
            x[:, SCALARS] = (x[:, SCALARS] - self.mu) / self.sd
            out.append(x)
        return out

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(N_IN, HIDDEN, batch_first=True)
        self.drop = torch.nn.Dropout(DROPOUT)
        self.proj = torch.nn.Linear(HIDDEN, DIM)

    def forward(self, seqs: list[torch.Tensor]) -> torch.Tensor:
        """L2-normalized embedding of each (non-empty) sequence: the projection of the final hidden state."""
        lens = torch.tensor([len(s) for s in seqs])
        assert (lens > 0).all()
        pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        packed = pack_padded_sequence(pad, lens, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return Fn.normalize(self.proj(self.drop(h[-1])), dim=-1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """Symmetric NT-Xent over 2B views: each view's positive is its pair, the other 2B - 2 views are negatives."""
    z = torch.cat([z1, z2])
    B = len(z1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return Fn.cross_entropy(sim, tgt)


def crop(x: np.ndarray, rng) -> np.ndarray:
    n = len(x)
    L = max(1, math.ceil(CROP * n))
    s = int(rng.integers(0, n - L + 1))
    return x[s:s + L]


def train(seqs: list[np.ndarray], dev="cuda", log=print) -> tuple[Encoder, list[float]]:
    """The fixed recipe on (standardized) real-train sequences; agents with no events are skipped."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    keep = [s for s in seqs if len(s)]
    enc = Encoder().to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=WD)
    losses = []
    for ep in range(EPOCHS):
        enc.train()
        order = rng.permutation(len(keep))
        tot, nb = 0.0, 0
        for b in range(0, len(order) - BATCH + 1, BATCH):  # full batches only: the last incomplete one is dropped
            idx = order[b:b + BATCH]
            v1 = [torch.as_tensor(crop(keep[i], rng), device=dev) for i in idx]
            v2 = [torch.as_tensor(crop(keep[i], rng), device=dev) for i in idx]
            loss = nt_xent(enc(v1), enc(v2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        losses.append(tot / max(nb, 1))
        log(f"epoch {ep + 1}: loss {losses[-1]:.4f}")
    enc.eval()
    return enc, losses


@torch.no_grad()
def embed(enc: Encoder, seqs: list[np.ndarray], dev="cuda", chunk=256) -> np.ndarray:
    """(n, DIM) embeddings of complete sequences, in input order; empty sequences give NaN rows."""
    enc.eval()
    out = np.full((len(seqs), DIM), np.nan, np.float32)
    nz = [i for i, s in enumerate(seqs) if len(s)]
    nz.sort(key=lambda i: len(seqs[i]))  # similar lengths per chunk: less padding
    for a in range(0, len(nz), chunk):
        ids = nz[a:a + chunk]
        out[ids] = enc([torch.as_tensor(seqs[i], device=dev) for i in ids]).cpu().numpy()
    return out
