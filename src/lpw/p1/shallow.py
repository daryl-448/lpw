"""Shallow-row correction offsets (P1 after drift screen v2; design: runs/p1_diag/after_v2/sol_r2/sol_r2.md §4).

Diagnosis (runs/p1_diag/after_v2): short accounts are under-fit one step ahead on real histories, in-sample. For
example, a light normal account's P(next owned event is OUT) after one owned event is 0.53 vs 0.64 observed on train.
Their shallow rows are ~2% of training prefixes, so per-prefix MLE barely weighs them, while they are most of the
agents in the agent-weighted fidelity metrics.

The correction is a frozen-backbone additive offset on three heads, for rows with a KNOWN plan and owned depth
d = n_own <= D_MAX. Every other row (deeper, or plan unknown) is bit-identical to the base, so the s-unknown NLL is
unchanged. The offset of a row is the sum of three role-specific effects:

    A[role, plan bucket, d] + Bk[role, d, previous token kind] + Bc[role, d, previous token cp class]

The previous token is the prefix's last token: PROFILE (and cp "none") at d = 0. Each effect is a vector of N_OUT
outputs:
- [0] kind: log-odds offset added to the OUT logit;
- [1..4] cp: offsets on the REPEAT and VENUE gate logits, NEW the reference, per next kind (OUT: 1-2, IN_exo: 3-4),
  added before the legality masks;
- [5..10] time, per next kind (OUT: 5-7, IN_exo: 8-10):
  - a same-second (bin 0) logit offset;
  - a dormant (bin 63) logit offset;
  - a slope on the standardized log midpoint of finite bins 1-62.
  These are added to the time head's bin logits; the normalized weekly law is rebuilt from them as usual.

The fitted offsets are learned by penalized maximum likelihood on training groups (lpw.p1.shallow_fit); none are
hand-set. `cols` selects the enabled output block(s) (the kind / +cp / +time ablations); disabled columns stay 0.
"""

import numpy as np
import torch
import torch.nn as nn

from lpw.p1.plan import N_PLAN_N
from lpw.p1.timelaw import EDGES

D_MAX = 4
N_PK = 4   # previous token kind: OUT, IN_exo, IN_agent, PROFILE (store codes 0-3)
N_PC = 4   # previous token cp class: REPEAT, VENUE, NEW, none (PROFILE fill, collate code 3)
N_OUT = 11
COLS = {"kind": [0], "cp": [1, 2, 3, 4], "time": [5, 6, 7, 8, 9, 10]}
VARIANTS = {"k": ("kind",), "kc": ("kind", "cp"), "kct": ("kind", "cp", "time")}

_mid = np.log(np.sqrt(EDGES[:-1] * EDGES[1:]))  # log midpoints of finite bins 1-62
Z_BIN = torch.as_tensor((_mid - _mid.mean()) / _mid.std(), dtype=torch.float32)


def col_mask(blocks) -> torch.Tensor:
    m = torch.zeros(N_OUT)
    for b in blocks:
        m[COLS[b]] = 1.0
    return m


class Shallow(nn.Module):
    def __init__(self, blocks=("kind", "cp", "time")):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(2, N_PLAN_N, D_MAX + 1, N_OUT))
        self.Bk = nn.Parameter(torch.zeros(2, D_MAX + 1, N_PK, N_OUT))
        self.Bc = nn.Parameter(torch.zeros(2, D_MAX + 1, N_PC, N_OUT))
        self.register_buffer("cols", col_mask(blocks))

    def offsets(self, role, plan, nown, pk, pc) -> tuple[torch.Tensor, torch.Tensor]:
        """(offsets (N, N_OUT), active (N,)) per row. A row is active iff its plan is known (plan < N_PLAN_N) and its
        owned depth is <= D_MAX; inactive rows get zero offsets and are left bit-identical by the apply_* functions."""
        act = (plan < N_PLAN_N) & (nown <= D_MAX)
        p, d = plan.clamp(max=N_PLAN_N - 1), nown.clamp(max=D_MAX)
        off = self.A[role, p, d] + self.Bk[role, d, pk] + self.Bc[role, d, pc]
        return torch.where(act[:, None], off * self.cols, torch.zeros_like(off)), act

    def row_offsets(self, B) -> tuple[torch.Tensor, torch.Tensor]:
        """offsets() for every row of a collated batch, using the effective plan of the batch's encode()."""
        rb, rj = B["row_b"], B["row_j"]
        return self.offsets(B["role"][rb], B["_plan_n_eff"][rb], B["row_nown"], B["tok_kind"][rb, rj], B["tok_cls"][rb, rj])

    def penalty(self) -> torch.Tensor:
        """sum of squares of the enabled entries (the L2 partial-pooling term)."""
        return sum(((t * self.cols) ** 2).sum() for t in (self.A, self.Bk, self.Bc))


def apply_kind(logpk: torch.Tensor, off: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """kind log-probs (N, 2) with the OUT log-odds offset, renormalized on active rows. Inactive rows are returned as
    they are (bit-identical: renormalizing log-probs again would round)."""
    add = torch.stack([off[:, 0], torch.zeros_like(off[:, 0])], -1).to(logpk.dtype)
    return torch.where(act[:, None], torch.log_softmax(logpk + add, -1), logpk)


def apply_time(out: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
    """time-head output (2N, >= 64), rows 2i + kind: same-second, dormant and log-gap-slope tilts of the bin logits."""
    t = off[:, 5:11].reshape(-1, 3).to(out.dtype)  # (2N, 3): row 2i + kind
    z = Z_BIN.to(out.device, out.dtype)
    bins = torch.cat([out[:, :1] + t[:, :1], out[:, 1:63] + t[:, 2:3] * z[None], out[:, 63:64] + t[:, 1:2]], 1)
    return torch.cat([bins, out[:, 64:]], 1)


def apply_gate(g: torch.Tensor, off: torch.Tensor, ko: torch.Tensor) -> torch.Tensor:
    """raw gate logits (n, 3) [REPEAT, VENUE, NEW] of owned rows with next kind ko and their row offsets off (n, N_OUT)."""
    c = torch.where((ko == 0)[:, None], off[:, 1:3], off[:, 3:5]).to(g.dtype)
    return g + torch.cat([c, torch.zeros_like(c[:, :1])], -1)
