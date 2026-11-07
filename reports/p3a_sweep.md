# P3a defect-1 sweep repair: REJECTED (mechanism, scam); readout reported as pre-registered

- **Protocol.** `reports/p3a_prereg.md` §15 (addendum A, commit `fc7a60e`, frozen before fitting). The design came
  from a Sol consult.
- **Build.** `7af42aa`, reviewed with sol-loop (gpt-6.1-sol, 2 rounds, approved).
- **Outputs:**
  - `runs/p3a/2018/sweep/{gate.pt,fit.json,acceptance.json,diag.{txt,json}}`;
  - `runs/p3a/2018/readout_sweep.{txt,json}`;
  - `world{10,11,12}/checkpoint0.txt`.
- **Data.** 2018 val is development data, and this experiment was informed by the first readout. The sealed test is
  untouched.
- **Compute.** About 45 GPU-minutes in all (fit 2.5 min; 3 worlds × 2 arms of generation at 4-6 min each), inside
  the 2 GPU-hour budget.

## Gate fit (train only)

- **Data.** 126,391 eligible train OUTs, of which 12,044 are sweeps (|b - eth - fee| ≤ 1e-4).
- **Out-of-fold quality.** On 3 whole-group folds, BCE per row is 0.170 / 0.129 / 0.166, against 0.366 / 0.263 /
  0.335 for an intercept-only gate.
- **Calibration** (mean OOF p vs observed rate): normal 0.090 vs 0.092, scam 0.130 vs 0.128. The deciles track the
  rates; the top scam decile has p 0.66 vs rate 0.56.
- **Generation.** The gate fires on about 10% of eligible OUTs (8.0-8.3k sweeps per world).

## Acceptance (locked in `acceptance.json` before any utility or seam number)

| gate | normal | scam |
|---|---|---|
| Terminal near-zero share, mean over worlds 10-12 (band) | **0.105** (0.067-0.200): pass | **0.126** (0.204-0.611): **fail** |
| Unmodified kct on the same populations | 0.002 | 0.000 |
| M-pass difference vs kct (≥ -0.10) | +0.02: pass | +0.10: pass |
| Safety (every population) | pass | pass |

**Decision: REJECTED.** kct stays the release recipe, and the original P3a readout (`reports/p3a_readout.md`) stays
the baseline. Under the addendum, nothing is retuned and no second candidate is tried.

## Why the mechanism fails: the policy keeps acting after a sweep (fixed diagnostics, A.5)

| per role, mean over worlds | real val normal | kct_sweep normal | real val scam | kct_sweep scam | kct scam |
|---|---|---|---|---|---|
| sweep events / OUTs | 0.046 | 0.054 | 0.133 | 0.081 | 0.000 |
| agents ever sweeping | 0.147 | 0.209 | 0.444 | 0.377 | 0.003 |
| terminal near zero | 0.105 | 0.105 | 0.398 | 0.126 | 0.000 |
| sweeps that are the last OUT | 0.32 | 0.30 | 0.46 | 0.26 | n/a |
| sweepers with a later IN | 0.49 | 0.56 | **0.32** | **0.64** | n/a |
| OUTs leaving the balance < -1e-4 | 0.33 | 0.52 | 0.33 | 0.58 | 0.52 |

- **Sweeps happen in roughly the right share of scam accounts** (0.38 vs 0.44). But synthetic scams then receive
  again twice as often (0.64 vs 0.32), and their sweeps are less often the final OUT (0.26 vs 0.46). So they rarely
  end at zero.
- **This needs a stopping or funding behaviour, not an amount atom,** which is what Sol's design anticipated. Under
  A2 it is out of scope.
- **A second, independent gap: overdrawing.** Synthetic OUTs leave the observed balance negative far more often than
  real ones (0.52-0.58 vs 0.33). The amount head ignores affordability, and the gate does not change continuous
  draws. Not tested; a candidate known defect.

## Pre-registered readout on worlds 10-12 (both arms; kct_sweep labelled REJECTED)

| | kct_sweep (REJECTED) | kct (control, same populations) |
|---|---|---|
| TSTR AUC, full / XGB (3 worlds) | 0.916 ± 0.004 | 0.909 ± 0.005 |
| ratio; D [CI95]; outcome | 0.963; +0.070 [+0.048, +0.088]; SUPPORTED | 0.948; +0.064 [+0.041, +0.083]; SUPPORTED |
| matched TSTR / behavioural gain | 0.886 / +0.250 | 0.877 / +0.241 |
| C2ST `local` (normal / scam) | 0.848 / 0.852 (fail; gap +0.004) | 0.870 / 0.890 (fail) |
| C2ST BAL-only | 0.793 / 0.812 | 0.817 / 0.853 |
| C2ST LOCAL + VENUE without BAL | 0.723 / 0.754 | 0.753 / 0.775 |
| difficulty parity, XGB AP | -0.133 (fail) | -0.145 (fail) |
| shortcut flags | 3: end_balance, ends_near_zero, fee_med | 7 (as in worlds 0-2) |
| mode coverage, scam P / R | 0.89 / **0.94** | 0.89 / 0.57 |
| DCR, 5th-percentile ratio (normal / scam) | 6.85 / **1.36** (pass) | 7.03 / 3.40 |

The kct control reproduces worlds 0-2: TSTR 0.909, the same 7 flags, and scam recall 0.57. The baseline readout is
therefore stable across independent world draws.

## Reading

1. **The decision stands on its pre-registered gate.** The repair is rejected for scams, whose terminal cash-out
   remains about a third of real. The readout improvements below **cannot** overturn that: the addendum rules out
   any utility-based rescue.
2. **Descriptively, restoring the sweep action alone moves several criticism tests in the right direction:**
   - flags 7 → 3;
   - C2ST 0.87 / 0.89 → 0.85 / 0.85, with the scam - normal gap at +0.004;
   - scam mode recall 0.57 → 0.94.

   The scam recall jump says real scam accounts that cash out had no synthetic neighbours before. These figures are
   development observations of a rejected candidate. They come from one unconfirmed run and are not
   generator-release results.
3. **The repaired arm sits closer to real-train scam records.** Its scam DCR ratio falls from 3.4 to 1.36: still a
   pass (≥ 1), but closer. A sweep is a deterministic amount (b - fee), so it reproduces real exact-cash-out
   signatures. If such a mechanism is ever promoted, DCR deserves attention.
4. **Transfer is not driven by balance realism.** TSTR moves by +0.007, inside the world-to-world spread.
5. **What remains:**
   - the seam without BAL is 0.72-0.75, still above 0.70;
   - difficulty parity fails by about 0.13 AP;
   - post-sweep behaviour (refills) and overdrawing are open.

   All are generator-behaviour defects that need more than an amount-only change. Under A2, P1 is not reopened for
   them.

## Known defects carried forward (updates `reports/p1_exit.md`)

- **Defect 1 (no sweeping):** still open in the release recipe. An amount-only gate restores the action but not
  terminal cash-out for scams (refill and continuation behaviour).
- **Candidate defect 8 (profile arithmetic incoherence):** unchanged, see `reports/p3a_readout.md`.
- **Candidate defect 9 (overdrawing):** synthetic OUTs leave the observed balance negative in 52-58% of cases,
  against 33% real.
