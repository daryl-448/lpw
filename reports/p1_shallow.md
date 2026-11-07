# P1 shallow-row correction (exploratory)

Written before any full-size fit or generation (2026-11-07). This is development exploration under the protocol
agreed after drift screen v2:
- the search is disclosed, with no gated pass / fail screen;
- val is development data;
- confirmation comes later, once, on the frozen phase-level pipeline.

Design: Sol round 2 (`runs/p1_diag/after_v2/sol_r2/sol_r2.md` §4). Code:
- `src/lpw/p1/shallow.py` (offsets), `src/lpw/p1/shallow_fit.py` (fit and selection);
- `scripts/p1_shallow/{run.sh,readout.py}` (generation and readouts).

## Why

Short accounts are under-fit one step ahead (`runs/p1_diag/after_v2/synthesis.md`).
- **The error.** A light normal account's P(next owned = OUT) after one owned event is 0.53 vs 0.64 observed, in
  sample on train and on val. The model also over-predicts incoming REPEAT and makes outgoing gaps too fast at
  depths 2-3.
- **Why training misses it.** These rows are ~1.6% of training prefixes but 57% of normal val agents.
- **Why it matters.** In oracle mode, replacing the light band with matched real streams raises normal M-pass
  0.66 -> 0.88; the heavy band, 0.66 -> 0.69.

## Method

- **Base.** `runs/p1_r0/2018/lpw_plan_prog_rp`, frozen.
- **Offsets.**
  - Additive offsets on the kind log-odds, the REPEAT / VENUE gate logits per next kind, and three time-bin tilts per
    kind: same-second, dormant, and log-gap slope.
  - They apply only on rows with a known plan and owned depth <= 4.
  - Each row's offset is A[role, plan bucket, depth] + Bk[role, depth, previous token kind] +
    Bc[role, depth, previous token cp class].
  - Every other row, and every row with the plan unknown, is bit-identical to the base, so the s-unknown NLL is
    unchanged. There are no quotas or hand-set rates.
- **Variants.** k (kind), kc (+ cp), kct (+ time).
- **Fit.** L-BFGS on the summed NLL of all shallow training rows, with L2 penalty lam / 2 x |offsets|^2. The rows
  are kind, time, censoring and gate terms, including terminal and IN_agent-censored rows. Pointer, venue and amount
  heads are untouched.
- **Selection.**
  - Train groups only, 3 group folds (seed 0); lam in {1, 10, 100}.
  - Score: mean over held-out agents of the agent's mean shallow-row NLL.
  - Per variant: the largest lam within one whole-group SE of the variant's best.
  - Primary: the first of (zero, k, kc, kct) within one SE of the overall best.
  - All three variants are refitted on all training groups and all are read out.

## Readouts (`readout.{json,txt}`)

- **Teacher-forced (val and train)**, per role x band (real owned count 1-4 / 5-99 / 100+):
  - shallow NLL change vs B;
  - P(OUT) and P(REPEAT | kind) at depths 0-3, model vs observed;
  - known-plan NLL change per owned event per role.
- **Oracle generation** (5 paired replicas, seed 7, real plans; via `lpw.p1.drift_exp`):
  - M-pass and ratio_gmean per role x band;
  - a matched real reference per band: train donors matched on role, cohort, age decile / activation month and plan
    count bucket, 3 draws;
  - gap closed = (M_arm - M_B) / (M_ref - M_B), unclipped, with the raw difference.
- **Prior generation** (5 replicas, PlanPrior rng 101, identical plans across arms):
  - normal / scam M-pass and gmean;
  - retention of the oracle gain;
  - safety counts: capped, finite-plan N > 3U, nonempty zero-plan.
- **Sanity:** the s-unknown val NLL of every arm equals B's.
- **Uncertainty:** replica sd and donor draws are reported. No group bootstrap of M-pass; differences of a few
  hundredths are within replica noise (B's replica spread is about +-0.1).

## Decision rules (primary variant)

- **Kill** if any of these holds:
  - the selection keeps zero;
  - the val shallow NLL of light normal agents does not improve;
  - the known-plan val NLL rises by more than 0.02 per owned event in either role.
- **Promote** to the next step (prior-mode work, then a frozen comparison) only if all of these hold:
  - light normal gap closed >= 25% in oracle mode;
  - mid normal M-pass not below B's;
  - prior-mode normal M-pass not below B's.
- **Otherwise:** stop promotion. Report it as an objective / capacity finding.
- **No tuning of offsets to val rates** and no re-selection after seeing generation results. The other variants
  are ablations.

## Compute

Local GTX 1650S, measured in smoke runs:
- the fit takes seconds to minutes (cached rows);
- each of the 8 generations (4 arms x oracle / prior, 5 replicas) takes ~3-6 min;
- the readout takes ~10 min.

Expected about 1 h.

## Result (2026-11-07, full run at cc94f18; outputs in `runs/p1_shallow/2018/`)

**Outcome: PROMOTE (primary kct).** Every pre-registered check passes. This is development exploration on val, not
confirmation.

- **Selection.** kct is the best variant, with lam 10 for all three. It beats zero by 0.022 nats per agent
  (SE 0.002); k and kc are 0.005-0.006 behind kct.
- **Teacher-forced, val, kct.**
  - Shallow NLL vs B: light normal -0.032 per agent, light scam -0.126; mid / heavy -0.01 to -0.03.
  - Known-plan NLL per owned event: -0.006 normal, -0.007 scam (kill threshold +0.02).
  - Light normal P(OUT) at depth 1: 0.627 vs 0.629 observed (B 0.517).
  - Incoming REPEAT at depth 1: 0.452 vs 0.425 (B 0.529).
  - New miss: incoming REPEAT at depth 2 is now 0.431 vs 0.551 observed (B 0.472).
  - s-unknown val NLL identical for every arm (13.127763760493428).
- **Generation.** M-pass, 5 paired replicas; replica sd 0.05-0.07 oracle, ~0.1 prior.

  | | B | k | kc | kct | matched real |
  |---|---|---|---|---|---|
  | oracle normal | 0.650 | 0.775 | 0.787 | **0.825** | 0.963 |
  | oracle normal light 1-4 | 0.425 | 0.662 | 0.800 | **0.863** | 0.963 |
  | oracle normal mid 5-99 | 0.738 | 0.575 | 0.625 | **0.738** | 0.879 |
  | oracle normal heavy 100+ | 0.875 | 0.825 | 0.775 | 0.850 | 0.979 |
  | oracle scam | 0.863 | 0.838 | 0.938 | 0.925 | 0.979 |
  | prior normal | 0.637 | 0.662 | 0.725 | **0.725** | - |
  | prior scam | 0.825 | 0.850 | 0.838 | 0.825 | - |

  - Gap closed (kct, oracle): light normal 81%, normal overall 56%.
  - Mid normal ties B exactly: the rule passes on equality only. k and kc worsen mid, so the time tilts are what keep
    it level.
  - Prior normal +0.088 vs B: 50% retention of the oracle gain, within about one replica sd.
  - Safety counts are no worse than B's.
- **Reference.** The matched-real reference (0.963) is higher than the brainstorm's profile-only ceiling (~0.84). It
  matches on the plan count bucket and is scored on the arms' split seeds, with 7 fallback donors per draw.
- **Reading.** The shallow under-fit was a cheap, fixable conditional error, and fixing it carries into rollouts.
  What remains:
  - the prior-mode loss (oracle 0.825 -> prior 0.725 normal);
  - the mid band, flat vs B and 0.14 below its reference;
  - B1 NLL non-inferiority: open, and untouched by this correction.
