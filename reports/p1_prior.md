# P1 prior-mode loss decomposition (exploratory)

Written before the full run (2026-11-07). This is development exploration under the post-drift-v2 protocol:
- the search is disclosed and ungated;
- val is development data;
- confirmation comes later, once, on the frozen pipeline.

Code: `src/lpw/p1/drift_exp.py` (new plan modes) and `scripts/p1_prior/{run.sh,decompose.py}`.

## Why

The shallow-row correction (kct, `reports/p1_shallow.md`) lifts oracle normal M-pass from 0.650 to 0.825, but prior
mode gets only 0.725. P1 exit is judged in prior mode. The prior loss is spread over metrics near the pass threshold
(normal M5 dow, M12 span, M9 repeat rate, M1 amount). For scam it sits in M1 amount (1.00 -> 2.75) and M3 kind
(0.88 -> 2.96).

Before changing the plan prior, we need to know which part of the loss each mechanism carries:
- **the plan marginal:** the prior's val distribution of plans;
- **the individual assignment:** each agent's real plan, partly predictable from profile, partly not;
- **the amount bucket.**

## Plan sources

All use the frozen kct model, generation seed 7 and 5 replicas, with plans drawn with rng 101.

| source | plans | what it isolates |
|---|---|---|
| oracle | real val plans | (from the shallow run) |
| prior | `PlanPrior`, default | (from the shallow run) |
| perm | real joint plans, permuted among val agents of the same role x cohort | exact val marginal, no individual information |
| perm_a | real count bucket; amount bucket permuted within role x count bucket | individual amount assignment |
| prior_exp | exposure-aware `PlanPrior` (activation-time inputs; val plan NLL 3.105 vs 3.144) | a cheap candidate fix |

## Readouts (`decompose.{json,txt}`)

- **Scoring.** Normal and scam M-pass and gmean per source, on the same split seeds (seed = replica index). Each
  group is scored in its own `compare()` call.
  - Why: `compare()` draws all groups' bootstrap floors from one rng, in order. Scored in one call, a group's floors
    would depend on which groups precede it.
  - Disclosed for the shallow readout: it scored prior-mode scam after only `normal`, but oracle scam after the
    normal bands, so its scam prior retention mixed floors. Normal comes first and is unaffected, and so are all
    decision rules.
- **Bands.** Real owned-count bands only for oracle and perm_a, the sources that keep each agent's count bucket.
- **Contrasts.** Each is the mean ± sd of per-replica differences:
  - assignment = oracle - perm;
  - amount assignment = oracle - perm_a;
  - prior vs exact marginal = prior - perm;
  - exposure prior = prior_exp - prior.
- **Plan diagnostics per role,** vs real val plans:
  - count / amount bucket total variation;
  - the share of agents whose sampled bucket equals their real one.

## Readings (fixed before the run; normal, with 0.05 as the material size)

Confounds:
- perm keeps the exact val marginal but drops every individual association, including the profile / exposure
  associations a prior can learn. oracle - perm therefore bounds the value of individual plan information, both
  learnable and not. It is not an irreducible ceiling.
- prior keeps profile conditioning but has its own marginal error. prior - perm mixes the two, and they can cancel.

Readings:
- **Assignment (oracle - perm) >= 0.05:** individual plan information matters. How much of it a prior could learn
  is not identified here.
- **prior - perm:**
  - <= -0.05: consistent with the prior's val marginal costing fidelity (miscalibration or group shift), net of the
    prior's profile information. Hypothesis only; any calibration work is judged on train-group folds, not val.
  - >= +0.05: the prior's profile conditioning is worth more than its marginal error costs.
  - otherwise: not resolved.
- **Amount assignment (oracle - perm_a) >= 0.05:** the individual amount bucket carries a material share. For scam,
  also read the M1 ratio from oracle to perm_a. This includes cohort / profile-amount associations a prior could
  learn, since perm_a permutes within role x count only.
- **prior_exp:** adopted as the frozen pipeline's prior if its normal and scam M-pass are both no lower than the
  default prior's. Otherwise the default stays. There is no other prior search on val.
- **Noise:** prior-mode replica sd is about 0.1, so contrasts under 0.05 are noise.

## Compute

Local GTX 1650S: 3 generations of ~5 min and a ~10 min readout, about 25 min in all.

## Result (2026-11-07, full run at be01897; outputs in `runs/p1_prior/2018/`)

M-pass (kct, 5 replicas, shared split seeds):

| | oracle | prior | prior_exp | perm | perm_a |
|---|---|---|---|---|---|
| normal | 0.825 ± 0.047 | 0.725 ± 0.116 | 0.688 ± 0.088 | 0.738 ± 0.061 | 0.812 ± 0.000 |
| scam | 0.938 ± 0.000 | 0.800 ± 0.025 | 0.775 ± 0.031 | 0.938 ± 0.000 | 0.925 ± 0.025 |

Contrasts (mean ± sd of per-replica differences):

| contrast | normal | scam |
|---|---|---|
| assignment (oracle - perm) | +0.087 ± 0.075 | +0.000 |
| amount assignment (oracle - perm_a) | +0.013 ± 0.047 | +0.013 ± 0.025 |
| prior - perm | -0.013 ± 0.139 | **-0.138 ± 0.025** |
| prior_exp - prior | -0.037 ± 0.140 | -0.025 ± 0.031 |

Readings, under the rules fixed above:
- **Normal.**
  - Individual plan information matters (+0.087, more than one sd). How much of it a prior could learn is not
    identified.
  - prior - perm is unresolved (-0.013 ± 0.139).
  - The amount assignment is not material (+0.013).
  - One prior replica scored 0.50, which drives most of the prior's spread.
- **Scam.**
  - Shuffled real plans match oracle exactly, so individual assignment does not matter for scam.
  - Prior-sampled plans lose 0.138 ± 0.025, mostly in M1 amount (1.07 -> 2.80), M3 kind (0.90 -> 2.98) and M6.
  - This is consistent with the prior's plan marginal costing scam fidelity (hypothesis).
  - Amount assignment is not material: M1 is 0.80 in perm_a.
- **prior_exp** is lower on both roles. Per the rule, the default prior stays.

Post-run check (not pre-registered), the plan marginals per role x cohort (TV):

| cell | n train / val | train vs val | prior on val vs val | prior on val vs train | prior on train vs train (in-sample) |
|---|---|---|---|---|---|
| normal pre | 4458 / 1103 | 0.079 / 0.083 | 0.069 / 0.073 | 0.013 / 0.013 | 0.007 / 0.010 |
| normal born | 11368 / 2289 | 0.025 / 0.040 | 0.025 / 0.039 | 0.003 / 0.005 | 0.003 / 0.005 |
| scam pre | 157 / 38 | 0.136 / 0.266 | 0.254 / 0.173 | 0.284 / 0.302 | **0.191 / 0.239** |
| scam born | 1471 / 311 | 0.066 / 0.075 | 0.083 / 0.107 | 0.026 / 0.033 | 0.026 / 0.033 |

Each TV pair is count / amount.

- **Normal and born scam:** the prior reproduces the train marginal. Its val error is about the train-vs-val group
  shift, which a train-fitted prior cannot remove. This is the new-group estimand.
- **Pre-existing scam:** the prior misfits even in sample. The logistic features are additive in role, cohort and
  profile, with no role x cohort term, and pre-existing scams are a small cell. Example: it gives count bucket 5 a
  probability of 0.30, against 0.12 in train.
- **Open:** whether the scam prior loss comes from this misfit (fixable) or from group shift (not fixable). A
  train-marginal plan source, sampled from the empirical train joint per role x cohort, would split the two.
