# P1 anchor re-screen (regularised): pre-registration

Written and committed before any re-screen adapter was fitted (2026-11-06, after the Sol own-generation round).

## Why

The first screen (`reports/p1_anchor_screen.md`) failed. The arms improved generation (pace score B 0.177, C 0.115,
R 0.095; reach-100 0.38 / 0.48 / 0.55), but the likelihood got worse (known-plan val NLL C +0.195, R +0.289 vs B).
That loss came from overfitting: C - B was -0.196 on the fit subset, +0.185 on held-out training groups and +0.194 on
val, and it was concentrated in plan >= 300 accounts.

Sol recommended this regularised re-screen before any training on the model's own rollouts. A rollout-consistency
screen is pre-registered separately, and only if this one gives an acceptable control.

## Arms (from `runs/p1_r0/2018/lpw_plan_prog_rp`, one seed)

The adapter architecture, corruption scheme and loss weight (0.25) are unchanged from the first screen.

| arm | training |
|---|---|
| B | untouched base checkpoint |
| C | adapters only, clean prefixes |
| R | adapters only, clean + 0.25 x corrupted prefixes (same shifts as the first screen) |

Fit (`python -m lpw.p1.anchor --reg`):
- **Data.** All training groups except a fixed internal holdout of 20% of training groups (seed 0); all their exact prefixes, real plans.
- **Optimiser.** AdamW on the adapters only, lr 3e-4, weight decay 0.1, 4096 rows per step, at most 4000 steps, seed 0.
- **Checkpoint selection.**
  - Every 200 steps, measure the mean clean per-prefix NLL on the internal holdout (time + censoring + amount, kind constant).
  - Keep the best checkpoint. Step 0, the untouched base, is a candidate, so C or R may legitimately end up equal to B.
- **Sampling.**
  - Clean rows: drawn uniformly (plain MLE).
  - R's corrupted rows: drawn with probability proportional to n_rows(agent)^-0.5, so a few long streams do not dominate the auxiliary term.

## Gates (all must pass; applied by `scripts/p1_anchor/rescreen.py`)

**L. Likelihood, for both C and R.**
- The s-unknown val terms are bit-identical to B on every exact prefix.
- Exact known-plan NLL per prefix (unweighted mean) satisfies arm - B <= +0.02:
  - overall, normal, scam, and pooled plan >= 300;
  - on both the internal held-out groups and val.
- The internal holdout also selects the checkpoint, so val is the independent check.

**S. Shift response, R vs C.**
- Measured on real val prefixes with plan >= 10, the next event owned, normal accounts, after 3 / 9 / 29 / 99 owned events (pooled).
- Mean |e_tt| and mean |e_aa| must each fall by >= 25%.
- The whole-group bootstrap 2.5% quantile of the reduction must be > 0.

**D. Rollout discrepancy, R vs C.**
- Population: normal val accounts with plan >= 100, at depths of 9 / 29 / 99 owned events.
- For each agent and depth, take the arm's forecast on its own generated prefix (mean over replicas) minus its forecast on the real prefix at the same depth. Two forecasts are compared:
  - D_t uses Elog10 of the next own gap;
  - D_a uses the amount-forecast mean at the next owned event.
- Average over agent-depth pairs present in both arms.
- |D_t| and |D_a| must each fall by >= 25%, with a bootstrap 2.5% quantile of the reduction > 0.
- This is the paired real-vs-generated discrepancy that the first screen left out.

**G. Pilot generation, R vs C.**
- Setup: val agents with plan >= 30, oracle plans, 3 replicas, seed 7, cap 150 events.
- Pace score (same definition as the first screen) must fall by >= 15%.
- Amount W1 must fall by >= 15%:
  - measured on normal plans >= 100, events 31-100;
  - weighted per agent: each agent's replicas share one unit of weight, and each real agent has one unit.
- Both of these must have a paired whole-group bootstrap 2.5% quantile of (C - R) > 0.
- Reach-100 (normal plans >= 100) must not be lower than C's.

**Bootstrap.** 1000 resamples of val groups (`agents.parquet` group_id) with replacement, seed 0. Arms are paired within each resample.

## Outcomes

- **Pass.** The full experiment, prior mode and the 3-seed confirmation get their own pre-registration, which needs the user's OK.
- **R fails S/D/G while C passes L.** C is an acceptable control for the pre-registered rollout-consistency screen.
- **C fails L.** The frozen-backbone adapter implementation is rejected, and the drift needs a different design.
