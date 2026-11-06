# P1 drift screen v2 (partial-backbone, shift consistency): pre-registration

Written before any R snapshot was evaluated (2026-11-07). Design: Sol drift-screen-v2 round (after the base
qualification), amended by Sol build review r1. Code: `src/lpw/p1/drift_ft.py`, `scripts/p1_drift/screen.py`.

## Why

The frozen-adapter anchor screens failed: the likelihood control forced step 0. Sol's v2 moves the anchor into the
upper backbone and replaces the corrupted-likelihood term with a consistency term toward the untouched base, so the
fine-tune is pulled towards "the plan, not the drifting history, sets the level" without new likelihood targets.

## Development amendment (disclosure)

This is a development amendment after inspection of C-only pilot likelihood trajectories (and the C shift-sensitivity
summaries printed with them; `runs/p1_drift_pilot/`). No R outcome selected the revised recipe.
- With the designed head lr 1e-4 and a 20% group exclusion, no C snapshot after step 0 met the budget: the excluded
  groups were in B's training data (forgetting, not overfitting) and fresh AdamW at 10x B's final lr moved val by
  +-0.04 between snapshots.
- With lr 1e-5 everywhere and all training groups, C moved val by +0.002 to +0.009 (all) and up to +0.039 (plan >= 300).
  C still selects step 0 there (every snapshot has a worse unknown-plan val NLL than B). C = step 0 is a legitimate outcome.
- The designed J weight 0.05 gave a J gradient 5e-5 to 4e-4 of the clean one at step 0; the weight is now set by the
  calibration rule below.

## Arms (2018, base `runs/p1_r0/2018/lpw_plan_prog_rp`, one seed 20261104)

| arm | training |
|---|---|
| B | untouched base |
| C | clean fine-tune |
| R | clean + w_J x J |

H (stopping term) is deferred.

Both arms (identical except J):
- **Trainable.** Upper two encoder layers, final norm, time_head, amt_head. Everything else frozen (asserted).
- **Data.** Every training group (the former 20% subset is reported-only retention data).
- **Loss.** The base's training loss: plain per-prefix likelihood with censoring and plan dropout, scaled by rows / the
  base's ref_rows, gradient clip 1.0 x scale, fp16 autocast with GradScaler.
- **Optimiser.** AdamW, weight decay 0.1, lr 1e-5 for every trainable parameter, constant, 1200 attempted updates.
  Snapshot numbering counts attempted updates; GradScaler-skipped updates are recorded.
- **Snapshots.** Steps 0, 25, 50, 100, 200, 400, 800, 1200 (step 0 = B). No recalibration.
- **Determinism.** Same seed, clean batch order and plan-dropout / dropout draws in C and R (the auxiliary forwards
  and the calibration consume no torch RNG).

**J (shift consistency).**
- Auxiliary prefixes: 128 per update. Agent uniform over training agents with at least one event, then prefix length
  uniform in [1, n]; independent of the clean batches.
- Corruption: `anchor.corrupt_view`. Prefix-wide positive-gap factor 10^{-0.3, 0, +0.3} and log-amount shift
  {-0.5, 0, +0.5}, probabilities 0.25 / 0.5 / 0.25, independently. Zero gaps, timestamps and kinds are kept;
  ledger-derived features scale with the amounts.
- Student: the arm on the corrupted prefix. Teacher: frozen B on the clean prefix. Both use the real plan and
  eval-mode forwards; the targets' own fields (kind, time, counterparty) come from the clean view.
- J = 1/2 mean over rows of the mean squared difference of the next-owned-event gap CDF at 64 points + 1/2 mean over
  owned-target rows of the mean squared difference of the truncated amount CDF at 64 points.
  - Gap points: the same-second atom, the 62 finite-bin log midpoints (integrated with the model's weekly law) and
    365 d (1 - F there is the dormant mass); mixture over the two kinds by p(kind).
  - Amount points: 64 equally spaced log10 amounts over [dust floor, total supply]; interval masses in log space.
- **Weight.** At step 0, w_J = 0.25 x median ||g_clean|| / median ||g_J||. This uses 16 fixed (clean batch,
  auxiliary batch) pairs (seed 777), the scaled clean objective, the trainable parameters and the training precision,
  with the RNG state restored. The calibration aborts on fewer than 16 pairs, non-finite losses or gradients, or a
  non-positive median or weight. No weight sweep or outcome-dependent adjustment.

## Likelihood budget and selection (`screen.py eval` / `select`)

- **Feasibility.** Val exact prefixes, known-plan AND unknown-plan full NLL per prefix. arm - B <= +0.02 for each of
  all, normal, scam and plan >= 300 (count bucket >= 300; 21 agents in 13 groups on 2018 val).
  - Reported, not gated: whole-group bootstrap 95% intervals (seed 4, 1000 draws) and agent / group / prefix counts
    per cell; the retention subset's deltas.
- **C.** Lowest unknown-plan val NLL among feasible snapshots.
- **R.** Lowest max(mean|e_tt| / B, mean|e_aa| / B) among feasible snapshots.
  - Mean absolute shift sensitivities on normal val items: agents with plan >= 10 owned events, real prefixes at
    owned depths 3 / 9 / 29 / 99.
  - e_tt is the change in the forecast E log10 positive gap per unit dt shift (+-0.3).
  - e_aa is the change in the forecast truncated log10-amount mean per unit amount shift (+-0.5).
- Step 0 is included. Ties go to the earlier snapshot. A non-finite score aborts.
- No threshold relaxation, runner-up substitution or unregistered refitting after a failure.

**Integrity.**
- Every evaluation and generation artifact is fingerprinted by its checkpoint's sha256 and checked on use.
- The selected checkpoints are re-checked.
- The C / R recipes must be equal apart from J.
- Step 0 of each arm must be state-identical to B.
- The generation recipe (replicas, caps, seeds, population) is recorded and checked.

## Generation (`screen.py gen`, selected B / C / R)

- **Pilot.** Val agents with plan >= 30 owned events, oracle plans, 3 replicas, seed 7, cap 150.
- **Prior mode.** The whole val population with plans from `PlanPrior` (no exposure inputs), rng 101.
  - 3 draws, identical across arms (asserted).
  - Seed 7, natural window stopping, cap 10,000.

## Gates (`screen.py report`; all must pass; R vs C, and R no worse than B in point estimate)

**S. Shift sensitivity.**
- Mean |e_tt| and mean |e_aa| (normal items) each reduced >= 25% vs C.
- Whole-group bootstrap 2.5% quantile of the reduction > 0 (1000 draws, seed 1).
- R <= B.

**D. Rollout discrepancy.**
- Population: normal pilot agents with plan >= 100; depths 9 / 29 / 99.
- Per pair: the forecast on the arm's generated prefix at the same owned-event depth minus the forecast on the real
  prefix. The generated forecast is averaged over the replicas that reach the depth.
- Statistic: mean |discrepancy| over the pairs with a value in all of B, C and R (common support).
- Gate: >= 25% reduction vs C, bootstrap lower bound > 0 (seed 3), and R <= B on the common support.
- Coverage guard: at every depth, R's share of real pairs with at least one reaching replica is >= max(C, B) - 0.05.
- D is conditional on reaching the prefix. Coverage and reach-100 are reported alongside it.

**G. Pilot generation.**
- Definitions as `reports/p1_anchor_rescreen.md`: pace score and agent-weighted amount W1. Non-reachers count as
  failures in the pace score.
- Each must be reduced >= 15% vs C, with the paired whole-group bootstrap 2.5% quantile of (C - R) > 0 (seed 2).
- Each must satisfy R <= B.
- reach-100: R >= C and R >= B.

**P. Safety (prior mode, all 3 replicas).**
- Zero capped streams.
- Zero finite-plan streams with N > 3U, where U is the plan bucket's upper count; zero-plan streams must be empty.
- Zero same-second runs longer than 10 x the real-training maximum (133 in 2018; recomputed and recorded).
- Reported, not gated:
  - the longest run per replica;
  - runs above the real maximum;
  - N > U;
  - open-bucket streams longer than the longest real training stream.

**F. Fidelity (prior mode).**
- Setup:
  - Real = the val population.
  - `compare` on the role groups normal and scam only, 200 real-half draws.
  - Draw seed = replica index, shared by every arm.
- Statistic: `ratio_gmean`, aggregated geometrically over the 3 replicas.
- Views: all groups, and without the largest val group by real owned events (fixed once).
- Normal: R <= 0.90 x C and R <= B.
- Scam: R <= 1.05 x B and R <= 1.05 x C.
- Every ratio_gmean must be finite, and n_ratio = 16 in every replica, role and view.
- Thresholded M-pass, per-metric ratios and M1 per-kind ratios are reported, not gated.

## Interpretation

- Val gates and bootstrap bounds are development diagnostics after selection, not independent confirmatory evidence.
- If the screen passes, the recipe is frozen before any sealed-test evaluation. A 2021 replication and multi-seed
  confirmation need separate approval.
- Final P1 still needs SPEC §8 with amendment A1 on the sealed test (>= 70% M-pass per role, role-AUC parity, baselines).
- Any mechanism claim needing confirmation must also be evaluated independently.

## Compute

Local GTX 1650S, measured on the pilot:
- about 2 min per checkpoint evaluation (15 evaluations);
- 4 min (C) and about 12 min (R) of fine-tuning;
- generation and fidelity not yet timed at full size.

Expected 2-3 h locally; the user is asked before a run that exceeds 4 h.
