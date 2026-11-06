# P1 base qualification (tf_alibi vs tf_ref): pre-registration

Written and committed before any qualification run was trained (2026-11-06). Approved by the user; designed in the Sol
round after the encoder screen.

## Why

The encoder screen (`reports/p1_encoder_screen.md`) failed its pre-registered B1 margin. Its failure clause therefore
stands: nothing is adopted from it. Some screen results did motivate this study:

| comparison (encoder screen) | result |
|---|---|
| tf_alibi vs tf_ref, val full NLL | -0.16 nats per owned event |
| normal M-pass | 0.50 vs 0.50 |
| scam M-pass | 0.94 vs 0.73 |

This study decides only one thing: whether tf_alibi replaces tf_ref as the transformer base. It does not satisfy P1. It
also does not retroactively pass the screen.

**Search history, disclosed:**
- 4 arms x 2 learning rates were searched on 2018 val (seed 20261104), plus a second seed for the challenger and B1.
- 2018 val has therefore been reused for selection, so it is not used here.

**Frozen candidates** (code at commit e21ee8a):
- configs `tf_alibi` and `tf_ref` in `src/lpw/p1/model.py`;
- lr 1e-3, the selected value for both;
- the current training recipe: 80-epoch cap, patience 6, staged stop recalibration.

B1 (`b1`, lr 1e-3) is trained alongside it and reported. Under SPEC amendment A1 it is not gated here.

## Untouched confirmation groups

`scripts/p1_qualify/make_conf_pilot.py` relabels a fixed 20% of each slice's training groups (seed 0) as split `conf`.

| slice | pilot | conf agents (normal / scam) | train agents left |
|---|---|---|---|
| 2018 | `data/pilot_2018_conf` | 3851 (3453 / 398) | 13603 |
| 2021 | `data/pilot_conf` | 1443 (394 / 1049) | 3599 |

How the conf groups are kept clean:
- No model in this study trains, early-stops, recalibrates or selects on them. Early stopping stays on the existing val split.
- Venues, hubs and groups are label-blind structure built over all agents before the split, exactly as for val and test.
- Profile normalisation, the fee pool and the role-parity classifier use the reduced train split.

**Disclosed overlap.** The 2018 conf groups are the same groups as the internal holdout of the anchor re-screen (same
seed and method). There they selected an adapter checkpoint on a different base model, which ended at step 0. They have
never been used for any encoder decision or by any model trained in this study. 2021 val and conf were not used by the
encoder screen.

## Procedure (`scripts/p1_qualify/qualify.py`, outputs under `runs/p1_qualify/`)

For each slice (2018, 2021), each arm (tf_alibi, tf_ref, b1) and each seed (20261104, 20261105):
1. Fit on the reduced train split.
2. Score exact per-prefix NLL on `conf`.
3. Generate 3 paired `conf` replicas (seed 7), then compute fidelity with the existing noise-floor procedure and role
   parity.

That is 12 fits in total. No runner-up substitution and no reselection. If a run crashes, it may be rerun unchanged.

## Gates (all 16 must pass for tf_alibi to qualify)

For each slice x seed cell (4 cells):

1. **L.** tf_alibi − tf_ref, exact conf full NLL per owned event, must be ≤ −0.03. The one-sided 95% upper bound of
   the paired whole-group bootstrap (1000 resamples, seed 0) must be < 0.
2. **G1.** tf_alibi's mean M-pass is not below tf_ref's, in each role.
3. **G2.** tf_alibi's mean per-replica |role-AUC delta| is ≤ 0.05.
4. **G3.** tf_alibi has zero capped streams.

**Reported, not gated:**
- Δ by prefix depth, for tf_alibi − tf_ref and tf_alibi − b1.
- The amended B1 criterion (SPEC A1). Non-inferiority means upper95 of tf_alibi − b1 ≤ +0.03. Generating better means
  higher mean M-pass than b1 in each role.
- The original B1 criterion: tf_alibi − b1 < 0.

## Outcomes

- **Pass:** tf_alibi becomes the transformer base. Next:
  1. Rebuild plan/progress on it and recheck the s-unknown likelihood.
  2. Run the drift consistency screen on that base. It gets its own pre-registration, with selection constrained by the
     likelihood budget.
- **Fail:** tf_ref stays the base.
- **Either way:** P1 exit is decided later on the full pipeline, with prior-mode generation at ≥ 70% M-pass per role
  and role parity on both slices.
- **Approval needed:** 3-seed confirmation and any run over 4 h need the user's OK.

## Compute

A JarvisLabs A30 (16 vCPU), with 4 fits run concurrently. Estimate: about 2 h and about $1.
