# P1 pace-shrinkage experiment: pre-registration

Written and committed before any arm was fitted or generated (Sol design and review rounds, 2026-11-06).
Diagnosis and earlier results: `runs/p1_diag/drift_results.md` (local; runs/ is not tracked).

## Hypothesis

Free-running generation is too slow for heavy plans because the policy infers pace from its own (slightly slow, and
amount-distorted) generated history and follows it with elasticity ~0.4 (depth 3-9) to ~0.9 (deep). Shrinking the
history-inferred mean log gap toward a history-free reference law, on known-plan BELOW rows with plan >= 10 and
owned depth >= 1, should recover heavy-plan pacing without training on generated histories.

## Arms (all from `runs/p1_r0/2018/lpw_plan_prog_rp`, 2018 slice, one training seed)

| arm | gamma | construction |
|---|---|---|
| control `lpw_pace_g1.00` | 1 | exact bypass; known-state `prog_dorm` refit (4000 steps, lr 1e-2, +-8) |
| `lpw_pace_g0.75` | 0.75 | shrinkage; then the same `prog_dorm` refit |
| `lpw_pace_g0.50` | 0.5 | shrinkage; then the same `prog_dorm` refit |

The reference (`lpw.p1.pace.PaceRef`) is fitted once, on real training prefixes with the real plan, and shared by the
arms; the fit must converge (max |gradient| <= 1e-6) or the experiment stops. Fixed inputs shared by all arms:
generation seed 7, five replicas, oracle = real val plans, prior = `PlanPrior` (default, not exposure-aware) drawn
with `numpy.random.default_rng(101)` (saved per replica as `.plans.npz`).

## Gates (every one must pass; control = the gamma = 1 arm)

Likelihood
- L1 s-unknown val terms (kind, time, censoring, counterparty, amount per row) bit-identical to the base checkpoint
  on every exact val prefix (the whole split; the checked prefix and batch counts are recorded in `pace.json`), so
  the reportable s-unknown NLL is unchanged.
- L2 known-plan full conditional val NLL (oracle-conditioned; not reportable): arm - control <= 0.02 nats/prefix,
  overall, normal and scam.

Oracle screen (real plans; diagnostic of pace, independent of the prior)
- O1 primary pace score (mean W1 of exposure-normalized time to events 10 and 30, normal plans 30-100 and 100-300,
  non-reachers at 1): relative reduction vs control >= 0.25 and the paired whole-group bootstrap 95% lower bound of
  the improvement > 0.
- O2 mechanism: calendar-frozen history-stretch elasticity (normal, depths 3 and 9, both stretch factors, fixed real
  prefixes): relative reduction vs control >= 0.20.
- O3 no capped streams; P(n >= 1000) normal <= max(2 x real, 1.5 x control).
- O4 endpoint pileup: per role (normal, scam) and window (last 1 / 7 / 30 days of the window), with A, C, Rl the
  arm, control and real event shares (A and C are means over the five replicas), H = max(C, Rl), SD_R the
  whole-group bootstrap SD of the real share, and s_A, s_C the SDs (ddof 1) of the per-replica shares, both of:
  (a) catastrophe guard: A <= H + 3 SD_R;
  (b) control-spread bound: A <= H + max(3 sqrt(s_A^2 / 5 + s_C^2 / 5), 0.10 H). The 0.10 H floor keeps a
      statistically detectable but practically negligible change (tiny replica spread) from failing the gate.
  Every value must be finite and each side needs >= 2 replicas, otherwise the gate fails.
- O5 full-length M-pass per role >= control - 0.10.

Prior gate (only arms passing L and O, plus the control)
- P1 P(n >= 2) normal >= 0.767 and P(n >= 5) normal >= 0.401.
- P2 gain retention: prior-mode relative pace-score reduction >= 2/3 of the arm's oracle reduction, with the
  bootstrap 95% lower bound of the improvement > 0.
- P3 = O3, P4 = O4 in prior mode.
- P5 full-length M-pass per role >= control - 0.10; plan compliance (normal) >= control - 0.05.

Every gate fails on a non-finite input.

Reported, not gated: normal >= 300 and scam >= 30 W1s and under-plan rates (sparse), event totals per role,
same-second share, events per active day, cut-k M-pass.

## Selection and what follows

The selected arm is the largest gamma passing every gate (0.75 before 0.5); if none passes, pace shrinkage is
rejected. Gates are evaluated one by one, each as pass, fail or missing (an input file or field absent or partially
written). Within a stage (L + O, then P), an arm is rejected if any gate fails, even when another gate's input is
missing; otherwise it is pending while any gate is missing. P gates are evaluated only for arms passing every L and
O gate. While a larger gamma is pending, the selection is pending (a smaller gamma is never selected early). Inputs are checked for
provenance: each score file must name the gamma = 1 arm as its control and map every label to that arm's own
generation (`gen_pace_g{gamma}_{mode}.parquet`, matching the `--save` path in its drift_exp output, which must name
`lpw_pace_g{gamma}`; the control's mapping is checked like every arm's), each arm's mechanism comparison must be
against `lpw_pace_g1.00`, and every generation (drift_exp metadata, score metadata, per-replica endpoint arrays) must
have exactly five replicas. A mismatch is an error, not a gate outcome. A selected arm is a candidate only: it goes to a three-training-seed confirmation against matched controls
(with the user's OK), and the P1 exit criteria (SPEC §8: likelihood wins over B0/B1/B2, >= 70% M-pass per role,
role-AUC parity within 0.05, both slices) are unchanged. The test split stays sealed. Selection is applied by
`scripts/p1_pace/paceselect.py`, which encodes exactly these thresholds (fixture tests: `tests/test_paceselect.py`).
The score and mechanism metrics are computed by `scripts/p1_pace/pacescore.py` and `scripts/p1_pace/feedback.py`;
all three are frozen in the same commit as this document, before any arm is fitted.

## Run order

1. `python -m lpw.p1.pace --pilot data/pilot_2018 --base runs/p1_r0/2018/lpw_plan_prog_rp --root runs/p1_r0/2018`
   (reference fit, then the three arms; `--ref-only` fits and caches the reference alone).
2. Oracle generation per arm: `python -m lpw.p1.drift_exp --model lpw_plan_prog_pace --ckpt ROOT/lpw_pace_g{g}
   --mode oracle --cuts 5,20,50,200 --out ROOT/drift_pace_g{g}_oracle.json --save ROOT/gen_pace_g{g}_oracle.parquet`.
3. `pacescore.py data/pilot_2018 oracle g1.00=... g0.75=... g0.50=... --json ROOT/pacescore_pace_oracle.json`.
4. `feedback.py ROOT/lpw_pace_g1.00 lpw_plan_prog_pace data/pilot_2018 - --sens-only`, then each arm with
   `--compare ROOT/lpw_pace_g1.00`.
5. Prior mode (steps 2-3 with `--mode prior`) for the control and every arm passing L and O.
6. `paceselect.py ROOT`.
