# P3b readout: shared-identity attachment on frozen P3a streams (2018, development evaluation on val)

- **Protocol.** `reports/p3b_prereg.md`, frozen at `c061ba7` before any fit. The readout code (`5167b9e`) was reviewed
  with sol-loop before any val output: 3 rounds, gpt-6.1-sol, approved. Findings fixed before the run: strength kept
  self-transfers; recovered-community summaries; cohort marginals; storage of every m_d and undefined day; full input
  hashes.
- **Run.** One complete val readout at `5167b9e`, clean tree, 2079 s. Every outcome is kept. Outputs:
  `runs/p3b/2018/readout.{json,txt}`, `readout_predictions.npz`, `readout_samples.npz` (sha256 in
  `readout_manifest.json`: json `88dcaffe…`, predictions `597125ca…`, samples `7ae89794…`). Code, inputs, kernel,
  group maps and environment are hashed inside `readout.json`.
- **Scope.** Real val is development data. The sealed test is untouched: the readout used the guarded train/val
  loader only. Worlds 0-2, arms kct (primary) and B (control); attachment arms P3b-0 (empirical-rate control),
  P3b-1 (learned gate + pointer) and the zero-sharing anchor.

## Headline

| claim (prereg §P9) | kct (primary) | B (control) | outcome |
|---|---|---|---|
| A. Incremental structural transfer: R, D | R +0.0016 [−0.0043, +0.0102]; D −0.0014 [−0.0094, +0.0071] | R +0.0016; D −0.0024 | **NOT TESTABLE** (R < 0.020) |
| B. Learning advantage H = T₁ − T₀ | **+0.0146** [+0.0044, +0.0243], one-sided lower +0.0056 | +0.0081 [−0.0010, +0.0146], lower +0.0004 | **SUPPORTED** (both arms) |
| Core coupling fidelity (8 items, d ≤ 2f) | FAIL (P3b-0 and P3b-1) | FAIL | **CORE FAIL** |

XGB, L feature set, 1000 whole-group bootstraps (1000 of 1000 valid).

1. **A is NOT TESTABLE, as predicted before freezing (§P7, A6: R_oof +0.003).** On real val, STRUCT adds nothing
   to the local features: R = +0.0016 for XGB, and between −0.011 and +0.003 across the detectors and the L / full
   bases. With no real increment to transfer, D cannot be tested.
2. **B is SUPPORTED, but read it narrowly.** T₀ = −0.0147 and T₁ = −0.0001:
   - the control attachment *damages* synthetic-trained detectors (L+S scores below L);
   - the learned attachment removes that damage, but adds no positive increment.

   So H measures "learned sharing is not misleading, and control sharing is". It does not mean "structural signal
   transfers". H > 0, with a one-sided lower bound > 0, holds for all 16 combinations of 2 arms, 4 detectors and
   2 bases (L, full). The weakest are B XGB on L (+0.0004) and B LGBM on full (+0.0002).
3. **Combined wording (§P9).** "Learned structural signal transfers" needs A and B; A is untestable, so it is not
   claimed. "The minimal coupling model meets the core fidelity targets" is not claimed: CORE FAIL.

**The one positive structural-transfer result is secondary: STRUCT-only retention** (P3a rules, TRTR(S) 0.636 ≥ 0.60).

| arm / attachment | TSTR(S) per world | mean | D_S [CI95] | one-sided lower | outcome |
|---|---|---|---|---|---|
| kct P3b-0 | 0.550, 0.520, 0.514 | 0.528 | −0.081 [−0.127, −0.041] | −0.121 | NOT SUPPORTED |
| kct P3b-1 | 0.586, 0.633, 0.697 | 0.639 | +0.030 [−0.007, +0.075] | −0.002 | CONSISTENT |
| B P3b-0 | 0.574, 0.584, 0.483 | 0.547 | −0.062 [−0.099, −0.027] | −0.093 | NOT SUPPORTED |
| B P3b-1 | 0.617, 0.665, 0.668 | 0.650 | +0.041 [+0.007, +0.082] | +0.011 | SUPPORTED |

A detector trained on synthetic STRUCT alone recovers the real STRUCT-only role signal when attachment is learned,
but not under the empirical-rate control. The learned kernel therefore puts real role information into who shares.
On real val, that information is redundant with the local features, which is why A cannot be tested.

## Utility detail

| quantity (XGB) | kct P3b-0 | kct P3b-1 | B P3b-0 | B P3b-1 | real (TRTR) |
|---|---|---|---|---|---|
| AUC L | 0.901 | 0.901 | 0.898 | 0.898 | 0.929 |
| AUC L+S | 0.886 | 0.900 | 0.889 | 0.897 | 0.931 |
| AUC full+S | 0.896 | 0.908 | 0.899 | 0.906 | 0.932 |
| AUC S | 0.528 | 0.639 | 0.547 | 0.650 | 0.636 |
| F1 full+S (grouped OOF threshold) | 0.670 | 0.663 | 0.665 | 0.660 | 0.760 |

**Secondary analyses.** None of these changes a verdict.

- **Full base** (full vs full+S), kct XGB: R −0.0002, T₀ −0.0134, T₁ −0.0017, D −0.0015, H +0.0118 (lower
  +0.0033).
- **Matched pairs** (348, the saved P3a rule): R −0.0008, T₀ −0.0235, T₁ −0.0065, H +0.0170 (kct, L).
  Behavioural gain full+S − confonly: TRTR +0.241; TSTR +0.221 (P3b-0) and +0.233 (P3b-1).
- **Giant val group** (group 84: 654 agents, 37 scam):
  - fixed scores without it: R +0.0024, T₁ −0.0029, H +0.0097;
  - graph removal (val STRUCT recomputed without it): R +0.0009, T₁ −0.0046, H +0.0083 (kct).

  The outcomes do not depend on the giant group.
- **Detector sensitivity.**
  - RF and LGBM agree with XGB: |T₁| ≤ 0.004, H +0.006 to +0.012.
  - **LR breaks on synthetic STRUCT.** T₀ −0.25 / −0.27 and T₁ −0.20 / −0.13 (kct / B). The linear detector
    over-weights the STRUCT columns, whose synthetic scale differs from the real scale (see the shortcut audit).
    Even so, H stays positive: +0.050 (kct), +0.144 (B).

## Criticism tests (prereg §P10; mean over worlds)

| test | kct P3b-0 | kct P3b-1 | B P3b-0 | B P3b-1 | rule | result |
|---|---|---|---|---|---|---|
| C2ST S-only, normal / scam | 0.793 / 0.869 | 0.778 / 0.885 | 0.794 / 0.871 | 0.789 / 0.876 | each ≤ 0.70, gap ≤ 0.05 | **fail** (both rules) |
| C2ST L+S | 0.955 / 0.960 | 0.940 / 0.963 | 0.958 / 0.962 | 0.948 / 0.963 | same | fail (inherited local seam) |
| C2ST full+S | 0.956 / 0.961 | 0.939 / 0.964 | 0.959 / 0.963 | 0.948 / 0.964 | same | fail |
| Difficulty parity, AP at val prevalence (XGB) | 0.666 vs 0.815 | 0.687 vs 0.815 | 0.651 vs 0.815 | 0.680 vs 0.815 | ±0.05 | **fail** (all 4 detectors) |
| Profile-gain parity, AUC | +0.004 vs +0.000 | +0.001 vs +0.000 | +0.005 vs +0.000 | +0.004 vs +0.000 | same sign, ≤ 0.05 | pass (uninformative, as in P3a) |
| Shortcut flags (\|Δ\| > 0.1) | 11 | **7** | 11 | **7** | none | fail |

- **The structural seam is real and role-asymmetric.** A classifier separates synthetic from real STRUCT at AUC
  0.78-0.79 for normal agents and 0.87-0.89 for scams.
  - The learned arm lowers the normal C2ST slightly but raises the scam one: the role gap widens from +0.076 to
    +0.107 (kct).
  - The real feature most out of reach is cross-group sharing: synthetic pools are group-exclusive by
    construction (§P10).
  - The L+S and full+S seams are dominated by P3a's local seam (P3a `local` C2ST 0.87 / 0.90).
- **The learned attachment removes the STRUCT shortcut flags.** Under P3b-0, four STRUCT features are flagged:
  `shared_frac` +0.117, `mean_cp_other_frac` +0.124, `max_cp_other_frac` +0.112 and `other_agents_frac` +0.112.
  Under P3b-1 none is. The 7 remaining flags are exactly P3a's: the five balance sentinels, `out_amt_med` and
  `fee_med`, which are inherited and untouched by design.
- **Difficulty parity.** The worlds remain harder than real, as in P3a (P3a: 0.684 vs 0.813). Learned
  attachment narrows the gap by about 0.02-0.03 AP.

## Coupling fidelity (prereg §P8)

**Core family.**
- n = 1860 per half view; 20 seeds; mean over seeds, then over worlds.
- The table shows the ratio d/f (pass ≤ 2) and, in brackets, the number of worlds passing.

| item | kct P3b-0 | kct P3b-1 | B P3b-0 | B P3b-1 |
|---|---|---|---|---|
| shared_frac, normal | 12.0 (0) | 6.4 (0) | 12.0 (0) | 5.6 (0) |
| shared_frac, scam | 2.5 (0) | **1.5 (3)** | 2.4 (0) | **1.5 (3)** |
| other_agents_frac, normal | 7.7 (0) | 10.9 (0) | 7.8 (0) | 10.2 (0) |
| other_agents_frac, scam | 2.9 (0) | 2.8 (1) | 3.0 (0) | 2.8 (1) |
| sharing_component_frac, normal | 5.0 (0) | 4.7 (0) | 5.0 (0) | 4.6 (0) |
| sharing_component_frac, scam | 2.8 (0) | 2.8 (0) | 2.8 (0) | 2.7 (0) |
| log1p node degree | 2.0 (1) | **1.6 (3)** | 2.2 (0) | **1.4 (3)** |
| log1p recovered community | 7.5 (0) | 5.3 (0) | 7.7 (0) | 5.1 (0) |
| verdict | CORE FAIL | CORE FAIL | CORE FAIL | CORE FAIL |

- **P3b-1 moves the model closer to real on 6 of 8 items.** The P3b-1 − P3b-0 distance is negative for the
  shared fraction (normal −0.081, scam −0.026), the components, node degree and community size (−0.18).
- **Two items pass.** The scam sharing share and node degree pass in all 3 worlds under P3b-1.
- **It is still far from passing overall.**
  - The normal-agent sharing share is about 6× the noise floor.
  - `other_agents_frac`, the breadth of who an agent shares with, gets *worse* for normal agents.
  - The component reach does not move.

  All three are the signature of group-exclusive pools.
- **In real val, sharing crosses group boundaries.** 23% of the shared-node incidence in the val graph is
  cross-group, and synthetic worlds have 0% by construction.
- **Synthetic communities are too concentrated.** Louvain on the full real val graph recovers 1972 communities,
  with max 76 agents (2.0%) and an effective count of 222. Synthetic hold views under P3b-1 (kct, world 0) recover
  max 243 (6.5%), with an effective count of 97. The template prior plants large coherent groups: the assigned
  max is 487, which is 13% of the world-0 hold, while real val's agents fragment more in their own eligible graph.
- **Full-view descriptive W1 distances** (val vs world-0 hold, kct):
  - `shared_frac` normal: anchor 0.311, P3b-0 0.094, P3b-1 0.055;
  - recovered community: 1.25, 0.63 and 0.50.

  Cohort and age-tercile marginals are in `readout.json` (`core_full_view`).

**SPEC §7.3 daily topology** (descriptive).
- Every metric is defined on ≥ 88% of common days, and every discrepancy is within 2× the A/B scale, with venues
  included and excluded.
- Ratios with venues: GCC 1.0, components 1.6-1.7, max k-core 0.63-0.65, assortativity 0.80-0.82, transitivity
  0.51-0.56.
- **With venues, the zero-sharing anchor scores the same as both attachment arms** to two decimals. Without
  venues, only the max k-core moves (anchor 0.62 vs P3b-1 0.76, kct). Daily-sampled topology (median m_d ≈ 25 agents
  per day) is dominated by venue and periphery stars, so it barely sees agent-level sharing. This "pass" says almost
  nothing about coupling.

**Clauset fits** (descriptive; α, xmin, GoF p).

| quantity | real val | synthetic hold, P3b-1 (kct, 3 worlds) | anchor (kct) |
|---|---|---|---|
| unique degree | 1.92, 2, 0.00 (lognormal favoured, LR −5.3) | 2.26 / 2.02 / 2.33; xmin 15 / 2 / 15 | 2.27-2.30, xmin 14-16 |
| transaction degree | 2.16, 24, **0.79** | 1.97-2.06, xmin 1-3, p 0.00 | 1.94-1.95, xmin 3 |
| strength (ETH) | 1.64, 34.2, 0.00 | 1.44-1.52, xmin 2.4-3.0 | 1.45-1.53 |
| amount (ETH) | 1.67, 13.5, 0.00 | 1.32-1.50, xmin 1.2-30.5 | same |

- The real transaction-degree tail fits a power law above 24. Synthetic tails are heavier in value (α 1.3-1.5 vs
  1.6-1.7) and start much lower.
- Attachment barely moves these whole-window distributions: P3b-1 ≈ anchor. They come from P3a's local streams.

**Coupling diagnostics** (val).
- Agent↔agent transactions are 0.34% of val transactions and 0.71% of agent events (synthetic 0%).
- Same-second co-creation: 0.03%.
- Real first touches with a non-empty choice set: 0.83 (synthetic train worlds 0.37-0.54, as noted before freezing).

## Reading

1. **Utility.** On this data, structure does not help a role detector beyond the local stream features. The
   primary utility question is NOT TESTABLE by the pre-registered floor, which is the outcome predicted at freeze.
   No feature search, GNN or other rescue follows (§P7).
2. **Learning.** What the learned attachment demonstrably adds:
   - it does not mislead the detector (H SUPPORTED, from T₀ < 0 to T₁ ≈ 0);
   - it carries the real STRUCT-only role signal (D_S CONSISTENT for kct, SUPPORTED for B);
   - it removes the STRUCT shortcut flags;
   - it moves 6 of 8 coupling items toward real.
3. **Fidelity.** The coupling model fails its own fidelity targets. The failure is the predicted one: real sharing
   crosses groups (§P10, A4), and group-exclusive pools with large planted templates cannot reproduce it. The next
   structural model would need cross-group pools. That is a design change, not a tuning step.
4. **Seam.** The structural seam is role-asymmetric: scams are easier to tell from real by their STRUCT features.
   Its scale (0.78-0.89) is smaller than the inherited local seam (0.87-0.96).

## Next (prereg §P12)

The joint sealed-test addendum for P3a and P3b comes next. It covers:
- P3a kct primary;
- both P3b attachment arms, whatever their development outcome;
- every model, tape and attached world unchanged;
- test STRUCT on the test split's own graph.

The test is opened once, only after the user authorizes it.
