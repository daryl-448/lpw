# LPW on the 2018 slice: consolidated write-up (arc closed 2026-11-08)

This is the end of the 2018 arc of the Learned-Policy World (LPW, `SPEC.md`). It summarises what was built and
tested, and what the single sealed-test opening confirmed. It adds no new test analysis (§J10 of
`reports/p3_joint_addendum.md`). Every number is from a committed report, named in each section.

## 1. The question

Can a learned next-event policy, run as an agent population, generate synthetic AML-style data that is useful for
training real detectors? Illicit agents run the same policy, conditioned on a learned role, and are never written
from templates. Usefulness is judged on real labels: train on synthetic, test on real (TSTR). It is also judged by
criticism tests (the seam C2ST, the shortcut audit, profile-gain parity) after an observation operator applies the
dataset's own recording rules (SPEC §0-2, §6.8, §7.4).

## 2. The arc

| stage | what was done | outcome | source |
|---|---|---|---|
| P0 data | Two 12-month pilot slices of the HF Ethereum fraud dataset (2021 and 2018), with deterministic group-wise train / val / test splits (Louvain groups). 2018 was added because 2021 cannot support the confound-matched utility test. | 2018: 22,610 normal / 2,326 scam agents | `reports/p0_audit.md` |
| P1 policy | A next-event model (kind, time, censoring, counterparty, amount) with plan latents. Screens: anchor, pace, drift, encoder, shallow correction, prior decomposition. | **Stopped without meeting its exit criteria** (SPEC amendment A2). It beats B0 and B2 but not B1 on likelihood, and role-AUC parity is missed narrowly. Generator kct frozen for 2018 P3 feasibility, with B as control. | `reports/p1_exit.md` |
| P3a | A sampled population (roles, cohorts, activation, profiles, balances and plans, from frozen train-fitted models), uncoupled. Observed through O and compared on one shared node-feature extractor. | val: transfer SUPPORTED (kct D +0.064, ratio 0.948); seam, difficulty-parity and shortcut audits fail | `reports/p3a_summary.md`, `reports/p3a_readout.md`, `reports/p3a_sweep.md` |
| P3b | Shared-identity attachment on the frozen P3a streams: a conditional group prior, group-exclusive pools, and a learned gate + pointer (P3b-1) against an empirical-rate / uniform control (P3b-0) and a zero-sharing anchor. | val: learning advantage SUPPORTED (kct H +0.0146); incremental utility NOT TESTABLE (R +0.0016 < 0.020); STRUCT-only retention CONSISTENT (kct) / SUPPORTED (B); CORE FAIL | `reports/p3b_prereg.md`, `reports/p3b_readout.md` |
| Joint test | One fixed-sequence confirmation, frozen before any test output, rehearsed exactly on val and opened once with explicit user authorization | all three steps SUPPORTED (§3) | `reports/p3_joint_addendum.md`, `reports/p3_joint_rehearsal.md`, `reports/p3_joint_test.md` |

## 3. What the sealed test confirmed

The 2018 test split was opened once, on 2026-11-08, at authorization commit `165696f` (the only child of the freeze
commit `193d674`). It has 3,741 agents, 349 of them scam, in 1,874 groups. Inference is one-sided α = 0.05 per step,
with nominal 5% familywise control, conditional on the frozen Louvain whole-group bootstrap. Every verdict had
1000 / 1000 valid draws.

| step | claim | test | val | verdict |
|---|---|---|---|---|
| 1 | P3a transfer (kct, XGB, `full`) | D +0.070, lower +0.061; ratio 0.962 (TSTR 0.918 vs TRTR 0.934) | D +0.064, SUPPORTED | **SUPPORTED** |
| 2 | P3b learning advantage (kct, L): H = T₁ − T₀ | H +0.0140, lower +0.0080 | +0.0146, SUPPORTED | **SUPPORTED** |
| 3 | STRUCT-only retention (kct, P3b-1) | D_S +0.025, lower +0.0041; ratio 0.963 | +0.030, CONSISTENT | **SUPPORTED** |

**In the frozen wording:**
1. A detector trained only on the frozen synthetic population keeps ≥ 80% of the real-trained excess AUC on 2018
   test.
2. The learned attachment beats the restricted empirical-rate / uniform control.
3. At least 80% of the real STRUCT-only excess AUC is retained.

**Nominal results point the same way** (`reports/p3_joint_test.md` §2):
- the confound-removed `noconf` set: D +0.069;
- the clean subset under P0's leakage rule: D +0.077, H +0.0149, D_S +0.026. The predictions and full-test STRUCT
  are kept fixed and only agents are restricted, so this is not an inductive-graph claim;
- P3a giant-group removal, the B arm and the other detectors, all SUPPORTED;
- P3b giant-group H: positive point estimates, +0.012 to +0.016, with no interval;
- matched pairs: behavioural gain (full − confonly) +0.260 synthetic-trained vs +0.265 real-trained.

**AP is weaker than AUC.** Synthetic-trained AP stays below real-trained AP: kct 0.725 vs 0.774 on test. The
transfer claim is about AUC only.

## 4. What did not work, or was not tested

- **The synthetic world is detectably not real.** A classifier separates real from synthetic agents at seam C2ST
  0.88-0.92 on local features.
- **Synthetic worlds are harder than real.** Synthetic-to-synthetic AP is 0.09-0.11 below real-to-real (test).
- **The shortcut audit flags the balance and fee sentinels.** These are signals the synthetic data lacks.
- **New on test: the learned attachment flags STRUCT features in the shortcut audit.** kct P3b-1 flags four of them
  (for example `shared_frac` +0.101), and B P3b-1 three. At val it flagged none.
- **New on test: the daily component count fails the within-2× check** in every arm and attachment, the zero-sharing
  anchor included (2.17-2.33; val 1.59-1.70).
- **Profile-gain parity on AP flips with the sign of the near-zero real AP gain.** The check is uninformative.
- **Scam mode recall is low in the tabular view:** 0.49-0.56 on test. In encoder space it is high (0.98-0.99), and
  the two disagree.
- **Coupling fidelity fails.** The group-exclusive pools miss cross-group sharing: CORE FAIL in every cell.
- **Incremental structural transfer remains NOT TESTABLE.** The observed real increment, R +0.0123 (CI95 +0.0035 to
  +0.0216), is below the frozen 0.020 floor. NOT TESTABLE is not evidence of absence, and "learned structural signal
  transfers" is not claimed.
- **Step 2 is narrow.** The learned attachment mostly removes the control's damage: T₀ −0.0094, T₁ +0.0047, and T₁'s
  lower bound is −0.0052. It is not non-harm against zero sharing.
- **P1 never qualified** (A2). On held-out likelihood the frozen policy still trails the GRU baseline B1: shared NLL
  11.695 vs 11.539 / 11.590 on test (descriptive table, `reports/p3_joint_test.md` §4).
- **2021 is a failed stress case.** Scam M-pass is about 0 for every qualification model, so there is no cross-era
  claim.

## 5. Licensed and not licensed (§J10)

**Licensed.**
- A sampled learned-policy population supplies useful 2018 node-level scam-label training data, despite detectable
  distributional defects.
- The minimal learned identity attachment beats its restricted control.

**Not licensed:**
- faithful AML or laundering dynamics;
- P1 or P3 qualification;
- behavioural coupling;
- incremental structural utility;
- typology completeness;
- privacy guarantees from DCR;
- cross-era generalization.

## 6. Method notes worth keeping

- **Freeze, rehearse, then open once.**
  - Each confirmatory claim was pre-registered before its development readout: `p3a_prereg`, `p3b_prereg`.
  - The test was frozen in an addendum before any test output, with a fixed sequence, an exposure ledger and
    test-specific rules.
  - The exact job was rehearsed on val and had to reproduce the committed outputs. A loader that checks the
    authorization guarded the test split, and the results were published by atomic rename.
- **Rehearsal found a real nondeterminism.** `hour_entropy` depended on Polars' hash order. It moved about 1e-15
  between runs, and LGBM amplified that into AUC drifts of up to 0.009. It was made deterministic before the freeze
  and disclosed (`reports/p3_joint_rehearsal.md` R2).
- **The rehearsal reproduced the archive under the ledger's explained exceptions, not literally.**
  - XGB, the detector behind all three claims, was exact throughout.
  - The exceptions:
    - one P1 B1 argmax at 1.18e-4, against the 1e-4 criterion;
    - the `hour_entropy` diagnostic at ≤ 2.5e-5;
    - LGBM drift within 0.01, with one verdict string changed and the `ranking_spearman` swap.
  - The frozen job reproduced itself bitwise on every JSON output.
- **Reviews.** The pre-registrations, the joint addendum and the readout code were reviewed with sol-loop
  (gpt-6.1-sol) before the outputs they govern.
  - The joint entry point was reviewed after two development val rehearsals (r1, r2) and before the frozen rehearsal
    and the authorized test job.
  - The test readout and this write-up were audited by Sol against the published outputs.

## 7. After this arc

The 2018 holdout is spent and is never re-sealed. Any follow-up needs fresh confirmation data, and none is ready for
the matched claim:
- the HF source is the same dataset;
- 2017-07..2018-07 overlaps the slice;
- windows from 2019 on have an estimated 47-83 test pairs;
- an unopened 2021 holdout (62 pairs under P0's matcher) could carry only a separately pre-registered cross-era or
  fidelity question (§J10).

**Open research directions:**
- cross-group sharing pools (the CORE FAIL);
- the missing balance and fee signals (the shortcut sentinels);
- the STRUCT shortcut flags and the component-count failure that appeared on test;
- why the tabular and encoder-space scam coverage disagree.
