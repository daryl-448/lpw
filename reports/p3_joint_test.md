# P3 joint sealed-test readout (2018 test split, opened once)

The joint sealed-test addendum (`reports/p3_joint_addendum.md`) fixed this test before any test output existed.
The test split was opened once, by one complete job, with no retry, tuning or new analysis (§J8.3).

| item | value |
|---|---|
| freeze commit | `193d674451565e4b771835897c4533a423b3c24c` (manifest of 161 hashed files) |
| authorization | `165696f331cb0e1a3c577ac37df1ab76891e0be7`, the only child of the freeze commit; record `reports/p3_joint_authorization.md` (user, 2026-11-08) |
| job | `uv run python scripts/p3_joint/run.py --split test`. All four sealed checks passed in process; published atomically to `runs/p3_joint/2018/test/` (19 files, output manifest verified, no staging left) |
| wall time | 2971 s: clean 2, P1 97, encoder 11, P3a 673, P3b 2187 |
| attempts | 1 (no technical failure) |
| evaluation set | 3,741 test agents (349 scam); 1,874 Louvain groups; giant group 46 (585 agents, 33 scam); inverse size-Herfindahl 32.8 (val: 654 agents, 37 scam, 28.1) |

The val column comes from the committed val readouts (`runs/p3a/2018/readout.json`, `runs/p3b/2018/readout.json`).
For sections that are new in the joint job, it comes from the frozen val rehearsal
(`runs/p3_joint/2018/rehearsal_val_final`). Test owns every confirmation verdict. Val is shown alongside it and is
never pooled with it (§J3).

## 1. Confirmation hierarchy (fixed sequence)

Inference: one-sided α = 0.05 per step, with nominal 5% familywise control, conditional on the frozen Louvain
whole-group bootstrap. That is 1000 seed-0 draws over the test evaluation groups; every verdict below had
1000 / 1000 valid draws, with at least 156 scams per draw. The bootstrap intervals are conditional on the evaluation
set; they are not guaranteed population coverage. Cross-group sharing leaves dependence between groups, the models
are fixed, and only three worlds were drawn.

| step | claim | test | val | test verdict |
|---|---|---|---|---|
| 1 | **P3a transfer.** kct, XGB, `full`; D = (mean TSTR − 0.5) − 0.8 (TRTR − 0.5) | TSTR 0.918 (worlds 0.921 / 0.916 / 0.917); TRTR 0.934; ratio 0.962; **D +0.070**, one-sided lower bound **+0.061**, CI95 [+0.059, +0.082] | TSTR 0.909, TRTR 0.932, ratio 0.948, D +0.064, lower bound +0.045, SUPPORTED | **SUPPORTED** (reached) |
| 2 | **P3b learning advantage.** kct, XGB, L; H = T₁ − T₀ | T₀ −0.0094, T₁ +0.0047; **H +0.0140**, lower bound **+0.0080**, CI95 [+0.0069, +0.0228] | T₀ −0.0147, T₁ −0.0001, H +0.0146, lower bound +0.0056, SUPPORTED | **SUPPORTED** (reached) |
| 3 | **STRUCT-only retention.** kct, P3b-1, XGB, S; D_S = (mean TSTR(S) − 0.5) − 0.8 (TRTR(S) − 0.5) | TSTR(S) 0.645, TRTR(S) 0.651 (gate ≥ 0.60 met); ratio 0.963; **D_S +0.025**, lower bound **+0.0041**, CI95 [+0.0012, +0.0555] | D_S +0.030, lower bound −0.0022, CONSISTENT | **SUPPORTED** (reached) |

**All three claims are confirmed on 2018 test.** Each is licensed only in the wording fixed in §J3:
1. A detector trained only on the frozen synthetic population keeps ≥ 80% of the real-trained excess AUC on 2018
   test (kct, XGB, `full`). This is not the 96% point estimate, and not P3 qualification.
2. The learned attachment beats the restricted empirical-rate / uniform control (kct, L).
   - It is not non-harm against zero sharing: T₁'s one-sided lower bound is −0.0052.
   - It is not positive incremental utility, and it is not realistic coupling.
3. At least 80% of the real STRUCT-only excess AUC is retained (kct, P3b-1). This is not structural information
   beyond the local features.

Step 3 was CONSISTENT at val and is SUPPORTED on test. Its margin is narrow: the lower bound is +0.0041, and the
two-sided interval starts at +0.0012. The planning power for it was 0.42 (§J6).

## 2. Nominal results (outside the sequence; nominal verdicts only)

### 2.1 P3a transfer, variants and sensitivities (XGB unless stated)

| analysis | kct test | kct val | B test | B val |
|---|---|---|---|---|
| `noconf` contrast (the SPEC §8 confound-removed set; disclosed, never swapped in) | D +0.069, lower +0.060, ratio 0.961, SUPPORTED | D +0.057, lower +0.038, SUPPORTED | D +0.062, lower +0.053, ratio 0.945, SUPPORTED | D +0.055, lower +0.037, SUPPORTED |
| `full` contrast, B arm | | | D +0.070, lower +0.060, ratio 0.961, SUPPORTED | D +0.063, lower +0.046, SUPPORTED |
| clean subset, `full` (2,809 agents, 270 scam; P0 rule) | D +0.077, lower +0.068, SUPPORTED | D +0.070, SUPPORTED | D +0.075, lower +0.067, SUPPORTED | D +0.066, SUPPORTED |
| clean subset, `noconf` | D +0.074, lower +0.064, SUPPORTED | | D +0.067, lower +0.057, SUPPORTED | |
| giant group removed, fixed scores, `full` | D +0.075, lower +0.067, SUPPORTED | D +0.075 | D +0.075, lower +0.066, SUPPORTED | D +0.070 |
| giant group removed, `noconf` | D +0.073, SUPPORTED | | D +0.065, SUPPORTED | |
| capped streams removed | D +0.070, lower +0.060, SUPPORTED | | D +0.069, lower +0.059, SUPPORTED | |
| LR / RF / LGBM, `full` | D +0.043 / +0.067 / +0.070, all SUPPORTED | | D +0.042 / +0.063 / +0.069, all SUPPORTED | |
| matched (349 pairs; val 348): TSTR / TRTR AUC, `full` | 0.864 / 0.889 | | 0.861 / 0.889 | |
| matched behavioural gain (full − confonly), TSTR / TRTR | +0.260 / +0.265 | +0.240 (TSTR) | +0.253 / +0.265 | +0.229 (TSTR) |
| clean + matched (266 pairs; val 257): gain TSTR / TRTR | +0.240 / +0.234 | +0.238 (TSTR) | +0.227 / +0.234 | +0.222 (TSTR) |
| owned-only real observation: TSTR worlds / TRTR | 0.920 / 0.912 / 0.917 vs 0.930 | | 0.910 / 0.914 / 0.916 vs 0.930 | |

The clean subset drops 932 test agents and keeps 2,539 normal / 270 scam. That reproduces P0's historical test counts
exactly. Every one of the 1,874 groups keeps at least one clean member: the excluded agents sit in large groups (at
val, their median group size was 654).

### 2.2 P3b, outside the sequence (XGB, L base unless stated)

| analysis | kct test | kct val | B test | B val |
|---|---|---|---|---|
| **claim A** (incremental utility; R ≥ 0.020 floor) | R +0.0123 (lower +0.0045) < 0.020: **NOT TESTABLE**; D −0.0052 | R +0.0016, NOT TESTABLE | same R: **NOT TESTABLE**; D +0.0012 | NOT TESTABLE |
| H, B arm | | | H +0.0127, lower +0.0062, SUPPORTED | H +0.0081, SUPPORTED |
| D_S, B arm (P3b-1) | | | D_S +0.040, lower +0.023, SUPPORTED | D_S +0.041, SUPPORTED |
| D_S, P3b-0 (control) | D_S −0.071, NOT SUPPORTED | −0.081, NOT SUPPORTED | −0.037, NOT SUPPORTED | −0.062, NOT SUPPORTED |
| full base: H | +0.0109, SUPPORTED | +0.0118, SUPPORTED | +0.0121, SUPPORTED | +0.0070, SUPPORTED |
| LR / RF / LGBM: H | +0.048 / +0.010 / +0.013, all SUPPORTED | | +0.139 / +0.007 / +0.012, all SUPPORTED | |
| clean subset: H (L) | +0.0149, lower +0.0070, SUPPORTED | +0.0132, SUPPORTED | +0.0126, lower +0.0045, SUPPORTED | +0.0091, SUPPORTED |
| clean subset: H (full) | +0.0110, SUPPORTED | | +0.0125, SUPPORTED | |
| clean subset: D_S (P3b-1) | +0.026, lower +0.009, SUPPORTED | +0.022, CONSISTENT | +0.038, lower +0.025, SUPPORTED | +0.032, SUPPORTED |
| giant group, fixed scores: H | +0.0156 | +0.0097 | +0.0125 | +0.0055 |
| giant group, graph removed: H | +0.0156 | +0.0083 | +0.0134 | +0.0047 |
| matched (349 pairs): H | +0.0168 | +0.0170 | +0.0153 | +0.0092 |
| clean + matched (266 pairs): H (L / full) | +0.0157 / +0.0135 | | +0.0138 / +0.0156 | |

Claim A stays NOT TESTABLE. R rose from +0.0016 at val to +0.0123 on test, but it is still below the frozen floor.
Claim A is nominal in any case, so no "learned structural signal transfers" claim follows (§J3).

The LR H values (+0.048 / +0.139) are much larger than XGB's. They are reported as computed. No analysis of why was
added (§J8.3).

## 3. Failures and defects (reported in full)

These are the same defects as at val; the test opening does not change any of them.

- **The C2ST seam fails.** The real-vs-synthetic classifier separates the two easily:
  - P3a local features, normal / scam: kct 0.884 / 0.914, B 0.882 / 0.918 (val 0.871 / 0.896, 0.869 / 0.902);
  - P3b, S only: 0.80 / 0.84-0.85;
  - P3b, L+S: 0.94-0.96 / 0.96.
- **Difficulty parity fails.** Synthetic AP minus real AP at matched prevalence: kct −0.090, B −0.109 (val −0.129,
  −0.147).
- **The shortcut audit flags six features in both arms:** the balance and fee features `sweep_out_share`, `fee_med`,
  `ends_near_zero`, `end_balance`, `net_flow` and `min_balance` (|Δ| 0.13-0.21). These are the defect-1 sentinels,
  as at val.
- **P3b core: FAIL in all four arm × attachment cells.** The group-exclusive pools miss cross-group sharing. For
  An item passes if the synthetic-to-real-half Wasserstein distance d is at most twice the real-half-to-half
  distance f. For kct P3b-1, normal `shared_frac` has d 0.109 against f 0.014 (val 0.094 vs 0.015), and
  `log1p_community_agents` has d 0.389 against f 0.072. P3b-1 is closer than P3b-0 on the sharing items but still
  fails.
- **Scam mode recall is low.** Tabular mode coverage: scam recall 0.49-0.56 (val 0.57-0.58), normal P / R ≈ 0.85.
  The encoder-space coverage, supplementary, is high: scam P / R 0.95-0.98 / 0.98-0.99, normal 0.975 / 0.98-0.99.
- **DCR** (5th-percentile synthetic / holdout ratio): normal 7.9-9.3, scam 3.3-4.0. That is, synthetic records sit
  further from real train than real holdout records do. There are 0 exact copies. This is not a privacy guarantee.

## 4. P1: descriptive held-out likelihood (test)

*Held-out descriptive likelihood of frozen instruments; P1 remains unqualified under A2. No P1 exit decision is
reopened.*

Values are nats per owned event. Test: 66,013 prefixes, 62,097 owned, 3,916 censored; scam 7,450 / 6,990 / 460.
The val NLL is in brackets.

| row | full NLL, all | shared, all | full NLL, scam | shared, scam | kind error, all |
|---|---|---|---|---|---|
| kct | 12.743 (13.128) | 11.695 (11.503) | 13.189 (12.548) | 11.623 (10.670) | 0.186 |
| B | 12.743 (13.128) | 11.695 (11.503) | 13.189 (12.548) | 11.623 (10.670) | 0.186 |
| B0 | 16.736 (16.230) | 14.498 (13.675) | 15.128 (14.340) | 12.857 (11.772) | 0.511 |
| B1 s20261104 | 12.561 (12.902) | 11.540 (11.317) | 13.068 (12.387) | 11.520 (10.536) | 0.175 |
| B1 s20261105 | 12.614 (12.943) | 11.590 (11.359) | 13.086 (12.356) | 11.550 (10.538) | 0.174 |
| B2 | N/A | 12.916 (12.684) | N/A | 12.990 (11.863) | 0.188 |

- **kct = B by construction.** Both use `plan_inference = False`; the equality is asserted, not an independent
  replication.
- **The ordering is the same as at val.** On the shared NLL: B1 (encoder) < kct = B < B2 < B0. On the full NLL, where
  it is defined: B1 < kct = B < B0.
- **Per-field and per-panel values** are in `runs/p3_joint/2018/test/p1_nll.json`.
- **Rehearsal note.** The rehearsal ledger (R1) attaches one note to the B1 s20261105 row: at val, one scam argmax
  out of 8,477 differed from the archive. The cause is the archived run's fp16 GPU arithmetic. Today's numerics are
  deterministic.
- **No significance marks** are given.

## 5. Deviations and disclosures

1. **`hour_entropy` made deterministic before the freeze** (rehearsal ledger R2). Polars returned `value_counts()` in
   hash order, so the entropy moved by about 1e-15 between runs.
   - **Fix:** the counts are sorted first. The definition is unchanged; this is a §J7 allowed, determinism-only fix.
   - **Effect at val:** XGB exact everywhere; LGBM ≤ 0.0087, with one nominal LGBM verdict changed (B
     `incremental_full/lgbm/learning`); the shortcut-AUC diagnostic ≤ 2.5e-5.
   - **The archived val numbers stay the development record.**
2. **Rehearsal tolerances.**
   - **Detector rule:** fixed in the ledger before the freeze: RF 1e-9; LR 1e-9 on scalars and 1e-5 on predictions;
     LGBM 0.01.
   - **Comparison tool:** the `--against` array comparison was aligned with the stated 1e-9 criterion. RF averages
     its trees in thread order, which differs by 2.2e-16.
   - **Reproducibility:** the frozen rehearsal reproduced r2 on every JSON output bitwise.
3. **Review-driven additions** (sol-loop round 1, S1-S8), all within §J4 and §J7:
   - refusal of artifact-root overrides in test mode;
   - NOT TESTABLE handling for fewer than 200 pairs and for empty or one-role subsets;
   - the INVALID headline format;
   - `min_scam_per_draw`;
   - matched AUCs in P3b.

   None of the special cases arose on test: matching gave 349 pairs and every draw was valid.
4. **Labels in the text readout.** `runs/p3_joint/2018/test/p3b/readout.txt` keeps the val script's wording ("val",
   "real val") on test quantities. The JSON records `eval_split: "test"`.
5. **The exposure ledger (§J1) is updated by this opening.** 2018 test has been read once, by this job. It is spent
   and is never re-sealed (§J8.5).

## 6. What this does and does not license (§J10)

**Licensed.**
- A sampled learned-policy population supplies useful 2018 node-level scam-label training data, despite detectable
  distributional defects (steps 1 and 3).
- The minimal learned identity attachment beats its restricted control (step 2).

**Not licensed:**
- faithful AML or laundering dynamics;
- P1 or P3 qualification;
- behavioural coupling;
- incremental structural utility (claim A is NOT TESTABLE);
- typology completeness;
- privacy guarantees from DCR;
- cross-era generalization.

The full inventory (§J5) is in `runs/p3_joint/2018/test/`: the P3a and P3b readouts, predictions, samples, pairs,
P1 table, encoder coverage, clean-subset ids and job record. The output manifest hashes every file.
