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
| claims audit (§J10) | Sol, gpt-6.1-sol xhigh. Round 1 findings A1-A13 are all corrected in this version. |

**Where the val column comes from:**
- the committed val readouts, `runs/p3a/2018/readout.json` and `runs/p3b/2018/readout.json`;
- for sections that are new in the joint job, the frozen val rehearsal (`runs/p3_joint/2018/rehearsal_val_final`).
  Rehearsal values are marked ʳ.

Test owns every confirmation verdict. Val is shown alongside it and is never pooled with it (§J3).

## 1. Confirmation hierarchy (fixed sequence)

**Inference.** One-sided α = 0.05 per step, with nominal 5% familywise control, conditional on the frozen Louvain
whole-group bootstrap: 1000 seed-0 draws over the test evaluation groups.
- The three hierarchy contrasts each had 1000 / 1000 valid draws, with at least 156 scams per draw.
- The clean-subset contrasts (§2) also had 1000 / 1000 valid draws, with at least 137 scams per draw.
- The intervals are conditional on the evaluation set; they are not guaranteed population coverage. Cross-group
  sharing leaves dependence between groups, the models are fixed, and only three worlds were drawn.

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

**Clean-subset estimand (§J4.3).**
- The P0 rule excludes 932 test agents, keeping 2,539 normal / 270 scam. That reproduces P0's historical test counts
  exactly. Every one of the 1,874 groups keeps at least one clean member.
- The scores and the full-test STRUCT features are kept fixed; agents are only restricted. No transaction is
  deleted and nothing is refitted.
- So the clean results are a restriction of the same fixed predictions. They are not an inductive-graph claim.

### 2.1 P3a utility and transfer (XGB unless stated)

| analysis | kct test | kct val | B test | B val |
|---|---|---|---|---|
| TRTR `full`: AUC / AP / F1 (shared by both arms) | 0.934 / 0.774 / 0.707 | 0.932 / 0.813 / 0.757 | | |
| TSTR `full`, mean over worlds: AUC / AP / F1 | 0.918 / 0.725 / 0.670 | 0.909 / 0.760 / 0.712 | 0.917 / 0.717 / 0.667 | 0.908 / 0.744 / 0.690 |
| `full` contrast (B arm; kct is step 1) | | | D +0.070, lower +0.060, SUPPORTED | D +0.063, lower +0.046, SUPPORTED |
| `noconf` contrast (the SPEC §8 confound-removed set; disclosed, never swapped in) | D +0.069, lower +0.060, ratio 0.961, SUPPORTED | D +0.057, lower +0.038, SUPPORTED | D +0.062, lower +0.053, SUPPORTED | D +0.055, lower +0.037, SUPPORTED |
| clean subset, `full`: TRTR AUC; D | TRTR 0.951; D +0.077, lower +0.068, SUPPORTED | 0.950ʳ; +0.070, lower +0.057ʳ, SUPPORTED | TRTR 0.951; D +0.075, lower +0.066, SUPPORTED | 0.950ʳ; +0.066, lower +0.052ʳ, SUPPORTED |
| clean subset, `noconf` | D +0.074, lower +0.064, SUPPORTED | +0.063, lower +0.050ʳ, SUPPORTED | D +0.067, lower +0.057, SUPPORTED | +0.060, lower +0.047ʳ, SUPPORTED |
| giant group removed, fixed scores, `full` | D +0.075, lower +0.067, SUPPORTED | +0.075, lower +0.062ʳ, SUPPORTED | D +0.074, lower +0.066, SUPPORTED | +0.070, lower +0.057ʳ, SUPPORTED |
| giant group removed, `noconf` | D +0.073, lower +0.065, SUPPORTED | +0.067ʳ, SUPPORTED | D +0.065, lower +0.057, SUPPORTED | +0.063ʳ, SUPPORTED |
| capped streams removed, `full` | D +0.069, lower +0.060, SUPPORTED | +0.063, SUPPORTED | D +0.069, lower +0.059, SUPPORTED | +0.061, SUPPORTED |
| LR / RF / LGBM, `full`: D | +0.043 / +0.067 / +0.070, all SUPPORTED | +0.041 / +0.062 / +0.064, all SUPPORTED | +0.042 / +0.063 / +0.069, all SUPPORTED | +0.038 / +0.056 / +0.059, all SUPPORTED |
| matched pairs (349 test, 348 val): TSTR / TRTR AUC, `full` | 0.864 / 0.889 | 0.876 / 0.906 | 0.861 / 0.889 | 0.871 / 0.906 |
| matched behavioural gain (full − confonly): TSTR / TRTR | +0.260 / +0.265 | +0.240 / +0.241 | +0.253 / +0.265 | +0.229 / +0.241 |
| clean + matched (266 test, 257ʳ val): gain TSTR / TRTR | +0.240 / +0.234 | +0.238 / +0.224ʳ | +0.227 / +0.234 | +0.222 / +0.224ʳ |
| owned-only real observation: TSTR per world vs TRTR | 0.920 / 0.912 / 0.917 vs 0.930 | 0.909 / 0.915 / 0.908 vs 0.931 | 0.910 / 0.914 / 0.916 vs 0.930 | 0.905 / 0.914 / 0.910 vs 0.931 |

AP and F1 are lower on test than at val, for both TRTR and TSTR. The synthetic-trained detector stays below the
real-trained one on AP (−0.048 kct, −0.057 B) and on F1 (−0.038 / −0.040), with the gaps computed from unrounded
values. The transfer claim is an AUC claim; it
says nothing about AP parity.

### 2.2 P3b, outside the sequence (L base unless stated)

| analysis | kct test | kct val | B test | B val |
|---|---|---|---|---|
| **claim A** (incremental utility; R ≥ 0.020 floor), XGB | R +0.0123, CI95 [+0.0035, +0.0216], < 0.020: **NOT TESTABLE**; D −0.0052 | R +0.0016, NOT TESTABLE | same R: **NOT TESTABLE**; D +0.0012 | NOT TESTABLE |
| H, B arm, XGB | | | H +0.0127, lower +0.0062, SUPPORTED | H +0.0081, lower +0.0004, SUPPORTED |
| D_S, B arm (P3b-1), XGB | | | D_S +0.040, lower +0.023, SUPPORTED | D_S +0.041, lower +0.011, SUPPORTED |
| D_S, P3b-0 (control), XGB | −0.071, NOT SUPPORTED | −0.081, NOT SUPPORTED | −0.037, NOT SUPPORTED | −0.062, NOT SUPPORTED |
| full base: H, XGB | +0.0109, SUPPORTED | +0.0118, SUPPORTED | +0.0121, SUPPORTED | +0.0070, SUPPORTED |
| RF: T₀ / T₁ / H | −0.0091 / +0.0008 / +0.0099, SUPPORTED | −0.0036 / +0.0019 / +0.0055, SUPPORTED | −0.0082 / −0.0011 / +0.0071, SUPPORTED | −0.0066 / −0.0006 / +0.0060, SUPPORTED |
| LGBM: T₀ / T₁ / H | −0.0042 / +0.0088 / +0.0130, SUPPORTED | −0.0104 / +0.0007 / +0.0111, SUPPORTED | +0.0029 / +0.0152 / +0.0122, SUPPORTED | −0.0109 / +0.0009 / +0.0118, SUPPORTED |
| **LR: T₀ / T₁ / H** | **−0.236 / −0.189** / +0.047, SUPPORTED | −0.250 / −0.200 / +0.050, SUPPORTED | **−0.261 / −0.122** / +0.139, SUPPORTED | −0.272 / −0.128 / +0.144, SUPPORTED |
| clean subset: H (L), XGB | +0.0149, lower +0.0070, SUPPORTED | +0.0132, lower +0.0051ʳ, SUPPORTED | +0.0126, lower +0.0045, SUPPORTED | +0.0091, lower +0.0004ʳ, SUPPORTED |
| clean subset: H (full), XGB | +0.0110, lower +0.0037, SUPPORTED | +0.0108, lower +0.0045ʳ, SUPPORTED | +0.0125, lower +0.0052, SUPPORTED | +0.0074, lower +0.0012ʳ, SUPPORTED |
| clean subset: D_S (P3b-1), XGB | +0.026, lower +0.009, SUPPORTED | +0.022, lower −0.002ʳ, CONSISTENT | +0.038, lower +0.025, SUPPORTED | +0.032, lower +0.010ʳ, SUPPORTED |
| giant group, fixed scores: H (point estimate only) | +0.0156 | +0.0097 | +0.0125 | +0.0055 |
| giant group, graph removed: H (point estimate only) | +0.0156 | +0.0083 | +0.0134 | +0.0047 |
| matched (349 / 348 pairs): T₀ / T₁ / H (point estimates) | 0.0000 / +0.0168 / +0.0168 | −0.0235 / −0.0065 / +0.0170 | +0.0067 / +0.0221 / +0.0153 | −0.0136 / −0.0045 / +0.0092 |
| matched behavioural gain (full+S − confonly): TRTR / TSTR P3b-0 / P3b-1 | +0.288 / +0.256 / +0.272 | +0.241 / +0.221 / +0.233 | +0.288 / +0.255 / +0.273 | +0.241 / +0.215 / +0.223 |
| clean + matched (266 / 257ʳ pairs): H, L / full (point estimates) | +0.0157 / +0.0135 | +0.0157 / +0.0113ʳ | +0.0138 / +0.0156 | +0.0088 / +0.0081ʳ |

**Claim A stays NOT TESTABLE.** R rose from +0.0016 at val to +0.0123 on test, but it is still below the frozen
floor. NOT TESTABLE is not evidence of absence (§J3). It means the incremental structural transfer could not be
tested, because the observed real increment is below the floor. Claim A is nominal in any case, so no "learned
structural signal transfers" claim follows.

**LR.** H is positive because P3b-1 damages LR less than P3b-0 does. Both attachments cut LR's TSTR(L+S) far below
TSTR(L): T₀ −0.236 / −0.261 and T₁ −0.189 / −0.122. This is a large structural harm to the LR detector, at test and at
val alike. LR H is a learning-advantage reading only, not non-harm. No analysis of why was added (§J8.3).

## 3. Failures and defects (reported in full)

### 3.1 Unchanged from val

- **The C2ST seam fails.** The real-vs-synthetic classifier separates the two easily:
  - P3a local features, normal / scam: kct 0.884 / 0.914, B 0.882 / 0.918 (val 0.871 / 0.896, 0.869 / 0.902);
  - P3b, S only: 0.80 / 0.84-0.85 (val 0.78-0.79 / 0.87-0.89);
  - P3b, L+S: 0.94-0.96 / 0.96.
- **Difficulty parity fails.** Synthetic AP minus real AP at matched prevalence: kct −0.090, B −0.109 (val −0.129,
  −0.147).
- **P3a shortcut audit:** six flags in both arms: the balance and fee sentinels `sweep_out_share`, `fee_med`,
  `ends_near_zero`, `end_balance`, `net_flow` and `min_balance` (|Δ| 0.12-0.21). Val had seven: the same six plus
  `out_amt_med`.
- **P3b core: FAIL in all four arm × attachment cells.** An item passes if the synthetic-to-real-half Wasserstein
  distance d is at most twice the real-half-to-half distance f.
  - The group-exclusive pools miss cross-group sharing.
  - For kct P3b-1, normal `shared_frac` has d 0.109 against f 0.014 (val 0.094 vs 0.015), and
    `log1p_community_agents` has d 0.389 against f 0.072.
  - P3b-1 minus P3b-0 (kct) is smaller on six of the eight items: `shared_frac` (−0.081 normal / −0.021 scam),
    `sharing_component_frac` (−0.005 / −0.002), `log1p_node_degree` (−0.002) and `log1p_community_agents` (−0.215).
  - It is larger, i.e. worse, on `other_agents_frac` (+0.0011 normal / +0.00002 scam).
- **Scam mode recall is low.** Tabular mode coverage: scam recall 0.49-0.56 (val 0.57-0.58), normal P / R ≈ 0.85.
  The encoder-space coverage, supplementary, is high: scam P / R 0.95-0.98 / 0.98-0.99, normal 0.975 / 0.98-0.99.
- **DCR** (5th-percentile synthetic / holdout ratio): normal 7.9-9.3, scam 3.3-4.0. That is, synthetic records sit
  further from real train than real holdout records do. There are 0 exact copies. This is not a privacy guarantee.

### 3.2 New on test, or worse than at val

- **P3b shortcut audit: the learned attachment now flags STRUCT features.**

  | cell | flags: test | flags: val | STRUCT features flagged on test |
  |---|---|---|---|
  | kct P3b-1 | 10 | 7 | `shared_frac` +0.101, `mean_cp_other_frac` +0.116, `max_cp_other_frac` +0.115, `other_agents_frac` +0.114 |
  | B P3b-1 | 9 | 7 | `mean_cp_other_frac` +0.114, `max_cp_other_frac` +0.113, `other_agents_frac` +0.112 |
  | kct P3b-0 | 10 | 11 | the same four, +0.133 to +0.147 |
  | B P3b-0 | 10 | 11 | the same four, +0.129 to +0.143 |

  - At val, P3b-1 flagged none of these STRUCT features in either arm, and P3b-0 flagged all four.
  - On test, their single-feature role AUCs on P3b-1 synthetic data exceed real by more than the 0.1 flag threshold.
  - The other flags are the six P3a sentinels; `out_amt_med` is no longer flagged.
- **Daily topology: the component count fails the within-2× check everywhere.** This covers every arm and
  attachment, the zero-sharing anchor included, with and without venues.
  - Test ratios are 2.17-2.33; val ratios were 1.59-1.70 and passed.
  - Example: kct P3b-1 with venues, 2.198 (val 1.585).
  - The other daily items (GCC ratio, k-core, core fraction, assortativity, transitivity) stay within 2×.
- **Profile-gain parity on AP flips with the sign of the real AP gain,** which is near 0 on both splits: real
  −0.0030 on test vs +0.0064 at val for P3a, and −0.0009 vs +0.0032 for P3b.
  - Now failing: P3a kct and P3b kct (both attachments), and B P3b-1.
  - Now passing: P3a B and P3b B P3b-0.
  - Profile-gain parity on AUC passes everywhere. As at val, the check is uninformative: the profile adds about 0 AP
    on real data.

## 4. P1: descriptive held-out likelihood (test)

*Held-out descriptive likelihood of frozen instruments; P1 remains unqualified under A2. No P1 exit decision is
reopened.*

Values are nats per owned event, with val in brackets.
- Test: 66,013 prefixes, 62,097 owned, 3,916 censored; scam 7,450 / 6,990 / 460.
- Val: 63,360 / 59,406 / 3,954; scam 8,935 / 8,477 / 458.

| row | full NLL, all | shared, all | full NLL, scam | shared, scam | kind error, all |
|---|---|---|---|---|---|
| kct | 12.743 (13.128) | 11.695 (11.503) | 13.189 (12.548) | 11.623 (10.670) | 0.186 (0.207) |
| B | 12.743 (13.128) | 11.695 (11.503) | 13.189 (12.548) | 11.623 (10.670) | 0.186 (0.207) |
| B0 | 16.736 (16.230) | 14.498 (13.675) | 15.128 (14.340) | 12.857 (11.772) | 0.511 (0.421) |
| B1 s20261104 | 12.561 (12.902) | 11.539 (11.317) | 13.068 (12.387) | 11.520 (10.536) | 0.175 (0.181) |
| B1 s20261105 | 12.614 (12.943) | 11.590 (11.359) | 13.086 (12.356) | 11.550 (10.538) | 0.174 (0.182) |
| B2 | N/A | 12.916 (12.684) | N/A | 12.990 (11.863) | 0.188 (0.211) |

- **kct = B by construction.** Both use `plan_inference = False`; the equality is asserted, not an independent
  replication.
- **The ordering is the same as at val.** On the shared NLL: B1 (encoder) < kct = B < B2 < B0. On the full NLL, where
  it is defined: B1 < kct = B < B0.
- **Per-field and per-panel values** are in `runs/p3_joint/2018/test/p1_nll.json`.
- **Rehearsal note on the B1 s20261105 row** (rehearsal ledger R1). At val, one scam argmax out of 8,477 differed
  from the archive: `kind_error` |Δ| 1.18e-4, which exceeds §J7's 1e-4 criterion. The cause is the archived run's
  fp16 GPU arithmetic. Today's numerics are deterministic.
- **No significance marks** are given.

## 5. Deviations and disclosures

1. **`hour_entropy` made deterministic before the freeze** (rehearsal ledger R2). Polars returned `value_counts()` in
   hash order, so the entropy moved by about 1e-15 between runs.
   - **Fix:** the counts are sorted first. The definition is unchanged; this is a §J7 allowed, determinism-only fix.
   - **Effect at val:** XGB exact everywhere; LGBM ≤ 0.0087; the shortcut-AUC diagnostic ≤ 2.5e-5.
   - **The archived val numbers stay the development record.**
2. **The rehearsal against the archive did not meet every §J7 criterion literally.** The frozen rehearsal
   `rehearsal_val_final.compare.json` reports these exceptions. All were explained, and the detector tolerances were
   fixed in the ledger before the freeze (R3):
   - P1: `ok: false`, the B1 s20261105 scam `kind_error` 1.18e-4 > 1e-4 (R1; see §4).
   - Strict class: the 22 / 44 `shortcut/hour_entropy` single-feature AUCs, |Δ| ≤ 2.5e-5, with no flag changes (R2).
   - Detector rule:
     - RF: 1e-9;
     - LR: 1e-9 on scalars and 1e-5 on predictions;
     - LGBM: 0.01 on AUC, AP, ratio and contrast values;
     - `ranking_spearman`: any value, since it is a function of the LGBM AUCs (observed 0.2, one rank swap);
     - LGBM verdict strings may differ (observed: B `incremental_full/lgbm/learning` CONSISTENT vs SUPPORTED).
   - Reproducibility: the frozen rehearsal reproduced r2 bitwise on every JSON output. The RF prediction arrays
     differed by up to 2.2e-16 (RF averages its trees in thread order). For that, the `--against` array comparison
     was aligned with the ledger's stated 1e-9 criterion.
3. **Review chronology.**
   - The entry-point code was reviewed with sol-loop (2 rounds, approved) after rehearsals r1 and r2 had run.
   - r2's P3b stage used the pre-review readout.
   - The frozen rehearsal and the authorized test job ran on the reviewed code.
   - The review-driven additions (S1-S8) all stay within §J4 and §J7:
     - refusal of artifact-root overrides in test mode;
     - NOT TESTABLE handling for fewer than 200 pairs and for empty or one-role subsets;
     - the INVALID headline format;
     - `min_scam_per_draw`;
     - matched AUCs in P3b.
   - None of the special cases arose on test: matching gave 349 pairs and every draw was valid.
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
- incremental structural utility (claim A is NOT TESTABLE, which is not evidence of absence);
- typology completeness;
- privacy guarantees from DCR;
- cross-era generalization.

The full inventory (§J5) is in `runs/p3_joint/2018/test/`: the P3a and P3b readouts, predictions, samples, pairs,
P1 table, encoder coverage, clean-subset ids and job record. The output manifest hashes every file.
