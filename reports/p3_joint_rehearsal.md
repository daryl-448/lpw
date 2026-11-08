# P3 joint sealed-test job: val rehearsal ledger

Required by reports/p3_joint_addendum.md §J7. These rehearsals ran the exact job with the evaluation split set to val:
`scripts/p3_joint/run.py --split val`, under the guarded `("train", "val")` store. Each run is reconciled with the
committed val outputs by `scripts/p3_joint/compare_val.py`. Every discrepancy is listed below with its cause. No
test row was read.

## Runs

| run | code | wall time | purpose |
|---|---|---|---|
| r1 `runs/p3_joint/2018/rehearsal_val_r1` | step-2 build, before the R2 fix | 2913 s (P1 106, enc 11, P3a 634, P3b 2160) | first reconciliation |
| r2 `runs/p3_joint/2018/rehearsal_val_r2` | + R2 fix; P3b stage on pre-review readout | 2907 s (P1 102, enc 10, P3a 644, P3b 2148) | reconciliation after the fix |
| final `runs/p3_joint/2018/rehearsal_val_final` | reviewed code (sol-loop, 2 rounds, gpt-6.1-sol xhigh, approved), the freeze candidate | 2872 s (P1 ~102, enc ~10, P3a ~645, P3b 2130) | reproduction of r2, the frozen rehearsal |

The review (round 1: S1-S8, all fixed; round 2: approved) changed only:
- refusals (S1);
- formatting (S2);
- the comparison tool (S3) and the freeze file list (S8);
- the guards that apply when a set is not testable (S4, S5);
- new keys: `min_scam_per_draw` (S6) and the matched AUCs (S7).

None of these changes an existing val value. `--against` compares only the keys present in r2, so the new keys do
not enter it.

Cost: one run takes about 50 min on this machine, well under the SPEC §9 limit of 4 h.

## R1. P1 B1 s20261105: one scam kind argmax, at an fp16 near-tie

**Observation (r1).** Every P1 count is identical in all six rows and all three panels. Every field is within 1e-4
of its per-row reference, with one exception: B1 s20261105, scam panel, `kind_error`, |Δ| = 1.18e-4. That is one
argmax among 8,477 scam prefixes (1/8477 = 1.18e-4). The same row's NLL fields differ by at most 8e-7.

**Cause.** The archived evaluation ran with different fp16 GPU arithmetic from today's.
- **Today's runs are deterministic.** Two runs today give bitwise-identical per-row terms. Across r1 and r2,
  `p1_nll.json` and the encoder outputs are bitwise identical.
- **The difference is fp16-level.** Replaying the archive's exact order (`scripts/p1_encoder/screen.py eval_run`:
  `terms()` over every exact batch, then `evaluate`), today's per-row `kind + time + cens` terms differ from the
  archived `val_rows.npz` by up to 2.0e-3 (s20261105). s20261104 differs by up to 2.3e-3 against its own archive. The
  backbone runs under fp16 autocast.
- **Excluded:**
  - Code: lpw at e21ee8a, the HEAD when the archive was written (2026-11-06 21:35), reproduces today's values, not
    the archive's.
  - Data: the `data/pilot_2018` mtimes are 2026-11-04.
  - Weights: `model.pt` was written at 21:35:27, before `val_eval.json` at 21:35:49.
  - Library versions: `uv.lock` torch/CUDA entries are unchanged since 2026-11-04, and the driver has been
    615.78.08 since 2026-11-03.
  - Store layout: the full and guarded stores give identical results.
- **Likely cause, not proven:** each archived evaluation ran while another GPU job was running on the 4 GB card
  (the s20261104 generation overlapped the s20261105 evaluation). Under memory pressure, cuBLAS and cuDNN can choose
  different fp16 algorithms.

**Disposition.** No code change. The P1 table is descriptive and carries no verdict (§J5). The 1e-4 tolerance on an
error rate is finer than one event in the scam panel. The test P1 table reports this row with today's deterministic
numerics, and this entry is attached to it.

## R2. `hour_entropy` was not deterministic (value_counts hash order)

**Observation (r1).** There were 22 (P3a) and 44 (P3b) strict failures. All were the single-feature AUC diagnostic
`shortcut/hour_entropy` (real, synth, delta), |Δ| ≤ 4.5e-5. There were 155 (P3a) and 227 (P3b) detector differences,
almost all LGBM:
- P3a: |Δ| ≤ 0.006 on AUC/AP/ratio, plus `ranking_spearman` steps of 0.2 or 0.067, which are rank swaps among four
  detectors;
- P3b: |Δ| ≤ 0.0087.

**Every XGB value and array was exact.**

**Cause.** `lpw.p3.features.node_features` computed the hour entropy as a dot product over
`pl.col("hour").value_counts()`. Polars returns those counts in hash order, which changes between runs. The float
sum then moves by up to 1.3e-15.
- The measurement: four repeated builds of the same real val features, plus one build from the all-split store.
  `hour_entropy` was the only feature column that differed (max 1.3e-15).
- Two agents with the same hour-count multiset can therefore get unequal entropies. That breaks ties in the
  single-feature AUC. LGBM bins on distinct values, so a split of 1e-15 can move a bin edge and change the trees.
- The addendum's pre-review check ("LGBM full TRTR AUC 0.93070 against 0.93127", §J7) attributed this to the store
  layout. It is in fact run-to-run nondeterminism. The all-split store matched the archive only by chance in that
  check.
- **The archived val outputs are one draw of this tie-breaking.**

**Fix (determinism only, §J7 allowed fixes).** The counts are sorted before the dot product, so the value no longer
depends on hash order. The feature's definition is unchanged. Regression test:
`tests/test_p3a.py::test_hour_entropy_is_bitwise_equal_for_equal_hour_count_multisets` (fails before the fix, passes
after). After the fix, four repeated builds and the all-split build are bitwise identical on every column.

**Deviation, disclosed.** The archived val numbers are kept as the development record. The joint job's quantities
that depend on `hour_entropy` differ from them by the tie-breaking only. The size of that effect is measured in r2
(R3).

## R3. r2 against the archive: what the archived tie-break leaves

The archive is one draw of the R2 tie-breaking, so a deterministic job cannot reproduce it bitwise. The r2
reconciliation (`runs/p3_joint/2018/rehearsal_val_r2.compare.json`) gives the size of that draw:

| class | P3a | P3b |
|---|---|---|
| XGB, every scalar and verdict | exact (≤ 1e-9) | exact (≤ 1e-9) |
| XGB prediction arrays | n/a | all equal (≤ 1e-9) |
| non-detector quantities | exact, except `shortcut/hour_entropy` (22 values, ≤ 2.5e-5; no flag changes) | exact, except `shortcut/hour_entropy` (44 values, ≤ 1.8e-5; no flag changes) |
| RF | exact | exact (scalars and predictions) |
| LR | exact scalars | exact scalars; predictions ≤ 1.1e-6 |
| LGBM | 258 values ≤ 0.0050; `ranking_spearman` 0.2 in B world 1 (one rank swap among four detectors), 0.067 in the B aggregate | 238 values ≤ 0.0087; one nominal verdict: B `incremental_full/lgbm/learning` CONSISTENT (archive SUPPORTED) |
| encoder | embeddings and coverage exact | |
| clean subset (P0 counts) | reproduced: 1016 excluded, 2468 / 257 clean | |
| P1 | R1 only | |

**Per detector, r1 and r2 alike.**
- **RF:** exact in both runs. §J7's 2.2e-16 between calls is below the strict tolerance.
- **LR:** differs only in its predictions: 3.3e-6 in r1, 1.1e-6 in r2. The solver converges to a tolerance, so a
  1e-15 input change can move it that much. No readout scalar moves.
- **LGBM:** the only detector whose scalars move: ≤ 0.0059 / 0.0087 in r1, ≤ 0.0050 / 0.0087 in r2. Its histogram bins
  sit on distinct feature values, so a 1e-15 split can move a bin edge and change the trees.

The size of the drift is the same before and after the fix. This is expected: r1 had one random tie-break and r2 a
fixed one, and both are compared with the archive's random one.

**Detector rule (§J7), fixed here before the freeze.**
- **The cause is identified** (R2), and for every detector it lies in the archive, not in the job.
- **Tolerances for the rehearsal against the archive:**
  - RF: 1e-9;
  - LR: 1e-9 on readout scalars and 1e-5 on predictions;
  - LGBM: 0.01 on any AUC, AP, ratio or contrast;
  - `ranking_spearman`: any value, since it is a function of the LGBM AUCs;
  - LGBM verdict strings may differ; every one of them is nominal.
- r2 meets all of these.
- **Test LR / RF / LGBM rows:** reported normally, as nominal secondaries. Their val references carry the
  archive-draw drift above, which is attached to the comparison table.
- **No hierarchy claim is affected.** All three claims are XGB, and XGB is exact.

**The job itself is deterministic.** The final rehearsal must reproduce r2 bitwise (see the criterion below). That
is the property the single test opening needs.

## R4. Label equivalence

The P3b readout's existing label for too few valid bootstrap draws is "NOT TESTABLE (bootstrap)". P3a's new label is
INVALID (§J4). Both mean the same thing: fewer than 950 of the 1000 whole-group draws are valid, and no verdict is
given.

## Reproducibility criterion for the freeze

The final rehearsal must reproduce r2 bitwise. Every number in every output must match within 1e-9 (detectors
included), strings must be identical and arrays equal. Only timings are excepted
(`compare_val.py FINAL --against r2`). Against the archive, it must meet the §J7 criteria, with R1 and R2 as the
explained exceptions.

**Result (final against r2, `runs/p3_joint/2018/rehearsal_val_final.against_r2.json`): PASS.**
- **JSON outputs bitwise identical:** P1, the encoder coverage, the clean subset, the matched pairs, and both
  readouts with every detector.
- **Arrays bitwise identical except RF:** the encoder embeddings and the P3b samples are bitwise equal. The P3b
  predictions are equal except 65 of 286 arrays, all RF, which differ by at most 2.2e-16.
  - This is the RF call-to-call variation of §J7: `predict_proba` averages the trees in thread order.
  - It is below the 1e-9 strict tolerance, and it reaches no readout value: every readout is bitwise identical.
- **Tool change, disclosed:** `--against` first compared arrays with exact equality. That was stricter than the
  criterion stated above. It now uses the same 1e-9 as the archive comparison, and reports the largest non-bitwise
  difference.

**Against the archive:** the final run meets the §J7 criteria, with R1 (P1 near-tie) and R3 (the archived
`hour_entropy` draw) as the explained exceptions. Its per-class numbers are identical to r2's (R3 table).
