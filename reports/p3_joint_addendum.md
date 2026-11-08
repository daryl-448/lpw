# Joint sealed-test addendum: P3a + P3b (and descriptive P1 likelihood), 2018

Written 2026-11-08. This is before any test-split detector, model or metric output, and before the test-aware entry
point is built.

- **Design.** An Opus + Sol brainstorm, two rounds. Materials are in `runs/p3_joint/brainstorm/` (gitignored;
  `synthesis.md`, `sol_r1.md`, `sol_r2.md`, `opus_r1.md` and the check scripts).
- **Approval.** The user approved the decisions in §J0 on 2026-11-08.
- **What this freezes.** Everything a single authorized test job computes, how it is judged, and what may be said
  afterwards.
- **What it does not authorize.** Opening the test. That needs a separate, explicit user authorization naming the
  final commit (§J8).
- **Parent protocols.** `reports/p3a_prereg.md` (including addendum B, §16) and `reports/p3b_prereg.md`. Where this
  addendum is silent, they apply unchanged. Where it differs, it says so and why.

## J0. Decisions (user, 2026-11-08)

1. **The test opens once, for P3a and P3b together** (decision B of 2026-11-07, unchanged).
2. **No new model tranche before the opening.** In particular: no cross-group sharing pools, no sweep arm, no GNN,
   no P2 conditioning, no AGENT deliveries, no refits.
   - Cross-group pools cannot change R, which is fixed by the real data and STRUCT on the evaluation split. So they
     cannot make P3b claim A testable.
   - They cannot touch the local seam that drives the failing P3 gates either.
   - CORE FAIL limits a realistic-coupling claim. It was predicted at freeze (P3b prereg §P10, pre-freeze A4); it is
     not a blocking defect.
3. **Fixed-sequence confirmation** of three claims (§J3). Everything else is nominal or descriptive.
4. **A descriptive P1 likelihood table** is included (§J5.3). There is no P1 generation on test, and P1
   qualification is not reopened.
5. **This opening ends the 2018 arc.** There is no further 2018 model repair. The consolidated write-up follows
   (§J10).

## J1. Exposure ledger (replaces "never opened")

No detector, generator, kernel, encoder or threshold has been fitted, selected or scored on test rows. To our
knowledge this holds, checked by a filename inventory of `runs/`. But test is not pristine:

1. **The P0 audit (2026-11-04) computed label-dependent test summaries** (`reports/p0_audit.md`):
   - the role counts (3,392 normal / 349 scam);
   - the confounder-only AUC (0.610);
   - the common-support counts (scam 348 / normal 3,375) and the 348 caliper pairs under the P0 matcher;
   - the 932 test agents that share a non-venue counterparty with train, and the clean-subset counts (2,539 normal /
     270 scam).
2. **The 2018 window was chosen** by scanning candidate windows for estimated test matched-pair yield (SPEC §5.4).
3. **Label-blind transductive preprocessing.** Venues and Louvain groups were built on all W data, including the
   test structure, without labels (SPEC §5.4 step 1).
4. **All-split stores.** The P3a scripts and P1 `run.py` built all-split stores in memory (`build_store(PILOT)`), but
   their real-data APIs refuse test (`scripts/p3a/common.py:29`; the P1 val runs used `--split val`). The P3b loader
   never collects test rows (`src/lpw/p3b/access.py`).

**Development exposure of val is total.** Every val number in `reports/p3a_*.md`, `reports/p3b_readout.md` and the P1
reports stays a development result. Test results do not replace them, and the two are never pooled.

## J2. What is frozen (hashed in the manifest, §J8)

- **Code:** the test-aware entry point and every module it imports from `src/lpw/` (§J7).
- **P3a**, unchanged:
  - the worlds 0-2, train and hold parts, for kct and B: `runs/p3a/2018/world{0,1,2}/`, i.e. populations, raw
    events and `gen.json`;
  - the feature order and sets (`lpw.p3.features`);
  - the detector settings and seed (`lpw.p3.utility`);
  - the matcher (`lpw.p3.matching`: a train-fitted propensity model; support, caliper and tie order from train);
  - the encoder: `runs/p3a/2018/encoder/encoder.pt`, its normalizer state, and the saved synthetic-hold embeddings in
    `embeddings.npz`. Their hashes are already recorded in `encoder.json`.
- **P3b**, unchanged:
  - the group prior and assigned group maps: `runs/p3b/2018/world{w}/groups_{train,hold}.npz`;
  - the prior and kernel parameters (`runs/p3b/2018/{prior,kernel}/`);
  - the attachment sidecars for P3b-0 and P3b-1;
  - the STRUCT definitions;
  - the core, daily-topology and Clauset recipes, including the continuous xmin grid of P3b prereg §P8.
- **P1:** the six checkpoints in §J5.3 and the scorer `lpw.p1.evaluate`.
- **Excluded:** the rejected sweep arm and worlds 10-12.

## J3. Confirmation hierarchy (fixed sequence)

There are three claims. Each is tested at one-sided α = 0.05, with the test-group bootstrap of §J4.4. A claim is
**reached for confirmation** only if every earlier claim is SUPPORTED.

| step | claim | rule (unchanged from the parent protocol) | a SUPPORTED result licenses |
|---|---|---|---|
| 1 | **P3a transfer.** kct, XGB, `full`, mean AUC over worlds 0-2. D = (mean TSTR − 0.5) − 0.8 (TRTR − 0.5). | NOT TESTABLE if TRTR < 0.60; otherwise SUPPORTED if D ≥ 0 and the one-sided 95% lower bound is ≥ 0; CONSISTENT if D ≥ 0 only; NOT SUPPORTED if D < 0 (P3a prereg §7) | "a detector trained only on the frozen synthetic population keeps ≥ 80% of the real-trained excess AUC on 2018 test". Not the val point estimate (95%), and not P3 qualification. |
| 2 | **P3b learning advantage.** kct, XGB, L. H = T₁ − T₀ (learned P3b-1 minus empirical-rate / uniform P3b-0) | SUPPORTED if H > 0 and the one-sided 95% lower bound is > 0; CONSISTENT if H > 0 only; NOT SUPPORTED otherwise (P3b prereg §P9 B) | "the learned attachment beats the restricted empirical-rate / uniform control". Not non-harm against zero sharing (T₁'s val CI crosses 0), not positive incremental utility, not realistic coupling. |
| 3 | **STRUCT-only retention.** kct, P3b-1, XGB, S. D_S = (mean TSTR(S) − 0.5) − 0.8 (TRTR(S) − 0.5) | NOT TESTABLE if TRTR(S) < 0.60; otherwise P3a's rule (P3b prereg §P9, secondary list) | "≥ 80% of the real STRUCT-only excess AUC is retained". Not structural information beyond the local features. |

**Rules.**
- **The sequence stops at the first claim that is not SUPPORTED.** Later claims are still computed and reported with
  their raw verdicts, labelled **"not reached for confirmation"**.
- **Outside the sequence:** B's H and D_S, P3b claim A, every detector or feature-set variant, and every sensitivity
  analysis. They are reported with nominal verdicts only. B never rescues kct.
- **P3b claim A** keeps its rule unchanged, including the R ≥ 0.020 floor. Test R is expected to be about 0
  (val +0.0016; train OOF +0.003). Even if R clears the floor, A stays nominal. No family-controlled "learned
  structural signal transfers" claim can follow, because P3b's combined wording needs A and B.
- **Inference wording:** "nominal 5% familywise control, conditional on the frozen Louvain whole-group bootstrap".
  - Cross-group sharing leaves dependence between bootstrap groups.
  - The models are fixed, and only three worlds were drawn.
  - So the intervals are conditional on the evaluation set, not guaranteed population coverage. This limits the
    primary as much as the secondaries.
- **Disclosed discrepancy.** The frozen primary uses `full` features. SPEC §8's qualification text asks for the
  confound-removed ratio. `noconf` is reported as a secondary and is never swapped in.

**How each verdict is read.**

| verdict | reading |
|---|---|
| SUPPORTED | the claim's threshold holds with one-sided 95% confidence |
| CONSISTENT | inconclusive: the point estimate meets the threshold, the bound does not |
| NOT SUPPORTED | the point estimate is below the threshold. Confident failure is stated only if the two-sided 95% upper bound is also below 0 |
| NOT TESTABLE | the gate failed (TRTR, R or a role is missing). Neither success nor generator failure |
| INVALID | fewer than 950 / 1000 valid bootstrap draws (§J4.4). No inferential verdict |

**Test and val disagreeing.** Test owns the confirmation verdict. Val is shown alongside it and never pooled with it
to rescue a result. Not confirming a claim is not, by itself, evidence that there is no effect.

## J4. Test-specific rules

### J4.1 Evaluation split substitution

Real val becomes real test wherever val served as the evaluation set. Real train stays the training and fitting
source. Specifically:
- TRTR = real train → test;
- TSTR = synthetic train world → test;
- the C2ST pits real test against the independent hold world;
- difficulty parity uses test prevalence;
- the DCR uses test as the real holdout reference;
- STRUCT is computed on the test split's own graph;
- the P3b core half views are drawn from test, with n set by the frozen formula
  `n = min(N_test // 2, min_hold // 2)`;
- daily topology, Clauset fits and the coupling diagnostics use the test view;
- encoder coverage embeds real test with the frozen encoder, using the inference path only.

Nothing is refitted, and no threshold, bin or normalizer is re-estimated on test. Train-fitted quantities are used
exactly as at the val readouts: the F1 thresholds from training-source OOF, the matcher, the feature normalizers and
the encoder normalizer.

### J4.2 Matching

- `Matcher(F_train, y_train).match(F_test, y_test)`, with the frozen rule. New pair IDs are constructed and saved.
- No pair count is asserted (the P3b val script asserted 348). Fewer than 200 pairs makes the matched analyses NOT
  TESTABLE. Support, caliper and balance rules are never relaxed.
- Full-split STRUCT is used for the matched rows.
- Matched analyses are point estimates, as at val: TSTR and TRTR AUC, and the behavioural gain full − confonly
  (P3a) and full+S − confonly (P3b). No pair bootstrap is used; at val, 339 / 348 pairs cross Louvain groups.

### J4.3 Clean subset (SPEC §7.4, P0 rule)

**Definition.** A test agent is excluded if any of its non-venue counterparties also occurs among the train agents'
counterparties. This is the rule of `lpw.data.audit.leakage_report`, applied to the same `events.parquet` and venue
table. P3b's extra non-agent restriction is not substituted. It is computed only inside the authorized job; P0's
historical counts were 932 excluded and 2,539 / 270 clean.

**Analyses on clean agents.** The same clean agent set is used for TRTR and TSTR.
- **Fixed scores.** Scores and full-test STRUCT are kept; agents are restricted, with no transaction deletion and no
  refit. This is not an inductive-graph claim.
- **P3a.** Every `transfer` contrast of the primary detector (`full` and `noconf`, kct and B), with its bootstrap
  over clean-agent groups.
- **P3b.** R / T₀ / T₁ / D / H (kct and B, L and full bases) and D_S. All nominal.
- **Clean + matched.** The frozen matcher is rerun inside the clean pool, with no refitting of the propensity model.
  This is a separately labelled estimand: ≥ 200 pairs, or NOT TESTABLE. Point estimates only.

**Sensitivity only.** At val, the clean rule left 2,725 agents (257 scam), and only 209 of the 348 original pairs
stayed intact.

### J4.4 Bootstrap validity (prospective amendment)

- **The amendment.** `GroupBoot` (`src/lpw/p3/utility.py:70`) silently drops invalid draws. Every bootstrap verdict
  (P3a transfer, P3b R / T / D / H, D_S, and the clean-subset contrasts) now requires ≥ 950 / 1000 valid two-role
  draws, otherwise INVALID.
- **Reported:** the valid count, and the minimum scam count per draw.
- **Recipe:** 1000 seed-0 whole-group draws over the test evaluation set's Louvain group ids, using P3a's and P3b's
  definitions.
- **Effect on val:** none. Every val verdict had 1000 / 1000 valid draws (Sol check, `checks_r2.json`).

### J4.5 Giant group

- **The group:** the largest test Louvain group, by size, then id.
- **P3b:** both of its sensitivities, unchanged: fixed scores without the group, and graph removal (test STRUCT
  recomputed without the group, same models).
- **P3a:** a new influence table, the primary detector's `full` and `noconf` contrasts with fixed scores and the giant
  group removed.
- **Both:** the group's size and scam count, and the test inverse group-size Herfindahl, are reported. At val the
  group had 654 agents and the Herfindahl was 28.1.
- These sensitivities explain leverage; they never replace a verdict.

## J5. Report inventory (computed in full, regardless of the hierarchy)

### J5.1 P3a: everything `scripts/p3a/readout.py` computes, on test (code unchanged since `ab81c64`; val readout `33a92df`)

**The readout sections, per world and aggregated over worlds 0-2, for kct and B:**
- TRTR and TSTR AUC / AP for every detector × {full, noconf, confonly, noprof, nobal};
- F1 (primary, full);
- matched results and the behavioural gain;
- owned-only real observation;
- the cap-removal sensitivity, when any stream is capped;
- difficulty parity, ranking agreement and profile-gain parity;
- C2ST on local and full features, plus the top-10 single-feature seams (exploratory);
- the shortcut audit with its sentinels;
- tabular mode coverage and DCR.

**Added in this addendum:** the clean-subset analyses (§J4.3), the giant-group influence table (§J4.5) and the
valid-draw counts.

**Encoder mode coverage (supplementary, addendum B).**
- `encoder.pt` and its normalizer are hash-verified, then real test is embedded.
- The synthetic-hold embeddings are the saved ones, hash-checked against `encoder.json`.
- Scoring is `scripts/p3a/encoder_score.py`'s recipe, with real test in place of real val: k = 5 per role,
  equal-size pools, seed-0 subsampling.
- An INVALID exit applies on any non-finite embedding.
- Reported, with no gate.

### J5.2 P3b: everything `scripts/p3b/readout.py` at `5167b9e` computes, on test

- the utility contrasts A and B, and every secondary of §P9, including D_S for both arms;
- the full-base contrasts, matched pairs and the giant-group sensitivities;
- F1, difficulty and profile-gain parity;
- the grouped C2ST (S, L+S, full+S) and the shortcut audit;
- the eight-item core family on 20 half-view draws, the full-view distances, and the cohort and age-tercile
  marginals;
- daily topology with venues included and excluded, the Clauset fits, and the coupling diagnostics;
- the zero-sharing anchor and all per-world results;
- the provenance hashes, predictions and samples.

**Added in this addendum:** the clean-subset analyses and the valid-draw counts.

**Rule changes from the val script:**
- the 348 assertion is removed (§J4.2);
- the evaluation split is a parameter (§J7).

### J5.3 P1: descriptive held-out likelihood of frozen instruments

**Rows** (each hashed before access; never best-seed selection):

| row | checkpoint | config |
|---|---|---|
| kct | `runs/p1_shallow/2018/kct` (P1-freeze manifest) | `lpw_plan_prog_shallow_kct` |
| B | `runs/p1_r0/2018/lpw_plan_prog_rp` | `lpw_plan_prog` |
| B0 | `runs/p1/2018/b0` | `b0` |
| B1 s20261104 | `runs/p1_enc/2018/b1_lr1e-3_s20261104` | `b1` |
| B1 s20261105 | `runs/p1_enc/2018/b1_lr1e-3_s20261105` | `b1` |
| B2 | `runs/p1/2018/b2` | `b2` |

**Scoring.**
- The scorer is `lpw.p1.evaluate.evaluate(model, store, split, views)`.
- Prefixes are exact and identical for every row; the identity of the agent / depth / owned arrays is asserted
  across rows.
- Plan models run with `plan_inference = False`, i.e. the unknown-plan likelihood. kct therefore equals B by
  construction, which is asserted, not reported as independent replication.
- No recalibration, fitting, generation or temporal experiment is run.

**Reported** for all / normal / scam:
- prefixes, owned and censored counts;
- the NLL of each field (kind, time, cens, cp, amt), plus the full and shared NLL and the counterparty-marginal
  amount, in nats per owned event;
- the error metrics `evaluate()` returns.

B2's counterparty and full NLL are **N/A**; it has no counterparty head.

**Caption, verbatim:** "Held-out descriptive likelihood of frozen instruments; P1 remains unqualified under A2. No P1
exit decision is reopened." No significance marks and no P1 inferential family. Paired M-pass generation stays
val-only, as recorded.

## J6. Planning power (not a guarantee)

**Method.** From the exact val 95% interval widths: SE = width / 3.920, and P(one-sided lower > 0) = Φ(effect / SE −
1.645). This is a marginal normal approximation that excludes the TRTR gates and the sequence. Asymmetric bootstrap
distributions, network dependence, selected val effects and world variation prevent a calibrated power claim.

| contrast | val effect | SE | at the val effect | at half of it |
|---|---|---|---|---|
| P3a D (step 1) | +0.064 | 0.0104 | 1.00 | 0.92 |
| H, kct (step 2) | +0.0146 | 0.0051 | 0.89 | 0.42 |
| H, B (nominal) | +0.0081 | 0.0040 | 0.65 | 0.26 |
| D_S, kct P3b-1 (step 3) | +0.030 | 0.0209 | 0.42 | 0.18 |

The clean-subset analyses lose sample. Under simple scaling, their intervals are about 14-16% wider.

## J7. Test-aware entry point and rehearsal

**Entry point.** A new script runs P3a, P3b, the encoder coverage and P1 in one process, with the evaluation split
as its only scientific parameter.

**Development guards stay.**
- The val-only APIs keep refusing test.
- Test rows are reachable only through a separate sealed loader. It refuses unless every one of these holds:
  1. `HEAD` is an authorization-only commit: its single parent is the frozen commit named in the authorization
     record (§J8), and `git diff HEAD^ HEAD` touches only `reports/p3_joint_authorization.md`;
  2. the working tree is clean;
  3. every hash in the frozen manifest matches;
  4. the output directory does not exist yet.
- It loads `("train", "test")` only, never val.

**Output discipline in test mode.**
- The console shows progress lines only, with no metrics.
- Results are written to a staging directory, `runs/p3_joint/2018/.test_staging/`. Only after every file and the
  output hash manifest are written and fsynced is the staging directory renamed, atomically, to
  `runs/p3_joint/2018/test/`. A run that ends without the rename publishes nothing (§J8.4).

**Fixtures** (no real test rows): the sealed branch on a fake split; wrong or absent authorization; an authorization
commit that touches another file or has the wrong parent; a dirty tree; a changed hash; a publish interrupted before
the rename; a one-role or empty evaluation set; undefined metrics; fewer than 950 valid draws; fewer than 200 pairs;
a giant fold.

**Rehearsal.** Before the freeze, the exact job runs with the evaluation split set to **val**, writing to scratch,
under the guarded `("train", "val")` loader. It must reproduce the committed val outputs.

| output | criterion |
|---|---|
| P3a `runs/p3a/2018/readout.json`, XGB and every non-detector quantity | every scalar to \|Δ\| ≤ 1e-9, every verdict string identical |
| P3b `runs/p3b/2018/readout.json` and predictions, XGB and every non-detector quantity | every scalar to \|Δ\| ≤ 1e-9, verdicts identical, prediction arrays equal |
| LR / RF / LGBM rows of both readouts | see the detector rule below |
| encoder | re-embedded real val within \|Δ\| ≤ 1e-4 of the saved `emb_real_val` (GPU inference), and every coverage P / R within 0.01 of `coverage.json` |
| P1, every row | the prefix, owned and censored counts identical, and every NLL field, the shared and marginal-amount NLL and every error metric within 1e-4, in all three panels (all / normal / scam). The full NLL is compared wherever it is defined (not B2). Identical prefix arrays across rows. |

**P1 references, per row:**
- kct and B: `runs/p3_joint/2018/p1_ref/B_val_eval.json`, a three-panel reference. It is made before the rehearsal,
  outside the new entry point: `lpw.p1.evaluate.evaluate` is called directly on B, with the loading recipe of
  `scripts/p1_audit/b1gap.py` (`plan_inference = False`), on the guarded `("train", "val")` store. Its `all` panel
  must match the archived aggregate `runs/p1_audit/2018/base_plan_prog_rp_val_eval.json` within 1e-4 in every field.
  It is hashed into the frozen manifest;
- B1 s20261104 and s20261105: `runs/p1_enc/2018/b1_lr1e-3_s{seed}/val_eval.json`;
- B0 and B2: `runs/p1/2018/results_val.json` (`nll.b0`, `nll.b2`).

**Detector rule (LR / RF / LGBM).** All three hierarchy claims are XGB, and XGB must reproduce exactly. A pre-review
check (Sol r1, guarded train / val) found three non-XGB differences against the archived outputs:
- RF: prediction differences of 2.2e-16 between calls on the same fitted model;
- LR: about 1e-6;
- LGBM: full TRTR AUC 0.93070 against 0.93127 archived.

The LGBM difference is not rounding. Before the freeze, the rehearsal ledger must identify the cause of each
non-XGB difference, using repeated identical calls and a feature-matrix comparison between the all-split and guarded
stores. It then fixes, in that ledger and before the freeze commit, a detector-specific tolerance from the measured
cause.
- **Allowed fixes:** determinism-only ones, such as fixed thread counts or library determinism flags, disclosed as
  deviations, with the archived val numbers kept.
- **Not allowed:** changing a hyperparameter.
- **If a cause stays unexplained,** that detector's test rows are reported with the unexplained val drift attached.
  They never carry a verdict.

- **Exempt from the comparison:** new sections that did not exist at val (clean subset, P3a giant-group influence,
  valid counts) and timing or provenance fields.
- **Any discrepancy** is recorded in a ledger and explained before the freeze. A code fix goes back to a scoped Sol
  review.
- **Cost.** The rehearsal measures it. A projection over 4 h goes to the user (SPEC §9).

## J8. Freeze, authorization, opening

1. **Freeze.** After review (sol-loop) and a successful rehearsal, a commit freezes the code. Its manifest (§J2)
   records sha256 for every code file, artefact, checkpoint and input parquet, plus the environment (`uv.lock`,
   Python, torch, CUDA).
2. **Authorization.** The user explicitly authorizes the opening, naming the freeze commit. The record
   `reports/p3_joint_authorization.md` states that commit, the date and the user's words. It is committed as the
   freeze commit's only child, changing nothing else (§J7 check 1).
3. **Opening.** One complete job at the authorization commit publishes `runs/p3_joint/2018/test/`:
   - the json, txt and npz files: predictions, folds, pairs, samples and counts;
   - a hash manifest of every output.

   No selective peeking, no automatic retry, no tuning and no new analysis.
4. **Technical failure** (any run that ends without the atomic rename, including a serialization or disk failure).
   - The attempt is INCOMPLETE.
   - The staging directory is kept, moved unread to `runs/p3_joint/2018/failed_attempt_<n>/` with a hash listing of
     its files, and disclosed with the traceback. Its contents are never inspected or reported.
   - A fix is a reviewed commit with a fresh authorization record. The rerun uses the same frozen science.
5. **A bug found after the opening.** It counts only if the code contradicts this frozen text. The original output is
   kept, and the corrected one is reported beside it, with the user's authorization for the correction. The holdout
   is spent; it is never re-sealed by re-splitting or new seeds.

## J9. Budget

- **Build:** about 1-2 days for the entry point, fixtures and rehearsal.
- **Rehearsal and opening:** each projected at about 1-2 h of CPU wall time and ≤ 10 min of GPU, with P1 inference
  at about 25-35 s per model. Archived anchors: P3a readout 945 s; P3b readout 2,079 s.
- **Over budget:** the SPEC §9 rule applies, and nothing is silently reduced.

## J10. After the opening

- **`reports/p3_joint_test.md`.** Every test number next to its val counterpart, the hierarchy verdicts, the nominal
  secondaries and every failure.
- **A consolidated write-up of the 2018 arc.** Sol audits its claims; no new test analyses are added.

**Licensed if confirmed:**
- a sampled learned-policy population supplies useful 2018 node-level scam-label training data, despite detectable
  distributional defects;
- if step 2 confirms, the minimal learned identity attachment beats its restricted control.

**Not licensed:**
- faithful AML or laundering dynamics;
- P1 or P3 qualification;
- behavioural coupling;
- incremental structural utility;
- typology completeness;
- privacy guarantees from DCR;
- cross-era generalization.

**Later work.** Follow-ups need fresh confirmation data, and none is ready for the matched claim:
- the HF source is the same dataset;
- 2017-07..2018-07 overlaps the slice;
- windows from 2019 on have an estimated 47-83 test pairs;
- an unopened 2021 holdout (62 pairs under P0's matcher) could carry only a separately pre-registered cross-era or
  fidelity question.

## J11. Deviations

None yet.
