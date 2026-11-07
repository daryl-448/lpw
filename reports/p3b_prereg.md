# P3b pre-registration: learned shared-counterparty attachment on frozen streams (2018)

Written 2026-11-07. This is before any P3b val role-utility, C2ST or coupling-fidelity readout, and before the
attachment kernel is fitted.

- **Design.** Sol consult (`reports/p3b/sol_consult.md`), adopted with the frozen numbers below.
- **Scope.** User decision B: P3b is part of the confirmatory package, and the sealed test is opened once, for P3a and
  P3b together (§P12).
- **Changes.** Any change to what is listed here, made after the first P3b readout, is a deviation. It is recorded
  in §P13, with the frozen-protocol result reported next to it.
- **Code.**
  - `src/lpw/p3b/{access,replay,groups,kernel,graph,topology}.py`
  - `scripts/p3b/{common,access_check,prefreeze,prefreeze_topology,fit,attach,checkpoint0,readout}.py`
  - tests: `tests/test_p3b.py`

## P0. Scope and claims

**The model.** A frozen-stream periphery-identity attachment model, on the 2018 slice.
- The saved P3a local tapes (kct primary, B control) are left unchanged.
- A train-fitted conditional group prior places agents in groups.
- A causal kernel then decides, at each first local touch of a non-venue, non-agent counterparty, whether the touch
  reuses a node that another group member touched in an earlier recorded second, and which one.

**Arms.**
- **P3b-0: empirical-rate / uniform control.** It attaches with probability p0[role] when the opportunity set is
  nonempty, and picks uniformly from it.
- **P3b-1: the one learned candidate.** A state-only ridge-logistic gate and a ridge conditional-logit pointer.
- **Anchor: zero sharing.** The original P3a private identity map.

**Why local results do not move: pathwise invariance.**
- kct and B see only {REPEAT, VENUE, NEW} plus local partner-table state, never a non-venue global identity.
- So an injective, per-agent relabelling of non-venue entries into shared global nodes changes no token, amount,
  fee, ledger value or local feature. The same holds for every local feature block (CONF, PROF, LOCAL, BAL, VENUE).
- P3a's local results are therefore unchanged by construction. They are not new P3b evidence.
- Only cross-agent structure changes, and with it the STRUCT block and the graph metrics.

**Not in P3b:**
- a GROUP input token, or any behavioural feedback from sharing;
- new event generation, or any policy refit (A2 is not reopened);
- AGENT, deliveries, taint or a GNN;
- a learned membership model (P3b uses uniform placement within role);
- the rejected sweep arm, or worlds 10-12.

**Relation to the SPEC.** This is a narrowed, restricted implementation of SPEC §6.6 / §6.10, not the coupled
policy and engine of §6.7.
- The P3 plan's word "refit" means fitting this attachment component.
- Synthetic agent↔agent edges are zero by design.
- Pools are exclusive to a group, so cross-group sharing cannot be produced (§P10).
- Even an excellent P3b result does not qualify the unchanged full P3.

**Development data.** 2018 val is used for development and is not fresh confirmation. The design consult saw val
summaries computed by role:
- the GROUP share of first touches (0.139 normal, 0.064 scam);
- per-agent shared-counterparty fractions;
- the val group-size facts.

These are disclosed here as development exposure.

## P1. Access and provenance

**Guarded loader.** All P3b development reads go through `lpw.p3b.access.load()`:
- it calls `build_store(splits=("train", "val"))`;
- agent rows are filtered by split inside the parquet scan, and events, profiles and labelled-contract flags by the
  kept node ids, before collection;
- every development API rejects `test`.

**Not used in P3b development:** `build_store()` without `splits`, `scripts/p3a/common.store` and `group_ids`, and the
brief's unfiltered `facts.py`.

**Reproduction check.** `scripts/p3b/access_check.py` shows the filtered store reproduces P3a's real side exactly:
- TRTR XGB AUC for full 0.93172, noconf 0.92942 and confonly 0.79789;
- 348 matched val pairs.

**Disclosure.** P3a's loaders built the full store in memory, with all splits, and computed only on train and val.
P3b does not do this.

**Transductive provenance.** The venue set and the Louvain groups were built transductively before the split
(SPEC §5.4). They are reused as historical artefacts, so P3b groups are not claimed to come from an inductive graph.

## P2. Populations, worlds, arms, seeds

**Inputs.**
- Only the unmodified `runs/p3a/2018/world{0,1,2}/pop_{train,hold}.npz` and `{kct,B}_{train,hold}.parquet` are used,
  and they are never overwritten.
- The P3a cap and retention rules carry through unchanged: cap 10,000, and a safety stop if more than 0.001 of the
  original population is capped.
- Zero-event agents are dropped after attachment and counted.

**Group seed** `4000 + 10w + p`, with p = 0 for train and 1 for hold.
- Templates: `SeedSequence([seed, 0])`.
- Within-role permutation for role r: `[seed, 1, r]`.
- The map is shared by kct / B and by P3b-0 / P3b-1.

**Attachment seed** `5000 + 10w + p`.
- Agent i's q-th eligible first touch takes the q-th draw of `[seed, 0, i]` (gate uniform) and of `[seed, 1, i]`
  (pointer uniform).
- One of each is consumed at every eligible first touch, even if the pool is empty or the gate says NEW.
- These are common random numbers across P3b-0 / P3b-1, and partly across kct / B.
- NumPy PCG64, numpy as locked in `uv.lock`.

**Order.**
- Canonical processing order: (recorded second, original agent id, original local event index).
- Candidate order: creation order (mint second, minter id, minter local index).
- **Real mint direction** (a pointer feature), when several members first touch one real node in the same second:
  the direction of the touch first in the stable real order (ts, block_number, tx_hash, OUT first). Visibility still
  follows the second-batch rule.

**Fit seeds.** Fitting, CV and readout use seed 0; noise floors use seeds 0-19. No seed shopping.

## P3. Membership prior (`lpw.p3b.groups`)

**The prior.**
- Real-train templates (s, k), with s ≥ 2: 122 templates, 732 groups.
- M[s,k] ~ Poisson(N / 17,454 · c[s,k]).
- A whole draw is rejected if it needs more slots than the role totals of the original sampled population allow,
  with at most 1,000 proposals.
- Remaining slots become role-specific singletons, and agents are placed uniformly within role.
- The canonical slot layout is in the module docstring.

**Artefact.** `runs/p3b/2018/prior/templates.npz` (keys, counts, N_train), hashed.

**Pre-freeze facts** (`prefreeze.{json,txt}`, 200 non-frozen seeds):

| | train-size worlds | hold-size worlds |
|---|---|---|
| proposals | 1 | ≤ 4 |
| singleton share (train 0.460) | 0.454 | 0.47-0.49 |
| P(largest group ≥ 500) | 0.63 | 0.19-0.20 |

Real val holds one 654-agent group (17% of val).

**Rules.**
- No val reweighting, behaviour-aware mapping or co-activation sorting.
- Proposal counts, singleton remainders and the sizes of the sampled and retained groups are reported per world and
  part.

**Known limitation (pre-freeze A5).** Real groups are co-activated: the mean within-group SD of first-event day is
21.4 days, against 58.2 ± 0.9 under uniform within-role placement. Uniform placement therefore spreads synthetic
members in time and removes pool opportunities: the world-0 smoke run had 0.40 of first touches with C_e nonempty,
against 0.81 real. This is reported, not modelled.

## P4. Eligibility and causal replay (`lpw.p3b.replay`)

**Eligible events** touch a counterparty that is non-venue and non-agent.
- Real: `venue < 0` and `!cp_is_agent`.
- Synthetic: `venue < 0`.
- The class precedence (REPEAT before GROUP) is implied: identity decisions happen only at an entry's first local
  touch (real `cls == NEW`; synthetic NEW draws, asserted equal to the first touch).

**Recorded-second batch rule.**
- Within a second, an agent's own entry map and touched set update at once.
- Pool additions and pool statistics (members, events, last touch, directions seen) become visible to other members
  only after the whole second.
- Repeat touches update pool statistics but make no decision.

**The opportunity set.** C_e = the visible pool nodes the actor has not touched. If C_e is empty, the touch is forced
NEW.
- Real label: the target is in C_e.
- Real same-second co-first touches of one node are not positives. Pre-freeze they are 0.06% of train first touches
  (79), and they are reported as unmodelled.

**Synthetic attachment.** It keeps the original frozen periphery metadata and tokens. Choices are logged as a sidecar
keyed by original (agent, local event index).

**Real pool facts** (train):

| | normal | scam |
|---|---|---|
| first touches | 102,663 | 24,792 |
| share with C_e nonempty | 0.799 | 0.873 |
| unconditional attachment rate | 0.1322 | 0.0914 |
| conditional rate p0 | 0.165482 | 0.104781 |

Exact p0 values are frozen in `runs/p3b/2018/kernel/p0.json`.

**Choice sets.**
- With C_e nonempty, the median size is 294 and the maximum 6,109.
- 15,842 positives, with 16.8M pointer alternatives in total.
- 55% of positives are in groups larger than 100.

## P5. Fitting (`lpw.p3b.kernel`, `scripts/p3b/fit.py`)

**Gate.**
- Fitted on real-train first touches with C_e nonempty.
- Continuous inputs:
  1. log1p(the actor's events before this one);
  2. log1p(eligible partners already seen);
  3. log1p(|C_e|);
  4. log1p(visible pool size);
  5. log1p(other group members);
  6. log1p(other members with an event in an earlier second);
  7. log1p(days since origin);
  8. window fraction.
- Binary inputs: role, born, OUT.
- Design: standardized continuous inputs, the binaries, role × the continuous, role × born, role × OUT, and an
  unpenalized intercept.
- Loss: summed BCE + 10/2 ‖β‖².

**Pointer.**
- Fitted on gate positives, over all of C_e, with no intercept.
- Features: log1p(distinct members who touched v), log1p(group events on v), log1p(days since v's last group touch),
  any earlier same-direction group touch, and v minted by an OUT. Each has a role interaction.
- Loss: summed choice NLL + 10/2 ‖θ‖².
- Standardization: over every alternative of the fitting sets.

**Both fits.**
- Arithmetic mean and population SD, with zero SD → 1.
- Deterministic zero-start L-BFGS on CPU in float64: lr 1, history 50, strong Wolfe, max 500 iterations, gradient
  tolerance 1e-6, change tolerance 1e-12.
- No backbone states, amounts, balances, plans, IDs, pool role mix or future information.
- No class weights, thresholds, calibration, early stopping or search.
- Applied by sampling, never by hard decisions.

**Diagnostics** (3 train whole-group folds, seed 0, the same fold rule as pre-freeze A6). They are reported, and they
gate nothing.
- Gate BCE and Brier vs the p0 control (fitted on the fitting folds); calibration deciles by role.
- Pointer NLL vs uniform NLL (log |C_e|).
- Opportunity counts and choice-set quantiles.

**Final fit.** Once on all train. Parameters, convergence (objective, max gradient, iterations) and hashes go to
`runs/p3b/2018/kernel/`.

**Failure policy.** A non-finite or invalid fit is a technical failure. Predictive weakness is reported, and no other
model is swapped in.

## P6. Observation and local invariance

**Local side unchanged.** The P3a O operator and feature code are used without change. Global endpoints are a
separate field:
- `cp_local` is the old per-agent key;
- `cp_global` holds real raw `events.cp`, or synthetic venue / attached-node keys.

**Required exact equalities**, across the original and both attachment arms:
- every non-identity event field;
- every local cp equality relation;
- the local feature matrices;
- the retained agent IDs and cap counts.

**Required properties of every attachment:**
- it has an earlier-second witness in the same group;
- it is a previously unseen entry of the actor;
- no non-venue node is shared between groups or between worlds.

**Real transactions** (`lpw.p3b.topology.tx_table`). For graph counts:
- One row per transaction identity (block_number, tx_hash).
- Every recording of one transaction must agree **exactly** in directed endpoints, timestamp, value and venue flag.
  If any differ, it is a schema failure, and the run stops.
- Distinct transaction identities stay distinct even when everything else coincides.
- Pre-freeze: train has 357,060 incident rows and 355,798 transactions; val has 59,619 rows and 59,419.
- Synthetic rows are distinct transactions. Self-loops are dropped from simple-graph topology.

**Node namespace.**
- Synthetic graph keys are typed: agent, venue or periphery in the top bits, then the view-local index
  (`node_key`). Real keys are raw addresses.
- Sidecars record their view context (world, part, arm, attachment), and loading checks it.
- Keys are view-local: independent views are never merged, and numeric equality across views is not identity.
- Checkpoint 0 asserts that every incidence node is in the periphery namespace.

## P7. STRUCT (`lpw.p3b.graph.struct_features`)

**Notation.** C_i is agent i's distinct eligible counterparties, d(c) the number of focal agents touching c in this
split or world view, and N the retained focal agents.

**The six features:**
1. `shared_any`;
2. `shared_frac` = #{c ∈ C_i : d(c) ≥ 2} / |C_i|;
3. `mean_cp_other_frac` = mean (d(c) − 1) / (N − 1);
4. `max_cp_other_frac`;
5. `other_agents_frac` = |U_i| / (N − 1);
6. `sharing_component_frac` = (K_i − 1) / (N − 1).

All are 0 on an empty set.

**Context.** Each view uses its own unlabelled graph: real train, real val (later real test), and each synthetic
world part. No IDs, group labels, community IDs, pool flags or roles enter. The list was not selected on role AUC.

**Pre-freeze A6** (train-only OOF, pre-specified in `prefreeze.py`):
- AUC(L) 0.922-0.937; AUC(L+S) adds −0.002 to +0.008, giving **R_oof = +0.003**;
- STRUCT alone gives AUC 0.66-0.72.

So the primary incremental-utility verdict (§P9 A) is expected to be NOT TESTABLE on val. That outcome was chosen in
advance. It is not a reason to search features or add a GNN.

## P8. Coupling fidelity and SPEC §7.3 (`lpw.p3b.graph`, `lpw.p3b.topology`)

### Core family (8 items; all W1)

| # | item |
|---|---|
| 1 | per-agent non-venue `shared_frac`, normal |
| 2 | `shared_frac`, scam |
| 3 | `other_agents_frac`, normal |
| 4 | `other_agents_frac`, scam |
| 5 | `sharing_component_frac`, normal |
| 6 | `sharing_component_frac`, scam |
| 7 | log1p(focal agents per eligible node), pooled, including degree-one nodes |
| 8 | log1p(agent count of the recovered community), agent-weighted |

Item 8 uses Louvain (seed 20261104, sorted insertion) on the eligible bipartite incidence graph, with isolated agents
as size-one communities.

**Half-view noise floor.**
- n = min(⌊N_real_val / 2⌋, min_w ⌊N_hold_retained,w / 2⌋), one n for every arm and world.
- For each seed s = 0-19: two disjoint uniform label-blind n-subsets A and B of real val, and one n-subset of each
  synthetic hold. The selected agents keep their complete incident streams, and every view is rebuilt from scratch
  (degrees, components, communities).
- d_m = W1(synthetic, A) and f_m = W1(A, B).
- Role items subsample each view's agent measurements to the triple's minimum role count, with the same seed; at
  least 20 per role is required.
- Each distance is averaged over the 20 draws, then over the 3 worlds, separately for kct and B, and separately for
  P3b-0 and P3b-1.

**Verdicts.**
- **CORE PASS:** d_m ≤ 2 f_m for every item, inclusive. If f_m = 0, only d_m = 0 passes.
- **CORE FAIL:** any item fails.
- **INCOMPLETE:** any item is not estimable.
- Per-world flags and the P3b-1 − P3b-0 distance for each item are reported.
- The eight measurements are also reported on the **full** views (val vs each full synthetic hold), as descriptive
  W1 distances with no pass claim.
- **Marginals.** The agent-level items are also reported by cohort and by account-age tercile, descriptively. Roles
  are already separate.
  - Tercile cut points: train-defined terciles of P3a `log_age` over real-train pre-existing agents.
  - Born agents form their own bin.
- This is descriptive SPEC-style thresholding, not an equivalence test.

### SPEC §7.3 descriptive topology

**Daily topology.**
- Actual undirected transaction graphs: agents plus boundary counterparties.
- Venues included (primary) and excluded.
- Per (A, B, synthetic) triple and UTC day, m_d = the minimum daily active focal count; m_d active agents are drawn
  per side, keeping all their transactions that day.
- Metrics: GCC ratio, component count, max k-core, max-core fraction (vertices in the highest nonempty k-core over
  all vertices), degree assortativity (undefined for constant-degree endpoints) and transitivity.
- The daily selection draws depend only on daily activity, which is identical across attachment arms. They are
  drawn once per (world, policy arm, seed, day), with seed-0..19 substreams `SeedSequence([6000 + s, w, day])`, and
  reused for P3b-0, P3b-1 and the anchor.
- Every m_d and every excluded day is stored.
- Reported: mean absolute paired daily discrepancy vs the A/B scale, on days defined on all sides.
- No metric passes if fewer than 80% of nonempty common days are defined.

**Clauset** (`powerlaw` 2.0.0, `lpw.p3b.topology.clauset`).
- **Inputs.** Full-window unique degree and transaction-count degree (discrete), strength (ETH, excluding fees) and
  per-transaction ETH (continuous). They are fitted on the full val view and on each full synthetic hold view.
- **xmin.**
  - Discrete: powerlaw's full KS-minimizing search.
  - Continuous, a **pre-freeze change**: the KS-minimizing candidate among the 100 empirical quantiles 0, 0.01, ...,
    0.99 of the sample being fitted, with closed-form MLE alpha.
- **Why the change** (preliminary v0 measurements on val, earlier implementation). The full continuous search took 412 s for 10 bootstrap refits on val strength, about 1.1 h per
  view at 100 refits. The grid search agrees closely with the full one on that data: xmin 34.2 vs 29.4, α 1.637 vs
  1.631, D 0.033 vs 0.030.
- **Tests.** The lognormal likelihood ratio uses powerlaw at the chosen xmin.
- **Goodness of fit.** 100 semi-parametric bootstrap refits, each with its own xmin search. The tail draws come from
  the fitted law: Pareto (continuous) or the exact discrete zeta law (`zeta_sample`). The body is resampled below
  xmin. p is the share of refits
  with D ≥ the observed D.
- **Minimum.** At least 50 tail points; undefined fits are reported.
- **Corrections after Sol r1, before freezing.**
  - The continuous KS is the two-sided supremum, including each empirical jump's left limit.
  - The discrete bootstrap draws from the exact fitted zeta law (`zeta_sample`), not rounded Pareto draws.
  - gof_p is defined only if at least 90 of the 100 attempted refits are valid.
- **Record.** The earlier implementation's pre-freeze output is kept as `prefreeze_topology_v0.*`, computed on val;
  the cost check was rerun on train.
- **Cost.** About 4 min for all four fits on the full train view (a train-only cost check, 127k nodes and 356k
  transactions). The val-size and hold-size views being fitted are about 5-6× smaller.
- **Preliminary v0 val results** (label-blind; earlier implementation, with one-sided KS and rounded-Pareto discrete
  draws; not results under the corrected protocol). Unique degree α 1.92 (xmin 2), lognormal favoured (LR −5.3). Transaction
  degree α 2.16 (xmin 24, GoF p 0.82). Strength α 1.64. Amount α 1.67 (lognormal favoured, p 0.004).

**Coupling diagnostics** (reported):
- assigned and recovered group sizes (group- and agent-weighted, H, 1 / H);
- agent↔agent fractions with stated denominators (synthetic 0);
- cross-group shared-node incidence;
- same-second co-creation.

These are descriptive only, not extra ways to pass.

## P9. Utility (`lpw.p3.utility`, unchanged settings, seed 0)

**Models.**
- Sources: real train, or one attached synthetic train-size world; worlds are never pooled.
- Feature sets: L = P3a `local` (LOCAL + BAL + VENUE), L+S, S, full, full+S and confonly.
- Detectors: LR / RF / XGB / LGBM; XGB is primary and the others are sensitivity checks.
- F1 folds: the training source's five whole-group folds use the C2ST rule below (largest-first into the
  least-populated fold, ties by a seed-0 permutation), on real Louvain and synthetic assigned groups.
- Scores: AUC and AP; F1 for XGB full+S, with a training-source 5-fold OOF threshold. The folds are whole groups on
  both real and synthetic sources, and ties go to the highest threshold.
- Real val STRUCT uses the val graph; synthetic STRUCT uses that world part's graph.

### A. Incremental structural-signal transfer (primary; kct, XGB)

**Contrasts.**
- R = AUC(TRTR, L+S) − AUC(TRTR, L).
- T_j = mean_w [AUC(TSTR_j,w, L+S) − AUC(TSTR_w, L)].
- D = T_1 − 0.8 R.
- H = T_1 − T_0.

**Bootstrap.** 1,000 seed-0 whole-group bootstraps of the real val evaluation set; fixed predictions are re-scored,
with no refits. At least 950 valid two-role draws are required, otherwise the inferential outcome is NOT TESTABLE.

**Outcome rule.**
- **NOT TESTABLE:** R < 0.020, or the evaluation set lacks a role.
- Otherwise:
  - **SUPPORTED:** D ≥ 0 with a one-sided 95% lower bound ≥ 0, and R's one-sided 95% lower bound > 0;
  - **CONSISTENT:** D ≥ 0 but the D condition or the R condition fails;
  - **NOT SUPPORTED:** D < 0.
- Reported: two-sided 95% intervals for R, T_1 and D.

**Status of the thresholds.** The 0.020 floor is a minimum useful effect, not a val calibration. The 0.8 extends
P3a's convention.

### B. Learning advantage

**Outcome rule** for H:
- **SUPPORTED:** H > 0, with a one-sided 95% lower bound > 0;
- **CONSISTENT:** H > 0, with a lower bound ≤ 0;
- **NOT SUPPORTED:** otherwise.

The R floor does not apply. A gate fitted to data is learned by provenance; whether that learning helps is a
separate claim.

### Secondary analyses (none changes the verdicts)

- **STRUCT-only excess-AUC retention**, with P3a's rules: NOT TESTABLE if TRTR(S) < 0.60, then D_S = (TSTR − 0.5) −
  0.8 (TRTR − 0.5).
- **R / T / D / H on full and full+S.**
- **Matched pairs.** The saved P3a 348 pairs, using full-split STRUCT (never recomputed on the subset). Fewer than 200
  pairs makes this NOT TESTABLE. Also full+S − confonly.
- **Giant group, fixed scores.** Utility with the largest real val group (by size, then id) removed.
- **Giant group, graph removal.** Remove that group, recompute val STRUCT and re-score the trained models.

These explain leverage; they do not rescue a result.

**Combined wording.**
- "Learned structural signal transfers" needs A and B to be supported.
- "The minimal coupling model meets the core fidelity targets" needs CORE PASS.
- All three together give the strongest wording.

## P10. Seam, shortcut, inherited criticism

**C2ST**, per role, XGB.
- Source sizes are balanced by seed-0 subsampling.
- Five group-disjoint folds on both sides: real Louvain groups and synthetic assigned groups, placed largest-first
  into the least-populated fold, with ties broken by a seed-0 permutation. NOT TESTABLE if five valid two-domain folds
  cannot be made.
- Within each training fold, each domain's total weight is half the fold's rows.
- Scored on pooled OOF probabilities.
- Feature sets: S-only (the new structural seam), L+S (combined) and full+S (secondary).
- **Pass:** both roles ≤ 0.70, and scam − normal ≤ 0.05.

**Shortcut audit.**
- The top 20 full+S features by |signed AUC − 0.5| on real train, plus all six STRUCT features and the five balance
  sentinels.
- Compared: synthetic train vs real val signed AUC; flag |Δ| > 0.10, with the mean over worlds.

**Parity tests.** Difficulty and profile-gain parity on full+S, with P3a's definitions.

**Not repeated.** The encoder, DCR and coverage are not repeated: their inputs are locally invariant, and they do not
see the graph.

**Expected and reported, not hidden:**
- the bipartite synthetic graph has no agent↔agent edges, so triangles come only from shared nodes;
- group-exclusive pools cannot reproduce real cross-group sharing. Pre-freeze A4: 21% of shared train nodes, 36% of
  shared incidences and 45% of events on shared nodes are cross-group. The largest real sharing component has 6,888
  agents, against 341 with only single-group nodes;
- the inherited local seam is untouched.

## P11. Checkpoint 0 (world 0, train + hold, kct and B, P3b-0 then P3b-1; before any utility or C2ST)

**Checked:**
- hashes and schema; caps and no-event counts;
- local invariance (exact) and namespaces;
- group-slot totals and proposal counts;
- empty-pool and opportunity counts; attachment shares by role and direction; choice-set sizes;
- within-agent collisions (must be 0); same-seed reproducibility;
- earlier-second witnesses, no future pool access, no node shared between groups or worlds, and venue stability;
- STRUCT finite values and ranges;
- hand-built checks (`tests/test_p3b.py`).

**Failure policy.**
- Safety, schema, injection or causality failures stop the readout.
- A demonstrated bug may be fixed, logged and rerun with the same seeds.
- Weak attachment, poor OOF quality, giant-group leverage or poor fidelity trigger no search. One mechanistic
  diagnosis is written, and the fixed comparison is completed.

## P12. Development readout, freezing, sealed opening

**Before any val output.** All train-fit diagnostics and Checkpoint 0 decisions are written first.

**The readout.** Worlds 1-2 are attached, the readout code is reviewed, and then one complete val readout runs, with
every outcome kept.

**Hashed:** code, environment, inputs, group maps, kernel parameters, feature order, O rules, raw predictions,
matching, noise-floor samples and folds, and bootstrap recipes.

**Joint test addendum** (written later, before the user authorizes the opening).
- P3a: kct primary, B control, worlds 0-2, features and thresholds unchanged, no sweep arm.
- P3b: both attachment arms, whatever their development outcome.
- Models: the prior, kernel, metadata conventions, encoder and population models, all unchanged.
- Tapes: the same saved tapes and attached worlds. The size-matched structural subsampling follows the formula, with
  no tuning to test sizes or labels.
- Test STRUCT is computed on the test split's own graph. Matching and age bins are frozen. The clean-subset
  sensitivity check applies where the SPEC requires it.
- One authorized job produces the complete report, with no selective peeking or retries. Bug fixes, if any, are
  reported separately from the original result.

## P13. Budget and deviations

**Budget.**
- Build: at most two days.
- No generation and no GPU.
- At most 1 CPU-hour for the sampler, attachment and fitting with CV.
- At most 3 CPU-hours for the full development readout.

These are caps to be validated with measured throughput.

**Measured pre-freeze:**
- train replay 4.4 s;
- synthetic attachment 2-3 s per world part;
- pre-freeze A/B 30 s;
- daily topology 0.7 s per half view (both venue variants);
- Clauset about 4 min on the full train view (corrected implementation), less on the smaller val and hold views.

**Daily-topology definedness** on val halves: venues included, every metric is defined on all 365 days. Without
venues, at most 0.1% of days are undefined, except assortativity at 2.6%.

**Over-budget rule.** If a prescribed fit or metric will not fit in the budget, stop and ask; nothing is silently
reduced. A single run projected over 4 h needs the user (SPEC §9). A technical or budget failure is INCOMPLETE.

**Deviations:** none yet. The pre-freeze change to the continuous Clauset xmin search (§P8) was made before freezing.
