# P3a pre-registration: sampled-population, uncoupled policy transfer on local node features

Written 2026-11-07, before any P3a utility readout. It implements the approved plan `reports/p3_plan.md` (D1-D4 = yes).
SPEC A2 scope: the 2018 slice; frozen generator **kct** (primary) with **B** as control.

Code: `src/lpw/p3/{population,observe,features,matching,utility}.py` and `scripts/p3a/{freeze,generate,checkpoint0,readout}.py`.
Tests: `tests/test_p3a.py`. Changing anything listed here after the first readout counts as a deviation. Each deviation
is recorded in §14 with its reason, and the frozen-protocol result is reported next to the changed one.

## 1. Claim and scope

- **Claim tested:** a node-level role detector trained on a fully synthetic, uncoupled population keeps at least 80% of
  the excess AUC that a detector trained on real data achieves. Features are local node features. Evaluation is on
  real held-out groups.
- **What the claim covers:** T4 (TSTR, partly), C1 (profile-gain parity) and T2 (the seam, local features only).
- **What it does not cover:** the coupled-world contribution, and P3 is not qualified under the unchanged SPEC.
  Structure-aware seams and §7.3 are P3b.
- **Development data:** real **val**, already used heavily in P1, so it is not fresh confirmation. The sealed test is
  opened once, at the end (§13).

## 2. Population (frozen: `runs/p3a/2018/population/`)

- **Order of sampling.** Role ~ Bernoulli(π_train = 0.09327). Then cohort, activation week, raw profile and b0 come
  from `ProfileGen`, and the plan (count bucket, amount bucket) from the default `PlanPrior` (no exposure inputs).
- **Fitting.** Both models were fitted once on the train split by `scripts/p3a/freeze.py`, with seed 0.
  `freeze()` refuses to overwrite.

| artefact | sha256 |
|---|---|
| `profile_gen.pt` (net, binner, p_born, week_hist) | `d001e09e51c2cf77477745d928e5e933bde9b7e6316f41bab45c082e7277b20f` |
| `plan_prior.pkl` | `fc41cbd3a6bc4f0273ecb282fea22c7875a204cf96549994a1a909961819cdfa` |

- **Val use.** ProfileGen's epoch was chosen by early stopping on val NLL (roles included), as in P1. This is the
  only val input to the population models; nothing reads test.
- **Fit diagnostics.** ProfileGen val NLL 31.56 (116 epochs). PlanPrior val NLL 3.144. p_born: normal 0.718, scam
  0.904.
- **Activation fix.** Activation is uniform inside the sampled week's part of W. Before the fix, a full-week draw
  clipped to `w_end - 1` piled the partial last week onto the final second. A test covers this.
- **Raw profiles** are persisted with every population (`pop_{train,hold}.npz`).
- **Population seeds.** World w uses a train-size population (17,454 agents) with seed 1000 + w, and an independent
  val-size **holdout** population (3,741) with seed 2000 + w.

## 3. Worlds and the observation operator

- **Worlds.** w ∈ {0, 1, 2}, with no seed shopping. World 0 is Checkpoint 0 (§9) and the first of the three
  replications.
- **Arms.** Both arms run the same populations with generation seed 100 + w (common random numbers), cap 10,000
  events, plan inference on. Checkpoint hashes are in `runs/p1_freeze/2018/manifest.json`.
  - kct: `lpw_plan_prog_shallow_kct`, `runs/p1_shallow/2018/kct`;
  - B: `lpw_plan_prog`, `runs/p1_r0/2018/lpw_plan_prog_rp`.
- **O (synthetic), `lpw.p3.observe.observe_synth`:**
  - times are floored to whole recorded seconds;
  - IN_exo becomes IN;
  - venue counterparties are keyed by venue index, because the generator opens a new partner entry on every VENUE
    draw but an observer sees one address;
  - asserts: no IN_agent, the dust floor, finite values, every event inside [origin, w_end).
- **Retention.** Synthetic agents with no recorded event are dropped and counted per role, since real agents are
  active in W by construction. If more than 2% of a population is dropped, it is a Checkpoint 0 finding.
- **Real reference, `observe_real`:**
  - IN_exo and IN_agent both become IN, so every incident edge is kept (**primary**);
  - an **owned-only** sensitivity analysis drops IN_agent. It matters because 0.4% of events are IN_agent, but
    13-19% of scams receive one vs 2% of normals;
  - the uncoupled world cannot produce IN_agent, so this gap is part of the seam by design.

## 4. Features (`lpw.p3.features`, one function for real and O(world))

| block | features |
|---|---|
| CONF | cohort, log age, activation fraction of W, log exposure days, days from origin to first event |
| PROF | the 20 profile fields, log-transformed (zero for born-in-W agents) |
| LOCAL | counts (all / OUT / IN), out-fraction, active days, span, end idle days, median log gap, same-second share, max one-second burst, max events in a day, hour entropy, weekend share, OUT and IN log-amount median and max, amount SD, OUT and IN totals, distinct counterparties, repeat share, top-counterparty share, distinct senders / receivers, median log fee |
| BAL | end balance (b0 + observed flows), net flow, ends near zero (\|end\| ≤ 1e-4 ETH, the dust floor), minimum running balance, share of OUTs leaving balance < -1e-4, share of OUTs leaving \|balance\| ≤ 1e-4 |
| VENUE | venue share of events, distinct venues, top-venue share of venue events, OUT share of venue events (no venue identities) |

- **Sentinel values.** An undefined statistic is -1, or -9 for a log amount or log fee. The value is the same on
  both sides.
- **STRUCT.** Cross-agent sharing is not a P3a block.

**Feature sets:**

| set | blocks | used for |
|---|---|---|
| `full` | all | primary |
| `noconf` | LOCAL + BAL + VENUE | §7.5 (2): without the confounders and their proxies |
| `confonly` | CONF + PROF | conditional baseline |
| `noprof` | full minus PROF | profile gain |
| `nobal` | full minus BAL | defect-1 ablation |
| `local` | LOCAL + BAL + VENUE | C2ST primary |

## 5. Detectors (`lpw.p3.utility`)

**Settings** (seed 0, no tuning):

| detector | settings |
|---|---|
| LR | StandardScaler + C = 1 |
| RF | 500 trees, min leaf 5 |
| **XGB (primary)** | 300 trees, depth 4, lr 0.05, subsample / colsample 0.8 |
| LGBM | 300 trees, lr 0.05, 15 leaves, subsample / colsample 0.8 |

- **Training sources:** real train (TRTR) or one synthetic train-size world (TSTR). Scores are AUC and AP.
- **F1** is reported for the primary detector on `full` only. Its threshold maximizes F1 on 5-fold out-of-fold
  predictions of the **training source**, never the evaluation data. Every distinct OOF score is a candidate, and
  ties go to the highest threshold.

## 6. Matched evaluation (§7.5)

- **Propensity model:** logistic regression for role, fitted on **real train** only, on these confounders:
  - log age;
  - days active and days since active;
  - normal sent / received counts and ERC-20 sent / received counts;
  - ETH sent / received;
  - cohort (= profile missingness);
  - activation fraction of W (era inside the slice).
- **Support and caliper,** frozen from train: support is [max of the two role minima, min of the two maxima] of the
  train logit, and the caliper is 0.2 × the train logit SD.
- **Matching.** 1:1 without replacement: exact on cohort, then greedy nearest neighbour on the logit within the
  caliper, highest-propensity scams first. Pair IDs are saved (`matched_pairs_val.json`). Below 200 pairs the result
  is "not testable".
- **Design facts known before any readout:**
  - val gives 348 pairs (348 of 349 scams are in the support);
  - every confounder has |SMD| ≤ 0.05 after matching;
  - the confounder-logit AUC inside the matched set is 0.501.

  The match was redesigned once before any utility readout, from a two-pointer greedy that left act_frac SMD at 0.33
  (§14).
- **Reported three ways** for every utility number:
  1. `full`;
  2. `noconf`;
  3. matched AUC at 50% prevalence, for `full` and the `confonly` baseline. Behavioural gain = full - confonly,
     inside the matched sample.

## 7. Headline: TSTR vs TRTR (per arm; kct is the headline arm and B the control; no arm is selected on results)

- **Gate.** If TRTR AUC (full, XGB, real train → val) is below 0.60, the outcome is **NOT TESTABLE**, and no ratio
  or outcome is computed.
- **Statistics.**
  - Excess-AUC ratio = (mean over worlds of TSTR AUC - 0.5) / (TRTR - 0.5).
  - Paired contrast D = (mean TSTR - 0.5) - 0.8 (TRTR - 0.5).
  - D gets a whole-group bootstrap over real-val groups (1,000 draws, seed 0). In every draw, all worlds and TRTR
    are rescored on the same resampled agents.
- **Outcome:**

| outcome | condition | meaning |
|---|---|---|
| **SUPPORTED** | one-sided 95% lower bound of D ≥ 0 | ratio ≥ 0.8 with confidence |
| **CONSISTENT** | D ≥ 0 but the lower bound < 0 | ratio ≥ 0.8 on the point estimate only |
| **NOT SUPPORTED** | D < 0 | the 95% CI of D is reported; an upper bound < 0 means "confidently below 0.8" |

- **What the uncertainty covers.** It is evaluation uncertainty only. Three worlds cannot estimate world-to-world
  variance, so each world's TSTR and the SD are reported.
- **Also reported, not gated:**
  - the other detectors and feature sets;
  - the matched results;
  - owned-only TSTR / TRTR;
  - the `nobal` ablation.

## 8. Criticism tests (per arm; per world, then the mean over worlds is the arm's result)

**The arm's result is the gate applied to the mean over worlds** (e.g. the mean C2ST AUC per role, the mean
synthetic AP, the mean single-feature Δ, the mean DCR ratio). Per-world pass flags are reported only as spread.
TSTR AUC / AP / F1 are also averaged over worlds per detector and set.

| test | rule |
|---|---|
| Difficulty parity (T3) | Synthetic train → synthetic holdout AP, at real-val prevalence (subsampled, mean of 20 draws), within ±0.05 of real train → val AP at the same prevalence. Primary detector, `full`; others reported. |
| Detector-ranking agreement | Spearman of the 4 detectors' AUC, synthetic → synthetic vs real → real. Reported only: n = 4 has no useful test. |
| Profile-gain parity (C1) | Gain = AUC(full) - AUC(noprof), on synthetic → synthetic and on real → real. Pass: same sign and \|difference\| ≤ 0.05 (AUC primary, AP reported). |
| Seam C2ST (T2) | Per role: real val vs the independent holdout world, sizes balanced by subsampling. 5-fold out-of-fold primary detector; real agents folded by group. Pass: AUC ≤ 0.70 for each role and scam - normal ≤ 0.05. `local` is primary, `full` secondary. The top-10 single-feature seams are exploratory. |
| Shortcut audit (C2) | The 20 features with the largest \|AUC - 0.5\| on **real train**, plus the balance sentinels (end_balance, net_flow, ends_near_zero, sweep_out_share, min_balance). Single-feature role AUC on the synthetic train world minus on real val. Flag \|Δ\| > 0.1. |
| DCR (memorization) | Within role, the nearest real-train record on `full` features scaled by real-train median / IQR (Euclidean). Queries are the synthetic holdout world vs real val, at equal size per role. Pass: 5th-percentile ratio (synthetic / val) ≥ 1. A val 5th percentile of 0 makes the ratio **undefined**, not passed. Exact copies are counted. |
| Mode coverage (C2), primary | Improved precision and recall (Kynkäänniemi et al., k = 5), per role, of the synthetic holdout world vs real val, on standardized `full` vectors. Reported, no gate. |
| Mode coverage (C2), encoder tranche | An evaluation-only encoder embedding. Its recipe is frozen by an addendum **before** the encoder is trained, and the addendum uses no TSTR result. Until it runs, C2 is incomplete. |

- **Failed tests stay reported** as negative results.
- **Gates within ±0.05** are set against the P1 replica scale. They are not recalibrated after the readout.

## 9. Checkpoint 0 (world 0, both arms; `scripts/p3a/checkpoint0.py`)

- **Safety stop** (exit 2): more than 0.1% of a population's streams reach the 10,000-event cap, a non-finite value,
  or an O assertion failure. A stop means no readout. It is reported, and if the cause is a fixable bug, the fix is
  disclosed.
- **Capped streams below that share** are kept, truncated at the cap (an agent observed up to the cap), and listed
  with role, cohort and plan. Streams longer than the longest real stream (4,839 events) are counted. The readout
  adds a sensitivity analysis with capped agents removed (headline statistic only).
- **Enforcement.** The readout's loader re-applies the 0.1% stop to every population it loads, so a replication
  that fails it cannot enter the readout.
- **Population checks** against real train:
  - role and born rates, activation quantiles, and plan histograms by role × cohort;
  - origins on the last second;
  - a profile + confounder C2ST of the pre-existing agents.
- **Fidelity:** M1-M16 for the holdout world vs real val (unpaired, amended scorer).
- **Weak fidelity**, i.e. either role's M-pass below 0.50: one written diagnosis, with no generator change, before the
  replications. The readout goes ahead unless there is a safety stop.

## 10. Known defect 1 (no balance sweeping): bounded repair after the first readout

- **Trigger, either of:**
  - the shortcut audit flags any balance sentinel for kct;
  - the synthetic ends-near-zero share is below half the real-train share for either role.
- **Budget:** one learned "sweep" atom in the amount head (an OUT of amount = balance - fee), fitted on train only.
  At most 1 day of build and 2 h GPU, as one bounded P1 experiment under A2.
- **Acceptance, on these alone:**
  - mechanism: the synthetic ends-near-zero share per role is within ±50% (relative) of real train;
  - safety: the §9 rules;
  - fidelity: holdout M-pass per role no lower than kct's by more than 0.10.

  **Never on TSTR or any utility number.**
- **Reporting.** The kct baseline readout stays reported. The repaired generator gets the same readout, with fresh
  worlds 10-12. The `nobal` ablation is reported for both, and the full-feature result is never replaced by `nobal`.

## 11. Exploratory (labelled as such, not used for any gate)

New interactions; per-cohort and per-age subgroup probes; group-remapping controls; single-feature seam rankings;
alternative O variants.

## 12. Aggregation and outputs

- `runs/p3a/2018/world{w}/` holds the populations, events, `gen.json` and `checkpoint0.*`.
- `runs/p3a/2018/readout.{json,txt}` holds every number above, per world and aggregated.
- Real-side predictions are never pooled across worlds.

## 13. Sealed test

- **When:** opened once, after the user authorizes, and after deciding which stages belong in the frozen evaluation.
- **How:** the same scripts with real train → **test**. TRTR is also reported on the clean test subset (SPEC §5.2).
- **Afterwards:** nothing is tuned, and negative results are confirmed too.

## 14. Disclosures and deviations

**Before any readout (2026-11-07):**
1. **Feature design looked at real-val role signal.** While building, I printed real-val single-feature role AUCs
   and a 400-agent synthetic smoke world's feature means (ends-near-zero 0.3% synthetic vs 13% real). Two changes
   followed:
   - `first_delay_days` moved from LOCAL to CONF, because it marks cohort (0 for most born agents, never 0 for
     pre-existing ones);
   - the near-zero threshold was set to the dust floor, which reproduces the P1 exit's real zero-balance shares
     (normal 10.5%, scam 39.8% on val).

   No detector, TSTR or C2ST was run on val before this pre-registration.
2. **Matching was redesigned** (two-pointer greedy → exact cohort plus nearest neighbour) after its balance on val
   came out poor. This is a design-diagnostic use of val labels.
3. **Plumbing smoke tests** of Checkpoint 0 and the readout ran on a tiny scratch world (1,500 + 750 agents) to
   check that the code runs. Their utility outputs were not read. They are not results.
4. **The cap rule changed after the smoke world's safety lines were read.** The plan said that caps stop the run.
   In the smoke world:
   - B capped one stream and kct reached 9,627 events. Both streams came from sampled pre-existing normal agents
     with extreme, partly incoherent profile draws (e.g. 85k unique peers with 91 sends) and the open-ended plan
     bucket (≥ 1,000 owned events);
   - P1 prior mode never met this case, because it kept real profiles;
   - about 0.27% of sampled agents fall in that bucket (real train 0.22%), so in a 17k world "any cap stops" would
     fail almost surely.

   The rule is now a 0.1% capped share, plus the capped-removed sensitivity analysis. The profile + confounder C2ST
   at Checkpoint 0 reports how coherent the sampled profiles are.

**After the readout (`reports/p3a_readout.md`):** no protocol change. The Checkpoint 0 diagnosis (world 0) and one
exploratory C2ST probe (with and without BAL) are labelled as such. Their scripts are in `reports/p3a/`.

## 15. Addendum A (2026-11-07, before fitting): the defect-1 sweep repair

Written after `reports/p3a_readout.md` fired the §10 trigger (kct balance sentinels flagged). The user approved
(D4, then "yes go"). The design is Sol's consult (`sol_consult_sweep/consult.md`, scratch). This addendum replaces
§10's informal wording where the two differ.

**Status.** This is development evaluation informed by the first val readout, not fresh confirmation. It does not
change any P1 qualification claim, and success would not qualify P3 overall.

### A.1 Mechanism (kct only; one candidate)

- **What is added.** One frozen-backbone Bernoulli **sweep gate** on OUT events. Everything else stays frozen: the
  kct base, the kct offsets, the plan prior, the population models and FeePool.
- **Branch.** On a sweep, amount = b - fee, persisted in float64 as `eth`, with `amt = log10(eth)`. Otherwise the
  unchanged truncated-MoG draw is used, with no affordability clipping.
- **Order and randomness:**
  - the fee is the existing FeePool draw, reused for eligibility, amount and ledger;
  - the MoG draw is still consumed for every event;
  - the branch uses a separate NumPy RNG with seed 3000 + world;
  - with the gate disabled, generation is bit-identical to the current code (a unit test checks this).
- **Eligibility:** an OUT, with b and fee finite, fee ≥ 0, and 1e-4 ≤ b - fee ≤ SUPPLY. Ineligible rows never
  sweep and are counted.
- **A sweep does not end a stream.** Later IN, OUT and repeat sweeps are allowed.

### A.2 Labels and fit (real train only)

- **Label.** On eligible train OUTs: y = 1 iff |b - eth - fee| ≤ 1e-4, where b is the pre-event ledger (`store.bal`
  = b0 + all observed incident flows; an exact replay was verified). All other eligible OUTs, including
  proxy-negative ones, have y = 0.
- **Features** (fixed; no search):
  - the frozen amount-head input `amt_x` of the target OUT (kct, plan inference on, real train plan,
    generation-matched exact windows);
  - one-hot role, cohort, count-plan bucket, amount-plan bucket, and `progress_state(plan_n, n_own + 1)`;
  - the scalars log1p(n_own), log10((b - fee) / 1e-4), fee / b, and log1p((w_end - t) / 86400).
- **Standardization.** Continuous coordinates (amt_x and the scalars) use train statistics. One-hot coordinates are
  left as they are.
- **Fit.** Logistic regression minimizing summed BCE + 10/2 · ||β||², with the intercept unpenalized.
  Deterministic L-BFGS from zero, max_iter 500.
- **Diagnostics only.** Three whole-train-group folds (seed 0) give OOF BCE and calibration by role. They select
  nothing. The final gate is refitted on all train once.
- **Banned:** class weighting, a threshold, calibration to rollouts, val stopping, any change of ε or the label
  band.
- **Reported:** the gate BCE, separately. The frozen P1 NLL / B1 results stand as published, labelled as the frozen
  backbone's. No repaired full-event NLL is claimed.

### A.3 Worlds and seeds

- **Worlds** w ∈ {10, 11, 12}: populations as in §2 (seeds 1010-1012 and 2010-2012), generation seed 100 + w, sweep
  seed 3000 + w.
- **Arms:** `kct_sweep` (candidate) and unmodified `kct` (the control on the same populations). There is no
  repaired B. The original worlds 0-2 and their readout stay archived.

### A.4 Acceptance (decided and written to `runs/p3a/2018/sweep/acceptance.json` before any utility or C2ST number)

| gate | rule |
|---|---|
| Mechanism | For each role, the share of retained holdout agents ending at \|b0 + Σ flows\| ≤ 1e-4, averaged over the three worlds, lies in [0.5 R_r, 1.5 R_r]. R_r is frozen from real train (all-incident O): **R_normal = 0.13345, R_scam = 0.40725**, so the bands are [0.0667, 0.2002] and [0.2036, 0.6109]. |
| Fidelity | For each role, mean over worlds of (M-pass kct_sweep - M-pass kct) ≥ -0.10, on the same holdout population. Unchanged scorer, unpaired vs real val, seed 0, 20 half-splits. |
| Safety | Every population of both arms individually passes §9. |

- Boundaries count as a pass. A missing or undefined quantity fails.
- No world is stopped early except for safety or budget.

### A.5 Readout and failure

- **Readout.** The §5-§8 readout runs once on worlds 10-12 for both arms (`--tag _sweep`), whatever the decision,
  and a rejected result is labelled REJECTED. It also reports the fixed attribution diagnostics:
  - per-role BAL-only C2ST and LOCAL + VENUE C2ST without BAL;
  - per-role single-feature seam AUC for every local feature;
  - per-role sweep-event share, the share of agents ever sweeping, the terminal near-zero share, the negative-OUT
    share, the share of sweeps that are the last OUT, and the share of sweeping agents with a later IN.

  None of these is a tuning target.
- **Budget:** one build day and ≤ 2 GPU-hours in all.
- **Stops and failures:**
  - a safety failure stops the experiment, with no readout;
  - a budget or technical failure is reported as inconclusive;
  - nothing is retuned: no seed replacement, no wider band, no second head, no backbone change, no utility-based
    rescue. A demonstrable bug fix is allowed within budget, disclosed, and rerun with the same seeds.

### A.6 Outcome (2026-11-07)

**REJECTED** on the scam mechanism gate (0.126 vs band 0.204-0.611). Normal mechanism, fidelity and safety pass.
The labelled readout and diagnostics were run once, as pre-registered: `reports/p3a_sweep.md`. Deviations: none.

