# P3 plan (2026-11-07; revised after a Sol design round; awaiting user approval)

**Scope.**
- SPEC A2: the 2018 slice. Frozen generator kct, with B as control.
- The research question comes first (SPEC §0-2, §7.4).
- Nothing is run yet. Sol's notes: scratch `sol_consult_p3/consult.md`.

## Claim of stage P3a

**"Sampled-population, uncoupled policy transfer on local node features."**
- It answers part of T4 (TSTR), C1 (profile-gain parity) and T2 (the seam, on local features only).
- It does **not** complete the coupled-world contribution, and it does not qualify P3 under the unchanged SPEC.
- A structure-aware seam cannot pass in an uncoupled world. The missing non-venue sharing alone gives a domain AUC
  of about 0.76 (normal) / 0.83 (scam) on val (Sol's probe).

**Why uncoupled first** (measured on 2018 real streams; degrees within the split, no test agents):
- The share of counterparties touched by >= 2 agents has role AUC 0.30, the venue share 0.41, and non-venue sharing
  0.49.
- Near-chance marginal AUCs do not prove sharing carries no role signal. They do make an uncoupled tabular test
  informative.
- Non-venue sharing is mostly within one group: 78% by counterparty, 61% by agent incidence. This is partly induced by
  how Louvain builds the groups.

## P3a steps

### 1. Freeze the protocol before any readout (a pre-registration, `reports/p3a_prereg.md`)

Fixed in advance:
- claims, artefacts and hashes (policy, profile model and bins, normalizers, activation rates, fee pool, RNG seeds);
- population and retention rules;
- feature blocks;
- the primary detector and every detector's settings, preprocessing and F1-threshold rule;
- thresholds, the matching rules, C2ST splits, the DCR recipe and the encoder recipe;
- aggregation, uncertainty, stop rules and the repair rule.

Exploratory, and labelled as such: new interactions, subgroup probes, group-remapping controls.

### 2. Build (about 0.5-1 day)

- **Population sampler.**
  - Roles are Bernoulli(π_train). Then come cohort, activation week, raw profile and b0 (`ProfileGen.population`),
    then the plan (`PlanPrior`).
  - Raw profiles are persisted. The final-week activation clipping is fixed.
  - The active-agent estimand is explicit. Real agents need at least one event in W, while synthetic agents can
    have zero (0.45% of current replicas). Rule: keep only synthetic agents with at least one recorded event,
    and report how many are dropped.
- **Observation operator O.**
  - Project synthetic times to integer recorded seconds. Synthetic `ts` is float64 and 96% fractional; real `ts` is
    int64. Left unprojected, this alone gives domain AUC 0.96-0.97.
  - Assert the dust floor and agent-incident edges.
  - Map both incoming kinds to the observable IN, so real streams keep every incident event.
- **Feature extractor** (`src/lpw/eval/features.py`). One function for real streams and O(world). Frozen blocks:
  - confounders: age, cohort, era and their derived proxies;
  - profile;
  - local behaviour, including the balance trajectory;
  - venue-agnostic venue usage;
  - cross-agent structure: reported only, outside the P3a primary.

  The primary analysis uses every observed incident event; an owned-only version is a sensitivity analysis.
- **Matched evaluation (§7.5), done properly.** A train-only propensity model, frozen support and caliper rules,
  saved pair IDs and balance diagnostics. Below 200 pairs per role, the result is "not testable". Matched AP is at
  50% prevalence.
  - The existing code uses an all-split score SD for its caliper and only counts pairs: the 348 is a feasibility
    count.

### 3. Checkpoint 0 (the first paired B / kct world: train-size world plus an independent holdout world; about 12 min GPU)

- **What it covers:** both roles, both cohorts, the activation x plan joints, profile joints and arithmetic,
  continuous fidelity distances and M1-M16. All of it against matched-size frozen references.
- **Safety or schema failure** (caps, runaways, non-finite values, broken invariants): stop immediately.
- **Weak fidelity:** one bounded diagnosis, not a search.

### 4. Three replications (about 36-40 min GPU in all)

- Each replication is a train-size world plus an independent holdout world, for B and kct.
- The worlds are separate, not pooled. No seed shopping: the generation seeds are fixed in the pre-registration.

### 5. One frozen tabular readout (about 20-40 min CPU)

Development data is real **val**. It is not fresh confirmation, since it was used heavily in P1.

- **TSTR vs TRTR.**
  - LR, RF, XGB and LGBM, with one named primary.
  - Metrics are averaged over worlds (predictions are never pooled). Each world and the spread are reported.
- **Gates.**
  - TRTR AUC >= 0.60.
  - Excess-AUC ratio >= 0.8.
  - A paired contrast D = (mean TSTR - 0.5) - 0.8 (TRTR - 0.5), bootstrapped over real-val groups.

  Evaluation uncertainty is reported separately from world-to-world variability, which 3 worlds cannot pin down.
- **On the synthetic holdout worlds:** difficulty parity, detector-ranking agreement and profile-gain parity.
- **C2ST seam test:** held-out real groups against independent synthetic populations, balanced within role. Local
  features only.
- **Shortcut audit:** the selection rule and balance sentinels are frozen.
- **DCR:** fixed scaling, distance, conditioning and reference size. If the real 5th percentile is 0, the ratio is
  "undefined", not passed.
- **Failed criticism tests stay negative results.**

### 6. Encoder and mode-coverage tranche (up to 1 h GPU, plus scoring)

This is scheduled, not optional. Until it runs, C2 is incomplete.

### 7. Known defect 1 (no balance sweeping)

- **When.** One bounded repair, **after** the first frozen baseline readout.
- **Fixed now:** the trigger, the budget (one learned "sweep" amount atom), and a balance-block ablation.
- **Selection** is on mechanism, safety and fidelity, never on improved TSTR.
- **What it must not hide.** The original baseline is kept, and the full-feature result is not suppressed because
  dropping balances helps.

## P3b (conditional; a short protocol, not on the critical path)

- **Cheap structural-identity control, no learning.** Assign group pools post hoc while keeping each agent's own
  partner distinctions. Local-feature TSTR must stay unchanged; the control tests the structure metrics' plumbing.
- **Minimal learned coupling.**
  - What it includes: groups, a learned GROUP choice, and chronological pool updates. It needs relabelling and a
    refit.
  - What it leaves out: AGENT, deliveries, taint, and initially a GNN.
  - Test the §7.3 structure metrics and structural-feature tabular transfer first.
- **The go decision comes after P3a,** and is the user's.

## Sealed test

It is opened once, with authorization, after deciding which stages belong in the frozen evaluation. Negative results
are confirmed too, and nothing is tuned afterwards.

## Compute

| item | estimate |
|---|---|
| complete P3a | about 1-3 h compute, plus 0.5-1 day of build |
| generation (local GTX 1650S; measured about 5 min per train-size world) | about 40 min |
| CPU readouts | about 1 h |
| encoder | up to 1 h |

Every item is under the 4 h single-run limit.

## Decisions for the user

- **D1.** Is development on val, with one authorized sealed-test evaluation at the end, OK?
- **D2.** Is the P3a claim wording above accepted as the first headline, with P3b as a separate go decision?
- **D3.** Mode coverage: is it scheduled inside P3a, as step 6 (Sol's recommendation)?
- **D4.** Is the sweep repair after the first readout, with its trigger and budget pre-registered now?
