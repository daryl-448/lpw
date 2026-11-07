# P3a summary: a learned agent policy transfers to real detection (2018, development evaluation)

Status as of 2026-11-07:
- **Development results** on real val. The sealed test is untouched.
- **Scope decision (user, option B).** P3b, the coupled world, belongs in the confirmatory package. The sealed test
  therefore stays closed until P3b is developed. It will then be opened once, for P3a and P3b together.
- **Not claimed:** P3 is not complete, and P1 is not qualified (it stopped under SPEC A2).

Sources: `reports/p3a_prereg.md`, `reports/p3a_readout.md`, `reports/p3a_sweep.md`, `reports/p1_exit.md`.

## Question and design

- **Question.** Does a node-level role detector trained only on a fully synthetic LPW population keep most of a
  real-trained detector's skill on real held-out groups (T4, local node features)?
- **Population.**
  - Roles, cohorts, activation times, profiles, balances and plans are all sampled from frozen train-fitted models.
  - The policy is frozen kct, with B as control.
  - Agents are uncoupled: no shared counterparties beyond venues, and no deliveries between agents.
- **Comparison.** O(world) vs the real all-incident reference, on one shared node-feature extractor.
- **Protocol.** Pre-registered before any utility number, with the outcome rule fixed in advance:
  - SUPPORTED: the paired contrast D's one-sided 95% lower bound is ≥ 0, i.e. the excess-AUC ratio is ≥ 0.8 with
    confidence;
  - CONSISTENT: D ≥ 0 on the point estimate only;
  - NOT SUPPORTED: D < 0.

## Results

| | kct | B |
|---|---|---|
| TSTR AUC, full / XGB (worlds 0-2) | 0.909 ± 0.004 | 0.908 ± 0.006 |
| TRTR AUC | 0.932 | 0.932 |
| excess-AUC ratio; D [CI95] | 0.948; +0.064 [+0.041, +0.082] | 0.945; +0.063 [+0.042, +0.079] |
| outcome | **SUPPORTED** | **SUPPORTED** |
| replication: kct control on worlds 10-12 | 0.909, ratio 0.948, SUPPORTED | n/a |

- **Robustness.** Every detector × feature set keeps a ratio of 0.89-0.97. The result is unchanged with owned-only
  real observation and with capped streams removed.
- **Confounding.**
  - Inside 348 age / era / cohort-matched val pairs (max |SMD| 0.054), TSTR is 0.876 vs TRTR 0.906.
  - Behavioural gain over the confounder-only detector: +0.240 synthetic-trained vs +0.241 real-trained. These are
    estimates; no equivalence test was pre-registered.

## Criticism tests: the synthetic world is useful but not realistic

| test | result |
|---|---|
| Seam C2ST, local features | **fails**: 0.87 (normal) / 0.89-0.90 (scam). Scams are no easier to spot than normals (gap +0.02-0.03). Balance features alone give 0.81-0.85; without them, 0.74-0.78. |
| Difficulty parity | **fails**: synthetic → synthetic AP is 0.13-0.15 below real → real. Synthetic worlds are harder. |
| Shortcut audit | **fails**: 7 flags (all 5 balance sentinels, median OUT amount, median fee). Each is a signal the synthetic data **lacks**. None is template-like. |
| Profile-gain parity | passes, **uninformative**: the profile adds about 0 AUC even on real data |
| DCR (memorization) | passes, ratios 3.4-7.9 |
| Mode coverage, tabular (standardized full vectors, k = 5) | normal P / R 0.87 / 0.87; **scam recall 0.57**. This is point coverage in that representation, not "57% of scam typologies". |
| Mode coverage, encoder space (supplementary, addendum B; not a gate) | normal P / R 0.98 / 0.98 (both arms); scam P / R 0.96 / 0.98 (kct), 0.95 / 0.99 (B). n_r 3382 / 307. |

**Encoder coverage (addendum B, `runs/p3a/2018/encoder/`).**
- **The fit.** One fixed recipe: 20 epochs, loss 0.68 → 0.165, 139 s. The train-only checks passed (finite values,
  minimum coordinate SD 0.13, mean pairwise cosine 0.003), so the representation did not collapse.
- **The two analyses disagree for scams.** In a role-blind, balance-free event-sequence view, the synthetic scams
  sit on the real-val scam manifold and cover it almost entirely: recall 0.98 per world, against 0.57 tabular.
- **The likely reason, which is not tested.**
  - The tabular gap comes from aggregate state features (balances, terminal cash-out, amount and fee medians).
  - The encoder leaves out the balance by design.
  - This matches the sweep finding: tabular recall rose to 0.94 once sweeps existed.
- **Caveats.**
  - Near-saturated P/R may also mean the 32-d CoLES space is coarse in the directions where the roles differ.
  - No real-vs-real reference was pre-registered, so it cannot be calibrated here.
  - Tabular stays the primary analysis, and C2 is **not** passed.

**Interpretation.**
- The synthetic agents carry role signal in the right direction, so detectors learn the right boundary.
- But the role signal is weaker (difficulty), less diverse (scam recall), and stylistically off in balance, amount
  and fee behaviour (seam).
- Transfer holds despite a clearly detectable seam, so the two must be reported together.

## Defect ledger (what limits the claims)

| # | defect | status |
|---|---|---|
| 1 | No balance sweeping: synthetic accounts almost never end at zero | **Open.** The bounded repair (a learned sweep gate) was **rejected** on its pre-registered scam mechanism gate. It restored the action, but synthetic scams keep acting afterwards (a later IN in 0.64 of sweepers vs 0.32 real). |
| 2 | Weaker role separability than real | open (difficulty parity; P1 role AUC -0.06) |
| 3-5 | P1 fidelity and likelihood limits (mid band, noisy normal M-pass, B1 gap) | open; no predictive-superiority claim |
| 6 / 8 | Sampled profile joint incoherent (e.g. 50% have days_active > age; profile C2ST 0.96) | open; small effect on P3a utility, matters for the released node table (C1, T5) |
| 7 | 2021: scam generation collapses | out of scope (A2); no cross-era claim |
| 9 | Overdrawing: synthetic OUTs leave the observed balance negative in 52-58% of cases vs 33% real | open; untested |

## What this licenses

**Licensed (2018, development; to be confirmed on the sealed test):** a fully synthetic, uncoupled LPW population
trains node-level scam detectors that keep about 95% of real-trained excess AUC on real held-out groups, including
inside confound-matched pairs.

**Not licensed:**
- that the synthetic world is indistinguishable from real;
- realistic difficulty;
- coverage of real scam behaviour;
- anything about coupled-world structure (P3b);
- anything about another era.

## Next

1. ~~Encoder mode-coverage tranche~~ done (above).
2. **P3b**, inside the confirmatory package:
   - first, the structural-identity control;
   - then one learned GROUP / pool candidate;
   - judged on §7.3 structure and on structural signal beyond local features.
3. **One authorized sealed-test evaluation** of the frozen P3a and P3b protocols.
