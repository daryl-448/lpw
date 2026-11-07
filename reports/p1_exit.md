# P1 exit: bounded freeze for P3 feasibility (SPEC amendment A2, 2026-11-07)

P1 stops here **without meeting its exit criteria**. Neither the original criteria nor the A1 ones are met. This
report freezes a generator for **2018 P3 feasibility**. It does not declare P1 passed.

- **Decision.** Made with the user and a Sol strategy round (scratch notes `sol_consult_stop/consult.md`), after the
  scorer / B1 audit (`reports/p1_audit.md`) and the step-1 diagnostics.
- **Evidence.** All of it is 2018 val, development data. The sealed test is untouched.

## Why stop

- **What P3 needs from P1.** The research claim is P3's: a learned-policy world, judged by TSTR, the seam test, the
  shortcut audit and profile-gain parity (SPEC §0-2, §7.4). P1 is the instrument for it.
- **Diminishing returns.** The P1 screens returned changes within replica noise, or exposed measurement issues:
  - anchor, pace, drift v1 / v2, encoder;
  - base qualification;
  - the shallow correction;
  - the prior decomposition.
- **The B1 diagnosis ended on two accounts.** The remaining tf_alibi - B1 gap comes from two hub accounts of a single
  val group (`runs/p1_diag/step1/summary.md`).
- **Beating a GRU on likelihood is not the research question** (amendment A1 already moved generation fidelity ahead
  of it).

## Frozen recipe (`runs/p1_freeze/2018/manifest.json`, with sha256 of checkpoints and scorer code)

- **Primary: kct.** This is `lpw_plan_prog_shallow_kct`, the shallow-row correction on base `lpw_plan_prog_rp`.
  - Plans come from the default `PlanPrior` (no exposure inputs), fitted on train, drawn with rng 101.
  - Generation: seed 7, cap 10,000 events.
  - Its shallow readout outcome under the amended scorer is **STOP PROMOTION** (mid band 0.688 vs B 0.750, within
    noise; `runs/p1_shallow/2018/readout_v2.txt`). It is the exploratory primary, not a qualified model.
- **Sensitivity control: B,** the same base without the correction.
- **Scope.**
  - 2018 only.
  - 2021 is a documented failed stress case: scam M-pass is about 0 for every qualification model.
  - P2 conditioning is deferred. Mode coverage will be evaluated with an evaluation-only encoder.

## Criterion ledger (each criterion as written; 2018 val unless stated)

| criterion | status | evidence |
|---|---|---|
| beats B0 on full NLL | **met** | base 13.128 vs B0 16.230 (B0 not retrained under the current recipe; margin > 3 nats) |
| beats B2 on shared NLL | **met** | base 11.503 vs B2 12.684 (B2 not retrained; margin > 1 nat) |
| B1: original "beats B1 on full NLL" | **unmet** | base - b1 = +0.226 / +0.185 (two B1 seeds) |
| B1 non-inferiority (A1): upper95 <= +0.03 | **unmet** | upper95 +0.349 / +0.309 (`runs/p1_audit/2018/b1gap.txt`); no gate under A2 |
| B1 generation superiority (A1) | **not measured under a matched protocol** | B1 has no plan pathway; the encoder screen (unplanned, 3 replicas) had B1 normal 0.188 vs transformer 0.500; no gate under A2 |
| >= 70% M1-M16 pass per role (prior mode) | **met on the mean, fragile** | kct normal 0.725 ± 0.116, scam 0.800 ± 0.025 (5 replicas, amended scorer); one replica at 0.50. Diagnostic target under A2 |
| role-AUC within ±0.05 of real (prior mode) | **unmet, narrowly** | kct with profile: synthetic 0.844-0.856 vs real 0.908, delta -0.056 mean, 0/5 replicas within ±0.05 (B: -0.058). Behaviour-only detector: -0.071 |
| identical evaluation prefixes | **met** | b1gap asserts identical agent / depth / own arrays across 9 models |
| each slice (2021) | **unmet** | kct not built on 2021; every 2021 qualification model has scam M-pass of 0.00-0.06 |

## Readiness diagnostics (`scripts/p1_freeze/audit.py` -> `runs/p1_freeze/2018/audit.txt`; saved streams, 5 replicas each)

**Rollout safety (kct, prior mode):**
- No capped streams and no non-finite values.
- Longest stream 2,174 events; longest equal-timestamp burst 103 events.
- Plan violations: 5 streams with N > 3U (finite plan), 3 of them non-empty streams with a zero plan. B in prior mode
  has 6 and 5. These are disclosed, not runaways.

**Role parity.** Synthetic roles are **less separable than real** in every arm and mode, by 0.04-0.08 AUC. Every
group CI95 of the delta is below zero except one B oracle replica's ([-0.060, +0.005]).

| prior mode | with profile | behaviour only |
|---|---|---|
| kct | -0.056 | -0.071 |
| B | -0.058 | -0.077 |

**Single-feature shortcut gaps** (AUC of role on one feature, synthetic minus real; the P1 version of the §7.4
audit):
- Flagged in **every** arm and mode (|delta| > 0.1):
  - `end_balance`, +0.17 to +0.18;
  - `monthly_net`, +0.13 to +0.15.
- Every other feature stays below the 0.1 flag threshold (largest: out_frac +0.054, B oracle).

**Mechanism: the generator never empties an account.**

| share of accounts ending at about zero balance | normal | scam |
|---|---|---|
| real | 9% | **33%** |
| synthetic, B and kct, prior mode | 0.3% | 1% |

Real scams sweep their balance out, so a low end balance marks a scam (real AUC 0.40). Synthetic scams keep a
residual, which inverts the direction. This is a role-differentiating cash-out behaviour, and AML-relevant.

## Known defects carried into P3

1. **Balance sweeping is missing** (end-balance / monthly-net inversion). This is the first candidate for a bounded
   P1 repair under A2, e.g. a "sweep" atom in the amount head at amount = balance - fee. It is not started. It is to
   be decided against P3's own shortcut audit.
2. **Weaker role separability** than real: -0.06 AUC in prior mode.
3. **Mid band fidelity** below the matched-real reference (0.69-0.75 vs 0.88), oracle mode.
4. **Prior-mode normal M-pass is noisy** (sd 0.12), and the prior's plan marginal costs scam fidelity (-0.14;
   `reports/p1_prior.md`).
5. **Likelihood:** +0.19 to +0.23 behind B1. Any predictive-superiority claim is excluded.
6. **Fully synthetic populations are untested.** Prior mode samples plans but keeps real val profiles, cohorts and
   activation times. Sampling them is the first P3 checkpoint.
7. **2021:** scam generation collapses. No cross-era claim.

## Rules from here (A2)

- **Resuming P1.** P1 work resumes only for an identified P3-blocking defect, under one explicitly bounded
  experiment with its budget set beforehand. Defect 1 is the known candidate.
- **What does not reopen P1:** another architecture, the B1 margin, or noisy movement around 70%.
- **P3 control.** P3 should include an uncoupled control (same policy and population recipe, no coupling), so that
  generator defects and world-engine effects can be told apart.
