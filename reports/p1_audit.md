# P1 scorer and qualification audit (2026-11-07)

This is step 1 of the plan from the Sol consult after the prior decomposition
(scratch notes, not in the repo; summary in `reports/p1_prior.md`). It uses train/val only, and val is
development data. The sealed test was not touched.

Code:
- `src/lpw/p1/fidelity.py` (`compare()` amendment) and `tests/test_fidelity.py`;
- `scripts/p1_audit/b1gap.py`;
- a `--tag` option on `scripts/p1_shallow/readout.py` and `scripts/p1_prior/decompose.py`.

## 1. `compare()` amendment: per-group rngs

**The defect.** One rng ran through the groups in dict order, so a group's bootstrap floors depended on which groups
preceded it. Example: the same scam generation scored 0.6875 alone and 0.75 after `normal`.

**The fix.** Each group now gets `default_rng(seed)` for its real half-splits, plus a separate
`default_rng([seed, 1])` for the unpaired synthetic draw.
- Paired single-group scores are unchanged.
- The first group of paired calls is unchanged; every later group's score changes.
- Unpaired scores change: only `run.py`'s unconditional generation uses them.

**Tests.** Two new tests fail without the fix:
- a group's score equals its score when scored alone and when the groups are reversed;
- unpaired floors equal paired floors.

**Rescoring of the active results** (`_v2` files; the earlier files are kept):
- **prior decomposition (`runs/p1_prior/2018/decompose_v2.txt`):** identical. It already scored each group alone.
- **shallow readout (`runs/p1_shallow/2018/readout_v2.txt`):** the outcome changes from PROMOTE to **STOP
  PROMOTION**.

  | | B | kct | matched real |
  |---|---|---|---|
  | oracle normal mid 5-99 | 0.750 | 0.688 | 0.875 |
  | oracle normal light 1-4 | 0.400 | 0.850 | 0.975 |

  - The mid rule fails, by -0.06, within the replica sd of about 0.07. The other four checks still pass.
  - Light gap closed: 78%. Normal overall is unchanged, since it is the first group.
  - Prior scam: B 0.800, kct 0.800.

Older reports (drift, pace, anchor, encoder, qualification) are not rescored. Their decisions keyed on `normal` /
the first group, or on likelihood, but their later-group numbers carry the old order dependence.

## 2. B1 likelihood gap (SPEC A1: non-inferior iff upper95 of model - b1 <= +0.03 per owned event)

**Setup** (`runs/p1_audit/2018/b1gap.txt`):
- Exact per-prefix val terms on the encoder screen's prefix set: 63,360 prefixes, 59,406 owned events.
- Paired whole-group bootstrap, 1000 draws, seed 0, 1,873 groups.
- Plan models are scored with the plan unknown: the likelihood kct shares with its base.

**Matched conventions.** Every model uses the same pilot, the same lr 1e-3, the same recalibration (4000 steps,
374,514 rows) and the same exact prefix and owned-row sets; no agent weighting. They differ in seed, encoder,
plan / progress and parameter count. B1 has 0.58M parameters, the transformers 1.15-1.18M. The protocol reproduces the
encoder screen's tf_alibi - b1 exactly (+0.0167 / +0.0232).

**Results** (full NLL per owned event; b1 12.902 / 12.943 for its two seeds):

| model | full | - b1 s04 (upper95) | - b1 s05 (upper95) |
|---|---|---|---|
| **current base** (lpw_plan_prog_rp, s 20261104) | 13.128 | **+0.226 (+0.349)** | +0.185 (+0.309) |
| lpw, same seed, no plan / progress | 13.084 | +0.182 (+0.302) | +0.141 (+0.265) |
| lpw seed 1 | 13.165 | +0.263 (+0.409) | +0.222 (+0.372) |
| lpw seed 2 | 13.057 | +0.155 (+0.267) | +0.115 (+0.227) |
| tf_ref (independent init) | 13.077 | +0.175 (+0.291) | +0.135 (+0.253) |
| tf_alibi s 20261104 | 12.919 | +0.017 (+0.066) | -0.024 (+0.039) |
| tf_alibi s 20261105 | 12.966 | +0.064 (+0.103) | +0.023 (+0.076) |

**Readings:**
- **The current base fails B1 non-inferiority by a wide margin:** +0.19 to +0.23, upper95 +0.31 to +0.35.
  - P1 cannot pass on this base. No generation-side work (shallow offsets, prior, rollout training) changes this
    number.
- **Where the gap comes from:**
  - the absolute-position transformer itself: lpw - b1 is +0.11 to +0.26 over 3 seeds, and the seed spread alone is
    0.11;
  - the plan / progress extension: +0.044 on the same seed;
  - about 60% of the gap is in cp / amount, and about 40-50% comes from prefixes deeper than 255 events.
  - Both encoders see the same input: PROFILE plus at most 255 events (`exact_specs()`). The GRU keeps no hidden
    state between windows, and both get the full-history partner / ledger summaries. So the deep-prefix share is not
    a context-length difference between them. Whether a longer context would help is an untested hypothesis.
- **tf_alibi is the only transformer near B1:** -0.02 to +0.06, upper95 +0.04 to +0.10. It is still not
  non-inferior on any pairing.
  - It also failed base qualification on the 2021 generation gates (`runs/p1_qualify/qualify.txt`).

## Consequences for the plan

- **Do not provisionally freeze kct-on-the-current-base as a P1 candidate.** Its likelihood rules P1 out. It can
  still serve as a generation reference.
- **The binding P1 question is the backbone:**
  - whether a transformer can reach B1 non-inferiority at all (ALiBi / recency, longer context as a hypothesis, the depth > 255
    regime);
  - or whether SPEC should be amended again, a user decision.
  - Shallow offsets would then be refitted on any new base, at about 1 h.
