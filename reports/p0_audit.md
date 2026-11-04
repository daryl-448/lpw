# Phase 0 audit: pilot slices

- **Date:** 2026-11-04
- **Slices:** two windows, both 12 months UTC, built with identical preprocessing.
  - **2021 slice:** W = [2021-05-01, 2022-05-01). Output in `data/pilot`.
  - **2018 slice:** W = [2018-01-01, 2019-01-01). Output in `data/pilot_2018`. Added because the 2021 slice cannot support the confound-matched utility test (see Confound below).
- **Reproduce:**
  ```
  uv run python -m lpw.data.download --months 2021-05:2022-04
  uv run python -m lpw.data.download --months 2018-01:2018-12
  uv run python -m lpw.data.slice                                                    # 2021 → data/pilot
  uv run python -m lpw.data.slice --window 2018-01:2019-01 --out data/pilot_2018
  uv run python -m lpw.data.audit                                                    # → runs/p0_audit.json
  uv run python -m lpw.data.audit --pilot data/pilot_2018 --out runs/p0_audit_2018.json
  ```
  `runs/` is git-ignored. Each build is deterministic: two independent builds give identical groups and splits (unit-tested).

## Downloaded data

There are 30 files in `data/raw/` (7.3 GB). Each one was checked against the sha256 that HuggingFace publishes for it (`lpw.data.download`):
- 24 monthly edge files: 2021-05 to 2022-04 (101.5M edges) and 2018-01 to 2018-12 (29.3M edges, 1.5 GB);
- labels, address mapping, month metadata and `month_targets`;
- `addr_labels_balanced.csv.zst`;
- `lstm_dataset/daily_filtered.parquet`.

## Slices

| | 2021 normal | 2021 scam | 2018 normal | 2018 scam |
|---|---|---|---|---|
| Agents (non-hub EOA seeds with ≥1 E1 event in W) | 2,867 | 4,337 | 22,610 | 2,326 |
| Pre-existing / born in W | 2,819 / 48 | 1,476 / 2,861 | 6,515 / 16,095 | 219 / 2,107 |
| Events | 123,085 | 356,083 | 420,026 | 58,925 |
| Median events per agent | 2 | 23 | 3 | 8 |
| p90 / p99 events per agent | 24 / 723 | 205 / 855 | 30 / 256 | 51 / 301 |
| Agents with > 255 events | 54 | 324 | 227 | 33 |
| Median out-fraction | 0.69 | 0.62 | 0.50 | 0.33 |
| Median repeat-counterparty rate | 0.00 | 0.43 | 0.09 | 0.08 |
| Median distinct counterparties | 2 | 11 | 2 | 7 |
| Median share of events with counterparties shared by ≥5 agents | 0.33 | 0.60 | 0.50 | 0.20 |

| | 2021 slice | 2018 slice |
|---|---|---|
| Events (OUT / `IN_exo` / `IN_agent`) | 479,168 (315,897 / 160,632 / 2,639) | 478,951 (219,558 / 257,711 / 1,682) |
| Agent↔agent share of events | 1.1% | 0.7% |
| Self-transfers (kept, flagged by `cp == node_id`) | 936 | 368 |
| Hubs (>2,000 events in some month) → venues | 75 (74 normal, 1 scam), 64.2M events | 117 (114 normal, 3 scam), 23.6M events |
| Venues (hubs + counterparties shared by ≥50 agents) | 147 (75 + 72); 73 sender-capable; 2 labelled scam | 282 (117 + 165); 231 sender-capable; 3 labelled scam |
| `month_targets` upper bound (difference = hubs) | 2,941 normal, 4,338 scam | 22,724 normal, 2,329 scam |
| Scam prevalence among agents | 60% | 9.3% |

The two eras look different. In 2021, scam agents are busy, repeat their counterparties and lean on shared counterparties. In 2018 they are quieter, rarely repeat, and mostly *receive* (median out-fraction 0.33). A role learned in one era is therefore not assumed to transfer to the other.

## Checks

- **Daily records.** `daily_filtered.parquet` has 42.9M rows for 85,147 nodes, 2015-08-07 to 2025-08-17. 524,889 rows are exact duplicates of just 5,401 distinct rows, so they are deduplicated. In-W native ETH received per agent matches the E1 edges exactly in both slices (value ratio median 1.000, p10–p99 within 1e-8). The count ratio has median 1.0; in 2021 its p10 is 0.80, because sub-dust transactions are dropped from the edges. Every pre-existing agent has a pre-W profile in both slices.
- **`activity_end_ts`.** In both slices it is never earlier than the agent's last edge, and the first edge is never before `activity_start_ts`. Agents whose activity ends inside W: 2,303 in 2021 (median lag from last edge 0 days, p99 134) and 11,567 in 2018 (median 0 days, p99 241). So "activity ended inside W" is a reliable end-of-stream signal.
- **Observed balance.** It is unreliable in both slices, though less so in 2018.
  - The pre-W balance proxy is negative for 43.3% (2021) and 16.1% (2018) of pre-existing agents.
  - OUT events leave the running balance negative for 54.9% / 89.2% (2021, normal / scam) and 26.4% / 32.5% (2018).
  - In 2021, adding the daily internal-transaction flows only lowers this to 50.9% and 85.0%.
  - **Decision (unchanged):** the balance is an input feature only, and no conservation rule is enforced (SPEC §5.2). Even 26% is far above any usable conservation tolerance.
- **Groups.**
  - Connected components of the 2021 agent–counterparty graph (venues removed) put 47% of agents in one component (78% scam). That stays true for every venue threshold above 2:

    | `S_pub` | 50 | 20 | 10 | 5 | 3 | 2 |
    |---|---|---|---|---|---|---|
    | Largest component (% of agents) | 47.4% | 42.0% | 35.3% | 27.7% | 18.4% | 0.3% |

    At `S_pub = 2` every shared counterparty becomes a venue.
  - **Decision:** groups are Louvain communities.
    - 2021: 3,650 groups (3,328 singletons); the largest holds 9.3% of agents.
    - 2018: 12,507 groups (11,466 singletons); the largest holds 2.7%.
  - Split counts per role (normal / scam):

    | | train | val | test |
    |---|---|---|---|
    | 2021 | 2,007 / 3,035 | 430 / 651 | 430 / 651 |
    | 2018 | 15,826 / 1,628 | 3,392 / 349 | 3,392 / 349 |
- **Leakage across the split.**
  - 2021: 2,261 of 5,721 shared non-venue counterparties span splits, and 431 test agents share one with a train agent. The clean test subset has 316 normal and 334 scam agents.
  - 2018: 1,059 of 9,412 span splits, and 932 test agents share one with a train agent. The clean test subset has 2,539 normal and 270 scam agents.
- **Confound.**

  | | 2021 | 2018 |
  |---|---|---|
  | Age-only AUC (all / pre-existing cohort) | 0.960 / 0.915 | 0.597 / 0.500 |
  | Confounder-only AUC (train / val / test) | 0.961 / 0.959 / 0.972 | 0.611 / 0.613 / 0.610 |
  | Common support, test (scam / normal) | 637 / 427 | 348 / 3,375 |
  | 1:1 caliper-matched pairs (train / val / test) | 347 / 93 / **62** | 1,607 / 348 / **348** |
  | Matched utility test (SPEC §7.5 needs ≥ 200 per role) | **not testable** | **testable** |

  In 2018, account age carries almost no label signal, and 348 of the 349 test scam agents find a matched normal partner.

### Corrections

- **Caliper matcher.** A first version only searched the 20 nearest candidates. That undercounts when many scores tie (it reported 99 test pairs where the exact count was 107). It now uses the exact sorted two-pointer method (`caliper_pairs`, unit-tested).
- **Non-reproducible split.** The first 2021 numbers (Louvain 3,649 groups; clean test subset 344 / 327; matched pairs 312 / 85 / 107) came from one build that could not be reproduced. polars `unique()` and `group_by()` do not guarantee row order across runs. That order fed the Louvain graph and the seeded random group order in `split_groups`, so rebuilding moved 2,660 agents between splits. Both are now sorted before the seeded steps. Two builds are identical, a unit test shuffles the input to check this, and all numbers above come from the fixed build. The conclusion for 2021 stands, since every build gives far fewer than 200 test pairs.

## Throughput (GTX 1650 Super)

Measured on a stand-in transformer of the planned size: 826,112 parameters, 4 layers, d_model 128, fp16 AMP.

| Measurement | Value |
|---|---|
| Training throughput (B64 × L256) | 68.5–69.2k tokens/s, so about 7 s per epoch over either slice (both have about 479k events) |
| Batch-1 L256 inference pass | 1.65–1.67 ms, so the sequential world engine takes about 15–30 min per slice |
| Peak GPU memory | 471 MB |

Nothing in P1–P3 is near the 4 h limit, even with both slices.
