# Phase 0 audit: pilot slice

- **Date:** 2026-11-04
- **Window:** W = [2021-05-01, 2022-05-01) UTC
- **Reproduce:** `uv run python -m lpw.data.download`, then `uv run python -m lpw.data.slice`, then `uv run python -m lpw.data.audit`. Raw output: `runs/p0_audit.json` (git-ignored).

## Downloaded data

There are 18 files in `data/raw/` (5.8 GB). Each one was checked against the sha256 that HuggingFace publishes for it (`lpw.data.download`):
- 12 monthly edge files (2021-05 to 2022-04);
- labels, address mapping, month metadata and `month_targets`;
- `addr_labels_balanced.csv.zst`;
- `lstm_dataset/daily_filtered.parquet`.

## Slice

| | normal | scam |
|---|---|---|
| Agents (non-hub EOA seeds with ≥1 E1 event in W) | 2,867 | 4,337 |
| Pre-existing / born in W | 2,819 / 48 | 1,476 / 2,861 |
| Events | 123,085 | 356,083 |
| Median events per agent | 2 | 23 |
| p90 / p99 events per agent | 24 / 723 | 205 / 855 |
| Agents with > 255 events | 54 | 324 |
| Median out-fraction | 0.69 | 0.62 |
| Median repeat-counterparty rate | 0.00 | 0.43 |
| Median distinct counterparties | 2 | 11 |
| Median share of events with counterparties shared by ≥5 agents | 0.33 | 0.60 |

- **Event kinds:** 315,897 OUT, 160,632 `IN_exo`, 2,639 `IN_agent`.
- **Agent↔agent:** 1.1% of events.
- **Self-transfers:** 936 events. They stay in the data, flagged by `cp == node_id`.
- **Hubs:** 75 EOA seeds with more than 2,000 events in some month (74 normal, 1 scam). They hold 64,187,316 events and all become venues.
- **Venues:** 147 in total (75 hubs and 72 counterparties shared by ≥50 agents). 73 are sender-capable, and 2 are labelled scam.
- **Agents vs the `month_targets` upper bound** (2,941 normal, 4,338 scam): the difference is the hubs.

## Checks

- **Daily records.** `daily_filtered.parquet` has 42.9M rows for 85,147 nodes, 2015-08-07 to 2025-08-17. 524,889 rows are exact duplicates of just 5,401 distinct rows, so they are deduplicated. In-W native ETH received per agent matches the E1 edges exactly: value ratio median 1.000 (p10–p99 within 1e-8). The count ratio has median 1.0 and p10 0.80, because sub-dust transactions are dropped from the edges. Every pre-existing agent has a pre-W profile.
- **`activity_end_ts`.** It is never earlier than the agent's last edge, and the first edge is never before `activity_start_ts`. For the 2,303 agents whose activity ends inside W, the lag from last edge to activity end has median 0 days and p99 134 days. So "activity ended inside W" is a reliable end-of-stream signal.
- **Observed balance.** It is unreliable.
  - The pre-W balance proxy is negative for 43.3% of pre-existing agents.
  - OUT events leave the running balance negative 54.9% of the time for normal agents and 89.2% for scam agents.
  - Adding the daily internal-transaction flows only lowers this to 50.9% and 85.0%.
  - **Decision:** the balance is an input feature only, and no conservation rule is enforced (SPEC §5.2).
- **Groups.**
  - Connected components of the agent–counterparty graph (venues removed) put 47% of agents in one component (78% scam). That stays true for every venue threshold above 2:

    | `S_pub` | 50 | 20 | 10 | 5 | 3 | 2 |
    |---|---|---|---|---|---|---|
    | Largest component (% of agents) | 47.4% | 42.0% | 35.3% | 27.7% | 18.4% | 0.3% |

    At `S_pub = 2` every shared counterparty becomes a venue.
  - **Decision:** groups are Louvain communities. There are 3,649 of them (3,327 singletons), and the largest holds 9.3% of agents.
  - Split counts per role: train 2,007 normal / 3,035 scam; val 430 / 651; test 430 / 651.
- **Leakage across the split.** 2,206 of 5,721 shared non-venue counterparties span splits. 410 test agents share one with a train agent. The clean test subset has 344 normal and 327 scam agents.
- **Confound.**
  - Age-only AUC is 0.960, and 0.915 within the pre-existing cohort.
  - A confounder-only logistic model (age, cohort, proxies, missingness) scores AUC 0.966 on train, 0.956 on val and 0.947 on test.
  - 1:1 caliper matching on common support gives 280 train, 78 val and **99 test** pairs. That is below the 200 that SPEC §7.5 requires, so the confound-matched utility claim is **not testable in this window**.

## Throughput (GTX 1650 Super)

Measured on a stand-in transformer of the planned size: 826,112 parameters, 4 layers, d_model 128, fp16 AMP.

| Measurement | Value |
|---|---|
| Training throughput (B64 × L256) | 68,527 tokens/s, so about 7 s per epoch over the 479k-event slice |
| Batch-1 L256 inference pass | 1.665 ms, so the sequential world engine takes about 15–30 min for the pilot |
| Peak GPU memory | 471 MB |

Nothing in P1–P3 is near the 4 h limit.
