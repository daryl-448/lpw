# SPEC: Learned-Policy World (LPW)

From next-event prediction to an agent-based synthetic AML simulator.

- **Status:** Phase 0 done 2026-11-04 for two pilot slices, 2021 and 2018 (§5.2, [reports/p0_audit.md](reports/p0_audit.md)). Phases 1–3 not started.
- **Inputs:** the paper digests in [papers/](papers/README.md) and the HuggingFace dataset `fesevu/ethereum_fraud_dataset_by_activity`.
- **Hardware for the pilot:** GTX 1650 Super (4 GB VRAM, Turing sm_75, no tensor cores), 15 GB RAM, 12 CPU threads.

## 0. TL;DR

A next-event model `p_θ(e_{k+1} | history_k, profile, role)` (Banksformer-style) is already an agent *policy*. It only generates one account in isolation. LPW turns it into an agent-based simulator by adding five things:

1. **Who.** Each event picks a counterparty: a past partner, a public venue, a group-shared address, another simulated agent, or a fresh address. All of these choices are learned.
2. **When, coupled.** A shared world clock. An agent re-draws its next action whenever its history changes, including when another agent sends it money.
3. **How much is possible.** A per-agent ledger that the amount head conditions on.
4. **Who exists.** A learned prior over agent roles, activation times, profiles and (later) latent behaviour types and groups.
5. **What would have been recorded.** An *observation operator* that applies the dataset's own recording rules to the synthetic world (dust floor, seed-incident edges only). Only then is the result compared with the matching real data.

Illicit behaviour is **not injected from templates or edited into the graph**. Illicit agents run the same learned policy, conditioned on a role learned from real labeled scam addresses. The output is checked against **real labels** (train on synthetic, test on real: TSTR), not only against stylized facts.

## 1. Goal and non-goals

**Goal.** Build a generator of synthetic AML-style datasets. Each dataset has:
- a node table (agents with observable profiles),
- a transaction table,
- node-level and transaction-level labels,
- a hidden-state log, as in FINESSE `L_hidden`.

The behaviour in it is learned from data, not hand-programmed. Start small (local pilot), then scale.

**Non-goals for v1:**
- ERC-20, internal or zero-value transactions. They are not in the edge data (§5.1).
- Smart contracts as actors. Contracts never send in this data, so they are only passive counterparties.
- Second-hop (layering) flows beyond agents' own transactions. The released E2 edges cannot identify them (§5.3).
- Gas/fee modelling. Fees are still read, for the ledger.
- Multi-currency and payment formats (AMLworld/TransXion).
- Formal causal inference. FINESSE disclaims it too.
- Adversarial hardening of illicit agents. It is future work (§10).

**Framing caveat.** The real labels are *scam/phishing addresses*: fraud, not adjudicated laundering. LPW's "illicit" means "controlled by, or carrying funds from, a scam-labelled actor". Moving scam proceeds is a laundering-adjacent behaviour, but the outputs must be called "illicit-linked", not "laundering", unless better labels are added.

## 2. What "fixing the criticisms" means

Two sets of criticisms are in scope: the ones TransXion raises about existing AML benchmarks (C1, C2), and the weaknesses of TransXion itself (T1–T5). Each row names the test that checks the fix (§7).

| ID | Criticism | Source | LPW fix | Test |
|---|---|---|---|---|
| C1 | Edge-only logs, no node semantics, so "out-of-character" behaviour cannot be modelled | TransXion §1 on AMLSim, AMLworld, SAML-D | Every agent has an observable profile that conditions its policy. A node table and a hidden-state log are emitted. "Out-of-character" has a direct meaning: low likelihood under the agent's own policy. | Profile-gain parity (§7.4) |
| C2 | Template injection (fan-in, fan-out, smurfing...) teaches detectors shortcuts | TransXion §1 | No templates. Illicit behaviour comes from the learned policy. Coordination comes from learned group structure (§6.6). | Shortcut audit, mode coverage, TSTR (§7.4) |
| T1 | TransXion's normal backbone is hand-coded. Its counterparty mixture (locality, decayed memory, attractiveness, exploration) has no published formula or weights. | TransXion §3.1 | The same four ideas become *learned* counterparty classes: REPEAT (memory), VENUE (attractiveness), GROUP/AGENT (locality), NEW (exploration). The gate and the choices are fitted by maximum likelihood. | P1 next-event metrics (§7.1) |
| T2 | Anomalies (LLM edits of real subgraphs) and backbone (ABM) come from different generators, so detectors may learn which generator made the data | TransXion §3.2–3.3 | Normal and illicit agents come from one model and one engine. | Seam test (§7.4) |
| T3 | Hardness is optimized against a monitor with only validity constraints, so hardness is not the same as realism | TransXion App. A | The target is *difficulty parity with real data* at matched prevalence. Hardening is not part of v1 (§10). | Difficulty parity (§7.4) |
| T4 | Validation uses stylized facts and detector scores only. Nothing is tested against real labeled outcomes. | TransXion §5 | Real labels exist here, so TSTR on held-out real addresses is the headline test. | §7.4 |
| T5 | Labels are transaction-level only. There are no typology IDs and no hidden state. | TransXion App. B.1 | LPW emits node roles, group IDs, latent behaviour types (Phase 2), taint fractions and a daily hidden-state log. | Output schema (§6.9) |

## 3. Core idea

```
   learned population prior  p(group) · per member: p(role) p(cohort, activation | role) p(z | ·) p(profile | z, ·)
                              │
                              ▼
 ┌──────────── agent i: policy p_θ(next owned event | own stream, partner table, profile, role, z) ────┐
 │  kind ─► time (normalized Δt × hour × weekday law) ─► counterparty class+choice ─► amount           │
 └──────────────┬──────────────────────────────────────────────────────────────────────────────────────┘
                │ OUT to agent j   → j's stream gets an IN_agent token → j re-draws its clock
                │ OUT to non-agent → venue / group-shared / new periphery node (passive in v1)
                │ IN_exo           → generated by i's own policy (sender not simulated)
                ▼
     world engine: exact event-ordered clock, ledgers, fund inventories, groups, taint ──► full world log
                                                                         │
                       observation operator O (dust floor, agent-incident edges)
                                                                         ▼
                 synthetic dataset, compared like-for-like with the real reference R_W (§6.8)
```

**Ownership rule (§6.7).** Every event has exactly one generator. No other engine component generates inflows to agents.
- An OUT event belongs to its sender's policy.
- An IN event from a non-simulated source (`IN_exo`) belongs to the receiver's policy.
- An IN event from a simulated agent (`IN_agent`) is the sender's OUT. The receiver only *observes* it.

## 4. What each paper contributes

| Paper | Reused | Adapted or rejected |
|---|---|---|
| Banksformer | Decoder-only transformer with a metadata token at position 0 and per-field conditional generation inside each event (teacher forcing). The **date mechanism**: per-component date heads combined as a product renormalized over a candidate window. Metric suite (W1, JSD, 3-gram JSD, joint type×date JSD). Ablation pattern (BF-ND, BF-NC). | Calendar dates become continuous time with log-spaced Δt bins plus hour-of-day and day-of-week (§6.3). Fixed-length sequences become right-censored streams. Censoring needs a single time law, so the heads are trained *jointly* through the normalized product. Banksformer's independent head training stays in baseline B2. |
| BankGAN | Keep the discrete skeleton (kind, time, counterparty) separate from attribute generation. **Clock (sin/cos) encoding** for cyclic inputs (best overall in its Table 1). Multimodal amount modelling (BankGAN had the best Amount W1 of the three models, which its authors credit to GMM mode-specific normalization). | The CTGAN attribute generator becomes a mixture-density head, which is stable on a 4 GB GPU. A CTGAN head is an optional ablation (A4). DP-SGD is not needed because the chain is public. A memorization check is kept instead (§7.4). |
| FINESSE | Agents with latent behaviour policies that drive observed streams. A **hidden-state log** emitted as ground truth. Venues as "merchants" with popularity and affinity. The next-event TPP benchmark framing (type error, time error). | Hand-specified policy families (its Table 2) become learned latent types (Phase 2). The bank-intervention stream is future work (§10). |
| TransXion | Profile-conditioned agents, the counterparty mixture *concept*, labels propagated along fund traces, and its evaluation suite: RQ1 stylized facts, Clauset tails, AP/F1 over AUC. | Every hand-coded mechanism becomes learned (T1). Post-hoc subgraph embedding is dropped, because there is a single generator (T2). GRPO hardening with cross-family transfer is future work. If it is done, it acts on the small policy's own actions, so no LLM is needed (§10). |

## 5. Data

### 5.1 Verified facts (checked 2026-11-04)

Sources: the HF API file listing, the dataset READMEs, the pipeline code at `github.com/fesevu/eth-fraud-dataset-pipeline` (`bq/GNN/GNN`, `dataset/final/LSTM/build_unified_dataset.py`), and local polars scans of the label files, `month_targets.parquet`, `lstm_dataset/monthly.parquet` and the monthly edge files for 2017-06 and 2021-10. Phase 0 re-checks every number.

**Size and coverage**
- 121 monthly edge files, 2015-08 to 2025-08 (2025-08 is partial and ends at ts 1755475139). 380,566,661 edges in total.
- About 21.3 GB in `gnn_dataset/edges_by_month/`. The whole repo is about 73 GB, including raw exports.
- **Edge schema:** `src_id, dst_id` (UInt64 hashes of addresses), `ts` (s, UTC), `value_wei`, `tx_fee_wei` (decimal strings), `block_number`, `contract_creation`, `tx_hash`.

**How edges were sampled (pipeline SQL)**
- **Seeds** are labeled addresses with non-null `activity_start_ts`.
- **E1** is *every* top-level native-ETH transaction of a seed inside its activity window with `value ≥ 1e14 wei` (0.0001 ETH). So seeds have complete native-ETH histories above the dust threshold.
- **E2** consists of same-UTC-day transactions *between* first-hop (L1) and second-hop (L2) neighbours. The selection runs as follows:
  - **S2:** keep the top 15 L1 neighbours per (seed, day). They are ranked by summed value over *all* the seed's transactions, including those below the dust floor, so the ranking happens before the dust filter.
  - **S3:** keep the top 5 L2 neighbours per (seed, day) *within each of 16 L1-hash-shard batches*. That allows up to 80 L2 per (seed, day).
  - **S6:** keep transactions in both directions (L1→L2 and L2→L1) at or above the dust floor, capped at 20 per (seed, day, from, to) *directed* pair.
- The data has no ERC-20 transfers, no internal transactions and no zero-value calls. **Contracts never appear as senders** (the out-fraction of contract seeds is 0.0 in 2021-10).

**Labels**
- `targets_global.parquet` has 114,739 addresses:

  | | EOAs | Contracts | Total |
  |---|---|---|---|
  | scam | 14,982 | 39,671 | 54,653 |
  | non-scam | 30,059 | 30,027 | 60,086 |

- Descriptions are mostly `phishing`/`Phishing` (42,853), then `Scamming` (2,176), `ponzi` (374) and `rugpull` (266). The labels are node-level only.
- 29,592 labeled addresses (24,776 of them scam) have a null `activity_start_ts`, so they have no edges.
- 66,002 seeds are active in at least one month:

  | | EOAs | Contracts |
  |---|---|---|
  | non-scam | 29,643 | 14,005 |
  | scam | 12,389 | 9,965 |

**Severe label/time confound**
- Non-scam EOAs were mostly *born* in 2017–2018. Only **130** non-scam EOAs were born between 2020-07 and 2022-12, against 5,835 scam EOAs.
- Activity overlaps better. In every month from 2021-05 to 2022-04, `month_targets` lists 689–1,494 active non-scam seeds and 989–2,601 active scam seeds (all address types, hubs included).
- Among non-hub EOAs active in 2021-10, account age alone separates the classes with **AUC 0.93**. Median age is 1,386 days for non-scam and 146 days for scam.

**2021-10 profile (9,678,589 edges)**
- **Edge mix:** 98.5% touch a seed. 0.84% are seed↔seed. 1.5% are E2.
- **Hubs:** 67 seeds with more than 2,000 events/month (47 non-scam EOAs, 19 non-scam contracts, 1 scam contract) hold **99.3% of seed-events**.
- **Non-hub EOAs:**

  | Median per agent | non-scam (700 agents, 18,017 events) | scam (2,068 agents, 40,046 events) |
  |---|---|---|
  | events | 2 | 7 |
  | out-fraction | 0.61 | 0.67 |
  | repeat-counterparty rate | 0.00 | 0.33 |
  | counterparties shared with ≥5 seeds | 0.40 | 0.68 |

- Agent↔agent events are **about 1.0%** of all events.
- **Per-agent hour-of-day entropy** (agents with ≥10 events) has a median of about 3.1–3.2 bits, against a 4.58-bit maximum. The aggregate UTC profile is nearly flat (±15%), so diurnal structure is per-agent, not global.

**Pilot-window cohort** (`month_targets`, EOAs active anywhere in `W` = 2021-05 to 2022-04, before hub exclusion)

| | Active in W | Born before W | Born inside W (`activity_start_ts` in W) |
|---|---|---|---|
| non-scam | 2,941 | 2,875 | 66 |
| scam | 4,338 | 1,476 | 2,862 |

**Pre-window profile sources**
- `lstm_dataset/monthly.parquet` has monthly features for 85,147 labeled node_ids, 2015-08 to 2025-08. It includes ERC-20 counts, internal-transaction values, peer counts and fees.
- But `monthly` is built **from ISO weeks**, and ISO weeks cross month boundaries. For example, April includes May 1–2. So monthly rows leak activity from after a month-start cutoff, and cumulative sums count boundary weeks twice.
- Profiles therefore come from the **daily** records instead (`lstm_dataset/daily_filtered.parquet`, or the raw `LSTM/parquet/daily_final_*`), aggregated strictly before the cutoff (§5.4).

### 5.2 Phase 0 measurements (2026-11-04)

Full numbers: [reports/p0_audit.md](reports/p0_audit.md). Raw JSON: `runs/p0_audit.json` (2021) and `runs/p0_audit_2018.json` (2018), produced by `python -m lpw.data.slice` then `python -m lpw.data.audit` (commands in the report). Builds are deterministic.

There are two pilot slices (§5.4): the **2021 slice**, W = [2021-05-01, 2022-05-01), and the **2018 slice**, W = [2018-01-01, 2019-01-01).

| Item | 2021 slice | 2018 slice |
|---|---|---|
| Agents (non-hub EOA seeds in W) | 2,867 normal, 4,337 scam (60% scam) | 22,610 normal, 2,326 scam (9.3% scam) |
| Cohorts, pre-existing / born in W | normal 2,819 / 48; scam 1,476 / 2,861 | normal 6,515 / 16,095; scam 219 / 2,107 |
| Hubs → venues | 75 EOA hubs (74 normal, 1 scam), 64.2M events. Plus 72 shared counterparties at `S_pub = 50`: 147 venues. | 117 hubs (114 normal, 3 scam), 23.6M events. Plus 165 shared: 282 venues. |
| Events | 479,168: 315,897 OUT, 160,632 `IN_exo`, 2,639 `IN_agent`. Agent↔agent 1.1%. | 478,951: 219,558 OUT, 257,711 `IN_exo`, 1,682 `IN_agent`. Agent↔agent 0.7%. |
| Events per agent (median) | normal 2, scam 23. 378 agents over 255 events. | normal 3, scam 8. 260 agents over 255 events. |
| Daily records | 524,889 exact-duplicate rows (5,401 distinct) are dropped. In-W native ETH received matches E1 edges exactly (value ratio 1.0). Every pre-existing agent has a pre-W profile. | Same. |
| `activity_end_ts` | Never earlier than the last edge. 2,303 agents end inside W, median lag 0 days. | Same check passes. 11,567 end inside W, median lag 0 days. |
| Observed balance | Pre-W proxy negative for 43% of pre-existing agents. OUT events leaving it negative: 55% normal, 89% scam (51% and 85% with internal flows). | Proxy negative for 16%. OUT events leaving it negative: 26% normal, 32% scam. |
| Groups | Connected components fail: one holds 47% of agents (78% scam) for every `S_pub` > 2. Louvain: largest community 9.3% of agents. 2,261 of 5,721 shared non-venue counterparties span splits. | Louvain: largest 2.7%. 1,059 of 9,412 span splits. |
| Clean test subset (no non-venue counterparty shared with train) | 316 normal, 334 scam | 2,539 normal, 270 scam |
| Confound | Age-only AUC 0.96 (0.92 within pre-existing). Confounder-only AUC 0.96–0.97. Matched pairs: 347 train, 93 val, **62 test**. | Age-only AUC 0.60 (0.50 within pre-existing). Confounder-only AUC 0.61. Matched pairs: 1,607 train, 348 val, **348 test**. |
| Throughput (GTX 1650 Super, 0.83M-param model, fp16) | Training 68.5k tokens/s at B64×L256, so about 7 s per epoch per slice. Batch-1 L256 pass 1.66 ms, so the sequential engine needs about 15–30 min per slice. Peak memory 0.47 GB. | (same model and GPU) |

The eras differ in behaviour, not just in mix. 2021 scam agents are busy, repeat counterparties (median repeat rate 0.43) and lean on shared counterparties. 2018 scam agents are quieter, rarely repeat (0.08) and mostly receive (median out-fraction 0.33).

**Decisions taken from these measurements** (they supersede the matching text above and below):
1. **Groups are Louvain communities** (fixed seed, label-blind, inputs sorted so the result does not depend on row order) of the agent–counterparty graph after removing venues, not connected components. `component_id` is kept for reference. Leakage through shared non-venue counterparties does not affect TSTR, because the generator never sees those addresses as identities and TSTR detectors train only on synthetic data. Detectors trained on real data are also reported on each slice's **clean test subset**.
2. **The observed balance is an input feature only**, in every phase and both slices. With 26–89% of OUT events leaving it negative, a conservation tolerance would be meaningless, so the Phase 3 redraw rule is dropped. Labels still use the separate non-negative fund inventory (§6.7).
3. **Two slices, with different jobs.** The 2021 slice cannot support the confound-matched utility test (62 matched test pairs, below the 200 that §7.5 requires); the 2018 slice can (348). Both slices run P1–P3 with the same code and preprocessing. Utility claims under §7.5 come from the **2018 slice**. The 2021 slice reports them with the "not testable" flag, and serves as the scam-dense, high-repeat stress case for fidelity and world structure. Models are trained per slice; a shared model with an era covariate is P5.

### 5.3 Consequences for the design

1. **Only seeds are full agents.** Their neighbours are partial and biased samples. Population tiers (§6.1):
   - agents are non-hub EOA seeds;
   - venues are hub seeds plus widely shared counterparties;
   - periphery is everything else.
2. **Coupling is mostly indirect.** Agents rarely transact directly (about 1%). They mostly connect through shared counterparties: venues and group-shared addresses, especially among scam agents (0.68 vs 0.40). Groups (§6.6) are therefore the main coupling mechanism. Direct agent↔agent edges are second.
3. **Only complete data is compared.** An agent's E1 edges are complete above the dust floor, so v1 models and compares only agent-incident E1 edges (§6.8). E2 is excluded from v1 for two reasons:
   - It is not a record of forwarding episodes. It has reverse-direction transactions and transactions before the triggering transfer, and an edge missing from E2 does not mean "did not forward".
   - The released edges carry no seed provenance, and the frontier selection (ranked before the dust filter, shard-batched) cannot be replayed for a reduced seed set.
4. **Activity-window design with two cohorts.** In 2021, normal agents are mostly old accounts. They enter the window with pre-window profiles (left-censoring), and most scam agents (2,861 of 4,337) are *born inside* the window. In 2018, most agents of both roles are born inside the window (16,095 of 22,610 normal, 2,107 of 2,326 scam). Both cohorts are modelled (§5.4, §6.7), and every stream is right-censored at the end of observation.
5. **The confound is reproduced, then controlled for.** The generator conditions on account age and cohort, so it reproduces the real (age, role) joint. Utility claims use matched evaluation on common support (§7.5). Young normal accounts are almost absent from the 2021 slice but common in the 2018 slice, so they are validated there (risk R1).
6. **The output is a native-ETH view.** Phishing approval drains (ERC-20) are invisible in it.

### 5.4 Pilot slice

**Windows.** Two slices, each 12 months UTC, built independently with the same preprocessing below (`--window` in `lpw.data.slice`):
- **2021 slice:** `W = [2021-05-01, 2022-05-01)`. 12 monthly files, 101.5M edges, about 6 GB. Output `data/pilot`.
- **2018 slice:** `W = [2018-01-01, 2019-01-01)`. 12 monthly files, 29.3M edges, 1.5 GB. Output `data/pilot_2018`. It was chosen by scanning candidate windows for matched-pair yield: 2017-07..2018-07 gave about 274 estimated test pairs, 2018-01..2019-01 about 346, and every window from 2019 on 47–83.

`W` below means the window of whichever slice is being built.

**Preprocessing order.** It runs one way and is frozen afterwards:
1. **Label-blind structural preprocessing**, on all W data. Labels are not used.
   - *Hubs:* EOA seeds with more than 2,000 E1 events in any calendar month of W. They become venues. Phase 0 reports their label mix.
   - *Agents:* the remaining EOA seeds with at least 1 E1 event in W.
   - *Venues:* hubs, plus counterparties touched by at least `S_pub` distinct agents. `S_pub` is chosen from the label-blind distribution of distinct-agent degree (default 50). At most 1,024 venues, ranked by distinct agents. This step sees test agents' structure (transductive) but never their labels.
   - *Groups:* Louvain communities (fixed seed) of the agent–counterparty graph after removing the venues. Connected components were tried first and fail (§5.2).
2. **Split by group** (scam clusters share counterparties, so a split by agent would leak). 70/15/15, stratified by role. Labels are used here only to stratify.
3. **Train-only fitting.** Everything that uses labels or is learned is fitted on the train split only: models, quantile bins, the group prior, activation rates and the detectors. Thresholds and model selection use the validation split (§7).

A secondary temporal check holds out the last 2 months of W, for the next-event metrics only.

**Inputs per agent**
- *Events:* the agent's E1 events in W, as one merged IN/OUT stream.
- *Cohort:*
  - **pre-existing:** `activity_start_ts` < W start;
  - **born in W:** `activity_start_ts` in W. The stream starts at that activation time.
- *Profile (pre-existing cohort only).* Aggregated from daily records strictly before W start (§5.1). Fields:
  - account age at W start;
  - days active;
  - lifetime counts and volumes;
  - peers;
  - ERC-20 activity;
  - an initial balance proxy: the cumulative `eth_recv_sum + internal_in_value_eth_sum − eth_sent_sum − internal_out_value_eth_sum − tx_fee_eth_sum`.

  The born-in-W cohort has an empty profile: age 0, no prior activity, balance 0. Phase 0 checks the sign of the balance proxy, its consistency with edge sums, and the coverage of the daily files.
- *Observation end:* W end. For an agent whose `activity_end_ts` falls inside W, it is instead the data end (2025-08-18), because the data show no activity after that date. Phase 0 checks `activity_end_ts` against the last edge.

**Scale (measured, §5.2).** 2021: 7,204 agents, 479k events. 2018: 24,936 agents, 479k events.

## 6. Design

### 6.1 Entities

| Tier | Real-data source | In the simulator |
|---|---|---|
| Agent | Non-hub EOA seeds | Runs the policy. Has a ledger, a fund inventory, a partner table, a profile, a role, a cohort, a group and (Phase 2) latent type `z`. |
| Venue | Hub seeds and public shared counterparties | Persistent pool with learned embeddings (popularity and role-specific affinity, like FINESSE merchants). Not simulated as a sequence. Released under synthetic IDs, with `node_type` (EOA, contract, unknown; from labels where known) and a `sender_capable` flag (appeared as a sender in the real data). |
| Group-shared node | A non-venue, non-agent counterparty touched by ≥2 agents of one group | Created fresh per synthetic group (Phase 3). Never shared across groups, so no global blacklist shortcut appears. Has a fund inventory, for taint. |
| Periphery | Every other counterparty | Created by NEW draws. Passive in v1: it receives OUTs and is the source of `IN_exo` events. |

### 6.2 Event token

Each agent stream is `[PROFILE] e_1 e_2 …`. Events are sorted by `(ts, block_number, tx_hash)`, so order within a second is deterministic.

The fields of an event `e_k`:
- `kind ∈ {OUT, IN_exo, IN_agent}`.
- `ts` encoded as:
  - `log1p Δt` since the previous token,
  - hour-of-day and day-of-week as clock encodings (sin/cos),
  - window-relative month, as an era/macro covariate.
- `cp_class ∈ {REPEAT, VENUE, GROUP, AGENT, NEW}`. Phase 1 uses only `{REPEAT, VENUE, NEW}`. GROUP and AGENT events fall back to NEW or REPEAT depending on the agent's own history. The precedence is REPEAT > AGENT > GROUP > VENUE > NEW, judged at event time:
  - REPEAT: the agent touched this counterparty before. It resolves to the actual node, which may be an agent.
  - AGENT: the counterparty is another simulated agent.
  - GROUP: a non-agent, non-venue node that another member of the agent's group touched before `ts`. Group pools never contain agents or venues.
- `cp_ref`:
  - REPEAT: the entry in the agent's partner table (§6.3).
  - VENUE: the venue id.
  - AGENT: an inductive descriptor of the other agent.
- `amount = log10(ETH)`, which is at least −4 because of the dust floor.
- `ledger` before the event: `log1p |b|` and `sign(b)`. Observed balances can go negative because some flows are unobserved.

The PROFILE token holds:
- the role embedding;
- the cohort, with activation time for the born-in-W cohort;
- the profile fields;
- (Phase 2) `z`.

### 6.3 Policy model

**Architecture.** A Banksformer-style decoder: 4 layers, `d_model = 128`, 4 heads, about 1M parameters. The context is the PROFILE token *always prepended*, plus the most recent 255 event tokens. The same context is used in training and simulation. It runs in fp16 with AMP (TU116 has 2× fp16 rate, no tensor cores).

**Factorization.** It follows Banksformer's conditional field order. Each head sees the context vector plus the fields already generated, with teacher forcing during training:

```
p(next owned event | h_k) = p(kind | h) · p(time | h, kind) · p(cp | h, kind, time) · p(amount | h, kind, time, cp)
```

**Kind head.** Softmax over `{OUT, IN_exo}`. `IN_agent` is never a target (§6.4).

**Time law** (the Banksformer date mechanism, adapted as one normalized law). It is used identically for training, censoring and sampling.
- *Heads,* conditioned on the kind:
  - `p_Δ` over 64 bins: bin 0 is "same second", bins 1–62 are log-spaced over [1 s, 365 d), and bin 63 is "≥ 365 d (dormant)";
  - `p_hour` (24 classes, UTC);
  - `p_dow` (7 classes).
- *Weekly weight.* `w(c) = p_hour(h(c)) · p_dow(d(c))` for each of the 168 hour cells `c` of the week, normalized so its weekly mean is 1.
- *Density.* For absolute time `t` after the last token time `t_k`:
  - **finite bins 1–62:** the density in seconds is `f(t | κ) = p_Δ(b) / |b| · w(c(t)) / Z`, where `b` is the bin containing `t − t_k` and `|b|` its width in seconds;
  - **bin 0:** an atom at `t_k` with mass `p_Δ(0) · w(c(t_k)) / Z`;
  - **bin 63:** an atom ("dormant") with mass `p_Δ(63) / Z`.
- *Normalizer.* `Z = p_Δ(0) w(c(t_k)) + Σ_{b=1..62} p_Δ(b) · c̄_b + p_Δ(63)`. Here `c̄_b` is the exact mean of the piecewise-constant `w` over `[t_k + lo_b, t_k + hi_b)`, computed in closed form from 168-cell prefix sums.
- *Owned-event law.* `p(κ, t | h) = p(κ | h) · f(t | κ)`. Its survival function is `S(r) = Σ_κ p(κ | h) · ∫_{t_k + r}^∞ f(t | κ) dt`, with the dormant atom included.
- *Exact sampling.*
  1. Sample κ.
  2. Sample a bin with probability proportional to its mass (`p_Δ(b) · c̄_b` for finite bins).
  3. Inside a finite bin, sample an hour cell in proportion to `w` times the cell's overlap with the bin.
  4. Sample a time uniformly in seconds inside that overlap.
- *Ablations.* A1 sets `w ≡ 1` (Δt only; the BF-ND analogue). Baseline B2 keeps Banksformer's original recipe (§7.1).

**Counterparty head**
- A gate over the counterparty classes.
- **REPEAT:** a pointer softmax over the agent's **partner table**, which lives outside the transformer window. The table holds every distinct past counterparty of the agent, with no cap. Hubs are excluded as agents, so tables stay small. Each entry carries:
  - log interaction count, time since the last interaction, log total in and out, and the node tier;
  - the hidden state of the counterparty's latest occurrence if it is inside the context, or else a learned "out-of-context" vector plus a flag.

  This is the learned, time-decayed memory. A revisit is never silently turned into NEW.
- **VENUE:** a softmax over venue embeddings.
- **NEW:** a fresh periphery id.
- **Phase 3 adds GROUP and AGENT.**
  - GROUP is a pointer over the group pool, which grows with NEW draws inside the group.
  - AGENT is a sampled softmax over currently alive agents, using inductive descriptors `φ_j`: observable state such as activity, age, balance bucket and group-membership flag. Never per-ID embeddings, so it works for synthetic populations. Training uses 63 negatives drawn from agents active in the same ISO week.
- **Masks for `IN_exo`:** the AGENT class, REPEAT entries that are agents, and every node that is not `sender_capable` (contract venues and contract-typed nodes). Agent-to-agent INs are owned by the sender, and contracts never send in this data.

**Amount head.** A mixture of 8 Gaussians on `log10 ETH`, conditioned on the fields above and on the ledger features. This is the BankGAN multimodality lesson. Samples below −4 are rejected and redrawn, to match the dust floor; the likelihood is renormalized above −4 to match.
- Ablation A3: a single Gaussian (the Banksformer default).
- Ablation A4 (optional): a CTGAN-style attribute head conditioned on the hidden state (BankGAN).

**Profile generator.** Synthetic agents need profiles. A small autoregressive model over quantile-binned profile fields learns `p(profile | role, cohort)`, and `p(profile | z, role, cohort)` from Phase 2. Profiles are not bootstrapped from real rows. The born-in-W cohort has an empty profile and needs no model.

### 6.4 Training

**Targets per prefix (competing risks).** Every prefix `k` (from PROFILE to the last token) gets exactly one time term. The origin is `t_k`, the time of the prefix's last token (or the activation time, for PROFILE).

| Next token after prefix `k` | Time term | Other terms |
|---|---|---|
| Owned (OUT or `IN_exo`) | `−log p(κ, t | h_k)` | Counterparty and amount NLL at this same prefix. Any `IN_agent` the event depends on, and its sender, are already inside the prefix. |
| `IN_agent` | Censored: `−log S(t_{k+1} − t_k)` | None |
| None (last token) | Censored at the observation end (§5.4): `−log S(T_obs − t_k)` | None |

For ended streams, the censoring time is more than 365 days, so the term equals the dormant atom's probability. A standalone Phase 1 generator never produces `IN_agent` tokens. They are about 1% of real events, so the P1 fidelity metrics compare against real streams with `IN_agent` removed.

- **Loss:** the sum of the terms above. The field weights start at 1 and are tuned on validation NLL only.
- **Long streams:** random windows of PROFILE plus 255 event tokens. The partner table and ledger carry state from before the window.
- **Optimizer:** AdamW with early stopping on validation NLL. Seeds are fixed.

### 6.5 Latent behaviour types (Phase 2)

1. Train a CoLES-style contrastive GRU encoder on train-split agent sequences (positives are sub-sequences of the same agent).
2. Fit a GMM on the embeddings. Choose K ≤ 32 by BIC.
3. Use the cluster as `z`. Condition the policy on `z` through the PROFILE token. The prior becomes `p(z | role, cohort) p(profile | z, role, cohort)`.
4. **Inference.** The training assignment of `z` uses the whole sequence, so it must never condition a held-out prediction.
   - Next-event metrics on held-out agents *marginalize* `z` over the prefix posterior `p(z | role, cohort, profile, observed prefix)`. It is proportional to the prior `p(z | role, cohort, profile)` times the policy's own likelihood of the prefix under each `z`, and it never uses future tokens. One pass over the sequence for each of the ≤ 32 values of `z` gives the posterior at every prefix, from cumulative per-token log-likelihoods.
   - Generation samples `z` from the prior.
   - Conditioning on the full-sequence `z` is reported only as a labelled oracle diagnostic.
5. Log `z` in the hidden-state log. It is LPW's typology ID (T5).
6. **Keep `z` only if it helps** (exit criteria, §8).

v1 has no latent drift. FINESSE-style switching is listed under §10.

### 6.6 Groups and shared counterparties (Phase 3)

- Real groups (§5.4) give a learned group prior: the empirical joint over group size and role mix in the train split.
- Each synthetic group gets an empty pool of group-shared nodes:
  - a member's NEW draw adds to its group's pool;
  - a member's GROUP draw picks from that pool.
- This reproduces *structure*: shared sinks and funders inside a cluster, plus cluster sizes. It does not create global identifiers that a detector could memorize.
- Without this, Phase 1 synthetic agents never share counterparties, and the graph metrics will show it. That is the reason for the phase.

### 6.7 World engine

**Population.** The configuration sets the number of agents N, the illicit prevalence π (independent of the balanced training set) and the horizon (W by default). Sampling order:
1. **Groups** (Phase 3) are drawn from the group prior, reweighted to hit π. Before Phase 3 every agent is its own group and roles are Bernoulli(π).
2. **Per member, in this order:** role from the group's role mix; then cohort and activation time from `p(cohort | role)`, using weekly activation rates per role from the train split (a scenario input, like TransXion's daily controller); then `z ~ p(z | role, cohort)` (Phase 2); then `profile ~ p(profile | z, role, cohort)`.

Pre-existing agents start at t0 with their profile balances. Born-in-W agents enter at their activation times.

**Clock (competing clocks).** Each agent keeps one pending "next owned event" time, drawn from the time law on its current prefix.
- Whenever a token is appended to an agent's stream, its pending time is discarded and redrawn from the new prefix. The token can be its own event, or an `IN_agent` delivered by another agent.
- The new prefix's origin is that token's timestamp. No truncation is needed.
- This matches training exactly (§6.4). A prefix followed by an `IN_agent` is censored at the delivery, and the next prefix's time is measured from the delivery.

**Exact sequential execution.** Agents couple through deliveries, group pools and the AGENT candidate descriptors `φ_j`, so commits must happen strictly in global time order.
1. Each agent holds its pending `(κ, t)` and the cached hidden state `h` of its current prefix. They sit in a priority queue keyed by `(t, agent id)`.
2. **Pop and commit.** Pop the earliest agent `A`. Evaluate its counterparty and amount heads from the cached `h`, with the candidate sets as they are at `t` (the group pool, alive agents, `φ_j`). This is valid because `A`'s own prefix has not changed since `h` was computed; if it had, `A` would have been redrawn. Commit the event: append it to `A`'s stream and update ledgers, inventories, partner tables and pools.
3. **Deliver.** If the event is an OUT to agent `R`, append an `IN_agent` token to `R`'s stream at `t`. Every earlier commit is at or before `t`, so streams stay time-ordered.
4. **Redraw immediately.** Run one transformer pass on the new prefix of `A`, and of `R` if there was a delivery. Push their new pending times. This happens *before* anything later is committed, so a redraw can never land before an event that is already committed.

**Invariant:** no context ever contains a token later than the action being generated, and commits are in global time order. The engine asserts both.

**Cost.** About one transformer pass per committed event (the actor's redraw), plus about 1% more for receivers. The pass covers PROFILE plus 255 tokens. A per-agent KV cache (in CPU RAM, about 0.5 MB per agent, about 3.5 GB for 7k agents) may be used only while an agent's stream is shorter than the window (at most 255 events). Once the window starts to slide, cached higher-layer states still encode evicted events, so every later pass for that agent is a full recomputation over PROFILE plus the current 255 events.

**Ledger** (the observed-balance feature): `b ← b + in − out − fee`.
- Fees are sampled from the empirical fee distribution for the same calendar day of the real window. This is an environment input, not a behaviour.
- The ledger is an input feature only, in every phase. Phase 0 found that 51–89% of real OUT events leave the observed balance negative, because of unobserved inflows (§5.2). So no conservation rule is enforced on it.

**Fund inventory** (for labels only; separate from `b`). Every engine-tracked node (agents and group-shared nodes) holds a non-negative clean part `c` and tainted part `x`.
- *Start:* pre-existing agents have `c = max(b0, 0)`, `x = 0`. Born-in-W agents and new group nodes start at 0.
- *Inflow* of amount `a` with taint `τ`: `x += τa`, `c += (1 − τ)a`. Untracked sources (venues, periphery) send with `τ = 0`.
- *Outflow* (including fees): if `a > c + x`, the shortfall is first added to `c`, as funding from an unknown, untainted source. Then the outflow takes the haircut share `τ_out = x / (c + x)` and is removed proportionally.
- *Source rule:* every OUT from an illicit agent has `τ = 1`.

### 6.8 Observation operator O and real reference R_W

- **O** applies the pipeline's recording rules that bear on agent-incident edges: the dust floor (`value ≥ 1e14 wei`) and the seed-incident rule, under which every edge with an agent endpoint is recorded.
- **R_W** is every real E1 edge in W with at least one non-hub agent endpoint. It is complete above the dust floor by construction (§5.1), so `O(world)` and `R_W` are like-for-like. Hub-only and E2 edges are excluded from both.
- Every real-vs-synthetic structural comparison uses `O(world)` against `R_W`, with the number of active agents matched per day.

### 6.9 Labels and outputs

- **`nodes`:** synthetic address id, tier, `node_type` (EOA for agents, otherwise as in §6.1), the observable profile fields, cohort, the role label and the group id. The released version separates the observable columns from the hidden ones.
- **`transactions`:** the same columns as the source edges, plus `is_illicit` and `taint`.
  - `is_illicit = 1` if the sender or receiver is an illicit agent, or if `taint ≥ θ` (default 0.5).
  - `taint` is the haircut share `τ` of the transfer, from the fund inventory (§6.7).
  - This labelling convention is a configuration choice, not a behaviour. TransXion also propagates labels along fund traces.
- **`hidden_state_log`** (daily, per agent): `z`, balance, inventory taint share, group, and alive/dormant.
- **Two releases:** the full world log and its projection `O(world)`.
- Only node-level labels can be checked against real data, because the real labels are node-level (§7.4). Transaction labels are synthetic by construction, and the documentation says so.

### 6.10 Learned vs hand-designed

| Learned from data | Hand-designed (engineering or measurement) |
|---|---|
| Event kinds, timing (Δt, diurnal and weekly), lifetimes and dormancy | Architecture and field order |
| Counterparty gate and choices (memory, venues, groups, agents, exploration) | The event loop and the redraw rule |
| Amounts, including their dependence on the ledger | Ledger and inventory arithmetic; fees drawn from the real fee distribution |
| Role differences, latent types, profiles | Thresholds: hub, `S_pub`, dust floor (copied from the pipeline) |
| Group sizes and role mix; activation rates | Observation operator (from the pipeline SQL) |
| | Taint rule and θ; scenario knobs (N, π, horizon) |

## 7. Evaluation

**Noise floor.** For every distributional metric, the noise floor is the same metric between two disjoint random halves of the real comparison split. It is bootstrapped 20 times, with sample sizes matched to the synthetic comparison. A metric "passes" at ≤ 2× its noise floor. Fidelity results (§7.2, §7.3) are reported per role, per cohort and per account-age tercile.

**Split discipline.** Every threshold, early stop, checkpoint and model choice uses the **validation** groups. The **test** groups are sealed. They are used once per phase, for the reported numbers.

### 7.1 Next-event prediction (P1; real test split)

- Per-field negative log-likelihood (NLL), with `z` marginalized over its prefix posterior from Phase 2 (§6.5).
- Next-kind error rate and Δt-bin accuracy. These are FINESSE/EasyTPP-style metrics, plus time MAE on `log Δt`.
- Counterparty-class accuracy and REPEAT Hits@{1,5} over the full partner table.
- Amount NLL and MAE on `log10`.

**Baselines:**
- B0: per-role marginals with a Markov-1 model on kind.
- B1: a GRU with the same heads (BankGAN's StackedLSTM analogue).
- B2: Banksformer-adapted. Δt, hour and weekday heads trained independently with cross-entropy and combined as a product at sampling (the original recipe), trained on uncensored prefixes only. It has no counterparty head and a single-Gaussian amount.

### 7.2 Single-agent generative fidelity (P1–P2)

Generate one stream per test agent, conditioned on its real role, cohort, activation time and profile (paired), with `z` drawn from `p(z | role, cohort, profile)`. Also generate unconditionally from sampled profiles.

| ID | Metric | Distance |
|---|---|---|
| M1 | log10 amount, overall and per kind | W1 |
| M2 | log Δt | W1 |
| M3 | kind | JSD |
| M4 | hour-of-day | JSD |
| M5 | day-of-week | JSD |
| M6 | kind 3-gram | JSD |
| M7 | joint (kind, hour), the Banksformer joint analogue | JSD |
| M8 | counterparty class | JSD |
| M9 | per-agent repeat rate | W1 |
| M10 | distinct counterparties per agent | W1 |
| M11 | events per agent | W1 |
| M12 | active span | W1 |
| M13 | out-fraction per agent | W1 |
| M14 | per-agent hour entropy | W1 |
| M15 | monthly net flow per agent (Banksformer cash-flow analogue) | W1 |
| M16 | end-of-window observed balance (sweeping) | W1 |

**Units.** Events per agent and distinct counterparties (M10, M11) are compared as `log1p` counts. Monthly net flow and end balance (M15, M16) are compared as signed `log1p` ETH. On a raw scale, W1 on these heavy-tailed quantities is set by a few whale accounts, so both the noise floor and the test become blind to typical accounts. Real and synthetic go through the same transform. M15 covers every agent-month from the agent's origin to the window end, with inactive months counted as 0. M10 counts zero-event agents as having 0 counterparties. M1 passes only if the overall, OUT and `IN_exo` amount distributions all pass.

**Role-signal parity.** Train XGBoost on real-train per-agent summaries (M9–M16 features plus profile) to predict role. Compare its AUC on real test with its AUC on synthetic agents. A large gap in either direction is a failure: a higher synthetic AUC means caricature; a lower one means the signal was lost.

### 7.3 World structure (P3; on O(world) vs R_W)

- **TransXion RQ1 set** on daily undirected projections: GCC ratio, number of components, max k-core, max-core fraction, degree assortativity and transitivity.
- **Clauset power-law fits** (with lognormal likelihood ratio) for unique degree, transaction-count degree, strength and amount.
- **Coupling metrics:**
  - shared-counterparty fraction per agent (real: 0.40 normal, 0.68 scam);
  - group size distribution;
  - agent↔agent event fraction (real: about 1%).
- **Engine check:** the causality and ordering invariants hold (§6.7), and a fixed seed reproduces the world exactly.

### 7.4 AML utility and criticism tests (P3+)

- **TSTR (headline; T4).** Detectors trained on synthetic and tested on real test-split groups, for node-level illicit classification. Report AUC, AP and F1, next to train-real/test-real (TRTR). TRTR is also reported on the clean test subset (§5.2), because real-trained detectors could exploit counterparties shared with train agents. Detectors: logistic regression, random forest, XGBoost, LightGBM, and a GNN once one is feasible (Multi-GIN, as in TransXion).
- **Detector-ranking agreement.** Spearman correlation between the detector ranking on synthetic→synthetic and on real→real.
- **Difficulty parity (T3).** At **matched prevalence** (the synthetic test set subsampled to the real test split's illicit rate), AP on synthetic→synthetic should be within ±0.05 of AP on real→real. Other prevalence scenarios are reported separately. LPW targets realism, not maximal hardness.
- **Shortcut audit (C2).** Single-feature AUC for the top 20 features, synthetic vs real. Flag any |Δ| > 0.1. Template-like shortcuts show up as single features that are near-perfect in synthetic data but not in real data.
- **Profile-gain parity (C1).** The change in detector AUC/AP from adding profile features. It should have the same sign on synthetic and real, with |difference| ≤ 0.05.
- **Seam test (T2).** C2ST: a classifier separating real from synthetic agents, run per role. Pass only if both conditions hold:
  - `AUC ≤ 0.70` for *each* role (absolute distinguishability);
  - `AUC_illicit − AUC_normal ≤ 0.05`, meaning illicit agents are no easier to spot as synthetic than normal agents are.
- **Mode coverage (C2).** Improved precision and recall (Kynkäänniemi et al., 2019) on Phase 2 encoder embeddings, per role.
- **Memorization.** Distance to closest record (DCR): synthetic→train distances should be no smaller than test→train distances (5th-percentile ratio ≥ 1). This guards against copying real phishing traces.

### 7.5 Confound controls (applies to all of §7.4)

Account age, era, cohort and their proxies are strongly confounded with the label. The proxies are days active, lifetime counts and volumes, and profile missingness. Every utility number is reported three ways:
1. with all features;
2. without age, era, cohort and proxy features;
3. **matched on common support.** Fit a propensity model for role on age, era, cohort and the proxies, using the train split. Trim to overlapping propensity, match illicit and normal agents, and report the effective N per role.

The **conditional baseline** is a detector that uses only those confounder features, evaluated in the same matched sample. Behavioural signal is the full detector's gain over that baseline inside the matched sample. If the matched sample has fewer than 200 agents per role, the claim is reported as "not testable" for that slice. In the pilot this applies to the 2021 slice; the 2018 slice is the one that carries the utility claims (§5.2).

The thresholds in §7 are initial targets, recalibrated on validation data once P1 results and noise floors exist. Any change is recorded in this file.

## 8. Phases, exit criteria and compute estimates

The estimates are for the GTX 1650 Super. Each one is replaced by a measurement before the matching run (§9).

| Phase | Work | Exit criteria | Est. wall time | Where |
|---|---|---|---|---|
| **P0** Data audit + pilot slices (**done** for 2021 and 2018, §5.2) | Set up the environment (uv, Python 3.12, PyTorch CUDA on sm_75, fp16 smoke test). Download W months (~6 GB for 2021, 1.5 GB for 2018) plus labels and daily feature records. Build agent streams, cohorts, profiles, venues, groups and splits (§5.4 order). Re-verify §5.1. Measure: negative-balance rate, `activity_end_ts` vs last edge, daily-file coverage, group sizes and purity, hub label mix, age-only AUC in W, matched-sample size (§7.5), training throughput. | Slice built. Audit report written. §5 updated with the measured numbers. | 1–2 h (mostly download) | local |
| **P1** Next-event predictor → single-agent generator | Policy model, B0–B2, ablations A1 and A3 (A4 optional), profile generator, M1–M16. | On each slice: beats B0 on held-out full-event NLL. Non-inferior to B1 on held-out full-event NLL: the one-sided 95% upper bound of LPW − B1 (paired bootstrap over whole groups) is ≤ +0.03 nats per owned event. Generates better than B1: in each role, LPW's mean M1–M16 pass rate over the same generation replicas is above B1's (amendment A1). Beats B2 on the NLL of the fields both models have (kind, time including censored terms, amount). Every model is scored on identical evaluation prefixes. At least 70% of M1–M16 pass at ≤ 2× noise floor for each role. Role-AUC within ±0.05 of real. | Minutes per training run (measured: about 7 s per epoch per slice); ≤ 4 h for the whole phase grid across both slices | local |
| **P2** Latent types | Encoder, GMM, `z`-conditioning, ablation against no-`z`. | Keep `z` only if M9–M16 or role parity improve on validation, with nothing getting worse beyond noise. Otherwise drop it and record why. | ≤ 1 h | local |
| **P3** World | Groups, AGENT class, engine, ledger, inventory, O, labels, §7.3 and §7.4. | On each slice, §7.3 metrics reported against that slice's `R_W`. With confounder features removed, the excess-AUC ratio `(AUC_TSTR − 0.5) / (AUC_TRTR − 0.5)` is ≥ 0.8 (initial target). This gate only counts when `AUC_TRTR − 0.5 ≥ 0.1`; otherwise the result is reported as not testable. Seam test and shortcut audit pass. The §7.5 matched utility result comes from the 2018 slice. | Sequential simulation per slice (7.2k agents in 2021 and 24.9k in 2018, each about 0.48M events at the measured 1.66 ms per batch-1 pass, plus overhead): 15–30 min each. GNN detector: 1–3 h. | local; GNN borderline |
| **P5** Scale-up | Multiple eras (2017–2025, about 21 GB), all non-hub seeds in one model with an era covariate, which tests transfer across eras. Contracts become passive sinks. | P1 and P3 metrics hold in each era. | Data build 2–4 h; training 1–3 h; simulation 3–8 h | **likely cloud** |

P4 is intentionally absent: hardening moved to future work (§10).

**Amendment A1 (2026-11-06): B1 criterion of P1.**
- **Original wording:** "beats B0 and B1 on held-out full-event NLL".
- **What prompted it:** the pre-registered encoder screen (`reports/p1_encoder_screen.md`) failed its B1 margin.
  - On val, ALiBi − B1 was +0.017 and +0.023 nats per owned event over two seeds.
  - B1 generated much worse populations: normal M-pass 0.19 vs 0.50, with capped streams.
- **Why the criterion changed:** better likelihood than a GRU with the same heads is not what the simulator is for. Generation fidelity is.
- **Disclosure:** the change was made after seeing those results. It was decided with the user and a Sol design round.
- **Reporting:** every P1 result also reports the original criterion, which is currently unmet.
- **Unchanged:** §6.3 (still a transformer) and every other exit criterion.

**Amendment A2 (2026-11-07): bounded P3 entry after incomplete P1.**
- **Decision.** After inspecting development results, P1 architecture search stops. It meets neither the original nor
  the A1 exit criteria. Decided with the user and a Sol design round; recorded in `reports/p1_exit.md`.
- **Frozen recipe.** kct (`reports/p1_shallow.md`) with the default train-fitted plan prior, for 2018 P3 feasibility.
  B (its base, without the shallow correction) is kept as a sensitivity control.
- **Entry requires:** a frozen recipe, documented rollout-safety and role-parity diagnostics, and the B0 likelihood
  comparison.
- **What stops gating.** B1 non-inferiority and generation superiority no longer gate entry. B1 / B2 comparisons and
  every original threshold are still reported, failures included.
- **Diagnostic targets.** The 70% per-role fidelity and ±0.05 role-AUC thresholds stay as diagnostic targets. They are
  not retrospectively passed criteria.
- **P3 requirements.** The utility and criticism-test requirements are unchanged, within the narrowed 2018 scope. They
  require genuinely sampled populations: P1's prior mode samples plans only, keeping real profiles, cohorts and
  activation times.
- **2021.** Its failure (scam M-pass about 0 for every model) is reported, with no cross-era qualification.
- **P2.** Conditioning is deferred. Mode coverage (§7.4) is still evaluated, with an evaluation-only, train-fitted
  encoder.
- **Sealed test.** It stays untouched until an authorized frozen evaluation.
- **Resuming P1.** Only for an identified P3-blocking defect, under one explicitly bounded experiment with its budget
  set beforehand.

## 9. Compute policy

- Before every run longer than a smoke test, project the wall time from measured throughput (tokens/s, events/s).
- If a single experiment is projected to take **more than 4 h** locally, or will not fit in 4 GB of VRAM or about 8 GB of free RAM even after reasonable reductions, **stop and ask the user** whether to rent a cloud GPU. The message gives the estimate and the reason.
- Never shrink the evaluation silently to stay inside the budget.

## 10. Risks and open questions

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Label/age confound: age-only AUC 0.96 in the 2021 slice, which has almost no young normal accounts (48 born in W). | Condition on age and cohort. Matched evaluation (§7.5) on the 2018 slice, where age-only AUC is 0.60 and 16k normal accounts are born in W. |
| R2 | Small populations: 2,867 normal agents in 2021, 2,326 scam agents in 2018 | Small model, early stopping, group split. P5 adds eras. |
| R3 | Native-ETH-only blind spot (ERC-20 approvals, internal transactions) | Documented as the scope of the output. ERC-20 counts appear only as profile fields. |
| R4 | Observed balances are unreliable (confirmed in Phase 0: 51–89% of OUT events leave them negative) | The ledger is a feature only. Labels use a separate non-negative inventory (§6.7). |
| R5 | The product-of-heads time law is too restrictive | Ablation A1 and baseline B2. |
| R6 | Biased 2-hop sampling | v1 uses only complete agent-incident E1 edges (§6.8). |
| R7 | Noisy curated labels; scam labels on token contracts | Report the description mix. Contracts are out of pilot scope. |
| R8 | Venue-ID shortcuts | Split by group, group-scoped pools, shortcut audit. |
| R9 | The exact sequential engine is slow at P5 scale | CUDA graphs for batch-1 passes, and a KV cache only for streams that have not yet filled the window (§6.7). For P5, a parallel optimistic (Time Warp-style) engine with rollback, validated against the sequential engine on small worlds. |

**Future work (not v1):**
- **Second-hop (layering) model.** It needs provenance: full transaction histories of L1 nodes, for example from the public BigQuery `crypto_ethereum.transactions` table that the pipeline itself used. Then forwarding and non-forwarding are both observable.
- **Realism-constrained hardening.** KL-regularized GRPO on the illicit policy's own actions (TransXion App. A advantage), with a monitor from one detector family and evaluation on another. Guards use validation groups only. Policies are not released. Likely needs a cloud GPU.
- Latent drift (FINESSE behaviour switching).
- A bank/exchange monitoring agent that freezes flagged accounts, analogous to the FINESSE policy stream, with illicit agents that adapt to it.
- ERC-20 streams.
- KYC-style profiles for transfer to bank AML.

## 11. Planned layout and tooling

```
SPEC.md
pyproject.toml            # uv; Python 3.12; torch (CUDA), polars, numpy, scikit-learn, xgboost, lightgbm, igraph, powerlaw, pytest
src/lpw/data/             # download, audit, slice building (streams, cohorts, profiles, venues, groups, splits)
src/lpw/model/            # policy transformer, time law, heads, partner table, profile generator, baselines
src/lpw/sim/              # world engine, ledger, fund inventory, observation operator, labels
src/lpw/eval/             # metrics M1–M16, structure, detectors, TSTR, matched evaluation, audits
configs/                  # one YAML per run; seeds fixed
tests/                    # unit tests, starting with the time law (normalization, survival, sampling) and the engine causality invariant
data/, runs/              # git-ignored
```

Downloaded files are untrusted. Raw data goes in `data/raw/` and is only read by scripts that live outside it, and no tool is run from inside it.
