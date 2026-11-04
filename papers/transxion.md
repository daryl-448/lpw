# TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering

- **Authors:** Keyang Chen\*, Mingxuan Jiang\* (equal contribution), Yongsheng Zhao, Zeping Li, Zaiyuan Chen, Weiqi Luo, Zhixin Li, Sen Liu, Yinan Jing, Guangnan Ye† (Fudan Univ.); Xihong Wu (Peking Univ.); Hongfeng Chai (Fudan). † = corresponding author (the footnote says "authors", but only Ye is marked).
- **Venue:** KDD '26 (32nd ACM SIGKDD, V.2), 9–13 Aug 2026, Jeju. DOI 10.1145/3770855.3817550. Pages 8707–8718.
- **Code + data:** `https://github.com/chaos-max/TransXion`
- **Task:** A synthetic AML **transaction-graph benchmark** with entity profiles and laundering subgraphs that are not built from templates.

## Core idea (TL;DR)

The paper names two flaws in existing AML benchmarks (AMLSim, AMLWorld/IBM, SAML-D, Elliptic, T-Finance):

1. **Edge-only logs with no node semantics.** Without KYC-style context, models cannot catch "out-of-character" behaviour: activity that looks plausible on its own but contradicts the entity's socio-economic profile.
2. **Template-driven anomaly injection.** Fixed motifs such as fan-in, fan-out and smurfing lead models to learn shortcuts, so robustness looks better than it is.

TransXion is a three-step pipeline:
1. A **profile-aware agent-based normal backbone** that generates profiles and transactions together.
2. **Non-template anomaly synthesis.** An LLM policy edits *real illicit subgraphs*, trained with **GRPO** against a detector "monitor" so the edits reduce detectability.
3. **Profile-conditioned embedding** of each illicit cluster, intact, into the backbone under structural, profile and temporal constraints.

The result is about **3M transactions among about 50K entities** with rich demographic profiles. Detectors score much lower on it than on prior benchmarks, mainly on AP and F1.

## Generative framework

### 3.1 Normal backbone (agent-based, closed loop, hourly windows)

Each **one-hour window** runs four steps:
1. Sample initiators.
2. Sample counterparties.
3. Sample event attributes.
4. Update the interaction state.

Components:
- **Persistent profiles:**
  - Accounts have socio-economic attributes and behavioural priors: daily intensity, typical amount scale and diurnal rhythm.
  - Merchants have a business type and an operating scale.
  - Profiles set the **per-window activation probability** and the **attribute samplers**, so "out-of-character" means a deviation from the profile's expectations.
- **Scenario controller (macro layer):**
  - Updates once a day, for example for holidays and regional demand.
  - Outputs multiplicative modifiers: a **global intensity factor** and **region-level mixing weights**.
  - These change only aggregate volume and regional mix, never individual events or micro behaviour. A fixed update cadence keeps the layer reproducible.
- **Socially structured sampling:**
  - Initiators are sampled in proportion to profile-derived activity weights.
  - Counterparties come from a **weighted mixture of four terms**: locality preference, **time-decayed interaction memory**, merchant attractiveness, and limited exploration for new ties.
  - Amounts and other attributes stay consistent with the initiator's profile.
  - The paper gives no explicit formula, only "in minimal form, this is a weighted mixture".
- **Evolving interaction state:**
  - Stores time-decayed partner preferences and region hotspots, updated after each hourly window.
  - Creates path dependence: repeated ties become more likely while new links still appear stochastically.
  - Produces temporal persistence, a stable diurnal structure and heavy-tailed activity.

### 3.2 Anomalous transaction synthesis

Edits are applied to **realistic illicit transaction subgraphs**. Each edit keeps basic sanity: valid endpoints, non-negative amounts and a consistent time order.

**Four action families:**
- **Intermediary Injection:** insert intermediate accounts to lengthen chains and hide where funds came from.
- **Account Merging:** combine behaviourally similar accounts to mask repetitive structure.
- **Account Splitting:** spread flows across several recipients to dilute concentrated signals.
- **Transaction Adjustment:** perturb amounts or timestamps within feasible ranges.

Selecting the edits is framed as a **constrained sequential decision process**. A synthesis agent proposes edits, and a **pre-trained monitor** (a detector) scores the result.

### Appendix A: RL details

The paper withholds institution-specific rules and case details to reduce misuse risk.

**Rollouts:**
- For each cluster `C` (a subgraph), the policy draws `K` trajectories: `{τ_1, …, τ_K} ∼ π_θ(· | C)`.
- At each step the agent observes a state derived from the graph, emits a **structured action** (a graph edit), and the environment applies it.

**Monitor score**, computed before and after each action:
```
S_pre = S(G_t),   S_post = S(G_{t+1})
S = w1·F1 + w2·AUC + w3·AP            (w1, w2, w3 ≥ 0, set per config)
```

**Step reward:**
```
R_t = R_validity(a_t) + λ_mon · ΔS_t,     ΔS_t = S_pre − S_post
```
`R_validity` penalizes malformed or non-executable actions and may end the trajectory after severe violations.

**GRPO advantage.** `J_i` is the cumulative return of trajectory `τ_i`:
```
μ = (1/K) Σ_{i=1}^{K} J_i,     σ = sqrt( (1/K) Σ_{i=1}^{K} (J_i − μ)² )
A_i = (J_i − μ) / (σ + ε),     ε > 0
```

**Objective** (no critic network):
```
L(θ) = −(1/K) Σ_{i=1}^{K} A_i Σ_t log π_θ(x_{i,t} | x_{i,<t}, C)
```
`x_{i,t}` are the action tokens, so the policy is an autoregressive (LLM) generator.

**Hyperparameters:**
- AdamW, lr 1e-5, weight decay 0.01, gradient accumulation 4, max gradient norm 1.0.
- **LoRA** with r = 8, α = 16, dropout 0.05, updating about 0.07% of the base parameters.
- The **base LLM is not named**.
- This is a one-time cost when building the benchmark.

### 3.3 Anomaly-to-normal embedding

Each synthesized cluster is inserted **as an intact subgraph**. **Labels are transferred along fund traces**, so the cluster stays illicit by construction.

The embedding is accepted only if all three constraints hold:
1. **Structural preservation:** every sender–receiver pair is kept under a one-to-one mapping of the cluster's participants.
2. **Profile alignment:** each anomalous role maps to a normal account whose profile-consistent activity regime falls in the role's required range.
3. **Temporal coherence:** the cluster sits in a feasible time window, and its internal ordering is preserved.

This is an **atomic injection policy**: a cluster that breaks any constraint is rejected. Afterwards, stylized facts of the whole dataset (e.g. heavy tails) are re-checked.

## Dataset

**Tables:** Persons, Merchants and Transactions.

**Person profile attributes (Fig. 3 Sankey):**
- Gender
- Age band: <25, 25–35, 36–45, 46–55, 56–65, >65
- Education: below high school, high school, associate, bachelor's, graduate or higher
- Marital status: married, single, divorced
- Occupation: white-collar, blue-collar, business owner, freelancer, student, retired, unemployed, other

**Table 7:**

| Days | Accounts | Transactions | Laundering | 1 per N |
|---|---|---|---|---|
| 365 | 47,526 | 3,029,170 | 4,641 | 652.70 |

Prevalence is **0.153%**, about 1 in 653. The headline figure of "50,000 entities" is rounded, or includes merchants; Table 7 counts 47,526 accounts.

**Transaction schema (App. B.1):**
- `From Bank`, `From Account`, `To Bank`, `To Account`, `Timestamp`
- `Amount Paid` + `Payment Currency`
- `Amount Received` + `Receiving Currency` (paid and received amounts can differ through FX)
- `Payment Format`
- `is_laundering` (0/1)

This matches the AMLWorld/IBM column layout. Build account keys as bank + account to avoid ID collisions. Labels are **transaction-level only**, with no typology IDs.

**Payment formats (Fig. 6, approximate):** Mobile ≈ 0.9M, Card ≈ 0.8M, Transfer ≈ 0.67M, Cash ≈ 0.67M. The paper warns that channel can be a shortcut and recommends stratified analysis.

**Account activity:** most accounts make 10–250 transactions per year, with a small number that are rarely or highly active.

## Evaluation

### RQ1: fidelity to payment-network stylized facts

**Method:**
- Build daily directed networks, following Soramäki et al. 2007.
- Compute invariants on **daily undirected simple-graph projections**, reported as mean ± std.
- Fit tails by **MLE with automatically selected x_min** (Clauset, Shalizi & Newman 2009).

**Table 1:**

| Indicator | TransXion | AMLWorld | AMLSim | SAML-D |
|---|---|---|---|---|
| GCC ratio | 0.8924 ± 0.0321 | 0.3824 ± 0.3373 | 0.8355 ± 0.0130 | 0.0009 ± 0.0001 |
| # connected components | 308.5 ± 49.7 | 24,102.7 ± 41,639.3 | 389.5 ± 24.7 | 9,624.9 ± 461.9 |
| Max k-core | 2.9151 ± 0.4264 | 1.8333 ± 0.7638 | 2.93 ± 0.2551 | 1.0000 ± 0.0000 |
| Max-core fraction | 0.0703 ± 0.0875 | 0.4581 ± 0.4456 | 0.0602 ± 0.0616 | 1.0000 ± 0.0000 |
| Degree assortativity | −0.4943 ± 0.0279 | −0.2593 ± 0.2817 | −0.2009 ± 0.0114 | −0.1188 ± 0.0048 |
| Global transitivity | 0.0021 ± 0.0005 | 0.0007 ± 0.0028 | 0.0012 ± 0.0002 | 0.0000 ± 0.0000 |
| Strength-tail KS D (x_min \| n_tail) | 0.0570 (120.8k \| 20.0k) | 0.0204 (531.0M \| 5.2k) | 0.1046 (21.5M \| 4.4k) | 0.0445 (80.6k \| 402.1k) |
| Amount-tail α (x_min \| n_tail) | 2.9255 (40.3k \| 30.3k) | 1.5423 (623.8k \| 253.9k) | 1.3267 (85.5k \| 26.5k) | 3.0000\* (10.2k \| 2,471.3k) |

\* α hit the boundary of the fitting range, typically [1, 3].

**Claims:**
- **Connected backbone:** the GCC ratio is about 0.89.
- **Sparse but non-zero triadic closure.**
- **Strong disassortativity** (−0.49), suggesting hub-and-spoke tiering.
- **Multi-layer cores:** max core about 2.9, against degenerate 1-cores in SAML-D.
- **Plausible heavy tails.**

AMLSim is close on several invariants (GCC, k-core, max-core fraction). AMLWorld has a lower strength-tail KS D. The paper argues TransXion gives the best *balance* overall, not that it wins every metric.

**Table 8 (appendix): tail fits under automatic x_min.** R < 0 favours lognormal over a power law.

| Dataset | Distribution | x_min \| n_tail | α | D | R, p |
|---|---|---|---|---|---|
| TransXion | degree_unique | 56 \| 24.0k | 2.904 | 0.190 | −4.34×10², 4.92×10⁻³⁰ |
| TransXion | degree_tx_count | 79 \| 23.8k | 2.904 | 0.108 | −3.88, 6.25×10⁻¹ |
| TransXion | strength | 120.8k \| 20.0k | 2.405 | 0.057 | −1.12, 1.48×10⁻¹ |
| TransXion | amount | 40.3k \| 30.3k | 2.926 | 0.042 | −3.26×10¹, 3.11×10⁻³ |
| AMLWorld | degree_unique | 4 \| 111.3k | 3.000 | 0.045 | −1.79×10³, 1.04×10⁻⁸⁰ |
| AMLWorld | degree_tx_count | 40 \| 79.6k | 3.000 | 0.049 | −2.97×10³, 1.82×10⁻⁸⁷ |
| AMLWorld | strength | 531.0M \| 5.2k | 1.788 | 0.020 | −4.23, 6.84×10⁻² |
| AMLWorld | amount | 623.8k \| 253.9k | 1.542 | 0.022 | −1.32×10³, 2.01×10⁻²⁰² |
| AMLSim | degree_unique | 12 \| 4.5k | 2.931 | 0.044 | −5.96×10¹, 6.42×10⁻¹¹ |
| AMLSim | degree_tx_count | 219 \| 4.9k | 2.968 | 0.030 | −2.23×10¹, 3.86×10⁻⁵ |
| AMLSim | strength | 21.5M \| 4.4k | 2.148 | 0.105 | −2.56×10², 2.47×10⁻³⁹ |
| AMLSim | amount | 85.5k \| 26.5k | 1.327 | 0.158 | −5.56×10³, < 10⁻³⁰⁰ |
| SAML-D | degree_unique | 1 \| 855.5k | 3.000 | 0.118 | −3.52×10⁴, < 10⁻³⁰⁰ |
| SAML-D | degree_tx_count | 215 \| 17.1k | 2.235 | 0.240 | −5.77×10³, < 10⁻³⁰⁰ |
| SAML-D | strength | 80.6k \| 402.1k | 2.228 | 0.045 | 1.57×10¹, 7.05×10⁻³⁰⁸ |
| SAML-D | amount | 10.2k \| 2471.3k | 3.000 | 0.030 | −2.19×10², 4.54×10⁻²³ |

**Appendix C argument:**
- Exponents should *differ* between structural and monetary variables.
- Several baselines hit the α = 3.000 boundary, which the authors read as rigid or over-regularized behaviour.
- SAML-D degree_unique has x_min = 1 (n_tail = n), so it is degenerate.
- TransXion amounts are **lognormal-like**, which is plausible for multiplicative or mixture amount mechanisms. Its structural variables keep heavy tails.

### RQ2: detection hardness

**Protocol:**
- Supervised **edge classification** on a directed temporal multigraph, keeping parallel edges.
- **Edge features only**, for fairness, because some benchmarks have no node attributes.
- Chronological **60/20/20** split.
- Metrics: AUC, AP and F1.

**Detectors:**
- GBTs: LightGBM and XGBoost, on per-transaction attributes.
- GNNs, using the directed-multigraph framework of Egressy et al. 2024: Multi-GIN, Multi-GIN+EU (edge updates), Multi-PNA, Multi-PNA+EU and Multi-GAT.

**Table 2: GBT**

| Dataset | Metric | LightGBM | XGBoost |
|---|---|---|---|
| AMLSim | AUC / AP / F1 | 0.4906 / 0.2035 / 0.3598 | 0.9931 / 0.9206 / 0.9171 |
| AMLWorld | AUC / AP / F1 | 0.8549 / 0.1369 / 0.2569 | 0.9631 / 0.3761 / 0.4046 |
| SAML-D | AUC / AP / F1 | 0.7906 / 0.1361 / 0.2654 | 0.8521 / 0.2436 / **0.1872** |
| TransXion | AUC / AP / F1 | **0.4128 / 0.0313 / 0.1383** | **0.7869 / 0.1716** / 0.2375 |

Bold marks the lowest value per detector and metric. XGBoost F1 is actually lowest on **SAML-D** (0.1872), not TransXion, although the text says both GBTs are weakest on TransXion.

**Table 3: GNN (edge-only)**

| Dataset | Metric | Multi-GIN | Multi-GIN+EU | Multi-PNA | Multi-PNA+EU | Multi-GAT |
|---|---|---|---|---|---|---|
| AMLSim | AUC | 0.929836 | 0.931760 | 0.948663 | 0.940602 | 0.931399 |
| AMLSim | AP | 0.693337 | 0.728561 | 0.637610 | 0.596686 | 0.458262 |
| AMLSim | F1 | 0.775000 | 0.802885 | 0.697337 | 0.700535 | 0.486216 |
| AMLWorld | AUC | 0.986481 | 0.982223 | 0.982513 | 0.985354 | 0.978961 |
| AMLWorld | AP | 0.638040 | 0.504771 | 0.630889 | 0.672263 | 0.395064 |
| AMLWorld | F1 | 0.636644 | 0.652040 | 0.627935 | 0.709499 | 0.451544 |
| SAML-D | AUC | 0.999975 | 0.999746 | 0.999934 | 0.999984 | 0.999757 |
| SAML-D | AP | 0.991920 | 0.991262 | 0.988842 | 0.994835 | 0.971794 |
| SAML-D | F1 | 0.912689 | 0.937154 | 0.968137 | 0.976988 | 0.939498 |
| TransXion | AUC | 0.968837 | 0.952841 | 0.968291 | 0.968088 | 0.951603 |
| TransXion | AP | 0.287842 | 0.268952 | 0.350831 | 0.345992 | 0.277935 |
| TransXion | F1 | 0.433980 | 0.424686 | 0.486515 | 0.467446 | 0.433566 |

TransXion has the lowest AP and F1 for every GNN, and the second-lowest AUC. AMLSim has the lowest AUC but high AP and F1.

The authors' reading:
- AUC measures global pairwise ranking.
- AP and F1 reflect the high-score, thresholded region that matters for alerting.
- TransXion puts its difficulty in that decision-critical region, so it is **SAML-D that is near-trivial** (AUC ≈ 0.9999).

### Node-profile context (§5.3.2)

**Fig. 4:** XGBoost and LightGBM AUC and AP generally rise as the share of node-profile attributes used grows from 0 to 1. Approximate relative gains at ratio 1.0, read from the plot: LightGBM AP ≈ +173%, LightGBM AUC ≈ +55%, XGBoost AP ≈ +40%, XGBoost AUC ≈ +14%.

**Fig. 5:** gain from adding profile features to GBTs.

| | XGBoost ΔAUC | XGBoost ΔAP | LightGBM ΔAUC | LightGBM ΔAP |
|---|---|---|---|---|
| TransXion | +0.085 | +0.034 | +0.295 | +0.124 |
| AMLWorld | −0.033 | −0.021 | +0.046 | +0.064 |

**Table 4:** GNNs on TransXion, with node attributes fused with edge features before edge classification.

| Setting | Metric | Multi-GIN | Multi-GIN+EU | Multi-PNA | Multi-GAT |
|---|---|---|---|---|---|
| Edge-only | AUC | 0.9688 | 0.9528 | 0.9683 | 0.9516 |
| Edge-only | AP | 0.2878 | 0.2690 | 0.3508 | 0.2779 |
| Edge-only | F1 | 0.4340 | 0.4247 | 0.4865 | 0.4336 |
| Profile+Edge | AUC | 0.9550 | 0.9718 | 0.9759 | 0.9452 |
| Profile+Edge | AP | 0.3173 | 0.3176 | 0.3746 | 0.2916 |
| Profile+Edge | F1 | 0.4666 | 0.4660 | 0.4732 | 0.4654 |

Adding profiles improves AP for all four backbones and F1 for three of four (Multi-PNA drops). AUC drops for Multi-GIN and Multi-GAT.

### Anomaly-synthesis ablation (§5.3.3)

The graph substrate is held fixed: the agent "hardens" **AMLWorld**, not TransXion, so the effect can be isolated.

Configurations:
- **Original:** AMLWorld unmodified.
- **LLM-Only:** an LLM synthesis agent with no monitor-guided training.
- **RL-Optimized:** the same pipeline, using a policy trained with monitor guidance.

**Table 5:**

| Config | Multi-GIN AUC / AP / F1 | XGBoost AUC / AP / F1 |
|---|---|---|
| Original | 0.9864 / 0.6380 / 0.6366 | 0.9631 / 0.3761 / 0.4046 |
| LLM-Only | 0.9841 / 0.5384 / 0.5741 | 0.9481 / 0.4008 / 0.4573 |
| RL-Optimized | 0.9796 / 0.5121 / 0.5037 | 0.9474 / 0.3276 / 0.3849 |

LLM-Only gives mixed effects: XGBoost AP and F1 actually *rise*. RL-Optimized lowers every metric for both detectors.

**Table 6: cross-family transfer.** The monitor is a GBT; evaluation uses Multi-GIN on AMLWorld.

| Config | AUC | AP | F1 |
|---|---|---|---|
| Original | 0.9864 | 0.6380 | 0.6366 |
| LLM-Only | 0.9841 | 0.5384 | 0.5741 |
| GBT-Guided | 0.9738 | 0.4297 | 0.4807 |

So hardening against one detector family carries over to another. GBT-guided hardening degrades Multi-GIN *more* than the Table 5 RL-Optimized row does. The paper does not say which monitor that row used.

## Takeaways for synthesis / replication

**Reusable design patterns:**
- **Profile → activity/attribute samplers:** gives a definition of "out-of-character" anomalies relative to the profile.
- **Counterparty mixture:** locality + decayed memory + attractiveness + exploration.
- **Daily macro modifiers** that never touch micro events.
- **Atomic, constraint-checked subgraph embedding** with label propagation along fund traces.
- **Adversarial LLM editing of real illicit subgraphs** with GRPO, rewarded by the drop in a monitor score, using four edit families.

**Evaluation patterns to reuse:**
- Daily undirected invariants: GCC, component count, k-core, assortativity, transitivity.
- Clauset power-law tails with power-law vs lognormal tests.
- Chronological 60/20/20 split, edge-only features, reporting AP and F1 rather than only AUC.

**Unspecified in the paper:** the counterparty-mixture weights and formula, the source of the "real illicit subgraphs", the monitor architecture and weights `w1..w3` and `λ_mon`, K, and the base LLM. Check the repo.
