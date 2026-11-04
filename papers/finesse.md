# FINESSE: An Agent-Based Simulator and Benchmark Dataset for Multimodal Financial Event Sequences

- **Authors:** Tyler Farnan, Benjamin Eng, Adam Abate, Xirui Hou, Rizal Fathony, Nam H. Nguyen, Senthil Kumar (Capital One)
- **Venue:** arXiv:2609.11993v1 [cs.LG], 9 Sep 2026, preprint under review
- **Code:** `https://github.com/CapitalOne-Research/FINESSE`
- **Dataset:** `https://huggingface.co/datasets/capitalone/FINESSE`
- **Type:** A *simulator and benchmark* paper. There is no learned generative model: the data come from a rule-based **agent-based model (ABM)**.

## Core idea (TL;DR)

**FINESSE** (Financial Event Sequence Simulation Environment) is an ABM of **credit-card customers, merchants and a bank**. It emits **four coupled event streams** per customer:

1. Transactions
2. Loan/card payments
3. Account state
4. Bank policy interventions

The streams are coupled through each customer's **hidden, evolving latent state**. Because the generator is not fitted to real data, the authors argue it avoids the privacy problems of distribution-matching synthetic data.

**FINESSE-Bench** is a released dataset from the simulator with four tasks:
1. Transaction fraud detection
2. Missed payment prediction
3. Balance forecasting
4. Next event prediction

The paper also gives baselines from four paradigms: temporal graphs, sequence models, time-series forecasting and temporal point processes.

The authors say the tasks are for research and are not granular real-world tasks of any organization, and that FINESSE does **not** aim to support formal causal inference.

## Simulator: FINESSE-ABM

### Algorithm 1 (verbatim structure)

```
Input:  N_c customers, T days, merchants M {MCC, ID, R_f, R_i},
        customer behaviors C, credit_limits[i], RCT intervention cohorts
Output: L_txn, L_pay, L_acct, L_policy, L_hidden

for t = 1 to T:
  for c in {1..N_c}:
    Transactions:   if p_t > θ_t: sample m ~ P(M; A[c]); append {t, c, m, amt} to L_txn; update R_i[c]
    Payments:       if p_p > θ_p: compute payment from S[c]; append {t, c, pay} to L_pay
    Policy Changes: if t mod P = 0 and c ∉ control: sample policy; append {t, c, policy} to L_policy
    Hidden State:   if p_h < θ_h: update A[c] or S[c]; append {t, c, state} to L_hidden
    Account State:  update missed payments and account status; append {t, c, acct} to L_acct
return L_txn, L_pay, L_acct, L_policy, L_hidden
```

The algorithm notation is not defined in the text. The likely reading is:
- `A[c]`: the customer's Merchant Affinity policy
- `S[c]`: the customer's payment Strategy
- `R_f`, `R_i`: merchant fraud and identity-theft risk
- `R_i[c]`: the customer's accumulated identity-theft / fraud-risk score
- `p_*`: random draws
- `θ_*`: thresholds
- `P`: the policy period

The simulator advances in **discrete daily steps**.

### Merchants

They are sampled from pre-configured attribute distributions.
- **Observable** (recorded in transactions): `MCC` and `merchant_id`.
- **Unobserved** (drive the latent dynamics): `popularity` (selection frequency), `online_presence`, `id_theft_risk` and `tx_fraud_risk`.

### Customers

Each customer gets one of `P_ma` **Merchant Affinity** policies and one of `P_ps` **Payment Strategy** policies.
- **Merchant Affinity:** transaction frequency, spend amounts and online-merchant usage *per MCC*.
- **Payment Strategy:** payment frequency and the proportion of the balance paid down.

Behaviour parameters are sampled from policy-specific distributions, which gives distinct but diverse segments.

Table 2 shows the heterogeneity pattern. Each type mixes distribution families:

| Type | Frequency | Amount |
|---|---|---|
| Type 1 | F ~ Poisson(λ₁) | A ~ N(μ₁, σ₁) |
| Type 2 | F ~ N(μ₁, σ₁) | A ~ Poisson(λ₁) |
| ⋮ | ⋮ | ⋮ |
| Type N | F ~ Poisson(λ_N) | A ~ N(μ_N, σ_N) |

### Bank

- **RCT assignment:** each customer is in either a control group or an intervention group.
- **Dynamic interventions:** changes to credit limit, interest rate and minimum-payment factor.
- **Static policies:** minimum-payment rules and due dates, the same for everyone.

### Interaction graph (Fig. 2)

Customers → Merchants: transactions. Merchants → Customers: fraud risks. Customers → Bank: payments. Bank → Customers: policy adjustments.

### Latent dynamics

- **Behaviour drift:** at each step a customer may switch, at random, to a new Merchant Affinity or Payment Strategy. Bank interventions can raise the chance of a switch, depending on account state.
- **Fraud:** a customer's hidden fraud-risk score **increases with repeated transactions at high identity-theft-risk merchants**. The chance of a fraudulent transaction grows with that evolving risk and with the risk profiles of the merchants used.
- **Missed payment:** no minimum payment by the statement due date.

## FINESSE-Bench schema (Table 3)

| Stream | Attributes | Frequency |
|---|---|---|
| Transactions | status (approved/declined), amount (cont.), merchant_category, merchant_id (str), online (bool), **fraud (bool)**, timestamp (Y/M/D/H/M/S) | per transaction |
| Loan Payments | amount, timestamp (Y/M/D/H/M/S) | per payment |
| Policy Interventions | intervention_type ∈ {credit limit, interest rate, min payment factor}, timestamp (Y/M/D) | daily (if applied) |
| Account State | current_balance, utilization_ratio, interest_rate, min_payment_factor (all cont.), current_missed_payments (int) | daily |

The paper does not report dataset size (number of customers or days). Its §4 "Dataset Overview" contains only this schema table.

### Comparison with other datasets (Table 1)

FINESSE is the only dataset listed with Payment, Credit-state and Policy-change streams. It lacks clickstream, geotags and natural language, which some others have.

| Feature | Berka (KDD'99) [9] | Data Fusion 2022 [13] | MBD [16] | FinBench [35] | FINESSE |
|---|---|---|---|---|---|
| Transactions | ✓ | ✓ | ✓ | ✗ | ✓ |
| Payment | ✗ | ✗ | ✗ | ✗ | ✓ |
| Credit state | ✗ | ✗ | ✗ | ✗ | ✓ |
| Policy changes | ✗ | ✗ | ✗ | ✗ | ✓ |
| Clickstream | ✗ | ✓ | ✓ | ✗ | ✗ |
| Geotags | ✗ | ✗ | ✓ | ✗ | ✗ |
| Natural language | ✗ | ✗ | ✓ | ✓ | ✗ |
| Account info | ✓ | ✗ | ✓ | ✓ | ✓ |

The paper labels column [35] as **FinBench**, but reference [35] in its bibliography is *FinPT* (Yin et al. 2023). It also calls the Berka dataset "from the 1999 KDD Cup", although it is the PKDD'99 Discovery Challenge.

## Tasks, protocols and baselines

All runs use **one A10 GPU**, with training times of 12–24 h.

### 1. Transaction fraud detection: temporal graph edge classification

- The graph is a bipartite customer–merchant **CTDG**, with edges as timestamped transactions. Each edge is classified as fraud or not.
- Baselines are **TGN** and **DyRep**, using the TGB implementations with modified negative sampling and memory-module updates.
- The split is chronological by edge time, 70/15/15. Metrics are ROC-AUC and PR-AUC.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| DyRep | 0.7561 | **0.0435** |
| TGN | **0.7607** | 0.0433 |

(Fraud is very rare, so PR-AUC is about 0.04.)

### 2. Missed payment prediction: sequence-level

- Predict the **number of missed payments** in the next 3, 6, 9 or 12 months from the full transaction and account history.
- It is framed as a regression/count task but evaluated as classification with **multiclass ROC-AUC**.
- Baselines are MLP, GRU, Transformer, MAMBA and CoLES, all at default settings.
- The values in the table were read off the Fig. 3 bar labels:

| Horizon | CoLES | GRU | Mamba | MLP | Transformer |
|---|---|---|---|---|---|
| 3 mo | 0.597 | 0.641 | 0.649 | 0.601 | **0.653** |
| 6 mo | 0.548 | 0.538 | 0.546 | 0.557 | **0.612** |
| 9 mo | 0.505 | **0.598** | 0.542 | 0.507 | 0.582 |
| 12 mo | 0.523 | **0.537** | 0.519 | 0.517 | 0.516 |

Performance mostly falls as the horizon grows. At 12 months all models are about the same.

### 3. Balance forecasting: time series

- Input is the previous **60 days of account balance**. The output is the **next 60 days**.
- Baselines are DLinear, PatchTST, Transformer and MambaSimple, all at default settings.
- Metrics are MAE, MSE, RMSE, MAPE and MSPE. **PatchTST wins 3 of 5** (MAE, MSE, RMSE). The **Transformer wins MAPE and MSPE**.
- The authors say MambaSimple "consistently ranks lowest", blaming the short 60-day input for wasting its O(L) advantage. Their own Fig. 4 shows it worst on 4 of 5 metrics; on MAE, the Transformer is marginally worse.
- Approximate values read from the Fig. 4 bars:

| Model | MAE | MSE | RMSE | MAPE | MSPE |
|---|---|---|---|---|---|
| DLinear | ≈0.83 | ≈1.17 | ≈1.08 | ≈2.5 | ≈57k |
| PatchTST | ≈0.72 | ≈1.00 | ≈1.00 | ≈3.35 | ≈100k |
| Transformer | ≈1.08 | ≈1.87 | ≈1.37 | ≈1.3 | ≈1k |
| MambaSimple | ≈1.07 | ≈2.45 | ≈1.57 | ≈6.1 | ≈337k |

### 4. Next event prediction: temporal point process

- Uses the EasyTPP package.
- Two tasks: next **payment** and next **transaction category**. Each predicts the event type and time.
- Metrics are type error rate and time RMSE.

| Model | Payment type err | Payment RMSE | Txn type err | Txn RMSE |
|---|---|---|---|---|
| RMTPP | 0.5245 | 13.7997 | 0.5904 | 0.6606 |
| NHP | 0.5241 | 13.5598 | 0.5877 | **0.6520** |
| THP | 0.5080 | **12.5381** | 0.5863 | 0.6576 |
| IFTPP | **0.4922** | 12.8061 | **0.5785** | 1.9132 |

Bold = actual column minimum. The paper itself bolds IFTPP's 12.8061 for payment RMSE, although THP's 12.5381 is lower.

## Related work positioning

- **Prior financial ABMs:**
  - LeBaron's agent-based computational finance (macro markets)
  - Firm–bank credit ABMs
  - Liquidity risk in interbank and payment systems
  - ABIDES (RL agents)
  - A credit-card promotions ABM, which uses only account-level demographics
  - AML simulators (e.g. Altman et al., IBM AMLworld), which encode known criminal priors as rules
- **Public fraud datasets:** IEEE-CIS, PaySim, Sparkov, the ULB credit-card PCA dataset and the Amazon fraud benchmark. These are single-modality and transaction-only.
- **Loan default datasets:** Vehicle Loan and Amex default. These are account-level or statement aggregates.
- **Multimodal datasets:** Berka, Data Fusion 2022 and MBD. None has policy-driven dynamics or event-level payment logs.

## Takeaways for synthesis / replication

- The ABM design pattern is to **latent per-agent policy (A, S) + latent risk score → several observed streams**, with an exogenous RCT treatment stream. The hidden-state log `L_hidden` is emitted as well, giving ground truth for latent-state inference and drift detection.
- The fraud label comes from a **hidden mechanism**: identity-theft exposure accumulated at risky merchants. Models must infer cumulative exposure from merchant history; it is not a per-transaction rule.
- The baselines are weak and run at default settings. They are reference points, not tuned state of the art.
- Missing from the paper: dataset size, the concrete parameter values, P_ma and P_ps, θ thresholds, the policy period P and the fraud-probability functional form. Get these from the repo or config.
