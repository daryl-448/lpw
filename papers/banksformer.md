# Banksformer: A Deep Generative Model for Synthetic Transaction Sequences

- **Authors:** Kyle Nickerson, Terrence Tricco, Antonina Kolokolova (Memorial Univ. of Newfoundland), Farzaneh Shoeleh, Charles Robertson, John Hawkin (Verafin), Ting Hu (Queen's Univ.)
- **Venue:** ECML PKDD 2022, LNAI 13718, pp. 121–136 (Springer, 2023). DOI 10.1007/978-3-031-26422-1_8
- **Code:** `github.com/BigTuna08/Banksformer_ecml_2022` (also has dataset copies and links to the baseline models). Supplementary PDF has the full hyperparameter list and UK-dataset figures/ablations.
- **Task:** Unconditional generation of synthetic, multivariate **banking transaction sequences** (irregular time intervals, date-dependent patterns).

## Core idea (TL;DR)

A transformer-decoder (TD, GPT-style autoregressive) that generates one transaction per step, with two innovations:

1. **Date mechanism.** Dates are encoded as several redundant cyclic features (DoM, DTME, DoW, month) plus a time delta. At generation time, the per-feature distributions are **multiplied together** into one distribution over candidate calendar dates, and the date is sampled from it.
2. **Conditional (sequential) field generation inside a transaction.** Fields are generated in a fixed order (tcode → date → amount), each conditioned on the earlier ones (teacher forcing at train time):
   `p(trans_k | hist) = p(tcode_k | hist) · p(date | hist, tcode_k) · p(amount | hist, tcode_k, date)`

Main result: Banksformer is the only model compared (vs TimeGAN, DoppelGANger) that learns **joint date × transaction-type patterns** (e.g., interest credited only on the last day of the month).

## Data and preprocessing

| | Accts | Total trans | Trans/acct (min / max / mean) | Tcodes | Date range |
|---|---|---|---|---|---|
| **czech** (real, PKDD'99 Discovery Challenge / Berka) | 4500 | 1.06×10⁶ | 9 / 675 / 235 | 16 | 01/01/1993 – 31/12/1998 |
| **uk** (synthetic, from a towardsai blog post) | 5000 | 10⁵ | 2 / 50 / 20 | 44 | 01/04/2017 – 25/05/2017 |

- **tcode:** all categorical type columns are concatenated into one categorical field.
- Only the transaction **date** is modeled (czech has no time of day, so uk time of day is also dropped).
- czech has day-of-month patterns (some types only at month end, some only early in the month) and no apparent day-of-week pattern. uk has day-of-week patterns (no Sunday transactions); day-of-month patterns are not considered there because uk spans under 2 months.
- Sequences: group by account, sort by date. Drop sequences shorter than `l_min` (uk 5, czech 20). Split sequences longer than `l_max` (uk 20, czech 80) into contiguous chunks, so all lengths are in `[l_min, l_max]`.
- **Per-sequence metadata:** starting balance, start date, and (czech only) customer age at sequence start.
- Continuous features are linearly scaled to variance 1. Categoricals are one-hot encoded.
- Generic preprocessing for all models (following DoppelGANger): start date as metadata, plus a per-transaction **time delta** feature.

## Method

### Date encoding

DoM, DTME (days until month end), DoW and month are each encoded as two features:

```
f1 = sin(2π i / n_i),   f2 = cos(2π i / n_i)
```

Here `i` is the ordering index and `n_i` the number of possible values (e.g. January is `i = 0`, `n_i = 12`). A time delta `Δ_t` feature is also modeled, as in TimeGAN and DoppelGANger.

### Date distribution (Eq. 1)

```
p(date) = (1/Z) · ∏_{field ∈ {DoM, DTME, DoW, month, Δ_t}} p_field(date[field])
```

`Z` is a normalizing constant. Implementation:
- A **maximum time between transactions** is set so the candidate set stays finite.
- `Δ_t` uses a **truncated Gaussian** on `[0, max time]`. The network outputs two values for it, interpreted as mean and variance.
- Every other date field gets a **softmax categorical** distribution over its options.
- `Z` is the sum of the normalized probabilities over every date from 0 to the max number of days after the current date. A date is **sampled** from the result and then converted back into the separate date features.

### Architecture (Fig. 1)

- **Input layer:** a fully connected map from `d_input` to `d_model`, plus positional encoding. The paper writes the PE as `sin(t/10000^{i/d})` for even `i` and `cos(t/10000^{i/d})` for odd `i`.
- **Decoder stack:** 4 identical decoder layers like Vaswani et al., with masked multi-head self-attention so position `i` attends only to positions `< i` (following Liu et al. 2018, "Generating Wikipedia"). Each layer has two sub-layers; the paper only names the first (masked MHA). `d_model = 128`.
- **Output layers:** produce tcode, then date, then amount, each from its own fully connected head. The **context** vector conditions all of them. When generating, each `*` step samples a value and feeds it forward. In training, the true value is fed forward (teacher forcing).
- **Date layer (Fig. 1C):** every date component (Month, DoM, DTME, DoW, TD) is predicted **independently** from the context and the true current tcode.

### Loss

A **weighted sum** of the per-field losses:
- **Continuous fields:** the network outputs normal-distribution parameters, and the loss is the negative log-probability.
- **Categorical fields:** categorical cross-entropy.

The TD background objective is `−log P(seq; Θ)`, with the autoregressive factorization `p(seq; Θ) = ∏_{i=1}^{n} p(x_i | x_1, …, x_{i−1}; Θ)`. The printed equation has an index typo (`x_n | x_1..x_{n1}` inside `∏_i`).

### Generation

1. The first sequence element is the **metadata**, turned into a vector of the same dimension as a transaction.
2. Generate autoregressively for `l` steps, appending each output to the input.
3. Within a step, generate attributes in a predefined order:
   - **Categorical:** softmax, then sample.
   - **Continuous:** sample from a normal with the predicted mean and variance.
   - **Date:** sample from Eq. 1.
4. Evaluation setup:
   - Banksformer and TimeGAN use fixed lengths (uk 20, czech 80), and their start dates (and czech ages) are sampled from the empirical real distribution.
   - DoppelGANger generates its own lengths and metadata.

## Evaluation protocol (reusable)

- **Univariate fidelity:**
  - Wasserstein-1 distance on continuous variables: **amount**, and **monthly cash flow** (the sum of credits and debits per account per month; it is not modeled directly, so it is a derived check).
  - Jensen–Shannon divergence (JSD) on categoricals: **tcode** and **DoM**.
- **Sequence:** JSD over **tcode 3-gram** distributions. Other N gave similar results. Larger N makes the empirical distributions poor estimates (curse of dimensionality). Additive smoothing did not change results, so raw empirical distributions are reported.
- **Joint:**
  - JSD of the joint **(tcode, Date\*)** distribution, where Date\* is DoM for czech and DoW for uk.
  - PCA plots following TimeGAN: flatten sequences along time, fit PCA on real data only, and project all datasets with it.
  - Conditional plots: p(DoM | tcode) and p(amount | tcode).

## Key results (Table 2; lower is better)

Bold marks the best value per dataset and column, as bolded in the paper.

| Data | Model | Amt (W1) | CF (W1) | Tcode JSD | DoM JSD | Tcode 3G JSD | (Tcode, Date\*) JSD |
|---|---|---|---|---|---|---|---|
| czech | **BF** | 2102 | **2738** | **0.004** | 0.011 | **0.042** | **0.251** |
| czech | DG | 1939 | 57800 | 0.007 | 0.090 | 0.132 | 0.660 |
| czech | TG | **1931** | 4980 | 0.075 | 0.059 | 0.337 | 0.638 |
| czech | BF-ND (no date mech.) | 3705 | 4191 | 0.009 | 0.059 | 0.059 | 0.595 |
| czech | BF-NC (no cond. gen.) | 3580 | 4775 | 0.158 | **0.006** | 0.411 | 0.542 |
| czech | TF-V (vanilla transformer) | 4726 | 4138 | 0.185 | 0.059 | 0.445 | 0.674 |
| uk | **BF** | **42.6** | **541.8** | 0.015 | **0.024** | 0.156 | **0.008** |
| uk | DG | 179.0 | 1051 | **0.011** | 0.034 | **0.135** | 0.061 |
| uk | TG | 116.0 | 1460 | 0.237 | 0.087 | 0.622 | 0.077 |

Findings:
- **Joint (tcode, date) is the main gain.** Only BF learns month-end interest credits (last day of the month, i.e. day 28–31) and a debit type that occurs only on days 5–14.
- **Amount:** BF's W1 is worse than TG and DG on czech, but on a log scale it better captures the **three modes** of the real distribution. BF is best on uk amount.
- **Conditional amount:** BF gives narrower, symmetric p(amount | tcode) distributions centred near the real mean. DG and TG give much wider, asymmetric ones.
- **PCA (czech):** BF reproduces the multiple peaks. DG is bimodal, and TG collapses to a single mode.
- **Ablations:**
  - Both mechanisms help most metrics, especially joint and amount metrics.
  - Removing conditional generation (BF-NC) badly hurts tcode and 3-gram JSD (comparable to TF-V), even though tcode is the *first* generated field. The authors' guess is that the other fields become harder to model and absorb capacity.
  - Removing the date mechanism (BF-ND) hurts DoM JSD (comparable to TF-V).

## Limitations / open directions stated by the authors

- Hybrid models: put BF's date mechanism into TimeGAN/DoppelGANger-style GANs, or add an adversarial training step to BF.
- **Privacy** of the generated data is not analysed (they name it as critical future work).
- Log-likelihood evaluation is avoided because it is known to be unreliable and is not available for GANs.

## Replication notes / gotchas

- Uses the TD-style decoder only (no encoder). Metadata is fed as token 0.
- Date candidates are bounded by a max inter-transaction gap, and the product of field probabilities is **renormalized over that window** before sampling.
- Δ_t distributions: the date mechanism uses a **truncated** Gaussian on [0, max time] at generation, while the training loss for continuous fields is a plain normal NLL. The paper does not say whether Δ_t's training loss is truncated.
- Other hyperparameters (layers beyond 4, heads, loss weights, max gap) are only in the supplementary material or code, not this paper.
