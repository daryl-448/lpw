# BankGAN: A Generative Model for Synthetic Financial Transactions

- **Authors:** Hamideh Mehri, John Hawkin, Kyle Nickerson, Alexander Bihlo, Farzaneh Shoeleh (Memorial Univ. of Newfoundland, and Verafin / Nasdaq)
- **Venue:** 37th Canadian Conference on Artificial Intelligence (Canadian AI 2024). CC BY 4.0.
- **Code:** `http://github.com/AI4Banking/BankGAN-CANAI2024`. TensorFlow 2, run on a Quadro RTX 8000 with 48 GB RAM.
- **Task:** Generate synthetic **bank transaction sequences** with irregular, non-uniform date patterns, plus a DP-SGD privacy study.
- **Lineage:** follows on from [Banksformer](banksformer.md), with an overlapping Verafin and MUN team.

## Core idea (TL;DR)

BankGAN is a **two-stage hybrid**:

1. A **decoder-only transformer** generates the per-account **sequence of transaction codes (Tcodes)**. Customer metadata such as age is the first token.
2. A **CTGAN-style conditional tabular GAN** generates each transaction's other fields (amount, date parts, time delta) row by row. The conditional vector is the Tcode at that sequence position (the **sequential conditional vector**).

Dates are produced with Banksformer's **date mechanism**, adapted with new "feature to probability" decoders for cyclic encodings. The GAN itself has no recurrence: sequence structure comes only from the Tcode transformer and from chaining dates.

## Data

- **Czech (Berka/PKDD'99) only:** about 4,500 accounts and more than 1M transactions.
- Fields: amount, timestamp, customer age, and 3 type columns. The 3 type columns are merged into one **Tcode**, as in Banksformer.
- Derived features:
  - Time delta (Td) between successive transactions.
  - DoW, DoM, DTME (days till month end) and month, taken from the timestamp.

## Method

### GAN backbone (identical to CTGAN)

- **Generator:**
  - Two residual-concat layers: FC → BatchNorm → ReLU, with each layer's input concatenated to its output.
  - A final FC layer maps to the dimension of the encoded real row.
- **Discriminator:**
  - Two layers of Linear → LeakyReLU → Dropout.
  - A final dense layer outputs one scalar.
- **Loss:** WGAN with gradient penalty (WGAN-GP).
- **Generator input:** noise plus a conditional vector, where the condition is the Tcode (the only categorical attribute).
  - **Training-by-sampling** (from CTGAN) samples Tcodes so every category appears in training based on its training-data frequency. The exact weights are unspecified; CTGAN itself uses log-frequency.
  - An **extra generator loss term** (from CTGAN) penalizes mismatch between the conditioned Tcode and the generated Tcode. This is what lets the GAN follow an externally supplied Tcode sequence.
- **Discriminator input:** each real row is encoded variable by variable and the parts concatenated. Generator outputs pass through type-specific activations to match that encoding.

### Three variable types: date, continuous, categorical

**Date encodings compared.** `n` is the number of units in the period, and `t ∈ {0, …, n−1}`.
- **One-hot:** an n-dim binary vector.
- **Clock (Eq. 4.1):**
  ```
  C(t) = [ sin(2πt/n), cos(2πt/n) ]
  ```
- **RBF (Eq. 4.2):** a two-dimensional repeating basis, following sklego `RepeatingBasisFunction`:
  ```
  R(t) = [ exp(−( min(|t/(n−1)|,       1 − |t/(n−1)|       ) )²),
           exp(−( min(|t/(n−1) − 0.5|, 1 − |t/(n−1) − 0.5|) )²) ]
  ```

**Continuous: mode-specific normalization (Eq. 4.3), from CTGAN.**
```
G(τ) = [ (τ − μ_i) / (4σ_i),  one-hot(i) ]   where i = argmax_j ρ_j(τ)
```
Here `ρ_j(τ)` is the density of τ under GMM mode j, with mean μ_j and standard deviation σ_j.

**Categorical:** one-hot.

### Generation pipeline (Fig. 1)

1. The trained transformer generates a Tcode sequence for an account, starting from the metadata token.
2. Each Tcode in turn becomes the conditional vector, concatenated with N(0, I) noise, and goes into the generator. The output is the vector `V`.
3. Each segment of `V` is decoded:
   - **Continuous:** `i = argmax(Softmax(V_continuous[1:]))`, then `value = tanh(V_continuous[0]) × 4 × σ_i + μ_i`.
   - **Tcode:** `i = argmax(Softmax(V_Tcode))`, then map back to the category.
   - **Date, one-hot encoded:** `p_date = Softmax(V_date)`.
   - **Date, cyclic encoded (clock or RBF), Eqs. 4.4–4.5:**
     ```
     p_date = [ (1/d(t)) / Σ_{u=0}^{n−1} (1/d(u)) ]_{t=0}^{n−1}
     d(t)   = Σ_{i=1}^{2} ( E(t)_i − pt_i + ε )²,   pt = tanh(V_date)
     ```
     `E = C` for clock or `E = R` for RBF. `d(t)` is the squared distance from the generated point to each unit's encoding, so the probability is inverse-distance weighted. `ε` is a small stability constant.
   - **Combine the date parts with the Banksformer mechanism (Eq. 4.6):**
     ```
     p(transaction date) ∝ p_month(m) × p_DoM(d) × p_DoW(w) × p_DTME(e) × p_td(td)
     ```
     - `p_td` is a **normal with mean equal to the decoded V_td** (via the continuous decoder) and a **fixed standard deviation**.
     - The probability is evaluated for "max days" starting from the last transaction date.
     - The **argmax date is chosen**: it is deterministic, unlike Banksformer, which samples.

**Tcode transformer:**
- Decoder-only: FC embedding, positional encoding, a decoder stack with attention, and a final dense layer.
- Invariant metadata (e.g. age) is the first token.
- Trained on next-Tcode prediction, then generates iteratively.

## Baseline built in the paper: Stacked LSTM Autoencoder

- **Encoder and decoder:** 2 LSTM layers each.
- The encoder's last-layer final hidden state is the context, repeated over the timesteps (RepeatVector).
- Each decoder layer is initialized with the final (h, c) states of the matching encoder layer.
- **Output dense heads:** predict features sequentially, each conditioned on the earlier predicted features (Banksformer-style conditional generation). Teacher forcing in training, own predictions at generation.
- **Encoding:**
  - Date and categorical features are encoded as above.
  - Continuous features are only scaled by their standard deviation.
- **Outputs and losses:**
  - Date and categorical heads: softmax with categorical cross-entropy.
  - Continuous heads: ReLU dense layers output mean and standard deviation, trained with normal-likelihood loss.
- The date-head outputs and the predicted (μ, σ) of td feed Eq. 4.6.

**Trainable parameters:** Banksformer about 450k, StackedLSTM about 490k, BankGAN about 320k.

## Evaluation (same metrics as Banksformer)

Lower is better for every metric:
- **Wasserstein distance:** Amount, monthly Cash Flow (credits minus debits per customer per month) and Time delta.
- **JSD:** Tcode, Day (DoM), Tcode 3-gram and joint Tcode–Day.

The models are also scored on **recurring vs non-recurring** transactions, which are about 58% and 42% of the data. A transaction is recurring if it happens regularly each month around the same date.

**"None" date encoding:** no date mechanism. Td is an ordinary continuous variable, and dates come from a random start date plus cumulative time deltas.

### Table 1: date encoding × model

| Model | Date enc. | Amount WD | Cash Flow WD | Time delta WD | Tcode JSD | Day JSD | Tcode-3g JSD | Tcode-Day JSD |
|---|---|---|---|---|---|---|---|---|
| BankGAN | onehot | 2627 | 8402 | 2.7 | 0.09 | 0.066 | 0.25 | 0.26 |
| BankGAN | clock | 2057 | 9542 | 0.73 | 0.09 | 0.06 | 0.25 | 0.26 |
| BankGAN | rbf | 2173 | 6332 | 1.6 | 0.09 | 0.09 | 0.25 | 0.32 |
| BankGAN | None | 1884 | 8472 | 1.01 | 0.09 | 0.11 | 0.25 | 0.37 |
| StackedLSTM | onehot | 3769 | 2500 | 2.86 | 0.089 | 0.13 | 0.29 | 0.25 |
| StackedLSTM | clock | 4277 | 3674 | 1.93 | 0.064 | 0.13 | 0.28 | 0.26 |
| StackedLSTM | rbf | 4039 | 2392 | 1.96 | 0.12 | 0.15 | 0.35 | 0.30 |
| StackedLSTM | None | 4376 | 2285 | 2.5 | 0.12 | 0.14 | 0.34 | 0.38 |
| Banksformer | onehot | 5916 | 7162 | 6.1 | 0.17 | 0.028 | 0.42 | 0.21 |
| Banksformer | clock | 4236 | 4338 | 0.15 | 0.016 | 0.010 | 0.058 | 0.035 |
| Banksformer | rbf | 3383 | 3394 | 0.98 | 0.021 | 0.026 | 0.079 | 0.064 |
| Banksformer | None | 4931 | 5015 | 1.32 | 0.074 | 0.11 | 0.21 | 0.32 |

### Table 2: DP-SGD (δ = 1e-5, ε from TensorFlow Privacy, clock-encoding variants)

| Model | DP | Amount WD | Cash Flow WD | Time delta WD | Tcode JSD | Day JSD | Tcode-3g JSD | Tcode-Day JSD |
|---|---|---|---|---|---|---|---|---|
| BankGAN | NO | 2057 | 9542 | 0.73 | 0.09 | 0.06 | 0.25 | 0.26 |
| BankGAN | YES (ε=0.60) | 2033 | 6703 | 0.8 | 0.09 | 0.06 | 0.25 | 0.25 |
| Banksformer | NO | 4236 | 4338 | 0.15 | 0.016 | 0.010 | 0.058 | 0.035 |
| Banksformer | YES (ε=1.16) | 4918 | 5241 | 0.6 | 0.036 | 0.016 | 0.11 | 0.058 |
| StackedLSTM | NO | 4277 | 3674 | 1.96 | 0.064 | 0.15 | 0.28 | 0.26 |
| StackedLSTM | YES (ε=1.29) | 4174 | 4226 | 10.25 | 0.035 | 0.15 | 0.24 | 0.21 |

## Findings as claimed

- **Clock encoding wins overall.**
  - For Banksformer, clock is best on every metric except Amount and Cash Flow.
  - For BankGAN, it is best except Cash Flow.
  - Cyclic encodings (clock, RBF) beat one-hot on the Time delta metric across models.
  - Clock beats RBF except on Cash Flow and Amount, so clock is used for the further analysis.
- **Date mechanism:**
  - Removing it makes Banksformer worse on every metric.
  - For StackedLSTM it is generally worse without it, except on Cash Flow.
  - For BankGAN, removing it improves Amount and Cash Flow but is comparable or worse elsewhere.
- **BankGAN vs others:**
  - Best Amount WD, which the authors credit to GMM mode-specific normalization against the single-Gaussian assumption of the other two models.
  - Beats StackedLSTM on nearly all metrics.
  - **Does not beat Banksformer on most metrics.** Banksformer with clock encoding is clearly best on Tcode, Day, 3-gram, Tcode-Day and Time delta.
- **Recurring transactions (Fig. 3):** BankGAN best matches log-amount and monthly cash flow, but Banksformer best matches the Tcode distribution.
- **Non-recurring transactions (Fig. 4):** BankGAN fails to generate the single **credit** category, so its cash flow is only negative. Its log-amount peak is right, but it adds a pronounced tail.
- **DP-SGD:**
  - BankGAN at ε = 0.60 is not degraded and even slightly improves some metrics. The authors read this as DP acting as a regularizer (citing PATE).
  - Banksformer degrades under DP.
  - StackedLSTM roughly matches or improves on most metrics, but its Time delta WD gets much worse (1.96 → 10.25).

## Inconsistencies in the paper (verify against code before relying on them)

- **Mismatched StackedLSTM rows.** Table 2's StackedLSTM "NO" row (Time delta 1.96, Day 0.15) does not match Table 1's StackedLSTM clock row (1.93, 0.13). The values 1.96 and 0.15 match Table 1's **rbf** row.
- **Overstated DP claim for BankGAN.** The text says BankGAN under DP "improves synthetic data quality in all aspects", but its Time delta WD worsens slightly (0.73 → 0.8).
- **Wrong section reference.** Section 4.3 says the transformer is "described in section 4.2". It is actually described later in 4.3.
- **Confusing "Banksformer study" wording.** The paper says "in the Banksformer study [12], DP degrades data quality". The original Banksformer paper has no DP experiments, so this must mean Banksformer as re-run in this paper.

## Ideas worth reusing / open directions (from authors)

- **Decouple the discrete event-type sequence (an autoregressive model) from per-event attribute generation** (a conditional tabular GAN).
- The inverse-distance probability decoder for cyclic encodings turns any generated point on the circle into a categorical distribution, so Banksformer's product-of-marginals date mechanism can be used.
- Future work stated by the authors:
  - Improve the Tcode generator.
  - **Split into two models**, one for recurring and one for non-recurring transactions.
