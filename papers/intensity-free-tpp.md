# Intensity-Free Learning of Temporal Point Processes

- **Authors:** Oleksandr Shchur\*, Marin Biloš\*, Stephan Günnemann (Technical University of Munich). \* equal contribution.
- **Venue:** ICLR 2020.
- **Code and datasets:** `https://github.com/shchur/ifl-tpp`
- **Common names:** "IFTPP" or "IFL-TPP"; the main model is **LogNormMix**. EventFlow uses it as a baseline; see [eventflow.md](eventflow.md).
- **Task:** Autoregressive next-event-time modelling: learn `p*(τ_i) = p(τ_i | H_{t_i})` for inter-event times `τ_i = t_i − t_{i−1}`. Optional marks and side information.

## Core idea (TL;DR)

**Do not parametrise the conditional intensity `λ*(t)`. Parametrise the conditional density of the next inter-event time `p*(τ)` directly**, with neural density estimators. An RNN encodes the history, and an affine layer maps the context to the distribution's parameters.

Two families:
1. **Normalizing flows (DSFlow, SOSFlow):** flexible, with a closed-form likelihood. But there is no closed-form mean, and sampling needs iterative root finding.
2. **Log-normal mixture (LogNormMix), the recommended model:** flexible (universal approximator), with a closed-form likelihood, mean and moments, and **closed-form, reparametrisable sampling**.

LogNormMix matches or beats the flows and the intensity-based baselines on NLL. Because sampling is differentiable, it also enables training with missing data (imputation by sampling). Separately, the context vector supports learned per-sequence embeddings, which are trained by plain likelihood.

### Table 1: model properties (history encoded with an RNN)

| | Exponential intensity | Neural Hawkes | Fully NN | Normalizing Flows | Mixture Distribution |
|---|---|---|---|---|---|
| Closed-form likelihood | ✓ | ✗ | ✓ | ✓ | ✓ |
| Flexible | ✗ | ✓ | ✓ | ✓ | ✓ |
| Closed-form E[τ] | ✗ | ✗ | ✗ | ✗ | ✓ |
| Closed-form sampling | ✓ | ✗ | ✗ | ✗ | ✓ |

## Background (§2)

- A TPP realisation is a sequence of strictly increasing arrival times `T = {t_1, …, t_N}`. Equivalently, it is a sequence of strictly positive inter-event times `τ_i = t_i − t_{i−1} ∈ R+`.
- History: `H_t = {t_j ∈ T : t_j < t}` (strict).
- Conditional intensity: `λ*(t) := λ(t | H_t)`.
- From intensity to density:

```
p*(τ_i) := p(τ_i | H_{t_i}) = λ*(t_{i−1} + τ_i) exp( −∫_0^{τ_i} λ*(t_{i−1} + s) ds )
```

- MLE for an intensity model:

```
θ* = argmax_θ Σ_i log p*_θ(τ_i) = argmax_θ [ Σ_i log λ*_θ(t_i) − ∫_0^{t_N} λ*_θ(s) ds ]
```

The integral runs to the last event `t_N`. **There is no survival term for an observation window past `t_N`.** Compare EventFlow Eq. 3.

- Cumulative intensity: `Λ*(τ_i) := ∫_0^{τ_i} λ*(t_{i−1} + s) ds`.

The trade-off the paper targets:
- A "simple" intensity has a closed-form `Λ*` but limited expressiveness.
- A rich intensity (Neural Hawkes) needs Monte Carlo integration.
- FullyNN (Omi et al. 2019) models `Λ*` with a monotone network. But its density is not valid (see below), sampling is expensive, and it has no closed-form mean.

## Models (§3)

§3.1–3.2 first treat `τ_i` as independent of the history (`p*(τ_i) = p(τ_i)`). §3.3 then adds the conditioning.

### 3.1 Normalizing flows on τ

- Change of variables: `p(x) = q(g^{−1}(x)) |∂/∂x g^{−1}(x)|`.
- Sampling uses the forward map `x = (g_M ∘ ⋯ ∘ g_1)(z)`.
- The density uses the inverse `z = (g_1^{−1} ∘ ⋯ ∘ g_M^{−1})(x)`.

The inverse chain used for maximum likelihood:
1. `z_M = g_M^{−1}(τ) = log τ`, which maps `R+ → R`.
2. Stacked layers `g_m^{−1} = f_{θ_m}`.
3. Finally `g_1^{−1} = σ`, which maps `z_2` to `z_1 ∈ (0, 1)`.
4. The base is `z_1 ~ q_1(z_1) = Uniform(0, 1)` (App. D.4/D.5).

Batch-normalisation flow layers go between consecutive layers. Dimension-splitting flows (RealNVP) do not apply to 1-D data.

Layer choices, with `σ(x) = 1/(1 + e^{−x})`:

```
f^DSF(x) = σ^{−1}( Σ_{k=1}^K w_k σ( (x − µ_k) / s_k ) )                                    (1)
f^SOS(x) = a_0 + Σ_{k=1}^K Σ_{p=0}^R Σ_{q=0}^R ( a_{p,k} a_{q,k} / (p + q + 1) ) x^{p+q+1}
```

- DSF: Krueger et al. 2018, with `Σ_k w_k = 1, w_k ≥ 0, s_k > 0`.
- SOS: Jaini et al. 2019. The coefficients `a ∈ R^{(R+1)×K}` are unconstrained.
- Both are invertible only numerically. So **sampling needs root finding, and `E_p[τ]` has no closed form.**

### 3.2 Log-normal mixture (LogNormMix)

```
p(τ | w, µ, s) = Σ_{k=1}^K w_k · 1/(τ s_k √(2π)) · exp( −(log τ − µ_k)² / (2 s_k²) )      (2)
```

- **Mean:** `E_p[τ] = Σ_k w_k exp(µ_k + s_k²/2)`. Higher moments come from the per-component moments.
- **CDF (App. A):** `F(τ) = Σ_{k=1}^K w_k Φ( (log τ − µ_k) / s_k )`.
- **Sampling, closed form:**

```
z ~ Categorical(w)     (one-hot, size K)
ε ~ Normal(0, 1)
τ = exp( s^T z · ε + µ^T z )
```

The sample is differentiable in `µ` and `s`. For gradients with respect to `w`, use the **Straight-Through Gumbel** estimator (App. D.6):
- relaxed sample `z* = softmax((log w + o)/T)`, with `o_i` i.i.d. Gumbel(0, 1) and `T` the temperature;
- forward pass: `z = onehot(argmax_k z*_k)`;
- backward pass: gradients flow through `z*`.

The estimator is slightly biased, which the authors found negligible in practice.

**As a flow (App. D.2), the trick that matters for implementation:**

```
z_1 ~ GaussianMixture(w, µ, s)
z_2 = a z_1 + b
τ   = exp(z_2)
```

The fixed affine layer uses dataset-level statistics (not batch statistics): `b = (1/N) Σ_i log τ_i`, and `a` = the std of `log τ_i` (see Gotchas). This gives a better initialisation and faster convergence.

Why log-normal: Theorem 1 holds for any base density. The authors note that log-logistic does not always have finite moments, and gamma does not allow simple reparametrised sampling.

### 3.3 Conditioning

- **History:** an RNN over the past events gives `h_i ∈ R^H`. The RNN input is the **log-transformed, centred** `τ`, plus a mark embedding if there are marks.
- **Metadata `y_i`:** side information that conditions only `p*(τ_i | y_i)`. It is not part of the objective, and it may be shared across a whole sequence. Example: weekday or evening indicators.
- **Sequence embedding `e_j`:** a learnable vector per training sequence `j`.
- **Context:** `c_i = [h_i || y_i || e_j]`. The parameters are affine in `c_i`, as in a mixture density network:

```
w_i = softmax(V_w c_i + b_w)      s_i = exp(V_s c_i + b_s)      µ_i = V_µ c_i + b_µ        (3)
```

The flow parameters come from the context the same way, e.g. SOS: `a = V_a c + b_a`.

### 3.4 Theory and discussion

**Theorem 1** (DasGupta 2008, Thm 33.2). Let `p(x)` be a continuous density on R, and `q(x)` any continuous density on R. For any `ε > 0` and compact `S ⊂ R`, there exist:
- a number of components `K ∈ N`;
- weights `w ∈ Δ^{K−1}`;
- locations `µ ∈ R^K`;
- scales `s ∈ R^K_+`;

such that `p̂(x) = Σ_{k=1}^K w_k (1/s_k) q((x − µ_k)/s_k)` satisfies `sup_{x∈S} |p(x) − p̂(x)| < ε`.

So mixtures are universal approximators, like DSFlow and SOSFlow. The conditional version also assumes the RNN encodes all relevant history (RNN universal approximation).

**From density back to intensity (App. A):**

```
λ*(t) = p*(t − t_{i−1}) / (1 − F*(t − t_{i−1}))
Λ*(τ_i) := ∫_0^{τ_i} λ*(t_{i−1} + s) ds = −log(1 − F*(τ_i))
```

For flows, `F(τ) = Q(g^{−1}(τ))`, where `Q` is the CDF of `z`. Here `τ = g(z)`.

**Superposition argument.** Merging independent processes adds intensities, `λ* = λ*_1 + λ*_2`. Equivalently, in CDF form:

```
F*(τ) = F*_1(τ) + F*_2(τ) − F*_1(τ) F*_2(τ)
```

So modelling `p*` loses none of the "reusability" of intensities.

**Classic intensities as distributions (App. B):**
- **Constant intensity**, `λ*(t_i) = exp(v^T h_i + b)`, gives an **exponential distribution** with `c = exp(v^T h_i + b)`.
- **RMTPP exponential intensity**, `λ*(t_i) = exp(w(t_i − t_{i−1}) + v^T h_i + b)` with `w ∈ R+`, gives a **Gompertz distribution**. Let `d = v^T h_i + b`:

```
p(τ | w, d) = exp( wτ + d − (1/w) exp(wτ + d) + (1/w) exp(d) )
```

- This is Gompertz with `α = exp(d)` and `β = w`:
  - `p(τ | α, β) = α exp( βτ − (α/β) exp(βτ) + α/β )`, for `α, β > 0`.
  - Mean: `E[τ] = (1/β) exp(α/β) Ei(−α/β)`, with `Ei(z) = ∫_{−z}^∞ exp(−v)/v dv`.
  - It is unimodal and light-tailed for any parameters. This is why RMTPP fits heavy-tailed real data badly.

**FullyNN flaws (App. C).** The model (Omi et al. 2019) uses non-negative weights:

```
Λ*(τ) := f(τ) = softplus( W^(3) tanh( W^(2) tanh( W^(1) τ + b̃^(1) ) + b^(2) ) + b^(3) )      (4)
b̃^(1) = V h + b^(0)
```

Shapes: `W^(1) ∈ R_+^{D×1}`, `W^(2) ∈ R_+^{D×D}`, `W^(3) ∈ R_+^{1×D}`. Then:
1. `lim_{τ→∞} Λ*(τ) < softplus( Σ_{d=1}^D |w_d^(3)| + b^(3) ) < ∞`, so the **density does not integrate to 1**.
2. `Λ*(0) = f(0) = softplus(W^(3) tanh(W^(2) tanh(b̃^(1)) + b^(2)) + b^(3)) > 0`, so `F*(0) = 1 − exp(−Λ*(0)) > 0`. This puts **probability mass on τ ≤ 0**.
3. Sampling uses the inverse method: draw `z̃ ~ Exp(1)`, then solve `f(τ) = z̃` (e.g. by bisection).

## Experiments (§5, App. E–F)

### Data

**Real datasets (Table 2):**

| Dataset | #Seq | #Events |
|---|---|---|
| LastFM | 929 | 1268385 |
| Reddit | 10000 | 672350 |
| Stack Overflow | 6633 | 480414 |
| MOOC | 7047 | 396633 |
| Wikipedia | 1000 | 157471 |
| Yelp | 300 | 215146 |

**Marks:**
- Reddit: 984 subreddits.
- MOOC: 97 interaction types.
- LastFM: artists as event types.

**Synthetic datasets** (after Omi et al. 2019): 64 sequences × 1024 events each.
- **Poisson:** `λ*(t) = 1`.
- **Renewal:** log-normal `p(τ)` with `µ = 1.0`, `σ = 6.0`.
- **Self-correcting:** `λ*(t) = exp(t − Σ_{t_i<t} 1)`.
- **Hawkes:** `λ*(t) = µ + Σ_{t_i<t} Σ_{j=1}^M α_j β_j exp(−β_j (t − t_i))`.
  - Hawkes1: `M = 1`, `µ = 0.02`, `α_1 = 0.8`, `β_1 = 1.0`.
  - Hawkes2: `M = 2`, `µ = 0.2`, `α_1 = 0.4`, `β_1 = 1.0`, `α_2 = 0.4`, `β_2 = 20`.

### Protocol (App. D, F.1)

- **Splits:** 60/20/20 by sequence, 10 random splits.
- **Training data:** training sequences are chunked to length ≤ 128.
- **Optimiser:** Adam, lr 10⁻³, batches of 64 sequences.
- **Stopping:** up to 2000 epochs; early stopping with patience 100, reverting to the best validation parameters.
- **Tuning:**
  - L2 strength `C ∈ {0, 10⁻⁵, 10⁻³}` for every model.
  - Layers: `M ∈ {1, 2, 3}` for SOSFlow, `M ∈ {1, 2, 3, 5, 10}` for DSFlow.
  - **`K = 64`** for LogNormMix, DSFlow and FullyNN, which gives roughly matched parameter counts.
  - SOSFlow uses `K = 4`, `R = 3` (degree 7 per layer). Larger `R` was unstable.
- **Input scaling:** the baselines use `τ̃ = aτ` with `a = (1/N) Σ_i τ_i` (see Gotchas). LogNormMix uses the D.2 affine layer.
- **Loss:** `L_time(θ) = −(1/N) Σ_{i=1}^N log p*_θ(τ_i)`.
- **Marks:** `L_total(θ) = −(1/N) Σ_{i=1}^N [log p*_θ(τ_i) + log p*_θ(m_i)]`, with:

```
π_i = softmax( V_π^(2) tanh( V_π^(1) h_i + b_π^(1) ) + b_π^(2) )
```

  Marks are assumed **conditionally independent of the time given the history**.
- **Reporting:** NLL differences relative to LogNormMix, computed per split and then aggregated. Rescaling the data by `a` shifts every log-likelihood by a constant (`log q(x) = log p(τ) − log a` for `x = aτ`; the paper prints `+ log a`, see Gotchas), so absolute NLLs are not comparable across scalings.

### Results: time NLL on real data (Table 4; lower is better)

| Model | Reddit | LastFM | MOOC | Stack Overflow | Wikipedia | Yelp |
|---|---|---|---|---|---|---|
| LogNormMix | 10.19 ± 0.078 | -2.88 ± 0.147 | 6.03 ± 0.092 | 14.44 ± 0.013 | 8.39 ± 0.079 | 13.02 ± 0.070 |
| DSFlow | 10.20 ± 0.074 | -2.88 ± 0.148 | 6.03 ± 0.090 | 14.44 ± 0.019 | 8.40 ± 0.090 | 13.09 ± 0.065 |
| SOSFlow | 10.27 ± 0.106 | -2.56 ± 0.133 | 6.27 ± 0.058 | 14.47 ± 0.049 | 8.44 ± 0.120 | 13.21 ± 0.068 |
| FullyNN | 10.23 ± 0.072 | -2.84 ± 0.179 | 6.83 ± 0.152 | 14.45 ± 0.014 | 8.40 ± 0.086 | 13.04 ± 0.073 |
| LogNormal | 10.38 ± 0.077 | -2.60 ± 0.140 | 6.53 ± 0.016 | 14.62 ± 0.013 | 8.52 ± 0.078 | 13.44 ± 0.074 |
| RMTPP | 10.88 ± 0.293 | -1.30 ± 0.164 | 10.65 ± 0.023 | 14.51 ± 0.014 | 10.02 ± 0.085 | 13.36 ± 0.056 |
| Exponential | 11.07 ± 0.070 | -1.28 ± 0.152 | 10.64 ± 0.026 | 18.48 ± 3.257 | 10.03 ± 0.083 | 13.78 ± 1.250 |

- The paper says the universal approximators (LogNormMix, DSFlow, SOSFlow, FullyNN) always beat the unimodal models. Table 4 has an exception: on MOOC, FullyNN (6.83) is worse than LogNormal (6.53).
- A single LogNormal beats RMTPP/Gompertz, because real inter-event times are heavy-tailed.
- **LogNormMix and DSFlow are consistently best.**

### Results: synthetic data (Table 5, test NLL)

| Model | Poisson | Renewal | Self-correcting | Hawkes1 | Hawkes2 |
|---|---|---|---|---|---|
| True model | 0.999 | 0.254 | 0.757 | 0.453 | -0.043 |
| LogNormMix | 0.99 ± 0.006 | 0.25 ± 0.010 | 0.78 ± 0.003 | 0.52 ± 0.047 | 0.02 ± 0.049 |
| DSFlow | 0.99 ± 0.006 | 0.25 ± 0.010 | 0.78 ± 0.002 | 0.52 ± 0.047 | 0.02 ± 0.050 |
| SOSFlow | 1.00 ± 0.013 | 0.25 ± 0.010 | 0.88 ± 0.011 | 0.59 ± 0.056 | 0.06 ± 0.046 |
| FullyNN | 1.00 ± 0.006 | 0.28 ± 0.013 | 0.78 ± 0.004 | 0.55 ± 0.047 | 0.06 ± 0.047 |
| LogNormal | 1.08 ± 0.008 | 0.25 ± 0.010 | 1.03 ± 0.006 | 0.55 ± 0.047 | 0.06 ± 0.049 |
| RMTPP | 0.99 ± 0.006 | 1.01 ± 0.023 | 0.78 ± 0.003 | 0.74 ± 0.057 | 0.69 ± 0.058 |
| Exponential | 0.99 ± 0.006 | 1.00 ± 0.023 | 0.94 ± 0.002 | 0.74 ± 0.055 | 0.69 ± 0.054 |

### Sensitivity to K (Table 3, LogNormMix test NLL)

| K | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Reddit | 10.239 | 10.208 | 10.189 | 10.185 | 10.191 | 10.192 |
| LastFM | -2.828 | -2.879 | -2.881 | -2.880 | -2.877 | -2.860 |
| MOOC | 6.246 | 6.053 | 6.055 | 6.055 | 6.050 | 5.660 |
| Stack Overflow | 14.461 | 14.438 | 14.435 | 14.435 | 14.436 | 14.428 |
| Wikipedia | 8.399 | 8.389 | 8.385 | 8.384 | 8.384 | 8.386 |
| Yelp | 13.169 | 13.103 | 13.058 | 13.045 | 13.032 | 13.024 |
| Poisson | 1.006 | 0.992 | 0.991 | 0.991 | 0.990 | 0.991 |
| Renewal | 0.256 | 0.254 | 0.254 | 0.254 | 0.256 | 0.259 |
| Self-correcting | 0.831 | 0.785 | 0.782 | 0.783 | 0.784 | 0.784 |
| Hawkes1 | 0.530 | 0.523 | 0.532 | 0.532 | 0.523 | 0.523 |
| Hawkes2 | 0.036 | 0.026 | 0.024 | 0.024 | 0.026 | 0.024 |

The model is robust once `K ≳ 8`.

### Results: with marks (Table 6)

| Model | Time NLL Reddit | Time NLL MOOC | Total NLL Reddit | Total NLL MOOC | Mark acc. Reddit | Mark acc. MOOC |
|---|---|---|---|---|---|---|
| LogNormMix | 10.28 ± 0.066 | 5.75 ± 0.040 | 12.40 ± 0.094 | 7.58 ± 0.047 | 0.62±0.014 | 0.45±0.003 |
| DSFlow | 10.28 ± 0.073 | 5.78 ± 0.067 | 12.39 ± 0.064 | 7.52 ± 0.074 | 0.62±0.013 | 0.45±0.004 |
| SOSFlow | 10.35 ± 0.106 | 6.06 ± 0.084 | 12.49 ± 0.158 | 7.78 ± 0.107 | 0.62±0.013 | 0.46±0.009 |
| FullyNN | 10.41 ± 0.079 | 6.22 ± 0.224 | 12.51 ± 0.094 | 7.93 ± 0.230 | 0.63±0.013 | 0.46±0.004 |
| LogNormal | 10.42 ± 0.076 | 6.38 ± 0.019 | 12.51 ± 0.080 | 8.11 ± 0.026 | 0.62±0.013 | 0.42±0.005 |
| RMTPP | 11.15 ± 0.061 | 10.29 ± 0.209 | 13.26 ± 0.085 | 12.14 ± 0.220 | 0.62±0.014 | 0.41±0.006 |

- The authors say marks as RNN input improve time prediction for all models. But Table 6's Reddit time NLLs (e.g. LogNormMix 10.28) are *higher* than Table 4's unmarked ones (10.19), so the evidence is only MOOC.
- Mark accuracy is similar across models, because the mark head is the same and conditionally independent of time.

### Other experiments

- **Extra conditioning (Yelp, §5.3, F.3):**
  - `y_i ∈ {0,1}²` = (weekday vs weekend, whether `t_{i−1}` falls in 5PM–11PM).
  - Each of the 4 combinations gets a learnable 64-dim embedding.
  - Adding `y_i` lowers NLL with or without `h_i` (Fig. 5).
  - Fig. 9: LogNormMix recovers the **daily multimodality** (peaks at 0h, 24h, 48h, …) of time-to-next-check-in. RMTPP cannot, and FullyNN struggles.
- **Missing-data imputation (§5.4, F.4):**
  - Data: one Hawkes1 sequence of 100 events, with a block of length ≈ `t_N/3` removed.
  - Three strategies, scored by NLL on the full sequence:

    | Strategy | How | NLL |
    |---|---|---|
    | (a) Ignore the gap | the gap enters as one long `τ` | 1.01 ± 0.20 |
    | (b) Mean imputation | fill with the mean `τ̂`; fed to the RNN but not to the loss | 0.81 ± 0.27 |
    | (c) Sampling with reparametrization | sample `τ^(imp)` from the model, feed it to the RNN, and maximise `E_{τ^(imp)∼p*}[Σ_i log p*(τ_i^(obs))]` | **0.36 ± 0.05** |

  - (a) and (b) overfit the partial sequence. Only (c) learns the true distribution, and it **requires reparametrised sampling**.
- **Sequence embeddings (§5.5, F.5):**
  - Pre-train `e_j` with the history disabled (`−(1/N) Σ_i log p_θ(τ_i | e_j)`), then enable the history (`−(1/N) Σ_i log p_θ(τ_i | e_j, h_i)`).
  - A t-SNE of the `e_j` separates the 5 synthetic process types without supervision.
  - Interpolating `½(e_SC + e_RN)` generates intermediate behaviour.
  - Sequence embeddings were *not* used in the main NLL experiments, because they cannot be learned for validation or test sequences.

## Gotchas and inconsistencies in the paper

- **App. D.2:** the formula for the affine scale `a` is printed as `a = sqrt( (1/N) Σ_i (log τ_i − b) )`, with the square missing. The intended value is `a = sqrt( (1/N) Σ_i (log τ_i − b)² )`, the std of `log τ`.
- **App. B, constant intensity:** printed as `p*(τ) = c exp(−c)`. It should be `c exp(−cτ)`.
- **App. B, Gompertz pdf:** printed with `exp(βt)`. It should be `exp(βτ)`.
- **App. D.3:** "linear scaling `τ̃ = aτ`, where `a = (1/N) Σ τ_i`" multiplies by the mean. To normalise you would divide, i.e. use `a = 1/mean`. Check the code.
- **App. F.1, sign of the scaling shift:** for `x = aτ` the paper writes `log q(x) = log p(τ) + log a`. The change of variables gives `q(x) = p(x/a)/a`, so it should be `− log a`; the NLL goes up by `log a`. The conclusion (absolute NLL is arbitrary) still holds.
- **§5.1 overclaim:** "Simple unimodal distributions … are always dominated by the more flexible models", but on MOOC FullyNN 6.83 ± 0.152 is worse than LogNormal 6.53 ± 0.016 (Table 4).
- **App. F.5:** the Figure 6 description says the bottom row used "`e_SC`, embedding of a renewal sequence". That should be `e_RN`.
- **Yelp is described two ways:** "reviews for the 300 most visited restaurants in Toronto" (E.2) and "check-ins" (§5.3, F.3).
- **K = 64 vs the MOOC sensitivity row:** Table 3 MOOC at K = 64 is 5.660, but the Table 4 LogNormMix MOOC result (also K = 64) is 6.03 ± 0.092.
- **The log-likelihood has no window-end survival term.** The likelihood stops at `t_N`, so the model only scores `n` events. It says nothing about stopping, so the count must come from elsewhere. In autoregressive generation it comes implicitly from "stop when past `T`" (this is how EventFlow runs IFTPP).
- **The flexibility vs inductive-bias trade-off.** In EventFlow's benchmark, IFTPP wins the simple synthetic MMDs but loses on real data and in multi-step forecasting. One-step NLL wins do not imply good rollouts.

## Takeaways for synthesis

- **LogNormMix is a strong, cheap time head for any autoregressive event model:**
  - closed-form NLL, CDF, survival, mean and sampling;
  - a valid density on `R+`;
  - robust to `K`;
  - reparametrisable.
  It drops into a transformer or RNN as an MDN head on the context `c_i`.
- **A window-truncated variant is easy.** The CDF `Σ w_k Φ((log τ − µ_k)/s_k)` gives the survival term `1 − F(T − t_N)` in closed form. This makes finite-window likelihoods (EventFlow Eq. 3) tractable.
- **Conditioning by concatenation** (`[h || y || e]`) handles static profiles (`y`) and per-entity latent embeddings (`e`). The Yelp result shows that explicit calendar features beat expecting the RNN to infer the clock. This mirrors the Banksformer date-features lesson.
- **Reparametrised sampling** opens up imputation, low-variance gradients through samples (no score-function estimator), and end-to-end training of multi-step objectives. These are the levers for fixing autoregressive drift that EventFlow attacks non-autoregressively.
- **Implementation tips:**
  - Feed `log τ` (centred) into the encoder.
  - Initialise the output affine layer with the dataset mean and std of `log τ`.
  - Report NLL differences per split, not raw NLLs.
