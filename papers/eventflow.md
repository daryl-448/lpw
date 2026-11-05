# EventFlow: Forecasting Temporal Point Processes with Flow Matching

- **Authors:** Gavin Kerrigan† (Univ. of Oxford), Kai Nelson† (UC Berkeley), Padhraic Smyth (UC Irvine). † work done at UC Irvine.
- **Venue:** AISTATS 2026 (PMLR vol. 300), Tangier, Morocco.
- **Code:** `https://github.com/GavinKerrigan/eventflow`
- **Task:** **Unmarked** TPPs (event times only, no marks). Two settings:
  1. **Forecasting:** generate the events in `[T0, T0 + ΔT]` given the history `H_{T0}`.
  2. **Unconditional generation:** generate whole sequences on `[0, T]`.

## Core idea (TL;DR)

EventFlow is a **non-autoregressive** generator of whole event sequences. It factorises a TPP into two parts:

1. **A count distribution:**
   - Unconditional: the empirical count distribution `µ1(n)`.
   - Forecasting: a learned classifier `p_φ(n | H)`.
2. **A flow-matching model over the joint event times, given n.** To sample:
   - draw `n` i.i.d. reference times from `q` and sort them;
   - push all `n` times together through a learned velocity field `v_θ(γ_s, s, e_H)` from flow time `s = 0` to `s = 1`.

The number of events is **fixed** during the flow, so count modelling is decoupled from time modelling. The model needs no likelihood, intensity, CDF integral or thinning. Training is a regression (MSE) loss. Sampling is an ODE solve, and 1 Euler step (1 NFE) is already competitive.

Headline claim: 20%–53% lower multi-step forecast error than the nearest baseline on 7 real benchmarks, with fewer model calls. Check this range yourself; see Gotchas.

## Background the paper argues against (§3)

Autoregressive TPPs model `p(t^k | H_{t^{k−1}})`, which gives the joint `p(t^1, …, t^n) = p(t^1) ∏_{k=2}^n p(t^k | H_{t^{k−1}})`.

History notation: `H_t = {t^k : t^k ≤ t}` (inclusive) and `H_{t−} = {t^k : t^k < t}`.

Conditional intensity:

```
λ*(t) := λ(t | H_{t−}) = p(t | H_{t^n}) / (1 − F(t | H_{t^n})),   F(t | H_{t^n}) = ∫_{t^n}^t p(s | H_{t^n}) ds

F(t | H_{t^n}) = 1 − exp( −∫_{t^n}^t λ*(s) ds )                       (1)
p(t | H_{t^n}) = λ*(t) exp( −∫_{t^n}^t λ*(s) ds )                       (2)
```

The likelihood on the window `[0, T]` needs a **survival term** for "no events in `(t^n, T]`":

```
L({t^k}_{k=1}^n) = p(t^1, …, t^n) (1 − F(T | H_{t^n}))                 (3)
                 = ( ∏_{k=1}^n λ*(t^k) ) exp(−Λ*(T)),   Λ*(T) = ∫_0^T λ*(s) ds    (4)
```

Pain points named by the paper:
- `Λ*(T)` (or the CDF) often needs Monte Carlo integration, with many forward passes.
- Thinning-based sampling needs an intensity upper bound (chosen by heuristic), is hard to parallelise and hard to implement.
- Diffusion TPPs (Lin et al. 2022) only maximise an ELBO of the next inter-event time, so they cannot compute Eq. 3.
- Multi-step autoregressive rollouts accumulate errors.

## Method (§4)

### Setup and notation

- `T = [0, T]` is the support. `Γ` is the configuration space: finite counting measures `γ = Σ_{k=1}^n δ[t^k]`. A TPP is `µ ∈ P(Γ)`.
- `N(γ)` is the event count.
- **Flow time** `s ∈ [0, 1]` and **event time** `t ∈ [0, T]` are separate ("orthogonal") axes.
- A TPP is fully characterised by a count distribution `µ(n)` plus joint densities `{µ^n(t^1, …, t^n)}_{n=1}^∞`, each supported on ordered tuples `t^1 < … < t^n` (Daley & Vere-Jones Prop. 5.3.II).
- `µ1` is the data TPP and `µ0` is the reference TPP.

### Balanced couplings

A coupling `ρ ∈ P(Γ × Γ)` with marginals `µ0, µ1` is **balanced** if `N(γ0) = N(γ1)` almost surely. `Π_b(µ0, µ1)` is the set of balanced couplings. Pairing equal-length sequences means no events are added or removed along the path.

**Proposition 1.** `Π_b(µ0, µ1)` is nonempty **iff** `µ0(n) = µ1(n)` for every `n ∈ Z≥0`.

The proof (App. B) builds `ρ(A1 × A2) = Σ_{n=0}^∞ µ(n) µ^n(A1^n) ν^n(A2^n)` (Eq. 14). The converse is a pushforward argument on `N`.

**Practical choice.**
- `µ0` is a **mixed binomial process**: draw `n ~ µ1(n)` (the data count distribution), then `n` i.i.d. times from a density `q` on `T`, then sort.
- Coupling: draw `γ1 ~ µ1`, then draw `N(γ1)` i.i.d. times from `q` and sort them to get `γ0`. So `γ0` and `γ1` are **independent given n**. There is no OT or minibatch pairing.
- `q = N(0, I_n)` in practice (App. C), even though §4 calls `q` "a density on `T`".

### Interpolant

For a pair `z = (γ0, γ1)` with `n` events each, the events are matched **by sorted rank**:

```
γ_s^z = Σ_{k=1}^n δ[t_s^k],   t_s^k = (1 − s) t_0^k + s t_1^k,   0 ≤ s ≤ 1          (5)
γ̂_s^z = Σ_{k=1}^n δ[t_s^k + ε^k],   ε^k ~ N(0, σ²)                                  (6)
η_s = ∫ η_s^z dρ(z)                                                                   (7)
```

- The count distribution of `η_s` equals `µ1(n)` for all `s`.
- At the endpoints, the joint time densities are `µ_0^n` and `µ_1^n` convolved with `N(0, σ² I_n)`. The path becomes exact as `σ² ↓ 0`.
- The noise is "instrumental in obtaining a well-specified model". The paper says `σ²` is not a critical hyperparameter and gives no value for it.
- The noise means the support can leak outside `T` (a stated limitation).

Conditional (per-pair) vector field. It is **constant** in `s` and `γ`:

```
v_s^z(γ) := [v_s^1, …, v_s^n]^T = [t_1^1 − t_0^1, …, t_1^n − t_0^n]^T                (8)
```

Marginal field (intractable, because `dη_s^z / dη_s` is unknown):

```
v_s(γ) = E_{z∼ρ}[v_s^z(γ) | γ_s^z = γ] = ∫ v_s^z(γ) (dη_s^z / dη_s)(γ) dρ(z)          (9)
```

### Training objective (conditional flow matching)

```
J(θ) = E_{s, (γ0, γ1), γ̂_s^z} [ ‖γ1 − γ0 − v_θ(γ̂_s^z, s)‖² ]                        (10)
```

This equals the MSE against the marginal `v_s`, up to a θ-independent constant. Sampling scheme: `s ~ U[0, 1]`, `z ~ ρ`, `γ̂_s^z` from Eq. 6.

For forecasting, a learned encoder gives `e_H = f_θ(H)`. It is trained **jointly** with `v_θ(γ_s, s, e_H)`.

**Algorithm 1 (training), condensed:**

```
1  γ1 ~ µ1;  s ~ U[0,1];  ε ~ N(0,1)
2  e_H = ∅
3  if forecast:
4      T0 ~ [ΔT, T − ΔT]                       (uniform, per §5.1)
5      H ← {t ∈ γ1 : t ≤ T0}
6      e_H ← f_θ(H)
7      γ1 ← {t ∈ γ1 : T0 < t ≤ T0 + ΔT}
8  n ← N(γ1)
9  t_0^k ~ q i.i.d., sorted, k = 1..n
10 γ0 ← (t_0^1, …, t_0^n)
11 t_s^k ← (1 − s) t_0^k + s t_1^k
12 γ_s^z ← (t_s^1, …, t_s^n)
13 gradient step on ‖γ1 − γ0 − v_θ(γ_s^z + ε, s, e_H)‖²
```

### Event-count model (forecasting only)

`p_φ(n | H)` is a categorical over counts up to `N_max` (Eq. 22 sums `k = 1..N_max`; zero-count handling is unspecified, see Gotchas), with `N_max` = the maximum sequence length in the training data. It is trained with cross-entropy plus an **OT-style smoothness regulariser** (App. C):

```
L(φ) = E_{n,H} [ −log p_φ(n | H) + (α / N_max) Σ_{k=1}^{N_max} p_φ(k | H) (n − k)² ]   (22)
```

- The regulariser is the squared OT distance between `p_φ(· | H)` and `δ[n]`. It penalises mass far from the true `n`.
- Grid: `α ∈ {0, 1/N_max, 10/N_max, 100/N_max, 1000/N_max}`.
- The paper reports "α/N_max" in Table 7 as 1.0 (PUBG), 100.0 (Yelp-M) and 1000.0 (the other five). Read literally, `α ∈ {N_max, 100·N_max, 1000·N_max}`, which is not on the grid. More likely the column holds `α·N_max`, i.e. the grid multiplier.
- Ablation `α = 0` (Table 16): MARE gets much worse, e.g. Reddit-C 0.70 → 2.01 and Twitter 0.46 → 1.23.
- Mixtures of Poissons or of Negative Binomials (which avoid `N_max`) did **worse** than the categorical logit.
- For unconditional generation, `n ~` the empirical training count distribution.

### Sampling (Algorithm 2)

```
choose 0 = s_0 < s_1 < … < s_K = 1         (uniform in practice)
e_H ← ∅
if forecast: e_H ← f_θ(H);  n ~ p_φ(n | H)
else:        n ~ µ1(n)
t_0^k ~ q i.i.d., sorted;  γ_0 ← (t_0^1, …, t_0^n)
for k = 1..K:
    h_k ← s_k − s_{k−1}
    γ_{s_k} ← γ_{s_{k−1}} + h_k v_θ(γ_{s_{k−1}}, s_{k−1}, e_H)          (forward Euler, Eq. 11)
return γ_{s_K}         (the printed algorithm says "return γ_0"; see Gotchas)
```

Learned paths are close to straight in practice, so Euler with few steps is enough.

## Architecture and training (App. C)

**Unconditional `v_θ` (Fig. 3):**
- Three linear embeddings, **added** together per element:
  - the event times `t^k` (one shared linear layer);
  - the flow time `s`;
  - the position index `k` (sinusoidal embedding followed by a linear layer).
- Then a standard transformer, then a shared linear head mapping each of the `N(γ_s)` outputs to a scalar velocity.

**Conditional `v_θ` (Fig. 4):**
- A transformer **encoder** embeds the history times and positions to give `e_H`.
- **`T0` is appended as the final event of `H`**. This worked better than encoding it separately.
- The **decoder** embeds `(γ_s, s, positions)` the same way and cross-attends to `e_H` ("in the intermediate layer"). Its output is `N(γ_s)` velocities.

**Count model `p_φ` (Fig. 5):** a transformer with the same overall architecture as the unconditional model (the text calls it a "transformer decoder"). Its input is the observed history `H` (event and position embeddings). It mean-pools the outputs, then applies a 3-layer residual MLP to get the logits over `n`.

**Training:**
- Normalise all sequences to `[−1, 1]` using the global min/max event time in the training data. Rescale after sampling.
- Adam (β1 = 0.9, β2 = 0.999), **30,000 steps**, cosine schedule restarting every 10,000 steps.
- Select the best validation checkpoint from 10 evaluations during training.
- Grid: lr ∈ {5×10⁻³, 10⁻³, 5×10⁻⁴}, dropout ∈ {0, 0.1, 0.2}. Learning rates ≥ 10⁻² often diverged.
- Default size: 6 layers, 8 heads, embedding dim 512, MLP dim 2048.
- Reddit-C and Reddit-S: 4 heads, embedding dim 128, MLP dim 256 (memory).
- Each model trains on a single A6000 GPU.
- Splits: 60/20/20 by sequence.

## Experiments (§5)

### Datasets (Table 4)

| Dataset | #Seq | Mean len | Std len | Range len | Support | ΔT |
|---|---|---|---|---|---|---|
| Hawkes1 | 1000 | 95.4 | 45.8 | [14, 300] | [0, 100] | − |
| Hawkes2 | 1000 | 97.2 | 49.1 | [18, 355] | [0, 100] | − |
| Nonstationary Poisson | 1000 | 100.3 | 9.8 | [71, 134] | [0, 100] | − |
| Nonstationary Renewal | 1000 | 98 | 2.9 | [86, 100] | [0, 100] | − |
| Stationary Renewal | 1000 | 109.2 | 38.1 | [1, 219] | [0, 100] | − |
| Self-Correcting | 1000 | 100.3 | 0.74 | [98, 102] | [0, 100] | − |
| PUBG | 3001 | 76.5 | 8.8 | [26, 97] | [0, 38] | 5 |
| Reddit-C | 1356 | 295.7 | 317.9 | [1, 2137] | [0, 24] | 4 |
| Reddit-S | 1094 | 1129 | 359.5 | [363, 2658] | [0, 24] | 4 |
| Taxi | 182 | 98.4 | 20 | [12, 140] | [0, 24] | 4 |
| Twitter | 2019 | 14.9 | 14 | [1, 169] | [0, 24] | 4 |
| Yelp-Airport | 319 | 30.5 | 7.5 | [9, 55] | [0, 24] | 4 |
| Yelp-Miss. | 319 | 55.2 | 15.9 | [3, 107] | [0, 24] | 4 |

- The synthetic sets come from Omi et al. (2019). The real sets are the Shchur et al. (2020b) unmarked benchmarks. ΔT follows Lüdke et al. (2023).
- The datasets mostly cover one day each (`[0, 24]`). PUBG is one game, up to 38 minutes.

### Baselines

All baselines were retuned by the authors with batch size 64, up to 1,000 epochs and early stopping.
- **IFTPP:** Shchur et al. 2020a, a log-normal mixture. See [intensity-free-tpp.md](intensity-free-tpp.md).
- **NHP:** the EasyTPP implementation, with thinning.
- **Diffusion:** Lin et al. 2022, 1000 steps with a cosine schedule. The authors modified it: an IFTPP RNN encoder, no log-scaling, and scaling to `[−1, 1]` with clipping.
- **Add-and-Thin:** Lüdke et al. 2023, a non-autoregressive diffusion model with 100 NFEs.
- AttNHP was dropped because it needed more than 24 GB of VRAM.

### Metrics

Sequence distance (Xiao et al. 2017a), with `n ≤ m`. It is the L1 distance between counting processes, i.e. W1 for unequal mass:

```
d(γ, η) = Σ_{k=1}^n |t_γ^k − t_η^k| + Σ_{k=n+1}^m (T − t_η^k)                       (12)
```

- **Forecasting:** use `T0 + ΔT` in place of `T`, then divide by `ΔT`. 50 values of `T0` per test sequence, 1 forecast each.
- **Unconditional:** MMD with kernel `k(γ, η) = exp(−d(γ, η) / (2σ²))`, where `σ` = the median pairwise distance. 1,000 generated sequences against the test set.
- **Counts:** `MARE = E_{n,H,n̂∼p_φ(n|H)} |(n̂ − n)/n|` (Eq. 23).

### Results: forecasting (Fig. 2 / Table 20; sequence distance, mean ± std over 5 seeds; lower is better)

| Model | PUBG | Reddit-C | Reddit-S | Taxi | Twitter | Yelp-A | Yelp-M |
|---|---|---|---|---|---|---|---|
| IFTPP | 4.2±0.7 | 25.6±2.3 | 61.2±3.2 | 5.1±0.4 | 2.9±0.2 | 2.1±0.2 | 3.4±0.2 |
| NHP | 2.8±0.1 | 31.0±1.4 | 95.7±0.7 | 4.5±0.3 | 3.4±0.5 | 1.8±0.1 | 3.0±0.2 |
| Diffusion | 5.4±1.2 | 25.7±0.9 | 80.3±11.4 | 4.6±0.7 | 2.4±0.2 | 1.8±0.1 | 3.3±0.7 |
| Add-and-Thin | 2.5±0.04 | 22.2±4.6 | 34.3±0.4 | 3.7±0.1 | 3.1±0.2 | 1.8±0.1 | 3.0±0.2 |
| **EventFlow (25 NFEs)** | **2.0±0.03** | **15.8±2.7** | **16.0±0.2** | **3.2±0.1** | **1.4±0.01** | **1.3±0.02** | **1.9±0.1** |
| EventFlow (10 NFEs) | 2.0±0.03 | 15.8±2.7 | 15.8±0.2 | 3.1±0.1 | 1.3±0.02 | 1.3±0.1 | 1.9±0.1 |
| EventFlow (1 NFE) | 2.0±0.01 | 15.8±2.7 | 15.8±0.2 | 3.2±0.2 | 1.4±0.03 | 1.8±0.3 | 1.9±0.1 |
| EventFlow (25 NFEs, **true n**) | 1.2±0.01 | 5.5±0.3 | 8.8±0.2 | 1.8±0.02 | 0.7±0.01 | 0.7±0.02 | 1.1±0.02 |

- The **true-n oracle** roughly halves the error. **The count predictor is the main bottleneck**, especially on long sequences (Reddit).
- The NFE sweep in main-text Table 1 matches Table 20, except **Taxi at 1 NFE (3.7 in Table 1 vs 3.2 in Table 20)** and **Twitter at 10 NFEs (1.4 vs 1.3)**.

### Results: event counts (Table 16, MARE; lower is better)

| Model | PUBG | Reddit-C | Reddit-S | Taxi | Twitter | Yelp-A | Yelp-M |
|---|---|---|---|---|---|---|---|
| IFTPP | 1.05 | 1.69 | 0.79 | 0.60 | 0.88 | 0.76 | 0.76 |
| NHP | 1.02 | 0.95 | 1.00 | 0.67 | 2.48 | 0.80 | 1.07 |
| Diffusion | 1.95 | 1.28 | 1.12 | 0.49 | 0.66 | 0.65 | 0.72 |
| Add-and-Thin | 0.43 | 0.99 | 0.38 | 0.33 | 0.60 | **0.42** | **0.46** |
| EventFlow | **0.40** | **0.70** | **0.16** | **0.28** | **0.46** | 0.56 | 0.50 |
| EventFlow (α = 0) | 0.69 | 2.01 | 0.26 | 0.47 | 1.23 | 0.66 | 0.80 |

The baselines model `n` only implicitly: the autoregressive ones stop at the first event outside `T`.

### Results: one-step-ahead MSE (Table 19)

On first-next-event MSE, all methods are similar, and EventFlow is often **not** best (e.g. Reddit-C: EventFlow 1.06 vs Add-and-Thin 0.30). The authors use this to argue that one-step metrics hide multi-step differences.

### Results: unconditional MMD (Table 3 / 17 / 18; ×10⁻², 1,000 samples, 5 seeds; lower is better)

| Model | H1 | H2 | NSP | NSR | SC | SR | PUBG | Red.C | Red.S | Taxi | Twitter | YelpA | YelpM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Data (held-out floor, App. E) | 1.3 | 1.3 | 1.8 | 3.0 | 5.7 | 1.1 | 1.3 | 0.6 | 0.4 | 3.1 | 2.6 | 3.6 | 3.1 |
| IFTPP | **1.5** | **1.4** | **2.3** | 6.2 | **5.8** | **1.3** | 5.7 | 1.3 | 1.9 | 5.8 | **2.9** | 8.2 | 5.1 |
| NHP | 1.9 | 5.2 | 3.6 | 12.6 | 25.4 | 5.0 | 7.2 | 2.2 | 22.5 | 5.0 | 7.3 | 6.7 | 6.1 |
| Diffusion | 4.8 | 5.5 | 10.8 | 15.0 | 9.1 | 5.1 | 14.3 | 3.9 | 6.2 | 11.7 | 12.5 | 10.9 | 10.5 |
| Add-and-Thin | 1.9 | 2.5 | 2.6 | 7.4 | 22.5 | 2.2 | 2.8 | 1.2 | 2.7 | 5.2 | 4.8 | **4.5** | **3.0** |
| EventFlow (25 NFEs) | 1.9 | 2.2 | 3.8 | **4.2** | 8.3 | 1.7 | **1.5** | **0.7** | **0.7** | **3.5** | 4.9 | 6.6 | **3.0** |

- Mean ranks: EventFlow 1.8, IFTPP 2.1, Add-and-Thin 2.4, NHP 3.7, Diffusion 4.8.
- EventFlow is best or second best on 11 of 13 datasets, and best on 5 of 7 real ones (YelpM is a tie).
- **IFTPP wins most synthetic sets.** The authors attribute this to the strong inductive bias of its parametric log-normal mixture. EventFlow, with no parametric form, wins on the real data.

### Runtimes (Table 21; seconds for 1,000 sequences)

| Model | PUBG | Reddit-C | Reddit-S | Taxi | Twitter | Yelp-A | Yelp-M |
|---|---|---|---|---|---|---|---|
| IFTPP | 16.15 | 20.77 | 69.14 | 22.62 | 1.85 | 6.55 | 9.60 |
| NHP | 0.99 | 6.13 | 11.94 | 1.19 | 0.83 | 0.74 | 0.89 |
| Diffusion | 43.77 | 99.82 | 262.26 | 78.77 | 43.41 | 29.82 | 32.85 |
| Add-and-Thin | 149.09 | 261.53 | 292.51 | 42.60 | 16.71 | 34.73 | 34.87 |
| EventFlow (1 NFE) | 35 | 739 | 885 | 8 | 42 | 6 | 10 |
| EventFlow (25 NFEs) | 769 | 17889 | 19624 | 187 | 960 | 142 | 233 |

- The transformer is costly on long sequences (Reddit).
- At 1 NFE, EventFlow is about as fast as IFTPP on moderate lengths (Taxi, Yelp).

## Limitations stated by the authors

- **No marks** and no spatial dimension. Adding them "may require new coupling and interpolation strategies".
- No hard support constraint on `[0, T]`, because of the Gaussian noise in Eq. 6. Constrained flows are suggested.
- The architectures differ across baselines, so gains are not isolated from the architecture.

## Gotchas and inconsistencies in the paper

- **"20%–53% lower" vs Table 20.** Against the best baseline: PUBG 20%, Reddit-C 29%, Reddit-S 53%, **Taxi 14% (3.2 vs 3.7)**, Twitter 42% (vs Diffusion 2.4), Yelp-A 28%, Yelp-M 37%. Taxi falls below the stated lower bound.
- **Table 1 vs Table 20:** Taxi at 1 NFE is 3.7 vs 3.2. Twitter at 10 NFEs is 1.4 vs 1.3 ± 0.02.
- **Algorithm 2 line 13 prints `return γ0`.** It should return the final state `γ_{s_K}` (= `γ1`).
- Line 1 of **Algorithm 1** samples `ε ~ N(0, 1)` as a scalar, but line 13 adds it to the n-vector `γ_s^z`. Eq. 6 has independent per-event noise `ε^k ~ N(0, σ²)`. Use a per-event vector with variance `σ²`. The paper gives no value for `σ`.
- The paper writes `(η_s)_{s∈[0.1]}` for `[0, 1]`. It also says "the unconditional path η_s defined in Equation (6)", but η_s is defined in Eq. 7.
- Eq. 10 omits `e_H`; Algorithm 1 includes it.
- **Eq. 9 conditions on the noiseless interpolant** (`γ_s^z = γ`), but the density ratio `dη_s^z / dη_s` belongs to the noisy path. The intended conditioning is on `γ̂_s^z = γ`.
- **Zero-count windows.** Eq. 22 sums `p_φ(k | H)` over `k = 1..N_max`, and Eq. 23 (MARE) divides by `n`. But forecast windows can be empty (Table 4 has sequences with only 1 event, e.g. Reddit-C and Twitter). The paper does not say how `n = 0` is handled in training, sampling or MARE. Check the code.
- `q` is called "a density on `T` (e.g., uniform)", but in practice it is `N(0, I_n)` on `[−1, 1]`-normalised data, so the reference samples are not confined to `T`.
- §3 mixes `{t_k}` with `{t^k}` and `p(t_k | t_1, …, t_{k−1})` with `p(t^k | H_{t^{k−1}})`.
- §3 defines `H_t` inclusively (`t^k ≤ t`). The IFTPP paper uses an exclusive history.
- Table 7 labels its column "α/N_max", but the values (1.0, 100.0, 1000.0) look like the grid multipliers, i.e. `α·N_max`.
- The "Data" MMD floor row is only in App. E (Tables 17–18), not in main-text Table 3.
- Table 8 (IFTPP tuned hyperparameters) lists 6 mixture components for Twitter, but the stated grid is {8, 16, 32, 64}.
- The checklist says code would be "released at a later date", but footnote 2 gives a URL.

## Takeaways for synthesis

- **Count × times factorisation.** "Choose `n` first, then move `n` particles jointly" is a clean way to make a sequence generator non-autoregressive. It is attractive for generating whole account histories over a fixed window, such as one month of transactions.
- **Balanced coupling with sorted-rank matching** removes birth/death events from the flow. The cost is that the count model carries the whole burden of `n`, and it is the measured bottleneck (see the true-n oracle).
- **Windowed forecasting recipe:** sample `T0 ∈ [ΔT, T − ΔT]` uniformly, encode `H` with `T0` appended as a pseudo-event, and predict `(n, times)` in `(T0, T0 + ΔT]`. This avoids rollout error accumulation.
- **The OT-regularised count CE (Eq. 22)** is a cheap, general trick for any ordinal count head.
- **Open extensions** the paper leaves to others:
  - Marks: categorical types and continuous amounts. Discrete or mixed flow matching per particle is a natural fit.
  - Support-respecting flows, e.g. in logit or inter-event space.
  - Conditioning on static profiles (cf. the IFTPP context vector `[h || y || e]`).
- **Evaluation:** use the sequence distance (Eq. 12) for forecasts, and MMD with the median-heuristic exponential kernel for unconditional samples. Report the held-out-data MMD as a floor. One-step MSE is not enough.
