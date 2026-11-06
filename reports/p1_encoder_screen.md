# P1 encoder screen: pre-registration

Written and committed before any screen run was trained (2026-11-06; Sol B1-gap design round).

## Why this screen

The GRU baseline B1 beats the transformer on full NLL.

| model | full NLL, val (nats per owned event) |
|---|---|
| B1 (GRU) | 12.938 |
| lpw | 13.105 |
| current plan/progress checkpoint, s-unknown | 13.128 |

Beating B1 is a P1 exit criterion. Where the gap comes from (`runs/p1_diag/b1gap/breakdown.txt`):
- B1 is better on every NLL term.
- The gap grows with prefix depth: +0.006 at depth 0, +0.47 per prefix at depth 256 and beyond, with both models seeing the same 255-event window.
- 61% of it comes from accounts with 300 or more events. That is only about 21 val accounts.

Leading hypothesis: the GRU gets a recency/order bias for free and is more sample-efficient. Neither "missing context" nor "attention dilution" is established. The old pair (`runs/p1/2018/{lpw,b1}`) also predates the current training recipe. So every arm here, B1 included, is retrained under the same recipe.

## Arms

All arms use the same code, data, CTX 255, prefix weighting, heads, stop recalibration, optimiser schedule, 80-epoch cap and patience 6. Each arm is tried at lr 1e-3 and 3e-4, with seed 20261104.

| arm | config | description |
|---|---|---|
| B1 | `b1` | 2-layer GRU, the comparator, given the same LR tuning |
| reference | `tf_ref` | existing 4-layer absolute-position transformer, with independently initialised layers |
| ALiBi | `tf_alibi` | 4 layers, per-head linear recency bias between event tokens (slopes 2^-2, 2^-4, 2^-6, 2^-8); attention to the PROFILE token is never penalised; no position embeddings |
| small | `tf_small` | 2 layers, ff 128, absolute positions, independent init; same size as B1 (524,288 vs 523,264 parameters here) |

Independent layer initialisation applies to every screen transformer. `nn.TransformerEncoder` deep-copies one initialised layer, so the old `lpw` started with identical layers.

## Procedure (`scripts/p1_encoder/screen.py`, outputs in `runs/p1_enc/2018/`)

1. **Fit.** 8 runs (4 arms x 2 LRs), seed 20261104. Early stopping uses the existing tiled-val loss. Stop recalibration is staged.
2. **Score.** Score every run on exact val full NLL per owned event (`lpw.p1.evaluate`).
3. **Select once** (`selection.json`, written once and never overwritten).
   - Each arm keeps its better LR.
   - The challenger is the transformer arm with the lowest exact val full NLL.
   - If the challenger later fails, no runner-up is substituted.
4. **Second seed.** Retrain the challenger and B1 at their selected LRs with seed 20261105.
5. **Generation.** Run 3 paired val replicas (seed 7) for the seed-1 challenger, reference and B1, with the existing noise-floor procedure and role parity.

## Gates (all must pass)

**L, likelihood.** Define Δ = challenger − B1, in exact val full NLL per owned event.
- **Margin, in both seeds:** Δ ≤ −0.03, and the one-sided 95% upper bound of Δ is below 0. The bound uses a paired bootstrap that resamples whole val groups (1000 resamples, seed 0, `agents.parquet` group_id).
- **Other baselines, in both seeds:** the challenger beats B0 on full NLL (16.231) and B2 on shared NLL (12.684), using the counterparty-marginalised amount. These come from `runs/p1/2018/results_val.json`. B0 and B2 are not retrained, because the margins are above 1 nat.

**G, generation veto** (seed-1 runs).
- The challenger's mean M-pass in each role is not below the reference's.
- Role-AUC |delta| ≤ 0.05.
- No capped streams.
- This is a screening safeguard only. P1 still requires ≥70% per role.

**Reported, not gated:**
- Δ by prefix depth.
- Leave-one-group-out Δ over the groups that contain an account with 300 or more events.
- Per-metric synth/floor ratios.

## After the screen

**If it passes:**
- **Plan/progress carry-over.** Rebuild the plan/progress transfer on the challenger: same plan definitions, 20% unknown dropout, shared dropout draw, whole-stream owned-count progress, staged recalibration. Gate the reportable s-unknown NLL again.
- **Confirmation.** Train the frozen challenger and a comparably tuned B1 on the 2021 slice. Success is required separately per slice.
- **Then the drift work.** Rebase the deferred drift gates on the new base.
- **Needs the user's OK:** a 3-seed confirmation, and any run over 4 hours.

**If it fails:** no encoder change is adopted from this screen. SPEC §6.3 specifies a 4-layer transformer, so adopting the GRU itself would need a recorded SPEC amendment. Renaming B1 would not meet the "beats B1" criterion.

**Deferred diagnostics, not part of this screen:**
- Attention-window restriction and position-shift probes.
- Counterparty-loss decomposition.

## Compute

| GPU | per transformer fit | per GRU fit | whole screen |
|---|---|---|---|
| GTX 1650S | about 25–40 s per epoch, ~25–35 min including recalibration | about 15 min | 4–6 h |
| A30 (JarvisLabs) | not benchmarked | not benchmarked | runs go in parallel |

On the A30, fits are independent, so stage-1 fits may run concurrently. That does not change any result.
