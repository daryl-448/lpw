# P1 anchor screen: pre-registration

Written and committed before any adapter was fitted (Sol anchor design round, 2026-11-06). Diagnosis:
`runs/p1_diag/drift_results.md` (local): heads calibrated on real histories, sampler faithful, but heavy accounts drift
jointly to slower pace and larger amounts in free-running generation. This screen is a cheap directional test of one
implementation (frozen backbone + plan-conditioned residual adapters); failing it rejects this adapter implementation,
not every robustness design. Passing it leads to a separately pre-registered full experiment.

## Arms (from `runs/p1_r0/2018/lpw_plan_prog_rp`, one seed)

| arm | training |
|---|---|
| B | untouched base checkpoint (reference) |
| C | adapters only, clean prefixes (`lpw.p1.anchor`, arm C) |
| R | adapters only, clean + 0.25 x corrupted prefixes (prefix-wide gaps x 10^{-0.3, 0, +0.3}, log10 amounts + {-0.5, 0, +0.5}, p = 0.25 / 0.5 / 0.25 each) |

Adapters: zero-initialised residuals on the time logits (bins 0-62 and the weekly logits; not the dormant bin) and the
amount-mixture parameters, input = the head's own input + plan embeddings; rows with an unknown plan bypass them exactly.
Fit: fixed 30% of training groups (seed 0), all their exact prefixes with real plans, cached head inputs, 1000 Adam steps,
4096 rows per step (plus 4096 corrupted rows for R), lr 3e-4, seed 0. No pace class in this screen (added only if R
passes, as a further arm).

## Gates (R vs C unless stated; all must pass)

1. Likelihood: exact full-val known-plan NLL per prefix, R - B <= 0.02 overall, normal and scam; s-unknown val terms
   bit-identical to B on every exact prefix.
2. Shift response (real val prefixes, plan >= 10, next event owned, normal, after 3 / 9 / 29 / 99 owned events pooled):
   mean |e_tt| (gap-forecast response to a history gap shift) and mean |e_aa| (amount-forecast response to a history
   amount shift) each reduced by >= 25%. Cross terms reported, not gated.
3. Pilot generation (val agents with plan >= 30, oracle plans, 3 replicas, seed 7, cap 150 events): pace score (mean W1
   of exposure-normalized tau_10 / tau_30, normal plans 30-100 and 100-300) reduced by >= 15%; amount W1 (normal plans
   >= 100, events 31-100, streams weighted equally) reduced by >= 15%; reach-100 (normal plans >= 100) not lower.

Applied by `scripts/p1_anchor/screen.py` (writes `screen.json` / `screen.txt`). Known deviation from the design
round: the paired real-vs-generated prefix forecast discrepancy is not computed; the pilot generation stands in for it.
