# P3a readout: sampled-population, uncoupled policy transfer (2018, development evaluation on val)

- **Protocol.** `reports/p3a_prereg.md`, frozen at commit `ab81c64` before any utility number. It was reviewed with
  sol-loop: 3 rounds, gpt-6.1-sol, approved.
- **Outputs.** `runs/p3a/2018/readout.{txt,json}` (the readout ran at `ab81c64`, clean tree), plus
  `world{0,1,2}/checkpoint0.txt` and the Checkpoint 0 diagnosis `world0/diag_cp0.md`.
- **Scope.** Real val is development data and was used heavily in P1. The sealed test is untouched. Worlds are
  uncoupled, and features are local.

## Headline (prereg §7): SUPPORTED, in both arms

| arm | TSTR AUC (3 worlds) | TRTR AUC | excess-AUC ratio | D [CI95] | outcome |
|---|---|---|---|---|---|
| **kct** | 0.909 ± 0.004 (0.908, 0.915, 0.906) | 0.932 | 0.948 | +0.064 [+0.041, +0.082] | **SUPPORTED** |
| B (control) | 0.908 ± 0.006 (0.901, 0.914, 0.909) | 0.932 | 0.945 | +0.063 [+0.042, +0.079] | **SUPPORTED** |

The setup: an XGBoost role detector on `full` node features, trained on one fully synthetic train-size world, and
tested on real val groups.

- **AP and F1.** TSTR AP is 0.760 vs TRTR 0.813, and F1 is 0.712 vs 0.757 (kct; F1 thresholds from training-source
  OOF).
- **Robustness:**
  - every detector × feature set keeps a ratio of 0.89-0.97;
  - the owned-only real observation gives TSTR 0.909-0.915 vs TRTR 0.931;
  - removing capped streams leaves the headline SUPPORTED (ratio 0.945).
- **Confound control (§7.5).** Matching gives 348 val pairs, with max |SMD| 0.054 after matching.
  - Inside the matched sample, TSTR is 0.876 vs TRTR 0.906.
  - The behavioural gain over the confounder-only baseline transfers almost fully: **+0.240 synthetic-trained vs
    +0.241 real-trained**.
  - The `noconf` set (no age, era, cohort or profile) gives TSTR 0.901 vs TRTR 0.929.
- **The arms do not differ.** The shallow-row correction (kct) neither helps nor hurts utility.

## Criticism tests (prereg §8; arm result = gate on the mean over worlds)

| test | kct | B | rule | result |
|---|---|---|---|---|
| Seam C2ST, `local` (normal / scam) | 0.871 / 0.896 | 0.869 / 0.902 | each ≤ 0.70; scam - normal ≤ 0.05 | **fail** on absolute distinguishability; role gap passes (+0.024 / +0.033) |
| Difficulty parity, AP at val prevalence (XGB) | 0.684 vs 0.813 | 0.665 vs 0.813 | within ±0.05 | **fail**: synthetic worlds are harder (all 4 detectors, -0.12 to -0.15) |
| Profile-gain parity, AUC | +0.003 vs +0.002 | +0.002 vs +0.002 | same sign, \|Δ\| ≤ 0.05 | pass, but **uninformative**: profile adds almost nothing beyond behaviour, even on real |
| Shortcut audit (\|Δ\| > 0.1 flagged) | 7 flags | 7 flags | none flagged | **fail**: all 5 balance sentinels, plus OUT median amount and median fee |
| DCR, 5th-percentile ratio (normal / scam) | 7.6 / 3.5 | 7.9 / 4.1 | ≥ 1 | pass: no copying of real records |
| Mode coverage P / R, normal | 0.87 / 0.87 | 0.87 / 0.86 | reported | |
| Mode coverage P / R, scam | 0.87 / 0.58 | 0.89 / 0.57 | reported | scam recall: synthetic scams cover about 58% of real scam modes |
| Detector-ranking ρ | +0.87 | +0.93 | reported | |

**Shortcut flags** (single-feature role AUC, synthetic minus real; kct):

| feature | Δ |
|---|---|
| end_balance | +0.174 |
| ends_near_zero | -0.148 |
| min_balance | +0.130 |
| sweep_out_share | -0.129 |
| net_flow | +0.121 |
| fee_med | -0.106 |
| out_amt_med | -0.103 |

Every flag is a signal the synthetic data **lacks** or inverts. None is a template-like signal that is stronger in
synthetic data.

## Reading

**Transfer holds.**
- A detector trained only on a sampled, uncoupled LPW population keeps about 95% of the real-trained detector's
  excess AUC on real held-out groups.
- This holds inside the confound-matched sample too: behavioural signal transfers as fully as the overall AUC.
- This is the T4 result for local node features.

**But the synthetic world is distinguishable and too hard.**
- C2ST is 0.87-0.90, synthetic → synthetic AP is 0.13-0.15 below real, and scam mode recall is 0.58.
- Synthetic scams carry the right direction of role signal but less of it, and cover a narrower set of behaviours.
  This matches P1 known defect 2 (role AUC -0.06).

**The seam is mostly the balance trajectory** (known defect 1). Exploratory, not a gate
(`runs/p3a/2018/explore_seam_nobal.txt`):

| C2ST on (normal / scam) | kct | B |
|---|---|---|
| BAL alone | 0.81 / 0.86 | 0.82 / 0.88 |
| LOCAL + VENUE, without BAL | 0.74-0.75 / 0.76 | 0.75-0.76 / 0.79-0.80 |

- Even a perfect balance repair would leave a seam above 0.70.
- **Defect 1 harms the seam, not transfer.** Dropping BAL (`nobal`) leaves TSTR unchanged (0.908 vs 0.909).
- The single-feature seam ranking (exploratory) also leads with balance features: neg_out_share, sweep_out_share,
  ends_near_zero, min_balance, net_flow, then fee_med.

**The profile incoherence found at Checkpoint 0 hardly matters here.**
- Sampled pre-existing profiles break real arithmetic (profile + confounder C2ST 0.96), but the profile carries
  almost no role information beyond behaviour in 2018: real profile gain is +0.002 AUC.
- The `full` C2ST matches `local` (0.878 vs 0.871).
- **Consequence for C1.** The profile-gain parity "pass" says little about whether profiles carry the right signal.

## Triggers and next steps

1. **The defect-1 repair trigger fired** (prereg §10): the balance sentinels are flagged. The budget is
   pre-registered: one learned sweep atom, at most 1 day of build and 2 h GPU. Acceptance is on mechanism, safety and
   fidelity only, then a readout on fresh worlds 10-12. This readout stays the baseline.
2. **Candidate defect 8: profile arithmetic.** Low priority for P3a utility, since profiles carry little signal. It
   matters for the released node table (C1, T5).
3. **The encoder mode-coverage tranche** (prereg §8) is still owed. C2 stays incomplete until it runs.
4. **P3b go / no-go** is the user's decision.
5. **The sealed test** is opened once, with authorization, at the end. TRTR is also reported on the clean test
   subset.
