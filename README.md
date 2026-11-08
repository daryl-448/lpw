# LPW: Learned-Policy World

A synthetic AML-style data simulator built from a learned next-event policy. Every agent runs the same model, and
illicit agents are conditioned on a learned role, not written from templates. The synthetic data is judged on real
labels: train on synthetic, test on real (TSTR).

The full design is in [SPEC.md](SPEC.md). The data is the HuggingFace dataset
`fesevu/ethereum_fraud_dataset_by_activity`, using 12-month pilot slices from 2018 and 2021.

## Result (2018 arc, closed 2026-11-08)

The confirmation was pre-registered and run once on the sealed 2018 test split. All three steps came out SUPPORTED:

| claim | test |
|---|---|
| P3a transfer: synthetic-trained XGB keeps ≥ 80% of the real-trained excess AUC | D +0.070 (lower bound +0.061) |
| P3b: the learned identity attachment beats its restricted control | H +0.0140 (lower bound +0.0080) |
| STRUCT-only retention ≥ 80% | D_S +0.025 (lower bound +0.0041) |

The synthetic world is still easy to tell apart from real data (C2ST 0.88–0.92). Synthetic-trained AP stays below
real-trained AP, coupling fidelity fails, and incremental structural utility is NOT TESTABLE. The 2018 test split is
now spent. For what the results do and do not license, see
[reports/lpw_2018_arc.md](reports/lpw_2018_arc.md).

## Layout

- `src/lpw/`: the library: `data` (P0 slices and splits), `p1` (next-event policy), `p3` (population, observation
  operator, features, utility statistics), `p3b` (shared-identity attachment), `sealed.py` (the test-split guard)
- `scripts/`: one directory per experiment stage; `scripts/p3_joint/` is the sealed-test entry point
- `reports/`: pre-registrations, readouts and audits, one per stage
- `tests/`, and `papers/` (paper digests)

## Setup

```sh
uv sync
uv run pytest
```

The pilot ran on one 4 GB GPU (GTX 1650 Super). A full P3 joint val rehearsal takes about 50 minutes.
