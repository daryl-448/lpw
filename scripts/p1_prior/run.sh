#!/usr/bin/env bash
# Prior-mode loss decomposition, 2018 (reports/p1_prior.md). usage: scripts/p1_prior/run.sh [ROOT]
set -euo pipefail
ROOT=${1:-runs/p1_prior/2018}
PILOT=data/pilot_2018
SH=${SH:-runs/p1_shallow/2018}
SEEDS=${SEEDS:-5}
mkdir -p "$ROOT/gen"
echo "commit $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" > "$ROOT/run.log"
for mode in prior_exp perm perm_a; do
  echo "gen kct $mode" | tee -a "$ROOT/run.log"
  uv run python -m lpw.p1.drift_exp --ckpt "$SH/kct" --model lpw_plan_prog_shallow_kct --mode "$mode" --pilot "$PILOT" \
    --seeds "$SEEDS" --no-nll --save "$ROOT/gen/kct_${mode}.parquet" --out "$ROOT/gen/kct_${mode}.json" > "$ROOT/gen/kct_${mode}.log" 2>&1
done
uv run python scripts/p1_prior/decompose.py "$ROOT" --base-gen "$SH/gen" --pilot "$PILOT" ${DECOMP_ARGS:-} 2>&1 | tee -a "$ROOT/run.log"
