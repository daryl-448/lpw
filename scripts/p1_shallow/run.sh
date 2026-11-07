#!/usr/bin/env bash
# Shallow-row correction, 2018 (reports/p1_shallow.md). usage: scripts/p1_shallow/run.sh [ROOT] [extra shallow_fit args]
set -euo pipefail
ROOT=${1:-runs/p1_shallow/2018}; shift || true
PILOT=data/pilot_2018
BASE=runs/p1_r0/2018/lpw_plan_prog_rp
SEEDS=${SEEDS:-5}
mkdir -p "$ROOT/gen"
echo "commit $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" > "$ROOT/run.log"
uv run python -m lpw.p1.shallow_fit --base "$BASE" --pilot "$PILOT" --out "$ROOT" "$@" 2>&1 | tee -a "$ROOT/run.log"
declare -A CK=([B]="$BASE lpw_plan_prog" [k]="$ROOT/k lpw_plan_prog_shallow_k" [kc]="$ROOT/kc lpw_plan_prog_shallow_kc" [kct]="$ROOT/kct lpw_plan_prog_shallow_kct")
for arm in B k kc kct; do
  read -r dir model <<< "${CK[$arm]}"
  for mode in oracle prior; do
    nll=$([ "$mode" = oracle ] && echo "" || echo "--no-nll")
    echo "gen $arm $mode" | tee -a "$ROOT/run.log"
    uv run python -m lpw.p1.drift_exp --ckpt "$dir" --model "$model" --mode "$mode" --pilot "$PILOT" --seeds "$SEEDS" \
      --save "$ROOT/gen/${arm}_${mode}.parquet" --out "$ROOT/gen/${arm}_${mode}.json" $nll > "$ROOT/gen/${arm}_${mode}.log" 2>&1
  done
done
uv run python scripts/p1_shallow/readout.py "$ROOT" --pilot "$PILOT" ${READOUT_ARGS:-} 2>&1 | tee -a "$ROOT/run.log"
