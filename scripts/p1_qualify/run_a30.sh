#!/bin/bash
# base qualification on a JarvisLabs A30 (reports/p1_base_qualification.md); outputs under runs/p1_qualify
set -euo pipefail
cd /home/tx-pred-gen
export UV_PYTHON_INSTALL_DIR=/home/.uv-python UV_CACHE_DIR=/home/.uv-cache
for p in data/pilot data/pilot_2018; do [ -d ${p}_conf ] || uv run python scripts/p1_qualify/make_conf_pilot.py $p; done
R=runs/p1_qualify; mkdir -p $R/logs
Q="uv run python -u scripts/p1_qualify/qualify.py"
one() {  # slice name seed
  local run=$R/$1/${2}_s${3}
  $Q fit $1 $2 $3 > $R/logs/fit_$1_$2_$3.log 2>&1
  [ -f $run/conf_eval.json ] || $Q eval $run > $R/logs/eval_$1_$2_$3.log 2>&1
  [ -f $run/gen_conf.json ] || $Q gen $run > $R/logs/gen_$1_$2_$3.log 2>&1
}
export -f one; export R Q
echo "$(date +%T) fits"
for sl in 2018 2021; do for n in tf_alibi tf_ref b1; do for s in 20261104 20261105; do echo "$sl $n $s"; done; done; done \
  | xargs -P 4 -L 1 bash -c 'one $0 $1 $2'
echo "$(date +%T) report"
$Q report
echo "$(date +%T) done"
