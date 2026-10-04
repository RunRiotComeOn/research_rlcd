#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_expected_brier_v1
nohup bash run_jev10_expected_brier_eval_route.sh \
  > logs/jev10_expected_brier_v1/eval.log 2>&1 < /dev/null &
echo $! > logs/jev10_expected_brier_v1/eval.pid
echo "started expected_brier_eval=$(cat logs/jev10_expected_brier_v1/eval.pid)"
