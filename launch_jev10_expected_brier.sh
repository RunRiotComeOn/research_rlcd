#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_expected_brier_v1
nohup python train_jev10_expected_brier.py --device cuda:0 --limit 300 \
  --beta 0.2 --run-name jev10_expected_brier_v1_300 \
  > logs/jev10_expected_brier_v1/train.log 2>&1 < /dev/null &
echo $! > logs/jev10_expected_brier_v1/train.pid
echo "started expected_brier=$(cat logs/jev10_expected_brier_v1/train.pid)"
