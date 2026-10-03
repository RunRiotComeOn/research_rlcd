#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_blind_v1
nohup python eval_jev10_blind.py --model existing --device cuda:0 \
  > logs/jev10_blind_v1/existing.log 2>&1 < /dev/null &
echo $! > logs/jev10_blind_v1/existing.pid
nohup python eval_jev10_blind.py --model ce_brier --device cuda:6 \
  > logs/jev10_blind_v1/ce_brier.log 2>&1 < /dev/null &
echo $! > logs/jev10_blind_v1/ce_brier.pid
echo "started existing=$(cat logs/jev10_blind_v1/existing.pid) ce_brier=$(cat logs/jev10_blind_v1/ce_brier.pid)"
