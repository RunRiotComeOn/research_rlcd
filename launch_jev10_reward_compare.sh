#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_reward_compare_v2
nohup python train_jev10_reward_compare.py --mode old_brier --device cuda:0 --limit 300 \
  > logs/jev10_reward_compare_v2/old_brier.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_compare_v2/old_brier.pid
nohup python train_jev10_reward_compare.py --mode proposed --device cuda:3 --limit 300 \
  > logs/jev10_reward_compare_v2/proposed.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_compare_v2/proposed.pid
echo "started old_brier=$(cat logs/jev10_reward_compare_v2/old_brier.pid) proposed=$(cat logs/jev10_reward_compare_v2/proposed.pid)"
