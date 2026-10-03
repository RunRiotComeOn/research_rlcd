#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_reward_blind_v1
nohup bash -c 'python eval_jev10_reward_compare.py --dataset blind --mode baseline --device cuda:0 && python eval_jev10_reward_compare.py --dataset blind --mode old_brier --step 300 --device cuda:0' \
  > logs/jev10_reward_blind_v1/baseline_old.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_blind_v1/baseline_old.pid
nohup python eval_jev10_reward_compare.py --dataset blind --mode proposed --step 300 --device cuda:3 \
  > logs/jev10_reward_blind_v1/proposed.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_blind_v1/proposed.pid
echo "started baseline_old=$(cat logs/jev10_reward_blind_v1/baseline_old.pid) proposed=$(cat logs/jev10_reward_blind_v1/proposed.pid)"
