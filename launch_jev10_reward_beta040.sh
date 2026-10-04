#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_reward_beta040_v1
nohup python train_jev10_reward_compare.py --mode proposed --beta 0.4 \
  --device cuda:0 --limit 300 --run-name jev10_reward_proposed_beta040_v1_300 \
  > logs/jev10_reward_beta040_v1/train.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_beta040_v1/train.pid
echo "started beta040=$(cat logs/jev10_reward_beta040_v1/train.pid)"
