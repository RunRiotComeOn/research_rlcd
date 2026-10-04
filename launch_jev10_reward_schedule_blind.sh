#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_reward_schedule_blind_v1
nohup bash run_jev10_reward_schedule_blind_route.sh \
  > logs/jev10_reward_schedule_blind_v1/eval.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_schedule_blind_v1/eval.pid
echo "started schedule_blind=$(cat logs/jev10_reward_schedule_blind_v1/eval.pid)"
