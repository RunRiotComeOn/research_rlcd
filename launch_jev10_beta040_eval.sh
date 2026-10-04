#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_reward_beta040_v1
nohup bash run_jev10_beta040_eval_route.sh \
  > logs/jev10_reward_beta040_v1/eval.log 2>&1 < /dev/null &
echo $! > logs/jev10_reward_beta040_v1/eval.pid
echo "started beta040_eval=$(cat logs/jev10_reward_beta040_v1/eval.pid)"
