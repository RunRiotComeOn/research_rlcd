#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_frozen_action_meanq_v1
nohup bash run_jev10_frozen_action_meanq_eval_route.sh \
  > logs/jev10_frozen_action_meanq_v1/eval.log 2>&1 < /dev/null &
echo $! > logs/jev10_frozen_action_meanq_v1/eval.pid
echo "started frozen_action_meanq_eval=$(cat logs/jev10_frozen_action_meanq_v1/eval.pid)"
