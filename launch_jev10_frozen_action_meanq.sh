#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
mkdir -p logs/jev10_frozen_action_meanq_v1
nohup python train_jev10_frozen_action_meanq.py --device cuda:0 --limit 300 \
  --run-name jev10_frozen_action_meanq_v1_300 \
  > logs/jev10_frozen_action_meanq_v1/train.log 2>&1 < /dev/null &
echo $! > logs/jev10_frozen_action_meanq_v1/train.pid
echo "started frozen_action_meanq=$(cat logs/jev10_frozen_action_meanq_v1/train.pid)"
