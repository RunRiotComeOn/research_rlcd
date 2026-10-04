#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
for step in 100 200 300; do
  printf -v checkpoint 'runs/jev10_reward_proposed_schedule_v1_300/checkpoints/step_%04d' "$step"
  printf -v name 'schedule_step_%04d' "$step"
  python eval_jev10_reward_compare.py --mode proposed --step "$step" \
    --adapter "$checkpoint" --name "$name" --device cuda:0
done
echo 'COMPLETED schedule evaluation route'
