#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
python eval_jev10_reward_compare.py --mode proposed --step 300 \
  --dataset schedule_blind --name fixed_step_0300 --device cuda:0
python eval_jev10_reward_compare.py --mode proposed --step 300 \
  --dataset schedule_blind --name schedule_step_0300 \
  --adapter runs/jev10_reward_proposed_schedule_v1_300/checkpoints/step_0300 \
  --device cuda:0
echo 'COMPLETED schedule blind evaluation route'
