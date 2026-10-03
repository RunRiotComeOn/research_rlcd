#!/usr/bin/env bash
set -euo pipefail
mode="$1"
device="$2"
cd /pfs/hyx/videojev-rlcd
source env.sh
if [[ "$mode" == old_brier ]]; then
  python eval_jev10_reward_compare.py --mode baseline --device "$device"
fi
for step in 100 200 300; do
  python eval_jev10_reward_compare.py --mode "$mode" --step "$step" --device "$device"
done
echo "COMPLETED evaluation route $mode"
