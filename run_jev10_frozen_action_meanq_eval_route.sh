#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source env.sh
for step in 100 200 300; do
  python eval_jev10_frozen_action_meanq.py --device cuda:0 --step "$step"
done
echo 'COMPLETED frozen-action mean-q validation route'
