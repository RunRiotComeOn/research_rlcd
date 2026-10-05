#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
status=runs/jevfull_action_fold0_v1/STATUS.txt
until [[ -f "$status" ]]; do sleep 60; done
grep -qx 'COMPLETED 24063' "$status"
.venv/bin/python train_jev_full_action.py --dataset fold1 --device cuda:1
