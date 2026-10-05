#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
for fold in fold0 fold1; do
  marker="runs/jevfull_action_${fold}_v1/STATUS.txt"
  until [[ -f "$marker" ]]; do sleep 60; done
  grep -qx 'COMPLETED 24063' "$marker"
done
.venv/bin/python collect_jev_full_oof.py --target-fold fold0 --device cuda:0 > logs/jevfull_oof_fold0_v1.log 2>&1 &
pid0=$!
.venv/bin/python collect_jev_full_oof.py --target-fold fold1 --device cuda:1 > logs/jevfull_oof_fold1_v1.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
.venv/bin/python fit_jev_full_oof_platt.py > logs/jevfull_oof_platt_v1.log 2>&1
.venv/bin/python train_jev_full_confidence.py --device cuda:1 > logs/jevfull_confidence_v1.log 2>&1
