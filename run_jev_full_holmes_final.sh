#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
for item in \
  'runs/jevfull_confidence_oof_v1/STATUS.txt:COMPLETED 48126' \
  'evaluations/jevfull_holmes_v1/actions/STATUS.txt:COMPLETED 1837' \
  'calibration/jevfull_oof_v1/platt/STATUS.txt:COMPLETED 48126'; do
  marker="${item%%:*}"
  expected="${item#*:}"
  until [[ -f "$marker" ]]; do sleep 60; done
  grep -qx "$expected" "$marker"
done
.venv/bin/python eval_jev_full_holmes_confidence.py --device cuda:1 > logs/jevfull_holmes_final_v1.log 2>&1
