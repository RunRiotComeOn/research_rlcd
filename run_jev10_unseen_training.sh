#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
fit_status=calibration/jev10_unseen_v1/fit/STATUS.txt
dev_status=calibration/jev10_unseen_v1/development/STATUS.txt
until [[ -f "$fit_status" && -f "$dev_status" ]]; do sleep 20; done
grep -qx 'COMPLETED 4000' "$fit_status"
grep -qx 'COMPLETED 500' "$dev_status"
.venv/bin/python fit_jev10_unseen_calibration.py
.venv/bin/python train_jev10_unseen_confidence.py --device cuda:1
