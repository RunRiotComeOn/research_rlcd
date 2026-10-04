#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
dev_status=evaluations/jev10_unseen_confidence_v1/examples_4000/development/STATUS.txt
final_status=calibration/jev10_unseen_v1/final/STATUS.txt
until [[ -f "$dev_status" && -f "$final_status" ]]; do sleep 20; done
grep -qx 'COMPLETED 500' "$dev_status"
grep -qx 'COMPLETED 1000' "$final_status"
.venv/bin/python select_jev10_unseen_checkpoint.py
examples=$(python -c 'import json; print(json.load(open("calibration/jev10_unseen_v1/selected_checkpoint.json"))["examples"])')
.venv/bin/python eval_jev10_unseen_confidence.py --split final --examples "$examples" --device cuda:2
.venv/bin/python report_jev10_unseen_final.py --examples "$examples"
