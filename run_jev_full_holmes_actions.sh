#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
marker=runs/jevfull_action_all_v1/STATUS.txt
until [[ -f "$marker" ]]; do sleep 60; done
grep -qx 'COMPLETED 48126' "$marker"
.venv/bin/python collect_jev_full_holmes_actions.py --device cuda:0 > logs/jevfull_holmes_actions_v1.log 2>&1
