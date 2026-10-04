#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
for examples in 1000 2000 3000 4000; do
  checkpoint="runs/jev10_unseen_confidence_v1/checkpoints/examples_${examples}/STATUS.txt"
  until [[ -f "$checkpoint" ]]; do sleep 20; done
  grep -qx "COMPLETED ${examples}" "$checkpoint"
  .venv/bin/python eval_jev10_unseen_confidence.py --split development --examples "$examples" --device cuda:2
done
