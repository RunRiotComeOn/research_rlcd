#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
mkdir -p logs
for spec in lam0:2 lam05:3 lam1:4; do
  variant=${spec%%:*}
  if [[ -e "runs/f0joint_${variant}_v1" ]]; then
    echo "Existing run: ${variant}" >&2
    exit 1
  fi
done
for spec in lam0:2 lam05:3 lam1:4; do
  variant=${spec%%:*}
  gpu=${spec##*:}
  nohup .venv/bin/python train_f0_joint_confidence.py \
    --variant "$variant" --device "cuda:$gpu" \
    > "logs/f0joint_${variant}_v1.log" 2>&1 < /dev/null &
  echo "${variant} gpu=${gpu} pid=$!"
done
