#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
mkdir -p logs
for spec in lam0:5 lam05:6 lam1:7; do
  variant=${spec%%:*}
  if [[ -e "runs/f0joint_${variant}_kl50_v1" ]]; then
    echo "Existing run: ${variant} KL=50" >&2
    exit 1
  fi
done
for spec in lam0:5 lam05:6 lam1:7; do
  variant=${spec%%:*}
  gpu=${spec##*:}
  nohup .venv/bin/python train_f0_joint_confidence.py \
    --variant "$variant" --action-kl-weight 50 --device "cuda:$gpu" \
    > "logs/f0joint_${variant}_kl50_v1.log" 2>&1 < /dev/null &
  echo "${variant} kl=50 gpu=${gpu} pid=$!"
done
