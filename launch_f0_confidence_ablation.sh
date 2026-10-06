#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
mkdir -p logs
for spec in base_low:2 base_high:3 f0_low:4 f0_high:5; do
  variant=${spec%%:*}
  gpu=${spec##*:}
  if [[ -e "runs/f0conf_${variant}_v1" ]]; then
    echo "Existing run: ${variant}" >&2
    exit 1
  fi
done
for spec in base_low:2 base_high:3 f0_low:4 f0_high:5; do
  variant=${spec%%:*}
  gpu=${spec##*:}
  nohup .venv/bin/python train_f0_confidence_ablation.py \
    --variant "$variant" --device "cuda:$gpu" \
    > "logs/f0conf_${variant}_v1.log" 2>&1 < /dev/null &
  echo "${variant} gpu=${gpu} pid=$!"
done
