#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source ./env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for tag in std none; do
  for step in 25 50 75 100 200; do
    padded=$(printf "%04d" "$step")
    name="ablation_${tag}_${step}_dev92"
    python evaluate.py \
      --run-name "$name" \
      --adapter "runs/ablation_${tag}_200/checkpoints/step_${padded}" \
      --data data/dev_from_train_92.jsonl \
      > "evaluations/${name}.log" 2>&1
    echo "EVALUATED $name"
  done
done
python analyze_ablation.py
echo COMPLETE > comparisons/advantage_ablation_200/STATUS.txt
