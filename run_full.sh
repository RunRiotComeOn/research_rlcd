#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
source ./env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python train.py --mode rl_only --run-name full_rl_only_2406 --adapter runs/sft_balanced_100/adapter > runs/full_rl_only_2406.log 2>&1
python train.py --mode rlcd --run-name full_rlcd_2406 --adapter runs/sft_balanced_100/adapter > runs/full_rlcd_2406.log 2>&1
python evaluate.py --run-name full_rl_only_92 --adapter runs/full_rl_only_2406/adapter > evaluations/full_rl_only_92.log 2>&1
python evaluate.py --run-name full_rlcd_92 --adapter runs/full_rlcd_2406/adapter > evaluations/full_rlcd_92.log 2>&1
python compare.py --vanilla evaluations/vanilla_92/summary.json --rl-only evaluations/full_rl_only_92/summary.json --rlcd evaluations/full_rlcd_92/summary.json --output comparisons/full_2406_92
echo COMPLETE > full_pipeline.STATUS
