#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
mkdir -p logs
variants=(base_low base_high f0_low f0_high)
gpus=(2 3 4 5)
steps=(4000 8000 12000 16000 20000 20063)

while true; do
  ready=true
  for variant in "${variants[@]}"; do
    if [[ "$(cat "runs/f0conf_${variant}_v1/STATUS.txt" 2>/dev/null || true)" != 'COMPLETED 20063' ]]; then
      ready=false
    fi
  done
  if [[ "$(cat evaluations/f0_confidence_ablation_v1/holmes_actions/STATUS.txt 2>/dev/null || true)" != 'COMPLETED 1837' ]]; then
    ready=false
  fi
  if [[ "$ready" == true ]]; then break; fi
  sleep 30
done

run_eval() {
  local split=$1 variant=$2 step=$3 gpu=$4
  local out="evaluations/f0_confidence_ablation_v1/${split}/${variant}/examples_$(printf '%05d' "$step")"
  local expected=2000
  if [[ "$split" == holmes ]]; then expected=1837; fi
  if [[ "$(cat "$out/STATUS.txt" 2>/dev/null || true)" == "COMPLETED $expected" ]]; then
    return
  fi
  local resume=()
  if [[ -d "$out" ]]; then resume=(--resume); fi
  .venv/bin/python eval_f0_confidence_ablation.py \
    --split "$split" --variant "$variant" --step "$step" \
    --device "cuda:$gpu" "${resume[@]}" \
    >> "logs/f0conf_eval_${split}_${variant}_v1.log" 2>&1
}

for i in "${!variants[@]}"; do
  (
    for step in "${steps[@]}"; do
      run_eval dev "${variants[$i]}" "$step" "${gpus[$i]}"
    done
  ) &
  pids[$i]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ ! -f evaluations/f0_confidence_ablation_v1/selection.json ]]; then
  .venv/bin/python select_f0_confidence_checkpoints.py \
    > logs/f0conf_selection_v1.log 2>&1
fi
for i in "${!variants[@]}"; do
  (
    step=$(.venv/bin/python -c 'import json,sys; print(json.load(open("evaluations/f0_confidence_ablation_v1/selection.json"))["variants"][sys.argv[1]]["selected_step"])' "${variants[$i]}")
    run_eval test "${variants[$i]}" "$step" "${gpus[$i]}"
    run_eval holmes "${variants[$i]}" "$step" "${gpus[$i]}"
  ) &
  pids[$i]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done
.venv/bin/python summarize_f0_confidence_ablation.py \
  > logs/f0conf_final_summary_v1.log 2>&1
