#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
mkdir -p logs
variants=(lam0 lam05 lam1)
gpus=(2 3 4)
steps=(4000 8000 12000 16000 16063)

while true; do
  ready=true
  for variant in "${variants[@]}"; do
    if [[ "$(cat "runs/f0joint_${variant}_v1/STATUS.txt" 2>/dev/null || true)" != 'COMPLETED 16063' ]]; then
      ready=false
    fi
  done
  if [[ "$ready" == true ]]; then break; fi
  sleep 30
done

run_eval() {
  local split=$1 variant=$2 step=$3 gpu=$4
  local out="evaluations/f0_joint_confidence_v1/${split}/${variant}/examples_$(printf '%05d' "$step")"
  local expected=2000
  if [[ "$split" == holmes ]]; then expected=1837; fi
  if [[ "$(cat "$out/STATUS.txt" 2>/dev/null || true)" == "COMPLETED $expected" ]]; then
    return
  fi
  local resume=()
  if [[ -d "$out" ]]; then resume=(--resume); fi
  .venv/bin/python eval_f0_joint_confidence.py \
    --split "$split" --variant "$variant" --step "$step" \
    --device "cuda:$gpu" "${resume[@]}" \
    >> "logs/f0joint_eval_${split}_${variant}_v1.log" 2>&1
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

if [[ ! -f evaluations/f0_joint_confidence_v1/selection.json ]]; then
  .venv/bin/python select_f0_joint_confidence.py \
    > logs/f0joint_selection_v1.log 2>&1
fi
read -r variant step < <(.venv/bin/python -c 'import json; d=json.load(open("evaluations/f0_joint_confidence_v1/selection.json"))["global_selection"]; print(d["variant"],d["step"])')
run_eval test "$variant" "$step" 2
run_eval holmes "$variant" "$step" 2
.venv/bin/python summarize_f0_joint_confidence.py \
  > logs/f0joint_final_summary_v1.log 2>&1
