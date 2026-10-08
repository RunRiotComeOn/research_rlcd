#!/usr/bin/env bash
set -euo pipefail
cd /pfs/hyx/videojev-rlcd
mkdir -p logs
specs=(lam0:5:2 lam05:5:3 lam1:5:4 lam0:50:5 lam05:50:6 lam1:50:7)
steps=(4000 8000 12000 16000 16063)

while true; do
  ready=true
  for spec in "${specs[@]}"; do
    IFS=: read -r variant kl gpu <<< "$spec"
    suffix=''
    if [[ "$kl" == 50 ]]; then suffix=_kl50; fi
    if [[ "$(cat "runs/f0joint_${variant}${suffix}_v1/STATUS.txt" 2>/dev/null || true)" != 'COMPLETED 16063' ]]; then
      ready=false
    fi
  done
  if [[ "$ready" == true ]]; then break; fi
  sleep 30
done

run_eval() {
  local split=$1 variant=$2 kl=$3 step=$4 gpu=$5
  local suffix=''
  if [[ "$kl" == 50 ]]; then suffix=_kl50; fi
  local out="evaluations/f0_joint_confidence_v1/${split}/${variant}${suffix}/examples_$(printf '%05d' "$step")"
  local expected=2000
  if [[ "$split" == holmes ]]; then expected=1837; fi
  if [[ "$(cat "$out/STATUS.txt" 2>/dev/null || true)" == "COMPLETED $expected" ]]; then
    return
  fi
  local resume=()
  if [[ -d "$out" ]]; then resume=(--resume); fi
  .venv/bin/python eval_f0_joint_confidence.py \
    --split "$split" --variant "$variant" --step "$step" \
    --action-kl-weight "$kl" --device "cuda:$gpu" "${resume[@]}" \
    >> "logs/f0joint_eval_${split}_${variant}${suffix}_v1.log" 2>&1
}

for i in "${!specs[@]}"; do
  (
    IFS=: read -r variant kl gpu <<< "${specs[$i]}"
    for step in "${steps[@]}"; do
      run_eval dev "$variant" "$kl" "$step" "$gpu"
    done
  ) &
  pids[$i]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ ! -f evaluations/f0_joint_confidence_v1/selection_grid.json ]]; then
  .venv/bin/python select_f0_joint_confidence_grid.py \
    > logs/f0joint_selection_grid_v1.log 2>&1
fi
read -r variant step kl < <(.venv/bin/python -c 'import json; d=json.load(open("evaluations/f0_joint_confidence_v1/selection_grid.json"))["global_selection"]; print(d["variant"],d["step"],d["action_kl_weight"])')
run_eval test "$variant" "$kl" "$step" 2
run_eval holmes "$variant" "$kl" "$step" 2
.venv/bin/python summarize_f0_joint_confidence.py --grid \
  > logs/f0joint_final_summary_grid_v1.log 2>&1
