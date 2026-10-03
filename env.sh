#!/usr/bin/env bash
set -euo pipefail
export VIDEOJEV_ROOT=/pfs/hyx/videojev-rlcd
export HF_HOME="$VIDEOJEV_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$VIDEOJEV_ROOT/.cache"
export TORCH_HOME="$VIDEOJEV_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$VIDEOJEV_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$VIDEOJEV_ROOT/.cache/inductor"
export UV_CACHE_DIR="$VIDEOJEV_ROOT/.cache/uv"
export PIP_CACHE_DIR="$VIDEOJEV_ROOT/.cache/pip"
export TMPDIR="$VIDEOJEV_ROOT/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PATH="$VIDEOJEV_ROOT/.venv/bin:$PATH"
