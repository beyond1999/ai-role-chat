#!/usr/bin/env bash
set -euo pipefail


export CTRANSLATE2_CUDA_ALLOCATOR=cuda_malloc_async
export CTRANSLATE2_NUM_THREADS=$(nproc)
# 只用 WSL 的 12.x 运行时，避免 /usr/local/cuda 污染
exec env -i \
  PATH=/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu \
  CUDA_VISIBLE_DEVICES=0 \
  CTRANSLATE2_CUDA_ALLOCATOR=cuda_malloc_async \
  CTRANSLATE2_NUM_THREADS=$(nproc) \
  CT2_CUDA_TRUE_FP16_GEMM=0 \
  python3 backend/asr.py
