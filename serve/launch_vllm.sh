#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the BASE zai-org/GLM-OCR model.
# Serves the raw OCR vision-language model (NOT the block-diffusion conversion).
# Endpoint: http://<node>:8080/v1/chat/completions   served-model-name: glm-ocr
#
# Run this ON A COMPUTE NODE (gpu-node-0/001). 0.9B model -> TP=1, 1 GPU.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export TOKENIZERS_PARALLELISM=false
# FlashInfer's top-k/top-p sampler JIT-compiles at startup (needs ninja+nvcc) and
# was the cause of an early EngineCore crash. Use vLLM's native Torch sampler instead
# (quality-identical; OCR eval runs greedy temperature=0 anyway).
export VLLM_USE_FLASHINFER_SAMPLER=0

VENV="${VENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv-vllm}"
PORT="${PORT:-8080}"
GPU="${GPU:-0}"                       # which local GPU (ignored under slurm)
MAXLEN="${MAXLEN:-32768}"             # raise for very large pages
GPUUTIL="${GPUUTIL:-0.90}"

# Under slurm, the allocation already masks GPUs via CUDA_VISIBLE_DEVICES — don't override.
# Standalone (no slurm), pin to the requested local GPU.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Speculative MTP decoding (lossless speedup; model has num_nextn_predict_layers=1).
# Set SPEC=0 to disable if it fails to load.
SPEC_ARGS=""
if [[ "${SPEC:-1}" == "1" ]]; then
  SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC:-3}}"
fi

exec "$VENV/bin/vllm" serve zai-org/GLM-OCR \
  --served-model-name glm-ocr \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPUUTIL" \
  --trust-remote-code \
  $SPEC_ARGS
