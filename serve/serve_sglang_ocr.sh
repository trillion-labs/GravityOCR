#!/bin/bash
# One-command SGLang HTTP server for a GLM-OCR block-diffusion checkpoint — AR or our self-spec.
# Bakes in EVERY gotcha this project hit (see serve/SGLANG_SERVE.md). Another session can just run this.
#
#   bash serve/serve_sglang_ocr.sh MODE=spec GPU=0 PORT=30000          # our block-diffusion self-spec (lossless)
#   bash serve/serve_sglang_ocr.sh MODE=ar   GPU=1 PORT=30001          # plain AR baseline
#   (optional) CKPT=<hf checkpoint dir or hub id>   (default = trillionlabs/GravityOCR)
#
# Verify:  curl -s http://127.0.0.1:$PORT/v1/models
# Then hit /v1/chat/completions with {image_url(base64), text} — see serve/SGLANG_SERVE.md for the eval client.
set -u
for kv in "$@"; do export "$kv"; done          # MODE=.. GPU=.. PORT=.. CKPT=..
MODE="${MODE:-spec}"; GPU="${GPU:-0}"; PORT="${PORT:-30000}"
# ROOT derived from this script's location (serve/..) so a MOVED tree still works.
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CKPT="${CKPT:-trillionlabs/GravityOCR}"
# VBIN may be overridden to launch the same server from a different venv
# (e.g. to measure rollout speed under another framework environment with the same client).
VBIN="${VBIN:?set VBIN=<patched sglang venv>/bin}"
# env-specific bits (override on a different server): CUDA toolkit + ninja dir on PATH.
CUDA_HOME_DIR="${CUDA_HOME:-/usr/local/cuda-12.8}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# --- GOTCHAS (all required; each cost this project a debugging cycle) ---
# 1. venv bin ON PATH  -> flashinfer cuda-graph capture shells out to `ninja` (lives in VBIN); without it,
#    capture dies "FileNotFoundError: ninja". Also USE $VBIN/python directly (NOT `uv run`, which picks the
#    wrong nested venv sglang/python/.venv that lacks deps).
# 2. VLLM_USE_FLASHINFER_SAMPLER=0 -> else the FlashInfer sampler JIT needs ninja/config and crashes.
# 3. CUDA_HOME=cuda-12.8 -> nvcc for JIT (driver is 570/cu12.8; everything is cu129).
# 4. HF_HUB_OFFLINE=1 -> model is cached; 8 concurrent loads otherwise throttle on unauthenticated HF Hub.
export PATH=$VBIN:$CUDA_HOME_DIR/bin:$PATH   # $VBIN has ninja; ensure a `ninja` is reachable on any server
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=$CUDA_HOME_DIR HF_HUB_OFFLINE=1 HF_HOME=$HF_CACHE

MAXREQ_ARG=""; [ -n "${MAXREQ:-}" ] && MAXREQ_ARG="--max-running-requests $MAXREQ"  # raise batching cap (spec defaults to 48)
# GRAPHBS (opt-in): --cuda-graph-max-bs. Unset = SGLang default (256).
#   Why: in spec mode a concurrency sweep at MAXREQ=256 dies during cuda-graph capture with a
#   flashinfer workspace overflow (the draft window K=33 needs larger prefill scratch than AR).
#   AR is fine at 256. Lowering only the capture batch keeps the running concurrency (max-running-requests).
GRAPHBS_ARG=""; [ -n "${GRAPHBS:-}" ] && GRAPHBS_ARG="--cuda-graph-max-bs $GRAPHBS"
SPEC_ARGS=""
if [ "$MODE" = "spec" ]; then
  # our block-diffusion self-spec — needs the SELFSPEC_DIFFUSION CLI choice patched into server_args.py
  # (patches/sglang, applied to the SGLang checkout $VBIN was built from) + page_size 1. Lossless == AR-greedy.
  SPEC_ARGS="--speculative-algorithm SELFSPEC_DIFFUSION --speculative-num-draft-tokens ${DRAFT:-33} --page-size 1"
fi

echo "[serve] MODE=$MODE GPU=$GPU PORT=$PORT CKPT=$CKPT"
CUDA_VISIBLE_DEVICES=$GPU exec $VBIN/python -m sglang.launch_server \
  --model-path "$CKPT" --served-model-name glm-ocr --host 127.0.0.1 --port "$PORT" \
  --attention-backend flashinfer --mem-fraction-static "${MEMFRAC:-0.6}" --context-length "${CONTEXT:-8192}" \
  --trust-remote-code $MAXREQ_ARG $GRAPHBS_ARG $SPEC_ARGS
