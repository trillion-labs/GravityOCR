#!/usr/bin/env bash
#SBATCH --job-name=sgl_glmocr
#SBATCH --partition=p0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G           # serving job: cap RAM (else slurm reserves whole-node 1.9TB)
#SBATCH --time=4:00:00
#SBATCH --output=serve/sgl_%j.log
# Official GLM-OCR on SGLang — base model (NOT our block-diffusion), to answer "is MTP(=NEXTN) slower than
# AR with the OFFICIAL code, on an optimized (cudagraph) engine?" — the fair test vLLM couldn't give.
#   SPEC=nextn PORT=8090 sbatch serve/sglang_serve.slurm   # official MTP/NEXTN spec (README recipe)
#   SPEC=ar    PORT=8091 sbatch serve/sglang_serve.slurm   # plain AR baseline, same engine
# Then measure BOTH with the SAME client (identical HTTP boundary -> directly comparable, docs/DECODE_PATHS.md):
#   serve/.venv... python serve/vllm_throughput.py --port 8090   # and --port 8091
# NOTE: uses the self-spec-PATCHED sglang clone (glm_ocr.py +2 lines) -> official number carries a small asterisk.
set -euo pipefail
SGL="${SGLANG_SRC:?set SGLANG_SRC=<patched sglang checkout>}"
VENV="${SGLANG_VENV:-$SGL/.venv-sglang}"
PORT="${PORT:-8090}"; SPEC="${SPEC:-nextn}"
export PATH="$VENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "[sgl_serve] node=$(hostname) job=$SLURM_JOB_ID SPEC=$SPEC PORT=$PORT $(date)"
COMMON="--model-path zai-org/GLM-OCR --port $PORT --served-model-name glm-ocr --mem-fraction-static 0.8 --host 0.0.0.0"
if [[ "$SPEC" == "nextn" ]]; then
  export SGLANG_ENABLE_SPEC_V2=1
  exec python -m sglang.launch_server $COMMON \
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
else
  exec python -m sglang.launch_server $COMMON
fi
