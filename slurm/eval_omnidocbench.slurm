#!/bin/bash
# OmniDocBench inference, multi-GPU page-sharded (default 16 GPUs = 2 nodes × 8). Each task is one
# independent single-GPU process OCRing a disjoint page subset; all write the same --out_dir. No NCCL
# (embarrassingly parallel); output identical to a 1-GPU run. Normally submitted by scripts/eval.sh.
#   DOCR_ROOT=$PWD sbatch --export=ALL,CKPT=<hf ckpt dir>,GT=<OmniDocBench.json>,IMAGES=<images dir>,OUT=<pred dir> \
#          slurm/eval_omnidocbench.slurm
#SBATCH --job-name=odb_eval
#SBATCH --partition=p0
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8          # 8 tasks/node × 2 = 16 shards; one GPU per task
#SBATCH --cpus-per-task=10
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval/odb_%j.out
#SBATCH --error=logs/eval/odb_%j.err

set -euo pipefail
ROOT="${DOCR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; PROJECT=$ROOT; export PYTHONPATH=$ROOT/src RUNS_ROOT=$ROOT/runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:?set CKPT=<hf_ checkpoint dir>}"
GT="${GT:?set GT=<OmniDocBench GT json>}"
OUT="${OUT:?set OUT=<prediction out_dir>}"
# REQUIRED: without --images_dir the GT image names resolve relative to CWD and every page is "missing".
IMAGES="${IMAGES:?set IMAGES=<dir holding the GT-referenced page images>}"
FAST="${FAST:-1}"; BS="${BS:-8}"; LIMIT="${LIMIT:-0}"
# Full resolution. The inference default (768) downscales pages and wrecks OCR — never evaluate below this.
LONGSIDE="${LONGSIDE:-99999}"
THRESHOLD="${THRESHOLD:-0.99}"
cd "$PROJECT"; mkdir -p logs/eval "$OUT"

FAST_FLAG=""; [ "$FAST" = "1" ] && FAST_FLAG="--fast --batch_size $BS"
# SPEC=natcache|prefixcache|uncached → self-speculative decode (== AR-greedy output, fewer forwards).
# Uses the bs=1 per-crop sampler, so it overrides --fast batching.
SPEC_FLAG=""
case "${SPEC:-}" in
  natcache)    SPEC_FLAG="--spec_natcache";  FAST_FLAG="" ;;
  prefixcache) SPEC_FLAG="--spec";           FAST_FLAG="" ;;
  uncached)    SPEC_FLAG="--spec_uncached";  FAST_FLAG="" ;;
esac
LIMIT_FLAG=""; [ "$LIMIT" != "0" ] && LIMIT_FLAG="--limit $LIMIT"
# LoopMDM checkpoints: LOOP_INFER_S=<int> fixes the mid-block loop count at inference.
LOOP_FLAG=""; [ -n "${LOOP_INFER_S:-}" ] && LOOP_FLAG="--loop_infer_S $LOOP_INFER_S"

# one independent process per GPU; shard_id = global task rank, num_shards = total tasks. No NCCL.
T0=$(date +%s)
srun --cpu-bind=cores bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID            # 1 GPU per task (local 0..7)
  cd '"$PROJECT"'
  export HF_HOME='"$HF_HOME"' TOKENIZERS_PARALLELISM=false PYTHONPATH='"$ROOT"'/src
  echo "[task $SLURM_PROCID/$SLURM_NTASKS] host=$(hostname) localid=$SLURM_LOCALID CUDA=$CUDA_VISIBLE_DEVICES"
  uv run python src/infer_omnidocbench.py \
    --checkpoint '"$CKPT"' --omnidocbench_gt '"$GT"' --images_dir '"$IMAGES"' --out_dir '"$OUT"' \
    --max_long_side '"$LONGSIDE"' --threshold '"$THRESHOLD"' --num_shards $SLURM_NTASKS --shard_id $SLURM_PROCID \
    '"$FAST_FLAG"' '"$SPEC_FLAG"' '"$LIMIT_FLAG"' '"$LOOP_FLAG"'
'
echo "[done] all shards finished -> $OUT  (score with the OmniDocBench end2end scorer on this dir)"
echo "[wall] WALL_SECONDS=$(($(date +%s)-T0)) OUT=$OUT"
