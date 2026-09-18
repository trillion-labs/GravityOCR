#!/bin/bash
# OmniDocBench inference with the plain autoregressive path (src/infer_odb_ar.py: native HF greedy
# generate), page-sharded over 16 GPUs like eval_omnidocbench.slurm. Same layout-detect + assemble
# pipeline, so scores are directly comparable to the diffusion / self-spec runs. Submitted by
# scripts/eval.sh MODE=ar; direct use:
#   DOCR_ROOT=$PWD sbatch --export=ALL,MODEL=<hf ckpt dir or zai-org/GLM-OCR>,OUT=<pred dir>,GT=<OmniDocBench.json>,IMAGES=<images> slurm/eval_ar.slurm
#SBATCH --job-name=odb_ar
#SBATCH --partition=p0
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval/odbar_%j.out
#SBATCH --error=logs/eval/odbar_%j.err
set -euo pipefail
ROOT="${DOCR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; PROJECT=$ROOT; export PYTHONPATH=$ROOT/src RUNS_ROOT=$ROOT/runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
MODEL="${MODEL:?set MODEL=<hf dir or zai-org/GLM-OCR>}"
OUT="${OUT:?set OUT=<prediction out_dir>}"
GT="${GT:-$ROOT/omnidocbench/OmniDocBench.json}"
IMAGES="${IMAGES:-$ROOT/omnidocbench/images}"
LIMIT="${LIMIT:-0}"; BS="${BS:-1}"   # BS>1 batches region crops per HF generate (validate vs bs=1 first)
cd "$PROJECT"; mkdir -p logs/eval "$OUT"
LIMIT_FLAG=""; [ "$LIMIT" != "0" ] && LIMIT_FLAG="--limit $LIMIT"
echo "[eval_ar] MODEL=$MODEL OUT=$OUT GT=$GT LIMIT=$LIMIT BS=$BS"
T0=$(date +%s)
srun --cpu-bind=cores bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  cd '"$PROJECT"'
  export HF_HOME='"$HF_HOME"' TOKENIZERS_PARALLELISM=false PYTHONPATH='"$ROOT"'/src
  echo "[task $SLURM_PROCID/$SLURM_NTASKS] host=$(hostname) localid=$SLURM_LOCALID"
  uv run python src/infer_odb_ar.py \
    --model_id '"$MODEL"' --omnidocbench_gt '"$GT"' --images_dir '"$IMAGES"' --out_dir '"$OUT"' \
    --max_long_side 99999 --batch_size '"$BS"' --num_shards $SLURM_NTASKS --shard_id $SLURM_PROCID '"$LIMIT_FLAG"'
'
echo "[done] all shards finished -> $OUT"
echo "[wall] WALL_SECONDS=$(($(date +%s)-T0))"
