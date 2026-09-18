#!/bin/bash
# ONE-COMMAND OmniDocBench eval. Produces a single self-contained runs/ folder containing:
#   config.json + launch/ (reproducible: command, args, slurm copy)
#   predictions/*.md
#   score.json   (overall + text_block/display_formula/table/reading_order + table_TEDS)
#   speed.json   (throughput_pg_s, tps_tok_s, tok_per_fwd)
# It chains 3 slurm jobs:  inference (GPU) -> finalize=score+perf (CPU, afterok) [+ tok/fwd bench (GPU)].
#
#   bash scripts/eval.sh MODEL=checkpoints/<run>/hf_step_40000 SCALE=full NAME=blockdiff
#   bash scripts/eval.sh MODE=ar MODEL=zai-org/GLM-OCR SCALE=full NAME=base_ar
#
# Required: MODEL.   Key options:
#   MODE=diffusion|ar (default diffusion)   SCALE=full|200 (default full = the benchmark)
#   THR=0.99   BS=8   LIMIT=0   NAME=<label>   STEP=<n>   TOKFWD=1|0 (default 1; needs VAL_PAGES)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/_lib.sh"
require_compute_node
cd "$ROOT"
for kv in "$@"; do export "$kv"; done
export RUN_CMD="scripts/eval.sh $*"   # real invocation -> config.json cmd (reproducibility)

MODE="${MODE:-diffusion}"
SCALE="${SCALE:-full}"
MODEL="${MODEL:?set MODEL=<hf checkpoint dir, or zai-org/GLM-OCR for MODE=ar>}"
THR="${THR:-0.99}"; BS="${BS:-8}"; LIMIT="${LIMIT:-0}"
case "$MODE" in diffusion|ar) ;; *) echo "ERROR: MODE must be diffusion|ar (got $MODE)"; exit 2 ;; esac

read -r GT EVALSET < <(gt_for_scale "$SCALE")
IMAGES="${IMAGES:-$ROOT/omnidocbench/images}"
# labels
NAME="${NAME:-$(basename "$(dirname "$MODEL")")}"
STEP="${STEP:-$(echo "$MODEL" | grep -oE 'step_[0-9]+|final' | tail -1 | tr -d 'step_')}"
TOKFWD="${TOKFWD:-1}"

echo "=== EVAL  MODE=$MODE  SCALE=$SCALE ($EVALSET)  THR(confidence)=$THR  BS=$BS  LIMIT=$LIMIT ==="
echo "    MODEL=$MODEL   NAME=$NAME  STEP=${STEP:-?}"
echo "    GT=$GT"
[ "$SCALE" = full ] && echo "    (FULL 1651-page benchmark — the headline number)"

RD="$(mkrun "$NAME" "${STEP:-}" "$MODE" "$THR" "$EVALSET" "$MODEL" "$GT")"
PREDS="$RD/predictions"
echo "    run folder: $RD"

# ===== never re-run an eval that is already done =====
# The run-folder name is deterministic for (model,step,mode,thr,scale). If it already has a complete
# score.json + speed.json, report and STOP. Use FORCE=1 to deliberately re-run.
score_complete() { python3 -c "import json,sys;d=json.load(open('$RD/score.json'));sys.exit(0 if d.get('overall') not in (None,'') else 1)" 2>/dev/null; }
have_preds()      { ls "$PREDS"/*.md >/dev/null 2>&1; }
if [ "${FORCE:-0}" != "1" ] && score_complete && [ -f "$RD/speed.json" ]; then
  echo "==> ALREADY DONE — results exist; NOT re-running (pass FORCE=1 to override):"
  cat "$RD/score.json" "$RD/speed.json"
  exit 0
fi

# ---- 1) inference (GPU, page-sharded) — SKIP if predictions already exist (only re-score) ----
IJOB=""; INFER_LOG=""
if [ "${FORCE:-0}" != "1" ] && have_preds; then
  echo "==> inference already done ($(ls "$PREDS"/*.md | wc -l) pages) — skipping GPU inference, only (re)scoring."
else
  # The page-sharded inference is embarrassingly parallel (independent processes, NO NCCL):
  # num_shards = SLURM_NTASKS, one GPU per task. The slurm files default to 2 nodes x 8 GPU = 16
  # shards; a single node runs the SAME eval with fewer shards:  NODES=1 GPN=8 bash scripts/eval.sh ...
  SBO=""
  [ -n "${NODES:-}" ]    && SBO="$SBO --nodes=$NODES"
  [ -n "${GPN:-}" ]      && SBO="$SBO --gpus-per-node=$GPN --ntasks-per-node=$GPN"
  [ -n "${NODELIST:-}" ] && SBO="$SBO --nodelist=$NODELIST"
  [ -n "$SBO" ] && echo "    (sbatch override:$SBO )"
  case "$MODE" in
    diffusion)
      SLURM="${EVAL_SLURM:-$ROOT/slurm/eval_omnidocbench.slurm}"
      IOUT="$(sbatch --parsable $SBO --export=ALL,DOCR_ROOT="$ROOT",CKPT="$MODEL",GT="$GT",IMAGES="$IMAGES",OUT="$PREDS",LONGSIDE=99999,FAST=1,BS="$BS",LIMIT="$LIMIT",THRESHOLD="$THR" "$SLURM")" ;;
    ar)
      SLURM="$ROOT/slurm/eval_ar.slurm"
      IOUT="$(sbatch --parsable $SBO --export=ALL,DOCR_ROOT="$ROOT",MODEL="$MODEL",GT="$GT",IMAGES="$IMAGES",OUT="$PREDS",BS="$BS",LIMIT="$LIMIT" "$SLURM")" ;;
  esac
  IJOB="${IOUT%%;*}"
  INFER_LOG="$ROOT/logs/eval/$([ "$MODE" = ar ] && echo odbar || echo odb)_${IJOB}.out"
  echo "    inference jobid=$IJOB  log: $INFER_LOG"
  snapshot_launch "$RD" "$IJOB" "${SLURM:-}" "scripts/eval.sh $*"
fi

# ---- 2) finalize = score + perf (CPU; afterok inference if it ran, else immediately) ----
FSLURM="$ROOT/slurm/score_finalize.slurm"
DEP=""; [ -n "$IJOB" ] && DEP="--dependency=afterok:$IJOB"
FOUT="$(sbatch --parsable $DEP \
  --export=ALL,DOCR_ROOT="$ROOT",RD="$RD",GT="$GT",SCALE="$SCALE",INFER_LOG="$INFER_LOG" "$FSLURM")"
FJOB="${FOUT%%;*}"
echo "    finalize jobid=$FJOB  (${IJOB:+afterok:$IJOB }-> writes score.json + speed.json)"
snapshot_launch "$RD" "$FJOB" "$FSLURM" "finalize (score+perf)"

# ---- 3) tok/fwd + accept/round bench (GPU). Runs for any block-diffusion checkpoint
#         (block_diffusion.json) — AR-head checkpoints get spec accept/round + tok/fwd, pure
#         diffusion checkpoints get plain-diffusion tok/fwd. Skipped for base-AR models and when
#         already measured. Needs VAL_PAGES (a save_to_disk dataset of region crops). ----
if [ -f "$MODEL/block_diffusion.json" ] && [ "$TOKFWD" = 1 ]; then
  if [ -z "${VAL_PAGES:-}" ]; then
    echo "    tok/fwd bench skipped: set VAL_PAGES=<crop dataset> to enable"
  elif [ "${FORCE:-0}" != "1" ] && python3 -c "import json,sys;d=json.load(open('$RD/speed.json'));sys.exit(0 if d.get('tok_per_fwd') and d.get('accept_per_round') else 1)" 2>/dev/null; then
    echo "    tok/fwd + accept/round already measured — skipping bench"
  else
    BSLURM="$ROOT/slurm/bench_forwards.slurm"
    BOUT="$(sbatch --parsable --export=ALL,DOCR_ROOT="$ROOT",VAL_PAGES="$VAL_PAGES",CKPT="$MODEL",N=100,ACCEPT_CONF=0,OUT="$RD/tokfwd_bench.json",RD="$RD" "$BSLURM" || true)"
    echo "    tok/fwd+accept bench jobid=${BOUT%%;*}"
  fi
fi

log_index "${IJOB:-reuse} eval $RD mode=$MODE scale=$SCALE thr=$THR (finalize=$FJOB)"
echo "When all finish:  cat $RD/score.json $RD/speed.json"
