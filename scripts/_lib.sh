#!/bin/bash
# Shared helpers for the launchers. Sourced by eval.sh.
# Every run drops ONE self-contained folder under runs/ that records
#   (a) exactly how it was launched (command + args + a copy of the slurm script)  -> reproducible
#   (b) every result (score + tps + tok/fwd + throughput)                          -> never lost
# Nothing here is hand-edited later; the wrappers + finalize step fill the folder.
set -euo pipefail

export ROOT="${DOCR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SRC=$ROOT/src
export RUNS_ROOT=$ROOT/runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# OmniDocBench GT + images, by scale.  SCALE = full | 200
gt_for_scale() {  # echo "<gt_json> <evalset_label>"
  case "$1" in
    full) echo "$ROOT/omnidocbench/OmniDocBench.json full-1651" ;;
    200)  echo "${ODB_GT_200:?set ODB_GT_200=<200-page subset of OmniDocBench.json>} seed-200p" ;;
    *)    echo "ERROR: SCALE must be 'full' or '200' (got '$1')" >&2; exit 2 ;;
  esac
}

# make a run dir via src/run_output.py (stdlib-only; no uv needed). echoes the dir path.
# usage: mkrun model step mode thr evalset ckpt gt
mkrun() {
  PYTHONPATH="$SRC" RUNS_ROOT="$RUNS_ROOT" python3 - "$@" <<'PY'
import sys, os
from run_output import make_run_dir
m,step,mode,thr,evalset,ckpt,gt = sys.argv[1:8]
meta = dict(model=m, step=step, mode=mode, thr=thr, eval_set=evalset, ckpt=ckpt, gt=gt)
cmd = os.environ.get("RUN_CMD", "")
if cmd: meta["cmd"] = cmd   # the real wrapper invocation (else config.json cmd shows the mkrun heredoc argv)
print(make_run_dir(meta))
PY
}

# snapshot how a job was launched into <rundir>/launch/. Call right after sbatch.
# usage: snapshot_launch <rundir> <jobid> <slurm_file> <full command line>
snapshot_launch() {
  local rd="$1" jobid="$2" slurm="$3"; shift 3
  mkdir -p "$rd/launch"
  {
    echo "date:    $(date -Is)"
    echo "host:    $(hostname)"
    echo "jobid:   $jobid"
    echo "slurm:   $slurm"
    echo "command: $*"
  } >> "$rd/launch/launch.txt"
  [ -f "$slurm" ] && cp -p "$slurm" "$rd/launch/$(basename "$slurm")"
  # record the resolved config knobs that were exported into the environment
  ( set -o posix; set ) | grep -E '^(MODEL|MODEL_ID|CKPT|OUT|GT|IMAGES|SCALE|MODE|THR|THRESHOLD|STEP|NAME|BS|FAST|LIMIT|BD_SIZE|MAX_LENGTH|MAX_RESP_LENGTH|MAX_PACKED_ROWS|GRAD_ACCUM|WARMUP|MAX_STEPS|SAVE_EVERY|SAVE_DIR|DATASET|GRAD_CKPT|GC_MIN_LEN|LR|ENCODER_LR_MULT|RESUME|N|ACCEPT_CONF|DRAFT_STEPS)=' \
    > "$rd/launch/resolved_env.txt" 2>/dev/null || true
}

# append a one-line row to the global run index (at-a-glance history of every launch).
log_index() {  # usage: log_index "<jobid> <kind> <rundir-or-note> <extra>"
  mkdir -p "$RUNS_ROOT"
  echo "$(date -Is) | $*" >> "$RUNS_ROOT/INDEX.log"
}

require_compute_node() {
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Submit from a node that can reach the slurm controller." >&2
    exit 1
  fi
}
