#!/bin/bash
# Single-GPU tokens-per-forward / accept-per-round benchmark (tools/bench_forwards.py): self-speculative
# decode paths vs plain AR on a fixed page set. This is the canonical source of every tok/fwd number.
#   DOCR_ROOT=$PWD sbatch --export=ALL,CKPT=<hf ckpt dir>,N=100,OUT=runs/_bench/bench_forwards.json slurm/bench_forwards.slurm
#SBATCH --job-name=benchfwd
#SBATCH --partition=p0
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench/benchfwd_%j.out
#SBATCH --error=logs/bench/benchfwd_%j.err
set -euo pipefail
ROOT="${DOCR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; PROJECT=$ROOT; export PYTHONPATH=$ROOT/src RUNS_ROOT=$ROOT/runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT"; mkdir -p logs/bench
OUT="${OUT:-$ROOT/runs/_bench/bench_forwards_${SLURM_JOB_ID}.json}"; mkdir -p "$(dirname "$OUT")"
echo "[benchfwd] CKPT=$CKPT N=${N:-100} ACCEPT_CONF=${ACCEPT_CONF:-0} DRAFT_STEPS=${DRAFT_STEPS:-iterative} OUT=$OUT"
CKPT="$CKPT" N="${N:-100}" ACCEPT_CONF="${ACCEPT_CONF:-0}" DRAFT_STEPS="${DRAFT_STEPS:-}" OUT="$OUT" \
  uv run python tools/bench_forwards.py
echo "[benchfwd-done] wrote $OUT"

# when launched as part of an eval run (RD set), fold tok/fwd and accept/round into that run's speed.json
if [ -n "${RD:-}" ] && [ -f "$OUT" ]; then
  PYTHONPATH="$ROOT/src" python3 - "$RD" "$OUT" <<'PY'
import json, sys, os
from run_output import write_speed
rd, out = sys.argv[1:3]
agg = json.load(open(out))
by, acc = {}, {}
for method, a in agg.items():
    fwd = a.get("fwd", 0); toks = a.get("toks", 0); rnds = a.get("rounds", 0)
    if fwd: by[method] = round(toks / fwd, 3)
    if rnds: acc[method] = round(toks / rnds, 3)     # committed tokens / round = accept/round
# headline = the natcache spec if present, else any spec, else diffusion
def pick(d):
    return (next((m for m in d if "natcache" in m), None)
            or next((m for m in d if "spec" in m.lower()), None)
            or next((m for m in d if "diff" in m.lower()), None)
            or next(iter(d), None))
spd = {"tok_per_fwd_by_method": by, "accept_per_round_by_method": acc}
k = pick(by);  spd["tok_per_fwd"] = by[k] if k else None
ka = pick(acc); spd["accept_per_round"] = acc[ka] if ka else None
write_speed(rd, {x: y for x, y in spd.items() if y is not None})
print("[bench] folded tok/fwd + accept/round into", rd, "->", json.dumps(spd))
PY
fi
