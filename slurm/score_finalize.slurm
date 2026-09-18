#!/bin/bash
# Finalize an eval run: score the predictions with the official OmniDocBench scorer (CPU) and measure
# throughput / tokens-per-second, writing score.json + speed.json into the run folder. Submitted by
# scripts/eval.sh with --dependency=afterok on the inference job; standalone:
#   DOCR_ROOT=$PWD sbatch --export=ALL,RD=<run dir>,GT=<gt json>,SCALE=full|200,INFER_LOG=<infer .out> slurm/score_finalize.slurm
# Requires the OmniDocBench repository checked out at $DOCR_ROOT/scorer with its venv at scorer/.venv-score
# (see docs/04_EVAL.md).
#SBATCH --job-name=odb_finalize
#SBATCH --partition=p0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval/finalize_%j.out
#SBATCH --error=logs/eval/finalize_%j.err
set -euo pipefail
ROOT="${DOCR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export SRC=$ROOT/src RUNS_ROOT=$ROOT/runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false

RD="${RD:?set RD=<run dir>}"; GT="${GT:?set GT}"; SCALE="${SCALE:-full}"; INFER_LOG="${INFER_LOG:-}"
PREDS="$RD/predictions"
RUNNAME="$(basename "$RD")"
SCORER="${SCORER:-$ROOT/scorer}"

[ -d "$PREDS" ] && [ -n "$(ls -A "$PREDS"/*.md 2>/dev/null)" ] || { echo "ERROR: no predictions in $PREDS (inference produced nothing — check IMAGES)"; exit 1; }
NPRED=$(ls "$PREDS"/*.md | wc -l); echo "[finalize] $NPRED prediction pages in $PREDS"

# --- 1) SCORE: unique pred symlink so the scorer result prefix == run name (no collisions) ---
mkdir -p "$SCORER/runs_preds"
ln -sfn "$PREDS" "$SCORER/runs_preds/$RUNNAME"
# The TEDS matcher is very slow on the full 1651-page set -> TEDS only on the 200-page subset here.
TABLE_METRIC='[Edit_dist]'; [ "$SCALE" = "200" ] && TABLE_METRIC='[TEDS, Edit_dist]'
YAML="$RD/score_config.yaml"
cat > "$YAML" <<YML
end2end_eval:
  metrics:
    text_block: {metric: [Edit_dist]}
    display_formula: {metric: [Edit_dist]}
    table: {metric: ${TABLE_METRIC}}
    reading_order: {metric: [Edit_dist]}
  dataset:
    dataset_name: end2end_dataset
    ground_truth: {data_path: ${GT}}
    prediction: {data_path: ${SCORER}/runs_preds/${RUNNAME}}
    match_method: quick_match
YML
echo "[finalize] scoring (this is the slow CPU step)..."
( cd "$SCORER" && .venv-score/bin/python pdf_validation.py --config "$YAML" )

# --- 2) parse scorer result -> score.json (overall + 4 categories + TEDS) ---
PYTHONPATH="$SRC" python3 - "$RD" "$SCORER" "$RUNNAME" <<'PY'
import json, os, sys
from run_output import write_score
rd, scorer, runname = sys.argv[1:4]
rf = os.path.join(scorer, "result", f"{runname}_quick_match_metric_result.json")
CATS = ["text_block", "display_formula", "table", "reading_order"]
out = {}
if os.path.exists(rf):
    d = json.load(open(rf))
    vals = []
    for c in CATS:
        try:
            v = round(float(d[c]["all"]["Edit_dist"]["ALL_page_avg"]), 4)
        except Exception:
            v = None
        out[c] = v
        if v is not None: vals.append(v)
    out["overall"] = round(sum(vals)/len(vals), 4) if len(vals) == 4 else None
    tf = os.path.join(scorer, "result", f"{runname}_quick_match_table_per_table_TEDS.json")
    if os.path.exists(tf):
        td = json.load(open(tf))
        ts = [x.get("TEDS") for x in (td.values() if isinstance(td, dict) else td) if x.get("TEDS") is not None]
        out["table_TEDS"] = round(sum(ts)/len(ts), 4) if ts else None
    json.dump(d, open(os.path.join(rd, "score_raw.json"), "w"), indent=2)   # raw scorer json for provenance
    write_score(rd, out)
    print("[finalize] score.json:", json.dumps(out))
else:
    print("[finalize] WARN: scorer result json not found:", rf)
PY

# --- 3) PERF: throughput (pages/s) + TPS (tok/s) -> speed.json ---
WALL=0
[ -n "$INFER_LOG" ] && [ -f "$INFER_LOG" ] && WALL=$(grep -oE 'WALL_SECONDS=[0-9]+' "$INFER_LOG" | tail -1 | cut -d= -f2 || true)
echo "[finalize] perf (wall=${WALL:-0}s from $INFER_LOG)..."
CKPT_TOK="$(python3 -c "import json;print(json.load(open('$RD/config.json')).get('ckpt',''))" 2>/dev/null || true)"
TOK_FLAG=""; [ -n "$CKPT_TOK" ] && [ -e "$CKPT_TOK" ] && TOK_FLAG="--tokenizer $CKPT_TOK"
PERF_JSON="$(cd "$ROOT" && uv run python src/measure_eval_perf.py --pred_dir "$PREDS" --elapsed_s "${WALL:-0}" --tag "$RUNNAME" $TOK_FLAG 2>/dev/null | sed -n 's/^\[EVAL PERF\] //p' | tail -1 || true)"
PYTHONPATH="$SRC" python3 - "$RD" "$PERF_JSON" <<'PY'
import json, sys, os
from run_output import write_speed
rd = sys.argv[1]; raw = sys.argv[2] if len(sys.argv) > 2 else ""
try:
    p = json.loads(raw) if raw else {}
except Exception:
    p = {}
spd = {"throughput_pg_s": p.get("throughput_pg_s"), "tps_tok_s": p.get("tps_tok_s"),
       "pages": p.get("pages"), "out_tokens": p.get("out_tokens"), "wall_s": p.get("elapsed_s")}
write_speed(rd, {k: v for k, v in spd.items() if v is not None})
print("[finalize] speed.json:", json.dumps(spd))
PY

# --- 4) status file ---
{
  echo "run:    $RUNNAME"
  echo "scale:  $SCALE"
  echo "preds:  $NPRED pages"
  echo "score:  $(cat "$RD/score.json" 2>/dev/null || echo MISSING)"
  echo "speed:  $(cat "$RD/speed.json" 2>/dev/null || echo MISSING)"
  echo "(tok/fwd from bench_forwards lands in speed.json separately if the bench job ran)"
} > "$RD/STATUS.txt"
echo "[finalize] done -> $RD/STATUS.txt"
