#!/usr/bin/env python3
"""Standardised run-output layout. Inference / measurement scripts call this so EVERY run drops one
self-contained folder you can open later and find everything — instead of scattering outputs.

Layout:  code/runs/<model>__s<step>__<mode>__thr<thr>__<evalset>/
    config.json     run metadata (model, ckpt, step, mode, threshold, eval_set, gt, cmd, date)
    predictions/    *.md  (OmniDocBench prediction dir — point pdf_validation here)
    score.json      {overall, text_block, display_formula, table, reading_order, table_TEDS}
    speed.json      {tok_per_fwd, accept_per_round, tps_bs1, throughput_tokps, throughput_bs}
    samples/        a few colored page renders (optional)

Every number a run produces lands in runs/<name>/score.json + speed.json — nothing is hand-typed. Usage:

    from run_output import make_run_dir, write_score, write_speed
    rd = make_run_dir(dict(model="gravityocr", step=500, mode="diffusion", thr=0.99, eval_set="seed-200p",
                           ckpt=ckpt_path, gt=gt_path))
    # write predictions into rd/"predictions"/<stem>.md  (infer loop)
    write_speed(rd, dict(tok_per_fwd=1.37))            # measure script
    write_score(rd, dict(overall=0.109, table_TEDS=0.88))  # after scoring
"""
import json, os, sys

RUNS_ROOT = os.environ.get(
    "RUNS_ROOT", "${DOCR_ROOT}/runs")

def run_name(meta):
    thr = meta.get("thr", "")
    thr_s = f"thr{thr}" if thr not in (None, "") else (f"tau{meta['tau']}" if meta.get("tau") else "na")
    return f"{meta['model']}__s{meta.get('step','')}__{meta['mode']}__{thr_s}__{meta.get('eval_set','seed-200p')}"

def make_run_dir(meta, root=RUNS_ROOT):
    """Create runs/<name>/{predictions,samples} and write config.json. Returns the run dir path."""
    rd = os.path.join(root, run_name(meta))
    os.makedirs(os.path.join(rd, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(rd, "samples"), exist_ok=True)
    cfg = dict(meta)
    cfg.setdefault("cmd", " ".join(sys.argv))
    _write(os.path.join(rd, "config.json"), cfg)
    return rd

def predictions_dir(run_dir):
    return os.path.join(run_dir, "predictions")

def write_score(run_dir, score):   _merge(os.path.join(run_dir, "score.json"), score)
def write_speed(run_dir, speed):   _merge(os.path.join(run_dir, "speed.json"), speed)

def _write(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def _merge(path, obj):
    cur = {}
    if os.path.exists(path):
        try: cur = json.load(open(path))
        except Exception: pass
    cur.update({k: v for k, v in obj.items() if v not in (None, "")})
    _write(path, cur)
