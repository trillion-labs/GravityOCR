#!/usr/bin/env python3
"""Measure eval THROUGHPUT (pages/s) + TPS (output tokens/s) for an OmniDocBench prediction dir.
RULE: run this after every eval inference (alongside the score) — always report score + TPS + throughput.

  python measure_eval_perf.py --pred_dir eval_out/curve_step2000 --elapsed_s 1234 [--tokenizer <hf_ckpt>]
  # elapsed_s = the eval INFERENCE wall-clock seconds (from the slurm job). If omitted, estimated
  # from the prediction dir's file-mtime span (rough; pass --elapsed_s for accuracy).
Prints one line: pages, out_tokens, elapsed_s, throughput(pg/s), TPS(tok/s).
"""
import argparse, glob, os, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--elapsed_s", type=float, default=0, help="eval inference wall seconds (from slurm)")
    ap.add_argument("--tokenizer", default="trillionlabs/GravityOCR",
                    help="hf checkpoint dir or hub id for the output-token count (TPS). If it cannot be loaded, falls back to chars/4.")
    ap.add_argument("--tag", default="", help="label for the log line")
    a = ap.parse_args()
    preds = sorted(glob.glob(os.path.join(a.pred_dir, "*.md")))
    pages = len(preds)
    texts = [open(p, encoding="utf-8", errors="ignore").read() for p in preds]
    # output tokens
    toks = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
        toks = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
    except Exception as e:
        toks = int(sum(len(t) for t in texts) / 4)   # rough fallback: ~4 chars/token
        print(f"[warn] tokenizer load failed ({str(e)[:60]}); TPS uses chars/4 estimate", flush=True)
    # elapsed
    el = a.elapsed_s
    if el <= 0 and preds:                              # estimate from file mtime span (rough)
        mt = [os.path.getmtime(p) for p in preds]; el = max(mt) - min(mt)
        print(f"[warn] --elapsed_s not given; estimated {el:.0f}s from mtime span (pass --elapsed_s for accuracy)", flush=True)
    el = max(el, 1e-9)
    res = {"tag": a.tag or os.path.basename(a.pred_dir.rstrip("/")), "pages": pages,
           "out_tokens": toks, "elapsed_s": round(el, 1),
           "throughput_pg_s": round(pages / el, 3), "tps_tok_s": round(toks / el, 1)}
    print("[EVAL PERF] " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
