#!/usr/bin/env python3
"""FAIR, cuda-synced single-stream tps (bs=1) — apples-to-apples ours(spec) vs AR(HF .generate) vs base.

Why this exists: bench_forwards tps is NOT cuda-synced (rough). measure_tps_throughput is repointed to
the OLD repo. This one: warmup + torch.cuda.synchronize around the timed loop, SAME val pages, SAME
max_new_tokens, and BOTH paths use the fast-vision patch (our GlmOcrBlockDiffusion auto-patches in
__init__; HF AR gets infer_odb_ar.patch_vision_fast) -> the only difference is the decode path itself.

Modes: spec (our generate_speculative_natcache on GlmOcrBlockDiffusion) | ar (native HF .generate via
GlmOcrForConditionalGeneration). Appends results/speed_tps_fair.csv: model,decode_mode,step,n,tps_bs1,cuda_synced.

  CKPT=<ckpt dir or zai-org/GLM-OCR> LABEL=arjit STEP=10000 MODES=spec,ar N=40 python tools/measure_tps_fair.py
"""
import os, sys, time, csv, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from datasets import load_from_disk
from transformers import GlmOcrForConditionalGeneration, AutoProcessor
from models.glm_ocr_dllm import GlmOcrBlockDiffusion
import fast_sampling
import infer_odb_ar as arlib

ROOT = "${DOCR_ROOT}"
CKPT = os.environ["CKPT"]; LABEL = os.environ.get("LABEL", "model"); STEP = os.environ.get("STEP", "")
MODES = os.environ.get("MODES", "spec,ar").split(","); N = int(os.environ.get("N", "40"))
OUT = os.environ.get("OUT", f"{ROOT}/results/speed_tps_fair.csv")
dev = "cuda"
ds = load_from_disk(os.environ["VAL_PAGES"], keep_in_memory=False)   # same crop dataset as bench_forwards
random.seed(0); idxs = sorted(random.sample(range(ds.num_rows), min(N, ds.num_rows)))
def mnt_of(ex): return 256 if ex["task_type"] == "text" else (768 if ex["task_type"] == "table" else 256)


def time_loop(genfn):
    # warmup (2) then cuda-synced timed loop over N pages; tps = total decoded tokens / wall
    for w in idxs[:2]:
        genfn(ds[w])
    torch.cuda.synchronize(); t0 = time.perf_counter(); toks = 0
    for i in idxs:
        txt = genfn(ds[i]); toks += len(tok_fn(txt))
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    return round(toks / max(1e-9, wall), 1), toks, round(wall, 1)


rows = []
if "spec" in MODES:
    bd = json.load(open(os.path.join(CKPT, "block_diffusion.json"))).get("bd_size", 32)
    m = GlmOcrBlockDiffusion(model_id=CKPT, bd_size=bd, dtype=torch.bfloat16).to(dev).eval()  # auto fast-vision
    tok_fn = lambda t: m.tokenizer(t, add_special_tokens=False).input_ids
    with torch.no_grad():
        tps, toks, wall = time_loop(lambda ex: fast_sampling.generate_speculative_natcache(
            m, ex["image"], prompt=ex.get("prompt", "Text Recognition:"), max_new_tokens=mnt_of(ex), draft_steps=1)[0])
    print(f"[fair] {LABEL} spec(natcache)  tps_bs1={tps}  ({toks}tok/{wall}s, n={N}, cuda-synced)")
    rows.append([LABEL, "spec/AR-greedy", STEP, N, tps, "yes"]); del m; torch.cuda.empty_cache()
if "ar" in MODES:
    proc = AutoProcessor.from_pretrained(CKPT)
    glm = GlmOcrForConditionalGeneration.from_pretrained(CKPT, dtype=torch.bfloat16).to(dev).eval()
    arlib.patch_vision_fast(glm)
    tok_fn = lambda t: proc.tokenizer(t, add_special_tokens=False).input_ids
    with torch.no_grad():
        tps, toks, wall = time_loop(lambda ex: arlib.generate_ar(
            proc, glm, ex["image"], ex.get("prompt", "Text Recognition:"), mnt_of(ex), dev, 99999))
    print(f"[fair] {LABEL} ar(HF.generate)  tps_bs1={tps}  ({toks}tok/{wall}s, n={N}, cuda-synced)")
    rows.append([LABEL, "ar-native-HF", STEP, N, tps, "yes"])

new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["model", "decode_mode", "step", "n", "tps_bs1", "cuda_synced"])
    for r in rows: w.writerow(r)
print("wrote", len(rows), "rows ->", OUT)
