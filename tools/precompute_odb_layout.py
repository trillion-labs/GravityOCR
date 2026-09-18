"""Precompute PP-DocLayout-V3 region detection for OmniDocBench ONCE and cache it, so future evals skip
layout entirely (deterministic per page). Reuses src/infer_omnidocbench.py's LayoutDetector + resolve_image_list
(NO parallel pipeline). Also measures layout speed once (constant, engine-agnostic) — record it in
docs/DECODE_PATHS.md.

Usage (on a compute node w/ a free GPU, PYTHONPATH=src):
  CUDA_VISIBLE_DEVICES=<idx> uv run python tools/precompute_odb_layout.py \
      --omnidocbench_gt omnidocbench/OmniDocBench.json --images_dir omnidocbench/images \
      --out runs/_layout_cache/odb_layout_regions.json

Then eval reuses it:  infer_omnidocbench.py ... --layout_cache runs/_layout_cache/odb_layout_regions.json
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from PIL import Image
from infer_omnidocbench import LayoutDetector, resolve_image_list

ap = argparse.ArgumentParser()
ap.add_argument("--omnidocbench_gt", required=True)
ap.add_argument("--images_dir", required=True)
ap.add_argument("--out", required=True, help="output JSON: {stem: [region dicts]}")
a = ap.parse_args()

class _Args: image = None
_Args.omnidocbench_gt = a.omnidocbench_gt; _Args.images_dir = a.images_dir
items = sorted(resolve_image_list(_Args), key=lambda x: x[1])
print(f"[precompute-layout] {len(items)} OBD pages", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
layout = LayoutDetector(dev)

# warmup (excluded from timing)
layout.detect(Image.open(items[0][0]).convert("RGB"))
if dev == "cuda": torch.cuda.synchronize()

cache = {}
t0 = time.time(); n = 0
for i, (img_path, stem) in enumerate(items):
    img = Image.open(img_path).convert("RGB")
    if dev == "cuda": torch.cuda.synchronize()
    regions = layout.detect(img)
    if dev == "cuda": torch.cuda.synchronize()
    cache[stem] = regions
    n += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(items)}  ({1000*(time.time()-t0)/n:.1f} ms/page so far)", flush=True)
wall = time.time() - t0

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(cache, open(a.out, "w"))
tot_regions = sum(len(v) for v in cache.values())
print(f"[precompute-layout] wrote {a.out}: {len(cache)} pages, {tot_regions} regions", flush=True)
print(f"[LAYOUT-SPEED] {wall:.1f}s for {n} pages = {1000*wall/n:.1f} ms/page mean "
      f"(cuda-synced, warmup excluded, single GPU) — CONSTANT, record in docs/DECODE_PATHS.md", flush=True)
