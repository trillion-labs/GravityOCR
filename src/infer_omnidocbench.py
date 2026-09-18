"""End-to-end page OCR with the GLM-OCR Fast-dLLM v2 block-diffusion model,
producing OmniDocBench-ready predictions.

Flow (mirrors GLM-OCR's official pipeline, only the OCR step swapped for block diffusion):

    page image
      -> PP-DocLayout-V3 region detection (drop abandon, skip image/chart)
      -> per-region crop
      -> GlmOcrBlockDiffusion.generate(crop, task_prompt)        # Fast-dLLM v2 sampling
      -> per-region post-processing (ported from glmocr/result_formatter.py)
      -> merge regions in reading order  ->  page Markdown
      -> write <image_stem>.md into --out_dir

OmniDocBench reads predictions as one ``<image_stem>.md`` per GT page
(``_resolve_prediction_path``: ``img_name[:-4] + '.md'``), then matches GT elements
into the predicted markdown and scores text(Edit), formula(CDM), table(TEDS),
reading-order(Edit). So this script's --out_dir is a drop-in ``prediction.data_path``.

Examples
--------
# Whole OmniDocBench set (GT json + the page images it references):
python infer_omnidocbench.py \
    --checkpoint checkpoints/glm_ocr_dllm/hf_final \
    --omnidocbench_gt /path/OmniDocBench.json \
    --images_dir /path/OmniDocBench/images \
    --out_dir preds/glm_dllm

# Or just a folder of images:
python infer_omnidocbench.py --checkpoint <ckpt> --images_dir imgs/ --out_dir preds/

Then evaluate (in the OmniDocBench repo) by pointing the config's
``prediction.data_path`` at --out_dir and ``ground_truth.data_path`` at the GT json:
    python pdf_validation.py --config configs/end2end.yaml
(use --write_config here to emit that yaml automatically).
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from layout_postprocess import apply_layout_postprocess   # GLM-OCR-exact NMS/merge/order
# `from models.glm_ocr_dllm import GlmOcrBlockDiffusion` is done LAZILY inside the model-build path
# (see line with "import GlmOcrBlockDiffusion") so this module's layout/assembly helpers (LayoutDetector,
# TASK_PROMPTS, MAX_NEW_TOKENS, assemble_page) can be imported in an env WITHOUT the HF model deps —
# e.g. the SGLang cu129 crop-scorer (runners/sglang_odb_score.py).

Image.MAX_IMAGE_PIXELS = None
SPEC_STATS = []   # populated by make_sampler when --spec; one dict, summed across regions
DIFF_STATS = []   # populated by make_sampler diffusion path; forwards+tokens summed across regions

LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
# GLM-OCR layout config (glmocr/config.yaml): low threshold + NMS + containment merge.
LAYOUT_THRESHOLD = 0.3                    # GLM-OCR default (was 0.5; lower, cleaned up by NMS/merge)
LAYOUT_NMS = True                         # NMS iou_same=0.6 / iou_diff=0.98 (in layout_postprocess)
LAYOUT_UNCLIP_RATIO = (1.0, 1.0)          # GLM-OCR default (1.0 = no expansion)
LAYOUT_MERGE_MODE = "large"               # drop boxes contained in a larger one (kills nested sub-boxes)
MAX_LONG_SIDE = 768

# PP-DocLayout-V3 native label -> OCR task type (GLM-OCR config.yaml label_task_mapping).
# Covers ALL 21 labels PP-DocLayout-V3 actually emits (verified against model.config.id2label);
# kept identical to build_olmocr_test_dataset.py so eval and dataset-build route regions the same.
# NOTE: the model emits a single "formula" label (NOT display_formula/inline_formula) — earlier
# those were dead keys, so every formula region fell through .get(...,"abandon") and was dropped.
LABEL_TO_TASK = {
    "abstract": "text", "algorithm": "text", "aside_text": "text", "content": "text",
    "doc_title": "text", "figure_title": "text", "formula_number": "text",
    "paragraph_title": "text", "reference_content": "text", "seal": "text",
    "text": "text", "vision_footnote": "text",
    "table": "table",
    "formula": "formula",
    "image": "skip", "chart": "skip",
    "header": "abandon", "footer": "abandon", "footnote": "abandon",
    "number": "abandon", "reference": "abandon",
}

# native label -> merged "visualization" category (GLM-OCR label_visualization_mapping).
LABEL_TO_CATEGORY = {
    "table": "table",
    "formula": "formula",
    "chart": "image", "image": "image",
}  # everything else -> "text"

TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}
# formula was 512 -> long display equations (multi-line arrays) exceed it and get TRUNCATED,
# tanking formula Edit/CDM vs the official SDK (which uses page max_tokens 8192/4096). Match table.
MAX_NEW_TOKENS = {"text": 512, "table": 4096, "formula": 4096}


# ======================================================================
# Layout detection (PP-DocLayout-V3) — keeps the native label for formatting.
# ======================================================================
class LayoutDetector:
    def __init__(self, device: str):
        from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor
        self.proc = PPDocLayoutV3ImageProcessor.from_pretrained(LAYOUT_MODEL_ID)
        self.model = (
            PPDocLayoutV3ForObjectDetection.from_pretrained(LAYOUT_MODEL_ID).to(device).eval()
        )
        self.device = device

    @torch.inference_mode()
    def detect(self, img: Image.Image) -> list[dict]:
        inputs = self.proc(images=[img], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        raw = self.proc.post_process_object_detection(
            outputs, threshold=LAYOUT_THRESHOLD, target_sizes=[img.size[::-1]]
        )
        # GLM-OCR's exact layout post-process: NMS + large-image filter + containment merge
        # ("large" -> drop boxes contained in a bigger one) + order_seq reading-order sort.
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        pp = apply_layout_postprocess(
            raw, id2label, [img.size], layout_nms=LAYOUT_NMS,
            layout_unclip_ratio=LAYOUT_UNCLIP_RATIO, layout_merge_bboxes_mode=LAYOUT_MERGE_MODE,
        )[0]
        dets = []
        for item in pp:                          # already sorted by order_seq (reading order)
            label = item["label"]
            task_type = LABEL_TO_TASK.get(label, "abandon")
            if task_type == "abandon":           # dropped entirely (header/footer/page-no/...)
                continue
            x0, y0, x1, y1 = item["coordinate"]
            dets.append({
                "label": label, "task_type": task_type, "score": round(item["score"], 3),
                "bbox": [int(x0), int(y0), int(x1), int(y1)], "order": item["order"],
            })
        return dets


# ======================================================================
# Post-processing — ported from GLM-OCR glmocr/postprocess/result_formatter.py
# and glmocr/utils/result_postprocess_utils.py (kept behaviourally faithful).
# ======================================================================
def _find_consecutive_repeat(s, min_unit_len=10, min_repeats=10):
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{" + str(min_unit_len) + "," + str(max_unit_len) + r"}?)\1{" + str(min_repeats - 1) + ",}",
        re.DOTALL,
    )
    m = pattern.search(s)
    return (s[: m.start()] + m.group(1)) if m else None


def clean_repeated_content(content, min_len=10, min_repeats=10, line_threshold=10):
    stripped = content.strip()
    if not stripped:
        return content
    if len(stripped) > min_len * min_repeats:
        r = _find_consecutive_repeat(stripped, min_len, min_repeats)
        if r is not None:
            return r
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if len(lines) >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / len(lines)) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(1 for j in range(i, min(i + 3, len(lines))) if lines[j] == common)
                    if consecutive >= 3:
                        orig = content.split("\n")
                        ne = 0
                        for idx, ol in enumerate(orig):
                            if ol.strip():
                                ne += 1
                                if ne == i + 1:
                                    return "\n".join(orig[: idx + 1])
                        break
    return content


def clean_formula_number(s):
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1]
    if s.startswith("（") and s.endswith("）"):
        return s[1:-1]
    return s


_INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$\s*((?:[^$\\]|\\.)+?)\s*\$(?!\$)")


def normalize_inline_formula(content):
    if "$" not in content:
        return content
    parts, last_end = [], 0
    for m in _INLINE_FORMULA_RE.finditer(content):
        formula = m.group(1).strip()
        if not formula:
            continue
        before = content[last_end:m.start()]
        if before and before[-1].isalnum():
            before += " "
        parts.append(before)
        parts.append(f"${formula}$")
        if m.end() < len(content) and content[m.end()].isalnum():
            parts.append(" ")
        last_end = m.end()
    if not parts:
        return content
    parts.append(content[last_end:])
    return "".join(parts)


def clean_content(content):
    if content is None:
        return ""
    # In a raw string r"\\t" is backslash+t, so the old `^(\\t)+` pattern did not only strip
    #   literal tabs but swallowed whole LaTeX commands that start with `\t`:
    #     \therefore→herefore, \times→imes, \text{→ext{, \theta→heta, \tau→au, \triangle→riangle
    #   (anchored at `^`, so only the first command of a block was affected).
    #   The base model wraps formulas in `$$` and never reaches this path; our fine-tuned model emits
    #   bare LaTeX and does, so only our outputs were damaged.
    #   Fix: leave `\t` alone when a lowercase letter follows (a LaTeX command); real tabs are still removed.
    #   Verified on all predicted regions: no false positives.
    #   (The GLM-OCR SDK has the same bug in glmocr/postprocess/result_formatter.py.)
    content = re.sub(r"^(?:\\t(?![a-z]))+", "", content).lstrip()
    content = re.sub(r"(?:\\t(?![a-z]))+$", "", content).rstrip()
    content = re.sub(r"(\.)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(·)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(_)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(\\_)\1{2,}", r"\1\1\1", content)
    if len(content) >= 2048:
        content = clean_repeated_content(content)
    content = normalize_inline_formula(content)
    return content.strip()


def map_label(native_label):
    return LABEL_TO_CATEGORY.get(native_label, "text")


def format_content(content, label, native_label):
    """Per-region formatting (glmocr result_formatter._format_content)."""
    if content is None:
        return content
    if label == "table":
        content = content.strip() if (content.startswith("<table") and content.endswith("</table>")) \
            else clean_content(str(content))
    elif label == "formula":
        content = content.strip() if (content.startswith("$$") and content.endswith("$$")) \
            else clean_content(str(content))
    else:
        content = clean_content(str(content))

    if native_label == "doc_title":
        content = re.sub(r"^#+\s*", "", content)
        content = "# " + content
    elif native_label == "paragraph_title":
        if content.startswith("- ") or content.startswith("* "):
            content = content[2:].lstrip()
        content = re.sub(r"^#+\s*", "", content)
        content = "## " + content.lstrip()

    if label == "formula":
        if content.startswith("$$") or content.startswith("\\[") or content.startswith("\\("):
            content = content[2:].strip()
        if content.endswith("$$") or content.endswith("\\]") or content.endswith("\\)"):
            content = content[:-2].strip()
        content = "$$\n" + content + "\n$$"

    if label == "text":
        if content.startswith("```") and not content.endswith("```"):
            content = content + "\n```"
        if content.startswith("·") or content.startswith("•") or content.startswith("* "):
            content = "- " + content[1:].lstrip()
        m = re.match(r"^(\(|\（)(\d+|[A-Za-z])(\)|\）)(.*)$", content)
        if m:
            _, sym, _, rest = m.groups()
            content = f"({sym}) {rest.lstrip()}"
        m = re.match(r"^(\d+|[A-Za-z])(\.|\)|\）)(.*)$", content)
        if m:
            sym, sep, rest = m.groups()
            sep = ")" if sep == "）" else sep
            content = f"{sym}{sep} {rest.lstrip()}"
        content = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content)
    return content


def merge_formula_numbers(page):
    """Merge formula_number into an adjacent formula via \\tag{} (result_formatter._merge_formula_numbers)."""
    out, skip = [], set()
    for i, blk in enumerate(page):
        if i in skip:
            continue
        nl = blk["native_label"]
        if nl == "formula_number":
            if i + 1 < len(page) and page[i + 1]["label"] == "formula":
                num = clean_formula_number(blk["content"].strip())
                nxt = dict(page[i + 1])
                if nxt["content"].endswith("\n$$"):
                    nxt["content"] = nxt["content"][:-3] + f" \\tag{{{num}}}\n$$"
                out.append(nxt)
                skip.add(i + 1)
            continue  # drop a formula_number with no following formula too
        if blk["label"] == "formula" and i + 1 < len(page) and page[i + 1]["native_label"] == "formula_number":
            num = clean_formula_number(page[i + 1]["content"].strip())
            cur = dict(blk)
            if cur["content"].endswith("\n$$"):
                cur["content"] = cur["content"][:-3] + f" \\tag{{{num}}}\n$$"
            out.append(cur)
            skip.add(i + 1)
            continue
        out.append(blk)
    return out


def assemble_page(regions):
    """regions: list of {native_label, content} in reading order -> page markdown string."""
    page = []
    for r in regions:
        native = r["native_label"]
        label = map_label(native)
        content = format_content(r["content"], label, native)
        is_image = label == "image" or r.get("task_type") == "skip"
        if not is_image and (content is None or not str(content).strip()):
            continue
        page.append({"native_label": native, "label": label, "content": content})
    page = merge_formula_numbers(page)
    return "\n\n".join(b["content"] for b in page if b["content"])


# ======================================================================
# Driver
# ======================================================================
def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="checkpoint dir (or hub id) with block_diffusion.json")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single page image")
    src.add_argument("--images_dir", help="folder of page images")
    p.add_argument("--omnidocbench_gt", help="OmniDocBench GT json; only its referenced images are run")
    p.add_argument("--out_dir", required=True, help="prediction folder (drop-in OmniDocBench prediction.data_path)")
    p.add_argument("--bd_size", type=int, default=None, help="override; default = block_diffusion.json")
    p.add_argument("--loop_infer_S", type=int, default=None,
                   help="LoopMDM: fixed mid-block loop count at inference (default = ckpt loop_smax). "
                        "Only used if the checkpoint was trained with loop_enabled.")
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--max_long_side", type=int, default=MAX_LONG_SIDE,
                   help="crop long-side cap fed to generate(); MATCH the training "
                        "crop policy (use a huge value for full-res-trained ckpts)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=0, help="process at most N pages (0 = all)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="multi-GPU eval: partition pages into this many shards (one process/GPU per "
                        "shard). Pages are split by sorted index (stride), all shards write to the SAME "
                        "--out_dir (disjoint per-page .md → no merge, no collision). Embarrassingly "
                        "parallel: no NCCL/rendezvous. Output is identical to a single 1-GPU run.")
    p.add_argument("--shard_id", type=int, default=0, help="this process's shard index in [0, num_shards)")
    p.add_argument("--overwrite", action="store_true", help="re-OCR pages whose .md already exists")
    p.add_argument("--layout_cache", help="JSON {stem: [region dicts]} from tools/precompute_odb_layout.py. "
                   "If set, PP-DocLayout is SKIPPED (regions loaded from cache) — deterministic per page, "
                   "so scores are identical while dropping the layout model load+inference from every eval.")
    p.add_argument("--write_config", help="path to write a ready-to-run OmniDocBench end2end.yaml")
    p.add_argument("--fast", action="store_true",
                   help="use the KV-cached block-diffusion sampler (fast_sampling.generate_cached). "
                        "UNTESTED speedup path; auto-falls back to model.generate() on any error.")
    p.add_argument("--spec", action="store_true",
                   help="speculative decode == AR-greedy via generate_speculative_prefixcache "
                        "(byte-identical to AR-greedy). Overrides --fast. Per-region forward-count stats.")
    p.add_argument("--spec_natcache", action="store_true",
                   help="Nemotron-style O(N) cached speculative decode (token-causal committed cache, "
                        "2 forwards/round). Not byte-identical on long pages (bf16); judge by score.")
    p.add_argument("--spec_uncached", action="store_true",
                   help="UNCACHED speculative decode (model.generate_speculative) — the VERIFIED working "
                        "spec path. Cached spec under-accepts on real crops; this is the canonical one.")
    p.add_argument("--draft_steps", type=int, default=1,
                   help="number of diffusion unmask steps used to build the draft (default 1 = a single bidirectional forward). "
                        ">1 refines the draft over several steps before verification: draft quality (acceptance) rises "
                        "but the round costs 1+draft_steps forwards, so whether tokens/forward improves must be measured. "
                        "Output is identical to AR greedy for any value because verification guarantees correctness.")
    p.add_argument("--no_dynamo", action="store_true",
                   help="disable torch._dynamo (forces one consistent eager flex kernel -> reproducible "
                        "decode, no compile/recompile jitter). Slower but numerically stable.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="OCR this many region crops per page at once via the batched sampler "
                        "(fast_sampling_batched.generate_cached_batched). DEFAULT 1 = current "
                        "per-crop behavior, untouched. Only active with --fast; >1 falls back to "
                        "bs=1 per crop on any batched error.")
    p.add_argument("--batch_group_len", action=argparse.BooleanOptionalAction, default=True,
                   help="group a page's crops by size before batching so each batch holds "
                        "similar-length crops (avoids the slowest row gating the whole batch). "
                        "On by default; --no-batch_group_len keeps reading order.")
    return p.parse_args()


def make_sampler(args):
    """Return a fn(model, crop, prompt, max_new_tokens) -> text. With --fast, try the cached
    sampler and fall back (once-warned) to the verified slow model.generate() on any failure."""
    slow = lambda model, crop, prompt, mnt: model.generate(
        crop, prompt=prompt, max_new_tokens=mnt, threshold=args.threshold,
        temperature=args.temperature, top_p=args.top_p, max_long_side=args.max_long_side)

    if getattr(args, "spec_uncached", False):
        # UNCACHED speculative decode == model.generate_speculative — the VERIFIED "working" spec path.
        # The cached path under-accepts on real crops (diagnostic: same whole page uncached accept ~13
        # vs cached ~2). Use this for the canonical real-crop spec speed.
        st = {"warned": False, "draft_fwd": 0, "verify_fwd": 0, "commit_fwd": 0,
              "tokens": 0, "regions": 0, "rounds": 0}
        SPEC_STATS.append(st)

        def spec_unc(model, crop, prompt, mnt):
            try:
                txt, s = model.generate_speculative(
                    crop, prompt=prompt, max_new_tokens=mnt, temperature=args.temperature,
                    top_p=args.top_p, max_long_side=args.max_long_side, draft_steps=args.draft_steps, return_stats=True)
                st["draft_fwd"] += s["draft_fwd"]; st["verify_fwd"] += s["verify_fwd"]
                st["tokens"] += s["resp_tokens"]; st["rounds"] += s["rounds"]; st["regions"] += 1
                return txt
            except Exception as exc:
                if not st["warned"]:
                    print(f"[spec_unc->slow] generate_speculative failed ({type(exc).__name__}: {exc}); "
                          f"falling back to model.generate().", flush=True)
                    st["warned"] = True
                return slow(model, crop, prompt, mnt)
        return spec_unc

    if getattr(args, "spec_natcache", False):
        # Nemotron-style O(N) cached spec: committed cached token-causally, 2 forwards/round (pending fold).
        # Not byte-identical to AR-greedy on long pages (bf16 cache drift); judge by SCORE. ~1.1-2x faster.
        from fast_sampling import generate_speculative_natcache
        st = {"warned": False, "draft_fwd": 0, "verify_fwd": 0, "commit_fwd": 0,
              "tokens": 0, "regions": 0, "rounds": 0, "kacc": 0}
        SPEC_STATS.append(st)

        def spec_nat(model, crop, prompt, mnt):
            try:
                tr = []   # per-round detail; tr[i]["k"] = # of the bd drafted tokens accepted that round
                txt, s = generate_speculative_natcache(
                    model, crop, prompt=prompt, max_new_tokens=mnt, temperature=args.temperature,
                    top_p=args.top_p, max_long_side=args.max_long_side, draft_steps=args.draft_steps,
                    return_stats=True, trace=tr)
                st["draft_fwd"] += s["draft_fwd"]; st["verify_fwd"] += s["verify_fwd"]
                st["commit_fwd"] += s["commit_fwd"]; st["tokens"] += s["resp_tokens"]
                st["rounds"] += s["rounds"]; st["regions"] += 1
                st["kacc"] += sum(int(d["k"]) for d in tr)   # REAL draft-accept (of bd drafted, how many matched AR)
                return txt
            except Exception as exc:
                if not st["warned"]:
                    print(f"[spec_nat->slow] natcache failed ({type(exc).__name__}: {exc}); "
                          f"falling back to model.generate().", flush=True)
                    st["warned"] = True
                return slow(model, crop, prompt, mnt)
        return spec_nat

    if getattr(args, "spec", False):
        # SPECULATIVE decode (== AR-greedy) = generate_speculative_prefixcache (caches the vision prefix,
        # re-forwards the short response with the exact uncached masks -> byte-identical to AR-greedy).
        from fast_sampling import generate_speculative_prefixcache
        st = {"warned": False, "draft_fwd": 0, "verify_fwd": 0, "commit_fwd": 0,
              "tokens": 0, "regions": 0, "rounds": 0}
        SPEC_STATS.append(st)

        def spec(model, crop, prompt, mnt):
            try:
                txt, s = generate_speculative_prefixcache(
                    model, crop, prompt=prompt, max_new_tokens=mnt, temperature=args.temperature,
                    top_p=args.top_p, max_long_side=args.max_long_side, draft_steps=args.draft_steps, return_stats=True)
                st["draft_fwd"] += s["draft_fwd"]; st["verify_fwd"] += s["verify_fwd"]
                st["commit_fwd"] += s["commit_fwd"]; st["tokens"] += s["resp_tokens"]
                st["rounds"] += s["rounds"]; st["regions"] += 1
                return txt
            except Exception as exc:
                if not st["warned"]:
                    print(f"[spec->slow] prefixcache spec failed "
                          f"({type(exc).__name__}: {exc}); falling back to model.generate().", flush=True)
                    st["warned"] = True
                return slow(model, crop, prompt, mnt)
        return spec

    if not args.fast:
        return slow

    from fast_sampling import generate_cached
    state = {"warned": False}
    dst = {"fwd": 0, "tokens": 0, "regions": 0}; DIFF_STATS.append(dst)
    # NOTE: token-causal commit (commit_causal) was HYPOTHESISED to help checkpoints with an AR loss,
    # since x_0 is trained token-causal; empirically it made their diffusion-only output clearly WORSE.
    # So we KEEP bidirectional commit (the default) for ALL models — it is the correct one. The
    # diffusion-vs-spec quality gap is genuine exposure bias, not a commit-mask bug. (commit_causal kept
    # in generate_cached as an off-by-default param; do NOT auto-enable.)
    _commit_causal = False

    def fast(model, crop, prompt, mnt):
        try:
            ct = []
            txt = generate_cached(
                model, crop, prompt=prompt, max_new_tokens=mnt, threshold=args.threshold,
                temperature=args.temperature, top_p=args.top_p, max_long_side=args.max_long_side,
                commit_trace=ct, commit_causal=_commit_causal)
            dst["fwd"] += len(ct); dst["tokens"] += sum(ct); dst["regions"] += 1   # real-eval tok/fwd
            return txt
        except Exception as exc:
            if not state["warned"]:
                print(f"[fast->slow] generate_cached failed ({type(exc).__name__}: {exc}); "
                      f"falling back to model.generate() for all regions.", flush=True)
                state["warned"] = True
            return slow(model, crop, prompt, mnt)

    return fast


def make_batched_sampler(args, bs1_sample):
    """Return fn(model, crops:list[Image], prompt, mnt) -> list[str]. Batches crops through
    fast_sampling_batched.generate_cached_batched; on any error falls back (once-warned) to the
    bs=1 per-crop sampler `bs1_sample`, mirroring make_sampler's --fast fallback. Imports the
    batched sampler from the NEW file only (no edits to fast_sampling.py / models/*)."""
    from fast_sampling_batched import generate_cached_batched
    state = {"warned": False}

    def batched(model, crops, prompt, mnt):
        try:
            return generate_cached_batched(
                model, crops, prompt=prompt, max_new_tokens=mnt, threshold=args.threshold,
                temperature=args.temperature, top_p=args.top_p, max_long_side=args.max_long_side)
        except Exception as exc:
            if not state["warned"]:
                print(f"[batched->bs1] generate_cached_batched failed ({type(exc).__name__}: "
                      f"{exc}); falling back to bs=1 per crop for all batches.", flush=True)
                state["warned"] = True
            return [bs1_sample(model, c, prompt, mnt) for c in crops]

    return batched


def ocr_regions_batched(model, page, regions, args, batched_sample):
    """Fill det['content'] for every OCR-eligible region by batching crops of the SAME task_type
    (same prompt + max_new_tokens) through `batched_sample`, optionally length-grouped. Region
    order / assemble_page are untouched — results are scattered straight back onto each det."""
    from collections import defaultdict
    by_task = defaultdict(list)
    for det in regions:
        if det["task_type"] == "skip":
            det["content"] = None
            continue
        by_task[det["task_type"]].append(det)

    for task_type, dets in by_task.items():
        crops = [page.crop(tuple(d["bbox"])) for d in dets]
        order = list(range(len(dets)))
        if args.batch_group_len:                       # similar-size crops batch together
            order.sort(key=lambda i: crops[i].size[0] * crops[i].size[1])
        prompt = TASK_PROMPTS[task_type]
        mnt = MAX_NEW_TOKENS.get(task_type, 512)
        for s in range(0, len(order), args.batch_size):
            idxs = order[s:s + args.batch_size]
            outs = batched_sample(model, [crops[i] for i in idxs], prompt, mnt)
            for i, txt in zip(idxs, outs):
                dets[i]["content"] = txt


def resolve_image_list(args):
    """Return list of (image_path, out_stem). out_stem matches OmniDocBench img_name[:-4]."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if args.image:
        p = Path(args.image)
        return [(str(p), p.stem)]

    if args.omnidocbench_gt:
        gt = json.load(open(args.omnidocbench_gt))
        images_dir = Path(args.images_dir) if args.images_dir else None
        items = []
        for e in gt:
            img_name = os.path.basename(e["page_info"]["image_path"])
            stem = img_name[:-4]  # exactly mirror OmniDocBench _resolve_prediction_path
            cand = (images_dir / img_name) if images_dir else Path(img_name)
            if not cand.exists():
                print(f"[warn] missing image for GT entry: {cand}", flush=True)
                continue
            items.append((str(cand), stem))
        return items

    d = Path(args.images_dir)
    return [(str(p), p.stem) for p in sorted(d.iterdir()) if p.suffix.lower() in exts]


def maybe_write_config(args):
    if not args.write_config:
        return
    gt = args.omnidocbench_gt or "<SET_GROUND_TRUTH_JSON>"
    cfg = f"""end2end_eval:
  metrics:
    text_block: {{metric: [Edit_dist]}}
    display_formula: {{metric: [Edit_dist, CDM]}}
    table: {{metric: [TEDS, Edit_dist]}}
    reading_order: {{metric: [Edit_dist]}}
  dataset:
    dataset_name: end2end_dataset
    ground_truth: {{data_path: {gt}}}
    prediction: {{data_path: {os.path.abspath(args.out_dir)}}}
    match_method: quick_match
"""
    Path(args.write_config).write_text(cfg, encoding="utf-8")
    print(f"[config] wrote {args.write_config}\n  run: python pdf_validation.py --config {args.write_config}", flush=True)


def main():
    args = get_args()
    if getattr(args, "no_dynamo", False):
        torch._dynamo.config.disable = True
        print("[no_dynamo] torch._dynamo.config.disable = True (consistent eager flex, reproducible)", flush=True)
    elif getattr(args, "spec", False) or getattr(args, "spec_uncached", False):
        # FUSED + reproducible: compile flex with dynamic=True so one kernel serves all decode KV_LEN
        # (default dynamic=auto recompiles per step -> recompile_limit -> unfused fallback + jitter).
        from fast_sampling import enable_fused_flex
        enable_fused_flex(dynamic=True)
        print("[fused] enable_fused_flex(dynamic=True): flex stays fused across cached-decode KV_LEN "
              "(fast + reproducible)", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    bd_size = args.bd_size
    if bd_size is None:
        _cfgp = os.path.join(args.checkpoint, "block_diffusion.json")
        with open(_cfgp) as f:
            bd_size = json.load(f).get("bd_size", 32)

    print(f"[load] model={args.checkpoint} bd_size={bd_size} device={device}", flush=True)
    from models.glm_ocr_dllm import GlmOcrBlockDiffusion  # lazy (see top-of-file note)
    model = GlmOcrBlockDiffusion(model_id=args.checkpoint, bd_size=bd_size, dtype=torch.bfloat16).to(device)
    # LoopMDM: if the checkpoint was trained with the looped mid-block, re-enable it at inference.
    # loop_* are not learned params (shared weights), so they must be restored from block_diffusion.json.
    try:
        with open(os.path.join(args.checkpoint, "block_diffusion.json")) as f:
            _bd = json.load(f)
    except Exception:
        _bd = {}
    if _bd.get("loop_enabled"):
        model.loop_enabled = True
        model.loop_start = _bd.get("loop_start", model.loop_start)
        model.loop_n_m = _bd.get("loop_n_m", model.loop_n_m)
        model.loop_smax = _bd.get("loop_smax", model.loop_smax)
        model.loop_infer_S = args.loop_infer_S  # None -> loop_smax (see _loop_S)
        print(f"[loopmdm] inference ON: mid[{model.loop_start}:{model.loop_start+model.loop_n_m}] "
              f"x S={args.loop_infer_S if args.loop_infer_S else model.loop_smax}", flush=True)
    layout_cache = None
    if getattr(args, "layout_cache", None):
        import json as _json_lc
        layout_cache = _json_lc.load(open(args.layout_cache))
        layout = None
        print(f"[layout] precomputed cache {args.layout_cache} ({len(layout_cache)} pages) — PP-DocLayout SKIPPED", flush=True)
    else:
        layout = LayoutDetector(device)
    sample = make_sampler(args)

    use_batched = args.batch_size > 1 and args.fast
    batched_sample = make_batched_sampler(args, sample) if use_batched else None
    if args.batch_size > 1 and not args.fast:
        print("[warn] --batch_size>1 requires --fast; running bs=1 per crop.", flush=True)

    items = resolve_image_list(args)
    items = sorted(items, key=lambda x: x[1])        # deterministic order so all shards agree on the split
    if args.limit:
        items = items[: args.limit]
    if args.num_shards > 1:                           # multi-GPU: this process takes a strided page subset
        assert 0 <= args.shard_id < args.num_shards, "shard_id must be in [0, num_shards)"
        total = len(items)
        items = items[args.shard_id :: args.num_shards]
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(items)}/{total} pages", flush=True)
    mode = "spec-uncached" if getattr(args, "spec_uncached", False) else (
        "spec-natcache" if getattr(args, "spec_natcache", False) else (
        "spec-prefixcache" if args.spec else (
        f"fast bs={args.batch_size}" if use_batched else ("fast" if args.fast else "slow"))))
    print(f"[run] {len(items)} page(s) -> {args.out_dir} (sampler={mode})", flush=True)

    import time as _time
    _infer_s = 0.0; _timed_pages = 0; _timed_chars = 0   # inference-only (excl model/layout load + warmup page 0)
    for n, (img_path, stem) in enumerate(items):
        out_md = Path(args.out_dir) / f"{stem}.md"
        if out_md.exists() and not args.overwrite:
            print(f"[{n + 1}/{len(items)}] skip (exists) {stem}", flush=True)
            continue
        try:
            page = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"[{n + 1}/{len(items)}] FAILED open {img_path}: {exc}", flush=True)
            out_md.write_text("", encoding="utf-8")
            continue

        if torch.cuda.is_available(): torch.cuda.synchronize()
        _t0 = _time.time()
        if layout_cache is not None:
            if stem not in layout_cache:
                raise KeyError(f"page '{stem}' missing from --layout_cache {args.layout_cache} "
                               f"(incomplete cache — re-run tools/precompute_odb_layout.py on the full set)")
            regions = [dict(d) for d in layout_cache[stem]]   # copy: det mutations below must not corrupt cache
        else:
            regions = layout.detect(page)
        for det in regions:
            det["native_label"] = det["label"]
        if use_batched:
            ocr_regions_batched(model, page, regions, args, batched_sample)
        else:
            for det in regions:
                if det["task_type"] == "skip":  # image/chart: placeholder block, no OCR prompt
                    det["content"] = None
                    continue
                crop = page.crop(tuple(det["bbox"]))
                det["content"] = sample(
                    model, crop, TASK_PROMPTS[det["task_type"]],
                    MAX_NEW_TOKENS.get(det["task_type"], 512),
                )
        markdown = assemble_page(regions)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _dt = _time.time() - _t0
        out_md.write_text(markdown, encoding="utf-8")
        if n > 0:                                         # skip page 0 (CUDA/flex warmup)
            _infer_s += _dt; _timed_pages += 1; _timed_chars += len(markdown)
        print(f"[{n + 1}/{len(items)}] {stem}: {len(regions)} regions, {len(markdown)} chars, {_dt:.2f}s", flush=True)

    print(f"[perf] INFER_SECONDS={_infer_s:.2f} PAGES={_timed_pages} CHARS={_timed_chars}", flush=True)
    if SPEC_STATS:
        s = SPEC_STATS[0]
        tot = s["draft_fwd"] + s["verify_fwd"] + s["commit_fwd"]
        rnd = max(1, s.get('rounds', 0))
        # tok/fwd = REAL committed tokens / REAL decoder forwards (independent of accept).
        _tpf = s['tokens'] / max(1, tot)
        # draft_accept = of the bd tokens DRAFTED each round, how many matched AR (kacc/rounds) — the
        # speculation success. NOT tok/fwd*2: tok/fwd counts forwards, draft_accept counts drafted hits.
        _dacc = s.get('kacc', 0) / rnd
        _commit = s['tokens'] / rnd   # committed tokens/round (= draft_accept + AR-correction + free block0)
        print(f"[spec] regions={s['regions']} tokens={s['tokens']} draft_fwd={s['draft_fwd']} "
              f"verify_fwd={s['verify_fwd']} commit_fwd={s['commit_fwd']} total_fwd={tot} rounds={s.get('rounds',0)} "
              f"kacc={s.get('kacc',0)} tokens_per_fwd={_tpf:.3f} draft_accept_per_round={_dacc:.3f} "
              f"commit_per_round={_commit:.3f}", flush=True)
    if DIFF_STATS and DIFF_STATS[0]["regions"]:
        d = DIFF_STATS[0]
        print(f"[diff] regions={d['regions']} tokens={d['tokens']} total_fwd={d['fwd']} "
              f"tokens_per_fwd={d['tokens']/max(1,d['fwd']):.3f}  (REAL eval crops, natural length)", flush=True)
    maybe_write_config(args)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
