"""AR (autoregressive) GLM-OCR page inference for OmniDocBench (NEW FILE).

Reuses infer_omnidocbench's FIXED LayoutDetector / assemble_page / TASK_PROMPTS (so layout label
mapping + region overlap handling match GLM-OCR official), but swaps the per-region OCR to the BASE
GLM-OCR weights run via plain HF autoregressive `.generate()` (greedy, KV cache). This is the
"our-code AR" reference to compare against the official `glmocr` pipeline on the same 100 ODB pages.

Vision patch_embed Conv3d->matmul fix is applied (identical math, avoids the cuDNN pathology).
"""
import argparse, json, os
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import GlmOcrForConditionalGeneration, AutoProcessor
import infer_omnidocbench as base

Image.MAX_IMAGE_PIXELS = None
STOP_IDS = (59246, 59253)   # GLM-OCR generation_config eos_token_id list


def patch_vision_fast(glm):
    pe = glm.model.visual.patch_embed
    def ff(h):
        return F.linear(h.reshape(h.shape[0], -1).to(pe.proj.weight.dtype),
                        pe.proj.weight.reshape(pe.embed_dim, -1), pe.proj.bias)
    pe.forward = ff


def _prep(image, max_long_side):
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_long_side:
        s = max_long_side / max(w, h)
        image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return image


@torch.no_grad()
def generate_ar(proc, glm, image, prompt, max_new_tokens, dev, max_long_side):
    image = _prep(image, max_long_side)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=[image], return_tensors="pt").to(dev)
    g = glm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                     use_cache=True, pad_token_id=59246)
    new = g[0, enc["input_ids"].shape[1]:].tolist()
    cut = next((i for i, t in enumerate(new) if t in STOP_IDS), len(new))
    return proc.tokenizer.decode(new[:cut], skip_special_tokens=True)


@torch.no_grad()
def generate_ar_batch(proc, glm, images, prompts, max_new_tokens, dev, max_long_side):
    """Batched AR OCR over a list of region crops (native HF batched generate, LEFT-padded).
    All rows share one max_new_tokens (take the group max). Returns list[str] aligned with inputs."""
    if not images:
        return []
    if len(images) == 1:
        return [generate_ar(proc, glm, images[0], prompts[0], max_new_tokens, dev, max_long_side)]
    imgs = [_prep(im, max_long_side) for im in images]
    texts = [proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pr}]}],
        add_generation_prompt=True, tokenize=False) for pr in prompts]
    old_side = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = "left"                       # decoder-only generation needs left pad
    try:
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(dev)
        g = glm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                         use_cache=True, pad_token_id=59246)
    finally:
        proc.tokenizer.padding_side = old_side
    P = enc["input_ids"].shape[1]                              # padded prompt len (same for all rows, left-pad)
    out = []
    for i in range(len(imgs)):
        new = g[i, P:].tolist()
        cut = next((j for j, t in enumerate(new) if t in STOP_IDS), len(new))
        out.append(proc.tokenizer.decode(new[:cut], skip_special_tokens=True))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="zai-org/GLM-OCR")
    p.add_argument("--omnidocbench_gt", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = full set")
    p.add_argument("--max_long_side", type=int, default=99999)
    p.add_argument("--batch_size", type=int, default=1, help=">1 = batch region crops per HF generate call")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0, help="this process's shard index in [0, num_shards)")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model_id)
    glm = GlmOcrForConditionalGeneration.from_pretrained(args.model_id, dtype=torch.bfloat16).to(dev).eval()
    patch_vision_fast(glm)
    layout = base.LayoutDetector(dev)
    items = base.resolve_image_list(argparse.Namespace(image=None, images_dir=args.images_dir,
                                                       omnidocbench_gt=args.omnidocbench_gt))
    items = sorted(items, key=lambda x: x[1])            # deterministic order so all shards agree on the split
    if args.limit:
        items = items[:args.limit]
    total = len(items)
    if args.num_shards > 1:                              # multi-GPU: strided page subset (mirror infer_omnidocbench)
        assert 0 <= args.shard_id < args.num_shards, "shard_id must be in [0, num_shards)"
        items = items[args.shard_id :: args.num_shards]
    print(f"[ar shard {args.shard_id}/{args.num_shards}] {len(items)}/{total} pages -> {args.out_dir}", flush=True)
    import time as _time, torch as _torch
    _infer_s = 0.0; _timed_pages = 0; _timed_chars = 0    # inference-only (excl load + warmup page 0)
    for n, (img_path, stem) in enumerate(items):
        out_md = Path(args.out_dir) / f"{stem}.md"
        if out_md.exists():
            continue
        try:
            page = Image.open(img_path).convert("RGB")
        except Exception:
            out_md.write_text("", encoding="utf-8"); continue
        if _torch.cuda.is_available(): _torch.cuda.synchronize()
        _t0 = _time.time()
        regions = layout.detect(page)
        ocr = []                                              # (det, crop, prompt, mnt) for non-skip regions
        for det in regions:
            det["native_label"] = det["label"]
            if det["task_type"] == "skip":
                det["content"] = None; continue
            ocr.append((det, page.crop(tuple(det["bbox"])),
                        base.TASK_PROMPTS[det["task_type"]],
                        base.MAX_NEW_TOKENS.get(det["task_type"], 512)))
        if args.batch_size <= 1:
            for det, crop, pr, mnt in ocr:
                det["content"] = generate_ar(proc, glm, crop, pr, mnt, dev, args.max_long_side)
        else:
            # group by similar crop size (so padding waste is small) then batch within MAX_NEW_TOKENS bucket
            ocr.sort(key=lambda x: (x[3], max(x[1].size)))    # by max_new_tokens, then size
            for i in range(0, len(ocr), args.batch_size):
                grp = ocr[i:i + args.batch_size]
                mnt = max(g[3] for g in grp)                  # shared cap = group max
                outs = generate_ar_batch(proc, glm, [g[1] for g in grp], [g[2] for g in grp],
                                         mnt, dev, args.max_long_side)
                for (det, *_), txt in zip(grp, outs):
                    det["content"] = txt
        md = base.assemble_page(regions)
        if _torch.cuda.is_available(): _torch.cuda.synchronize()
        _dt = _time.time() - _t0
        out_md.write_text(md, encoding="utf-8")
        if n > 0:
            _infer_s += _dt; _timed_pages += 1; _timed_chars += len(md)
        print(f"[{n+1}/{len(items)}] {stem}: {len(regions)} regions, {len(md)} chars, {_dt:.2f}s", flush=True)
    print(f"[perf] INFER_SECONDS={_infer_s:.2f} PAGES={_timed_pages} CHARS={_timed_chars}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
