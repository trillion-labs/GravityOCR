"""ENGINE-AGNOSTIC OCR throughput client — the SAME boundary for vLLM AND SGLang (and HF-served),
so cross-engine speed is apples-to-apples.

Fires C concurrent OCR requests at a local OpenAI /v1/chat/completions endpoint (base64 image + task
prompt), greedy. tok/s = sum of completion_tokens / wall at each concurrency. The SAME pages (VAL_PAGES,
seeded) regardless of engine. Point --port at whichever server (vLLM 8080 / SGLang 30000 / ...).

  python tools/http_ocr_throughput.py --port 30000 --model glm-ocr --concurrency 1,8 --n 40
"""
import argparse, base64, io, json, time, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk

import os
VAL = os.environ["VAL_PAGES"]   # save_to_disk dataset of region crops (image, prompt, task_type)


def b64(pil):
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mnt_of(tt): return 256 if tt == "text" else (768 if tt == "table" else 256)


def req(port, model, du, prompt, mx):
    body = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": du}}, {"type": "text", "text": prompt}]}],
        "max_tokens": mx, "temperature": 0.0}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        o = json.loads(resp.read())
    return o["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--model", default="glm-ocr")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", default="1,8")
    a = ap.parse_args()
    ds = load_from_disk(VAL, keep_in_memory=False)
    random.seed(a.seed); idxs = sorted(random.sample(range(ds.num_rows), min(a.n, ds.num_rows)))
    items = [(b64(ds[i]["image"]), ds[i].get("prompt", "Text Recognition:"), mnt_of(ds[i].get("task_type", "text"))) for i in idxs]
    req(a.port, a.model, *items[0])  # warmup
    print(f"[http-thrpt] port={a.port} model={a.model} n={len(items)} pages={VAL} seed={a.seed}", flush=True)
    for C in [int(x) for x in a.concurrency.split(",")]:
        t0 = time.time(); tot = 0
        with ThreadPoolExecutor(max_workers=C) as ex:
            for ct in ex.map(lambda it: req(a.port, a.model, *it), items):
                tot += ct
        wall = time.time() - t0
        print(f"  C={C:2d}: {tot} tok / {wall:.1f}s = {tot/wall:.0f} tok/s | {len(items)/wall:.2f} req/s", flush=True)


if __name__ == "__main__":
    main()
