#!/usr/bin/env python3
"""Smoke-test the GLM-OCR vLLM server with one image via the OpenAI chat API.

Usage:
  .venv-vllm/bin/python test_request.py [IMAGE_PATH] [--prompt "Text Recognition:"] [--host localhost:8080]
"""
import argparse, base64, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?",
                default="${DOCR_WORKSPACE}/diffusion-ocr/GLM-OCR/examples/source/page.png")
ap.add_argument("--prompt", default="Text Recognition:")
ap.add_argument("--host", default="localhost:8080")
ap.add_argument("--model", default="glm-ocr")
ap.add_argument("--max-tokens", type=int, default=4096)
args = ap.parse_args()

with open(args.image, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
ext = args.image.rsplit(".", 1)[-1].lower()
mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")

payload = {
    "model": args.model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
            {"type": "text", "text": args.prompt},
        ],
    }],
    "max_tokens": args.max_tokens,
    "temperature": 0.0,
}
req = urllib.request.Request(
    f"http://{args.host}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.load(r)
print("=== OCR output ===")
print(out["choices"][0]["message"]["content"])
print("=== usage ===", out.get("usage"))
