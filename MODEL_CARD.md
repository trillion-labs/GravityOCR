---
license: mit
base_model: zai-org/GLM-OCR
pipeline_tag: image-text-to-text
library_name: transformers
language:
- en
- zh
tags:
- ocr
- document-parsing
- block-diffusion
- speculative-decoding
- glm-ocr
---

<div align="center">

# GravityOCR

**Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding**

[![Code](https://img.shields.io/badge/GitHub-trillion--labs%2FGravityOCR-181717?logo=github)](https://github.com/trillion-labs/GravityOCR)
[![Tech Report](https://img.shields.io/badge/Tech%20Report-PDF-B31B1B?logo=adobeacrobatreader&logoColor=white)](https://github.com/trillion-labs/GravityOCR/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/trillion-labs/GravityOCR/blob/main/LICENSE)

One set of weights drafts a whole block of tokens with block diffusion and verifies it with its own
autoregressive path — the AR output, several tokens per forward pass.

</div>

<p align="center"><img src="assets/teaser.png" width="88%" alt="OmniDocBench Overall vs. pages per second"></p>

<p align="center"><sub>OmniDocBench v1.6 Overall vs. single-stream page rate on one H100 — every system measured on the same boundary.</sub></p>

## Results

OmniDocBench v1.6, official protocol. Speed: SGLang, one H100, batch size 1.

| Model | Decode | Overall ↑ | Tokens / forward ↑ | Pages / s ↑ |
|---|---|---|---|---|
| GLM-OCR (base) | AR | 95.48 | 1.0 | 0.571 |
| GravityOCR | AR | 95.16 | 1.0 | 0.554 |
| **GravityOCR** | **self-speculative** | **95.16** | **9.7** | **0.730** |

- **1.32×** more pages per second than the AR path, identical output → same score.
- On region crops: **3.94×** decode-only, **1.74×** end to end.
- Under bf16 serving kernels the two paths agree exactly on 96.6% of crops; the rest are floating-point tie-breaks.

## See it decode

<p align="center"><img src="assets/page_race.gif" width="100%" alt="One page, AR vs self-speculative, at measured speed"></p>

<p align="center"><sub>One OmniDocBench page, both sides at their measured per-region times (one H100, slowed 6.3×).</sub></p>

## Use

### Autoregressive decoding — stock `transformers`, nothing else needed

```python
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
import torch
from PIL import Image

repo = "trillionlabs/GravityOCR"
processor = AutoProcessor.from_pretrained(repo)
model = GlmOcrForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")

image = Image.open("page_region.png").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": "Text Recognition:"}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Prompts follow GLM-OCR: `Text Recognition:`, `Table Recognition:` (HTML), `Formula Recognition:` (LaTeX).
The model reads one layout region per request — full pages go through a layout detector first, as in the
GLM-OCR SDK. Fine-tuning was predominantly English; Chinese works at the base model's level.

### Self-speculative decoding — the point of the model

Needs the [code repository](https://github.com/trillion-labs/GravityOCR); same weights, same output, ~9.7 tokens per forward pass.

- **Serve:** SGLang v0.5.12 + `patches/sglang/`, then `bash serve/serve_sglang_ocr.sh MODE=spec CKPT=trillionlabs/GravityOCR` — an OpenAI-compatible endpoint (`MODE=ar` serves the same checkpoint autoregressively for a direct comparison).
- **In process:** `python src/infer_omnidocbench.py --checkpoint trillionlabs/GravityOCR --spec_natcache ...`

Both are written out step by step in the repository's [`AGENTS.md`](https://github.com/trillion-labs/GravityOCR/blob/main/AGENTS.md).

## Model details

- **Architecture:** `GlmOcrForConditionalGeneration` — CogViT vision encoder + 0.5B text decoder, unchanged from
  [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), so the checkpoint is a drop-in replacement in any AR pipeline.
- **Files:** `model.safetensors` (bf16, 2.2 GB), processor and tokenizer, and `block_diffusion.json`:

  ```json
  {"bd_size": 32, "mask_id": 59282, "ar_loss_weight": 1.0, "mask_schedule": "uniform", ...}
  ```
  `bd_size` — block size the model was trained with (serve with the same value) · `mask_id` — the `<|mask|>` token used for drafting · `ar_loss_weight > 0` — the checkpoint has a trained AR path and can verify its own drafts.

## Citation

```bibtex
@techreport{gravityocr2026,
  title  = {Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding},
  author = {Trillion Labs},
  year   = {2026}
}
```

## License

MIT, inheriting GLM-OCR's MIT license.
