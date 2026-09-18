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

# GravityOCR

**Diffusion drafts, AR verifies — lossless parallel decoding for document OCR.**

GravityOCR is [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (CogViT vision encoder + 0.5B
text decoder) jointly fine-tuned with a block-diffusion objective and an autoregressive objective on
the *same* weights. At inference the diffusion path drafts a block of tokens in parallel and the AR
path verifies them, so decoding advances several tokens per forward pass while producing exactly the
AR-greedy output. The released checkpoint is additionally trained with GRPO on the AR path using
sequence- and structure-level OCR rewards.

| Model | Decode | OmniDocBench v1.6 Overall ↑ | tokens / forward | pages/s ↑ |
|---|---|---|---|---|
| GLM-OCR (base) | AR | 95.48 | 1.0 | 0.571 |
| GravityOCR | AR | 95.16 | 1.0 | 0.554 |
| **GravityOCR** | **self-speculative** | **95.16** | **9.7** | **0.730** (1.32× the AR path) |

The two GravityOCR rows score the same because they produce the same text. On region crops the gain is
3.94× on decode alone and 1.74× end to end.

![AR vs self-speculative decoding](assets/ar_vs_selfspec.gif)

*The same weights decoding the same crop, one frame per forward pass: autoregressive on the left,
self-speculative on the right. 150 forwards against 10, same output.*

Speed: SGLang serving, one H100, batch size 1, measured at the HTTP boundary. Score: official OmniDocBench protocol
and aggregation. Paper: *Diffusion Drafts, AR Verifies: Lossless Parallel Decoding for Document OCR*
(Trillion Labs, 2026). Code, serving patch and evaluation protocol:
[github.com/trillion-labs/GravityOCR](https://github.com/trillion-labs/GravityOCR).

## What is in this repository

A standard `GlmOcrForConditionalGeneration` checkpoint (`model.safetensors`, bf16, 2.2 GB) with its
processor/tokenizer, plus `block_diffusion.json`:

```json
{"bd_size": 32, "mask_id": 59282, "ar_loss_weight": 1.0, "mask_schedule": "uniform", ...}
```

- `bd_size` — diffusion block size the model was trained with (serve with the same value).
- `mask_id` — the `<|mask|>` token id used for diffusion drafting.
- `ar_loss_weight > 0` — the checkpoint has a trained AR path and can therefore verify its own drafts.

The architecture and prompt format are unchanged from GLM-OCR, so the model is a drop-in replacement
for it in any AR pipeline.

## Usage

### Plain autoregressive decoding (stock `transformers`)

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
Prompts follow GLM-OCR: `Text Recognition:`, `Table Recognition:` (HTML output), `Formula Recognition:`
(LaTeX output), applied per layout region as in the GLM-OCR SDK.

### Self-speculative decoding (the point of this model)

Same weights, ~9.7 tokens committed per forward pass, output identical to the AR path above.

- **Serving:** SGLang v0.5.12 with the patch in the code repository (`patches/sglang/`), then
  `bash serve/serve_sglang_ocr.sh MODE=spec CKPT=trillionlabs/GravityOCR` — an OpenAI-compatible
  endpoint. `MODE=ar` serves the same checkpoint autoregressively for a direct comparison.
- **In-process:** `python src/infer_omnidocbench.py --checkpoint trillionlabs/GravityOCR --spec_natcache ...`
  from the code repository.

## Training summary

1. **Joint AR + block-diffusion fine-tuning** of GLM-OCR: 40k steps, ~26B tokens, 16×H100, DeepSpeed
   ZeRO-2 bf16, block size 32, block-phase jitter, vision-encoder LR ×0.1, AR loss weight 1.0.
   Vision is never noised.
2. **GRPO on the AR path** (TRL) from the 40k checkpoint: edit-distance / TEDS / CDM rewards with
   structure penalties and scorer-identical markup normalization; lr 3e-6, 28 generations per prompt,
   KL β=1e-3, token-level truncated importance sampling. This card's checkpoint is step 500 of that run.

Training data: 10.8M region-level examples from public document datasets (DocGenome, Docmatix,
PubTables-1M, FinTabNet, SynthTabNet, PubTabNet, RVL-CDIP, DocLayNet, UniMER), ~60/20/20 text /
table / formula, predominantly English; pages overlapping the evaluation sets were removed.
The assembled pool is not redistributed.

## Intended use and limitations

Document image → text/HTML/LaTeX transcription of layout regions (full-page use goes through a
layout detector, as in GLM-OCR). Training data is predominantly English; Chinese is supported by the
base model but under-represented in fine-tuning. Verification guarantees the output equals the AR
path's greedy output — it does not make the AR path more accurate than it is.

## License

MIT, inheriting GLM-OCR's MIT license.

## Citation

```bibtex
@techreport{gravityocr2026,
  title  = {Diffusion Drafts, AR Verifies: Lossless Parallel Decoding for Document OCR},
  author = {Trillion Labs},
  year   = {2026}
}
```
