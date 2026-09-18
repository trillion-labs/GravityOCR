# 09 — Code map (`src/`)

The canonical code, kept deliberately minimal. Everything here is on the inference / evaluation
critical path (or a direct import of it). The training and RL code is not part of this release.

## src/ — the canonical set

| file | role |
|---|---|
| `models/glm_ocr_dllm.py` | `GlmOcrBlockDiffusion`: loads GLM-OCR, adds `<\|mask\|>`, the block-diffusion forward, `generate` (AR/diffusion sampler) and `generate_speculative` (the reference spec-decode path) |
| `models/__init__.py` | package marker |
| `infer_omnidocbench.py` | OmniDocBench inference, diffusion (`--fast`), page-sharded (`--num_shards/--shard_id`) |
| `infer_odb_ar.py` | OmniDocBench inference, native HF AR-greedy (the ~0.09 baseline path); also provides `patch_vision_fast`/`generate_ar` reused by the tok/fwd bench |
| `fast_sampling.py` | KV-cached diffusion sampler, bs=1 (imported by `infer_omnidocbench --fast`) |
| `fast_sampling_batched.py` | batched diffusion sampler, bs>1 |
| `layout_postprocess.py` | GLM-OCR-exact NMS / containment-merge / reading-order (verbatim copy of GLM-OCR's util; hard dep of `infer_omnidocbench`) |
| `run_output.py` | the standardized run-folder helper (`make_run_dir`/`write_score`/`write_speed`). `RUNS_ROOT` defaults to this folder's `runs/` (override with `$RUNS_ROOT`). |
| `measure_eval_perf.py` | throughput (pages/s) + TPS (tokens/s) for a prediction dir |

## Data flow

**Eval (diffusion):** `infer_omnidocbench.py` loads an `hf_*` checkpoint → per page: layout-detect →
crop regions → `fast_sampling`/`fast_sampling_batched` diffusion decode → `layout_postprocess` to
assemble reading order → write `<stem>.md`. Then the OmniDocBench scorer (in `scorer/`, CPU) compares
predictions vs GT.

**Eval (AR):** `infer_odb_ar.py`, same layout+assemble pipeline but native HF greedy generate — so
its scores are directly comparable to the diffusion path at the same scale.

For the *why* behind the architecture, see the paper and `SGLANG_SELFSPEC_IMPL.md`.
