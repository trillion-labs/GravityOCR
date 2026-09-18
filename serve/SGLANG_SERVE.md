# Serving with SGLang — self-speculative (spec) and autoregressive (ar) modes

`serve/serve_sglang_ocr.sh` serves the same checkpoint either way over an OpenAI-compatible HTTP
endpoint. `MODE=spec` is block-diffusion self-speculative decoding (lossless, faster); `MODE=ar` is
plain autoregressive decoding and exists for direct comparison.

## 1. One command

```bash
cd $DOCR_ROOT
bash serve/serve_sglang_ocr.sh MODE=spec GPU=0 PORT=30000 CKPT=<hf checkpoint dir or trillionlabs/GravityOCR> \
     CONTEXT=16384 VBIN=$SGLANG_SRC/.venv-sglang/bin
bash serve/serve_sglang_ocr.sh MODE=ar   GPU=1 PORT=30001 CKPT=...      # AR baseline, same weights
curl -s http://127.0.0.1:30000/v1/models                                  # readiness
```
Multi-GPU data parallel: `DP=8 GPU=0,1,2,3,4,5,6,7`. Batch cap for spec mode: `MAXREQ=96` (see gotcha 6).
Draft window: `DRAFT=33` is the default (K = training block size 32 + the boundary token; see
`docs/SGLANG_SELFSPEC_IMPL.md` §9-2) — override only for K sweeps, output is identical for any K.

The launcher bakes in the gotchas that each cost a debugging cycle:
1. **venv bin on PATH** — flashinfer cuda-graph capture shells out to `ninja`; missing → capture dies with
   `FileNotFoundError: ninja`. Use `$VBIN/python` directly, not `uv run`, which may pick a different venv.
2. `VLLM_USE_FLASHINFER_SAMPLER=0` — else the FlashInfer-sampler JIT needs ninja/config → crash.
3. `CUDA_HOME=/usr/local/cuda-12.8` — nvcc for JIT (a 12.8 driver runs cu129 builds).
4. `HF_HUB_OFFLINE=1` — the model is cached; concurrent loads otherwise throttle on the Hub.
5. `--speculative-algorithm SELFSPEC_DIFFUSION` is accepted only because the patch adds it to the CLI
   `choices` in `server_args.py`, plus `--page-size 1`.
6. In spec mode the draft window (K=33) needs more prefill scratch than AR; at very large batch caps
   cuda-graph capture can overflow the flashinfer workspace. Lower `MAXREQ` or `GRAPHBS`
   (`--cuda-graph-max-bs`).

Spec mode runs the **full-draft round** by default (`SELFSPEC_FULLDRAFT=1`): the draft window is the boundary
token plus 32 masks, every mask logit is used, and a fully accepted draft also commits the verifier's bonus
token, so a round commits up to K+1 = 34 tokens and reserves K+1 KV slots per request. `SELFSPEC_FULLDRAFT=0`
selects the legacy round (at most K = 33 tokens). Both are lossless — the parity check in §2 holds for either
(`docs/SGLANG_SELFSPEC_IMPL.md` §2).

The endpoint is a standard `/v1/chat/completions`: send `{image_url: data:image/png;base64,..., text:
"<task prompt>"}` with `temperature: 0`. Task prompts follow GLM-OCR (`Text Recognition:`,
`Table Recognition:`, `Formula Recognition:`), one request per layout region.

## 2. Parity check (spec ≡ AR)

Run the same crops through `MODE=ar` and `MODE=spec` and compare **token ids**, not text. They must be
identical with `repetition_penalty` off (see `docs/SGLANG_SELFSPEC_IMPL.md` §9-7 and §11). Acceptance
statistics are available from `Engine.get_server_info()["internal_states"]`
(`avg_spec_accept_length`; tok/fwd = that value / 2); a server driven through the Python API can
re-export it as an `/internal_state/` route.

## 3. Evaluating OmniDocBench through a server

The OmniDocBench pipeline is: layout detection (PP-DocLayout) → per-region crop → OCR each crop →
assemble `<stem>.md` → official scorer. Detection is deterministic and engine-independent, so it is
precomputed once with `tools/precompute_odb_layout.py`; an HTTP client then sends each page's crops
concurrently to the server (the server batches them), assembles the page with
`infer_omnidocbench.assemble_page`, and the result is scored with `slurm/score_finalize.slurm`
(`RD=<run dir>`). Both `MODE=ar` and `MODE=spec` give the same predictions; the difference is speed.

## 4. Throughput

`tools/http_ocr_throughput.py --port <p> --concurrency 1,8,32` fires concurrent OCR requests at any
OpenAI-compatible endpoint with one identical client, so engines (this server in either mode, vLLM,
…) are compared at the same boundary. Always report the page set (`VAL_PAGES`), concurrency and
`max_tokens` next to a number.

## 5. Where the time goes (for optimization work)

Both AR and spec are host/launch-bound on a 0.9B model even with cuda graphs. A spec round is two
block forwards (draft + verify) plus a small amount of accept/mask bookkeeping; the two forwards are
dispatched serially (CPU dispatch → wait for GPU). Overlapping the host-side orchestration with GPU
compute is the main remaining lever for latency; multi-GPU data parallelism is the lever for aggregate
throughput. Removing the verify mask is not (it breaks losslessness under cuda graphs).
