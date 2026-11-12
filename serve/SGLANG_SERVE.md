# Serving with SGLang — self-speculative (spec) and autoregressive (ar) modes

`serve/serve_sglang_ocr.sh` serves the same checkpoint either way over an OpenAI-compatible HTTP
endpoint. `MODE=spec` is block-diffusion self-speculative decoding: every committed token is checked
by the model's own AR head, so the output follows AR decoding up to the position layout and bf16
near-ties described in §2. `MODE=ar` is plain autoregressive decoding and exists for direct comparison.

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

**Optimized runtime: `FAST=1` is the default.** It turns on the runtime optimizations shipped in the SGLang patch
(`docs/SGLANG_SELFSPEC_IMPL.md` §7-5): in both modes the GLM ViT runs as CUDA graphs captured per patch count up to
2048, pixel values cross process boundaries as bf16, the server's CPU ops run single-threaded and two tokenizer
worker processes preprocess regions in parallel; in spec mode additionally the mask-free attention path, the M-RoPE
cache fix, the GPU-side commit, host-side lengths, device-side positions and the draft launched ahead of the host
bookkeeping. The output is bit-identical to `FAST=0`, which serves the unoptimized runtime. Any single flag can be
overridden from the environment (e.g. `SELFSPEC_DRAFT_AHEAD=0`), and `EXTRA_ARGS` is appended to `launch_server`.
The ViT graphs lengthen startup and, because they are captured before SGLang sizes the KV cache, take about 3.3 GB
of GPU memory from it (on an 80 GB H100 about 7.7% of the KV pool at `MEMFRAC=0.6`). `SS_GLM_VIT_GRAPH_MAXM=0` gives
it back; a capture that fails (e.g. out of memory) is logged as `[ss-vit-graph] OFF: capture failed (...)` and the
server continues with the eager ViT, with the same output (`docs/SGLANG_SELFSPEC_IMPL.md` §7-5, §10). Spec mode also
sets `SELFSPEC_POS_OFFSET=1`, which puts every generated token one RoPE position after where AR decoding and training
put it (the layout the released self-spec results were produced with; `0` is the AR layout, §2), and, when `CKPT` has
no local `block_diffusion.json` (e.g. a hub id), `SELFSPEC_MASK_ID=59282` (`docs/SGLANG_SELFSPEC_IMPL.md` §9-1, §10).

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
selects the legacy round (at most K = 33 tokens). Both verify every token the same way, so the parity check in §2
holds for either (`docs/SGLANG_SELFSPEC_IMPL.md` §2).

The endpoint is a standard `/v1/chat/completions`: send `{image_url: data:image/png;base64,..., text:
"<task prompt>"}` with `temperature: 0` (greedy, what the released results use). Task prompts follow GLM-OCR
(`Text Recognition:`, `Table Recognition:`, `Formula Recognition:`), one request per layout region. Spec mode
also serves sampling settings, penalties, `logit_bias`, `min_tokens` and structured output (`response_format`)
with AR's semantics (`docs/SGLANG_SELFSPEC_IMPL.md` §9-6, §9-7), and log probabilities (`logprobs` /
`top_logprobs`, `/generate`'s `return_logprob`): one entry per generated token, the first token of each round from
SGLang's sampler and the others from the verify pass (`docs/SGLANG_SELFSPEC_IMPL.md` §5-4).

**Security.** The launcher binds `127.0.0.1` and sets no API or admin key, so every user on the host can reach
SGLang's management endpoints (weight reloads from a caller-chosen path, profiling, cache control). On a shared host
give the server keys through a mode-0600 YAML file, `EXTRA_ARGS="--config <file>"` with `api-key:` and
`admin-api-key:` entries, rather than on the command line. The patch makes every tokenizer worker check them (upstream
v0.5.10 checks keys only with one worker), and `/server_info` / `/get_server_info` return the configuration with
both keys blanked. The server's torch.distributed rendezvous runs on `127.0.0.1` with a store that listens on
loopback only, whatever `--host` is (upstream v0.5.10 binds that store on every interface); gloo and NCCL still open
sockets of their own, so a host reachable from untrusted networks needs a firewall that admits only the HTTP port.

## 2. Parity check (spec vs AR)

Run the same crops through `MODE=ar` and `MODE=spec` and compare **token ids**, not text. Two things separate
them. With the launcher's `SELFSPEC_POS_OFFSET=1`, every generated token sits one RoPE position after where AR
decoding puts it, so spec verifies against the AR head under that shifted layout; at that layout the two modes
agree on 96.6% of 8,922 region crops. And the verify forward scores K positions at once with other kernels than
AR's one-token steps, so a bf16 near-tie can resolve differently. For a parity check start spec mode with
`SELFSPEC_POS_OFFSET=0` (the AR layout) and AR mode with `EXTRA_ARGS=--disable-overlap-schedule` (SGLang's overlap
scheduler counts each token for the penalties one step late); then every difference must be a near-tie
(`docs/SGLANG_SELFSPEC_IMPL.md` §9-6, §9-7 and §11). Acceptance
statistics are available from `Engine.get_server_info()["internal_states"]`
(`avg_spec_accept_length`; tok/fwd = that value / 2); a server driven through the Python API can
re-export it as an `/internal_state/` route.

## 3. Evaluating OmniDocBench through a server

The OmniDocBench pipeline is: layout detection (PP-DocLayout) → per-region crop → OCR each crop →
assemble `<stem>.md` → official scorer. Detection is deterministic and engine-independent, so it is
precomputed once with `tools/precompute_odb_layout.py`; an HTTP client then sends each page's crops
concurrently to the server (the server batches them), assembles the page with
`infer_omnidocbench.assemble_page`, and the result is scored with `slurm/score_finalize.slurm`
(`RD=<run dir>`). `MODE=ar` and `MODE=spec` give the same predictions up to the differences in §2; the
difference that matters is speed.

## 4. Throughput

`tools/http_ocr_throughput.py --port <p> --concurrency 1,8,32` fires concurrent OCR requests at any
OpenAI-compatible endpoint with one identical client, so engines (this server in either mode, vLLM,
…) are compared at the same boundary. Always report the page set (`VAL_PAGES`), concurrency and
`max_tokens` next to a number.

## 5. Where the time goes (for optimization work)

On a 0.9B model both AR and spec are host/launch-bound even with cuda graphs: a spec round is two block forwards
(draft + verify) plus accept/commit bookkeeping, and the unoptimized runtime spends more of a round in host work —
mask construction and copies, blocking device↔host copies, per-token Python bookkeeping, vision preprocessing — than
in GPU compute. `FAST=1` removes most of it, bit-identically: the verify and draft masks are gone (built-in FlashInfer
mask modes), a blocking copy per forward is gone (M-RoPE cache fix), the commit is one kernel plus one readback, and
the next round's draft is launched before the current round's host bookkeeping so that bookkeeping runs under GPU
time. Multi-GPU data parallelism is the lever for aggregate throughput.

An older mask-free experiment (`SELFSPEC_NOMASK=1`, a dual-graph path that read a stale captured mask buffer under cuda
graphs) has been removed from the patch; the `FAST=1` attention path is a different mechanism
(`docs/SGLANG_SELFSPEC_IMPL.md` §7-1).
