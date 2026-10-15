# Implementing block-diffusion self-speculative decoding in SGLang

This document plus the patch in `patches/sglang/` is meant to be enough to implement the same
decoding scheme for another model or task. It describes what code you have to write, not what we did.

Shipped artifacts — `patches/sglang/`

| file | content |
|---|---|
| `sglang_BASE_COMMIT.txt` | upstream SGLang commit the patch applies to (`1519acf`, v0.5.12) |
| `sglang_0.5.12_selfspec.patch` | changes to 24 existing source files plus `python/pyproject.toml` — `git apply` |
| `python/sglang/srt/speculative/selfspec_diffusion_worker.py` | the new worker (the core of the implementation) |
| `python/sglang/srt/speculative/selfspec_fast_attn.py` | mask-free draft / verify attention planning for FlashInfer (`SELFSPEC_FASTATTN`, §7-1) |
| `python/sglang/srt/speculative/selfspec_fast_commit.py` | GPU-side commit + single-readback host bookkeeping (`SELFSPEC_FAST_COMMIT`, §7-5) |
| `python/sglang/srt/multimodal/glm_vit_graph_runner.py` | CUDA graphs for the GLM-OCR vision encoder (`SS_GLM_VIT_GRAPH_MAXM`, §7-5) |

The three files after the worker implement runtime optimizations only (§7-5). They are imported unconditionally but
act only under their environment flags, which are off in the code; the output with them on is bit-identical to the
output with them off.

Check: applying the patch and copying the new files onto a clean checkout of the base commit reproduces our source tree
file for file.

---

## 1. Prerequisite — can your model do this?

Both are required:

1. **Block-diffusion training** — the model fills K masked slots at once, bidirectionally (non-causal).
2. **An AR (causal) head** — the same weights must also predict the next token causally.

Diffusion drafts, AR judges. Without the AR head there is no reference to verify against and
losslessness cannot be guaranteed. Our checkpoints train both objectives jointly (`ar_loss_weight > 0`).
A pure diffusion model cannot use this scheme.

---

## 2. The algorithm (one round)

```
state: committed tokens are in the KV cache (causal). x0 = the boundary token: the last token committed
       by the previous round (its own K/V is not in the cache yet). B = training block size (32), K = B + 1.

(1) DRAFT    window win = [x0 | MASK x B]                  (length K)
             one bidirectional forward; x0's row is causal, the masks attend to each other
             logits[j] predicts win[j+1]                    (token shift)
             a0       = argmax(logits[0])                   the AR prediction after x0 (x0's row is causal, so it is exact)
             d_1..d_B = argmax(logits[1..B])                one draft token per mask; the LAST mask's logit predicts the
                                                            token right after the window, so every mask logit is used

(2) VERIFY   one causal forward over [committed | a0, d_1..d_B]      (K positions again -- same shape as the draft)
             x0's K/V from the draft forward is kept (it was computed causally), so x0 is not re-forwarded
             accept the longest prefix of d_1..d_B that matches AR greedy;
             at the first mismatch take the AR token instead; if everything matched take the verifier's
             prediction after d_B                                        (= bonus token)

(3) COMMIT   a0 + accepted prefix + bonus  =  accept_length + 2 tokens, between 2 and B + 2 (= K + 1)
             seq_lens += committed ;  next x0 = the last committed token
```

Two forwards per round, at most B + 2 = K + 1 tokens committed per round → theoretical ceiling
`tok/fwd = (K+1)/2`. Even if every draft token is rejected the round still commits a0 and the AR
correction, so the worst case is AR speed and decoding never stalls.

`SELFSPEC_FULLDRAFT=0` selects the **legacy round**: verify runs over the draft as is, `[x0, a0, d_1..d_{B-1}]`
with root x0 — the last mask's logit is unused and a fully accepted draft gets no bonus token — so a round
commits at most K tokens. Both rounds are verified against AR greedy and produce identical output; the
full-draft round only commits more tokens per forward (about +1–3%).

**Correctness is guaranteed entirely by (2).** Whatever the draft contains, only the AR-greedy-matching
prefix is accepted, so changing K or degrading the draft changes speed only, never the output
(verified: K = 32/33/34/36/40/48/64 all produce identical token ids).

> This property hides porting mistakes. A wrong mask id still produces correct output — only the
> acceptance rate collapses. Failures therefore show up not as crashes but as "it works but is not
> faster". Read §9-1 first.

This is the fresh-window parallel-draft pattern (DFlash / Nemotron style): a new window every round,
sliding by the accepted length, never stuck at a block boundary.

---

## 3. Design decision — why on top of the spec framework, not the dllm path

SGLang already has a block-diffusion path (`srt/dllm/`), but it advances in **block_size lockstep**:
exactly block_size tokens per step. Self-spec has a variable accept length; built on the dllm
scheduler it stalled at ~2.5 tok/fwd.

The speculative-decoding framework's `NgramVerifyInput.verify()` already does **variable-length KV
advance**:

```python
batch.seq_lens.add_(self.accept_length + 1)          # ngram_info.py
batch.seq_lens_cpu.add_(accept_length_cpu + 1)
```

So we do not write accept/KV logic ourselves: the ngram verifier is reused as a "K-token linear-chain
verifier". This is the central idea; make the same choice. The full-draft round (§5-4) subclasses that
verifier rather than replacing it.

### Turning the tree into a linear chain (worker)
```python
retrive_index = arange(bs)[:,None]*n + arange(n)[None,:]
nxt = arange(1, n+1).repeat(bs,1); nxt[:, -1] = -1   # next = i+1
sib = full((bs, n), -1)                              # no siblings
```
With `next = i+1, sibling = -1` the `verify_tree_greedy` kernel degenerates from tree accept to
**longest-matching-prefix accept** — exactly the speculative-decoding accept rule.

---

## 4. Quick start

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout $(grep -oE '[0-9a-f]{40}' $DOCR_ROOT/patches/sglang/sglang_BASE_COMMIT.txt | head -1)
git apply $DOCR_ROOT/patches/sglang/sglang_0.5.12_selfspec.patch
cp -r $DOCR_ROOT/patches/sglang/python .          # adds the four new files (table above)
```
```bash
SELFSPEC_POS_OFFSET=1 \
python -m sglang.launch_server --model-path $CKPT --trust-remote-code \
  --speculative-algorithm SELFSPEC_DIFFUSION \
  --speculative-num-draft-tokens 33 \
  --page-size 1 \
  --attention-backend flashinfer --mem-fraction-static 0.6 --context-length 8192
```
The same engine is also usable through SGLang's Python `Engine` API with these arguments.

`$CKPT` is a checkpoint directory with `block_diffusion.json`, which supplies the mask id; for a hub id also set
`SELFSPEC_MASK_ID=59282` (§9-1). `SELFSPEC_POS_OFFSET=1` gives the positions GLM-OCR checkpoints are served with (§10).
This command runs the unoptimized runtime; the optimized one adds the flags of §7-5.
The launcher with every gotcha baked in is `serve/serve_sglang_ocr.sh` (`MODE=spec|ar`); its `FAST=1` default turns
the §7-5 flags on, and `FAST=0` serves the unoptimized runtime.

---

## 5. The code you write — the worker

The framework needs one interface: `forward_batch_generation(batch)`. The constructor signature is
dictated by SGLang (the `spec_info.py` factory calls it).

### 5-1. Constructor
```python
class SelfSpecDiffusionWorker:
    def __init__(self, server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank,
                 attn_cp_rank, moe_dp_rank, nccl_port, target_worker):
        self.target_worker   = target_worker            # draft == target (no separate draft model)
        self.model_runner    = target_worker.model_runner
        self.model_config    = target_worker.model_config    # the framework reads it from the worker
        self.draft_token_num = server_args.speculative_num_draft_tokens   # = K
        dllm_cfg = DllmConfig.from_server_args(server_args)
        # SELFSPEC_MASK_ID > <model_path>/block_diffusion.json > --dllm-algorithm config > error   <- §9-1
        self.mask_id, self._bd_size, _ = _resolve_block_diffusion(server_args, dllm_cfg)
```

### 5-2. Prefill branch — planting the seed
The first generated token becomes the `block[0]` seed of the first decode block.
```python
if batch.forward_mode.is_extend():
    r = self.target_worker.forward_batch_generation(batch.get_model_worker_batch())
    self._seeds()[batch.req_pool_indices] = r.next_token_ids.view(-1).to(torch.int64)
    for req in batch.reqs:
        req._spec_seed_ready = True
    return GenerationBatchResult(logits_output=r.logits_output,
                                 next_token_ids=r.next_token_ids,
                                 can_run_cuda_graph=r.can_run_cuda_graph)
```
**Seeds must be per request.** Under continuous batching the batch composition changes every step; a
worker-global `self._seed[bs]` mixes requests and corrupts output. Keep them in a GPU-resident pool
indexed by `req_pool_idx` and gather in the current batch order.

### 5-3. DRAFT
```python
K = self.draft_token_num
n = K                                         # window length = engine K (§9-3: a mismatch kills the kernel)
seed = self._seeds()[batch.req_pool_indices]  # GPU->GPU gather

win = torch.full((bs, n), self.mask_id, device=dev, dtype=torch.int64)
win[:, 0] = seed                              # [seed | MASK x (K-1)]

within = torch.ones((n, n), dtype=torch.bool)
within[0, 1:] = 0            # the seed is causal (does not see the masks); masks attend to each other
mask = self._full_mask(batch, within)         # all-ones prefix ++ the n x n tail above

spec = NgramVerifyInput(win.flatten(), mask, pos, ri, nt, ns, n)
batch.forward_mode = ForwardMode.TARGET_VERIFY      # reuse the spec path
batch.spec_info = spec
spec.prepare_for_verify(batch, self.page_size)
self._round_cache_loc = batch.out_cache_loc         # <- shared with verify (§7-2)

r = self.target_worker.forward_batch_generation(batch.get_model_worker_batch(), is_verify=True)
logits = r.logits_output.next_token_logits.view(bs, n, -1)   # logit[i,j] -> win[i,j+1]
am = logits.argmax(-1)

draft = torch.empty((bs, K), dtype=torch.int64, device=dev)
draft[:, 0] = seed
draft[:, 1:] = am[:, : K - 1]                 # token shift: draft[:, 1] = a0, draft[:, 2:] = d_1..d_{K-2}
if self._fulldraft:
    self._last_dK = am[:, K - 1]              # d_{K-1}: the last mask's logit, consumed by the full-draft verify
```
Viewing `logits` as `[bs, n, V]` requires the model to return logits at **every** position → §6-1.
With `SELFSPEC_FASTATTN=1` the worker builds no mask (`mask = None`) and tags the forward with
`spec.selfspec_role = "draft"` (`"verify"` in §5-4) instead; the attention backend derives the same pattern (§7-1).

### 5-4. VERIFY — full-draft round (`SELFSPEC_FULLDRAFT=1`, default)
The verify window is `[a0, d_1..d_{K-1}]`: the boundary token x0 is **not** re-forwarded. Its KV slot from
the draft forward is committed as is (row 0 of the draft was causal, so that K/V is already exact) and the
window moves one position to the right, so the forward keeps the K-token shape (cuda-graph compatible).
```python
d  = draft_tokens.view(bs, K)
a0 = d[:, 1]                                                          # AR prediction after x0
verify_tokens = torch.cat([d[:, 1:], self._last_dK.view(bs, 1)], dim=1).flatten()   # [a0, d_1..d_{K-1}]
pos         = self._positions(batch, K, extra=1)                      # p0+1 .. p0+K
custom_mask = self._full_mask(batch, torch.tril(torch.ones((K, K), dtype=torch.bool)), extra=1)

# KV slots: the draft's slots [1:K] plus ONE new slot at position p0+K  -> K+1 slots per request per round
# (reserved in ScheduleBatch.new_tokens_required_next_decode, see §6-2)
rl         = self._round_cache_loc.view(bs, K)
new_slots  = alloc_token_slots(batch.tree_cache, bs)
verify_loc = torch.cat([rl[:, 1:], new_slots.view(bs, 1).to(rl.dtype)], dim=1).flatten()
batch.req_to_token_pool.req_to_token[batch.req_pool_indices.long(), (batch.seq_lens + K).long()] = new_slots
batch.seq_lens.add_(1); batch.seq_lens_cpu.add_(1); batch.seq_lens_sum += bs   # x0 is committed

spec = _FullDraftVerifyInput(verify_tokens, custom_mask, pos, ri, nt, ns, K); spec.a0 = a0
batch.forward_mode = ForwardMode.TARGET_VERIFY; batch.spec_info = spec
batch.input_ids = verify_tokens; batch.out_cache_loc = verify_loc
batch_result = self.target_worker.forward_batch_generation(mwb, is_verify=True)
logits_output, vid, _ = mwb.spec_info.verify(batch, batch_result.logits_output, self.page_size, None)
```
`_FullDraftVerifyInput` subclasses `NgramVerifyInput` and overrides two methods:
- `_fill_requests` appends a0 to `req.output_ids` (and checks finish) **before** the verified tokens. If a0 is
  EOS or hits the length limit, the request ends at a0 and every verify token is dropped — the same mechanism
  SGLang uses for an EOS in the middle of the candidates (`accepted_indices` set to -1), applied to the whole row.
- `_free_cache` adds 1 to `kv_committed_len` for the x0 slot; otherwise the radix cache leaks that slot when
  the request finishes.

The worker then splices a0 in front of each request's returned tokens (ragged) and reports
`accept_length + 1` to the scheduler; `spec_verify_ct` stays 1 per round, so the engine metric in §8-1 is
unchanged. `return_logprob` is not supported on this path (a0's logit row lives in the draft forward) — RL
rollouts that need logprobs run with `SELFSPEC_FULLDRAFT=0`.

### 5-4b. VERIFY — legacy round (`SELFSPEC_FULLDRAFT=0`)
Verify over the draft as is (`[x0, a0, d_1..d_{K-2}]`), reusing exactly the K slots the draft allocated:
```python
within = torch.tril(torch.ones((K, K), dtype=torch.bool))     # chain-causal
custom_mask = self._full_mask(batch, within)

spec = NgramVerifyInput(draft_tokens, custom_mask, pos, ri, nt, ns, K)
batch.forward_mode = ForwardMode.TARGET_VERIFY
batch.spec_info = spec

# lightweight reuse: keep the slots the draft allocated (reallocating changes kv_indices and forces another plan)
if self._round_cache_loc is not None and self.page_size == 1:
    batch.input_ids     = spec.draft_token
    batch.out_cache_loc = self._round_cache_loc
else:
    spec.prepare_for_verify(batch, self.page_size)

batch_result = self.target_worker.forward_batch_generation(mwb, is_verify=True)
logits_output, next_token_ids, num_accepted = mwb.spec_info.verify(
    batch, batch_result.logits_output, self.page_size, None)
```
In both rounds `verify()` does the rest: greedy accept → update `req.output_ids` + `check_finished` → free
rejected slots and compact → update `req_to_token` → **slide `seq_lens += accept+1`**.

### 5-5. Next seed — ragged!
`verified_id` is a **flat concatenation** with a different length per request; do not view it as `[bs, -1]`.
```python
vid    = verify_input.verified_id.view(-1)
counts = verify_input.accept_length.to(torch.int64) + 1
ends   = torch.cumsum(counts, 0) - 1
self._seeds()[batch.req_pool_indices] = vid[ends].to(torch.int64)
```
In the full-draft round each request returns a0 followed by `accept_length + 1` verified tokens, so
`counts = accept_length + 2`; the seed is still the last returned token.
Finally restore `batch.forward_mode = ForwardMode.DECODE`.

---

## 6. The 24 modified files — what, why, and what is model-specific

### 6-1. Model side (MUST be adapted to your model)

**`models/glm4v.py`, `models/glm_ocr.py`** — logits at every position. Without this there is no draft.
```python
self.logits_processor = LogitsProcessor(config, return_full_logits=True)
```

**`dllm/config.py`** — architecture registration (block size and mask id). The self-spec worker reads it only when
`--dllm-algorithm` is passed; otherwise the mask id comes from `SELFSPEC_MASK_ID` or the checkpoint's
`block_diffusion.json` (§9-1).
```python
"GlmOcrForConditionalGeneration": {"block_size": 32, "mask_id": 59282},
```

**Token-shift convention** — our model is trained so that `logits[j]` predicts token `j+1`.
A model trained without the shift (`logits[j] -> token j`, SDAR/LLaDA style) indexes differently.
Check your training loss and match it.

**Draft window mask** `within[0,1:] = 0` — keeps the seed causal. Change it if the seed was
bidirectional during training.

**The M-RoPE dllm branch in `model_executor/forward_batch_info.py`** — the part that took longest.
One dllm forward covers both a prefill chunk (image tokens → 2-D grid M-RoPE) and a decode block
(plain text), so positions are split:
- `p < prompt_len` → slice of the precomputed `mm_input.mrope_positions[:, p]` (image-aware)
- `p >= prompt_len` → last prompt mrope + 1 per token (text continuation)

The earlier bug applied text positions to the image chunk and double-counted the image span, producing
garbage or output that started mid-response. **A text-only model does not need this branch at all.**
In the same file, computing `positions` from `extend_seq_lens` instead of a hard-coded `block_size` is
mandatory (a prefill chunk longer than block_size otherwise runs out of positions → illegal memory
access in rotary/KV).

### 6-2. Algorithm registration (copy almost verbatim)

**`speculative/spec_info.py`**
```python
SELFSPEC_DIFFUSION = auto()
def is_selfspec_diffusion(self): return self == SpeculativeAlgorithm.SELFSPEC_DIFFUSION
# inside create_worker:
elif self.is_selfspec_diffusion():
    if enable_overlap:
        raise ValueError("SELFSPEC_DIFFUSION does not support overlap worker creation.")
    from sglang.srt.speculative.selfspec_diffusion_worker import SelfSpecDiffusionWorker
    return SelfSpecDiffusionWorker
```

**`server_args.py`** — what is forced here is the constraint set.
```python
self.speculative_draft_model_path = None   # draft == target
self.disable_overlap_schedule = True       # cannot coexist with the overlap scheduler
self.enable_mixed_chunk = False
self.speculative_eagle_topk = 1            # a linear chain, not a tree
speculative_num_draft_tokens default 32 (= training block size) / max_running_requests default 48
# The fallback 32 only applies when --speculative-num-draft-tokens is omitted; serve/serve_sglang_ocr.sh
# passes DRAFT=33 (= block size + 1, §9-2).
```
Add `"SELFSPEC_DIFFUSION"` to the `--speculative-algorithm` `choices`.

**`model_runner.py`** — three enum registrations (required): KV-pool ownership (no draft model, so the
target owns everything) and a dummy `NgramVerifyInput` for cuda-graph capture (otherwise the graph is
captured in the EAGLE shape and mismatches replay).

**`scheduler.py`** — `draft_token_to_kv_pool = None`: the draft is the target, a second KV pool is waste.

**`managers/schedule_batch.py`** — `new_tokens_required_next_decode` reserves K+1 KV slots per request per
round when `SELFSPEC_FULLDRAFT=1` (the full-draft verify allocates one new slot on top of the draft's K).
Without it the scheduler can admit a batch whose verify allocation then fails under KV pressure.

**`ngram_info.py`** — three generic fixes: `import os`; a device-derivation fallback when
`custom_mask=None` (otherwise the constructor raises `AttributeError` and the scheduler dies); caching
`accept_length_cpu` (one D2H per round saved).

**`speculative/eagle_draft_cuda_graph_runner.py`** — keeps the stock EAGLE draft graph runner working with the
patched `CudaGraphRunner.capture` signature (the graph-runner changes themselves are §7).

### 6-3. Runtime optimizations (optional; §7-5)
Each lever is gated by its own flag, off in the code; skip all of them for a first port and add them once Gate 1
(§11) passes on the plain runtime.

| lever | files |
|---|---|
| `SELFSPEC_FASTATTN` | `speculative/selfspec_fast_attn.py` (new), `layers/attention/flashinfer_backend.py` (per-role planning, the draft's causal re-run), `model_executor/cuda_graph_runner.py` (graphs keyed by (bs, role)) |
| `SELFSPEC_MROPE_FIX` | `model_executor/forward_batch_info.py`, `model_executor/model_runner.py`, `model_executor/cuda_graph_runner.py` |
| `SELFSPEC_FAST_COMMIT` / `_HOSTLEN` / `_DEV_POS` | `speculative/selfspec_fast_commit.py` (new), the worker, `managers/schedule_batch.py`, `managers/scheduler_output_processor_mixin.py`, `managers/utils.py` (host token lists + `finish_checked` on `GenerationBatchResult`) |
| `SELFSPEC_DRAFT_AHEAD` | the worker, `managers/scheduler.py` (binding, release before prefill admission and `pause_generation`), `managers/scheduler_update_weights_mixin.py` (release before weight updates) |
| `SS_GLM_VIT_GRAPH_MAXM` | `multimodal/glm_vit_graph_runner.py` (new), `models/glm_ocr.py`, `layers/attention/vision.py` |
| `SS_GLM_PV_BF16` | `multimodal/processors/glm4v.py`, `multimodal/processors/base_processor.py` |

What is model-specific: the ViT graphs and bf16 pixels are written for the GLM-OCR vision encoder and processor, and
the M-RoPE fix matters only for M-RoPE models. The attention, commit and draft-ahead levers are model-independent but
tied to FlashInfer (§7-5).

### 6-4. Do not port
The profiling / dump blocks (`DLLM_PROF`, `SELFSPEC_*_DUMP`, `SELFSPEC_TIME`, `SELFSPEC_NOSYNC`, `DUMP_POS` in
`models/glm4v.py`), the decode-latency fields added in `managers/tokenizer_manager.py`, and the
`seq_lens_cpu` wiring in `flashinfer_backend.py` (§7-4). The `dllm/mixin/*`, `schedule_policy.py` and
`radix_cache.py` changes belong to an older experimental path gated on `req.dllm_config`; they are
inactive with `--speculative-algorithm SELFSPEC_DIFFUSION` and can be dropped.

---

## 7. cuda graphs and attention — how it actually runs

### 7-1. Two attention paths: mask-free (served) and custom mask (reference)
Draft and verify are both `TARGET_VERIFY` forwards of K tokens per request; only the attention pattern differs.

**Mask-free path — `SELFSPEC_FASTATTN=1`** (what the launcher runs with `FAST=1`, its default). No mask tensor is
built, copied or packed. The worker tags each forward with a role and the FlashInfer backend expresses the pattern
with built-in mask modes over the unchanged request layout (one request per sequence):
- verify runs `CAUSAL`. FlashInfer aligns a causal mask bottom-right, so row i sees the prefix plus window rows
  0..i — exactly all-ones prefix ++ `tril(K, K)`;
- draft runs the same plan `NON_CAUSAL` (the mask rows see prefix + seed + all masks), then once more `CAUSAL`;
  row 0 of that second run (the seed: prefix + seed only) replaces row 0.

`speculative/selfspec_fast_attn.py` plans FlashInfer directly from lengths the host already knows, and the graph
runner captures one graph per (bs, role). The output is bit-identical to the mask path: the plan and tile
configuration are the ones the custom mask uses, so every row reads the same KV chunks in the same merge order and
masked keys contribute the same exact zeros (compared with the mask path call by call, `SELFSPEC_SHADOW=1`).
Cheaper ways to get the seed row — a second sub-request in the same call, or its own one-row call — change
FlashInfer's KV splitting and are not bit-identical; the second causal run is what keeps row 0 exact.
Requirements: `--attention-backend flashinfer`, flashinfer 0.6.7.post2 with the fa2 prefill backend (the path calls
FlashInfer's planner directly; both are checked at startup), no `--enable-dp-attention`.

**Custom-mask path — `SELFSPEC_FASTATTN=0`** (the code default; what `FAST=0` runs). The direction of draft
(bidirectional) and verify (causal) is expressed by the **contents of `custom_mask`**. flashinfer runs in
`MaskMode.CUSTOM`, the real mask is copied into the buffer at replay, so one graph per bs handles both
(`SELFSPEC_MASK=1`, default). This is the reference the mask-free path is checked against; port it first.

> The tree also contains an older dual-graph (`(bs, causal)` key) implementation for `SELFSPEC_NOMASK=1`. It is a
> different, failed experiment, not the mask-free path above. It is OFF by default and recorded as worse on both
> correctness and speed: combined with cuda graphs it read a stale captured mask buffer and its output diverged from
> AR. It is refused at startup together with `SELFSPEC_FASTATTN=1` or the full-draft round. When porting, do not use it.

### 7-2. draft and verify share KV slots
Previously draft allocated → cloned → freed and verify allocated again. Sharing makes `kv_indices`
bit-identical so verify skips its attention `plan()`. Correctness holds because the verify forward
overwrites the same slots with causal K/V (the draft's bidirectional K/V was throwaway). In the full-draft
round verify reuses slots [1:K] and allocates one new slot; slot 0 (x0) is committed straight from the
draft forward, whose row 0 was causal.

### 7-3. How the `causal` flag reaches attention (dual-graph path only)
```
worker         spec.selfspec_causal = False (draft) / True (verify)
   ↓
forward_batch_info.py    ret.dllm_ragged_causal = bool(_ss_causal)
   ↓
flashinfer_backend.py    wrapper.forward(causal=...)   (paged and ragged call sites)
   ↓
cuda_graph_runner.py     replay graphs[(bs, causal)]
```
Porting trap: the paged call site is gated on `is_dllm_model or selfspec_nomask` but the ragged one only
on `is_dllm_model`. Multimodal models take the paged path (`use_ragged=False`); **a text-only model takes
the ragged path and needs `or self.selfspec_nomask` there too.** `dllm_ragged_causal` is a dynamic
attribute on `ForwardBatch` (read with `getattr(..., False)` everywhere), so a typo passes silently —
declare it as a dataclass field when porting.

The mask-free path of §7-1 does not use this route: the role on `spec_info` selects the planner's per-role wrapper and
sets the causal flag in the forward metadata. Its wrappers are paged for every model, so the paged/ragged gating above
does not apply to it.

### 7-4. cuda-graph lessons (do not repeat)
1. `causal` is a Python scalar and is baked into the graph at capture → N modes = N graphs.
2. flashinfer's mask mode is decided by whether a buffer was passed to the **constructor**; it cannot be
   changed through `plan()`. A `custom_mask_buf` in the constructor pins `CUSTOM` forever.
3. Capturing two graph sets at the same `bs` overwrites `prefill_cuda_graph_metadata[bs]` (12/12 → 4/12
   identical pages when tried on the mask path).
4. **The eager fallback must be the return value of `can_run()`.** Nothing in SGLang reads
   `batch.can_run_cuda_graph`; a guard placed there was a no-op and the crash stayed exposed.
5. On the mask path, `custom_mask` has `sum_req K*(seq_len+K)` elements and **grows with the context**. Overflowing
   the cuda graph's fixed buffer kills the server with an illegal memory access at replay → route only the
   overflowing batches to eager in `can_run`. (The mask-free path has no mask buffer to overflow.)
6. Multimodal prefill must run eager. The vision encoder / `input_embeds` scatter is not in the graph;
   replaying an image-less graph for an image prefill fails with a gather index out of bounds.

### 7-5. Runtime optimizations (`FAST=1`)
At this model size the unoptimized runtime is bound by host work — kernel launches, blocking copies, Python
bookkeeping — rather than by GPU compute, in AR and spec mode alike. The levers below remove that work. Each one is
off in the code and gated by its own flag, and each produces bit-identical output to the unoptimized runtime (the
same token ids in AR and spec mode). `serve/serve_sglang_ocr.sh` turns all of them on with `FAST=1`, its default;
`FAST=0` serves the unoptimized runtime. Flags and defaults: §10.

- **Mask-free attention** (`SELFSPEC_FASTATTN=1`) — §7-1. Removes mask construction, the mask copy and its packing
  from every forward; FlashInfer is planned from host-known lengths without a device→host copy; one cuda graph per
  (bs, role).
- **M-RoPE delta cache fix** (`SELFSPEC_MROPE_FIX=1`) — the draft/verify M-RoPE positions add a per-request delta
  that SGLang caches on the device, but the cache check compared the device string `"cuda"` with `cuda:0` and always
  missed, so every forward did a blocking host→device copy that waited for the queued GPU work. The fix caches the
  delta per batch composition and builds the positions in one kernel; it also reads the per-round switches once.
- **GPU commit** (`SELFSPEC_FAST_COMMIT=1`, `SELFSPEC_HOSTLEN=1`, `SELFSPEC_DEV_POS=1`;
  `speculative/selfspec_fast_commit.py`) — one kernel commits the accepted tokens (next seed, sequence lengths,
  rejected KV slots); the host finishes the round — output ids, finish checks with the per-token loop's exact
  semantics, KV accounting — from one pinned readback plus one event sync. Sequence lengths are kept exact on the
  host, and window positions are computed on the GPU. All three need `SELFSPEC_FASTATTN=1`; `FAST_COMMIT` also needs
  the full-draft round.
- **Draft ahead** (`SELFSPEC_DRAFT_AHEAD=1`) — right after round t's readback, when no request can finish in this
  round, round t+1's draft is launched first, and round t's host bookkeeping and the scheduler's output processing
  run under its GPU time. The next call uses the pre-launched draft only if the batch is unchanged; otherwise its KV
  slots are freed and a fresh draft runs. A pre-launched draft is also released before a prefill admission, a weight
  update or `pause_generation`. Rounds with logprobs, hidden states, a grammar or a custom logit processor, and
  outputs carrying `customized_info`, run without it. Needs the four flags above (startup error otherwise); it
  switches itself off with a warning under `--dllm-algorithm`, disaggregation, the prefill delayer, `pp_size > 1`,
  `tp_size > 1` or n-gram embeddings.
- **ViT CUDA graphs** (`SS_GLM_VIT_GRAPH_MAXM=<M>`, the launcher uses 2048; `multimodal/glm_vit_graph_runner.py`) —
  the GLM-OCR vision encoder is captured at startup as one CUDA graph per patch count up to M and replayed for a
  single image of a captured size; other calls run eager. Every captured size is compared with the eager forward at
  capture (`SS_GLM_VIT_GRAPH_CHECK`, default 1) and falls back to eager on a mismatch. Needs the FA3 vision attention
  backend (SGLang's default on Hopper; otherwise it switches itself off). Cost: longer startup and a few GB of GPU
  memory, taken from the KV cache.
- **bf16 pixels** (`SS_GLM_PV_BF16=1`) — the processor casts `pixel_values` to bf16 on the GPU, halving the bytes
  moved through D2H, shared memory, zmq and H2D. The values are unchanged: the model casts them to bf16 anyway
  (bf16 models only).
- **Server level** — `--tokenizer-worker-num 2` preprocesses regions in parallel processes, and
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `TORCH_NUM_THREADS=1` inside the server stop image preprocessing from
  oversubscribing every core.

---

## 8. Instrumentation — reading tok/fwd correctly

### 8-1. The engine metric (mind the conversion)
`observability/scheduler_metrics_mixin.py`
```python
self.spec_num_accepted_tokens += num_accepted_tokens + bs   # numerator: committed tokens (bonus included)
self.spec_num_forward_ct      += bs                          # denominator: verify rounds
```
`avg_spec_accept_length` = committed tokens per round (a0 and the bonus token already included). Two forwards per round:
```
tok/fwd = avg_spec_accept_length / 2        <- correct
tok/fwd = (avg_spec_accept_length + 1) / 2  <- wrong (double-counts the bonus)
```
It is not printed in the normal log (only when server args are updated); read it through
`Engine.get_server_info()["internal_states"]` when the engine is driven through the Python API.

### 8-2. Safer: count directly
Count generated tokens / decode forwards yourself. In the HF path hook the first decoder layer
(`tools/bench_forwards.py`). Do not hook at the `text_model` level — some paths iterate the layers
directly and the hook never fires.

### 8-3. Always state the page set
**tok/fwd is capped by output length.** A round can commit at most K+1 tokens (K in the legacy round), so a 10-token output
yields at most 10 tokens from that round regardless of draft quality. The same model and K give very
different tok/fwd on short-crop and long-crop sets. Wall clock depends on length for another reason:
the fixed vision + prefill cost dominates short outputs. Never compare numbers across page sets.

---

## 9. Porting traps

### 9-1. A silently wrong mask_id — the most dangerous
A wrong mask id does not crash: verification keeps the output correct and only acceptance collapses, which is
hard to diagnose. The worker therefore resolves the mask id at startup in this order, with no silent fallback:
1. `SELFSPEC_MASK_ID` (environment);
2. `<model_path>/block_diffusion.json` — its `mask_id`, plus `bd_size` when present. Only a local checkpoint
   directory has this file; a hub id does not;
3. the `--dllm-algorithm` config (`DllmConfig`, i.e. the architecture map in `dllm/config.py`);
4. none of these → startup error.

A mask id outside the vocabulary is also a startup error. When `bd_size` is known and `--speculative-num-draft-tokens`
is not `bd_size + 1`, the worker logs a warning and still runs (§9-2). Our launcher relies on (2) for a checkpoint
directory and sets `SELFSPEC_MASK_ID=59282` (GLM-OCR's `<|mask|>`) only when `CKPT` has no local
`block_diffusion.json`, e.g. a hub id.
→ When porting, ship `block_diffusion.json` (`mask_id`, `bd_size`) with your checkpoint or set `SELFSPEC_MASK_ID`, and
check the startup line `[selfspec-diffusion] K=.. mask_id=.. (from ..) bd_size=..`, which names where the id came from.

### 9-2. K = training block + 1
With a fully masked training block of `bd_size=32` and a token-shift loss, each mask attends to **32
masks** in training. The inference window `[seed | MASK x (K-1)]` has K-1 masks → **K = bd_size + 1 = 33**
is aligned. K=33 gives consistently (slightly) higher tok/fwd than K=32; K larger than the training
block is still lossless but acceptance saturates. When `block_diffusion.json` gives `bd_size`, the worker warns at
startup if K ≠ `bd_size + 1`.

### 9-3. window length ≠ engine K kills the kernel
```
RuntimeError: Tensor match failed for Tensor<33> @ jit_kernel/csrc/elementwise/kvcache.cuh:170
```
The KV-cache write JIT kernel is specialized on tensor size (not a cuda-graph issue; happens with
GRAPH=0 too). To enlarge the window, raise the engine K itself (`--speculative-num-draft-tokens 33`).

### 9-4. `--page-size 1` is required
The KV advance is partial (accept length), which misaligns with pages larger than 1. The full-draft verify
also asserts `page_size == 1` (it writes its one new slot into `req_to_token` directly).

### 9-5. No overlap scheduler
Blocked explicitly in `spec_info.py`; `server_args` forces `disable_overlap_schedule=True`. Host/GPU overlap
comes from inside the worker instead (`SELFSPEC_DRAFT_AHEAD`, §7-5).

### 9-6. temp > 0 uses a separate verify path
`verify()` branches on `sampling_info.is_all_greedy`: greedy → `_greedy_verify`; temp > 0 →
`_sampling_verify` (`tree_speculative_sampling_target_only`, `threshold` default 1.0 = distribution-
preserving lossless sampling). Acceptance does not collapse at temp 0.8, so it is usable for RL rollouts.

### 9-7. repetition_penalty makes spec ≠ AR
SGLang's speculative penalty is "relaxed": all K block positions get the pre-block penalty state. AR at
position r also includes block[0..r], so outputs diverge on pages with repetition.
`SELFSPEC_EXACT_PENALTY=1` adds the missing triangular share and matches AR exactly (default OFF).
**All "lossless" checks were done with the penalty OFF.**

### 9-8. Environment — cu129
A CUDA 12.8 driver runs cu129 builds (they link `libcudart.so.12`, minor-version compatible); the default
PyPI wheels are cu130 and fail with "NVIDIA driver too old". There is no cu128 wheel.
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0   # else the FlashInfer sampler JIT looks for ninja and dies
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=<venv>/bin:$PATH           # cuda-graph capture shells out to ninja
```
`env/sglang_venv_freeze.txt` is the tested environment.

---

## 10. Environment variables

Reproduce this combination first (the unoptimized runtime, `FAST=0`): `SELFSPEC_FULLDRAFT=1` (default),
`SELFSPEC_MASK=1` (default), `SELFSPEC_NOMASK=0`, `SELFSPEC_EXACT_PENALTY=0`, cuda graphs ON. Then add the
runtime optimizations (§7-5); the output must not change.

| env | default | purpose |
|---|---|---|
| `SELFSPEC_FULLDRAFT` | `1` | full-draft round (§2): every mask logit + the bonus token, up to K+1 tokens per round. `0` = legacy round (≤ K) |
| `SELFSPEC_MASK_ID` | unset | mask token id; takes precedence over `block_diffusion.json` and `--dllm-algorithm` (§9-1) |
| `SELFSPEC_POS_OFFSET` | `0` | constant added to the window's positions. `0` = AR-aligned (x0 at the position AR decoding gives that token). GLM-OCR checkpoints are served with `1`, the positions the released numbers were produced with; the launcher sets it in spec mode |
| `SELFSPEC_MASK` | `1` | custom-mask path: verify with custom_mask. **The reference path**; ignored when `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_FASTATTN` | `0` | mask-free draft/verify attention, one cuda graph per (bs, role) (§7-1). flashinfer backend only |
| `SELFSPEC_MROPE_FIX` | `0` | fixes the M-RoPE delta cache that always missed (removes a blocking copy per forward); reads the per-round switches once (§7-5) |
| `SELFSPEC_FAST_COMMIT` | `0` | commit on the GPU, host bookkeeping from one pinned readback (§7-5). Needs `SELFSPEC_FASTATTN=1`, `SELFSPEC_FULLDRAFT=1` |
| `SELFSPEC_HOSTLEN` | `0` | sequence lengths kept exact on the host (no device→host sum). Needs `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_DEV_POS` | `0` | window positions computed on the GPU. Needs `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_DRAFT_AHEAD` | `0` | launch the next round's draft before this round's host bookkeeping (§7-5). Needs the four flags above |
| `SS_GLM_VIT_GRAPH_MAXM` | `0` (off) | GLM-OCR ViT as one CUDA graph per patch count up to this value; other sizes run eager (§7-5) |
| `SS_GLM_VIT_GRAPH_CHECK` | `1` | compare every captured ViT graph with the eager forward at capture; a mismatching size runs eager |
| `SS_GLM_PV_BF16` | `0` | the processor casts `pixel_values` to bf16 on the GPU (bf16 models only) |
| `SELFSPEC_CHECK` | `0` | diagnostic: per-forward host/device length, position and plan checks; raises on a mismatch (syncs; never time with it) |
| `SELFSPEC_SHADOW` | `0` | diagnostic, with `SELFSPEC_FASTATTN=1`, eager forwards only: also run the mask path and compare every attention call bitwise |
| `SELFSPEC_PADFIX` | `1` | custom-mask path only: pad the mask for the dummy rows of a padded graph replay |
| `SELFSPEC_NOMASK` | `0` | older mask-free dual-graph path, not `SELFSPEC_FASTATTN`. Measured worse (correctness and speed) — experimental; refused with `SELFSPEC_FASTATTN=1` or `SELFSPEC_FULLDRAFT=1` |
| `SELFSPEC_TRAIN_MASKS` | `0` | window K+1. Kills the kernel (§9-3); refused at startup with `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_EXACT_PENALTY` | `0` | exact AR match when repetition_penalty is used (§9-7) |
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `0` | force greedy verify even at temp > 0 (A/B) |
| `SELFSPEC_DEBUG` | `0` | per-round accept + cumulative tok/fwd |
| `SELFSPEC_TIME` / `SELFSPEC_NOSYNC` | `0` | per-phase timers / launch-bound diagnosis |
| `SELFSPEC_ACCEPT_STATS` | `""` | path → per-position confidence + accept as jsonl |
| `SELFSPEC_ROUND_HIST` | `0` | histogram of tokens committed per round, printed every 200 rounds |
| `SELFSPEC_{LOGIT,DRAFT,VERIFY}_DUMP` | `0` | dump the first rounds |
| `DLLM_PROF` / `DLLM_GRAPH_DBG` | `0` | forward / graph instrumentation |

Capture, runtime and backend must read the **same** flag; splitting it in two was a real bug.

The launcher sets these through one knob. `FAST=1` (default) exports, in spec mode, `SELFSPEC_FASTATTN`,
`SELFSPEC_MROPE_FIX`, `SELFSPEC_FAST_COMMIT`, `SELFSPEC_HOSTLEN`, `SELFSPEC_DEV_POS` and `SELFSPEC_DRAFT_AHEAD` = `1`, and
in both modes `SS_GLM_VIT_GRAPH_MAXM=2048`, `SS_GLM_PV_BF16=1`, `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
`TORCH_NUM_THREADS` = `1` (`OMP_NUM_THREADS_SERVER` overrides the value) and `--tokenizer-worker-num 2`
(`TOKENIZER_WORKERS`). `FAST=0` exports none of them. A `SELFSPEC_*` / `SS_*` flag already set in the environment
wins, and `EXTRA_ARGS` is appended to `launch_server`. In spec mode the launcher also exports `SELFSPEC_FULLDRAFT=1`
and `SELFSPEC_POS_OFFSET=1`, and `SELFSPEC_MASK_ID=59282` when `CKPT` has no local `block_diffusion.json`.

---

## 11. Verifying a port (in this order)

Separate correctness from speed; mixing them hides the cause.

1. **Gate 1 — output equals AR.** Run the same prompts with `MODE=ar` and `MODE=spec` and compare
   **token ids** (text comparison is confounded by re-tokenization). On a mismatch check the penalty
   (§9-7) and whether the mask path is on (§7-1). Pass it with `FAST=0` first, then again with the §7-5 flags on;
   `SELFSPEC_CHECK=1` and `SELFSPEC_SHADOW=1` localize a mismatch on the optimized path.
2. **Gate 2 — acceptance is real.** `SELFSPEC_DEBUG=1` prints per-round accept. Values stuck at 1–2
   mean the draft is garbage: mask_id (§9-1) or K alignment (§9-2).
3. **Gate 3 — speed.** Only then measure tok/fwd and tok/s.
