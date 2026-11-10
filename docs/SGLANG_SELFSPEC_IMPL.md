# Implementing block-diffusion self-speculative decoding in SGLang

This document plus the patch in `patches/sglang/` is meant to be enough to implement the same
decoding scheme for another model or task. It describes what code you have to write, not what we did.

Shipped artifacts — `patches/sglang/`

| file | content |
|---|---|
| `sglang_BASE_COMMIT.txt` | upstream SGLang commit the patch applies to (`1519acf`, tag v0.5.10) |
| `sglang_0.5.10_selfspec.patch` | changes to 21 existing source files plus `python/pyproject.toml` — `git apply` |
| `python/sglang/srt/speculative/selfspec_diffusion_worker.py` | the new worker (the core of the implementation) |
| `python/sglang/srt/speculative/selfspec_sampling.py` | AR-exact sampling, penalties and grammar masks for the verify rows (§9-6, §9-7) |
| `python/sglang/srt/speculative/selfspec_commit_rules.py` | the host-side commit rules (finish checks with the per-token loop's semantics), dependency free |
| `python/sglang/srt/speculative/selfspec_fast_attn.py` | mask-free draft / verify attention planning for FlashInfer (`SELFSPEC_FASTATTN`, §7-1) |
| `python/sglang/srt/speculative/selfspec_fast_commit.py` | GPU-side commit + single-readback host bookkeeping (`SELFSPEC_FAST_COMMIT`, §7-5) |
| `python/sglang/srt/multimodal/glm_vit_graph_runner.py` | CUDA graphs for the GLM-OCR vision encoder (`SS_GLM_VIT_GRAPH_MAXM`, §7-5) |
| `python/sglang/srt/entrypoints/http_worker_keys.py` | how the API keys reach every tokenizer worker (§6-5) |

`selfspec_fast_attn.py`, `selfspec_fast_commit.py` and `glm_vit_graph_runner.py` implement runtime optimizations only
(§7-5). They are imported unconditionally but act only under their environment flags, which are off in the code; the
output with them on is bit-identical to the output with them off.

Check: applying the patch and copying the new files onto a clean checkout of the base commit reproduces our source tree
file for file.

---

## 1. Prerequisite — can your model do this?

Both are required:

1. **Block-diffusion training** — the model fills K masked slots at once, bidirectionally (non-causal).
2. **An AR (causal) head** — the same weights must also predict the next token causally.

Diffusion drafts, AR judges. Without the AR head there is no reference to verify against, and nothing
ties the output to AR decoding. Our checkpoints train both objectives jointly (`ar_loss_weight > 0`).
A pure diffusion model cannot use this scheme.

---

## 2. The algorithm (one round)

```
state: committed tokens are in the KV cache (causal). x0 = the boundary token: the last token committed
       by the previous round (its own K/V is not in the cache yet). B = training block size (32), K = B + 1.

(1) DRAFT    window win = [x0 | MASK x B]                  (length K)
             one bidirectional forward; x0's row is causal, the masks attend to each other
             logits[j] predicts win[j+1]                    (token shift)
             a0       = argmax(logits[0])                   the AR prediction after x0 (x0's row is causal, so it is exact);
                                                            with sampling / penalties / grammar: SGLang's sampler on logits[0]
             d_1..d_B = argmax(logits[1..B])                one draft token per mask; the LAST mask's logit predicts the
                                                            token right after the window, so every mask logit is used

(2) VERIFY   one causal forward over [committed | a0, d_1..d_B]      (K positions again -- same shape as the draft)
             x0's K/V from the draft forward is kept (it was computed causally), so x0 is not re-forwarded
             accept the longest prefix of d_1..d_B that matches AR greedy;
             at the first mismatch take the AR token instead; if everything matched take the verifier's
             prediction after d_B                                        (= bonus token)
             (sampling: speculative sampling against AR's distribution instead of the greedy match, §9-6)

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

What (2) reproduces is the AR head at the positions the windows are given. With `SELFSPEC_POS_OFFSET=0` those are
AR decoding's (and training's) positions, and verify computes what AR decoding computes, up to floating point. The
launcher's `SELFSPEC_POS_OFFSET=1`, the layout the released self-spec results were produced with, puts every generated
token one RoPE position later, so spec then verifies against the AR head under that shifted layout and can differ
from AR decoding where the shift changes the choice (§10). Either way, the verify forward scores K positions with other
kernels and batch shapes than AR's one-token steps, so a bf16 near-tie can resolve differently.

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
git apply $DOCR_ROOT/patches/sglang/sglang_0.5.10_selfspec.patch
cp -r $DOCR_ROOT/patches/sglang/python .          # adds the seven new files (table above)
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
`SELFSPEC_MASK_ID=59282` (§9-1). `SELFSPEC_POS_OFFSET=1` gives the positions GLM-OCR checkpoints are served with, one
RoPE position after AR decoding and training for every generated token (§2, §10); `0` is the AR layout.
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
        # SELFSPEC_MASK_ID > <model_path>/block_diffusion.json > error   <- §9-1
        self.mask_id, self._bd_size, _ = _resolve_block_diffusion(server_args)
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
`am[:, 0]` is a0 only for a batch that is greedy and uses no penalty, `logit_bias`, grammar or custom logit processor
(the common case, which runs exactly the code above). For any other batch a0 is drawn by SGLang's own sampler from
`logits[:, 0]` after the processing AR decoding applies (§9-6, §9-7).
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
  A token that the request's grammar rejects ends that request with an abort, as in AR decoding, instead of raising
  inside the scheduler.
- `_free_cache` adds 1 to `kv_committed_len` for the x0 slot; otherwise the radix cache leaks that slot when
  the request finishes.

The worker then splices a0 in front of each request's returned tokens (ragged) and reports
`accept_length + 1` to the scheduler; `spec_verify_ct` stays 1 per round, so the engine metric in §8-1 is
unchanged. `return_logprob` is not supported on this path (a0's logit row lives in the draft forward): the worker
raises on such a batch, inside the scheduler, which stops the server. Serve logprob requests with `MODE=ar`; RL
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

## 6. The modified files — what, why, and what is model-specific

### 6-1. Model side (MUST be adapted to your model)

**Logits at every window position.** Draft and verify are both `TARGET_VERIFY` forwards, for which SGLang's
`LogitsProcessor` already returns the logits of every position, so GLM-OCR needs no model change for it. A model whose
logits processor prunes `TARGET_VERIFY` rows would need one; check this first when porting.

**Token-shift convention** — our model is trained so that `logits[j]` predicts token `j+1`.
A model trained without the shift (`logits[j] -> token j`, SDAR/LLaDA style) indexes differently.
Check your training loss and match it.

**Draft window mask** `within[0,1:] = 0` — keeps the seed causal. Change it if the seed was
bidirectional during training.

**Positions.** The worker gives the draft and verify windows 1-D positions (§10 `SELFSPEC_POS_OFFSET`), and SGLang's
stock speculative M-RoPE branch (`_compute_spec_mrope_positions` in `model_executor/forward_batch_info.py`) adds each
request's M-RoPE delta, so an image prompt keeps its 2-D grid positions and the response continues as text. A
text-only model needs nothing here. `SELFSPEC_MROPE_FIX` changes only how the deltas reach the GPU (§7-5).

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
Add the new member after the existing ones, so every upstream `SpeculativeAlgorithm` member keeps its value.

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

**`server_args.py`** also refuses `--dllm-algorithm` together with `SELFSPEC_DIFFUSION`: SGLang's diffusion-LLM path
(§3) and self-spec are separate designs.

**`model_runner.py`** — three enum registrations (required): KV-pool ownership (no draft model, so the
target owns everything) and a dummy `NgramVerifyInput` for cuda-graph capture (otherwise the graph is
captured in the EAGLE shape and mismatches replay).

**`cuda_graph_runner.py`** — capture calls `capture_one_batch_size` with the self-spec graph variants (per role, §7-1)
only for a self-spec runner; every other runner, the EAGLE-family draft runners that reuse `CudaGraphRunner.capture`
included, gets upstream's exact call.

**`scheduler.py`** — `draft_token_to_kv_pool = None`: the draft is the target, a second KV pool is waste.

**`managers/schedule_batch.py`** — `new_tokens_required_next_decode` reserves K+1 KV slots per request per
round when `SELFSPEC_FULLDRAFT=1` (the full-draft verify allocates one new slot on top of the draft's K).
Without it the scheduler can admit a batch whose verify allocation then fails under KV pressure.

**`ngram_info.py`** — two generic fixes: a device-derivation fallback when `custom_mask=None` (otherwise the
constructor raises `AttributeError` and the scheduler dies); caching `accept_length_cpu` (one D2H per round saved).
The self-spec verify inputs subclass `NgramVerifyInput`; a batch that is greedy and uses no penalty, `logit_bias`,
grammar or custom logit processor runs its `verify()` unchanged, any other batch goes through `selfspec_sampling.py`
first (§9-6, §9-7).

Every change on a code path that other servers share is conditional on `--speculative-algorithm SELFSPEC_DIFFUSION`,
except the security fixes of §6-5 and the GLM-OCR-only ViT / pixel options of §6-3, which are off unless their
environment flags are set. SGLang's diffusion-LLM modules (`srt/dllm/`, the radix cache and schedule policy) and the
EAGLE draft runners are not modified.

### 6-3. Runtime optimizations (optional; §7-5)
Each lever is gated by its own flag, off in the code; skip all of them for a first port and add them once Gate 1
(§11) passes on the plain runtime.

| lever | files |
|---|---|
| `SELFSPEC_FASTATTN` | `speculative/selfspec_fast_attn.py` (new), `layers/attention/flashinfer_backend.py` (per-role planning, the draft's causal re-run), `model_executor/cuda_graph_runner.py` (graphs keyed by (bs, role)) |
| `SELFSPEC_MROPE_FIX` | `model_executor/forward_batch_info.py`, `model_executor/model_runner.py`, `model_executor/cuda_graph_runner.py` |
| `SELFSPEC_FAST_COMMIT` / `_HOSTLEN` / `_DEV_POS` | `speculative/selfspec_fast_commit.py` and `speculative/selfspec_commit_rules.py` (new), the worker, `managers/schedule_batch.py`, `managers/scheduler_output_processor_mixin.py`, `managers/utils.py` (host token lists + `finish_checked` on `GenerationBatchResult`) |
| `SELFSPEC_DRAFT_AHEAD` | the worker, `managers/scheduler.py` (binding, release before prefill admission and `pause_generation`), `managers/scheduler_update_weights_mixin.py` (release before weight updates) |
| `SS_GLM_VIT_GRAPH_MAXM` | `multimodal/glm_vit_graph_runner.py` (new), `models/glm_ocr.py`, `layers/attention/vision.py` |
| `SS_GLM_PV_BF16` | `multimodal/processors/glm4v.py`, `multimodal/processors/base_processor.py` |

What is model-specific: the ViT graphs and bf16 pixels are written for the GLM-OCR vision encoder and processor, and
the M-RoPE fix matters only for M-RoPE models. The attention, commit and draft-ahead levers are model-independent but
tied to FlashInfer (§7-5).

### 6-4. Do not port
The profiling / dump blocks (`DLLM_PROF`, `DLLM_GRAPH_DBG`, `SELFSPEC_*_DUMP`, `SELFSPEC_TIME`, `SELFSPEC_NOSYNC`) and
the decode-latency fields that `managers/tokenizer_manager.py` adds to a self-spec server's responses. All of them act
only on a self-spec server.

### 6-5. Security fixes (every server built from the patch)
Three upstream v0.5.10 behaviours change for every server, self-spec or not:
- **Keys in the server info.** `/server_info`, `/get_server_info`, `Engine.get_server_info()`, the scheduler's internal
  state and the request / crash dumps returned or stored the API key and the admin key. They now return those fields
  (and `ssl_keyfile_password`) as `null`, and `ServerArgs` keeps them out of its `repr`, which SGLang logs at startup
  (`utils/auth.py` `redact_server_args`, `entrypoints/http_server.py`, `entrypoints/engine.py`, `managers/scheduler.py`,
  `managers/tokenizer_manager.py`, `server_args.py`).
- **Keys with several tokenizer workers.** Upstream installs the key check only with one tokenizer worker and refuses
  to start a worker when an API key is set. The check is now part of the module-level app that every uvicorn worker
  serves; each worker receives the keys inside the pickled object uvicorn starts it with
  (`entrypoints/http_worker_keys.py`), not through shared memory or the environment, and answers 503 until it has them.
- **The tokenizer workers' shared-memory segment.** It is created exclusively with mode 0600, a stale one left by this
  user is replaced, one owned by another user stops the startup, and a segment is unpickled only when this user owns
  it (`managers/multi_tokenizer_mixin.py`). It no longer carries the keys.

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
> AR. It is refused at startup together with `SELFSPEC_FASTATTN=1` or the full-draft round, and its graphs and causal
> flag exist only on a self-spec server. When porting, do not use it.

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
Porting trap: the flag is read only at the paged call site, and only when `selfspec_nomask` is set. Multimodal
models take the paged path (`use_ragged=False`); **a text-only model takes the ragged path and would need the same
override there.** `dllm_ragged_causal` is a dynamic attribute on `ForwardBatch` (read with `getattr(..., False)`
everywhere), so a typo passes silently — declare it as a dataclass field when porting.

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
- **M-RoPE delta cache** (`SELFSPEC_MROPE_FIX=1`) — the draft/verify M-RoPE positions add a per-request delta.
  SGLang's stock speculative branch stacks the deltas on the host and uploads them with a blocking host→device copy
  in every forward with images, which waits for the queued GPU work. The fix caches the deltas on the device per
  batch composition and builds the positions in one kernel; it also reads the per-round switches once. Without the
  flag the stock branch runs unchanged.
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
  update or `pause_generation`. Rounds with logprobs or hidden states, rounds of any batch that is not greedy or uses a
  penalty, `logit_bias`, a grammar or a custom logit processor (its a0 depends on state the bookkeeping updates), and
  outputs carrying `customized_info`, run without it. Needs the four flags above (startup error otherwise); it
  switches itself off with a warning under disaggregation, the prefill delayer, `pp_size > 1`, `tp_size > 1` or
  n-gram embeddings.
- **ViT CUDA graphs** (`SS_GLM_VIT_GRAPH_MAXM=<M>`, the launcher uses 2048; `multimodal/glm_vit_graph_runner.py`) —
  the GLM-OCR vision encoder is captured at startup as one CUDA graph per patch count up to M and replayed for a
  single image of a captured size. Other calls (larger crops, several images) run eager with the sequence offsets
  passed once as a list (`SS_GLM_VIT_EAGER_LIST=1`, default) instead of a host copy and a sync per block; `0` keeps
  the original eager forward for them. Every captured size, and a few shapes of the list path, are compared bitwise
  with the original eager forward at capture (`SS_GLM_VIT_GRAPH_CHECK`, default 1); a mismatch falls back to eager
  for that size. Needs the FA3 vision attention backend (SGLang's default on Hopper; otherwise it switches itself
  off). The settings are parsed strictly (a malformed value stops the startup with an error naming the variable) and
  logged once as `[ss-vit-graph] settings: ...`. A failed capture (out of memory, a CUDA or capture error) is not
  fatal: the runner releases what it captured, logs `[ss-vit-graph] OFF: capture failed (...)`, and the model keeps
  its eager forward. The runner serves only the thread that built it (the scheduler's); a call from another thread
  takes the eager forward. Cost: longer startup and about 3.3 GB of GPU memory, taken from the KV cache because the
  graphs are captured before SGLang sizes it (on an 80 GB H100, about 7.7% of the KV pool at
  `--mem-fraction-static 0.6`, 17% at 0.3).
- **bf16 pixels** (`SS_GLM_PV_BF16=1`) — the processor casts `pixel_values` to bf16 on the GPU, halving the bytes
  moved through D2H, shared memory, zmq and H2D. The values are unchanged: the model casts them to bf16 anyway
  (bf16 models only).
- **Server level** — `--tokenizer-worker-num 2` preprocesses regions in parallel processes (each worker checks the
  API keys, §6-5), and
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
3. neither → startup error.

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
block gives the same output but acceptance saturates. When `block_diffusion.json` gives `bd_size`, the worker warns at
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

### 9-6. Sampling: speculative sampling against AR's distribution
`verify()` branches on whether the batch is greedy. Greedy → `_greedy_verify` (argmax = AR's greedy rule), on logits
processed as in §9-7 when the batch uses penalties, `logit_bias` or a grammar. Otherwise the worker reproduces AR
sampling exactly (`selfspec_sampling.py`, `selfspec_fast_commit.sampling_verify`):
- **a0** is drawn by SGLang's own sampler from the draft's causal row 0, after the processing AR decoding applies
  (custom logit processors, penalties, grammar mask, `logit_bias`, then temperature, top-k, top-p, min-p).
- **Verify row j** (prefix = committed + a0 + d_1..d_{j-1}) gets the same processing for exactly that prefix, then
  `tree_speculative_sampling_target_only` with the deterministic draft: d_j is accepted with probability p_j(d_j); at
  the first rejection the token is drawn from p_j with d_j removed and renormalized; if every draft token is accepted,
  the bonus token is drawn from the last row. p_j is AR's sampling distribution for the configured sampling backend,
  `min_p` and FlashInfer's joint top-k / top-p rule included (upstream's speculative verify ignores `min_p` and
  applies top-p after renormalizing the top-k).
- The acceptance thresholds must stay at 1.0 (`--speculative-accept-threshold-single` / `-acc`); the worker refuses
  to start otherwise. Verify rows draw from the device RNG, as SGLang's EAGLE and n-gram verify do, so per-request
  seeds under `--enable-deterministic-inference` reproduce a0 but not the verify rows; the distribution is the same.

### 9-7. Penalties, token constraints and grammars match AR
SGLang's speculative verify is "relaxed" upstream: all K positions get the penalty state from before the block, and
spec-v1 workers must feed committed tokens to the penalizers themselves. The worker does it exactly:
- every committed token (the prefill token, a0, each accepted draft token and the correction) is counted by the
  penalizers in commit order, so repetition / frequency / presence penalties and `min_new_tokens` see what AR
  decoding sees;
- verify row j applies each penalizer for its own prefix (the tokens drafted before j in the same round included),
  in the orchestrator's order, plus `min_new_tokens` at that length, `logit_bias`, and custom logit processors
  called with a read-only request view whose output is that prefix;
- a grammar (`response_format` json_schema / regex / ebnf) masks a0 through the sampler and every verify row with
  the grammar state after its prefix (accept, fill the row, roll back; the grammar ends in its entry state). The
  backend needs rollback and the packed bitmask: xgrammar (the default) and llguidance qualify, others are refused
  at startup. A token the grammar still rejects ends that request with an abort, as in AR decoding.

Two upstream effects remain. With `repetition_penalty` together with frequency or presence penalties, the penalizer
order depends on the process, so two AR servers can differ too; spec follows its own process's order. And SGLang's
overlap scheduler counts each token for the penalties one step late, so AR with and without it differ; spec matches AR
with `--disable-overlap-schedule`.
None of this runs for a greedy batch without these options: it takes the code of §5 unchanged.

### 9-8. Environment — cu129
A CUDA 12.8 driver runs cu129 builds (they link `libcudart.so.12`, minor-version compatible); the default
PyPI wheels are cu130 and fail with "NVIDIA driver too old". There is no cu128 wheel.
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0   # else the FlashInfer sampler JIT looks for ninja and dies
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=<venv>/bin:$PATH           # cuda-graph capture shells out to ninja
```
`env/sglang_venv_freeze.txt` is the tested environment. Its torchao / torchaudio / torchvision / torchcodec pins are
`+cu129` builds from the PyTorch index, so install it with `--extra-index-url https://download.pytorch.org/whl/cu129
--index-strategy unsafe-best-match` (the exact command is in `AGENTS.md` §2).

---

## 10. Environment variables

Reproduce this combination first (the unoptimized runtime, `FAST=0`): `SELFSPEC_FULLDRAFT=1` (default),
`SELFSPEC_MASK=1` (default), `SELFSPEC_NOMASK=0`, cuda graphs ON. Then add the runtime optimizations (§7-5); the
output must not change.

| env | default | purpose |
|---|---|---|
| `SELFSPEC_FULLDRAFT` | `1` | full-draft round (§2): every mask logit + the bonus token, up to K+1 tokens per round. `0` = legacy round (≤ K) |
| `SELFSPEC_MASK_ID` | unset | mask token id; takes precedence over `block_diffusion.json` (§9-1) |
| `SELFSPEC_POS_OFFSET` | `0` | constant added to the window's positions. `0` = the AR layout (x0 at the position AR decoding and training give that token); verify then computes what AR decoding computes, up to floating point. GLM-OCR checkpoints are served with `1`, the positions the released self-spec results were produced with: every generated token, in the windows and in the KV cache, one RoPE position after AR decoding and training (§2). The launcher sets it in spec mode |
| `SELFSPEC_MASK` | `1` | custom-mask path: verify with custom_mask. **The reference path**; ignored when `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_FASTATTN` | `0` | mask-free draft/verify attention, one cuda graph per (bs, role) (§7-1). flashinfer backend only |
| `SELFSPEC_MROPE_FIX` | `0` | caches the M-RoPE deltas on the device instead of the stock per-forward blocking upload; reads the per-round switches once (§7-5) |
| `SELFSPEC_FAST_COMMIT` | `0` | commit on the GPU, host bookkeeping from one pinned readback (§7-5). Needs `SELFSPEC_FASTATTN=1`, `SELFSPEC_FULLDRAFT=1` |
| `SELFSPEC_HOSTLEN` | `0` | sequence lengths kept exact on the host (no device→host sum). Needs `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_DEV_POS` | `0` | window positions computed on the GPU. Needs `SELFSPEC_FASTATTN=1` |
| `SELFSPEC_DRAFT_AHEAD` | `0` | launch the next round's draft before this round's host bookkeeping (§7-5). Needs the four flags above |
| `SS_GLM_VIT_GRAPH_MAXM` | `0` (off) | GLM-OCR ViT as one CUDA graph per patch count (a multiple of 4) up to this value; other sizes run eager (§7-5). Decimal digits only, rounded down to a multiple of 4, at most 16384 (2x the ViT's rotary table); anything else stops the startup |
| `SS_GLM_VIT_EAGER_LIST` | `1` | calls without a graph run eager with the sequence offsets passed once as a list; `0` = the original eager forward. Only `0` / `1` |
| `SS_GLM_VIT_GRAPH_CHECK` | `1` | compare every captured ViT graph with the eager forward at capture; a mismatching size runs eager. Only `0` / `1` |
| `SS_GLM_PV_BF16` | `0` | the processor casts `pixel_values` to bf16 on the GPU (bf16 models only) |
| `SELFSPEC_CHECK` | `0` | diagnostic: per-forward host/device length, position and plan checks; raises on a mismatch (syncs; never time with it) |
| `SELFSPEC_SHADOW` | `0` | diagnostic, with `SELFSPEC_FASTATTN=1`, eager forwards only: also run the mask path and compare every attention call bitwise |
| `SELFSPEC_PADFIX` | `1` | custom-mask path only: pad the mask for the dummy rows of a padded graph replay |
| `SELFSPEC_NOMASK` | `0` | older mask-free dual-graph path, not `SELFSPEC_FASTATTN`. Measured worse (correctness and speed) — experimental; refused with `SELFSPEC_FASTATTN=1` or `SELFSPEC_FULLDRAFT=1` |
| `SELFSPEC_TRAIN_MASKS` | `0` | window K+1. Kills the kernel (§9-3); refused at startup with `SELFSPEC_FASTATTN=1` |
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `0` | force greedy verify even at temp > 0 (A/B) |
| `SELFSPEC_DEBUG` | `0` | per-round accept + cumulative tok/fwd |
| `SELFSPEC_TIME` / `SELFSPEC_NOSYNC` | `0` | per-phase timers / launch-bound diagnosis |
| `SELFSPEC_ACCEPT_STATS` | `""` | path → per-position confidence + accept as jsonl |
| `SELFSPEC_ROUND_HIST` | `0` | histogram of tokens committed per round, printed every 200 rounds |
| `SELFSPEC_{LOGIT,DRAFT,VERIFY}_DUMP` | `0` | dump the first rounds |
| `DLLM_PROF` / `DLLM_GRAPH_DBG` | `0` | forward / graph instrumentation, self-spec servers only |

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
   **token ids** (text comparison is confounded by re-tokenization). Compare at `SELFSPEC_POS_OFFSET=0` (the AR
   layout, §2) against AR with `--disable-overlap-schedule` (§9-7); then every remaining difference should be a bf16
   near-tie, with the spec token AR's runner-up. On anything else check the mask path (§7-1) and the mask id (§9-1).
   Pass it with `FAST=0` first, then again with the §7-5 flags on; `SELFSPEC_CHECK=1` and `SELFSPEC_SHADOW=1`
   localize a mismatch on the optimized path. Repeat with the request options you serve (sampling, penalties,
   grammars, §9-6, §9-7).
2. **Gate 2 — acceptance is real.** `SELFSPEC_DEBUG=1` prints per-round accept. Values stuck at 1–2
   mean the draft is garbage: mask_id (§9-1) or K alignment (§9-2).
3. **Gate 3 — speed.** Only then measure tok/fwd and tok/s.
