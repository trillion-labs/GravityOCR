"""Fast-dLLM v2 block-diffusion training grafted onto GLM-OCR (zai-org/GLM-OCR).

The block-diffusion objective is applied to the GLM **text decoder** only; the CogViT
vision encoder + connector are used unchanged, and the visual input is never noised.

Shared-vision construction
---------------------------
The vanilla Fast-dLLM v2 forward concatenates the full noised and clean sequences into a
length-2L stream ``[x_t ; x_0]``. For a VLM that duplicates every vision/prompt token even
though they are never masked and carry no loss. Since vision/prompt tokens are *identical* in
both streams, we keep them once and only duplicate the diffused (response) tokens:

    sequence = [  S  |  x_t (response, noised)  |  x_0 (response, clean)  ]
                shared        region A tail            region B
    (region A = the original length-L sequence with the response span noised; region B = a
     clean copy of just the response span.)

Attention (custom flex block mask, see ``_build_shared_vision_block_mask``):
  * S (vision + prompt): causal among itself, visible to everything after it (conditioning).
  * x_t response block k: attends to all of S + its own block in x_t (bidirectional) +
    previous response blocks in x_0 (offset block-causal).
  * x_0 response: block-causal over [S, x_0]; never sees x_t.
Loss: token-shifted, masked-token-only cross-entropy over region A (the x_t / response span).

Assumption: the response (loss) tokens form a single contiguous span (true for single-turn
OCR SFT). Asserted at runtime.
"""

import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.checkpoint import checkpoint

from transformers import GlmOcrForConditionalGeneration, AutoProcessor

# segment tags used by the flex mask_mod
_PAD, _SHARED, _XT, _X0 = 0, 1, 2, 3


def bucket_up(x, buckets, cap):
    """Smallest bucket >= x, else cap. Used to keep the flex-attention sequence shape stable
    across batches (a handful of distinct lengths -> a handful of compiles, then cached)."""
    for b in buckets:
        if x <= b:
            return b
    return cap


def _allmask_viewb_mode(model):
    """(mode, shift) — only meaningful for the all-mask schedule; ('keep', 0) for every other schedule."""
    import os as _os
    if getattr(model, "mask_schedule", "uniform") != "allmask":
        return "keep", 0
    m = _os.environ.get("ALLMASK_VIEWB", "keep").lower()
    if m not in ("keep", "shift", "drop"):
        m = "keep"
    sh = int(_os.environ.get("ALLMASK_SHIFT", model.bd_size // 2))
    return m, sh


class GlmOcrBlockDiffusion(nn.Module):
    def __init__(self, model_id: str = "zai-org/GLM-OCR", bd_size: int = 32,
                 mask_token: str = "<|mask|>", dtype: torch.dtype = torch.bfloat16,
                 max_length: int = 1024, max_response_length: int = 512,
                 length_buckets=(128, 256, 512, 1024), response_buckets=(64, 128, 256, 512),
                 ar_loss_weight: float = 0.0, diff_loss_weight: float = None):
        super().__init__()
        self.bd_size = bd_size
        # Auxiliary AR loss on the clean x_0 (region B) stream — token-shifted next-token CE, à la
        # Fast-dLLM v2 / weDLM's memory-stream aux loss. 0.0 = pure block-diffusion (original v2).
        # When >0, the x_0 stream is made TOKEN-causal (not block-causal) so the next-token CE is a
        # valid AR objective (otherwise a token would attend to its own successor within a block).
        # total loss = (diff_loss + ar_loss_weight * ar_loss) / (1 + ar_loss_weight).
        self.ar_loss_weight = ar_loss_weight
        # Whether the AR loss excludes the EOS fill tail (pairs with EOS_BLOCK_FILL; default on).
        self.ar_eos_once = os.environ.get("AR_EOS_ONCE", "1") == "1"
        # Printed at startup: this filter once existed only on the unpacked path and therefore never ran
        # in training, with no trace in the log. Now the setting is visible.
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[ar-eos-once] exclude duplicate tail EOS from the AR loss: "
                  f"{'ON' if self.ar_eos_once else 'OFF'} (packed and unpacked paths)", flush=True)
        # --- diffusion/AR mixing coefficients (c_diff, c_ar) ------------------------------------------
        # Given in either parametrization; both are normalized so that c_diff + c_ar == 1.
        # Fixing the sum to 1 is the point: only the ratio moves, the total loss/gradient scale stays, so the
        # same lr / warmup / grad-clip settings behave as in previous runs (comparability).
        #   - diff_loss_weight = alpha  (Nemotron: L = L_AR + alpha*L_diff, alpha=0.3) -> c_ar=1/(1+alpha), c_diff=alpha/(1+alpha)
        #   - ar_loss_weight   = w      (original knob; w=1.0 is the 1:1 mix)      -> c_diff=1/(1+w), c_ar=w/(1+w)
        # Exactly equivalent via alpha = 1/w; the alpha form avoids values like 3.3333 in the checkpoint config.
        self.diff_loss_weight = diff_loss_weight
        if diff_loss_weight is not None:
            a = float(diff_loss_weight)
            self.c_ar, self.c_diff = 1.0 / (1.0 + a), a / (1.0 + a)
            self._has_ar = True                       # the alpha parametrization always keeps the AR term
        else:
            w = float(ar_loss_weight)
            self.c_diff, self.c_ar = (1.0 / (1.0 + w), w / (1.0 + w)) if w > 0 else (1.0, 0.0)
            self._has_ar = w > 0
        # Fixed-length bucketing: region A (full seq) is padded by the collator to a length_bucket;
        # region B (response copy) is padded here to a response_bucket. Samples exceeding the caps
        # are dropped by the collator. Keeps the flex sequence shape to <= 16 distinct values.
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.length_buckets = length_buckets
        self.response_buckets = response_buckets
        # gradient checkpointing on the decoder layer loop — recompute activations in backward to
        # fit long sequences (big OCR crops -> thousands of vision tokens). Off by default (recompute
        # is a compute cost); enable for long-max_length runs that would otherwise OOM in backward.
        # gc_min_len: only checkpoint when the (bucketed) input length is >= this — short buckets fit
        # without gc and shouldn't pay the ~2x recompute. 0 = checkpoint whenever the flag is on.
        self.grad_checkpoint = False
        self.gc_min_len = 0
        self.block_phase_jitter = False   # random per-segment block-phase offset (for spec-decode robustness)
        # Noise schedule = how the per-block mask probability p is sampled (default "uniform" = the
        # original Fast-dLLM v2 behaviour: t~U[0,1], p=(1-eps)t+eps). Other options bias the mask ratio:
        #   "power"   -> t=u**(1/g), g=mask_schedule_power>=1 is concave (skews p toward 1 = high mask).
        #   "cosine"  -> p=1-cos(pi/2 * u), u~U[0,1] (Improved-DDPM cosine; convex -> skews p toward 0 =
        #                low mask, E[p]~0.36). Grounded clean-leaning schedule: masked-diffusion LM work
        #                (arXiv:2509.05056) reports cosine masking beats linear/uniform for text.
        #   "allmask" -> p=1 deterministically: every response token is masked (view A = full block from
        #                all-mask, view B = empty/zero-loss). Trains ONLY the one-forward block draft used
        #                by spec-decode (draft_steps=1); no partial-context denoise signal.
        self.mask_schedule = "uniform"
        self.mask_schedule_power = 1.0
        # LoopMDM (arXiv 2605.26106): head(once) -> mid-block looped S times (SHARED weights) -> tail(once).
        # Loop a small # of EARLY-MIDDLE layers; S ~ U{1..smax} per step at train, fixed at inference.
        # loop_enabled=False -> plain single pass, identical to the base model. No added parameters.
        self.loop_enabled = False
        self.loop_start = 2       # first looped layer index (early-middle; paper used 1-2 on a 12-layer net)
        self.loop_n_m = 2         # # of contiguous mid layers in the looped block (small is best)
        self.loop_smax = 8        # max loop count (training samples U{1..smax}; inference default)
        self.loop_train_stochastic = True
        self.loop_infer_S = None  # fixed loop count at inference (None -> loop_smax)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.glm = GlmOcrForConditionalGeneration.from_pretrained(model_id, dtype=dtype)

        # Add the learned block-diffusion [MASK] token and grow the embedding/lm_head by 1.
        if mask_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
            self.glm.resize_token_embeddings(len(self.tokenizer))
        self.mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        self.image_token_id = self.glm.config.image_token_id

        # The text decoder uses flex attention so we can inject the block-diffusion BlockMask.
        # The vision tower keeps its own attention implementation (it runs separately).
        self.text_model.config._attn_implementation = "flex_attention"

        self._patch_fast_vision_patch_embed()

    def _patch_fast_vision_patch_embed(self):
        """Replace the vision Conv3d patch-embed with the equivalent matmul.

        The patch-embed Conv3d uses kernel == stride == full patch, so it is mathematically a
        per-patch linear projection. But Conv3d with a huge batch (one item per patch) and a
        1x1x1 output is a pathological shape for cuDNN: profiling a full-page crop (24,336 patches)
        showed the Conv3d taking **212s** while the 24 transformer blocks took 0.6s. cudnn.benchmark
        does not help; the equivalent matmul runs in ~0.5ms (bf16-equivalent, max|Δ|~3e-2). This is
        what made long (high-resolution) crops "impossible" — it was never attention/flex/max_length.
        """
        import torch.nn.functional as F
        pe = self.mm_model.visual.patch_embed

        def fast_forward(hidden_states):
            w = pe.proj.weight.reshape(pe.embed_dim, -1)            # [embed, in*t*ph*pw]
            x = hidden_states.reshape(hidden_states.shape[0], -1).to(w.dtype)
            return F.linear(x, w, pe.proj.bias)

        pe.forward = fast_forward

    # ---- convenience handles into the HF model ----
    @property
    def mm_model(self):              # GlmOcrModel (vision + connector + text)
        return self.glm.model

    @property
    def text_model(self):            # GlmOcrTextModel
        return self.glm.model.language_model

    # ------------------------------------------------------------------
    # noising
    # ------------------------------------------------------------------
    def _sample_p(self, shape, device, eps=1e-3):
        """Per-block mask probability p, sampled per the configured noise schedule.

        uniform: p=(1-eps)*U[0,1]+eps (original). power: p=(1-eps)*U^(1/g)+eps (concave for g>1,
        biases toward high mask). cosine: p=(1-eps)*(1-cos(pi/2*U))+eps (convex, biases toward low mask,
        E[p]~0.36). allmask: p=1 (every response token masked; view B becomes empty)."""
        sched = getattr(self, "mask_schedule", "uniform")
        if sched == "allmask":
            return torch.ones(shape, device=device)
        u = torch.rand(shape, device=device)
        if sched == "power":
            g = float(getattr(self, "mask_schedule_power", 1.0))
            if g != 1.0:
                u = u ** (1.0 / g)
        elif sched == "cosine":
            u = 1.0 - torch.cos((math.pi / 2.0) * u)
        return (1 - eps) * u + eps

    # ------------------------------------------------------------------
    def _make_views(self, input_ids, labels, s0):
        """Per-(response)block noising of the assistant span into two complementary views.

        Blocks are response-relative (aligned with the attention block structure). Returns
        noised ids and per-view labels (clean id where actually masked, else -100) for both
        the view and its complement, stacked along the batch dim -> [2B, L].
        """
        B, L = input_ids.shape
        device = input_ids.device
        answer_pos = labels != -100                                  # [B, L]

        rel = (torch.arange(L, device=device)[None, :] - s0[:, None]).clamp(min=0)
        rblk = rel // self.bd_size                                   # response-relative block id
        max_blocks = (L + self.bd_size - 1) // self.bd_size
        p_block = self._sample_p((B, max_blocks), device)            # [B, max_blocks]
        p_tok = torch.gather(p_block, 1, rblk)                       # [B, L]
        mask_indices = (torch.rand(B, L, device=device) < p_tok) & answer_pos

        def view(selected):
            apply = selected & answer_pos
            noisy = torch.where(apply, self.mask_id, input_ids)
            vlabels = labels.clone()
            vlabels[~apply] = -100
            return noisy, vlabels

        noisy_a, lab_a = view(mask_indices)
        # There are two forward paths. Training uses the packed path, so shift/drop are not implemented on
        #   this unpacked path; fail loudly instead of silently behaving differently.
        _vm, _ = _allmask_viewb_mode(self)
        if _vm != "keep":
            raise RuntimeError(
                f"ALLMASK_VIEWB={_vm} is only implemented on the packed path (unpacked forward entered). "
                "Check MAX_PACKED_ROWS / packing settings.")
        noisy_b, lab_b = view(~mask_indices)
        noisy = torch.cat([noisy_a, noisy_b], dim=0)
        view_labels = torch.cat([lab_a, lab_b], dim=0)
        return noisy, view_labels

    # ------------------------------------------------------------------
    # attention mask
    # ------------------------------------------------------------------
    @staticmethod
    def _build_shared_vision_block_mask(seg, opos, rblk, x0_token_causal=False):
        """Custom flex BlockMask for the [S | x_t | x_0] layout. All args are [B, T] int.

        x0_token_causal: when True the clean x_0 stream attends to itself TOKEN-causally (okv<=oq)
        instead of block-causally (bkv<=bq) — required for a valid AR aux loss on x_0 (so a token
        cannot peek at its own successor within a block). xt's view of x_0 (xt_offset) is a
        cross-stream rule on bkv<bq and is unaffected either way.
        """
        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]

            kv_ok = skv != _PAD
            # attend to shared (conditioning) tokens causally, from any non-pad query
            to_shared = (skv == _SHARED) & (okv <= oq)
            # x_t response query: own block in x_t (bidirectional) + previous blocks in x_0
            xt_diag = (sq == _XT) & (skv == _XT) & (bq == bkv)
            xt_offset = (sq == _XT) & (skv == _X0) & (bkv < bq)
            # x_0 clean response query: token-causal (AR) or block-causal (pure diffusion)
            if x0_token_causal:
                x0_causal = (sq == _X0) & (skv == _X0) & (okv <= oq)
            else:
                x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)
            return kv_ok & (to_shared | xt_diag | xt_offset | x0_causal)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device)

    # ------------------------------------------------------------------
    # LoopMDM decoder stack (head once -> mid-block looped S times -> tail once)
    # ------------------------------------------------------------------
    def _loop_S(self):
        """Loop count S: stochastic U{1..smax} during training (LoopMDM eq.5, exposes the shared
        mid-block to a range of effective depths); fixed (loop_infer_S or smax) at inference."""
        if self.training and self.loop_train_stochastic:
            return int(torch.randint(1, self.loop_smax + 1, (1,)).item())
        return self.loop_infer_S if self.loop_infer_S else self.loop_smax

    def _run_decoder_stack(self, hidden, position_embeddings, block_mask, gc_active):
        """Run the GLM decoder layers. loop_enabled=False -> plain single pass (identical to base).
        loop_enabled=True -> LoopMDM: layers[:start] once (head), layers[start:start+n_m] repeated S
        times reusing the SAME weights (mid-block), layers[start+n_m:] once (tail). gc_active toggles
        gradient checkpointing per layer application."""
        layers = self.text_model.layers

        def apply(layer, h):
            if gc_active:
                return checkpoint(lambda x, l=layer: l(x, position_embeddings=position_embeddings,
                                  attention_mask=block_mask, past_key_values=None),
                                  h, use_reentrant=False)
            return layer(h, position_embeddings=position_embeddings,
                         attention_mask=block_mask, past_key_values=None)

        if not getattr(self, "loop_enabled", False) or self.loop_n_m <= 0:
            for layer in layers:
                hidden = apply(layer, hidden)
            return hidden
        lo, nm, S = self.loop_start, self.loop_n_m, self._loop_S()
        for i in range(lo):                     hidden = apply(layers[i], hidden)   # head (once)
        for _ in range(S):                                                          # mid-block (S times, shared)
            for i in range(lo, lo + nm):        hidden = apply(layers[i], hidden)
        for i in range(lo + nm, len(layers)):   hidden = apply(layers[i], hidden)   # tail (once)
        return hidden

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, input_ids, labels, attention_mask, pixel_values, image_grid_thw,
                mm_token_type_ids, share_views=False, return_logits=False,
                seg_id=None, resp_pos=None, loss_parts=None):
        # Sequence-packing path: a row holds several [prefix | response] segments. seg_id/resp_pos
        # (written by the training-time packing collator) drive segment-isolated (block-diagonal) attention
        # and per-segment block-diffusion. The default single-sample path below is untouched.
        if seg_id is not None:
            return self._forward_packed(input_ids, labels, attention_mask, pixel_values,
                                        image_grid_thw, mm_token_type_ids, seg_id, resp_pos,
                                        return_logits=return_logits, loss_parts=loss_parts)
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()

        # response span = first..last labelled token; assert it is contiguous (single-turn)
        ans = labels != -100
        assert ans.any(dim=1).all(), "every sample must have at least one response token"
        s0 = torch.argmax(ans.int(), dim=1)                          # first response idx per sample
        last = L - 1 - torch.argmax(torch.flip(ans, [1]).int(), dim=1)
        assert (ans == ((torch.arange(L, device=device)[None] >= s0[:, None]) &
                        (torch.arange(L, device=device)[None] <= last[:, None]))).all(), \
            "response (loss) tokens must form one contiguous span"
        resp_len = last - s0 + 1                                      # contiguous response length

        # ---- 1) noise the response into two complementary views -> [2B, L] ----
        noisy, view_labels = self._make_views(input_ids, labels, s0)
        if share_views:
            # optional: process S + clean x_0 ONCE for both views (batch B, not 2B)
            return self._forward_shared(input_ids, attention_mask, pixel_values, image_grid_thw,
                                        mm_token_type_ids, noisy, view_labels, s0, last, resp_len)
        clean = input_ids.repeat(2, 1)
        s0 = s0.repeat(2)
        resp_len = resp_len.repeat(2)
        valid = valid.repeat(2, 1)
        mm_tt = mm_token_type_ids.repeat(2, 1)
        attn2 = attention_mask.repeat(2, 1)
        BB = 2 * B

        # ---- 2) MRoPE positions (computed on clean ids, shared by x_t and x_0) ----
        # get_rope_index iterates one grid per image across the (doubled) batch, so the grids
        # must be duplicated to match the complementary-view batch.
        grid_rope = image_grid_thw.repeat(2, 1) if image_grid_thw is not None else None
        pos3, _ = self.mm_model.get_rope_index(clean, mm_tt, image_grid_thw=grid_rope,
                                               video_grid_thw=None, attention_mask=attn2)  # [3, BB, L]

        # ---- 3) embed + splice image features into region A (noised stream) ----
        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.mm_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device)

        def embed_splice(ids, n_repeat):
            emb = self.text_model.embed_tokens(ids)
            if image_embeds is not None:
                feats = image_embeds.repeat(n_repeat, 1) if n_repeat > 1 else image_embeds
                mask = (ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
                emb = emb.masked_scatter(mask, feats.to(emb.dtype))
            return emb

        regionA = embed_splice(noisy, 2)                             # [BB, L, D] (S clean + x_t noised)
        cleanA = embed_splice(clean, 2)                              # clean embeds, to copy x_0 from

        # ---- 4) build region B (clean response copy) by gathering the response span ----
        # region B = the response span only, padded to a fixed response bucket (constant shape).
        rpad = bucket_up(int(resp_len.max().item()), self.response_buckets, self.max_response_length)
        r_idx = torch.arange(rpad, device=device)[None, :]
        src = (s0[:, None] + r_idx)                                  # [BB, rpad] physical index
        src_c = src.clamp(max=L - 1)
        b_valid = r_idx < resp_len[:, None]                          # slot valid iff within the response
        D = cleanA.size(-1)
        regionB = torch.gather(cleanA, 1, src_c.unsqueeze(-1).expand(-1, -1, D))  # [BB, rpad, D]
        posB = torch.gather(pos3, 2, src_c.unsqueeze(0).expand(3, -1, -1))        # [3, BB, rpad]

        combined = torch.cat([regionA, regionB], dim=1)              # [BB, L+rpad, D]
        position_ids = torch.cat([pos3, posB], dim=2)               # [3, BB, L+rpad]
        T = L + rpad

        # ---- 5) per-token metadata for the mask ----
        ar = torch.arange(L, device=device)[None, :].expand(BB, -1)
        valid_A = valid                                              # [BB, L]
        is_resp_A = (ar >= s0[:, None]) & valid_A
        segA = torch.where(valid_A, torch.where(is_resp_A, torch.full_like(ar, _XT),
                                                torch.full_like(ar, _SHARED)),
                           torch.full_like(ar, _PAD))
        oposA = ar
        rblkA = ((ar - s0[:, None]).clamp(min=0) // self.bd_size)

        segB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        oposB = src_c
        rblkB = (r_idx.expand(BB, -1) // self.bd_size)

        seg = torch.cat([segA, segB], dim=1).int()
        opos = torch.cat([oposA, oposB], dim=1).int()
        rblk = torch.cat([rblkA, rblkB], dim=1).int()
        block_mask = self._build_shared_vision_block_mask(seg, opos, rblk,
                                                          x0_token_causal=self._has_ar)

        # ---- 6) run the GLM text decoder stack with the injected block mask ----
        tm = self.text_model
        position_embeddings = tm.rotary_emb(combined, position_ids=position_ids)
        hidden = combined
        hidden = self._run_decoder_stack(hidden, position_embeddings, block_mask,
                                         self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len)
        hidden = tm.norm(hidden)

        # ---- 7) token-shifted, masked-only loss over region A (the x_t / response span) ----
        # Compute lm_head ONLY on the labelled (response) positions: most of the [BB, L] grid is
        # -100 (prefix/pad), so projecting the full [BB, L, V] and upcasting to fp32 wastes a huge
        # amount of memory (it OOMs at larger batches). Gathering the labelled predictor hiddens
        # first is mathematically identical (CE mean-reduces over the same non-ignored tokens) but
        # shrinks the logit tensor from [BB*(L-1), V] to [N_response, V].
        shift_h = hidden[:, :L - 1, :]                               # [BB, L-1, D] predictor hiddens
        shift_labels = view_labels[:, 1:]                            # [BB, L-1] next-token labels
        sel = shift_labels != -100
        sel_logits = self.glm.lm_head(shift_h[sel])                  # [N_response, V]
        diff_loss = F.cross_entropy(sel_logits.float(), shift_labels[sel])
        self._last_ntok = int(sel.sum())   # valid diffusion target tokens (for the trainer's loss logging)

        # ---- 7b) auxiliary AR loss on the clean x_0 stream (region B, positions L..L+rpad-1) ----
        # Token-shifted next-token CE: x_0 hidden at slot j predicts the clean response token j+1.
        # Valid because x_0 is TOKEN-causal here (x0_token_causal above). x_0 hiddens are identical
        # across the two complementary views (x_0 never attends to x_t), so computing over the full
        # [BB,...] just duplicates the per-view mean — same value, same gradient direction.
        if self._has_ar:
            hidB = hidden[:, L:, :]                                  # [BB, rpad, D] x_0 hiddens
            resp_ids = torch.gather(clean, 1, src_c)                 # [BB, rpad] clean response tokens
            ar_sel = (r_idx + 1 < resp_len[:, None])[:, :-1]         # slot j valid AND j+1 in-response
            # AR_EOS_ONCE: EOS block-fill (bd EOS tokens appended after the response in the data) exists for the
            #   diffusion stream only, to teach "the block still has bd MASKs but the document is over". The AR
            #   stream emits one token at a time, so a single EOS suffices; supervising all 32 of them measurably
            #   damages the AR path.
            #   So the AR loss supervises the tail only up to the first EOS and excludes the rest of the fill.
            #   Implementation: count the trailing EOS run of each row and mask out everything after the first EOS.
            if getattr(self, "ar_eos_once", False) and self.tokenizer.eos_token_id is not None:
                _eos = int(self.tokenizer.eos_token_id)
                _tgt = resp_ids[:, 1:]                                # tokens the AR path must predict
                _in = (r_idx + 1 < resp_len[:, None])[:, :-1]
                _is_eos = (_tgt == _eos) & _in
                # trailing EOS run: the stretch of consecutive EOS ending at the last valid position
                _rev = torch.flip(_is_eos.int(), dims=[1])
                _run = torch.cummin(_rev, dim=1).values               # 1 only on the trailing EOS run
                _tail = torch.flip(_run, dims=[1]).bool()
                # keep the FIRST EOS of the tail run, drop the rest
                _first = _tail & ~torch.cat(
                    [torch.zeros_like(_tail[:, :1]), _tail[:, :-1]], dim=1)
                ar_sel = ar_sel & (~_tail | _first)
            ar_logits = self.glm.lm_head(hidB[:, :-1, :][ar_sel]).float()
            ar_loss = F.cross_entropy(ar_logits, resp_ids[:, 1:][ar_sel])
            loss = self.c_diff * diff_loss + self.c_ar * ar_loss
            # Record the two terms separately, directly into the caller-provided dict (attribute side channels proved unreliable).
            if loss_parts is not None:
                loss_parts["diff"] = diff_loss.detach(); loss_parts["ar"] = ar_loss.detach()
            self.last_diff_loss = diff_loss.detach()
            self.last_ar_loss = ar_loss.detach()
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss
            if loss_parts is not None:
                loss_parts["diff"] = diff_loss.detach(); loss_parts["ar"] = ar_loss
            self.last_diff_loss = diff_loss.detach()
            self.last_ar_loss = ar_loss
        if return_logits:   # full [BB, L, V] logits — memory-heavy; for equivalence testing only
            return self.glm.lm_head(hidden[:, :L, :]), loss
        return None, loss

    # ------------------------------------------------------------------
    # forward — shared-views variant (process S + clean x_0 once for both views)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_shared_views_block_mask(seg, opos, rblk, view):
        """flex BlockMask for the layout ``[ base(S + x_0) | x_tA | x_tB ]``.

        Identical rules to ``_build_shared_vision_block_mask`` PLUS a view tag so the two
        complementary noised streams never attend to each other (``xt_diag`` now requires
        equal view). x_0 carries view 0 and is shared by both x_t views."""
        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]
            vq, vkv = view[b, q_idx], view[b, kv_idx]
            kv_ok = skv != _PAD
            to_shared = (skv == _SHARED) & (okv <= oq)                 # causal conditioning (S)
            x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)       # clean stream block-causal
            xt_to_x0 = (sq == _XT) & (skv == _X0) & (bkv < bq)         # x_t -> previous clean blocks
            xt_diag = (sq == _XT) & (skv == _XT) & (bkv == bq) & (vkv == vq)  # own block, same view
            return kv_ok & (to_shared | x0_causal | xt_to_x0 | xt_diag)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device)

    def _forward_shared(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                        mm_token_type_ids, noisy, view_labels, s0, last, resp_len):
        """Shared-views forward. Mathematically equivalent to the default forward, but processes
        the prefix S and the clean response x_0 ONCE for both complementary views instead of
        replicating them across a 2B batch.

        Layout per sample (batch B):  ``[ base = clean [S | x_0] | x_tA (noised resp, view A) |
        x_tB (noised resp, view B) ]``. x_0 never attends x_t (so it is view-independent and can
        be shared); the only new constraint vs default is x_tA ⊥ x_tB (view tag in the mask).

        Boundary token-shift: the predictor of the first response token (position s0) is the last
        prefix hidden ``base[s0-1]``; the rest come from the in-region shift of x_t. Assumes
        s0 >= 1 (image/prompt always precede the response — same assumption the default path makes
        implicitly, where the global shift drops position 0)."""
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        tm = self.text_model

        noisy_a, noisy_b = noisy[:B], noisy[B:]
        lab_a, lab_b = view_labels[:B], view_labels[B:]

        pos3, _ = self.mm_model.get_rope_index(input_ids, mm_token_type_ids,
                                               image_grid_thw=image_grid_thw, video_grid_thw=None,
                                               attention_mask=attention_mask)        # [3, B, L]

        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.mm_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device)

        def embed_splice(ids):
            emb = tm.embed_tokens(ids)
            if image_embeds is not None:
                mask = (ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
                emb = emb.masked_scatter(mask, image_embeds.to(emb.dtype))
            return emb

        base = embed_splice(input_ids)                               # [B, L, D] = clean [S | x_0]
        D = base.size(-1)

        # gather the response span (response-relative, padded to a fixed bucket -> stable shape)
        rpad = bucket_up(int(resp_len.max().item()), self.response_buckets, self.max_response_length)
        r_idx = torch.arange(rpad, device=device)[None, :]           # [1, rpad]
        src = s0[:, None] + r_idx                                     # [B, rpad]
        src_c = src.clamp(max=L - 1)
        b_valid = r_idx < resp_len[:, None]                          # [B, rpad]
        gidx = src_c.unsqueeze(-1).expand(-1, -1, D)
        xtA = torch.gather(embed_splice(noisy_a), 1, gidx)          # [B, rpad, D] noised resp (view A)
        xtB = torch.gather(embed_splice(noisy_b), 1, gidx)          # [B, rpad, D] noised resp (view B)
        posR = torch.gather(pos3, 2, src_c.unsqueeze(0).expand(3, -1, -1))   # [3, B, rpad]

        combined = torch.cat([base, xtA, xtB], dim=1)               # [B, L + 2*rpad, D]
        position_ids = torch.cat([pos3, posR, posR], dim=2)        # [3, B, L + 2*rpad]

        # ---- per-token metadata (seg / opos / rblk / view) ----
        ar = torch.arange(L, device=device)[None, :].expand(B, -1)
        is_resp = (ar >= s0[:, None]) & valid
        segBase = torch.where(valid, torch.where(is_resp, torch.full_like(ar, _X0),
                                                 torch.full_like(ar, _SHARED)),
                              torch.full_like(ar, _PAD))
        oposBase = ar
        rblkBase = ((ar - s0[:, None]).clamp(min=0) // self.bd_size)
        viewBase = torch.zeros_like(ar)

        segXt = torch.where(b_valid, torch.full_like(src, _XT), torch.full_like(src, _PAD))
        oposXt = src_c                                               # response positions (so x_t sees all S)
        rblkXt = r_idx.expand(B, -1) // self.bd_size
        viewA = torch.where(b_valid, torch.full_like(src, 1), torch.zeros_like(src))
        viewB = torch.where(b_valid, torch.full_like(src, 2), torch.zeros_like(src))

        seg = torch.cat([segBase, segXt, segXt], dim=1).int()
        opos = torch.cat([oposBase, oposXt, oposXt], dim=1).int()
        # ALLMASK_VIEWB=shift: view B on an offset block grid
        _vm, _vs = _allmask_viewb_mode(self)
        rblkXtB = ((r_idx.expand(B, -1) + _vs) // self.bd_size) if _vm == "shift" else rblkXt
        rblk = torch.cat([rblkBase, rblkXt, rblkXtB], dim=1).int()
        view = torch.cat([viewBase, viewA, viewB], dim=1).int()
        block_mask = self._build_shared_views_block_mask(seg, opos, rblk, view)

        # ---- run the GLM text decoder stack ----
        position_embeddings = tm.rotary_emb(combined, position_ids=position_ids)
        hidden = combined
        hidden = self._run_decoder_stack(hidden, position_embeddings, block_mask,
                                         self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len)
        hidden = tm.norm(hidden)

        base_h = hidden[:, :L, :]
        xtA_h = hidden[:, L:L + rpad, :]
        xtB_h = hidden[:, L + rpad:L + 2 * rpad, :]

        # boundary predictor for response[s0] = last prefix hidden base[s0-1] (token-shift seam)
        b0 = (s0 - 1).clamp(min=0)
        boundary = torch.gather(base_h, 1, b0[:, None, None].expand(-1, 1, D))   # [B, 1, D]

        def view_logits_labels(xt_h, lab):
            pred_h = torch.cat([boundary, xt_h[:, :rpad - 1, :]], dim=1)         # [B, rpad, D]
            logits = self.glm.lm_head(pred_h)                                    # [B, rpad, V]
            lab_resp = torch.gather(lab, 1, src_c)                              # [B, rpad]
            lab_resp = torch.where(b_valid, lab_resp, torch.full_like(lab_resp, -100))
            return logits, lab_resp

        logitsA, labA = view_logits_labels(xtA_h, lab_a)
        logitsB, labB = view_logits_labels(xtB_h, lab_b)
        V = logitsA.size(-1)
        all_logits = torch.cat([logitsA.reshape(-1, V), logitsB.reshape(-1, V)], dim=0).float()
        all_labels = torch.cat([labA.reshape(-1), labB.reshape(-1)], dim=0)
        loss = F.cross_entropy(all_logits, all_labels, ignore_index=-100)
        return (logitsA, logitsB), loss

    # ------------------------------------------------------------------
    # sequence packing (additive; default path above is unchanged)
    # ------------------------------------------------------------------
    def _make_views_packed(self, input_ids, labels, answer_pos, block_key):
        """Per-(segment, response-block) complementary noising for a PACKED row.

        Same two-view MLM masking as `_make_views`, but the block identity is `block_key`
        (unique per (segment, response-block) pair) instead of a single per-row s0 — so every
        packed segment is noised independently, exactly as if trained alone."""
        B, L = input_ids.shape
        device = input_ids.device
        nkey = int(block_key.max().item()) + 1 if block_key.numel() else 1
        p_key = self._sample_p((B, nkey), device)                            # [B, nkey]
        p_tok = torch.gather(p_key, 1, block_key.clamp(min=0))                # [B, L]
        mask_indices = (torch.rand(B, L, device=device) < p_tok) & answer_pos

        def view(selected):
            apply = selected & answer_pos
            noisy = torch.where(apply, self.mask_id, input_ids)
            vlabels = labels.clone()
            vlabels[~apply] = -100
            return noisy, vlabels

        noisy_a, lab_a = view(mask_indices)
        # With ALLMASK_VIEWB=shift view B is also fully masked (only the grid differs); a complement view would have no labels.
        _vm, _ = _allmask_viewb_mode(self)
        noisy_b, lab_b = view(answer_pos if _vm == "shift" else ~mask_indices)
        return torch.cat([noisy_a, noisy_b], dim=0), torch.cat([lab_a, lab_b], dim=0)

    @staticmethod
    def _build_packed_block_mask(seg, opos, rblk, segid):
        """`_build_shared_vision_block_mask` rules + segment isolation (block-diagonal across
        packed segments) + a self-diagonal so padded/empty query rows never get an all-masked
        (NaN) softmax. segid<0 marks padding (never matches a real query's segid)."""
        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]
            same_seg = segid[b, q_idx] == segid[b, kv_idx]
            kv_ok = skv != _PAD
            to_shared = (skv == _SHARED) & (okv <= oq)
            xt_diag = (sq == _XT) & (skv == _XT) & (bq == bkv)
            xt_offset = (sq == _XT) & (skv == _X0) & (bkv < bq)
            x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)
            rule = to_shared | xt_diag | xt_offset | x0_causal
            return (kv_ok & same_seg & rule) | (q_idx == kv_idx)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device)

    def _forward_packed_doublebatch(self, input_ids, labels, attention_mask, pixel_values, image_grid_thw,
                                    mm_token_type_ids, seg_id, resp_pos, return_logits=False):
        """[RETAINED for the equivalence gate, NOT used in production — see _forward_packed.]
        Packed-row forward in the DOUBLE-BATCH (2B) layout: regionA=[prefix|x_t] | regionB=[x_0],
        replicated across two views along the batch dim. Correct but forwards the vision-heavy
        prefix ONCE PER VIEW (2× waste); the shared-view _forward_packed below replaces it."""
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        answer_pos = (labels != -100) & valid
        bd = self.bd_size
        NB = L // bd + 2                                                  # max response-blocks/segment
        zero = torch.zeros_like(resp_pos)
        rblk_tok = torch.where(answer_pos, resp_pos.clamp(min=0) // bd, zero)     # resp-block within seg
        block_key = torch.where(answer_pos, seg_id.clamp(min=0) * NB + rblk_tok, zero)

        # ---- 1) two complementary noised views (per-segment blocks) -> [2B, L] ----
        noisy, view_labels = self._make_views_packed(input_ids, labels, answer_pos, block_key)
        clean = input_ids.repeat(2, 1)
        valid2 = valid.repeat(2, 1)
        mm2 = mm_token_type_ids.repeat(2, 1)
        attn2 = attention_mask.repeat(2, 1)
        seg2 = seg_id.repeat(2, 1)
        resp2 = resp_pos.repeat(2, 1)
        ans2 = answer_pos.repeat(2, 1)
        BB = 2 * B

        grid_rope = image_grid_thw.repeat(2, 1) if image_grid_thw is not None else None
        pos3, _ = self.mm_model.get_rope_index(clean, mm2, image_grid_thw=grid_rope,
                                               video_grid_thw=None, attention_mask=attn2)  # [3, BB, L]

        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.mm_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device)

        def embed_splice(ids, n_repeat):
            emb = self.text_model.embed_tokens(ids)
            if image_embeds is not None:
                feats = image_embeds.repeat(n_repeat, 1) if n_repeat > 1 else image_embeds
                mask = (ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
                emb = emb.masked_scatter(mask, feats.to(emb.dtype))
            return emb

        regionA = embed_splice(noisy, 2)                                 # [BB, L, D]
        cleanA = embed_splice(clean, 2)

        # ---- 2) regionB = clean x_0 copy of ALL response tokens (multi-span gather) ----
        rcount = ans2.sum(dim=1)                                         # [BB] response tokens/row
        rpad = bucket_up(int(rcount.max().item()), self.response_buckets, self.max_response_length)
        ar = torch.arange(L, device=device)
        # answer positions first (in index order), then the rest; keys unique -> order deterministic
        order = torch.argsort((~ans2).int() * (L + 1) + ar[None, :], dim=1)
        src = order[:, :rpad]                                           # [BB, rpad]
        b_valid = torch.arange(rpad, device=device)[None, :] < rcount[:, None]
        src_c = src.clamp(max=L - 1)
        D = cleanA.size(-1)
        regionB = torch.gather(cleanA, 1, src_c.unsqueeze(-1).expand(-1, -1, D))
        posB = torch.gather(pos3, 2, src_c.unsqueeze(0).expand(3, -1, -1))

        combined = torch.cat([regionA, regionB], dim=1)                  # [BB, L+rpad, D]
        position_ids = torch.cat([pos3, posB], dim=2)

        # ---- 3) per-token metadata (seg tag / opos / resp-block / segment id) ----
        arL = torch.arange(L, device=device)[None, :].expand(BB, -1)
        is_resp_A = ans2
        segtagA = torch.where(valid2, torch.where(is_resp_A, torch.full_like(arL, _XT),
                                                  torch.full_like(arL, _SHARED)),
                              torch.full_like(arL, _PAD))
        rblkA = torch.where(ans2, resp2.clamp(min=0) // bd, torch.zeros_like(resp2))
        segidA = torch.where(valid2, seg2, torch.full_like(seg2, -1))

        segtagB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        rblkB = torch.gather(rblkA, 1, src_c)
        segidB = torch.where(b_valid, torch.gather(segidA, 1, src_c), torch.full_like(src, -1))

        seg = torch.cat([segtagA, segtagB], dim=1).int()
        opos = torch.cat([arL, src_c], dim=1).int()                      # global index (monotonic in-seg)
        rblk = torch.cat([rblkA, rblkB], dim=1).int()
        segid = torch.cat([segidA, segidB], dim=1).int()
        block_mask = self._build_packed_block_mask(seg, opos, rblk, segid)

        # ---- 4) decoder stack ----
        tm = self.text_model
        position_embeddings = tm.rotary_emb(combined, position_ids=position_ids)
        hidden = combined
        hidden = self._run_decoder_stack(hidden, position_embeddings, block_mask,
                                         self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len)
        hidden = tm.norm(hidden)

        # ---- 5) token-shifted, masked-only loss over regionA (== default path) ----
        # Each response token's predictor hidden[p-1] sits in the SAME segment (segment layout is
        # [prefix | response]), so the shift never crosses a packed boundary.
        shift_h = hidden[:, :L - 1, :]
        shift_labels = view_labels[:, 1:]
        sel = shift_labels != -100
        sel_logits = self.glm.lm_head(shift_h[sel])
        loss = F.cross_entropy(sel_logits.float(), shift_labels[sel])
        if return_logits:
            return self.glm.lm_head(hidden[:, :L, :]), loss
        return None, loss

    @staticmethod
    def _build_packed_shared_views_block_mask(seg, opos, rblk, view, segid, x0_token_causal=False):
        """`_build_shared_views_block_mask` rules + per-segment isolation (seg_id block-diagonal)
        + self-diagonal (pad/empty rows never get an all-masked NaN softmax). segid<0 = padding.

        x0_token_causal: clean x_0 attends to itself TOKEN-causally (okv<=oq) instead of
        block-causally — required for a valid AR aux loss on the in-place x_0 (base) stream."""
        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]
            vq, vkv = view[b, q_idx], view[b, kv_idx]
            same_seg = segid[b, q_idx] == segid[b, kv_idx]
            kv_ok = skv != _PAD
            to_shared = (skv == _SHARED) & (okv <= oq)
            if x0_token_causal:
                x0_causal = (sq == _X0) & (skv == _X0) & (okv <= oq)
            else:
                x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)
            xt_to_x0 = (sq == _XT) & (skv == _X0) & (bkv < bq)
            xt_diag = (sq == _XT) & (skv == _XT) & (bkv == bq) & (vkv == vq)
            return (kv_ok & same_seg & (to_shared | x0_causal | xt_to_x0 | xt_diag)) | (q_idx == kv_idx)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device)

    def _forward_packed(self, input_ids, labels, attention_mask, pixel_values, image_grid_thw,
                        mm_token_type_ids, seg_id, resp_pos, return_logits=False, loss_parts=None):
        """Packed-row forward, SHARED-VIEW layout — generalizes _forward_shared to MULTIPLE
        [prefix | response] segments per row. Layout per row (batch B):

            [ base = clean (packed [S_k | x0_k] segments) | x_tA (all responses, view A)
                                                          | x_tB (all responses, view B) ]

        The vision-heavy prefix + clean x_0 are processed ONCE (not 2× as in the double-batch
        path), so forward length = packed_L + 2*rpad. Segments never cross-attend (seg_id
        block-diagonal); x_tA ⊥ x_tB (view tag); each segment keeps its own block-diffusion
        structure. Token-shift is PER-SEGMENT: a response slot's predictor is the segment's last
        prefix hidden if it's that segment's first response token (resp_pos==0), else the previous
        response token's noised hidden — so the shift never crosses a packed boundary.

        Mathematically identical to _forward_packed_doublebatch (and to unpacked); gated by
        test_packing_grad_equivalence.py. default(seg_id=None)/non-packed path is untouched."""
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        answer_pos = (labels != -100) & valid
        bd = self.bd_size
        NB = L // bd + 2                                                  # max response-blocks/segment
        tm = self.text_model
        zero = torch.zeros_like(resp_pos)
        # block-phase jitter: random per-segment offset δ∈[0,bd) so block boundaries vary across
        # samples (rblk=(resp_pos+δ)//bd). Makes the model robust to ANY block alignment — needed for
        # speculative decoding, where a partial accept leaves the next block starting off the bd grid.
        # NB=L//bd+2 already reserves the slot for the extra (partial) leading block. Same δ MUST be
        # used by both the noising (block_key) and the attention mask (rblkBase) below. Default off.
        if getattr(self, "block_phase_jitter", False) and self.training:
            max_seg = (int(seg_id.max().item()) + 1) if seg_id.numel() else 1
            dtab = torch.randint(0, bd, (B, max(1, max_seg)), device=device)
            delta_tok = torch.gather(dtab, 1, seg_id.clamp(min=0))        # [B,L] each token's segment phase
        else:
            delta_tok = torch.zeros_like(resp_pos)
        rblk_tok = torch.where(answer_pos, (resp_pos.clamp(min=0) + delta_tok) // bd, zero)
        block_key = torch.where(answer_pos, seg_id.clamp(min=0) * NB + rblk_tok, zero)

        # ---- 1) two complementary noised views (per-segment blocks) -> [2B, L] ----
        noisy, view_labels = self._make_views_packed(input_ids, labels, answer_pos, block_key)
        noisy_a, noisy_b = noisy[:B], noisy[B:]
        lab_a, lab_b = view_labels[:B], view_labels[B:]

        pos3, _ = self.mm_model.get_rope_index(input_ids, mm_token_type_ids,
                                               image_grid_thw=image_grid_thw, video_grid_thw=None,
                                               attention_mask=attention_mask)        # [3, B, L]
        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.mm_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device)

        def embed_splice(ids):
            emb = tm.embed_tokens(ids)
            if image_embeds is not None:
                mask = (ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
                emb = emb.masked_scatter(mask, image_embeds.to(emb.dtype))
            return emb

        base = embed_splice(input_ids)                                   # [B, L, D] prefix+x0 ONCE
        D = base.size(-1)

        # ---- 2) gather ALL response tokens (multi-span) for the two noised views ----
        rcount = answer_pos.sum(dim=1)                                   # [B]
        rpad = bucket_up(int(rcount.max().item()), self.response_buckets, self.max_response_length)
        # regionB gathers response tokens out of the L-wide row, so it can never need more than L
        # slots (rcount <= L always). bucket_up can round up to a response bucket > L (e.g. a row
        # bucketed to L=256 but response bucket 512/1024) -> src=order[:, :rpad] would be clipped to
        # width L while b_valid stays width rpad -> the line-~700 size mismatch. Clamp to L (safe:
        # min(rpad, L) >= rcount.max since L >= rcount.max), keeping x_t/x_0 region B consistent.
        rpad = min(rpad, L)
        ar = torch.arange(L, device=device)
        order = torch.argsort((~answer_pos).int() * (L + 1) + ar[None, :], dim=1)   # answer-first, ascending
        src = order[:, :rpad]
        b_valid = torch.arange(rpad, device=device)[None, :] < rcount[:, None]
        src_c = src.clamp(max=L - 1)
        gidx = src_c.unsqueeze(-1).expand(-1, -1, D)
        xtA = torch.gather(embed_splice(noisy_a), 1, gidx)               # [B, rpad, D] view A
        xtB = torch.gather(embed_splice(noisy_b), 1, gidx)               # [B, rpad, D] view B
        posR = torch.gather(pos3, 2, src_c.unsqueeze(0).expand(3, -1, -1))

        combined = torch.cat([base, xtA, xtB], dim=1)                    # [B, L + 2*rpad, D]
        position_ids = torch.cat([pos3, posR, posR], dim=2)

        # ---- 3) per-token metadata (seg tag / global pos / resp-block / view / segment id) ----
        arL = torch.arange(L, device=device)[None, :].expand(B, -1)
        segBase = torch.where(valid, torch.where(answer_pos, torch.full_like(arL, _X0),
                                                 torch.full_like(arL, _SHARED)),
                              torch.full_like(arL, _PAD))
        rblkBase = torch.where(answer_pos, (resp_pos.clamp(min=0) + delta_tok) // bd, torch.zeros_like(resp_pos))
        segidBase = torch.where(valid, seg_id, torch.full_like(seg_id, -1))

        segXt = torch.where(b_valid, torch.full_like(src, _XT), torch.full_like(src, _PAD))
        rblkXt = torch.gather(rblkBase, 1, src_c)
        segidXt = torch.where(b_valid, torch.gather(segidBase, 1, src_c), torch.full_like(src, -1))
        viewA = torch.where(b_valid, torch.full_like(src, 1), torch.zeros_like(src))
        viewB = torch.where(b_valid, torch.full_like(src, 2), torch.zeros_like(src))

        seg = torch.cat([segBase, segXt, segXt], dim=1).int()
        opos = torch.cat([arL, src_c, src_c], dim=1).int()
        rblk = torch.cat([rblkBase, rblkXt, rblkXt], dim=1).int()
        view = torch.cat([torch.zeros_like(arL), viewA, viewB], dim=1).int()
        segid = torch.cat([segidBase, segidXt, segidXt], dim=1).int()
        block_mask = self._build_packed_shared_views_block_mask(seg, opos, rblk, view, segid,
                                                                x0_token_causal=self._has_ar)

        # ---- 4) decoder stack ----
        position_embeddings = tm.rotary_emb(combined, position_ids=position_ids)
        hidden = combined
        hidden = self._run_decoder_stack(hidden, position_embeddings, block_mask,
                                         self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len)
        hidden = tm.norm(hidden)

        base_h = hidden[:, :L, :]
        xtA_h = hidden[:, L:L + rpad, :]
        xtB_h = hidden[:, L + rpad:L + 2 * rpad, :]

        # ---- 5) PER-SEGMENT token-shift loss (masked-only, both views) ----
        # slot j predicts response token src[j]; predictor = base_h[src[j]-1] if that token is its
        # segment's first response (resp_pos==0) else xt_h[j-1] (same-segment previous response).
        prev_global = (src_c - 1).clamp(min=0)
        boundary_h = torch.gather(base_h, 1, prev_global.unsqueeze(-1).expand(-1, -1, D))  # [B, rpad, D]
        is_first = (torch.gather(resp_pos, 1, src_c) == 0) & b_valid                       # [B, rpad]

        def view_logits_labels(xt_h, lab):
            shifted = torch.cat([torch.zeros_like(xt_h[:, :1, :]), xt_h[:, :rpad - 1, :]], dim=1)
            pred_h = torch.where(is_first.unsqueeze(-1), boundary_h, shifted)             # [B, rpad, D]
            logits = self.glm.lm_head(pred_h)                                            # [B, rpad, V]
            lab_resp = torch.gather(lab, 1, src_c)
            lab_resp = torch.where(b_valid, lab_resp, torch.full_like(lab_resp, -100))
            return logits, lab_resp

        logitsA, labA = view_logits_labels(xtA_h, lab_a)
        logitsB, labB = view_logits_labels(xtB_h, lab_b)
        V = logitsA.size(-1)
        all_logits = torch.cat([logitsA.reshape(-1, V), logitsB.reshape(-1, V)], dim=0).float()
        all_labels = torch.cat([labA.reshape(-1), labB.reshape(-1)], dim=0)
        diff_loss = F.cross_entropy(all_logits, all_labels, ignore_index=-100)
        self._last_ntok = int((all_labels != -100).sum())   # valid target tokens this loss averaged over

        # ---- 5b) auxiliary AR loss on the in-place clean x_0 stream (base, positions 0..L-1) ----
        # base_h[p] (token-causal x_0) predicts the next token labels[p+1]. labels are -100 on prefix
        # tokens, so a segment's last response token predicting the next segment's prefix is ignored —
        # the shift never leaks across packed boundaries; the first response token of each segment is
        # predicted from its segment's last prefix hidden (same-seg, valid). Mirrors weDLM's aux AR.
        if self._has_ar:
            ar_shift_h = base_h[:, :L - 1, :]                # [B, L-1, D] predictor hiddens
            ar_labels = labels[:, 1:]                        # [B, L-1] next-token labels (-100 off-response)
            ar_sel = ar_labels != -100
            # AR_EOS_ONCE: EOS block-fill (bd EOS appended after the response) is for the diffusion stream only,
            #   teaching "MASKs remain in the block but the document is over". The AR path emits one token at a
            #   time, so one EOS suffices; supervising all 32 damages the AR path.
            # NOTE: training runs through THIS packed path. A filter added only to the unpacked path is never
            #   applied; both paths must implement it.
            #   Check: with the same weights and samples, EOS_BLOCK_FILL 0 -> 1 must not change the AR loss
            #   once the filter is active.
            # Rule: keep only the first EOS of a trailing run. Prefix labels between packed segments are -100,
            #   so a run cannot cross a segment boundary and the rule is correct per segment in packed rows.
            if getattr(self, "ar_eos_once", False) and self.tokenizer.eos_token_id is not None:
                _is_eos = (ar_labels == int(self.tokenizer.eos_token_id)) & ar_sel
                _prev_eos = torch.cat([torch.zeros_like(_is_eos[:, :1]), _is_eos[:, :-1]], dim=1)
                ar_sel = ar_sel & ~(_is_eos & _prev_eos)
            ar_logits = self.glm.lm_head(ar_shift_h[ar_sel]).float()
            ar_loss = F.cross_entropy(ar_logits, ar_labels[ar_sel])
            loss = self.c_diff * diff_loss + self.c_ar * ar_loss
            if loss_parts is not None:               # per-term record on the packed path (the training path)
                loss_parts["diff"] = diff_loss.detach(); loss_parts["ar"] = ar_loss.detach()
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss
            if loss_parts is not None:
                loss_parts["diff"] = diff_loss.detach(); loss_parts["ar"] = ar_loss
        if return_logits:
            return (logitsA, logitsB), loss
        return None, loss

    # ------------------------------------------------------------------
    # inference (Fast-dLLM v2 block-diffusion decoding)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_decode_block_mask(seg, opos, blk):
        """Block-causal flex mask for single-stream decoding [S | resp blocks].

        S (vision+prompt): causal among itself, fully visible to every response token.
        Response token in block k: sees all of S + every response token in blocks <= k
        (current block bidirectional, earlier blocks are already-decoded clean context).
        This is the inference-time collapse of the training [S | x_t | x_0] layout: previous
        response blocks are clean, so block-causal over a single stream == attending to x_0.
        """
        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            to_shared = (skv == _SHARED) & (opos[b, kv_idx] <= opos[b, q_idx])
            resp = (sq == _XT) & (skv == _XT) & (blk[b, kv_idx] <= blk[b, q_idx])
            return to_shared | resp

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device)

    def _sample_top_p(self, logits, top_p, temperature):
        """Return (token_ids, probs). Greedy when temperature <= 0; else top-p sampling."""
        logits = logits.float()
        if temperature <= 0:
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1), probs
        probs = F.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        drop = (cumsum - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        Bn, Nn, Vn = probs.shape
        choice = torch.multinomial(sorted_probs.view(-1, Vn), 1).view(Bn, Nn)
        x = torch.gather(sorted_idx, -1, choice.unsqueeze(-1)).squeeze(-1)
        return x, probs

    def _decode_logits(self, ids, image_embeds, mm_tt, image_grid_thw, block_mask, position_ids,
                       override_block_emb=None):
        """One forward of the GLM text stack over the current single-stream sequence -> logits.

        override_block_emb=(soft, mpos): SOFT-EMBEDDING trick (dInfer-style). soft is [1, bd, D] =
        the expected input embedding Sigma p(tok)*embed(tok) for the trailing bd positions; mpos is
        [1, bd] bool marking which of them are still masked. Those slots' input embedding is replaced
        by the soft (continuous) embedding instead of the hard <|mask|> embedding, giving the next
        denoise iteration a richer signal. Committed (non-mask) slots keep their hard token embed."""
        emb = self.text_model.embed_tokens(ids)
        if image_embeds is not None:
            scatter = (ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(scatter, image_embeds.to(emb.dtype))
        if override_block_emb is not None:
            soft, mpos = override_block_emb
            bd = soft.shape[1]
            tail = emb[:, -bd:, :]
            emb = emb.clone()
            emb[:, -bd:, :] = torch.where(mpos.unsqueeze(-1), soft.to(emb.dtype), tail)
        tm = self.text_model
        position_embeddings = tm.rotary_emb(emb, position_ids=position_ids)
        hidden = self._run_decoder_stack(emb, position_embeddings, block_mask, gc_active=False)
        hidden = tm.norm(hidden)
        return self.glm.lm_head(hidden)                              # [1, T, V]

    @torch.no_grad()
    def generate(self, image, prompt: str = "Text Recognition:", max_new_tokens: int = 512,
                 threshold: float = 0.99, temperature: float = 0.0, top_p: float = 0.95,
                 max_long_side: int = 1344, soft_embed: bool = False):
        """Block-diffusion OCR of a single region crop (Fast-dLLM v2 decoding, batch size 1).

        soft_embed=True enables the dInfer-style SOFT-EMBEDDING trick: within a block, still-masked
        slots are fed the expected embedding Sigma p(tok)*embed(tok) (from the previous iteration's
        predicted distribution) instead of the hard <|mask|> embedding. Inference-only; can sharpen
        the iterative refinement. The decode loop's logit-shift already aligns "distribution for p"
        to index p, so no extra shift bookkeeping is needed.

        This is the region-level OCR primitive — the same granularity GLM-OCR is trained/served
        at (one PP-DocLayout-V3 region per call, with a task prompt like "Text Recognition:").
        Pre/post-processing mirrors the official transformers path (GLMOCRRunner.ocr): long-side
        resize to ``max_long_side``, chat template with the image, decode the response span.
        Full-page OCR = run PP-DocLayout-V3 first, then call this per region (see
        ``generate_glm_ocr_dllm.py``).

        Decoding is block-level autoregressive: append a length-`bd_size` block of <|mask|>,
        then iteratively unmask within it. Each step runs the text stack, applies the training
        token-shift (token at position p predicted from the logit at p-1), and unmasks every
        position whose predicted probability exceeds `threshold` (always at least the argmax,
        to guarantee progress). Move to the next block once the current one is fully unmasked;
        stop at EOS or `max_new_tokens`.

        Note: no KV / sub-block cache yet — each step recomputes the full sequence. Correct but
        unoptimized; the hierarchical caching from Fast-dLLM v2 is the obvious next speedup.
        """
        self.eval()
        device = next(self.parameters()).device
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_long_side:                                # match official preprocessing
            scale = max_long_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.processor(text=[text], images=[image], return_tensors="pt").to(device)
        input_ids = enc["input_ids"]                                 # [1, P]
        mm_tt = enc["mm_token_type_ids"]                             # [1, P]
        pixel_values = enc.get("pixel_values")
        image_grid_thw = enc.get("image_grid_thw")
        P = input_ids.shape[1]

        # vision features: computed once, spliced into image-token slots each step
        image_embeds = None
        if pixel_values is not None:
            feats = self.mm_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(feats, dim=0).to(device)

        eos_id = self.tokenizer.eos_token_id
        generated = input_ids
        num_blocks = (max_new_tokens + self.bd_size - 1) // self.bd_size
        import os as _os, json as _json
        _TR = [] if _os.environ.get("TRACE_OUT") else None; _bi = -1

        # MRoPE positions for the prompt, computed ONCE. The appended response tokens (generated +
        # <|mask|> block) are all plain text, so their 3 rope dims simply continue +1 per token from
        # the last prompt position. (Calling get_rope_index on the mask-extended sequence is buggy for
        # many-image-token inputs: its internal token-type tensor stays prompt-length while the
        # attention mask grows by bd_size -> shape mismatch crash.)
        base_attn = torch.ones((1, P), dtype=torch.long, device=device)
        base_pos3, _ = self.mm_model.get_rope_index(
            input_ids, mm_tt, image_grid_thw=image_grid_thw, video_grid_thw=None, attention_mask=base_attn)
        last_pos = base_pos3[:, :, -1:]                              # [3, 1, 1] last prompt position

        for _ in range(num_blocks):
            _bi += 1; _blkoff = generated.shape[1] - P; _it = -1
            block = torch.full((1, self.bd_size), self.mask_id, dtype=generated.dtype, device=device)
            cur = torch.cat([generated, block], dim=1)
            cur_mm = torch.cat([mm_tt, torch.zeros((1, self.bd_size), dtype=mm_tt.dtype, device=device)], dim=1)
            T = cur.shape[1]

            # positions: prompt positions + sequential continuation for the response/mask tokens
            ext = last_pos + torch.arange(1, T - P + 1, device=device).view(1, 1, -1)   # [3, 1, T-P]
            position_ids = torch.cat([base_pos3, ext], dim=2)        # [3, 1, T]
            ar = torch.arange(T, device=device)[None, :]
            seg = torch.where(ar < P, torch.full_like(ar, _SHARED), torch.full_like(ar, _XT)).int()
            opos = ar.int()
            blk = ((ar - P).clamp(min=0) // self.bd_size).int()
            block_mask = self._build_decode_block_mask(seg, opos, blk)

            prev_block_probs = None                                              # for soft-embedding
            while (cur[:, -self.bd_size:] == self.mask_id).any():
                override = None
                if soft_embed and prev_block_probs is not None:
                    emb_w = self.text_model.embed_tokens.weight                  # [V, D]
                    soft = prev_block_probs.to(emb_w.dtype) @ emb_w              # [1, bd, D] expected embed
                    override = (soft, cur[:, -self.bd_size:] == self.mask_id)
                logits = self._decode_logits(cur, image_embeds, cur_mm, image_grid_thw,
                                             block_mask, position_ids, override_block_emb=override)
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)  # token shift
                block_logits = logits[:, -self.bd_size:, :]
                x1, probs = self._sample_top_p(block_logits, top_p, temperature)
                prev_block_probs = probs                                         # feed next iter as soft embed
                conf = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)      # [1, bd_size]
                mask_pos = cur[:, -self.bd_size:] == self.mask_id
                conf = torch.where(mask_pos, conf, torch.full_like(conf, -float("inf")))
                unmask = conf > threshold
                unmask[0, conf.argmax(dim=-1)] = True                            # guarantee progress
                unmask &= mask_pos
                import os as _os
                if _os.environ.get("LTR")=="1":                                  # strict left-to-right test
                    fm=mask_pos[0].nonzero()
                    unmask=torch.zeros_like(mask_pos)
                    if len(fm): unmask[0,int(fm[0])]=True
                _it += 1
                if _TR is not None:                          # snapshot WHOLE block this iter (pre-commit masks + all confs)
                    cj = conf[0].tolist(); xj = x1[0].tolist(); mpj = mask_pos[0].tolist()
                    cb = cur[0, -self.bd_size:].tolist()
                    _TR.append({"blk": _bi, "it": _it, "off": _blkoff,
                                "cells": [{"m": bool(mpj[j]),
                                           "conf": (round(cj[j], 4) if mpj[j] else None),
                                           "t1": int(xj[j]), "cur": int(cb[j])}
                                          for j in range(self.bd_size)]})
                cur[:, -self.bd_size:][unmask] = x1[unmask]

            generated = cur
            if eos_id is not None and (generated[0, P:] == eos_id).any():
                break

        if _TR is not None:
            _json.dump({"trace": _TR, "resp_ids": generated[0, P:].tolist(), "bd": self.bd_size},
                       open(_os.environ["TRACE_OUT"], "w"))
        resp = generated[0, P:]
        if eos_id is not None:
            hit = (resp == eos_id).nonzero()
            if hit.numel() > 0:
                resp = resp[: hit[0, 0]]
        _txt = self.tokenizer.decode(resp, skip_special_tokens=True)
        import re as _re
        return _re.sub(r"<think>.*?</think>\s*", "", _txt, flags=_re.S)  # strip GLM template scaffolding (only the diffusion model emits it; base AR doesn't)

    @staticmethod
    def _build_causal_mask(opos):
        """Plain causal flex mask over a single [S | response] stream (opos = arange index).
        This is the inference-time AR-verification mask: every token attends to all earlier tokens
        (== training x_0: shared causal + x_0 token-causal). Used to score a diffusion draft with the
        model's AR head."""
        def mask_mod(b, h, q_idx, kv_idx):
            return opos[b, kv_idx] <= opos[b, q_idx]
        B, T = opos.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=opos.device)

    @torch.no_grad()
    def generate_speculative(self, image, prompt: str = "Text Recognition:", max_new_tokens: int = 512,
                             draft_threshold: float = 0.9, temperature: float = 0.0, top_p: float = 0.95,
                             max_long_side: int = 1344, accept_conf: float = 0.0, return_stats: bool = False,
                             draft_steps: int = 1,   # 1 = MTP-style ONE-forward block draft (best tok/total-fwd, exact); None = iterative denoise
                             trace: list = None,     # if a list is passed, append per-round detail (for the spec-decode video)
                             full_draft: bool = True):  # use every mask logit (+ bonus token) as in the method; False = legacy (<= bd/round)
        """Speculative OCR decode: block-diffusion DRAFTS a bd-block, the model's own AR head VERIFIES.

        Each round: (1) DRAFT — block-diffusion-decode one bd_size block (block-causal, the existing
        Fast-dLLM v2 step) to get bd candidate tokens; (2) VERIFY — one CAUSAL forward over
        [prompt | committed | draft] gives the AR next-token argmax at each draft slot; accept the
        longest prefix where draft==AR-argmax, REPLACE the first mismatch with the AR token, advance
        the committed prefix by (accepted+1) tokens — VARIABLE, so the next block starts off the bd
        grid (block-phase-jitter training makes that in-distribution). Greedy verify ⇒ the output is
        EXACTLY the model's AR-greedy output; the diffusion draft is pure speedup (commit up to bd
        tokens per draft+verify round instead of 1/forward). accept_conf>0 relaxes the rule (accept a
        drafted token if AR assigns it prob >= accept_conf even when not the argmax) for more accepts.
        """
        self.eval()
        device = next(self.parameters()).device
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_long_side:
            scale = max_long_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.processor(text=[text], images=[image], return_tensors="pt").to(device)
        input_ids = enc["input_ids"]; mm_tt = enc["mm_token_type_ids"]
        pixel_values = enc.get("pixel_values"); image_grid_thw = enc.get("image_grid_thw")
        P = input_ids.shape[1]
        bd = self.bd_size
        image_embeds = None
        if pixel_values is not None:
            feats = self.mm_model.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(feats, dim=0).to(device)
        eos_id = self.tokenizer.eos_token_id
        base_attn = torch.ones((1, P), dtype=torch.long, device=device)
        base_pos3, _ = self.mm_model.get_rope_index(input_ids, mm_tt, image_grid_thw=image_grid_thw,
                                                    video_grid_thw=None, attention_mask=base_attn)
        last_pos = base_pos3[:, :, -1:]

        def positions_for(T):
            ext = last_pos + torch.arange(1, T - P + 1, device=device).view(1, 1, -1)
            return torch.cat([base_pos3, ext], dim=2)

        generated = input_ids
        n_draft_fwd = n_verify_fwd = n_accepted = n_rounds = 0
        done = False
        while not done and (generated.shape[1] - P) < max_new_tokens:
            n_rounds += 1
            C = generated.shape[1] - P
            # ---- DRAFT: block-diffusion decode one bd-block (block-causal) ----
            block = torch.full((1, bd), self.mask_id, dtype=generated.dtype, device=device)
            cur = torch.cat([generated, block], dim=1)
            cur_mm = torch.cat([mm_tt, torch.zeros((1, bd), dtype=mm_tt.dtype, device=device)], dim=1)
            T = cur.shape[1]
            position_ids = positions_for(T)
            ar = torch.arange(T, device=device)[None, :]
            seg = torch.where(ar < P, torch.full_like(ar, _SHARED), torch.full_like(ar, _XT)).int()
            opos = ar.int()
            blk = ((ar - P).clamp(min=0) // bd).int()
            draft_mask = self._build_decode_block_mask(seg, opos, blk)
            # draft_steps=K caps the diffusion-draft forwards per block (MTP-style: K=1 = ONE forward,
            # argmax-fill the whole block). Default (None) = iterative denoise until the block fills
            # (committing only conf>threshold tokens per forward -> many forwards/block, slow draft).
            d_step = 0
            while (cur[:, -bd:] == self.mask_id).any():
                logits_raw = self._decode_logits(cur, image_embeds, cur_mm, image_grid_thw, draft_mask, position_ids)
                logits = torch.cat([logits_raw[:, :1, :], logits_raw[:, :-1, :]], dim=1)
                bl = logits[:, -bd:, :]
                x1, probs = self._sample_top_p(bl, top_p, temperature)
                conf = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)
                mask_pos = cur[:, -bd:] == self.mask_id
                conf = torch.where(mask_pos, conf, torch.full_like(conf, -float("inf")))
                d_step += 1
                if draft_steps is not None and d_step >= draft_steps:
                    unmask = mask_pos                                        # final allowed step: fill ALL remaining
                else:
                    unmask = conf > draft_threshold
                    unmask[0, conf.argmax(dim=-1)] = True
                    unmask &= mask_pos
                cur[:, -bd:][unmask] = x1[unmask]
                n_draft_fwd += 1
            draft = cur[:, -bd:]                                          # [1, bd] drafted tokens
            if full_draft:
                # the LAST mask's logit predicts the token right after the block (d_bd) -> use it too
                x_extra, _ = self._sample_top_p(logits_raw[:, -1:, :], top_p, temperature)
                draft = torch.cat([draft, x_extra], dim=1)               # [1, bd+1] = a0, d_1..d_bd
            nd = draft.shape[1]
            # ---- VERIFY: one causal AR forward over [prompt | committed | draft] ----
            cur_v = torch.cat([generated, draft], dim=1)
            cur_mm_v = torch.cat([mm_tt, torch.zeros((1, nd), dtype=mm_tt.dtype, device=device)], dim=1)
            Tv = cur_v.shape[1]
            opos_v = torch.arange(Tv, device=device)[None, :].int()
            verify_mask = self._build_causal_mask(opos_v)
            logits_vr = self._decode_logits(cur_v, image_embeds, cur_mm_v, image_grid_thw, verify_mask, positions_for(Tv))
            logits_v = torch.cat([logits_vr[:, :1, :], logits_vr[:, :-1, :]], dim=1)  # token shift
            n_verify_fwd += 1
            ar_slice = logits_v[:, P + C:P + C + nd, :]                   # predictions for the nd draft slots
            ar_probs = F.softmax(ar_slice.float(), dim=-1)
            ar_pred = ar_probs.argmax(dim=-1)                            # [1, nd]
            d = draft[0]; ap = ar_pred[0]
            match = (d == ap)
            if accept_conf > 0:                                          # relaxed: also accept if AR prob(draft) high
                pd = torch.gather(ar_probs[0], -1, d.unsqueeze(-1)).squeeze(-1)
                match = match | (pd >= accept_conf)
            # longest accepted prefix = #leading True
            nz = (~match).nonzero()
            k = int(nz[0, 0]) if nz.numel() > 0 else nd
            if k < nd:
                new = torch.cat([d[:k], ap[k:k + 1]], dim=0)            # accepted prefix + AR correction
            elif full_draft:
                bonus = logits_vr[0, -1, :].float().argmax(-1, keepdim=True).to(d.dtype)  # verifier's a_{bd+1}
                new = torch.cat([d, bonus], dim=0)                       # whole draft accepted + bonus
            else:
                new = d                                                  # whole block accepted
            if trace is not None:                                        # per-round detail for the spec video
                dec = lambda t: self.tokenizer.decode([int(t)], skip_special_tokens=False)
                trace.append({
                    "round": n_rounds,
                    "committed": self.tokenizer.decode(generated[0, P:], skip_special_tokens=False) if C > 0 else "",
                    "draft": [dec(t) for t in d.tolist()],               # 32 drafted tokens (diffusion fill)
                    "draft_ids": d.tolist(),                             # raw IDs (for cached-vs-uncached draft-match diag)
                    "ar_pred": [dec(t) for t in ap.tolist()],            # AR-verify argmax at each slot
                    "ar_pred_ids": ap.tolist(),
                    "match": match.tolist(),                             # which draft slots AR agreed with
                    "k": k,                                              # accepted prefix length (re-mask starts here)
                    "correction": dec(ap[k]) if k < bd else None,        # AR token that replaces the first mismatch
                    "accepted_text": self.tokenizer.decode(new, skip_special_tokens=False),
                })
            generated = torch.cat([generated, new.unsqueeze(0)], dim=1)
            n_accepted += int(new.shape[0])
            if eos_id is not None and (new == eos_id).any():
                done = True

        resp = generated[0, P:]
        if eos_id is not None:
            hit = (resp == eos_id).nonzero()
            if hit.numel() > 0:
                resp = resp[: hit[0, 0]]
        import re as _re
        txt = _re.sub(r"<think>.*?</think>\s*", "",
                      self.tokenizer.decode(resp, skip_special_tokens=True), flags=_re.S)
        if return_stats:
            fwd = n_draft_fwd + n_verify_fwd
            stats = {"rounds": n_rounds, "draft_fwd": n_draft_fwd, "verify_fwd": n_verify_fwd,
                     "total_fwd": fwd, "resp_tokens": int(resp.shape[0]),
                     "tokens_per_fwd": round(int(resp.shape[0]) / max(1, fwd), 3),
                     "accept_per_round": round(n_accepted / max(1, n_rounds), 2)}
            return txt, stats
        return txt
