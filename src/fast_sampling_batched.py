"""Batched (bs>1) KV-cached block-diffusion decoding for GlmOcrBlockDiffusion.

New file — does NOT import or mutate fast_sampling.py / models/*. It only
*reads* the model's public handles (text_model, mm_model, glm.lm_head, processor, mask_id,
image_token_id, bd_size, tokenizer, _sample_top_p), exactly as fast_sampling.py does, and mirrors
that file's block-causal KV-cache logic for a *batch* of region crops.

What's new vs fast_sampling.generate_cached (bs=1)
--------------------------------------------------
`generate_cached_batched` decodes a list of crops in one batched forward stream. Crops have
different prefix lengths (different #vision tokens), so the ragged prefixes are **left-padded** to
a common length ``Pmax`` and a per-sequence ``pad_len[b]`` (number of left-pad columns) drives the
attention mask — every query attends only ``kv >= pad_len[b]`` (real tokens), so the padded KV is
inert. The KV cache then stays perfectly rectangular: after prefill its length is ``Pmax`` for all
rows, and each decoded block grows it by ``bd`` uniformly. Per-row differences are carried entirely
by (1) the pad mask, (2) per-row MRoPE start positions, and (3) per-row early-finish bookkeeping.

Equivalence rationale
----------------------
Rows never cross-attend (attention is per-row), so each row's decode is identical to running it
alone at bs=1 — *up to floating-point*: batched GEMMs can pick different kernels / reduction orders
than a bs=1 GEMM, so a near-tie argmax can flip. At ``temperature=0`` the algorithm is otherwise
deterministic and bit-for-bit identical per op. A finished row is kept in the (rectangular) batch
but its tokens stop being recorded after EOS; its continued forwards are wasted compute that cannot
affect other rows.

!!! UNTESTED in the authoring environment — validate with ``test_batched_equivalence.py`` on CUDA.
Each load-bearing assumption is tagged ``[RISK]`` and mirrors a ``[RISK]`` already in fast_sampling.
"""

import torch
from PIL import Image


# --- cache helpers (re-defined locally; identical semantics to fast_sampling, no import coupling) ---

def _truncate_cache(cache, length):
    """Crop a DynamicCache back to `length` along the sequence axis (all layers).

    transformers 5.x stores per-layer K/V in ``cache.layers[i].keys/.values``."""
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None and layer.keys.shape[-2] > length:
            layer.keys = layer.keys[:, :, :length, :].contiguous()
            layer.values = layer.values[:, :, :length, :].contiguous()


def _run_layers(model, emb, position_ids, attn_mask, cache):
    """Run the GLM text decoder stack over `emb`, appending KV into `cache`. Returns hidden.

    [RISK] passes past_key_values=cache + use_cache=True to each layer and assumes the layer
    appends to its own slot (HF Glm/Qwen decoder layers do via self.layer_idx)."""
    tm = model.text_model
    pe = tm.rotary_emb(emb, position_ids=position_ids)
    hidden = emb
    for layer in tm.layers:
        out = layer(hidden, position_embeddings=pe, attention_mask=attn_mask,
                    past_key_values=cache, use_cache=True)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


def _preprocess_one(model, image, prompt, max_long_side, device):
    """Encode a single crop into (input_ids[1,P], mm_tt[1,P], pos3[3,1,P], next_pos, image_embeds).

    Identical preprocessing to fast_sampling.generate_cached / model.generate."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_long_side:
        s = max_long_side / max(w, h)
        image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = model.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = model.processor(text=[text], images=[image], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]                                      # [1, P]
    mm_tt = enc["mm_token_type_ids"]
    pixel_values = enc.get("pixel_values")
    image_grid_thw = enc.get("image_grid_thw")
    P = input_ids.shape[1]

    image_embeds = None
    if pixel_values is not None:
        feats = model.mm_model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True).pooler_output
        image_embeds = torch.cat(feats, dim=0).to(device)            # [n_vis, D]

    attn_prefix = torch.ones((1, P), dtype=torch.long, device=device)
    pos3, _ = model.mm_model.get_rope_index(
        input_ids, mm_tt, image_grid_thw=image_grid_thw, video_grid_thw=None,
        attention_mask=attn_prefix)                                  # [3, 1, P]
    next_pos = int(pos3.max().item()) + 1
    return input_ids, mm_tt, pos3, next_pos, image_embeds


@torch.no_grad()
def generate_cached_batched(model, images, prompt="Text Recognition:", max_new_tokens=512,
                            threshold=0.99, temperature=0.0, top_p=0.95, max_long_side=1344):
    """Batched block-diffusion OCR of a list of region crops. Returns list[str] (one per crop).

    Mirrors fast_sampling.generate_cached per-row; see module docstring for the batching scheme.
    """
    from transformers import DynamicCache
    from torch.nn.attention.flex_attention import create_block_mask

    if isinstance(images, Image.Image):
        images = [images]
    bs = len(images)
    model.eval()
    device = next(model.parameters()).device
    tm = model.text_model
    bd = model.bd_size
    mask_id = model.mask_id
    eos_id = model.tokenizer.eos_token_id
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    # Stop on the model's FULL generation_config eos set, not just tokenizer.eos_token_id. GLM-OCR's
    # generation_config.eos_token_id = [59246 <|endoftext|>, 59253 <|user|>]; stopping only on 59246
    # lets a <|user|>-terminated generation run on -> over-generation. (pad_id stays = eos_id 59246.)
    _gce = getattr(getattr(model.glm, "generation_config", None), "eos_token_id", None)
    stop_ids = set(int(x) for x in _gce) if isinstance(_gce, (list, tuple)) else set()
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    ids_dtype = torch.long

    # ---- per-crop preprocess; collect ragged prefixes ----
    per = [_preprocess_one(model, img, prompt, max_long_side, device) for img in images]
    lens = [p[0].shape[1] for p in per]
    Pmax = max(lens)
    pad_len = torch.tensor([Pmax - L for L in lens], device=device)   # [bs] left-pad columns/row
    next_pos = torch.tensor([p[3] for p in per], device=device)       # [bs] first response position

    def embed_block(ids):
        """Embed response/block tokens (never image tokens -> no vision splice)."""
        return tm.embed_tokens(ids)

    # ---- build left-padded prefix embeddings [bs, Pmax, D] and position_ids [3, bs, Pmax] ----
    sample_emb = tm.embed_tokens(torch.tensor([[pad_id]], device=device))
    D = sample_emb.shape[-1]
    emb_dtype = sample_emb.dtype
    prefix_emb = torch.zeros((bs, Pmax, D), dtype=emb_dtype, device=device)
    position_ids = torch.zeros((3, bs, Pmax), dtype=torch.long, device=device)
    pad_emb = tm.embed_tokens(torch.tensor([pad_id], device=device))[0]   # [D]
    for b, (input_ids, mm_tt, pos3, _np, image_embeds) in enumerate(per):
        L = input_ids.shape[1]
        off = Pmax - L
        e = tm.embed_tokens(input_ids)                                # [1, L, D]
        if image_embeds is not None:
            m = (input_ids == model.image_token_id).unsqueeze(-1).expand_as(e)
            e = e.masked_scatter(m, image_embeds.to(e.dtype))
        if off > 0:
            prefix_emb[b, :off] = pad_emb                             # inert (masked out)
        prefix_emb[b, off:] = e[0]
        position_ids[:, b, off:] = pos3[:, 0, :]                      # real positions right-aligned

    # ---- prefill: causal within real prefix, exclude pad columns; keep diagonal for empty-row safety ----
    pl = pad_len
    causal = create_block_mask(
        lambda b, h, q, kv: ((kv <= q) & (kv >= pl[b])) | (kv == q),
        B=bs, H=None, Q_LEN=Pmax, KV_LEN=Pmax, device=device)
    cache = DynamicCache()
    hidden = _run_layers(model, prefix_emb, position_ids, causal, cache)
    frozen = Pmax

    def seed_from(hidden_states):
        """Seed each row's next block[0] from its last (right-aligned) clean position's logit."""
        h = tm.norm(hidden_states[:, -1:, :])
        logits = model.glm.lm_head(h)                                 # [bs, 1, V]
        tok, _ = model._sample_top_p(logits, top_p, temperature)      # [bs, 1]
        return tok[:, 0]                                              # [bs]

    seed = seed_from(hidden)                                          # [bs]
    generated = [[] for _ in range(bs)]
    done = [False] * bs
    if stop_ids:
        for b in range(bs):
            if int(seed[b].item()) in stop_ids:
                done[b] = True
    num_blocks = (max_new_tokens + bd - 1) // bd
    rows = torch.arange(bs, device=device)

    for blk in range(num_blocks):
        if all(done):
            break
        block_ids = torch.full((bs, bd), mask_id, dtype=ids_dtype, device=device)
        block_ids[:, 0] = seed                                        # block[0] = clean seed (per row)
        base = next_pos + blk * bd                                    # [bs] per-row block start position
        block_pos = base[:, None] + torch.arange(bd, device=device)[None, :]   # [bs, bd]
        block_pos3 = block_pos[None, :, :].expand(3, bs, bd)
        KV = frozen + bd
        # current block sees ALL real cached past + the block itself bidirectionally (kv >= pad_len[b]);
        # block columns (>= frozen) always pass. [RISK] flex concatenates cached K/V (len frozen) + block.
        full = create_block_mask(
            lambda b, h, q, kv: kv >= pl[b], B=bs, H=None, Q_LEN=bd, KV_LEN=KV, device=device)

        # iterative unmask; runs until EVERY row's block is clean. A row with no masks left no-ops
        # (mpos all False -> unmask&=mpos clears it), matching its standalone bs=1 trajectory.
        while (block_ids == mask_id).any():
            _truncate_cache(cache, frozen)
            hb = _run_layers(model, embed_block(block_ids), block_pos3, full, cache)
            hb = tm.norm(hb)
            logits = model.glm.lm_head(hb)                            # [bs, bd, V]
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)  # in-block token shift
            x1, probs = model._sample_top_p(logits, top_p, temperature)
            conf = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)       # [bs, bd]
            mpos = block_ids == mask_id
            conf = torch.where(mpos, conf, torch.full_like(conf, -float("inf")))
            unmask = conf > threshold
            unmask[rows, conf.argmax(dim=-1)] = True                  # >=1 token/row/step
            unmask &= mpos
            block_ids[unmask] = x1[unmask]

        # persist the finished clean block's KV; seed next block from its last logit
        _truncate_cache(cache, frozen)
        hb = _run_layers(model, embed_block(block_ids), block_pos3, full, cache)
        frozen += bd
        seed = seed_from(hb)

        for b in range(bs):
            if done[b]:
                continue
            for t in block_ids[b].tolist():
                if t in stop_ids:
                    done[b] = True
                    break
                generated[b].append(t)
            if not done[b] and int(seed[b].item()) in stop_ids:
                done[b] = True                                        # next seed is a stop token -> stop (not emitted)

    import re as _re
    def _strip(_t):   # GLM template scaffolding the diffusion model emits; eval-side strip (matches single-row sampler)
        return _re.sub(r"<think>.*?</think>\s*", "", _t, flags=_re.S)
    return [_strip(model.tokenizer.decode(torch.tensor(g, device=device), skip_special_tokens=True))
            if g else "" for g in generated]
