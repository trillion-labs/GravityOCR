"""Fast (KV-cached) block-diffusion decoding for GlmOcrBlockDiffusion.

CACHING MAP (authoritative; full table in docs/DECODE_PATHS.md):
  model.generate (in models/glm_ocr_dllm.py) : NO cache — recomputes full seq each step (slow REFERENCE).
  generate_cached / generate_cached_batched   : REAL full block-causal KV cache (prefix + ALL committed
                                                blocks). eval.sh --fast uses generate_cached_batched.
                                                NOT just prefix. batched==bs1 only up-to-fp + UNTESTED.
  generate_speculative_prefixcache            : PREFIX-ONLY cache ("precache"); re-forwards response each
                                                round. byte-identical to AR-greedy (verified).
  generate_speculative_natcache               : REAL full committed cache (token-causal incremental,
                                                2 fwd/round). score==AR-greedy (verified). [seed-carry fix]
  generate_speculative_cached / _onecache     : SUPERSEDED / FAILED dead code — never use.
Equivalence NOT verified for diffusion: model.generate(bs1,uncached) == generate_cached(bs1) ==
generate_cached_batched(bs8). Do NOT vary diffusion batch size for scoring until that chain is verified.

Standalone on purpose — it does NOT edit models/glm_ocr_dllm.py (which has uncommitted WIP);
it only *reads* the model's public handles (text_model, mm_model, glm.lm_head, processor,
mask_id, image_token_id, bd_size, _sample_top_p). Used by infer_omnidocbench.py --fast, which
wraps it in try/except and falls back to the verified model.generate() on any error.

Why this is faster
------------------
model.generate() recomputes the FULL sequence ``[prefix | every response block]`` at every
unmask step. The prefix holds many vision tokens, so that O((P+R)^2) recompute dominates. Here
we mirror Fast-dLLM v2's ``batch_sample`` (Fast-dLLM/v2/generation_functions.py): keep a KV
cache for ``[prefix + already-decoded clean blocks]`` and, at each unmask step, forward ONLY the
current ``bd_size`` block (bd_size queries) against that cache. Within a block the cache must
stay frozen, so after each probe forward we truncate the DynamicCache back to the frozen length;
once the block is fully unmasked we persist its KV with one clean forward.

The unmask math is identical to model.generate() (training token-shift, top-p sample,
``conf > threshold`` unmask with an argmax guarantee), so results should match the slow path
up to sampling noise.

!!! UNTESTED in the authoring environment (no transformers / GPU there). Validate on the
instance. Each load-bearing assumption is tagged ``[RISK]``; if any is wrong the caller falls
back to the slow sampler, so eval never breaks — but the speedup won't kick in until these are
confirmed against the real GLM-OCR transformers build.

Possible further speedups not implemented here: a within-block sub-block cache (use_block_cache /
small_block_size / replace_position) and true multi-region batching with early-finish.
"""

import torch
from PIL import Image

_SHARED, _XT = 1, 2  # segment tags (match model._build_decode_block_mask)


def enable_fused_flex(dynamic=True):
    """Keep flex_attention FUSED across cached decode's varying KV_LEN.

    The model uses ``_attn_implementation='flex_attention'`` → transformers compiles flex via the
    ``WrappedFlexAttention`` singleton. On torch 2.7+ it does ``torch.compile(flex_attention)`` with
    ``dynamic=None`` (auto): it compiles statically and RE-compiles on every new (Q_LEN, KV_LEN).
    Cached decode changes KV_LEN every step → it blows past ``torch._dynamo.config.recompile_limit``
    (default 8) → dynamo DISABLES the compiled fn and silently falls back to the **unfused** kernel
    (the "flex_attention called without torch.compile" warning). That fallback is both slow AND the
    source of the run-to-run argmax jitter (fused vs unfused differ ~1e-3, flips near-tie greedy picks).

    Forcing ``dynamic=True`` makes the FIRST compile shape-polymorphic, so one fused kernel serves all
    KV_LEN — no recompiles, no fallback, fast AND reproducible. Call ONCE before any forward.
    Also lifts the recompile limit as a belt-and-suspenders for create_block_mask / the prefill shape.
    """
    import transformers.integrations.flex_attention as _fa
    from torch.nn.attention.flex_attention import flex_attention as _flex
    for attr in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                 "accumulated_cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, max(getattr(torch._dynamo.config, attr), 256))
    w = _fa.WrappedFlexAttention(False)             # singleton (training=False)
    w._compiled_flex_attention = torch.compile(_flex, dynamic=dynamic)
    w._is_flex_compiled = True
    w.training = False
    return w


def _truncate_cache(cache, length):
    """Crop a DynamicCache back to `length` along the sequence axis (all layers).

    transformers 5.x stores per-layer K/V in ``cache.layers[i].keys/.values`` (the older
    ``cache.key_cache[i]/.value_cache[i]`` lists were removed)."""
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None and layer.keys.shape[-2] > length:
            layer.keys = layer.keys[:, :, :length, :].contiguous()
            layer.values = layer.values[:, :, :length, :].contiguous()


def _run_layers(model, emb, position_ids, attn_mask, cache):
    """Run the GLM text decoder stack over `emb`, appending KV into `cache`. Returns hidden.

    [RISK] passes past_key_values=cache + use_cache=True to each layer and assumes the layer
    appends to its own slot (HF Glm/Qwen decoder layers do via self.layer_idx). The model's own
    code calls layers with past_key_values= but no use_cache; some builds return a tuple."""
    tm = model.text_model
    pe = tm.rotary_emb(emb, position_ids=position_ids)
    hidden = emb
    for layer in tm.layers:
        out = layer(hidden, position_embeddings=pe, attention_mask=attn_mask,
                    past_key_values=cache, use_cache=True)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


@torch.no_grad()
def generate_cached(model, image, prompt="Text Recognition:", max_new_tokens=512,
                    threshold=0.99, temperature=0.0, top_p=0.95, max_long_side=1344, commit_trace=None,
                    commit_causal=False, commit_detail=None):
    # commit_causal: this is the OLD generate_cached path; for the validated token-causal CACHED SPEC
    # see generate_speculative_natcache (Nemotron-style, byte-identical to AR-greedy on the eval workload).
    # Committing with token-causal context is quality-neutral versus block-causal context, and the
    # token-causal cache accepts more tokens per round; natcache is therefore the preferred cached path.
    from transformers import DynamicCache
    from torch.nn.attention.flex_attention import create_block_mask

    model.eval()
    device = next(model.parameters()).device
    tm = model.text_model
    bd = model.bd_size
    mask_id = model.mask_id
    eos_id = model.tokenizer.eos_token_id

    # ---- preprocess (identical to model.generate) ----
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_long_side:
        s = max_long_side / max(w, h)
        image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = model.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = model.processor(text=[text], images=[image], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    mm_tt = enc["mm_token_type_ids"]
    pixel_values = enc.get("pixel_values")
    image_grid_thw = enc.get("image_grid_thw")
    P = input_ids.shape[1]

    image_embeds = None
    if pixel_values is not None:
        feats = model.mm_model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True).pooler_output
        image_embeds = torch.cat(feats, dim=0).to(device)

    def embed(ids):
        e = tm.embed_tokens(ids)
        if image_embeds is not None:
            m = (ids == model.image_token_id).unsqueeze(-1).expand_as(e)
            e = e.masked_scatter(m, image_embeds.to(e.dtype))
        return e

    # ---- MRoPE positions: prefix via get_rope_index; response continues +1 on all 3 axes ----
    attn_prefix = torch.ones((1, P), dtype=torch.long, device=device)
    pos_prefix, _ = model.mm_model.get_rope_index(
        input_ids, mm_tt, image_grid_thw=image_grid_thw, video_grid_thw=None,
        attention_mask=attn_prefix)                                   # [3, 1, P]
    next_pos = int(pos_prefix.max().item()) + 1   # [RISK] GLM text-continuation start position

    # ---- 1) prefill prefix into the cache (causal within prefix) ----
    cache = DynamicCache()
    causal = create_block_mask(lambda b, h, q, kv: kv <= q, B=1, H=None,
                               Q_LEN=P, KV_LEN=P, device=device)
    hidden = _run_layers(model, embed(input_ids), pos_prefix, causal, cache)
    frozen = P                                                        # cache len = clean prefix

    def seed_from(hidden_states):
        """Seed the next block's FIRST token from the last position's logit (no token-shift).

        Mirrors batch_sample line 83 (next_token = logits[:,-1:]). Necessary because, with only
        the block fed against the cache, the in-block shift cannot predict block[0] — its
        predictor is the last clean token, whose logit lives outside the block window. The slow
        model.generate() gets this for free from the global shift; here we seed it explicitly."""
        h = tm.norm(hidden_states[:, -1:, :])
        tok, _ = model._sample_top_p(model.glm.lm_head(h), top_p, temperature)  # [1, 1]
        return int(tok.item())

    seed = seed_from(hidden)                                          # response[0]
    generated: list[int] = []
    done = False
    num_blocks = (max_new_tokens + bd - 1) // bd

    for _ in range(num_blocks):
        if done or (eos_id is not None and seed == eos_id):
            break
        block_ids = torch.full((1, bd), mask_id, dtype=input_ids.dtype, device=device)
        if commit_detail is not None:
            # Also record the block seed (first token). It is decided outside the unmask loop, so it used
            # to be missing from commit_detail, leaving a hole every bd positions in the reconstructed
            # sequence and breaking the GT alignment at each of them.
            commit_detail.append({"block": int(frozen // bd), "step": len(commit_detail),
                                  "pos": [int(frozen)], "in_block_pos": [0],
                                  "tok": [int(seed)], "conf": [1.0], "is_seed": True})
        block_ids[0, 0] = seed                                       # block[0] = clean seed
        base = next_pos + len(generated)
        block_pos3 = torch.arange(base, base + bd, device=device)[None, None, :].expand(3, 1, bd)
        # current-block query sees ALL cached past + itself bidirectionally -> full mask.
        # [RISK] assumes flex_attention concatenates cached K/V (len `frozen`) with the new
        # block (len bd) and accepts a [bd, frozen+bd] BlockMask.
        full = create_block_mask(
            lambda b, h, q, kv: kv >= 0, B=1, H=None, Q_LEN=bd, KV_LEN=frozen + bd, device=device)

        # iterative unmask of positions 1..bd-1 (position 0 is the clean seed) against frozen cache
        while (block_ids == mask_id).any():
            _truncate_cache(cache, frozen)                           # keep cache frozen
            hb = _run_layers(model, embed(block_ids), block_pos3, full, cache)
            hb = tm.norm(hb)
            logits = model.glm.lm_head(hb)                           # [1, bd, V]
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)  # in-block token shift
            x1, probs = model._sample_top_p(logits, top_p, temperature)
            conf = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)
            mpos = block_ids == mask_id
            conf = torch.where(mpos, conf, torch.full_like(conf, -float("inf")))
            unmask = conf > threshold
            unmask[0, conf.argmax(dim=-1)] = True                    # guarantee progress
            unmask &= mpos
            block_ids[unmask] = x1[unmask]
            if commit_trace is not None: commit_trace.append(int(unmask.sum()))   # tokens unmasked this forward
            # commit_detail: which token was committed at which position on which forward.
            #   Used for visualization of how positions inside a block are committed in parallel;
            #   counts alone (commit_trace) are not enough. Pure bookkeeping, the decode is unchanged.
            if commit_detail is not None:
                _idx = unmask[0].nonzero(as_tuple=True)[0].tolist()
                if _idx:
                    # pred_all: argmax + confidence of every still-masked position at this forward, so one can
                    #   see not only what was committed but what the alternatives looked like.
                    #   Recording only; bd=32 means 32 pairs per step, negligible.
                    _mp = mpos[0].nonzero(as_tuple=True)[0].tolist()
                    _pred = {"pos": _mp,
                             "tok": [int(x1[0, i]) for i in _mp],
                             "conf": [round(float(conf[0, i]), 4) for i in _mp]}
                    commit_detail.append({"pred_all": _pred,
                                          "block": int(frozen // bd), "step": len(commit_detail),
                                          "pos": [int(frozen + i) for i in _idx],
                                          "in_block_pos": _idx,
                                          "tok": [int(x1[0, i]) for i in _idx],
                                          "conf": [round(float(conf[0, i]), 4) for i in _idx]})

        # persist the finished clean block's KV; seed the next block from its last logit.
        # AR-loss models: commit TOKEN-causal (kv<=frozen+q) so the cached x_0 KV matches training.
        _truncate_cache(cache, frozen)
        commit_mask = full
        if commit_causal:
            commit_mask = create_block_mask(lambda b, h, q, kv, f=frozen: kv <= f + q,
                                            B=1, H=None, Q_LEN=bd, KV_LEN=frozen + bd, device=device)
        hb = _run_layers(model, embed(block_ids), block_pos3, commit_mask, cache)
        frozen += bd
        seed = seed_from(hb)

        for t in block_ids[0].tolist():
            if eos_id is not None and t == eos_id:
                done = True
                break
            generated.append(t)

    if not generated:
        return ""
    _txt = model.tokenizer.decode(torch.tensor(generated, device=device), skip_special_tokens=True)
    import re as _re
    return _re.sub(r"<think>.*?</think>\s*", "", _txt, flags=_re.S)  # strip GLM template scaffolding (only the diffusion model emits it)


@torch.no_grad()
def generate_speculative_prefixcache(model, image, prompt="Text Recognition:", max_new_tokens=512,
                                     draft_threshold=0.9, temperature=0.0, top_p=0.95,
                                     max_long_side=1344, accept_conf=0.0, draft_steps=1,
                                     return_stats=False, trace=None, full_draft=True):
    """FULL-FIX cached spec decode: caches ONLY the vision+prompt PREFIX KV, and each round RE-FORWARDS
    the (short) committed response + draft block against that cached prefix, using the EXACT uncached
    masks. This replicates model.generate_speculative's accept (the grid bd-block-causal DRAFT +
    token-causal VERIFY + global token-shift seed) while skipping the expensive vision recompute.

    Why this matches uncached where generate_speculative_cached did not (internal debugging notes):
      - DRAFT mask = bd-GRID block-causal (blk=idx//bd over the response), NOT one aligned bidirectional
        block -> off-grid commits (~14 tok/round) straddle grid blocks exactly like uncached.
      - block[0] is <mask> during the draft forward; its token comes from the global token-shift off the
        last committed token (in the re-forwarded response) -> no OOD pre-seed. (Round 0 only: block[0]
        comes from the prefill seed, since the last prefix token isn't in the response forward.)
      - committed response is RE-FORWARDED each round (cheap; it's short), so its representation is the
        same grid-block-causal one uncached uses every round -> no token-causal committed-KV drift.
    Cost: 2 forwards/round over (R+bd) response queries vs the cached P-token prefix (P = many vision
    tokens) -> per-forward compute ~ (R+bd)/(P+R+bd) of the uncached full re-forward."""
    from transformers import DynamicCache
    from torch.nn.attention.flex_attention import create_block_mask

    model.eval()
    device = next(model.parameters()).device
    tm = model.text_model; bd = model.bd_size
    mask_id = model.mask_id; eos_id = model.tokenizer.eos_token_id

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_long_side:
        s = max_long_side / max(w, h); image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = model.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = model.processor(text=[text], images=[image], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]; mm_tt = enc["mm_token_type_ids"]
    pixel_values = enc.get("pixel_values"); image_grid_thw = enc.get("image_grid_thw")
    P = input_ids.shape[1]
    image_embeds = None
    if pixel_values is not None:
        feats = model.mm_model.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
        image_embeds = torch.cat(feats, dim=0).to(device)

    def embed(ids):
        e = tm.embed_tokens(ids)
        if image_embeds is not None:
            mm = (ids == model.image_token_id).unsqueeze(-1).expand_as(e)
            e = e.masked_scatter(mm, image_embeds.to(e.dtype))
        return e

    attn = torch.ones((1, P), dtype=torch.long, device=device)
    pos_prefix, _ = model.mm_model.get_rope_index(input_ids, mm_tt, image_grid_thw=image_grid_thw,
                                                  video_grid_thw=None, attention_mask=attn)
    next_pos = int(pos_prefix.max().item()) + 1

    # prefill prefix (causal) -> cache; this is the ONLY thing kept across rounds
    cache = DynamicCache()
    causal = create_block_mask(lambda b, h, q, kv: kv <= q, B=1, H=None, Q_LEN=P, KV_LEN=P, device=device)
    hpre = _run_layers(model, embed(input_ids), pos_prefix, causal, cache)
    seed0 = int(model.glm.lm_head(tm.norm(hpre[:, -1:, :])).argmax(-1).item())   # AR pred of response[0]

    committed = torch.empty((1, 0), dtype=input_ids.dtype, device=device)
    n_draft_fwd = n_verify_fwd = n_rounds = n_accepted = 0
    done = False

    def draft_mask(R):   # bd-grid block-causal over [R+bd queries | P+R+bd kv], == _build_decode_block_mask
        Tr = R + bd
        def mm(b, h, q, kv):
            return (kv < P) | ((kv >= P) & (((kv - P) // bd) <= (q // bd)))
        return create_block_mask(mm, B=1, H=None, Q_LEN=Tr, KV_LEN=P + Tr, device=device)

    def verify_mask(R, extra=0):  # fully causal over [R+bd(+extra) | P+R+bd(+extra)], == _build_causal_mask
        Tr = R + bd + extra
        def mm(b, h, q, kv):
            return (kv < P) | ((kv - P) <= q)
        return create_block_mask(mm, B=1, H=None, Q_LEN=Tr, KV_LEN=P + Tr, device=device)

    while not done and committed.shape[1] < max_new_tokens:
        n_rounds += 1
        R = committed.shape[1]; Tr = R + bd
        block = torch.full((1, bd), mask_id, dtype=input_ids.dtype, device=device)
        resp = torch.cat([committed, block], dim=1)
        pos3 = torch.arange(next_pos, next_pos + Tr, device=device)[None, None, :].expand(3, 1, Tr)
        dm = draft_mask(R)

        # ---- DRAFT: forward [committed | mask*bd] vs cached prefix; shift; fill the bd block ----
        d_step = 0
        while (resp[:, -bd:] == mask_id).any():
            _truncate_cache(cache, P)
            logits_raw = model.glm.lm_head(tm.norm(_run_layers(model, embed(resp), pos3, dm, cache)))
            logits = torch.cat([logits_raw[:, :1, :], logits_raw[:, :-1, :]], dim=1)  # global token shift
            bl = logits[:, -bd:, :]
            x1, probs = model._sample_top_p(bl, top_p, temperature)
            conf = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)
            mpos = resp[:, -bd:] == mask_id
            conf = torch.where(mpos, conf, torch.full_like(conf, -float("inf")))
            d_step += 1
            if draft_steps is not None and d_step >= draft_steps:
                unmask = mpos
            else:
                unmask = conf > draft_threshold; unmask[0, conf.argmax(dim=-1)] = True; unmask &= mpos
            resp[:, -bd:][unmask] = x1[unmask]
            n_draft_fwd += 1
        if R == 0:
            resp[0, 0] = seed0                                                       # round-0 block[0] from prefill
        draft = resp[0, -bd:].clone()
        if full_draft:
            # the LAST mask's logit predicts the token right after the block (d_bd) -> use it too
            x_extra, _ = model._sample_top_p(logits_raw[:, -1:, :], top_p, temperature)
            draft = torch.cat([draft, x_extra[0]])                                   # [bd+1] = a0, d_1..d_bd
        nd = draft.shape[0]

        # ---- VERIFY: causal forward [committed | draft] vs cached prefix; shift; AR argmax ----
        _truncate_cache(cache, P)
        resp_v = torch.cat([committed, draft[None, :]], dim=1)
        Tv = resp_v.shape[1]
        pos3v = torch.arange(next_pos, next_pos + Tv, device=device)[None, None, :].expand(3, 1, Tv)
        lv_raw = model.glm.lm_head(tm.norm(_run_layers(model, embed(resp_v), pos3v, verify_mask(R, nd - bd), cache)))
        lv = torch.cat([lv_raw[:, :1, :], lv_raw[:, :-1, :]], dim=1)
        ar_probs = torch.softmax(lv[:, -nd:, :].float(), dim=-1)[0]                  # [nd, V]
        ar_pred = ar_probs.argmax(-1)
        if R == 0:
            ar_pred[0] = seed0
        n_verify_fwd += 1

        match = (draft == ar_pred)
        if accept_conf > 0:
            pd = torch.gather(ar_probs, -1, draft.unsqueeze(-1)).squeeze(-1)
            match = match | (pd >= accept_conf)
        nz = (~match).nonzero()
        k = int(nz[0, 0]) if nz.numel() > 0 else nd
        if k < nd:
            new = torch.cat([draft[:k], ar_pred[k:k + 1]])
        elif full_draft:
            bonus = lv_raw[0, -1, :].float().argmax(-1, keepdim=True).to(draft.dtype)  # verifier's a_{bd+1}
            new = torch.cat([draft, bonus])
        else:
            new = draft
        if trace is not None:
            trace.append({"round": n_rounds, "k": k, "seed": int(ar_pred[0]),
                          "draft": draft.tolist(), "ar_pred": ar_pred.tolist(), "new_tokens": new.tolist()})
        committed = torch.cat([committed, new[None, :]], dim=1)
        n_accepted += int(new.shape[0])
        if eos_id is not None and (new == eos_id).any():
            done = True

    gen = committed[0].tolist()
    if eos_id is not None and eos_id in gen:
        gen = gen[:gen.index(eos_id)]
    txt = ""
    if gen:
        import re as _re
        txt = _re.sub(r"<think>.*?</think>\s*", "",
                      model.tokenizer.decode(torch.tensor(gen, device=device), skip_special_tokens=True), flags=_re.S)
    if return_stats:
        fwd = n_draft_fwd + n_verify_fwd
        return txt, {"rounds": n_rounds, "draft_fwd": n_draft_fwd, "verify_fwd": n_verify_fwd,
                     "commit_fwd": 0, "total_fwd": fwd, "resp_tokens": len(gen),
                     "tokens_per_fwd": round(len(gen) / max(1, fwd), 3),
                     "accept_per_round": round(n_accepted / max(1, n_rounds), 2)}
    return txt

@torch.no_grad()
def generate_speculative_natcache(model, image, prompt="Text Recognition:", max_new_tokens=512,
                                  draft_threshold=0.9, temperature=0.0, top_p=0.95,
                                  max_long_side=1344, accept_conf=0.0, draft_steps=1,
                                  return_stats=False, trace=None, draft_bd=None, full_draft=True):
    """Nemotron-style O(N) cached speculative decode.

    full_draft (default True): use EVERY mask logit and the bonus token, exactly as the
    method describes -- the window [x0 | bd masks] yields a0 + d_1..d_bd (the last mask's logit predicts
    the token after the block), the verify pass runs over [x0, a0, d_1..d_bd] (bd+2 positions) and a
    fully accepted round also commits the verifier's bonus token, so a round commits up to bd+2 tokens.
    full_draft=False reproduces the earlier behaviour (last mask logit unused, no bonus, <= bd per round).

    draft_bd (opt): override the DRAFT block window (default = model.bd_size). Verify is causal-AR so
    output stays BYTE-IDENTICAL regardless of draft_bd; only the accept COUNT changes. This is the
    K-sweep knob (force a bd32-trained model to draft a larger block). Default None => no change.

    Unlike generate_speculative_prefixcache (which caches ONLY the vision prefix and RE-FORWARDS the
    growing committed response every round -> O(N^2)), this caches the committed response too, written
    CAUSALLY (token-causal native path, exactly like Nemotron's verify/commit with use_causal_mask),
    and each round drafts+verifies+commits only the bd block against the cache -> O(N).

    WHY this is correct (not the onecache bug):
      - The committed OUTPUT of any spec decode is the AR-greedy prefix: accepted draft tokens equal
        ar_pred by the accept test, so new == ar_pred[:k+1] regardless of the draft. ar_pred comes from
        the CAUSAL verify. A causal KV-cache of the committed response makes that verify identical to
        prefixcache's causal re-forward -> BYTE-IDENTICAL output to prefixcache / AR-greedy.
      - The DRAFT attends to the (token-causal) cached committed ctx instead of a block-causal re-forward.
        That commit-mode is quality-NEUTRAL (checks/probe/commit_mode_probe.py: tc 0.4735 vs bc 0.4931),
        so accept count ~ prefixcache; only speed changes.
      - SEED carry (what onecache got wrong): block[0]'s token comes from the token-shift off the last
        committed token's logit. That position is in the CACHE (not re-forwarded), so we carry its raw
        logit (seed_logit) from the commit forward to the next round's draft+verify. Round 0 uses the
        prefill seed (last prefix token's logit)."""
    from transformers import DynamicCache
    from torch.nn.attention.flex_attention import create_block_mask

    model.eval()
    device = next(model.parameters()).device
    tm = model.text_model; bd = int(draft_bd) if draft_bd else model.bd_size
    mask_id = model.mask_id; eos_id = model.tokenizer.eos_token_id

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_long_side:
        s = max_long_side / max(w, h); image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = model.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = model.processor(text=[text], images=[image], return_tensors="pt").to(device)
    input_ids = enc["input_ids"]; mm_tt = enc["mm_token_type_ids"]
    pixel_values = enc.get("pixel_values"); image_grid_thw = enc.get("image_grid_thw")
    P = input_ids.shape[1]
    image_embeds = None
    if pixel_values is not None:
        feats = model.mm_model.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
        image_embeds = torch.cat(feats, dim=0).to(device)

    def embed(ids):
        e = tm.embed_tokens(ids)
        if image_embeds is not None:
            mm = (ids == model.image_token_id).unsqueeze(-1).expand_as(e)
            e = e.masked_scatter(mm, image_embeds.to(e.dtype))
        return e

    attn = torch.ones((1, P), dtype=torch.long, device=device)
    pos_prefix, _ = model.mm_model.get_rope_index(input_ids, mm_tt, image_grid_thw=image_grid_thw,
                                                  video_grid_thw=None, attention_mask=attn)
    next_pos = int(pos_prefix.max().item()) + 1

    # prefill the FULL prefix CAUSALLY; seed0 = its last-token logit (predicts response[0]) — computed
    # exactly as uncached/prefixcache so the CRITICAL first generated token is faithful (a bf16-different
    # round-0 seed can flip a borderline first token -> whole-page divergence). Rounds 1+ use the pending
    # fold (1 carried token + bd block) for a flat 2 forwards/round.
    cache = DynamicCache()
    causal_pre = create_block_mask(lambda b, h, q, kv: kv <= q, B=1, H=None, Q_LEN=P, KV_LEN=P, device=device)
    hpre = _run_layers(model, embed(input_ids), pos_prefix, causal_pre, cache)
    seed_logit = model.glm.lm_head(tm.norm(hpre[:, -1:, :]))            # [1,1,V] predicts response[0]
    clen = P                                                            # current cache length

    def comb_pos(r0, n):   # MRoPE positions for response indices r0 .. r0+n-1
        return torch.arange(next_pos + r0, next_pos + r0 + n, device=device)[None, None, :].expand(3, 1, n)

    def draft_mask(cur, pre, Q):  # [pending(pre): CAUSAL | bd block: BIDIR], all see `cur` cached
        def mm(b, h, q, kv): return (kv < cur) | (q >= pre) | ((kv - cur) <= q)
        return create_block_mask(mm, B=1, H=None, Q_LEN=Q, KV_LEN=cur + Q, device=device)

    def mask_causal(cur, Q):  # Q queries see ALL `cur` cached + own block CAUSALLY (verify)
        def mm(b, h, q, kv): return (kv < cur) | ((kv - cur) <= q)
        return create_block_mask(mm, B=1, H=None, Q_LEN=Q, KV_LEN=cur + Q, device=device)

    committed = torch.empty((1, 0), dtype=input_ids.dtype, device=device)
    pending = None   # last committed token NOT yet cached; round 0 uses seed0, rounds 1+ prepend pending
    n_draft_fwd = n_verify_fwd = n_commit_fwd = n_rounds = n_accepted = 0
    done = False

    while not done and committed.shape[1] < max_new_tokens:
        n_rounds += 1
        pre = 1 if pending is not None else 0
        r0 = clen - P                                                  # response index of first forwarded token
        block = torch.full((1, bd), mask_id, dtype=input_ids.dtype, device=device)
        Q = pre + bd
        pos3 = comb_pos(r0, Q)

        def assemble():
            return torch.cat([pending[None, :], block], dim=1) if pre else block
        def shift_block(raw):
            # global token-shift, then take the bd block slice. block[0] <- pending's logit (raw[pre-1])
            # for rounds 1+, or the exact prefill seed0 for round 0. block[j>=1] <- block raw[j-1].
            used = torch.cat([raw[:, :1, :], raw[:, :-1, :]], dim=1) if pre else \
                   torch.cat([seed_logit, raw[:, :-1, :]], dim=1)
            return used[:, pre:, :]                                    # [1, bd, V]

        # ---- DRAFT: 1 forward of [pending | bd masks] vs cached committed; bidir block fill ----
        d_step = 0
        dm = draft_mask(clen, pre, Q)
        while (block == mask_id).any():
            _truncate_cache(cache, clen)                              # drop tentative pending+block KV
            raw = model.glm.lm_head(tm.norm(_run_layers(model, embed(assemble()), pos3, dm, cache)))
            used = shift_block(raw)
            x1, probs = model._sample_top_p(used, top_p, temperature)
            conf = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)
            mpos = block == mask_id
            conf = torch.where(mpos, conf, torch.full_like(conf, -float("inf")))
            d_step += 1
            if draft_steps is not None and d_step >= draft_steps:
                unmask = mpos
            else:
                unmask = conf > draft_threshold; unmask[0, conf.argmax(dim=-1)] = True; unmask &= mpos
            block[unmask] = x1[unmask]
            n_draft_fwd += 1
        draft = block[0].clone()
        if full_draft:
            # the LAST mask's logit predicts the token right after the block (d_bd in the paper's numbering)
            x_extra, _ = model._sample_top_p(raw[:, -1:, :], top_p, temperature)
            draft = torch.cat([draft, x_extra[0]])                       # [bd+1] = a0, d_1..d_bd

        # ---- VERIFY: 1 forward of [pending | draft] CAUSALLY; writes pending+draft KV; AR argmax ----
        _truncate_cache(cache, clen)
        if full_draft:
            # window = [pending | a0, d_1..d_bd] (pre + bd + 1 positions). Logit at position i predicts the
            # token at i+1: pending's logit -> a0 (trivially equal), a0's -> a_1 (vs d_1), ..., d_{bd-1}'s ->
            # a_bd (vs d_bd), and d_bd's logit -> the bonus token a_{bd+1}.
            Qv = Q + 1
            assemble_v = lambda: torch.cat([pending[None, :], draft[None, :]], dim=1) if pre else draft[None, :]
            hv = _run_layers(model, embed(assemble_v()), comb_pos(r0, Qv), mask_causal(clen, Qv), cache)
            lv = model.glm.lm_head(tm.norm(hv))                                          # [1, Qv, V]
            pred = torch.cat([lv[:, :1, :] if pre else seed_logit, lv[:, :-1, :]], dim=1)[:, pre:, :]  # [1, bd+1, V]
            ar_probs = torch.softmax(pred[0].float(), dim=-1)                            # [bd+1, V]
            ar_pred = ar_probs.argmax(-1)
            bonus = lv[0, -1, :].float().argmax(-1, keepdim=True)                        # a_{bd+1}
            n_verify_fwd += 1
            match = (draft == ar_pred)
            if accept_conf > 0:
                pd = torch.gather(ar_probs, -1, draft.unsqueeze(-1)).squeeze(-1)
                match = match | (pd >= accept_conf)
            nz = (~match).nonzero()
            nd = draft.shape[0]                                                          # bd + 1
            k = int(nz[0, 0]) if nz.numel() > 0 else nd
            new = torch.cat([draft[:k], ar_pred[k:k + 1]]) if k < nd else torch.cat([draft, bonus.to(draft.dtype)])
        else:
            hv = _run_layers(model, embed(assemble()), pos3, mask_causal(clen, Q), cache)
            ar_probs = torch.softmax(shift_block(model.glm.lm_head(tm.norm(hv)))[0].float(), dim=-1)  # [bd,V]
            ar_pred = ar_probs.argmax(-1)
            n_verify_fwd += 1

            match = (draft == ar_pred)
            if accept_conf > 0:
                pd = torch.gather(ar_probs, -1, draft.unsqueeze(-1)).squeeze(-1)
                match = match | (pd >= accept_conf)
            nz = (~match).nonzero()
            k = int(nz[0, 0]) if nz.numel() > 0 else bd
            new = torch.cat([draft[:k], ar_pred[k:k + 1]]) if k < bd else draft   # == ar_pred[:k+1]; block[0] always ok
        m = new.shape[0]

        # ---- COMMIT: keep verify KV for pending + new[:m-1] (inputs correct); leave new[-1] as next
        #      pending (uncached): partial -> new[-1]=corr (its verify KV used draft[k], wrong, must drop);
        #      full -> new[-1]=last accepted (correct, but dropped for uniform 2-fwd/round). 0 extra forwards. ----
        _truncate_cache(cache, clen + pre + (m - 1))
        clen = clen + pre + (m - 1)
        pending = new[-1:].clone().to(input_ids.dtype)
        committed = torch.cat([committed, new[None, :]], dim=1)
        n_accepted += m
        if trace is not None:
            trace.append({"round": n_rounds, "k": k, "draft": draft.tolist(),
                          "ar_pred": ar_pred.tolist(), "new_tokens": new.tolist()})
        if eos_id is not None and (new == eos_id).any():
            done = True

    gen = committed[0].tolist()
    if eos_id is not None and eos_id in gen:
        gen = gen[:gen.index(eos_id)]
    txt = ""
    if gen:
        import re as _re
        txt = _re.sub(r"<think>.*?</think>\s*", "",
                      model.tokenizer.decode(torch.tensor(gen, device=device), skip_special_tokens=True), flags=_re.S)
    if return_stats:
        fwd = n_draft_fwd + n_verify_fwd + n_commit_fwd
        return txt, {"rounds": n_rounds, "draft_fwd": n_draft_fwd, "verify_fwd": n_verify_fwd,
                     "commit_fwd": n_commit_fwd, "total_fwd": fwd, "resp_tokens": len(gen),
                     "tokens_per_fwd": round(len(gen) / max(1, fwd), 3),
                     "accept_per_round": round(n_accepted / max(1, n_rounds), 2)}
    return txt


@torch.no_grad()
def teacher_force_argmax(model, image, prompt, gen_ids, max_long_side=1344):
    """Lay out gen_ids and run ONE causal forward; return the AR argmax at every position.

    ar_tf[i] = the token the causal (AR) model would place at position i given gen_ids[:i].
    Hence `gen_ids[i] != ar_tf[i]` is exactly the condition under which the self-spec verifier rejects

    Preprocessing (resize, chat template, M-RoPE) replicates generate_cached exactly; if it diverged,
    "the AR path disagrees" would be contaminated by kernel/preprocessing differences.
    """
    from transformers import DynamicCache

    model.eval()
    device = next(model.parameters()).device
    tm = model.text_model

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_long_side:
        s = max_long_side / max(w, h)
        image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = model.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = model.processor(text=[text], images=[image], return_tensors="pt").to(device)
    input_ids, mm_tt = enc["input_ids"], enc["mm_token_type_ids"]
    pixel_values, image_grid_thw = enc.get("pixel_values"), enc.get("image_grid_thw")
    P = input_ids.shape[1]

    image_embeds = None
    if pixel_values is not None:
        feats = model.mm_model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True).pooler_output
        image_embeds = torch.cat(feats, dim=0).to(device)

    def embed(ids):
        e = tm.embed_tokens(ids)
        if image_embeds is not None:
            mm = (ids == model.image_token_id).unsqueeze(-1).expand_as(e)
            e = e.masked_scatter(mm, image_embeds.to(e.dtype))
        return e

    attn_prefix = torch.ones((1, P), dtype=torch.long, device=device)
    pos_prefix, _ = model.mm_model.get_rope_index(
        input_ids, mm_tt, image_grid_thw=image_grid_thw, video_grid_thw=None,
        attention_mask=attn_prefix)
    next_pos = int(pos_prefix.max().item()) + 1

    g = torch.tensor([list(gen_ids)], dtype=torch.long, device=device)
    L = g.shape[1]
    pos_gen = (torch.arange(L, device=device) + next_pos).view(1, 1, L).expand(3, 1, L)
    pos = torch.cat([pos_prefix, pos_gen], dim=-1)
    ids = torch.cat([input_ids, g], dim=1)
    T = P + L
    # SDPA 4D additive causal mask (flex attention fails on fp32 / long sequences)
    dt = tm.embed_tokens.weight.dtype
    m4 = torch.full((T, T), torch.finfo(dt).min, device=device, dtype=dt).triu(1)[None, None]
    hidden = _run_layers(model, embed(ids), pos, m4, DynamicCache())
    logits = model.glm.lm_head(tm.norm(hidden[:, P - 1:T - 1, :]))     # logits predicting position i
    return logits.argmax(-1)[0].tolist()
