# Portions Copyright 2023-2024 SGLang Team, licensed under the Apache License, Version 2.0 (the notice of SGLang's
# LICENSE; the source files have no per-file notice): adapted from sglang/srt/sampling/penaltylib/,
# sglang/srt/layers/sampler.py and sglang/srt/speculative/spec_utils.py (SGLang v0.5.10).
"""Self-spec sampling semantics: the AR distribution for a0 and for every verify row (torch only, no sglang import).

Target: for any request, self-spec commits tokens from the same distribution as autoregressive decoding with the same settings.
  a0 (the token after the seed x0, committed unconditionally) is drawn by SGLang's own sampler path on the draft's causal row
  (worker); this module supplies the pieces the verify rows need so each row gets exactly the logits processing AR decode gives
  the same prefix:
    row j of a round (prefix = committed tokens + the verify ids cand[0..j]; cand[0] = a0 in the full-draft round, x0 in the
    legacy round) sees the penalizer state of all committed tokens plus cand[1..j] (penalize_rows), the grammar state after
    cand[0..j] (fulldraft_bitmask / the stock generate_token_bitmask), logit_bias, and custom logit processors whose request
    view holds the same output ids (apply_custom_rows); then greedy (argmax) or speculative sampling against the AR target
    distribution of the configured sampling backend (verify_target_probs).
  Penalizer state is cumulated per committed token (cumulate_block), in the semantics of the upstream penalizers'
  cumulate_output_tokens (spec v1 leaves cumulation to the worker: ScheduleBatch.prepare_for_decode returns before it).

AR operation order (model_runner._preprocess_logits + Sampler): penalizers in orchestrator order (each its own _apply), grammar
vocab mask, logit_bias, custom logit processors, temperature, softmax, top-k / top-p / min-p. Every function below applies the same
elementwise op as its AR counterpart, so a row equals the AR row bitwise (validated against the real penalizers in the tests).

The penalizer classes are dispatched by name (this module must not import sglang); the worker checks at start-up that the
installed SGLang has exactly these four (a version with another penalizer fails at start, not silently).

Provenance: the per-penalizer ops reproduce SGLang v0.5.10's sampling/penaltylib (the four penalizers' _cumulate_output_tokens
and _apply), the target distributions reproduce its layers/sampler.py (flashinfer and pytorch backends), and the grammar rows
reuse speculative/spec_utils.traverse_tree; all Apache-2.0, Copyright 2023-2024 SGLang Team.
"""
from typing import Callable, List, Optional

import torch

REPETITION = "BatchedRepetitionPenalizer"
FREQUENCY = "BatchedFrequencyPenalizer"
PRESENCE = "BatchedPresencePenalizer"
MIN_NEW_TOKENS = "BatchedMinNewTokensPenalizer"
KNOWN_PENALIZERS = frozenset((REPETITION, FREQUENCY, PRESENCE, MIN_NEW_TOKENS))


def is_plain(sampling_info, has_grammar: bool) -> bool:
    """True for a batch that uses none of: sampling (every request greedy), penalties (repetition / frequency / presence /
    min_new_tokens), logit_bias, custom logit processors, grammar. Host attributes only (no tensor access). Such a batch runs the
    unmodified greedy round (no sampler call, no cumulation, no bitmask, no extra kernel or sync)."""
    si = sampling_info
    if si is None:
        return not has_grammar
    orch = si.penalizer_orchestrator
    return (
        bool(si.is_all_greedy)
        and (orch is None or not orch.is_required)
        and si.logit_bias is None
        and not si.has_custom_logit_processor
        and not has_grammar
    )


# ------------------------------------------------------------------------------------------------ penalizer state
def cumulate_block(orch, tok: torch.Tensor, valid: torch.Tensor) -> None:
    """Feed several committed tokens per request to the penalizers at once: the state afterwards equals calling the upstream
    orch.cumulate_output_tokens once per valid token, in any order (bitwise).

    tok [bs, L] token ids (entries where valid is False may hold anything, they are never used as an index);
    valid [bs, L] bool. Device ops only (no host sync).
    Set-type penalizers (repetition, presence) write one constant per request; inside one scatter_ every write to the same index must
    carry the same value (duplicate-index order is unspecified), so padded slots point at the row's first valid token (same value,
    idempotent) and a row with no valid token points every slot at column 0 and writes its current value back. Frequency adds f per
    valid token (+0.0 for padded slots: exact); min_new_tokens counts valid tokens."""
    if orch is None or not orch.is_required:
        return
    tok = tok.to(torch.int64)
    valid = valid.to(torch.bool)
    any_ = valid.any(1, keepdim=True)
    first_col = valid.to(torch.int64).argmax(1, keepdim=True)
    first = torch.where(any_, tok.gather(1, first_col), torch.zeros_like(first_col))
    idx = torch.where(valid, tok, first)
    for p in orch.penalizers.values():
        if not p.is_prepared():
            continue
        name = type(p).__name__
        if name == REPETITION:
            st = p.cumulated_repetition_penalties
            cur = st.gather(1, idx)
            st.scatter_(1, idx, torch.where(any_, p.repetition_penalties.expand_as(cur), cur))
        elif name == PRESENCE:
            st = p.cumulated_presence_penalties
            cur = st.gather(1, idx)
            st.scatter_(1, idx, torch.where(any_, p.presence_penalties.expand_as(cur), cur))
        elif name == FREQUENCY:
            st = p.cumulated_frequency_penalties
            st.scatter_add_(1, idx, p.frequency_penalties.expand(idx.shape) * valid)
        elif name == MIN_NEW_TOKENS:
            p.len_output_tokens += valid.sum(1, keepdim=True).to(p.len_output_tokens.dtype)
        else:
            raise NotImplementedError(f"self-spec: unknown penalizer {name}")


_PAIRS = {}


def _in_block_pairs(K: int, device):
    """(r, i) with 1 <= i <= r < K: row r must also see cand[i]."""
    key = (K, str(device))
    v = _PAIRS.get(key)
    if v is None:
        rr, ii = torch.tril_indices(K, K, offset=0, device=device)
        keep = ii >= 1
        v = _PAIRS[key] = (rr[keep].contiguous(), ii[keep].contiguous())
    return v


def penalize_rows(orch, logits: torch.Tensor, cand: torch.Tensor, apply_scaling: Callable) -> None:
    """In place on logits [bs * K, V] (row b * K + r). Row r gets the penalties AR decode applies to the token after cand[b, r]:
    the orchestrator state (every committed token) plus cand[b, 1..r], applied by each prepared penalizer in orchestrator order
    with the same op as its _apply (repetition: apply_scaling = the upstream compiled apply_scaling_penalties; presence /
    frequency: subtraction; min_new_tokens: the stop-token mask while the row's output length < min_new_tokens).
    One [bs * K, V] temporary per active penalizer at a time."""
    if orch is None or not orch.is_required:
        return
    bs, K = cand.shape
    dev = logits.device
    rr, ii = _in_block_pairs(K, dev)
    n = rr.numel()
    b = torch.arange(bs, device=dev).repeat_interleave(n)
    r = rr.repeat(bs)
    t = cand.to(torch.int64)[b, ii.repeat(bs)]
    row = b * K + r
    for p in orch.penalizers.values():
        if not p.is_prepared():
            continue
        name = type(p).__name__
        if name == REPETITION:
            S = p.cumulated_repetition_penalties.repeat_interleave(K, 0)
            if n:
                S[row, t] = p.repetition_penalties[b, 0]
            apply_scaling(logits, S)
        elif name == PRESENCE:
            S = p.cumulated_presence_penalties.repeat_interleave(K, 0)
            if n:
                S[row, t] = p.presence_penalties[b, 0]
            logits.sub_(S)
        elif name == FREQUENCY:
            S = p.cumulated_frequency_penalties.repeat_interleave(K, 0)
            if n:
                S.index_put_((row, t), p.frequency_penalties[b, 0], accumulate=True)
            logits.sub_(S)
        elif name == MIN_NEW_TOKENS:
            lens = p.len_output_tokens.repeat_interleave(K, 0) + torch.arange(K, device=dev, dtype=p.len_output_tokens.dtype).repeat(bs).view(-1, 1)
            m = (lens < p.min_new_tokens.repeat_interleave(K, 0)).expand_as(logits)
            logits[m] += p.stop_token_penalties.repeat_interleave(K, 0)[m]
        else:
            raise NotImplementedError(f"self-spec: unknown penalizer {name}")
        S = None   # release the [bs * K, V] temporary before the next penalizer builds its own


# ------------------------------------------------------------------------------------------------ custom logit processors
class RequestView:
    """Read-only stand-in for a Req inside a custom logit processor applied to verify row j: output_ids = the request's committed
    output ids + the verify ids that row j's prefix adds. Every other attribute is the request's own."""

    __slots__ = ("_req", "output_ids")

    def __init__(self, req, extra: List[int]):
        object.__setattr__(self, "_req", req)
        object.__setattr__(self, "output_ids", list(req.output_ids) + list(extra))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_req"), name)

    def __setattr__(self, name, value):
        raise AttributeError("RequestView is read-only")


def apply_custom_rows(logits: torch.Tensor, sampling_info, cand_cpu: List[List[int]], start: int) -> None:
    """Custom logit processors on the verify rows, one call per (processor, row position j) over the requests that use it, with
    each request's __req__ replaced by a RequestView whose output_ids = output_ids + cand_cpu[i][start : j + 1] (start = 0 in the
    full-draft round, where a0 is not in output_ids yet; 1 in the legacy round, where cand[0] = x0 already is). Exact for every
    processor that is a function of (logits, origin_input_ids + output_ids, params), which covers all of SGLang's own."""
    K = len(cand_cpu[0]) if cand_cpu else 0
    dev = logits.device
    for processor, batch_mask in sampling_info.custom_logit_processor.values():
        bidx = batch_mask.nonzero(as_tuple=True)[0].tolist()
        if not bidx:
            continue
        for j in range(K):
            params = []
            for i in bidx:
                prm = sampling_info.custom_params[i]
                if isinstance(prm, dict) and "__req__" in prm:
                    prm = dict(prm)
                    prm["__req__"] = RequestView(prm["__req__"], cand_cpu[i][start : j + 1])
                params.append(prm)
            rows = torch.tensor([i * K + j for i in bidx], device=dev, dtype=torch.int64)
            logits[rows] = processor(logits[rows], params)


# ------------------------------------------------------------------------------------------------ target distribution
def _renorm(p: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    out = torch.where(keep, p, torch.zeros_like(p))
    return out / out.sum(-1, keepdim=True)


def joint_top_k_top_p(probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """The distribution of flashinfer top_k_top_p_sampling_from_probs(filter_apply_order="joint") (AR's default backend without
    min_p): token q is kept iff #{v: p_v > p_q} < k and sum{p_v : p_v > p_q} < top_p (strictly greater, so tied tokens are kept or
    dropped together), then renormalized. top_p >= 1 keeps every token."""
    s, order = probs.sort(dim=-1, descending=True)
    V = probs.shape[-1]
    pos = torch.arange(V, device=probs.device).expand_as(s)
    new_group = torch.ones_like(s, dtype=torch.bool)
    new_group[:, 1:] = s[:, 1:] < s[:, :-1]
    start = torch.where(new_group, pos, torch.zeros_like(pos)).cummax(dim=-1).values   # = count of strictly greater
    excl = torch.cumsum(s, dim=-1) - s
    gt_sum = excl.gather(1, start)
    keep_s = (start < top_ks.view(-1, 1).to(start.dtype)) & ((gt_sum < top_ps.view(-1, 1)) | (top_ps.view(-1, 1) >= 1.0))
    keep = torch.zeros_like(keep_s).scatter_(1, order, keep_s)
    return _renorm(probs, keep)


def min_p_filter(probs: torch.Tensor, min_ps: torch.Tensor) -> torch.Tensor:
    """flashinfer min_p_sampling_from_probs: keep p >= min_p * max(p) (on the already top-k / top-p renormalized probs)."""
    keep = probs >= probs.max(dim=-1, keepdim=True).values * min_ps.view(-1, 1)
    return _renorm(probs, keep)


def torch_backend_filter(probs, top_ks, top_ps, min_ps, need_min_p: bool) -> torch.Tensor:
    """The distribution of SGLang's "pytorch" sampling backend (top_k_top_p_min_p_sampling_from_probs_torch)."""
    s, order = probs.sort(dim=-1, descending=True)
    csum = torch.cumsum(s, dim=-1)
    V = probs.shape[-1]
    keep_s = torch.arange(V, device=probs.device).view(1, -1) < top_ks.view(-1, 1)
    keep_s &= ~((csum - s) > top_ps.view(-1, 1))
    if need_min_p:
        keep_s &= ~(s < (s[:, 0] * min_ps).view(-1, 1))
    keep = torch.zeros_like(keep_s).scatter_(1, order, keep_s)
    return _renorm(probs, keep)


def verify_target_probs(logits: torch.Tensor, sampling_info, K: int, backend: str,
                        top_k_renorm: Optional[Callable] = None, top_p_renorm: Optional[Callable] = None) -> torch.Tensor:
    """AR's sampling distribution for every verify row (logits [bs * K, V], fully processed; not modified: the same values as
    Sampler.forward's in-place logits.div_(temperatures) + softmax). Batch-level flags select the same branch as the AR sampler;
    per-row parameters are each request's own (repeat_interleave), so a request whose top_k / top_p / min_p are neutral keeps
    its full distribution."""
    si = sampling_info
    probs = torch.softmax(logits / si.temperatures.repeat_interleave(K, 0), dim=-1)
    if not (si.need_top_p_sampling or si.need_top_k_sampling or si.need_min_p_sampling):
        return probs
    top_ks = si.top_ks.repeat_interleave(K, 0)
    top_ps = si.top_ps.repeat_interleave(K, 0)
    if backend == "flashinfer":
        if si.need_min_p_sampling:
            probs = top_k_renorm(probs, top_ks)
            probs = top_p_renorm(probs, top_ps)
            return min_p_filter(probs, si.min_ps.repeat_interleave(K, 0))
        return joint_top_k_top_p(probs, top_ks, top_ps)
    if backend == "pytorch":
        return torch_backend_filter(probs, top_ks, top_ps, si.min_ps.repeat_interleave(K, 0), si.need_min_p_sampling)
    raise ValueError(f"self-spec sampling verify: sampling backend {backend!r} is not supported")


def acceptance_coins(shape, device) -> torch.Tensor:
    """Uniform coins in (0, 1] for the rejection test `coin <= p_target(draft)`: P(accept) = p_target(draft), and a draft token the
    target gives probability 0 (grammar-masked, filtered by top-k / top-p / min-p) is never accepted. torch.rand alone is [0, 1),
    whose exact 0 would accept a zero-probability token."""
    return 1.0 - torch.rand(shape, dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------------------------ grammar
def chain_retrieve_cpu(K: int):
    """host copies of the linear-chain retrieve tensors (next = 1..K-1, -1; sibling = -1), as the worker's _chain_retrieve."""
    nxt = torch.arange(1, K + 1, dtype=torch.int64)
    nxt[-1] = -1
    sib = torch.full((K,), -1, dtype=torch.int64)
    return nxt, sib


def fulldraft_bitmask(reqs, vt_cpu: torch.Tensor, vocab_size: int, traverse_tree: Callable):
    """CPU vocab mask for the full-draft verify rows [bs * K, ...] and the first grammar (None when no request has one).

    vt_cpu [bs, K] = the verify ids [a0, d1..d_{K-1}]. Row j must allow what the request's grammar allows after a0, d1..dj.
    a0 is not in the grammar yet (it is committed after the verify), so it is accepted first, the stock traverse_tree (which
    treats node 0 as already accepted) fills rows 0..K-1 along the chain while each draft token passes its parent row's mask
    (accept, fill, recurse, rollback), and a0 is rolled back: the grammar leaves in the state it came in. a0 was drawn under
    this grammar's mask, so it is accepted; if a backend still rejects it, the rows stay all-allowed and the commit aborts the
    request (AR parity). Rows of a finished / terminated grammar stay all-allowed, as in AR's update_regex_vocab_mask.
    Every row starts all-allowed (-1: every bit of the packed int32 mask set), whatever the backend's allocator returns:
    llguidance hands back a buffer cached on the grammar object with the previous call's rows still in it."""
    bs, K = vt_cpu.shape
    nxt, sib = chain_retrieve_cpu(K)
    mask, first = None, None
    for i, req in enumerate(reqs):
        g = req.grammar
        if g is None:
            continue
        if mask is None:
            mask = g.allocate_vocab_mask(vocab_size=vocab_size, batch_size=bs * K, device="cpu")
            mask.fill_(-1)
            first = g
        if getattr(g, "finished", False) or g.is_terminated():
            continue
        try:
            g.accept_token(int(vt_cpu[i, 0]))
        except ValueError:
            continue
        try:
            traverse_tree(nxt, sib, vt_cpu[i], g, mask[i * K : (i + 1) * K], vocab_size=vocab_size)
        finally:
            g.rollback(1)
    return mask, first
