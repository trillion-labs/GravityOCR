# Portions Copyright 2023-2024 SGLang Team, licensed under the Apache License, Version 2.0 (the notice of SGLang's
# LICENSE; sglang/srt/speculative/ngram_info.py has no per-file notice): the n-gram verify and accept bookkeeping
# this module replaces is adapted from ngram_info.py (SGLang v0.5.10).
"""SELFSPEC_FAST_COMMIT=1: one packed readback per self-spec round, full-draft (SELFSPEC_FULLDRAFT=1) rounds only.

The inherited n-gram verify bookkeeping (ngram_info.NgramVerifyInput.verify -> _fill_requests / _free_cache, and the worker's
_FullDraftVerifyInput overrides + the _fulldraft_verify tail) reads the round's result back to the host in pieces: a0.tolist(),
accepted_indices.tolist(), predict.tolist(), a bool compaction (nonzero), accept_length.cpu(), an index_put with a Python scalar (pageable
H2D) and a bool index in _free_cache, torch.tensor(ends) (H2D) for the next seed, plus the scheduler's next_token_ids.tolist(): 9 blocking
points per round at bs = 1. Here:

  GPU (one triton launch, _ss_commit_kernel, right after verify_tree_greedy):
    per request i, with acc = accept_length[i] (the untruncated accept length; the chain is linear, so the accepted verify rows are
    i*K + 0..acc and the committed tokens are a0, predict[i*K + 0..acc]):
      packed[i] = (a0, acc, predict[i*K : i*K + K] with entries past acc zeroed)           -> the round's only D2H
      seed pool[req_pool_idx] = predict[i*K + acc] (next round's block[0])                -> no host round trip
      seq_lens[i] += acc + 1                                                                (the old verify(): seq_lens += accept_length + 1)
      free list <- verify slots i*K + acc + 1 .. i*K + K - 1, compacted in row order        (the slots the old _free_cache evicts)
  ONE non_blocking D2H of packed into a persistent page-locked buffer, one event record + event synchronize.
  HOST (no GPU access): per request, the committed tokens T = [a0] + predict[i*K : i*K + acc + 1] are cut at the position where the old
  per-token loop would stop, output_ids is extended once, check_finished is called once, the KV lengths / counters are set from the host
  counts, and the rejected KV slots are freed with the device free list (or plain views of verify_loc when a request finished inside
  the round, in the old eviction order).

The host rules (stop_set, is_fast, cut_index, cannot_finish, commit_fast, commit_slow, host_commit) live in the dependency-free
selfspec_commit_rules (unit-tested on CPU against the per-token loop) and are re-exported here.
Finish semantics (== the old per-token loop, checked by driving both on real Req objects):
  the old loop appends one token at a time and calls check_finished(1), whose order per position is to_finish, length
  (len(output_ids) >= max_new_tokens), grammar, stop token (unless ignore_eos), vocab boundary (id > vocab_size or < 0), stop strings.
  For a request with no grammar, no stop strings / regexes and no pending to_finish ("fast" requests) the first position at which any
  of length / stop token / vocab fires is  j = min(budget - 1, first stop-set id, first out-of-vocab id, len(T) - 1)  with
  budget = max_new_tokens - len(output_ids) and the stop set = stop_token_ids | eos_token_ids | tokenizer eos + additional stops (empty
  with ignore_eos). output_ids.extend(T[:j + 1]) followed by ONE check_finished(new_accepted_len=j + 1) then sets the same finish
  reason / finished_len / vocab-boundary replacement as the per-token loop (length is tested before tokens, so a length tie wins
  exactly as it does per token). Never one check_finished over the untruncated T: an EOS inside the round that also crosses
  max_new_tokens would finish FINISH_LENGTH with post-EOS tokens in the output. Requests with a grammar, stop strings /
  regexes or to_finish run the old per-token loop over the host row (still no GPU access).

KV accounting (FULLDRAFT=1): the worker already committed x0 (seq_lens += 1, kv +1) before the verify. With j as above, the kept verify
slots are 0..j-1 (a0 and the accepted draft tokens before the last committed token; the last committed token has no K/V yet, it is the
next seed): seq_lens_cpu += j, kv_committed_len += j + 1, kv_allocated_len = kv_committed_len. The device seq_lens got acc + 1 in the
kernel; a request that finished inside the round (j < acc + 1) is corrected by one scalar sub_ (no H2D). The old path's re-assign of
req_to_token for the kept slots is dropped: those entries already hold exactly those slots (the draft's prepare_for_verify wrote
positions S..S+K-1, the worker wrote S+K).

Scheduler contract: GenerationBatchResult.next_token_ids_cpu (per-request host lists = the tokens appended this round) replaces the
output processor's next_token_ids.tolist(); finish_checked=True skips its redundant check_finished (the release block still runs).
next_token_ids stays a device [bs] tensor (the next seeds): it becomes batch.output_ids, which filter_batch indexes on the device.
Logits / hidden states are not compacted: a batch with return_logprob or return_hidden_states takes the worker's per-token
commit path (_fulldraft_verify), which records the log probabilities.
Verify-row logits (verify_logits): a batch that uses no sampling feature runs verify_tree_greedy on the raw logits (the greedy
round); any other batch gets AR's processing per row (process_verify_logits) and AR's greedy rule or exact speculative sampling
(sampling_verify), and its committed tokens are fed to the penalizers (cumulate_committed) right after the commit kernel.
The pinned readback buffer is reused every round: safe in the spec-v1 loop because the host has finished reading it before the next round's
copy is enqueued. This still holds with SELFSPEC_DRAFT_AHEAD=1: the rows are turned into host lists right after
the event sync, and the next round's copy is enqueued only in the next worker call (after its verify), so no ping-pong buffer is needed.

SELFSPEC_CHECK=1 (syncs; diagnostic only): per round, the accepted-index rows must be the linear chain (i*K + 0..acc, then -1), and
after the commit the device seq_lens / seed pool / free list must equal what the host derived, and kv_committed_len == seq_lens_cpu.
"""
import logging
import os

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative import selfspec_commit_rules
from sglang.srt.speculative.selfspec_commit_rules import (  # noqa: F401  (re-exported: the host commit rules live there)
    cannot_finish,
    commit_fast,
    commit_slow,
    cut_index,
    grammar_accept_or_abort,
    host_commit,
    is_fast,
    stop_set,
)
from sglang.srt.speculative.selfspec_sampling import (
    acceptance_coins,
    apply_custom_rows,
    cumulate_block,
    penalize_rows,
    verify_target_probs,
)
from sglang.srt.speculative.spec_utils import TREE_SPEC_KERNEL_AVAILABLE
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob, tree_speculative_sampling_target_only
else:  # sampling verify needs the CUDA kernels; verify_logits falls back to greedy as upstream n-gram does
    top_k_renorm_prob = top_p_renorm_prob = tree_speculative_sampling_target_only = None

logger = logging.getLogger(__name__)

# A grammar rejection of a committed token aborts that request, as SGLang's AR output processor does (never the scheduler).
selfspec_commit_rules.set_abort_reason_factory(lambda msg: FINISH_ABORT(message=msg))


def fast_commit_enabled() -> bool:
    return os.environ.get("SELFSPEC_FAST_COMMIT", "0") == "1"


@triton.jit
def _ss_commit_kernel(
    predict_ptr,      # int32 [bs*K + 1]   verify_tree_greedy predicts (row i: i*K + 0..acc valid)
    acc_ptr,          # int32 [bs]         accept length (untruncated)
    a0_ptr,           # int64              a0 per request (strided view of the draft argmax)
    a0_stride,
    rpi_ptr,          # int64 [bs]         req_pool_indices
    seeds_ptr,        # int64 [pool]       worker seed pool
    seq_lens_ptr,     # int64 [bs]         batch.seq_lens (in place += acc + 1)
    loc_ptr,          # int64 [bs*K]       verify out_cache_loc
    packed_ptr,       # int32 [bs, K + 2]  (a0, acc, predict row)
    free_ptr,         # int64 [bs*K]       rejected verify slots, compacted in row order
    out_ptr,          # int64 [bs]         next seeds (result.next_token_ids)
    K: tl.constexpr,
    KP2: tl.constexpr,
    BS_POW2: tl.constexpr,
):
    i = tl.program_id(0)
    offs = tl.arange(0, BS_POW2)
    prev_acc = tl.load(acc_ptr + offs, mask=offs < i, other=K - 1)
    prev = tl.sum((K - 1 - prev_acc).to(tl.int64), axis=0)          # rejected slots of the rows before i
    a = tl.load(acc_ptr + i)
    j = tl.arange(0, KP2)
    inrow = j < K
    row = tl.load(predict_ptr + i * K + j, mask=inrow, other=0)
    row = tl.where(j <= a, row, 0)                                 # entries past acc are uninitialised in predict
    base = packed_ptr + i * (K + 2)
    tl.store(base + 2 + j, row, mask=inrow)
    tl.store(base + 1, a)
    a0 = tl.load(a0_ptr + i * a0_stride)
    tl.store(base, a0.to(tl.int32))
    seed = tl.load(predict_ptr + i * K + a).to(tl.int64)
    r = tl.load(rpi_ptr + i).to(tl.int64)
    tl.store(seeds_ptr + r, seed)
    tl.store(out_ptr + i, seed)
    sl = tl.load(seq_lens_ptr + i)
    tl.store(seq_lens_ptr + i, sl + (a + 1).to(sl.dtype))
    rej = (j > a) & inrow
    src = tl.load(loc_ptr + i * K + j, mask=rej, other=0)
    tl.store(free_ptr + prev + (j - a - 1), src, mask=rej)


def verify_logits(vi, batch, logits_output, vocab_mask=None, plain=True, cand_cpu=None, in_block_start=0):
    """Target decision for every verify row of a self-spec round; leaves vi.predict / vi.accepted_indices / vi.accept_length on the
    device for the commit.

    plain=True (selfspec_sampling.is_plain: every request greedy, no penalties / logit_bias / custom processor / grammar): greedy
    verify (verify_tree_greedy) on the raw logits -- exactly the kernels of the greedy round, nothing added.
    Otherwise process_verify_logits gives each row the logits processing AR decode gives the same prefix, then greedy verify
    (argmax, AR's greedy rule) or sampling_verify (speculative sampling against AR's target distribution)."""
    if plain:
        vi._greedy_verify(batch, logits_output)
        return
    si = process_verify_logits(vi, batch, logits_output, vocab_mask, cand_cpu, in_block_start)
    is_all_greedy = si.is_all_greedy or envs.SGLANG_NGRAM_FORCE_GREEDY_VERIFY.get()
    if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
        logger.warning("Tree speculative sampling kernel unavailable (likely AMD/HIP build). Falling back to greedy verification.")
    if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
        vi._greedy_verify(batch, logits_output)
    else:
        sampling_verify(vi, batch, logits_output, si)


def process_verify_logits(vi, batch, logits_output, vocab_mask=None, cand_cpu=None, in_block_start=0):
    """AR decode's logits processing for every verify row, in AR order (model_runner._preprocess_logits, then the sampler's custom
    logit processors). Row j of request i predicts the token after cand[i, j] (cand = vi.draft_token.view(bs, K)); its prefix holds
    the committed tokens and cand[i, 0..j]:
      penalties: orchestrator state (every committed token; a0 already cumulated in the full-draft round) + cand[i, 1..j], each
                 penalizer in orchestrator order with its own op (selfspec_sampling.penalize_rows);
      grammar:   vocab_mask row = the grammar state after cand[i, 0..j] (built by the worker; the a0 / x0 root included);
      logit_bias, then custom logit processors with a request view whose output_ids hold the same prefix
      (cand_cpu = host copy of cand; in_block_start = 0 in the full-draft round, 1 in the legacy round, see apply_custom_rows).
    In place on logits_output.next_token_logits; returns the batch's SamplingBatchInfo."""
    bs, K = vi.retrive_index.shape[0], vi.draft_token_num
    si = batch.sampling_info
    assert len(si) == bs, (len(si), bs)
    logits = logits_output.next_token_logits
    cand = vi.draft_token.view(bs, K)
    penalize_rows(si.penalizer_orchestrator, logits, cand, apply_scaling_penalties)
    if vocab_mask is not None:
        assert vi.grammar is not None
        vi.grammar.apply_vocab_mask(logits=logits, vocab_mask=vocab_mask)
    if si.logit_bias is not None:
        logits.add_(torch.repeat_interleave(si.logit_bias, K, dim=0))
    if si.has_custom_logit_processor:
        apply_custom_rows(logits, si, cand_cpu if cand_cpu is not None else cand.tolist(), in_block_start)
    return si


def sampling_verify(vi, batch, logits_output, si):
    """Speculative sampling of a deterministic draft chain against AR's distribution (replaces NgramVerifyInput._sampling_verify,
    whose target ignores min_p and applies top-k then top-p on the renormalized top-k probabilities, unlike AR's sampler).
    Per request: draft token d_j is accepted with probability p_j(d_j) (p_j = verify_target_probs, AR's sampling distribution for
    row j); at the first rejection the token is drawn from p_j with d_j removed and renormalized; when every draft is accepted the
    bonus is drawn from the last row. That is exactly a sample of AR decoding (standard speculative sampling with a point-mass draft).
    Acceptance thresholds are 1.0 (exact; the worker refuses other values at start-up); coins are in (0, 1], so a token with target
    probability 0 is never accepted."""
    bs, K = batch.batch_size(), vi.draft_token_num
    dev = vi.device
    candidates = vi.draft_token.reshape(bs, K)
    vi.predict = torch.empty((bs * K + 1,), dtype=torch.int32, device=dev)
    vi.accepted_indices = torch.full((bs, K), -1, dtype=torch.int32, device=dev)
    vi.accept_length = torch.empty((bs,), dtype=torch.int32, device=dev)
    target_probs = verify_target_probs(
        logits_output.next_token_logits, si, K, get_global_server_args().sampling_backend,
        top_k_renorm_prob, top_p_renorm_prob,
    ).reshape(bs, K, -1)
    tree_speculative_sampling_target_only(
        predicts=vi.predict,
        accept_index=vi.accepted_indices,
        accept_token_num=vi.accept_length,
        candidates=candidates.to(torch.int64),
        retrive_index=vi.retrive_index.to(torch.int64),
        retrive_next_token=vi.retrive_next_token.to(torch.int64),
        retrive_next_sibling=vi.retrive_next_sibling.to(torch.int64),
        uniform_samples=acceptance_coins((bs, K), dev),
        uniform_samples_for_final_sampling=torch.rand((bs,), dtype=torch.float32, device=dev),
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )


def cumulate_committed(vi, batch) -> None:
    """Penalizer cumulation of the verify tokens this round commits: predict[i * K + 0..acc] (accepted draft tokens + the bonus) per
    request, from the untruncated accept length. A request that finishes inside the round gets its later tokens cumulated too; it
    leaves the batch before the next forward, so its rows are never read again. Device ops only."""
    orch = batch.sampling_info.penalizer_orchestrator
    if orch is None or not orch.is_required:
        return
    bs, K = batch.batch_size(), vi.draft_token_num
    tok = vi.predict[: bs * K].view(bs, K)
    valid = torch.arange(K, device=tok.device).view(1, K) <= vi.accept_length.view(bs, 1).to(torch.int64)
    cumulate_block(orch, tok, valid)


class SelfSpecFastCommit:
    """owned by the self-spec worker when SELFSPEC_FAST_COMMIT=1: persistent device / page-locked buffers + the readback event."""

    def __init__(self, K: int, max_bs: int, device, check: bool):
        self.K = K
        self.max_bs = max_bs
        self.device = device
        self.check = check
        self.packed_dev = torch.zeros((max_bs, K + 2), dtype=torch.int32, device=device)
        self.packed_host = torch.zeros((max_bs, K + 2), dtype=torch.int32, pin_memory=True)
        self.free_dev = torch.zeros((max_bs * K,), dtype=torch.int64, device=device)
        self.event = torch.cuda.Event()
        self.n_rounds = 0
        self.n_checked = 0
        self.n_finish_fix = 0
        self.n_early = 0      # SELFSPEC_DRAFT_AHEAD=1: rounds whose lengths / frees were done before the bookkeeping (no request could finish)
        self.n_launched = 0   # SELFSPEC_DRAFT_AHEAD=1: of those, rounds that launched the next round's draft (the worker's ahead callback)
        self._warmup()

    def _warmup(self):
        """JIT-compile the commit kernel for every batch-size bucket (BS_POW2 is a constexpr) at startup, on scratch buffers, so no
        first-time compile stalls a serving round (the fast-attention planner's kernels are compiled by the graph capture)."""
        K, n, dev = self.K, self.max_bs, self.device
        predict = torch.zeros((n * K + 1,), dtype=torch.int32, device=dev)
        acc = torch.zeros((n,), dtype=torch.int32, device=dev)
        a0 = torch.zeros((n, K), dtype=torch.int64, device=dev)[:, 0]
        rpi = torch.arange(n, dtype=torch.int64, device=dev)
        seeds = torch.zeros((n,), dtype=torch.int64, device=dev)
        sl = torch.zeros((n,), dtype=torch.int64, device=dev)
        loc = torch.zeros((n * K,), dtype=torch.int64, device=dev)
        out = torch.empty((n,), dtype=torch.int64, device=dev)
        bs, sizes = 1, []
        while True:
            sizes.append(min(bs, n))
            if bs >= n:
                break
            bs *= 2
        for b in sizes:
            _ss_commit_kernel[(b,)](
                predict, acc, a0, a0.stride(0), rpi, seeds, sl, loc, self.packed_dev, self.free_dev, out,
                K=K, KP2=triton.next_power_of_2(K), BS_POW2=triton.next_power_of_2(max(b, 2)),
            )
        torch.cuda.synchronize()
        logger.info(f"[selfspec-fastcommit] commit kernel compiled for bs buckets {sizes} (K {K}, max_bs {n})")

    def commit(self, vi, batch, seeds: torch.Tensor, ahead=None):
        """after verify_tree_greedy / tree sampling on vi (a _FullDraftVerifyInput): the whole round bookkeeping. Returns
        (next seeds device [bs], tokens per request, j per request).

        ahead (SELFSPEC_DRAFT_AHEAD=1; None = the plain fast-commit round, unchanged): a callable ahead(inc) -> bool. When the host-only
        pre-check (cannot_finish) says no request can finish this round, the host lengths and the rejected-slot frees that the
        bookkeeping below would derive (j = acc + 1 for every request) are done right after the readback, in the same order as the old
        round (free(t) before the next draft's alloc), then ahead(inc) may launch round t + 1's draft; the host bookkeeping (host_commit,
        KV lengths) runs after it, under the draft's GPU time. The kernel, the readback and host_commit are the plain round's code."""
        if ahead is not None:
            return self._commit_ahead(vi, batch, seeds, ahead)
        bs = batch.batch_size()
        K = self.K
        assert vi.draft_token_num == K and bs <= self.max_bs, (vi.draft_token_num, K, bs, self.max_bs)
        verify_loc = batch.out_cache_loc
        a0 = vi.a0
        assert batch.seq_lens.dtype == torch.int64 and verify_loc.dtype == torch.int64 and a0.dtype == torch.int64, (
            batch.seq_lens.dtype, verify_loc.dtype, a0.dtype)
        out = torch.empty((bs,), dtype=torch.int64, device=self.device)
        if self.check:
            self._check_chain(vi, bs)
        _ss_commit_kernel[(bs,)](
            vi.predict, vi.accept_length, a0, a0.stride(0), batch.req_pool_indices, seeds, batch.seq_lens, verify_loc,
            self.packed_dev, self.free_dev, out,
            K=K, KP2=triton.next_power_of_2(K), BS_POW2=triton.next_power_of_2(max(bs, 2)),
        )
        host = self.packed_host[:bs]
        host.copy_(self.packed_dev[:bs], non_blocking=True)
        self.event.record()
        self.event.synchronize()            # the round's only readback
        rows = host.numpy().tolist()
        toks, js, accs = host_commit(batch.reqs, rows)
        # KV: kept verify slots = j (x0 was committed before the verify); seq_lens / kv lengths from the host counts
        sl = batch.seq_lens_cpu.numpy()
        fix = False
        n_free = 0
        for i, req in enumerate(batch.reqs):
            j, a = js[i], accs[i]
            sl[i] += j
            req.kv_committed_len += j + 1
            req.kv_allocated_len = req.kv_committed_len
            n_free += K - 1 - a
            if j < a + 1:
                # finished inside the round: the kernel added acc + 1 to the device seq_lens
                batch.seq_lens[i : i + 1].sub_(a + 1 - j)
                fix = True
        batch.seq_lens_sum += sum(js)
        alloc = batch.token_to_kv_pool_allocator
        if bs == 1:
            alloc.free(verify_loc[js[0]:])
        elif fix:
            # a request finished inside the round: free in the old eviction order (row-major, kept_i .. K-1 per row)
            self.n_finish_fix += 1
            alloc.free(torch.cat([verify_loc[i * K + js[i] : (i + 1) * K] for i in range(bs)]))
        elif n_free:
            # free() keeps the tensor itself inside a free group (free_group_begin .. end): never a view of the reused buffer there
            fl = self.free_dev[:n_free]
            alloc.free(fl if alloc.is_not_in_free_group else fl.clone())
        self.n_rounds += 1
        if self.check:
            self._check_after(batch, bs, js, accs, rows, seeds, out, verify_loc, n_free, fix)
        if self.n_rounds in (1, 1000) or self.n_rounds % 100000 == 0:
            logger.info(f"[selfspec-fastcommit] {self.n_rounds} rounds committed from one readback each "
                        f"(finished-inside-round free fixes {self.n_finish_fix})")
        return out, toks, js

    def _commit_ahead(self, vi, batch, seeds: torch.Tensor, ahead):
        """commit() with the SELFSPEC_DRAFT_AHEAD=1 segment between the readback and the bookkeeping (see commit's docstring). Rounds
        where the pre-check fails (a request may finish) take exactly the plain commit() sequence."""
        bs = batch.batch_size()
        K = self.K
        assert vi.draft_token_num == K and bs <= self.max_bs, (vi.draft_token_num, K, bs, self.max_bs)
        verify_loc = batch.out_cache_loc     # the next draft's prepare_for_verify replaces batch.out_cache_loc; keep this round's
        a0 = vi.a0
        assert batch.seq_lens.dtype == torch.int64 and verify_loc.dtype == torch.int64 and a0.dtype == torch.int64, (
            batch.seq_lens.dtype, verify_loc.dtype, a0.dtype)
        out = torch.empty((bs,), dtype=torch.int64, device=self.device)
        if self.check:
            self._check_chain(vi, bs)
        _ss_commit_kernel[(bs,)](
            vi.predict, vi.accept_length, a0, a0.stride(0), batch.req_pool_indices, seeds, batch.seq_lens, verify_loc,
            self.packed_dev, self.free_dev, out,
            K=K, KP2=triton.next_power_of_2(K), BS_POW2=triton.next_power_of_2(max(bs, 2)),
        )
        host = self.packed_host[:bs]
        host.copy_(self.packed_dev[:bs], non_blocking=True)
        self.event.record()
        self.event.synchronize()            # the round's only readback
        rows = host.numpy().tolist()        # host lists: packed_host may be rewritten by the next round's copy, the rows may not
        alloc = batch.token_to_kv_pool_allocator
        reqs = batch.reqs
        early = all(cannot_finish(q, [r[0]] + r[2 : 3 + r[1]]) for q, r in zip(reqs, rows))
        launched = False
        n_free = 0
        if early:
            # no request finishes: every j = acc + 1, so the device seq_lens (+= acc + 1 in the kernel) is already final and the frees
            # are commit()'s no-finish frees; do them now, before the next draft allocates (old allocator order)
            inc = [r[1] + 1 for r in rows]
            sl = batch.seq_lens_cpu.numpy()
            for i in range(bs):
                sl[i] += inc[i]
            batch.seq_lens_sum += sum(inc)
            n_free = K * bs - sum(inc)          # sum over i of K - 1 - acc_i
            if bs == 1:
                alloc.free(verify_loc[inc[0]:])
            elif n_free:
                fl = self.free_dev[:n_free]
                alloc.free(fl if alloc.is_not_in_free_group else fl.clone())
            self.n_early += 1
            launched = bool(ahead(inc))
            self.n_launched += launched
        toks, js, accs = host_commit(reqs, rows)
        fix = False
        if early:
            for i, req in enumerate(reqs):
                if js[i] != accs[i] + 1 or req.finished():
                    raise RuntimeError(f"[selfspec-draft-ahead] pre-check said no request can finish, but request {i} kept {js[i]} of "
                                       f"{accs[i] + 1} (finished={req.finished()}): cannot_finish disagrees with host_commit")
                req.kv_committed_len += js[i] + 1
                # the next draft's K slots sit at positions [committed, committed + K) of the request's req_to_token row; owning them in
                # kv_allocated_len keeps the scheduler's accounting exact (busy check, release on retract / finish frees them)
                req.kv_allocated_len = req.kv_committed_len + (K if launched else 0)
        else:
            sl = batch.seq_lens_cpu.numpy()
            for i, req in enumerate(reqs):
                j, a = js[i], accs[i]
                sl[i] += j
                req.kv_committed_len += j + 1
                req.kv_allocated_len = req.kv_committed_len
                n_free += K - 1 - a
                if j < a + 1:
                    batch.seq_lens[i : i + 1].sub_(a + 1 - j)
                    fix = True
            batch.seq_lens_sum += sum(js)
            if bs == 1:
                alloc.free(verify_loc[js[0]:])
            elif fix:
                self.n_finish_fix += 1
                alloc.free(torch.cat([verify_loc[i * K + js[i] : (i + 1) * K] for i in range(bs)]))
            elif n_free:
                fl = self.free_dev[:n_free]
                alloc.free(fl if alloc.is_not_in_free_group else fl.clone())
        self.n_rounds += 1
        if self.check:
            self._check_after(batch, bs, js, accs, rows, seeds, out, verify_loc, n_free, fix, ahead_slots=K if launched else 0)
        if self.n_rounds in (1, 1000) or self.n_rounds % 100000 == 0:
            logger.info(f"[selfspec-fastcommit] {self.n_rounds} rounds committed from one readback each "
                        f"(finished-inside-round free fixes {self.n_finish_fix}; draft-ahead: no-finish rounds {self.n_early}, "
                        f"next draft launched before the bookkeeping {self.n_launched})")
        return out, toks, js

    # ------------------------------------------------------------------ SELFSPEC_CHECK=1
    def _check_chain(self, vi, bs):
        K = self.K
        ai = vi.accepted_indices.view(bs, K).cpu()
        acc = vi.accept_length.cpu()
        for i in range(bs):
            a = int(acc[i])
            want = [i * K + t for t in range(a + 1)] + [-1] * (K - 1 - a)
            got = ai[i].tolist()
            if got != want:
                raise RuntimeError(f"[selfspec-check] fast commit: accepted_indices row {i} {got} is not the linear chain (acc {a})")

    def _check_after(self, batch, bs, js, accs, rows, seeds, out, verify_loc, n_free, fix, ahead_slots=0):
        K = self.K
        dev_sl = batch.seq_lens[:bs].cpu()
        if not torch.equal(dev_sl, batch.seq_lens_cpu[:bs].to(dev_sl.dtype)):
            raise RuntimeError(f"[selfspec-check] fast commit: seq_lens {dev_sl.tolist()} != seq_lens_cpu {batch.seq_lens_cpu.tolist()}")
        if int(batch.seq_lens_cpu[:bs].sum()) != batch.seq_lens_sum:
            raise RuntimeError(f"[selfspec-check] fast commit: seq_lens_sum {batch.seq_lens_sum} != {int(batch.seq_lens_cpu.sum())}")
        for i, req in enumerate(batch.reqs):
            if req.kv_committed_len != int(batch.seq_lens_cpu[i]):
                raise RuntimeError(f"[selfspec-check] fast commit: req {i} kv_committed_len {req.kv_committed_len} != seq_lens_cpu "
                                   f"{int(batch.seq_lens_cpu[i])}")
            if req.kv_allocated_len != req.kv_committed_len + ahead_slots:
                raise RuntimeError(f"[selfspec-check] fast commit: req {i} kv_allocated_len {req.kv_allocated_len} != committed "
                                   f"{req.kv_committed_len} + next-draft slots {ahead_slots}")
        want_seed = [rows[i][2 + accs[i]] for i in range(bs)]
        got_out = out.tolist()
        got_pool = seeds[batch.req_pool_indices].tolist()
        if got_out != want_seed or got_pool != want_seed:
            raise RuntimeError(f"[selfspec-check] fast commit: seeds out {got_out} pool {got_pool} != predict[acc] {want_seed}")
        vl = verify_loc.tolist()
        want_free = [vl[i * K + t] for i in range(bs) for t in range(accs[i] + 1, K)]
        got_free = self.free_dev[:n_free].tolist()
        if got_free != want_free:
            raise RuntimeError(f"[selfspec-check] fast commit: device free list != verify slots past acc")
        self.n_checked += 1
        if self.n_checked in (1, 10, 100, 1000) or self.n_checked % 10000 == 0:
            logger.info(f"[selfspec-check] {self.n_checked} fast commits checked (chain, seq_lens, kv lengths, seeds, free list)")
