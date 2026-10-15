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
Logits / hidden states are not compacted (FULLDRAFT=1 rejects return_logprob; a batch with return_hidden_states takes the old path).
The pinned readback buffer is reused every round: safe in the spec-v1 loop because the host has finished reading it before the next round's
copy is enqueued. This still holds with SELFSPEC_DRAFT_AHEAD=1: the rows are turned into host lists right after
the event sync, and the next round's copy is enqueued only in the next worker call (after its verify), so no ping-pong buffer is needed.

SELFSPEC_CHECK=1 (syncs; diagnostic only): per round, the accepted-index rows must be the linear chain (i*K + 0..acc, then -1), and
after the commit the device seq_lens / seed pool / free list must equal what the host derived, and kv_committed_len == seq_lens_cpu.
"""
import copy
import logging
import os
from typing import List

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.speculative.spec_utils import TREE_SPEC_KERNEL_AVAILABLE

logger = logging.getLogger(__name__)


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


def stop_set(req) -> frozenset:
    """the ids _check_token_based_finish matches (empty with ignore_eos); cached on the request (its sampling params, eos ids and
    tokenizer do not change after it is created)."""
    s = getattr(req, "_ss_stop", None)
    if s is None:
        sp = req.sampling_params
        ids = set()
        if not sp.ignore_eos:
            if sp.stop_token_ids:
                ids |= set(sp.stop_token_ids)
            if req.eos_token_ids:
                ids |= set(req.eos_token_ids)
            tk = req.tokenizer
            if tk is not None:
                if tk.eos_token_id is not None:
                    ids.add(tk.eos_token_id)
                extra = getattr(tk, "additional_stop_token_ids", None)
                if extra:
                    ids |= set(extra)
        s = frozenset(ids)
        req._ss_stop = s
    return s


def is_fast(req) -> bool:
    """requests whose finish check reduces to length / stop-token / vocab-boundary (see module docstring)."""
    sp = req.sampling_params
    return (
        req.grammar is None
        and not req.to_finish
        and not req.finished()
        and not sp.stop_strs
        and not sp.stop_regex_strs
        and req.vocab_size is not None
    )


def cut_index(T: List[int], budget: int, stop: frozenset, vocab: int) -> int:
    """index of the last token the per-token loop would append: min(budget - 1, first stop id, first out-of-vocab id, len(T) - 1)."""
    lim = len(T) - 1
    if budget - 1 < lim:
        lim = budget - 1 if budget > 0 else 0
    for t in range(lim):
        x = T[t]
        if x in stop or x > vocab or x < 0:
            return t
    return lim


def cannot_finish(req, T: List[int]) -> bool:
    """SELFSPEC_DRAFT_AHEAD=1 host-only finish pre-check: True iff host_commit(req, T) appends ALL of T
    and leaves the request unfinished. Exact for fast requests (== commit_fast's cut rule, which fires on length, stop id or vocab
    boundary: none can fire when budget > len(T), no id of T is in the stop set and every id is in [0, vocab]); always False for slow
    requests (grammar, stop strings / regexes, to_finish), whose finish depends on more than the ids."""
    if not is_fast(req):
        return False
    if req.sampling_params.max_new_tokens - len(req.output_ids) <= len(T):
        return False
    s = stop_set(req)
    if s and not s.isdisjoint(T):
        return False
    return min(T) >= 0 and max(T) <= req.vocab_size


def commit_fast(req, T: List[int]) -> int:
    """fast request: one extend + one check_finished. Returns j (T[:j + 1] appended)."""
    j = cut_index(T, req.sampling_params.max_new_tokens - len(req.output_ids), stop_set(req), req.vocab_size)
    req.output_ids.extend(T[: j + 1])
    req.check_finished(j + 1)
    return j


def commit_slow(req, T: List[int]):
    """the old per-token loop (worker _FullDraftVerifyInput._fill_requests) over the host row, no GPU access.
    Returns (j, a0_finished)."""
    req.output_ids.append(T[0])
    req.check_finished()
    if req.finished():
        return 0, True
    if req.grammar is not None:
        try:
            req.grammar.accept_token(T[0])
        except ValueError as e:
            logger.info(f"{req=} a0={T[0]}")
            raise e
    for t in range(1, len(T)):
        req.output_ids.append(T[t])
        req.check_finished()
        if req.finished():
            return t, False
        if req.grammar is not None:
            try:
                req.grammar.accept_token(T[t])
            except ValueError as e:
                logger.info(f"{req=} token={T[t]} committed={T}")
                raise e
    if req.grammar is not None and not req.finished():
        req.check_finished()   # the old path's scheduler re-check: a grammar may terminate on the last accepted token
    return len(T) - 1, False


def host_commit(reqs, rows) -> tuple:
    """host part of the commit for FULLDRAFT=1 rows (a0, acc, predict row). Mutates the requests exactly as the old per-token loop
    (tokens, finish state, grammar, spec counters). Returns (tokens per request, j per request, acc per request)."""
    toks, js, accs = [], [], []
    for req, r in zip(reqs, rows):
        a = r[1]
        T = [r[0]] + r[2 : 3 + a]
        if is_fast(req):
            j = commit_fast(req, T)
            a0_fin = j == 0 and req.finished()
        else:
            j, a0_fin = commit_slow(req, T)
        # counters, the old convention (worker _fill_requests): the untruncated verified count, 0 if a0 finished the request
        req.spec_verify_ct += 1
        adt = 0 if a0_fin else a + 1
        req.spec_accepted_tokens += adt
        req.update_spec_acceptance_histogram(adt)
        toks.append(T[: j + 1])
        js.append(j)
        accs.append(a)
    return toks, js, accs


def verify_logits(vi, batch, logits_output, vocab_mask=None, skip_custom=False):
    """the part of NgramVerifyInput.verify (ngram_info.py:446-510) before its _fill_requests, verbatim: sampling-info filter, custom
    logit processors, the relaxed penalties (+ SELFSPEC_EXACT_PENALTY), grammar vocab mask, then greedy (verify_tree_greedy) or tree
    sampling verify. Leaves vi.predict / vi.accepted_indices / vi.accept_length on the device for the commit."""
    bs = vi.retrive_index.shape[0]
    sampling_info = batch.sampling_info
    if bs != len(sampling_info):
        sampling_info = copy.deepcopy(sampling_info)
        sampling_info.filter_batch(vi.retrive_index.tolist(), vi.retrive_index)
    # skip_custom: the caller already applied its custom logit processors to every verify row
    if sampling_info.has_custom_logit_processor and not skip_custom:
        apply_custom_logit_processor(logits_output.next_token_logits, sampling_info, num_tokens_in_batch=vi.draft_token_num)
    if sampling_info.penalizer_orchestrator.is_required or sampling_info.logit_bias is not None:
        sampling_info.penalizer_orchestrator.apply(logits_output.next_token_logits, repeat=vi.draft_token_num)
        if sampling_info.logit_bias is not None:
            logits_output.next_token_logits.add_(torch.repeat_interleave(sampling_info.logit_bias, vi.draft_token_num, dim=0))
        if os.environ.get("SELFSPEC_EXACT_PENALTY", "0") == "1":
            vi._exact_block_repetition_penalty(sampling_info, logits_output, bs)
    if vocab_mask is not None:
        assert vi.grammar is not None
        vi.grammar.apply_vocab_mask(logits=logits_output.next_token_logits, vocab_mask=vocab_mask)
    is_all_greedy = sampling_info.is_all_greedy or envs.SGLANG_NGRAM_FORCE_GREEDY_VERIFY.get()
    if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
        logger.warning("Tree speculative sampling kernel unavailable (likely AMD/HIP build). Falling back to greedy verification.")
    if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
        vi._greedy_verify(batch, logits_output)
    else:
        vi._sampling_verify(batch, logits_output, sampling_info)


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
