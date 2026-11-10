# Portions Copyright 2023-2024 SGLang Team, licensed under the Apache License, Version 2.0 (the notice of SGLang's
# LICENSE; sglang/srt/speculative/ngram_info.py has no per-file notice): commit_slow is adapted from
# NgramVerifyInput._fill_requests in ngram_info.py (SGLang v0.5.10).
"""Host-side commit rules of the self-spec full-draft round (SELFSPEC_FULLDRAFT=1), dependency free.

Pure Python: no torch, triton or sglang import, so the rules can be unit-tested on any CPU against the per-token loop they
replace (`check_finished` after every appended token). selfspec_fast_commit re-exports every name defined here.

A round commits T = [a0] + the accepted verify tokens + the bonus. The old per-token loop (worker
_FullDraftVerifyInput._fill_requests, the n-gram verify bookkeeping) appends one token at a time and calls check_finished(1),
whose order per position is: to_finish, length (len(output_ids) >= max_new_tokens), grammar terminated, stop token (unless
ignore_eos), vocab boundary (id > vocab_size or < 0), stop strings. For a request with no grammar, no stop strings / regexes and
no pending to_finish ("fast" requests) the first position at which any of length / stop token / vocab fires is
j = min(budget - 1, first stop-set id, first out-of-vocab id, len(T) - 1), with budget = max_new_tokens - len(output_ids) and
the stop set = stop_token_ids | eos_token_ids | tokenizer eos + additional stops (empty with ignore_eos). output_ids.extend(T[:j + 1])
followed by ONE check_finished(new_accepted_len=j + 1) sets the same finish reason / finished_len / vocab-boundary replacement as
the per-token loop. Requests with a grammar, stop strings / regexes or to_finish run the per-token loop (commit_slow).

Grammar: every committed token was drawn under the request's grammar vocab mask (a0 by the sampler, the verify rows by the
self-spec bitmask), so accept_token cannot reject it. If a backend still rejects one, the request is aborted exactly as SGLang's
AR output processor does (logged, request finished with an abort reason) instead of raising inside the scheduler process.

Provenance: commit_slow follows SGLang v0.5.10's NgramVerifyInput._fill_requests (speculative/ngram_info.py) and the grammar
handling of its scheduler output processor; both Apache-2.0, Copyright 2023-2024 SGLang Team.
"""
import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Factory for the finish reason of a request whose grammar rejected a committed token (AR parity:
# scheduler_output_processor_mixin aborts the request). selfspec_fast_commit installs sglang's FINISH_ABORT at import;
# tests may install their own.
_abort_reason_factory: Optional[Callable[[str], object]] = None


def set_abort_reason_factory(fn: Callable[[str], object]) -> None:
    global _abort_reason_factory
    _abort_reason_factory = fn


def grammar_accept_or_abort(req, token: int) -> bool:
    """req.grammar.accept_token(token); on a grammar rejection abort the request as the AR output processor does (log the error,
    finish the request with an abort reason) and return False. True when the token was accepted."""
    try:
        req.grammar.accept_token(token)
        return True
    except ValueError as e:
        logger.error(f"Grammar accept_token failed for req {getattr(req, 'rid', None)} with token {token}: {e}")
        if _abort_reason_factory is None:
            raise
        req.to_finish = _abort_reason_factory(f"grammar rejected token {token}")
        req.check_finished()
        return False


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
    if req.grammar is not None and not grammar_accept_or_abort(req, T[0]):
        return 0, True
    for t in range(1, len(T)):
        req.output_ids.append(T[t])
        req.check_finished()
        if req.finished():
            return t, False
        if req.grammar is not None and not grammar_accept_or_abort(req, T[t]):
            return t, False
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
