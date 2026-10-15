"""Block-diffusion SELF-speculative decoding worker for SGLang (GLM-OCR).

natcache (src/fast_sampling.py) per round:
  DRAFT  : ONE bidirectional forward over a FRESH K-window [seed | mask*(K-1)] -> K draft tokens
           (token-shift; block[0]=seed). committed context lives in the KV cache (causal).
  VERIFY : ONE causal forward over [committed | draft]; AR-greedy; accept longest matching prefix.
  ADVANCE: SLIDE by accept count (the spec-decode framework advances seq_lens by accept_length+1).

This is the DFlash/Nemotron "fresh-window parallel draft" pattern. The dllm path could not slide
(block_size lockstep), which caps tok/fwd; here we reuse the NGRAM verify machinery (NgramVerifyInput.
verify(): verify_tree_greedy accept + seq_lens+=accept_length+1 KV advance) which slides natively.

SELF-spec: draft == target model (no separate draft model). The DRAFT is a fresh-K-window bidir
forward of the target; because the window is fresh (committed in KV, K pure masks after the seed),
its draft is byte-identical to natcache's draft (same model, same forward, same structure).

Correctness gates: the draft tokens byte-match natcache's draft, and the output byte-matches AR.
"""
import json
import logging
import os
from typing import Optional

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.selfspec_fast_commit import SelfSpecFastCommit, verify_logits
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)
_DBG = os.environ.get("SELFSPEC_DEBUG", "0") == "1"


class _FullDraftVerifyInput(NgramVerifyInput):
    """Verify input for the full-draft round (SELFSPEC_FULLDRAFT=1, the default).

    The one difference from the legacy round: a0 (the causal-logit argmax of the seed x0) is committed
    unconditionally, independent of verification. The chain root is a0 rather than x0 and the candidates
    are d1..d_{K-1} (all K-1 mask logits), so a round commits a0 + accepted prefix + bonus = accept_length + 2 (at most K+1).

    a0 is already committed and not part of the verify chain, so `_fill_requests` must append a0 and check
    finish BEFORE appending the verified tokens to output_ids. If a0 is EOS (or hits the length limit)
    the request ends at a0 and every verify token is dropped -- the same mechanism SGLang uses for an EOS
    in the middle of the candidates (clearing the trailing accepted_indices to -1), applied to the whole row.

    KV accounting: x0's K/V is what the draft forward wrote causally into slot(p0); it is kept as is (no recompute).
    That slot is not in verify's out_cache_loc, so `_free_cache` leaves it alone, and the worker adds 1 to
    seq_lens before verify. kv_committed_len is bumped by 1 here to match (the radix cache frees this many
    slots when the request finishes; without it the x0 slot would leak).
    """

    a0: Optional[torch.Tensor] = None  # [bs] int64, filled by the worker

    def _fill_requests(self, batch: ScheduleBatch, logits_output):
        assert self.a0 is not None, "fulldraft verify input needs a0"
        a0_cpu = self.a0.tolist()
        has_finished = False
        # (1) commit a0 first (it precedes the verify tokens).
        for i, req in enumerate(batch.reqs):
            req.output_ids.append(a0_cpu[i])
            req.check_finished()
            if req.finished():
                has_finished = True
                self.accepted_indices[i, :] = -1  # this request ends at a0 -- drop every verify token
            elif req.grammar is not None:
                try:
                    req.grammar.accept_token(a0_cpu[i])
                except ValueError as e:
                    logger.info(f"{i=}, {req=} a0={a0_cpu[i]}")
                    raise e
        # (2) from here identical to NgramVerifyInput._fill_requests (accepted_draft_tokens is +1: a0 is a committed token too).
        accept_index_cpu = self.accepted_indices.tolist()
        predict_cpu = self.predict.tolist()
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    self.accepted_indices[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n"
                                f"{self.accepted_indices=}\n"
                                f"{self.predict=}\n"
                            )
                            raise e
            req.spec_verify_ct += 1  # exactly 1 per round (keeps tok/fwd = toks/(2*verify_ct))
            # SGLang convention: accepted_draft_tokens = tokens committed this round - 1. Committed = 1 (a0) + kept verified.
            accepted_draft_tokens = sum(1 for idx in accept_index_row if idx != -1)
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        if has_finished:
            # an a0-EOS row is all -1 -> accept_length = -1 -> verify() adds 0 to seq_lens/kv.
            self.accept_length = (self.accepted_indices != -1).sum(dim=1) - 1
        self.accepted_indices = self.accepted_indices[self.accepted_indices != -1]

        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accepted_indices
        ]
        if logits_output.hidden_states:
            logits_output.hidden_states = logits_output.hidden_states[
                self.accepted_indices
            ]
        self.verified_id = self.predict[self.accepted_indices]

    def _free_cache(self, batch: ScheduleBatch, page_size: int, accept_length_cpu):
        super()._free_cache(batch, page_size, accept_length_cpu)
        # x0's slot (draft slot 0) was committed this round -- pair with the +1 on seq_lens.
        for req in batch.reqs:
            req.kv_committed_len += 1
            req.kv_allocated_len = req.kv_committed_len


class _DraftStash:
    """SELFSPEC_DRAFT_AHEAD=1: one pre-launched draft (the next round's), owned by the worker until the next call.
    batch / reqs (a copy of the list) / rpi / seq_lens identify the batch it was built for; sl = the host lengths it was planned with;
    rpi_host = each request's req pool index; cache_loc = its K KV slots per request (req_to_token positions [sl, sl + K)); am = the
    draft argmax [bs, K] (= the verify ids)."""
    __slots__ = ("batch", "reqs", "rpi", "seq_lens", "rpi_host", "sl", "cache_loc", "am")


def _resolve_block_diffusion(server_args: ServerArgs, dllm_cfg):
    """Mask id / bd_size of the served export.

    A hard-coded 59282 fallback (GLM-OCR's <|mask|>) would, for any other export, silently turn the draft
    into garbage: verify still keeps the output == AR, so the only symptom is accept ~1. Order: SELFSPEC_MASK_ID
    env > <model_path>/block_diffusion.json > --dllm-algorithm config; nothing found -> error (no silent fallback)."""
    bd_size = None
    path = os.path.join(str(server_args.model_path), "block_diffusion.json")
    bd = None
    if os.path.isfile(path):
        with open(path) as f:
            bd = json.load(f)
        bd_size = int(bd["bd_size"]) if "bd_size" in bd else None
    env = os.environ.get("SELFSPEC_MASK_ID", "")
    if env:
        return int(env), bd_size, "env SELFSPEC_MASK_ID"
    if bd is not None and "mask_id" in bd:
        return int(bd["mask_id"]), bd_size, path
    if dllm_cfg is not None:
        return int(dllm_cfg.mask_id), bd_size, "dllm_config"
    raise ValueError(
        f"SELFSPEC_DIFFUSION: no mask id — {path} missing/without mask_id and SELFSPEC_MASK_ID unset"
    )


class SelfSpecDiffusionWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config  # framework reads this on the spec worker
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        from sglang.srt.dllm.config import DllmConfig

        dllm_cfg = DllmConfig.from_server_args(server_args)
        self.mask_id, self._bd_size, self._mask_src = _resolve_block_diffusion(server_args, dllm_cfg)
        vocab = getattr(target_worker.model_config, "vocab_size", None)
        if vocab is not None and not (0 <= self.mask_id < vocab):
            raise ValueError(f"SELFSPEC_DIFFUSION: mask_id {self.mask_id} outside vocab {vocab}")
        if self._bd_size is not None and self.draft_token_num != self._bd_size + 1:
            logger.warning(
                f"[selfspec-diffusion] --speculative-num-draft-tokens={self.draft_token_num} but the export's "
                f"bd_size={self._bd_size}: the trained round is K = bd_size + 1 = {self._bd_size + 1} "
                f"(window [seed | mask x bd_size]; e.g. K=33 for bd=32)"
            )
        self._seed = None  # DEPRECATED: seed now carried PER-REQ on req._spec_seed (batch-safe)
        # Keep the block[0] seed resident on the GPU. Otherwise it crosses the host three times per round:
        # int(nti[i]) x bs at prefill (D2H), pageable list -> torch.tensor at draft (H2D),
        # int(last[i]) x bs at the end of verify (D2H). The value only ever needs to flow on the GPU.
        # Indexed by req_pool_idx, an identifier that stays with the request as the batch composition
        # changes under continuous batching, so the batch-safety of a per-request carry is preserved.
        self._nomask = os.environ.get("SELFSPEC_NOMASK", "0") == "1"   # default OFF: worse on both correctness and speed
        self._seed_pool = None
        self._seed_pool_dev = self.device
        # SELFSPEC_FULLDRAFT=1 (default): the full-draft round -- all K-1 mask logits are candidates and a0 is
        # committed unconditionally, so a round commits up to K+1 tokens. SELFSPEC_FULLDRAFT=0 = the legacy round (up to K).
        # Design: the `_fulldraft_verify` / `_FullDraftVerifyInput` docstrings and docs/SGLANG_SELFSPEC_IMPL.md.
        self._fulldraft = os.environ.get("SELFSPEC_FULLDRAFT", "1") == "1"   # default ON (paper algorithm); =0 -> legacy round
        if self._fulldraft:
            assert self.page_size == 1, "SELFSPEC_FULLDRAFT requires --page-size 1"
        # Positions are AR-ALIGNED by default: x0 (row 0) gets
        # len(origin)+len(output)-1 == batch.seq_lens == the position AR decode gives the same token
        # (forward_batch_info: clamp_position(seq_lens_after_prepare) = seq_lens_after_prepare-1).
        # The legacy GLM positions were len(origin)+len(output) (= AR+1) — harmless-looking under GLM's M-RoPE delta,
        # but with a plain 1-D RoPE that shifts every generated token one slot vs the prompt.
        # SELFSPEC_POS_OFFSET is added on top (SELFSPEC_POS_OFFSET=1 reproduces the legacy GLM positions).
        self._pos_offset = int(os.environ.get("SELFSPEC_POS_OFFSET", "0"))
        self._pos_checked = False
        # SELFSPEC_FASTATTN=1: no custom masks. The draft / verify forwards carry a role
        # (spec_info.selfspec_role) and the flashinfer backend expresses the pattern with built-in mask modes over the
        # unchanged request layout (one request per sequence): verify = CAUSAL (bottom-right aligned == prefix ones ++
        # tril(K, K)); draft = the same plan run NON_CAUSAL (mask rows: prefix + seed + all masks), then CAUSAL, whose
        # row 0 (seed: prefix + seed only) is copied into row 0. See speculative/selfspec_fast_attn.py.
        # Default OFF = the custom-mask path, unchanged.
        self._fastattn = os.environ.get("SELFSPEC_FASTATTN", "0") == "1"
        if self._fastattn and self._nomask:
            raise ValueError("SELFSPEC_FASTATTN=1 and SELFSPEC_NOMASK=1 are mutually exclusive")
        if self._fastattn and getattr(self.model_runner.attn_backend, "ss_planner", None) is None:
            # one source of truth: the worker drops the masks only if the backend (and so the graph runner, which keys its
            # graphs on the same planner) really runs the fast path; otherwise a mask-less forward would hit a mask wrapper
            raise ValueError(
                "SELFSPEC_FASTATTN=1 but the attention backend built no fast-path planner (needs --attention-backend "
                f"flashinfer; got {type(self.model_runner.attn_backend).__name__})"
            )
        if self._fastattn and os.environ.get("SELFSPEC_TRAIN_MASKS", "0") == "1":
            # The K + 1 row draft window is supported by the fast-path planner (it plans each forward with
            # spec_info.draft_token_num), but the worker's verify reuses the draft's K + 1 KV slots for its K tokens, so the
            # scheduler dies on the first round on the OLD path too (FULLDRAFT=1: a view-shape error in _fulldraft_verify;
            # FULLDRAFT=0: a K vs K + 1 size mismatch in set_kv_buffer / replay_prepare). Refuse at startup.
            raise ValueError("SELFSPEC_TRAIN_MASKS=1 is broken in this worker on both attention paths (verify reuses the draft's "
                             "K + 1 KV slots); not supported with SELFSPEC_FASTATTN=1")
        # SELFSPEC_CHECK=1: per-round host/device length asserts (syncs; diagnostic only, see _check_lens)
        self._check = os.environ.get("SELFSPEC_CHECK", "0") == "1"
        self._n_checked = 0
        # SELFSPEC_MROPE_FIX=1 also reads the per-round SELFSPEC_* switches once, here (instead of several os.environ
        # reads per round in _draft_block / forward_batch_generation / _round_hist_update). The process environment does
        # not change after start, so the values are the same; flag off = the per-round reads, unchanged.
        self._env_once = os.environ.get("SELFSPEC_MROPE_FIX", "0") == "1"
        _e = os.environ.get
        self._e_train_masks = _e("SELFSPEC_TRAIN_MASKS", "0") == "1"
        self._e_time = _e("SELFSPEC_TIME", "0") == "1"
        self._e_logit_dump = _e("SELFSPEC_LOGIT_DUMP", "0") == "1"
        self._e_accept_stats = _e("SELFSPEC_ACCEPT_STATS", "")
        self._e_draft_dump = _e("SELFSPEC_DRAFT_DUMP", "0") == "1"
        self._e_round_hist = _e("SELFSPEC_ROUND_HIST", "0") == "1"
        self._e_mask = _e("SELFSPEC_MASK", "1") == "1"
        self._e_nosync = _e("SELFSPEC_NOSYNC", "0") == "1"
        self._e_verify_dump = _e("SELFSPEC_VERIFY_DUMP", "0") == "1"
        # NOMASK makes the whole draft non-causal, and with FULLDRAFT=1 the draft's
        # seed row is committed (a0 unconditionally, x0's K/V reused), so the committed tokens would come from a non-causal row:
        # not lossless. Refused on both attention paths.
        if self._nomask and self._fulldraft:
            raise ValueError("SELFSPEC_NOMASK=1 with SELFSPEC_FULLDRAFT=1 is not lossless (the committed seed row / a0 would be "
                             "computed non-causally); use SELFSPEC_FASTATTN=1 for a mask-free path, or SELFSPEC_FULLDRAFT=0")
        # SELFSPEC_SHADOW=1 (with SELFSPEC_FASTATTN=1; diagnostic, eager forwards only, syncs): also build the OLD custom
        # masks (exactly as the flag-off branches below) and hand them to the backend on spec_info.ss_shadow_mask, which plans the
        # old custom-mask wrapper with them and compares every attention call bitwise (speculative/selfspec_fast_attn.py).
        self._shadow = self._fastattn and os.environ.get("SELFSPEC_SHADOW", "0") == "1"
        if os.environ.get("SELFSPEC_SHADOW", "0") == "1" and not self._fastattn:
            raise ValueError("SELFSPEC_SHADOW=1 needs SELFSPEC_FASTATTN=1 (it compares the fast path against the old one)")
        # Each default off, each needs SELFSPEC_FASTATTN=1:
        #   SELFSPEC_FAST_COMMIT=1: full-draft rounds commit on the GPU and do all host bookkeeping from ONE packed readback
        #     (speculative/selfspec_fast_commit.py); replaces _FullDraftVerifyInput._fill_requests / _free_cache, verify()'s
        #     accept_length.cpu() and the tail's torch.tensor(ends) / out assembly, and the scheduler's next_token_ids.tolist().
        #   SELFSPEC_HOSTLEN=1: seq_lens_sum kept exact on the host; ScheduleBatch.filter_batch sums seq_lens_cpu (no .item()).
        #     The other host-length syncs (generate_attn_arg_prefill sizing, begin_forward max_token_per_sequence) do not apply: the
        #     fast attention path plans from seq_lens_cpu itself and never calls either. The M-RoPE one is SELFSPEC_MROPE_FIX.
        #   SELFSPEC_DEV_POS=1: positions = seq_lens[:, None] + (arange(n) + extra + pos_offset) on the device (one kernel at
        #     any bs; the host per-request list + per-request add + cat are gone); full-draft verify ids = the draft argmax [bs, K] itself.
        _e = os.environ.get
        self._fast_commit = _e("SELFSPEC_FAST_COMMIT", "0") == "1"
        self._hostlen = _e("SELFSPEC_HOSTLEN", "0") == "1"
        self._dev_pos = _e("SELFSPEC_DEV_POS", "0") == "1"
        for _n, _on in (("SELFSPEC_FAST_COMMIT", self._fast_commit), ("SELFSPEC_HOSTLEN", self._hostlen), ("SELFSPEC_DEV_POS", self._dev_pos)):
            if _on and not self._fastattn:
                raise ValueError(f"{_n}=1 needs SELFSPEC_FASTATTN=1 (built and gated on the fast attention path)")
        if self._fast_commit and not self._fulldraft:
            raise ValueError("SELFSPEC_FAST_COMMIT=1 is implemented for SELFSPEC_FULLDRAFT=1 (paper) rounds only")
        # DEV_POS full-draft verify ids come straight from the draft argmax; the debug consumers of the assembled `draft` tensor keep it
        self._dev_ids = self._dev_pos and self._fulldraft and not (self._e_accept_stats or self._e_draft_dump)
        self._pos_consts = {}
        self._fc = (SelfSpecFastCommit(self.draft_token_num, int(self.model_runner.req_to_token_pool.size), self.device, self._check)
                    if self._fast_commit else None)
        self._am = None
        # SELFSPEC_DRAFT_AHEAD=1, default off. Right
        # after round t's single readback, if the host-only pre-check says no request can finish (selfspec_fast_commit.cannot_finish),
        # round t + 1's draft (its seed, positions, KV slots, plan and graph replay: all functions of the device state the commit kernel
        # already wrote plus host lengths known from the readback) is launched BEFORE round t's host bookkeeping and the scheduler's
        # output processing, which then run under the draft's GPU time. The next worker call uses it ("stash") when the batch is the
        # same (same ScheduleBatch, same Req objects in the same order, same host lengths, KV slots still owned); otherwise the stash's
        # KV slots are freed and a fresh draft runs. The spec-v1 loop is kept; the scheduler only binds the worker (read-only view of
        # the waiting queue / chunked request) and calls ss_release_stash() before a prefill admission reads the KV pool. Needs the
        # whole fast stack (FASTATTN, FAST_COMMIT, HOSTLEN, DEV_POS with the draft argmax as the verify ids).
        self._draft_ahead = _e("SELFSPEC_DRAFT_AHEAD", "0") == "1"
        self.ss_draft_ahead = self._draft_ahead      # the scheduler's hooks look for this attribute
        if self._draft_ahead:
            missing = [n for n, on in (("SELFSPEC_FASTATTN", self._fastattn), ("SELFSPEC_FAST_COMMIT", self._fast_commit),
                                       ("SELFSPEC_HOSTLEN", self._hostlen), ("SELFSPEC_DEV_POS", self._dev_pos)) if not on]
            if missing:
                raise ValueError(f"SELFSPEC_DRAFT_AHEAD=1 needs {' '.join(n + '=1' for n in missing)} (built and gated on that stack)")
            if not self._dev_ids:
                raise ValueError("SELFSPEC_DRAFT_AHEAD=1 needs the draft argmax as the verify ids (SELFSPEC_ACCEPT_STATS / SELFSPEC_DRAFT_DUMP "
                                 "keep the host-side draft assembly and are not supported with it)")
            alloc = self.model_runner.token_to_kv_pool_allocator
            if type(alloc).__name__ != "TokenToKVPoolAllocator":
                raise ValueError(f"SELFSPEC_DRAFT_AHEAD=1 supports the plain page-size-1 allocator only (got {type(alloc).__name__})")
        self._stash = None
        self._sched = None
        self._ahead_inc = None
        self._ahead_stats = {"launched": 0, "used": 0, "discarded": 0, "released": 0, "freed_slots": 0, "skip_sched": 0,
                             "skip_flags": 0, "skip_mem": 0, "no_precheck": 0}
        self._ahead_logged = 0
        logger.info(
            f"[selfspec-diffusion] K={self.draft_token_num} mask_id={self.mask_id} (from {self._mask_src}) "
            f"bd_size={self._bd_size} (self-spec) fulldraft={int(self._fulldraft)} "
            f"pos=AR-aligned{'+%d' % self._pos_offset if self._pos_offset else ''} fastattn={int(self._fastattn)} "
            f"check={int(self._check)} shadow={int(self._shadow)} mrope_fix={int(self._env_once)} "
            f"fast_commit={int(self._fast_commit)} hostlen={int(self._hostlen)} dev_pos={int(self._dev_pos)} dev_ids={int(self._dev_ids)} "
            f"draft_ahead={int(self._draft_ahead)}"
        )

    def update_weights_from_tensor(self, recv_req):
        """Load the actor weights an RL trainer pushes every step into the engine.

        Self-spec has no separate draft model (`self.model_runner is target_worker.model_runner`), so updating
        draft and target separately as the EAGLE worker does would write the same tensors twice; delegate instead.
        The serving path never updates weights and does not call this.
        """
        return self.target_worker.update_weights_from_tensor(recv_req)

    def clear_cache_pool(self):
        self._seed = None
        if self._stash is not None:
            # flush_cache runs only when fully idle (no running batch, so no stash); the allocator has just been cleared, so a stash
            # would be dropped without a free
            logger.warning("[selfspec-draft-ahead] clear_cache_pool with a pre-launched draft: dropped (allocator already cleared)")
            self._stash = None

    def _seeds(self) -> torch.Tensor:
        """Device-resident seed pool indexed by req_pool_idx; sized from req_to_token_pool on first use."""
        if self._seed_pool is None:
            size = self.target_worker.model_runner.req_to_token_pool.size
            self._seed_pool = torch.zeros(size, dtype=torch.int64, device=self._seed_pool_dev)
        return self._seed_pool

    def _chain_retrieve(self, bs: int, n: int):
        """Three constant tensors that depend only on (bs, n): no reason to rebuild them every round (and for
        draft and verify separately), so they are cached. ngram_info only reads them as kernel inputs and never
        mutates them (no assignment / copy_ / scatter; its .to(int64) is a no-op on int64 input)."""
        key = (bs, n)
        cached = getattr(self, "_chain_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        dev = self.device
        retrive_index = (
            torch.arange(bs, device=dev, dtype=torch.int64).unsqueeze(1) * n
            + torch.arange(n, device=dev, dtype=torch.int64).unsqueeze(0)
        )
        nxt = torch.arange(1, n + 1, device=dev, dtype=torch.int64).unsqueeze(0).repeat(bs, 1)
        nxt[:, -1] = -1
        sib = torch.full((bs, n), -1, device=dev, dtype=torch.int64)
        out = (retrive_index, nxt, sib)
        self._chain_cache = (key, out)
        return out

    def _positions(self, batch: ScheduleBatch, n: int, extra: int = 0) -> torch.Tensor:
        # arange(n) is constant: build it once and add the per-request base (no new arange per round).
        # extra: the full-draft verify window starts one position after the draft window (p0+1..p0+K), so it passes +1.
        if self._dev_pos:
            return self._positions_dev(batch, n, extra)
        ar = getattr(self, "_ar_n", None)
        if ar is None or ar.numel() != n:
            ar = torch.arange(n, device=self.device, dtype=torch.int64)
            self._ar_n = ar
        off = extra + self._pos_offset - 1          # -1: x0 at its AR position (see __init__)
        bases = [len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs]
        if self._check:
            self._check_lens(batch, bases)
        if not self._pos_checked:
            # one-time invariant check: x0's AR position == KV length (batch.seq_lens before this round)
            self._pos_checked = True
            sl = batch.seq_lens_cpu.tolist() if getattr(batch, "seq_lens_cpu", None) is not None else None
            if sl is not None and extra == 0 and [b - 1 for b in bases] != [int(x) for x in sl]:
                logger.warning(f"[selfspec-diffusion] position base {[b - 1 for b in bases]} != seq_lens {sl}")
            else:
                logger.info(f"[selfspec-diffusion] position check: x0 pos {[b - 1 + self._pos_offset for b in bases]} "
                            f"seq_lens {sl}")
        cols = [ar + (b + off) for b in bases]
        return cols[0] if len(cols) == 1 else torch.cat(cols)

    def _positions_dev(self, batch: ScheduleBatch, n: int, extra: int) -> torch.Tensor:
        """SELFSPEC_DEV_POS=1: the host formula of _positions, pos[i, j] = len(origin_i) + len(output_i) - 1 + j +
        extra + pos_offset, from the device lengths: len(origin) + len(output) - 1 == seq_lens (x0's AR position == the KV length before
        this round's x0 commit; _check_lens asserts it under SELFSPEC_CHECK=1), so pos = seq_lens[:, None] + (arange(n) + extra +
        pos_offset), one broadcast add at any bs. Both callers run before the full-draft seq_lens += 1."""
        c = self._pos_consts.get((n, extra))
        if c is None:
            c = self._pos_consts[(n, extra)] = torch.arange(n, device=self.device, dtype=torch.int64) + (extra + self._pos_offset)
        bs = batch.batch_size()
        pos = (batch.seq_lens.view(bs, 1) + c).view(-1)
        if self._check or not self._pos_checked:
            bases = [len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs]
            if self._ahead_inc is not None:
                # SELFSPEC_DRAFT_AHEAD=1: a draft launched ahead of round t's bookkeeping: output_ids does not hold round t's tokens yet (_ahead_inc[i])
                bases = [b + d for b, d in zip(bases, self._ahead_inc)]
            if self._check:
                self._check_lens(batch, bases)
                off = extra + self._pos_offset - 1
                want = [b + off + j for b in bases for j in range(n)]
                got = pos.tolist()
                if got != want:
                    raise RuntimeError(f"[selfspec-check] device positions {got[:4]}.. != host formula {want[:4]}.. (bs={bs} n={n})")
            if not self._pos_checked:
                self._pos_checked = True
                sl = batch.seq_lens_cpu.tolist() if getattr(batch, "seq_lens_cpu", None) is not None else None
                if sl is not None and extra == 0 and [b - 1 for b in bases] != [int(x) for x in sl]:
                    logger.warning(f"[selfspec-diffusion] position base {[b - 1 for b in bases]} != seq_lens {sl}")
                else:
                    logger.info(f"[selfspec-diffusion] position check: x0 pos {[b - 1 + self._pos_offset for b in bases]} "
                                f"seq_lens {sl} (device positions)")
        return pos

    def _check_lens(self, batch: ScheduleBatch, bases):
        """SELFSPEC_CHECK=1, every draft / verify (called from _positions, before the full-draft seq_lens += 1): the committed
        length the host derives from each request (len(origin) + len(output) - 1 == x0's AR position == KV length) must equal
        batch.seq_lens_cpu (the fast path's host plan input) and the device batch.seq_lens (what the kv-indices kernel reads)."""
        exp = [b - 1 for b in bases]
        host = [int(x) for x in batch.seq_lens_cpu.tolist()] if batch.seq_lens_cpu is not None else None
        dev = [int(x) for x in batch.seq_lens.tolist()]
        if host != exp or dev != exp:
            raise RuntimeError(f"[selfspec-check] committed lengths: requests {exp} seq_lens_cpu {host} seq_lens {dev}")
        self._n_checked += 1
        if self._n_checked in (1, 10, 100, 1000) or self._n_checked % 10000 == 0:
            logger.info(f"[selfspec-check] {self._n_checked} forwards: request / seq_lens_cpu / seq_lens lengths agree")

    def _full_mask(self, batch: ScheduleBatch, within: torch.Tensor, extra: int = 0) -> torch.Tensor:
        """FULL_MASK form (as ngram_worker builds): per req, n query rows attending to
        (committed prefix, all ones) ++ (within-window n x n block = `within`).
        extra: added to the prefix length -- the full-draft verify has x0 already committed (seq_lens+1), so +1."""
        # bool from the start: a float32 prefix would allocate n*(seq_len-1)*4 bytes per request
        # and forward (256 KB at K=32, seq~2000) plus a .to(torch.bool) cast kernel after the cat.
        # The bit pattern is identical, so output is unaffected (pure allocation/kernel savings).
        # Reuse a resident all-ones buffer. The prefix is all ones for both draft and verify; only the
        # trailing n x n block changes per request, so no fresh ones() + cat per round (and request)
        # -> two of three allocations removed.
        n = within.shape[0]
        within = within.to(torch.bool)
        parts = []
        inc = self._ahead_inc   # SELFSPEC_SHADOW with SELFSPEC_DRAFT_AHEAD=1: a draft launched ahead of round t's bookkeeping (output_ids lags by inc)
        for i, req in enumerate(batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids) + extra + (inc[i] if inc is not None else 0)
            w = seq_len - 1 + n
            buf = self._mask_buf(n, w)
            buf[:, seq_len - 1 : w] = within        # update only the tail (the front is always 1)
            parts.append(buf[:, :w].flatten())
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def _mask_buf(self, n: int, w: int) -> torch.Tensor:
        """Resident [n, >=w] all-ones bool buffer. The tail is overwritten, so the front is restored to 1 each time."""
        buf = getattr(self, "_mbuf", None)
        if buf is None or buf.shape[0] != n or buf.shape[1] <= w:
            # cap is ALWAYS larger than w. With w == cap, buf[:, :w] is contiguous and .flatten() returns a
            # view instead of a copy; at bs>1 the next request's tail write would then corrupt the previous
            # request's mask. Slack keeps the slice non-contiguous so flatten copies.
            cap = max(w, 4096) + n
            buf = torch.ones((n, cap), device=self.device, dtype=torch.bool)
            self._mbuf = buf
            self._mbuf_dirty = 0
        # restore to 1 only the span the previous tail overwrote (not the whole buffer)
        d = getattr(self, "_mbuf_dirty", 0)
        if d:
            buf[:, d - n : d] = True
        self._mbuf_dirty = w
        return buf

    def _draft_block(self, batch: ScheduleBatch) -> torch.Tensor:
        """Draft K tokens per req. Window = [block[0]=seed (real, AR/causal) | mask*K]  (K+1 positions
        = K masks bidir, matching the model's bd=K training block). block[0]=seed is the AR-greedy
        first token (from the previous verify's bonus) placed REAL and CAUSAL (token-causal AR path,
        doesn't attend the masks) so it's directly usable; the K masks are fully bidirectional and
        produce block[1..K-1] by token-shift of their logits."""
        bs = batch.batch_size()
        K = self.draft_token_num
        # n = K (NOT K+1): draft window length must match the target_verify cuda-graph shape
        # (draft_token_num). [seed | mask*(K-1)] gives the same draft as [seed | mask*K] (verified)
        # while staying cuda-graph compatible.
        #
        # Note on the window size: the training block has bd=K=32 fully masked slots and the loss is
        #   token-shifted (slot j's hidden state predicts token j+1), so in training each mask attends to
        #   32 masks bidirectionally. With n=K the window is [seed | mask*31] and each mask sees only 31,
        #   a slightly different representation that can lower draft quality (= acceptance = speed).
        #   Output correctness is unaffected (verify guarantees it); only speed is at stake.
        # SELFSPEC_TRAIN_MASKS=1 -> n=K+1, window [seed | mask*K], aligned with training.
        #   The draft width stays K (draft[1..K-1] = logits rows 0..K-2); the extra last row is discarded.
        #   This mismatches the cuda-graph shape (draft_token_num), so it must be measured with GRAPH=0,
        #   and any A/B must run both arms with GRAPH=0 (graphs dominate speed). The simpler fix is K = bd_size + 1.
        n = K + 1 if (self._e_train_masks if self._env_once else os.environ.get("SELFSPEC_TRAIN_MASKS", "0") == "1") else K
        dev = self.device
        # PER-REQUEST seed carry (batch-safe): each req carries its own block[0] seed on the Req object,
        # so continuous batching (batch composition changes every step) keeps each seed with its request.
        # Gather in the CURRENT batch order. (A single worker-global seed tensor [bs] would mix requests
        # at bs>1.)
        assert all(getattr(req, "_spec_seed_ready", False) for req in batch.reqs), \
            "no carried seed (prefill/extend must seed the pool)"
        seed = self._seeds()[batch.req_pool_indices]      # GPU->GPU gather (no pageable H2D)

        win = torch.full((bs, n), self.mask_id, device=dev, dtype=torch.int64)
        win[:, 0] = seed
        draft_window = win.flatten()
        shadow_mask = None
        if self._fastattn:
            mask = None   # draft pattern = backend role "draft" (seed row causal, mask rows bidirectional)
            if self._shadow:   # the old draft mask, built exactly as the else-branch below
                within = torch.ones((n, n), device=dev, dtype=torch.bool)
                within[0, 1:] = 0
                shadow_mask = self._full_mask(batch, within)
        elif self._nomask:
            # Mask-free: the bidirectional block equals the kernel's built-in NON_CAUSAL. Only row 0 (the seed)
            # differs from the old mask (the seed now sees the masks after it); correctness rests on VERIFY
            # (AR greedy + exact-match accept), not on the draft, so output still equals AR. A changed draft
            # quality only moves acceptance (= speed); check tok/fwd.
            mask = None
        else:
            within = torch.ones((n, n), device=dev, dtype=torch.bool)
            within[0, 1:] = 0  # block[0]=seed causal (AR); the K masks bidir
            mask = self._full_mask(batch, within)
        ri, nt, ns = self._chain_retrieve(bs, n)
        pos = self._positions(batch, n)
        spec = NgramVerifyInput(draft_window, mask, pos, ri, nt, ns, n)
        spec.selfspec_causal = False if self._nomask else None   # DRAFT = bidirectional block
        if self._fastattn:
            spec.selfspec_role = "draft"
            if shadow_mask is not None:
                spec.ss_shadow_mask = shadow_mask
        batch.spec_algorithm = SpeculativeAlgorithm.SELFSPEC_DIFFUSION
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec
        _tt = self._e_time if self._env_once else os.environ.get("SELFSPEC_TIME", "0") == "1"
        if _tt:
            import time as _tm
            torch.cuda.synchronize(); _a = _tm.perf_counter()
        spec.prepare_for_verify(batch, self.page_size)
        # The round's KV slots are allocated once at draft and REUSED by verify.
        # Separate allocations (draft alloc -> clone -> free, verify alloc again) would make out_cache_loc
        # and req_to_token differ -> kv_indices differ -> plan() runs twice.
        # Sharing the slots makes kv_indices bit-identical so verify's plan can be skipped.
        # Correctness: the verify forward overwrites the same slots with causally computed K/V, so later
        # blocks read verify's KV and stay consistent with AR (the draft's bidirectional K/V was throwaway).
        # Slots of rejected tokens are freed by verify()'s _free_cache based on batch.out_cache_loc.
        self._round_cache_loc = batch.out_cache_loc
        draft_cache_loc = None
        if _tt:
            torch.cuda.synchronize(); _b = _tm.perf_counter()

        mwb = batch.get_model_worker_batch()
        r = self.target_worker.forward_batch_generation(mwb, is_verify=True)
        if _tt:
            torch.cuda.synchronize(); _c = _tm.perf_counter()
            if getattr(self, "_tn2", 0) < 4:
                self._tn2 = getattr(self, "_tn2", 0) + 1
                print(f"[TIME2] draft prep+alloc={(_b-_a)*1e3:.2f}ms  model_fwd={(_c-_b)*1e3:.2f}ms", flush=True)
        logits = r.logits_output.next_token_logits.view(bs, n, -1)  # logit[i,j] predicts win[i,j+1]
        # With n=K+1 (SELFSPEC_TRAIN_MASKS=1) there is an extra row K that predicts the first token of the
        #   NEXT block; it is unused this round. The draft slice (:K-1) stays valid.
        am = logits.argmax(-1)  # [bs, n]
        if self._dev_ids:
            # SELFSPEC_DEV_POS=1, full draft: the verify input [a0, d1..d_{K-1}] == am[:, :K] and n == K (the fast
            # path refuses SELFSPEC_TRAIN_MASKS), so _fulldraft_verify takes am itself; `draft` ([seed | am[:, :K-1]]) is not assembled.
            assert n == K, (n, K)
            self._am = am
            if (self._e_logit_dump if self._env_once else os.environ.get("SELFSPEC_LOGIT_DUMP", "0") == "1") and getattr(self, "_ld", 0) < 1:
                self._ld = 1
                mp = getattr(mwb, "mrope_positions", None)
                probs = torch.softmax(logits[0].float(), dim=-1)
                for j in (0, 1, 2, 3):
                    top = torch.topk(probs[j], 3)
                    print(f"[LOGIT j={j}->block[{j+1}]] top3={[ (int(t),round(float(p),4)) for t,p in zip(top.indices,top.values)]}", flush=True)
                print(f"[LOGIT] mrope[:, :6]={None if mp is None else mp[:, :6].tolist()}", flush=True)
            return None

        draft = torch.empty((bs, K), device=dev, dtype=torch.int64)
        draft[:, 0] = seed
        draft[:, 1:] = am[:, : K - 1]  # token-shift: block[j] = argmax(logit[j-1])
        if self._fulldraft:
            # The last mask logit (row K-1 -> d_{K-1}, the token right after the window), which the legacy round
            # discards; the full-draft verify candidates are d1..d_{K-1}, so it is needed here.
            self._last_dK = am[:, K - 1]

        if (self._e_logit_dump if self._env_once else os.environ.get("SELFSPEC_LOGIT_DUMP", "0") == "1") and getattr(self, "_ld", 0) < 1:
            self._ld = 1
            mp = getattr(mwb, "mrope_positions", None)
            probs = torch.softmax(logits[0].float(), dim=-1)
            for j in (0, 1, 2, 3):  # logit[j] predicts block[j+1]
                top = torch.topk(probs[j], 3)
                print(f"[LOGIT j={j}->block[{j+1}]] top3={[ (int(t),round(float(p),4)) for t,p in zip(top.indices,top.values)]}", flush=True)
            print(f"[LOGIT] mrope[:, :6]={None if mp is None else mp[:, :6].tolist()}", flush=True)

        # Do not free the draft slots: verify reuses them (see above).
        # -- acceptance-predictability statistics (SELFSPEC_ACCEPT_STATS=<jsonl path>) --------------
        # Purpose: decide whether the verify forward could be replaced by a rule on the draft's own logits.
        #   Currently a round is 2 forwards (draft + verify). If draft confidence predicted "how far the draft
        #   is right" well enough, verify could be skipped: put masks after the uncertain position and let the
        #   next draft's causal path (the token right before the mask) commit it -> 1 forward per round.
        #   OCR is relatively deterministic, so the model's own confidence may carry that signal.
        # Recorded: top-1 probability / top1-top2 margin / entropy at positions j=1..K-1, plus the round's accept_length.
        #   (draft[:, j] is the argmax of draft-forward logits row j-1, so row j-1 statistics describe position j)
        # Cost: the softmax runs only when enabled via env; the normal path is unaffected.
        _as = self._e_accept_stats if self._env_once else os.environ.get("SELFSPEC_ACCEPT_STATS", "")
        if _as:
            with torch.no_grad():
                lp = torch.log_softmax(logits.float(), dim=-1)      # [bs, n, V]
                t2 = lp.topk(2, dim=-1)
                p1 = t2.values[..., 0].exp()                        # top-1 probability
                mg = (t2.values[..., 0] - t2.values[..., 1])        # logprob margin
                ent = -(lp.exp() * lp).sum(-1)                      # entropy
            self._as_pending = {
                "p1": p1.cpu().tolist(), "margin": mg.cpu().tolist(), "ent": ent.cpu().tolist(),
                "draft": draft.cpu().tolist(),
            }
        if (self._e_draft_dump if self._env_once else os.environ.get("SELFSPEC_DRAFT_DUMP", "0") == "1") and getattr(self, "_dn", 0) < 3:
            self._dn = getattr(self, "_dn", 0) + 1
            print(f"[DRAFTDUMP r{self._dn}] seed={int(seed[0])} draft={draft[0].tolist()}", flush=True)
        return draft.flatten()

    def _round_hist_update(self, committed_per_req):
        """SELFSPEC_ROUND_HIST=1 -> print a histogram (max/mean) of tokens committed per round every 200 rounds.
        Default OFF (one dict lookup when the env is unset). Used to confirm the per-round maximum."""
        if not (self._e_round_hist if self._env_once else os.environ.get("SELFSPEC_ROUND_HIST", "0") == "1"):
            return
        h = self.__dict__.setdefault("_rh", {})
        n = self.__dict__.get("_rh_n", 0)
        for c in committed_per_req:
            h[c] = h.get(c, 0) + 1
            n += 1
        self._rh_n = n
        if n % 200 == 0:
            tot = sum(k * v for k, v in h.items())
            print(f"[roundhist] rounds={n} max={max(h)} mean={tot / n:.3f} "
                  f"hist={dict(sorted(h.items()))}", flush=True)

    def _fulldraft_verify(self, batch: ScheduleBatch, draft_tokens: torch.Tensor) -> GenerationBatchResult:
        """Verify step of the full-draft round (SELFSPEC_FULLDRAFT=1, the default).

        Legacy round: verify input [x0, a0, d1..d_{K-2}] (root = x0, K-1 candidates, last mask logit discarded)
                  -> at most K tokens committed per round.
        Here:         verify input [a0, d1..d_{K-1}] (root = a0, candidates = all K-1 mask logits)
                  -> a0 committed unconditionally + accepted prefix + bonus -> at most K+1 tokens per round.

        Both forwards keep the K-token shape (cuda-graph compatible). What changes:
          positions: draft window p0..p0+K-1 -> verify p0+1..p0+K (`_positions(extra=1)`).
          KV slots: of the K slots the draft allocated, slot 0 (x0; a causal row, so its K/V is already exact) stays committed;
                   verify uses draft slots [1:K] + ONE new slot (position p0+K, written into req_to_token directly).
          seq_lens: +1 before verify (x0 committed) -> the mask prefix / kv_indices include the x0 slot.
          accounting: a0 is spliced in front of each request's returned tokens; accept_length_per_req_cpu / num_accepted /
                   accept_lens get +1; spec_verify_ct stays 1 per round. a0-EOS is handled in `_FullDraftVerifyInput`.
        """
        bs = batch.batch_size()
        K = self.draft_token_num
        dev = self.device
        if batch.return_logprob:
            raise NotImplementedError(
                "SELFSPEC_FULLDRAFT=1 does not support return_logprob (a0's logit row lives in the "
                "draft forward; splice not implemented — serving/bench do not use logprobs)"
            )
        if draft_tokens is None:
            # SELFSPEC_DEV_POS=1: [a0, d1..d_{K-1}] == the draft argmax [bs, K] (see _draft_block); no cat
            am, self._am = self._am, None
            a0 = am[:, 0]
            verify_tokens = am.view(-1)
        else:
            d = draft_tokens.view(bs, K)
            a0 = d[:, 1]                                          # causal-logit argmax of x0 (draft[:, 1])
            verify_tokens = torch.cat([d[:, 1:], self._last_dK.view(bs, 1)], dim=1).flatten()  # [a0,d1..d_{K-1}]

        ri, nt, ns = self._chain_retrieve(bs, K)
        pos = self._positions(batch, K, extra=1)              # p0+1 .. p0+K (draft window positions +1)
        shadow_mask = None
        if self._nomask or self._fastattn:
            custom_mask = None   # fastattn: verify pattern = backend role "verify" (flashinfer CAUSAL)
            if self._shadow:   # the old verify mask, built exactly as the else-branch below
                shadow_mask = self._full_mask(batch, torch.tril(torch.ones((K, K), device=dev, dtype=torch.bool)), extra=1)
        else:
            within = torch.tril(torch.ones((K, K), device=dev, dtype=torch.bool))
            custom_mask = self._full_mask(batch, within, extra=1)  # prefix = seq_lens+1 (includes x0)

        # KV slots: draft slots [1:] + one new slot, mapped to position p0+K = seq_lens+K.
        rl = self._round_cache_loc.view(bs, K)
        new_slots = alloc_token_slots(batch.tree_cache, bs)
        verify_loc = torch.cat([rl[:, 1:], new_slots.view(bs, 1).to(rl.dtype)], dim=1).flatten()
        r2t = batch.req_to_token_pool.req_to_token
        r2t[batch.req_pool_indices.long(), (batch.seq_lens + K).long()] = new_slots.to(r2t.dtype)

        # x0 committed -> seq_lens += 1 (the mask prefix, kv_indices and _free_cache's assign start all read this value)
        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        batch.seq_lens_sum += bs

        batch.spec_algorithm = SpeculativeAlgorithm.SELFSPEC_DIFFUSION
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        spec = _FullDraftVerifyInput(verify_tokens, custom_mask, pos, ri, nt, ns, K)
        spec.a0 = a0
        spec.selfspec_causal = True if self._nomask else None
        if self._fastattn:
            spec.selfspec_role = "verify"
            if shadow_mask is not None:
                spec.ss_shadow_mask = shadow_mask
        batch.spec_info = spec
        batch.input_ids = verify_tokens
        batch.out_cache_loc = verify_loc

        mwb = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(mwb, is_verify=True)
        verify_input: _FullDraftVerifyInput = mwb.spec_info
        if self._fast_commit and not batch.return_hidden_states:
            return self._fast_commit_round(batch, verify_input, batch_result)
        logits_output, vid, _ = verify_input.verify(
            batch, batch_result.logits_output, self.page_size, None
        )
        if self._hostlen:   # SELFSPEC_HOSTLEN=1: verify() leaves seq_lens_sum stale (only the x0 += bs above)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        # verify() already appended a0 to each request's output_ids; vid is the flat concat of (accept+1) verified
        # tokens per request (0 for an a0-EOS request). Splice a0 in front of each request's returned tokens.
        alc = verify_input.accept_length_cpu                  # [bs] int32 (-1 for an a0-EOS row)
        counts = (alc + 1).tolist()                           # verified tokens per request
        vid = vid.view(-1)
        a0v = a0.to(vid.dtype)
        if bs == 1:
            out = torch.cat([a0v, vid])
        else:
            a0_idx, v_idx, off = [], [], 0
            for c in counts:
                a0_idx.append(off)
                v_idx.extend(range(off + 1, off + 1 + c))
                off += 1 + c
            out = torch.empty(off, dtype=vid.dtype, device=dev)
            out[torch.tensor(a0_idx, device=dev, dtype=torch.int64)] = a0v
            if v_idx:
                out[torch.tensor(v_idx, device=dev, dtype=torch.int64)] = vid
        # next round's seed = each request's last returned token (the bonus; a0 for an a0-EOS request, which is finished anyway)
        ends = torch.tensor(
            [sum(c + 1 for c in counts[: i + 1]) - 1 for i in range(bs)], device=dev, dtype=torch.int64
        )
        self._seeds()[batch.req_pool_indices] = out[ends].to(torch.int64)

        accept_length_per_req_cpu = counts                    # = committed - 1 (SGLang convention: the scheduler slices with +1)
        num_accepted_tokens = sum(counts)                     # the scheduler adds bs -> tokens committed this round
        if _DBG:
            self._tf = getattr(self, "_tf", 0) + 2
            self._tt = getattr(self, "_tt", 0) + sum(c + 1 for c in counts)
            print(f"[selfspec-fulldraft] committed/round={[c + 1 for c in counts]} "
                  f"tok/fwd_cum={self._tt / max(1, self._tf):.2f}", flush=True)
        self._round_hist_update([c + 1 for c in counts])
        batch.forward_mode = ForwardMode.DECODE
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            accept_lens=verify_input.accept_length + 1,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def _fast_commit_round(self, batch: ScheduleBatch, verify_input, batch_result) -> GenerationBatchResult:
        """SELFSPEC_FAST_COMMIT=1: the full-draft round's verify + bookkeeping from ONE packed readback
        (speculative/selfspec_fast_commit.py). Same tokens, finish state, KV lengths, freed slots, seeds and counters as
        verify_input.verify() + the tail of _fulldraft_verify; the scheduler gets the per-request host token lists."""
        logits_output = batch_result.logits_output
        verify_logits(verify_input, batch, logits_output, skip_custom=False)
        ahead = self._ahead_callback(batch) if self._draft_ahead and getattr(logits_output, "customized_info", None) is None else None
        e0 = self._fc.n_early
        out, toks, js = self._fc.commit(verify_input, batch, self._seeds(), ahead=ahead)
        if ahead is not None and self._fc.n_early == e0:
            self._ahead_stats["no_precheck"] += 1   # a request may finish this round: no draft ahead (the plain fast-commit sequence)
        if self._draft_ahead and (self._fc.n_rounds in (1000, 10000) or self._fc.n_rounds % 100000 == 0):
            logger.info(f"[selfspec-draft-ahead] after {self._fc.n_rounds} rounds: {self._ahead_stats}")
        # js[i] = verified tokens kept (the old counts = accept_length_per_req_cpu); committed this round = js[i] + 1 (a0 included)
        if _DBG:
            self._tf = getattr(self, "_tf", 0) + 2
            self._tt = getattr(self, "_tt", 0) + sum(c + 1 for c in js)
            print(f"[selfspec-fulldraft] committed/round={[c + 1 for c in js]} "
                  f"tok/fwd_cum={self._tt / max(1, self._tf):.2f}", flush=True)
        self._round_hist_update([c + 1 for c in js])
        batch.forward_mode = ForwardMode.DECODE
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out,
            num_accepted_tokens=sum(js),
            accept_length_per_req_cpu=js,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            next_token_ids_cpu=toks,
            finish_checked=True,
        )

    # ------------------------------------------------------------------ SELFSPEC_DRAFT_AHEAD=1
    def ss_bind_scheduler(self, scheduler):
        """called once by the scheduler (maybe_init_draft_worker) when ss_draft_ahead: read-only access to the waiting queue, the chunked
        request and the running batch, so a draft is launched ahead only when the next iteration is expected to decode the same batch."""
        sa = scheduler.server_args
        bad = [n for n, c in (("dllm_algorithm", sa.dllm_algorithm is not None), ("disaggregation", sa.disaggregation_mode != "null"),
                              ("prefill_delayer", bool(sa.enable_prefill_delayer)), ("pp_size>1", sa.pp_size > 1),
                              ("tp_size>1", sa.tp_size > 1),
                              ("ngram_embedding", bool(getattr(scheduler.tp_worker.model_config, "use_ngram_embedding", False))))
               if c]
        if bad:
            self._draft_ahead = self.ss_draft_ahead = False
            logger.warning(f"[selfspec-draft-ahead] OFF: not supported with {bad} (these read the KV pool outside the release hook "
                           f"or are untested); the round runs without the pre-launched draft")
            return
        self._sched = scheduler
        logger.info("[selfspec-draft-ahead] ON: the next round's draft is launched right after the readback when no request can "
                    "finish; bound to the scheduler (waiting queue / chunked request gate, stash release before prefill admission)")

    def ss_release_stash(self, reason: str):
        """called by the scheduler before a prefill admission reads the KV pool (_get_new_batch_prefill_raw): free the pre-launched
        draft's KV slots, so admission (and any eviction it triggers) sees exactly the memory of the non-pipelined round."""
        st, self._stash = self._stash, None
        if st is not None:
            self._ahead_stats["released"] += 1
            self._discard_stash(st, reason)

    def _ahead_callback(self, batch: ScheduleBatch):
        """the ahead(inc) callable for SelfSpecFastCommit.commit, or None when the next scheduler iteration is not expected to decode
        this same batch (then the round is the plain fast-commit round, unchanged)."""
        s = self._sched
        if s is None or batch is not s.running_batch:
            self._ahead_stats["skip_sched"] += 1
            return None
        if s.chunked_req is not None or (len(s.waiting_queue) and not s.running_batch.batch_is_full) or \
                s.grammar_manager.has_waiting_grammars():
            # the next iteration will try a prefill admission, which releases a stash first (wasted draft): do not launch one
            self._ahead_stats["skip_sched"] += 1
            return None
        if batch.return_logprob or batch.return_hidden_states or batch.has_grammar or \
                (batch.sampling_info is not None and batch.sampling_info.has_custom_logit_processor):
            self._ahead_stats["skip_flags"] += 1   # custom logit processor (n-gram blocker): a0's ban needs the committed tokens
            return None
        return lambda inc: self._launch_ahead(batch, inc)

    def _launch_ahead(self, batch: ScheduleBatch, inc) -> bool:
        """inside commit, after the readback, the host lengths (+= inc) and this round's frees: launch round t + 1's draft now."""
        bs = batch.batch_size()
        K = self.draft_token_num
        alloc = batch.token_to_kv_pool_allocator
        # memory gate: the draft's K slots per request come out of free memory without evicting the radix tree, and the next
        # update_running_batch's check_decode_mem ((K + 1) per request, the full-draft round) still passes without evicting; both then
        # decide exactly as in the non-pipelined round, whose check sees these K slots as free (A >= (2K + 1) bs => A - K bs >= (K + 1) bs)
        if alloc.available_size() < (2 * K + 1) * bs:
            self._ahead_stats["skip_mem"] += 1
            return False
        assert self._am is None and all(getattr(r, "_spec_seed_ready", False) for r in batch.reqs)
        # output_ids lags by the tokens host_commit will append: T = a0 + the inc[i] kept verify tokens (the KV length grew by 1 for x0
        # before the verify and by inc[i] in the commit); only the SELFSPEC_CHECK / SHADOW host formulas read this
        self._ahead_inc = [d + 1 for d in inc]
        try:
            self._draft_block(batch)            # seed pool / seq_lens (device) / seq_lens_cpu are round t + 1's already
        finally:
            self._ahead_inc = None
        st = _DraftStash()
        st.batch, st.reqs, st.rpi, st.seq_lens = batch, list(batch.reqs), batch.req_pool_indices, batch.seq_lens
        st.rpi_host = [r.req_pool_idx for r in batch.reqs]
        st.sl = batch.seq_lens_cpu.numpy()[:bs].copy()
        st.cache_loc, st.am = self._round_cache_loc, self._am
        self._round_cache_loc, self._am = None, None
        self._stash = st
        self._ahead_stats["launched"] += 1
        return True

    def _stash_matches(self, st, batch: ScheduleBatch) -> bool:
        if batch is not st.batch or batch.req_pool_indices is not st.rpi or batch.seq_lens is not st.seq_lens:
            return False
        reqs = batch.reqs
        if len(reqs) != len(st.reqs) or any(a is not b for a, b in zip(reqs, st.reqs)):
            return False
        if batch.seq_lens_cpu is None or not np.array_equal(batch.seq_lens_cpu.numpy()[: len(reqs)], st.sl):
            return False
        K = self.draft_token_num
        return all(not r.kv_overallocated_freed and r.kv_allocated_len == r.kv_committed_len + K for r in reqs)

    def _use_stash(self, st, batch: ScheduleBatch):
        """this round's draft already ran (launched by the previous call): hand its KV slots and argmax to _fulldraft_verify."""
        self._round_cache_loc, self._am = st.cache_loc, st.am
        self._ahead_stats["used"] += 1
        if self._check:
            bs, K = batch.batch_size(), self.draft_token_num
            exp = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in batch.reqs]
            dev = batch.seq_lens[:bs].tolist()
            if exp != st.sl.tolist() or dev != exp:
                raise RuntimeError(f"[selfspec-check] draft-ahead stash: request lengths {exp} / device seq_lens {dev} != stash {st.sl.tolist()}")
            r2t = batch.req_to_token_pool.req_to_token
            rows = torch.stack([r2t[int(p), int(s) : int(s) + K] for p, s in zip(st.rpi_host, st.sl.tolist())]).to(torch.int64)
            if not torch.equal(rows.flatten(), st.cache_loc.to(torch.int64)):
                raise RuntimeError("[selfspec-check] draft-ahead stash: req_to_token rows [seq_len, seq_len + K) != the draft's KV slots")
            if self._ahead_stats["used"] in (1, 10, 100, 1000) or self._ahead_stats["used"] % 10000 == 0:
                logger.info(f"[selfspec-check] {self._ahead_stats['used']} pre-launched drafts used (lengths, seq_lens, KV slots agree)")

    def _discard_stash(self, st, reason: str):
        """free the pre-launched draft's KV slots of every request that still owns them (a request the scheduler released, finished /
        retracted / aborted, freed them through kv_allocated_len) and give the request back kv_allocated_len == kv_committed_len. The
        draft's GPU work is wasted; its KV writes went to slots that are either freed now or overwritten by the next draft before
        anything reads them (stream order)."""
        K = self.draft_token_num
        keep = []
        for i, r in enumerate(st.reqs):
            if (r.req_pool_idx is not None and r.req_pool_idx == st.rpi_host[i] and not r.kv_overallocated_freed
                    and r.kv_allocated_len == r.kv_committed_len + K):
                keep.append(i)
                r.kv_allocated_len = r.kv_committed_len
        if keep:
            loc = st.cache_loc.view(len(st.reqs), K)
            fl = loc.flatten() if len(keep) == len(st.reqs) else loc[torch.tensor(keep, dtype=torch.int64, device=loc.device)].flatten()
            alloc = self.model_runner.token_to_kv_pool_allocator
            alloc.free(fl if alloc.is_not_in_free_group else fl.clone())
        self._ahead_stats["discarded"] += 1
        self._ahead_stats["freed_slots"] += len(keep) * K
        if self._ahead_logged < 20:
            self._ahead_logged += 1
            logger.info(f"[selfspec-draft-ahead] pre-launched draft discarded ({reason}): bs {len(st.reqs)}, freed {len(keep) * K} KV slots "
                        f"of {len(keep)} requests; stats {self._ahead_stats}")

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
            if self._stash is not None:
                # SELFSPEC_DRAFT_AHEAD=1: not expected (a prefill is formed only after the scheduler's admission hook released the stash); defensive
                self.ss_release_stash("extend forward")
            mwb = batch.get_model_worker_batch()
            r = self.target_worker.forward_batch_generation(mwb)
            # the first generated token per req is the seed for that req's first decode block[0].
            nti = r.next_token_ids.view(-1)
            # write straight into the device-resident pool (no int() x bs D2H); only a readiness flag stays on the host.
            self._seeds()[batch.req_pool_indices] = nti.to(torch.int64)
            for req in batch.reqs:
                req._spec_seed_ready = True
            return GenerationBatchResult(
                logits_output=r.logits_output,
                next_token_ids=r.next_token_ids,
                can_run_cuda_graph=r.can_run_cuda_graph,
            )

        bs = batch.batch_size()
        K = self.draft_token_num

        if self._draft_ahead:
            # SELFSPEC_DRAFT_AHEAD=1: this round's draft may already be on the GPU (launched by the previous call right
            # after its readback). Use it only for the batch it was built for; otherwise free its KV slots and draft afresh.
            st, self._stash = self._stash, None
            if st is not None:
                if self._stash_matches(st, batch):
                    self._use_stash(st, batch)
                    return self._fulldraft_verify(batch, None)
                self._discard_stash(st, "batch changed")

        _t = self._e_time if self._env_once else os.environ.get("SELFSPEC_TIME", "0") == "1"
        if _t:
            import time as _time
            torch.cuda.synchronize(); _t0 = _time.perf_counter()

        draft_tokens = self._draft_block(batch)  # [bs*K]
        if _t:
            torch.cuda.synchronize(); _t1 = _time.perf_counter()

        if self._fulldraft:
            return self._fulldraft_verify(batch, draft_tokens)

        ri, nt, ns = self._chain_retrieve(bs, K)
        pos = self._positions(batch, K)
        # VERIFY is a topk=1 LINEAR CHAIN: each token attends [committed | earlier chain] causally,
        # which == plain causal attention. Passing custom_mask=None makes the attention kernel use
        # the fast inline-causal path (no [K, prefix+K] bool-mask global-memory loads) -> a much
        # faster forward. The chain structure for accept comes from retrive_* (not the mask).
        # Caveat: custom_mask=None is WRONG under cuda graphs captured with a mask (eager is correct):
        #   the mask path is lossless vs AR, but no-mask + such a graph diverges from AR, because the
        #   captured mask buffer is reused and attention is wrong. Keep the mask on that path.
        # SELFSPEC_MASK=0 selects the experimental no-mask path:
        #  (1) same semantics — the chain is plain causal and the accept structure lives in retrive_*
        #  (2) a much faster inline-causal kernel
        #  (3) no crash — the old mask of sum_req K*(seq_len+K) grew with the context and, once it
        #      overflowed the cuda graph's fixed custom_mask buffer, killed the server with an illegal
        #      memory access in init_forward_metadata_replay_cuda_graph.
        # The no-mask path alone is wrong because the cuda graph is captured "with mask" and reads a stale
        # buffer; with SELFSPEC_NOMASK the capture also uses tree_mask=None (cuda_graph_runner) so both
        # sides agree; still experimental.
        if self._nomask or self._fastattn:
            # The VERIFY mask (all-ones prefix ++ tril(K,K)) is element-wise identical to flashinfer's
            # bottom-right aligned CAUSAL (verified), so no mask is built and only causal=True is passed.
            custom_mask = None
            if self._shadow:   # SELFSPEC_SHADOW: the old verify mask (the SELFSPEC_MASK=1 branch below)
                shadow_mask_fd0 = self._full_mask(batch, torch.tril(torch.ones((K, K), device=self.device, dtype=torch.bool)))
        elif (self._e_mask if self._env_once else os.environ.get("SELFSPEC_MASK", "1") == "1"):  # mask path (verified lossless)
            within = torch.tril(torch.ones((K, K), device=self.device, dtype=torch.bool))  # chain-causal
            custom_mask = self._full_mask(batch, within)
        else:
            custom_mask = None
        if _t:
            torch.cuda.synchronize(); _t2 = _time.perf_counter()

        batch.spec_algorithm = SpeculativeAlgorithm.SELFSPEC_DIFFUSION
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        spec = NgramVerifyInput(draft_tokens, custom_mask, pos, ri, nt, ns, K)
        spec.selfspec_causal = True if self._nomask else None   # VERIFY = causal block
        if self._fastattn:
            spec.selfspec_role = "verify"
            if self._shadow:
                spec.ss_shadow_mask = shadow_mask_fd0
        # The cuda-graph mask-buffer overrun guard lives in CudaGraphRunner.can_run; a guard here would not work:
        # (1) the ModelRunner attribute is graph_runner (not cuda_graph_runner); (2) nothing reads
        # batch.can_run_cuda_graph; (3) the buffer that actually overflows is flashinfer's packed mask buffer.
        batch.spec_info = spec
        # Lightweight reuse instead of prepare_for_verify: update input_ids only and keep the slots draft allocated.
        # (prepare_for_verify does (1) input_ids, (2) slot reallocation, (3) req_to_token update; with shared
        #  slots (2)(3) were already done at draft, and reallocating would change kv_indices and force another plan.)
        _rl = getattr(self, "_round_cache_loc", None)
        if _rl is not None and self.page_size == 1:
            batch.input_ids = spec.draft_token
            batch.out_cache_loc = _rl
        else:
            batch.spec_info.prepare_for_verify(batch, self.page_size)

        mwb = batch.get_model_worker_batch()
        # NO-SYNC vs SYNC test: is the verify forward CPU-launch-bound (GPU hidden) or GPU-bound?
        # no-sync = CPU return time; sync = +GPU tail. Equal => CPU/launch-bound (like AR, GPU idle).
        _ns = self._e_nosync if self._env_once else os.environ.get("SELFSPEC_NOSYNC", "0") == "1"
        if _ns:
            import time as _tm2; torch.cuda.synchronize(); _n0 = _tm2.perf_counter()
        batch_result = self.target_worker.forward_batch_generation(mwb, is_verify=True)
        if _ns:
            _n1 = _tm2.perf_counter(); torch.cuda.synchronize(); _n2 = _tm2.perf_counter()
            a = self.__dict__.setdefault("_ns", {"disp": 0., "full": 0., "n": 0})
            a["disp"] += _n1 - _n0; a["full"] += _n2 - _n0; a["n"] += 1
            if a["n"] % 40 == 0:
                print(f"[NOSYNC verify n={a['n']}] CPU_dispatch(no-sync)={a['disp']/a['n']*1e3:.2f}ms "
                      f"full(sync)={a['full']/a['n']*1e3:.2f}ms  GPU_tail={(a['full']-a['disp'])/a['n']*1e3:.2f}ms", flush=True)
        if _t:
            torch.cuda.synchronize(); _t3 = _time.perf_counter()
            if getattr(self, "_tn", 0) < 6:
                self._tn = getattr(self, "_tn", 0) + 1
                print(f"[TIME] draft_fwd={(_t1-_t0)*1e3:.2f}ms mask/orch={(_t2-_t1)*1e3:.2f}ms "
                      f"verify_fwd+prep={(_t3-_t2)*1e3:.2f}ms", flush=True)
        if (self._e_verify_dump if self._env_once else os.environ.get("SELFSPEC_VERIFY_DUMP", "0") == "1") and getattr(self, "_vd", 0) < 1:
            self._vd = 1
            vl = batch_result.logits_output.next_token_logits.view(bs, K, -1)
            vp = torch.softmax(vl[0].float(), dim=-1)
            for j in (0, 1, 2, 3):  # verify logit[j] predicts block[j+1]
                t = torch.topk(vp[j], 3)
                print(f"[SSVERIFY block[{j+1}]] top3={[(int(a),round(float(b),4)) for a,b in zip(t.indices,t.values)]}", flush=True)
        verify_input: NgramVerifyInput = mwb.spec_info
        logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
            batch, batch_result.logits_output, self.page_size, None
        )
        if self._hostlen:   # SELFSPEC_HOSTLEN=1: verify() leaves seq_lens_sum stale
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        # next block[0] seed = the last verified token per req (the bonus). PER-REQUEST + RAGGED-safe:
        # verified_id is a FLAT concat of each req's (accept_length[i]+1) verified tokens (accept length
        # differs per req at bs>1), so it is NOT a uniform [bs, -1] view. Index the last token of each req
        # via the cumulative offsets from accept_length.
        vid = verify_input.verified_id.view(-1)
        counts = (verify_input.accept_length.to(torch.int64) + 1)   # tokens contributed per req
        ends = torch.cumsum(counts, 0) - 1                          # last-token flat index per req
        last = vid[ends]                                            # [bs]
        # verified_id is int32; the int64 pool needs a cast (index_put requires matching dtypes).
        self._seeds()[batch.req_pool_indices] = last.to(torch.int64)  # device-resident (no int() x bs D2H)
        # reuse the tensor verify() already moved to the CPU (one D2H per round saved).
        _alc = getattr(verify_input, "accept_length_cpu", None)
        accept_length_per_req_cpu = (
            _alc.tolist() if _alc is not None else verify_input.accept_length.cpu().tolist()
        )
        if _t:
            torch.cuda.synchronize(); _t4 = _time.perf_counter()
            a = self.__dict__.setdefault("_acc", {"draft":0.,"orch":0.,"vfwd":0.,"accept":0.,"n":0})
            a["draft"] += _t1-_t0; a["orch"] += _t2-_t1; a["vfwd"] += _t3-_t2; a["accept"] += _t4-_t3; a["n"] += 1
            if a["n"] % 40 == 0:
                n = a["n"]; tot = (a["draft"]+a["orch"]+a["vfwd"]+a["accept"])/n*1e3
                print(f"[TIMEAVG n={n}] draft={a['draft']/n*1e3:.2f} orch={a['orch']/n*1e3:.2f} "
                      f"vfwd={a['vfwd']/n*1e3:.2f} accept={a['accept']/n*1e3:.2f} | round_total={tot:.2f}ms", flush=True)
        _as = self._e_accept_stats if self._env_once else os.environ.get("SELFSPEC_ACCEPT_STATS", "")
        if _as and getattr(self, "_as_pending", None) is not None:
            import json as _json
            pend = self._as_pending; self._as_pending = None
            try:
                with open(_as, "a") as _f:
                    for _i, _al in enumerate(accept_length_per_req_cpu):
                        _f.write(_json.dumps({
                            "accept_len": int(_al),
                            "K": int(K),
                            "p1": [round(x, 5) for x in pend["p1"][_i]],
                            "margin": [round(x, 5) for x in pend["margin"][_i]],
                            "ent": [round(x, 5) for x in pend["ent"][_i]],
                            "draft": pend["draft"][_i],
                        }) + "\n")
            except Exception as _e:
                print(f"[accept-stats] write failed: {_e}", flush=True)
        if _DBG:
            self._tf = getattr(self, "_tf", 0) + 2  # draft + verify forwards this round
            self._tt = getattr(self, "_tt", 0) + sum(a + 1 for a in accept_length_per_req_cpu)
            print(
                f"[selfspec] accept/round={accept_length_per_req_cpu} "
                f"tok/fwd_cum={self._tt/max(1,self._tf):.2f}",
                flush=True,
            )
        self._round_hist_update([a + 1 for a in accept_length_per_req_cpu])
        # spec-v1 logprobs (RL only): on spec v1 the output processor leaves logprobs to the worker (same
        # contract as EAGLE), so they are filled here. verify() has already sliced next_token_logits to
        # the accepted rows, which this relies on. Serving does not request return_logprob, so this
        # block does not run there.
        if batch.return_logprob:
            from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1

            add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
        batch.forward_mode = ForwardMode.DECODE
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            accept_lens=verify_input.accept_length,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )
