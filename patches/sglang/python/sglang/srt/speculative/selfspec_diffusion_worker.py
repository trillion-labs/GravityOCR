"""Block-diffusion SELF-speculative decoding worker for SGLang (GLM-OCR).

natcache (src/fast_sampling.py) per round:
  DRAFT  : ONE bidirectional forward over a FRESH K-window [seed | mask*(K-1)] -> K draft tokens
           (token-shift; block[0]=seed). committed context lives in the KV cache (causal).
  VERIFY : ONE causal forward over [committed | draft]; AR-greedy; accept longest matching prefix.
  ADVANCE: SLIDE by accept count (the spec-decode framework advances seq_lens by accept_length+1).

This is the DFlash/Nemotron "fresh-window parallel draft" pattern. The dllm path could not slide
(block_size lockstep) -> ~2.5 tok/fwd; here we reuse the NGRAM verify machinery (NgramVerifyInput.
verify(): verify_tree_greedy accept + seq_lens+=accept_length+1 KV advance) which slides natively.

SELF-spec: draft == target model (no separate draft model). The DRAFT is a fresh-K-window bidir
forward of the target; because the window is fresh (committed in KV, K pure masks after the seed),
its draft is byte-identical to natcache's draft (same model, same forward, same structure).

Gate 1 = draft tokens byte-match natcache; Gate 2 = output byte-matches AR; then measure tok/fwd.
bs=1 first (the OCR validation path).
"""
import logging
import os
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
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
        self.mask_id = dllm_cfg.mask_id if dllm_cfg is not None else 59282
        self._seed = None  # DEPRECATED: seed now carried PER-REQ on req._spec_seed (batch-safe)
        # Keep the block[0] seed resident on the GPU. It used to cross the host three times per round:
        # int(nti[i]) x bs at prefill (D2H), pageable list -> torch.tensor at draft (H2D),
        # int(last[i]) x bs at the end of verify (D2H). The value only ever needs to flow on the GPU.
        # Indexed by req_pool_idx, an identifier that stays with the request as the batch composition
        # changes under continuous batching, so the batch-safety of a per-request carry is preserved.
        self._nomask = os.environ.get("SELFSPEC_NOMASK", "0") == "1"   # default OFF: measured worse on both correctness and speed
        self._seed_pool = None
        self._seed_pool_dev = self.device
        # SELFSPEC_FULLDRAFT=1 (default): the full-draft round -- all K-1 mask logits are candidates and a0 is
        # committed unconditionally, so a round commits up to K+1 tokens. SELFSPEC_FULLDRAFT=0 = the legacy round (up to K).
        # Design: the `_fulldraft_verify` / `_FullDraftVerifyInput` docstrings and docs/SGLANG_SELFSPEC_IMPL.md.
        self._fulldraft = os.environ.get("SELFSPEC_FULLDRAFT", "1") == "1"   # default ON (paper algorithm); =0 -> legacy round
        if self._fulldraft:
            assert self.page_size == 1, "SELFSPEC_FULLDRAFT requires --page-size 1"
        # SELFSPEC_POS_OFFSET (experimental, default 0 = unchanged): a constant added to the 1-D positions of the draft/verify window.
        # The current path gives x0 (row 0) position len(origin)+len(output_ids) = seq_lens+1, while AR decode gives the
        # same token seq_lens (mrope = (delta-1)+seq_lens_after_prepare = delta+S vs delta+S+1 for spec),
        # i.e. every generated token sits one position later than in AR. -1 aligns with AR (needs its own A/B).
        self._pos_offset = int(os.environ.get("SELFSPEC_POS_OFFSET", "0"))
        logger.info(
            f"[selfspec-diffusion] K={self.draft_token_num} mask_id={self.mask_id} (self-spec) "
            f"fulldraft={int(self._fulldraft)} pos_offset={self._pos_offset}"
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
        ar = getattr(self, "_ar_n", None)
        if ar is None or ar.numel() != n:
            ar = torch.arange(n, device=self.device, dtype=torch.int64)
            self._ar_n = ar
        off = extra + self._pos_offset
        cols = [
            ar + (len(req.origin_input_ids) + len(req.output_ids) + off) for req in batch.reqs
        ]
        return cols[0] if len(cols) == 1 else torch.cat(cols)

    def _full_mask(self, batch: ScheduleBatch, within: torch.Tensor, extra: int = 0) -> torch.Tensor:
        """FULL_MASK form (as ngram_worker builds): per req, n query rows attending to
        (committed prefix, all ones) ++ (within-window n x n block = `within`).
        extra: added to the prefix length -- the full-draft verify has x0 already committed (seq_lens+1), so +1."""
        # bool from the start. The prefix used to be float32, allocating n*(seq_len-1)*4 bytes per request
        # and forward (256 KB at K=32, seq~2000) plus a .to(torch.bool) cast kernel after the cat.
        # The bit pattern is identical, so output is unaffected (pure allocation/kernel savings).
        # Reuse a resident all-ones buffer. The prefix is all ones for both draft and verify; only the
        # trailing n x n block changes per request. Previously every round (and request) allocated a fresh
        # ones() and a second tensor via cat -> two of three allocations removed.
        n = within.shape[0]
        within = within.to(torch.bool)
        parts = []
        for req in batch.reqs:
            seq_len = len(req.origin_input_ids) + len(req.output_ids) + extra
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
        n = K + 1 if os.environ.get("SELFSPEC_TRAIN_MASKS", "0") == "1" else K
        dev = self.device
        # PER-REQUEST seed carry (batch-safe): each req carries its own block[0] seed on the Req object,
        # so continuous batching (batch composition changes every step) keeps each seed with its request.
        # Gather in the CURRENT batch order. (Was a single worker-global self._seed [bs] tensor -> crashed
        # / mixed reqs at bs>1; see git history.)
        assert all(getattr(req, "_spec_seed_ready", False) for req in batch.reqs), \
            "no carried seed (prefill/extend must seed the pool)"
        seed = self._seeds()[batch.req_pool_indices]      # GPU->GPU gather (no pageable H2D)

        win = torch.full((bs, n), self.mask_id, device=dev, dtype=torch.int64)
        win[:, 0] = seed
        draft_window = win.flatten()
        if self._nomask:
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
        batch.spec_algorithm = SpeculativeAlgorithm.SELFSPEC_DIFFUSION
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec
        _tt = os.environ.get("SELFSPEC_TIME", "0") == "1"
        if _tt:
            import time as _tm
            torch.cuda.synchronize(); _a = _tm.perf_counter()
        spec.prepare_for_verify(batch, self.page_size)
        # The round's KV slots are allocated once at draft and REUSED by verify.
        # Previously draft allocated -> cloned -> freed and verify allocated again; then out_cache_loc
        # and req_to_token differed -> kv_indices differed -> plan() had to run twice.
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

        draft = torch.empty((bs, K), device=dev, dtype=torch.int64)
        draft[:, 0] = seed
        draft[:, 1:] = am[:, : K - 1]  # token-shift: block[j] = argmax(logit[j-1])
        if self._fulldraft:
            # The last mask logit (row K-1 -> d_{K-1}, the token right after the window), which the legacy round
            # discards; the full-draft verify candidates are d1..d_{K-1}, so it is needed here.
            self._last_dK = am[:, K - 1]

        if os.environ.get("SELFSPEC_LOGIT_DUMP", "0") == "1" and getattr(self, "_ld", 0) < 1:
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
        _as = os.environ.get("SELFSPEC_ACCEPT_STATS", "")
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
        if os.environ.get("SELFSPEC_DRAFT_DUMP", "0") == "1" and getattr(self, "_dn", 0) < 3:
            self._dn = getattr(self, "_dn", 0) + 1
            print(f"[DRAFTDUMP r{self._dn}] seed={int(seed[0])} draft={draft[0].tolist()}", flush=True)
        return draft.flatten()

    def _round_hist_update(self, committed_per_req):
        """SELFSPEC_ROUND_HIST=1 -> print a histogram (max/mean) of tokens committed per round every 200 rounds.
        Default OFF (one dict lookup when the env is unset). Used to confirm the per-round maximum."""
        if os.environ.get("SELFSPEC_ROUND_HIST", "0") != "1":
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
        d = draft_tokens.view(bs, K)
        a0 = d[:, 1]                                          # causal-logit argmax of x0 (draft[:, 1])
        verify_tokens = torch.cat([d[:, 1:], self._last_dK.view(bs, 1)], dim=1).flatten()  # [a0,d1..d_{K-1}]

        ri, nt, ns = self._chain_retrieve(bs, K)
        pos = self._positions(batch, K, extra=1)              # p0+1 .. p0+K (draft window positions +1)
        if self._nomask:
            custom_mask = None
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
        batch.spec_info = spec
        batch.input_ids = verify_tokens
        batch.out_cache_loc = verify_loc

        mwb = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(mwb, is_verify=True)
        verify_input: _FullDraftVerifyInput = mwb.spec_info
        logits_output, vid, _ = verify_input.verify(
            batch, batch_result.logits_output, self.page_size, None
        )
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

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
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

        _t = os.environ.get("SELFSPEC_TIME", "0") == "1"
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
        # the fast inline-causal path (no [K, prefix+K] bool-mask global-memory loads) -> ~2.7x
        # faster forward. The chain structure for accept comes from retrive_* (not the mask).
        # Verification result: custom_mask=None is WRONG under cuda graphs.
        #   AR vs spec(mask)   : 12/12 pages identical (the mask path is lossless)
        #   AR vs spec(no-mask): 7/12 pages differ   (only combined with cuda graphs; eager is correct)
        #   -> "a chain is plain causal so no mask is needed" does not hold on the cuda-graph path
        #      (the captured mask buffer is reused and attention is wrong). Keep the mask.
        # SELFSPEC_MASK=0 selects the experimental no-mask path:
        #  (1) same semantics — the chain is plain causal and the accept structure lives in retrive_*
        #  (2) a much faster inline-causal kernel
        #  (3) no crash — the old mask of sum_req K*(seq_len+K) grew with the context and, once it
        #      overflowed the cuda graph's fixed custom_mask buffer, killed the server with an illegal
        #      memory access in init_forward_metadata_replay_cuda_graph.
        # On its own the no-mask path was wrong (7/12 pages differed) because the cuda graph had been
        # captured "with mask" and read a stale buffer. Capture now also uses tree_mask=None (SELFSPEC_NOMASK
        # in cuda_graph_runner) so both sides agree; still experimental.
        if self._nomask:
            # The VERIFY mask (all-ones prefix ++ tril(K,K)) is element-wise identical to flashinfer's
            # bottom-right aligned CAUSAL (verified), so no mask is built and only causal=True is passed.
            custom_mask = None
        elif os.environ.get("SELFSPEC_MASK", "1") == "1":  # default path (verified lossless)
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
        # The cuda-graph mask-buffer overrun guard lives in CudaGraphRunner.can_run.
        # An earlier guard here was ineffective for three reasons: (1) it looked up cuda_graph_runner but the
        # ModelRunner attribute is graph_runner, so it was always None; (2) nothing reads batch.can_run_cuda_graph;
        # (3) it compared against the wrong buffer (not the flashinfer packed buffer that actually overflows).
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
        _ns = os.environ.get("SELFSPEC_NOSYNC", "0") == "1"
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
        if os.environ.get("SELFSPEC_VERIFY_DUMP", "0") == "1" and getattr(self, "_vd", 0) < 1:
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
        _as = os.environ.get("SELFSPEC_ACCEPT_STATS", "")
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
