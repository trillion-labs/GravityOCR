"""SELFSPEC_FASTATTN=1: mask-free, sync-free flashinfer attention planning for the self-spec DRAFT / VERIFY forwards.

Both self-spec forwards are TARGET_VERIFY forwards of K = bd_size + 1 tokens per request. The inherited n-gram tree-verify path
expressed their attention patterns as a custom bool mask (ngram_info.generate_attn_arg_prefill + flashinfer plan(custom_mask=...),
MaskMode kCustom). Here the same patterns use flashinfer's built-in mask modes over the SAME request layout (one request per
sequence, q = the K window rows, kv = req_to_token[r, 0 : seq_len + K]), planned on the host from lengths the host already knows
(seq_lens_cpu), with no device->host copy:

  VERIFY: MaskMode CAUSAL. flashinfer aligns causal bottom-right: row i sees kv 0 .. seq_len + i == prefix all ones ++ tril(K, K).
  DRAFT:  the same plan run twice in the forward (per layer):
          NON_CAUSAL -> rows 1..K-1 (the masks) see prefix + seed + all masks (bidirectional) == the old mask's rows 1..K-1;
          CAUSAL     -> row 0 (the seed) sees prefix + seed only == the old mask's row 0 ([1, 0, .., 0]); copied into row 0.
  q / o layout, positions and KV slots are unchanged.

Bit-identity with the old path: flashinfer's plan ignores
the mask mode and kCausal / kNone / kCustom share the tile configuration, so with the same plan inputs every row is computed with the
same KV chunks, tiles and merge order as under the old custom mask, and masked keys contribute the same exact zeros. Two cheaper
variants are NOT bit-identical: (a) the seed row as a second attention sub-request next to the mask rows in one call ->
flashinfer's KV-split chunking starts at a shorter kv length, and the seed row's committed K/V no longer matches the old path
independently of the draft; (b) the seed row as its own qo=1 call over kv = prefix + seed (planned with the mask rows' lengths) ->
"kv truncated at the seed" can round differently from "kv with the masked tail" (the seed row then equals the verify's LAST row,
not its row 0). The causal run reproduces row 0.

Planning follows the wrapper's mode exactly like flashinfer's python plan() (CUDA-graph wrappers: graph semantics with the capture's
total_num_rows; eager wrappers: plain), so plan info == flashinfer plan() (checked at capture and on each eager wrapper's first use).

Per forward the host does: numpy arithmetic on seq_lens_cpu, ONE triton launch (_ss_kv_meta_kernel: kv_indptr + kv_indices on the
device from seq_lens / req_pool_indices), ONE call of flashinfer's C++ PrefillPlan (the same function flashinfer's python plan()
calls, minus its .to("cpu") copies, segment_packbits and mask copies). qo_indptr and last_page_len are constant patterns.

CUDA graphs: one captured graph per (bs, role). Each role has its own device metadata buffers, int workspace + page-locked staging
buffer (shared by that role's graphs, one of which is replayed at a time) and an event that guards the page-locked buffer against the
previous plan's still-pending cudaMemcpyAsync (queried first; waited for only if still pending, which the spec-v1 loop never hits).
No fixed split size under graphs (the plan-info layout must stay the capture layout).
Padded rows (raw_bs < captured bs) get seq_len = the fill value and read in-bounds req_to_token rows exactly like the old path;
there is no mask to over-read (cf. SELFSPEC_PADFIX on the old path).

Window length: the plan uses the forward's own window length (spec_info.draft_token_num), not the server's K, so an eager draft of
K + 1 rows (SELFSPEC_TRAIN_MASKS=1) is planned with its own qo pattern; CUDA graphs are captured at the server's K (a K + 1 draft
fails the graph runner's token-count check and runs eager).

Pinned to flashinfer 0.6.7.post2 + the fa2 prefill backend (asserted at startup): _cpp_plan calls the private JIT module's plan()
with that version's argument list and writes the wrapper's private plan fields (_plan_info, _qo_indptr_last, _batch_size).

SELFSPEC_CHECK=1 (diagnostic, syncs every forward; never time with it on): per forward, host seq_lens_cpu == device seq_lens, host
kv_indptr == device kv_indptr, device kv_indices == req_to_token rows, and plan info == flashinfer's own plan() on the same inputs;
any mismatch raises. The worker adds the per-request length check (speculative/selfspec_diffusion_worker.py _check_lens).

Capture-layout assert: a CUDA-graph replay's plan info must equal the plan info its graph was captured with
(all 15 fields: padded_batch_size, total_num_rows, every int-workspace offset, cta_tile_q, enable_cuda_graph, split_kv). The captured
kernels read the int workspace at those offsets, so a replay plan with another layout would silently mis-index; with flashinfer
0.6.7.post2 graph mode fixes all of them per wrapper (split_kv is forced on, scheduler.cuh:129; padded_batch_size / cta_tile_q depend
only on the capture's total_num_rows and bs), and a flashinfer upgrade that changed that raises here instead. The plan info is a
tvm_ffi Array whose element reads are host calls (a full 15-field tuple() is not free), so: EVERY replay checks the
three fields every offset is a function of (padded_batch_size, cta_tile_q, split_kv; the rest of PrefillPlan's allocation sizes are
the wrapper's fixed bs / total_num_rows / heads / head_dim), and the full 15-field compare runs on the first 64 replays of each
wrapper, then every 1024th, and on every replay with SELFSPEC_CHECK=1.

SELFSPEC_SHADOW=1 (diagnostic, eager forwards only, syncs): the worker also builds the OLD custom mask of every draft / verify, the
backend plans the old custom-mask wrapper with it, and every attention call's fast-path output is compared bitwise with the old
path's output on the same q / KV (end-to-end bs > 1 evidence on real batch compositions). Mismatches are counted
and logged, never raised, so a run reports the full count.
"""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

ROLE_DRAFT = "draft"
ROLE_VERIFY = "verify"
ROLES = (ROLE_DRAFT, ROLE_VERIFY)


FLASHINFER_PINNED = "0.6.7.post2"


def fastattn_enabled() -> bool:
    return os.environ.get("SELFSPEC_FASTATTN", "0") == "1"


def check_enabled() -> bool:
    return os.environ.get("SELFSPEC_CHECK", "0") == "1"


def shadow_enabled() -> bool:
    return os.environ.get("SELFSPEC_SHADOW", "0") == "1"


PLAN_INFO_FIELDS = ("padded_batch_size", "total_num_rows", "total_num_rows_offset", "cta_tile_q", "request_indices_offset",
                    "qo_tile_indices_offset", "kv_tile_indices_offset", "merge_indptr_offset", "o_indptr_offset",
                    "kv_chunk_size_ptr_offset", "v_offset", "s_offset", "block_valid_mask_offset", "enable_cuda_graph", "split_kv")


def spec_role(spec_info) -> Optional[str]:
    return getattr(spec_info, "selfspec_role", None) if spec_info is not None else None


@triton.jit
def _ss_kv_meta_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    r2t_stride,
    EXTRA: tl.constexpr,
    BS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """one program per sequence i: kv = req_to_token[r_i, 0 : L_i + EXTRA] at kv_indptr[i] (prefix sum over j < i computed
    in-program, bs <= BS_POW2); writes kv_indptr[i + 1] (and kv_indptr[0] = 0)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BS_POW2)
    prev = tl.sum(tl.load(seq_lens_ptr + offs, mask=offs < pid, other=0).to(tl.int64), axis=0)
    L = tl.load(seq_lens_ptr + pid).to(tl.int64)
    rpi = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    src = req_to_token_ptr + rpi * r2t_stride
    start = prev + pid * EXTRA
    ln = L + EXTRA
    if pid == 0:
        tl.store(kv_indptr_ptr, 0)
    tl.store(kv_indptr_ptr + pid + 1, (start + ln).to(tl.int32))
    for it in range(tl.cdiv(ln, BLOCK)):
        idx = it * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        v = tl.load(src + idx, mask=idx < ln)
        tl.store(kv_indices_ptr + start + idx, v, mask=idx < ln)


class _RoleState:
    """one role's plan target: qo pattern, host/device kv metadata, int workspace, eager wrapper (+ its graph wrappers)."""

    def __init__(self, role: str, max_bs: int, kv_cap: int, device):
        self.role = role
        self.max_bs = max_bs
        self.device = device
        self._qo: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}   # window length -> (host, device) arange(max_bs + 1) * n
        self.kv_indptr_host = torch.zeros(max_bs + 1, dtype=torch.int32)
        self.kv_indptr_np = self.kv_indptr_host.numpy()
        self.kvl_host = torch.zeros(max_bs, dtype=torch.int32)
        self.kvl_np = self.kvl_host.numpy()
        self.kv_indptr_dev = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        self.kv_indices_dev = torch.zeros(kv_cap, dtype=torch.int32, device=device)
        # int workspace (+ page-locked staging) of this role; flashinfer's default size (8 MB)
        self.int_ws = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=device)
        self.pinned_ws = torch.empty(self.int_ws.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
        self.plan_event = torch.cuda.Event()
        self.plan_event_recorded = False
        self.eager_wrapper = None
        self.total_kv = 0
        self.n = 0   # window length of the current plan
        self.pinned_waits = 0   # plans that had to wait for the previous plan's pinned -> device copy (0 in the spec-v1 loop)

    def qo(self, n: int):
        """constant qo_indptr pattern arange(max_bs + 1) * n (prefix-consistent across bs), host + device."""
        t = self._qo.get(n)
        if t is None:
            h = torch.arange(self.max_bs + 1, dtype=torch.int32) * n
            t = self._qo[n] = (h, h.to(self.device))
        return t


class SelfSpecFastPlanner:
    """owned by FlashInferAttnBackend when SELFSPEC_FASTATTN=1."""

    def __init__(self, backend, model_runner):
        import flashinfer
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        # private-API pin (_cpp_plan's argument list and the wrapper fields it writes are flashinfer 0.6.7.post2 / fa2 internals)
        if getattr(flashinfer, "__version__", None) != FLASHINFER_PINNED:
            raise RuntimeError(f"SELFSPEC_FASTATTN=1 is pinned to flashinfer {FLASHINFER_PINNED} (private plan API); "
                               f"found {getattr(flashinfer, '__version__', None)}")
        if backend.prefill_backend != "fa2":
            raise RuntimeError(f"SELFSPEC_FASTATTN=1 requires the fa2 prefill backend (got {backend.prefill_backend})")
        self._Wrapper = BatchPrefillWithPagedKVCacheWrapper
        self.backend = backend
        self.K = int(model_runner.server_args.speculative_num_draft_tokens)
        self.device = model_runner.device
        self.max_bs = int(model_runner.req_to_token_pool.size)
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        ctx = int(model_runner.model_config.context_len)
        # kv per sequence <= seq_len + window (+1 full-draft) <= ctx + K + 3 (window K, or K + 1 with SELFSPEC_TRAIN_MASKS=1)
        self.kv_cap = self.max_bs * (ctx + self.K + 4)
        self.ones = torch.ones((self.max_bs,), dtype=torch.int32, device=self.device)
        self.state: Dict[str, _RoleState] = {r: _RoleState(r, self.max_bs, self.kv_cap, self.device) for r in ROLES}
        self.graph_wrappers: Dict[Tuple[int, str], object] = {}
        # split-KV knobs, same env names and semantics as the old path (flashinfer_backend call_begin_forward)
        self.no_split = os.environ.get("SELFSPEC_VERIFY_NO_SPLIT", "0") == "1"
        self.fixed_split = int(os.environ.get("SELFSPEC_VERIFY_FIXED_SPLIT", "0") or 0)
        self._logged = set()
        self.check = check_enabled()
        self.n_checked = 0
        self.capture_info: Dict[Tuple[int, str], tuple] = {}   # (bs, role) -> plan info at capture (replay layout must match)
        self.replay_count: Dict[Tuple[int, str], int] = {}
        self.n_replay_layout = 0        # replays with the 3-field layout check
        self.n_replay_layout_full = 0   # replays with the full 15-field compare
        # SELFSPEC_SHADOW=1: per-role attention-call counters of the bitwise fast-vs-old comparison (flashinfer_backend.forward_extend)
        self.shadow = shadow_enabled()
        self.shadow_calls = {r: 0 for r in ROLES}
        self.shadow_mismatch = {r: 0 for r in ROLES}
        self.shadow_forwards = {r: 0 for r in ROLES}
        self.shadow_bs_hist: Dict[int, int] = {}
        upd = backend.indices_updater_prefill
        self.num_qo_heads, self.num_kv_heads, self.head_dim = upd.num_qo_heads, upd.num_kv_heads, upd.head_dim
        self.q_dtype, self.kv_dtype = upd.q_data_type, upd.data_type
        assert backend.num_wrappers == 1, "SELFSPEC_FASTATTN: single-wrapper models only (no sliding window / cross attention)"
        assert backend.page_size == 1, "SELFSPEC_FASTATTN requires --page-size 1"
        logger.info(
            f"[selfspec-fastattn] ON: verify=CAUSAL; draft=NON_CAUSAL (mask rows) + CAUSAL run of the same plan (seed row); "
            f"host plan, K={self.K} max_bs={self.max_bs} kv_cap={self.kv_cap} no_split={self.no_split} fixed_split={self.fixed_split} "
            f"flashinfer={flashinfer.__version__}/{backend.prefill_backend} check={int(self.check)}"
        )
        if self.fixed_split > 0 and not self.no_split:
            logger.warning(
                f"[selfspec-fastattn] SELFSPEC_VERIFY_FIXED_SPLIT={self.fixed_split} applies to EAGER forwards only on this path: "
                f"CUDA-graph forwards keep the captured split layout (a fixed split under graphs changes split_kv / the plan-info "
                f"layout baked at capture). The old path applied it under graphs too."
            )
        if self.check:
            logger.warning("[selfspec-fastattn] SELFSPEC_CHECK=1: per-forward host/device metadata + plan-info asserts (syncs; "
                           "diagnostic only, never time with it)")
        if self.shadow:
            logger.warning("[selfspec-fastattn] SELFSPEC_SHADOW=1: every eager attention call is compared bitwise with the old "
                           "custom-mask path (syncs; diagnostic only, never time with it; graph forwards are not compared)")

    # ------------------------------------------------------------------ helpers
    def _once(self, key, msg):
        if key not in self._logged:
            self._logged.add(key)
            logger.info(msg)

    def _meta(self, role: str, bs: int, n: int, req_pool_indices, seq_lens, seq_lens_cpu):
        """host kv lengths / indptr + ONE triton launch writing the device kv_indptr / kv_indices of the role (window length n)."""
        if seq_lens_cpu is None:
            raise RuntimeError("[selfspec-fastattn] seq_lens_cpu is None (host lengths are needed for a sync-free plan)")
        st = self.state[role]
        st.n = n
        kvl = st.kvl_np[:bs]
        kvl[:] = seq_lens_cpu[:bs].numpy() + n
        kvi = st.kv_indptr_np
        kvi[0] = 0
        np.cumsum(kvl, dtype=np.int32, out=kvi[1 : bs + 1])
        st.total_kv = int(kvi[bs])
        if st.total_kv > st.kv_indices_dev.numel():
            raise RuntimeError(f"[selfspec-fastattn] {role}: kv indices {st.total_kv} > buffer {st.kv_indices_dev.numel()}")
        _ss_kv_meta_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            st.kv_indptr_dev,
            st.kv_indices_dev,
            self.req_to_token.shape[1],
            EXTRA=n,
            BS_POW2=triton.next_power_of_2(max(bs, 2)),
            BLOCK=512,
        )
        if self.check:
            self._check_meta(st, role, bs, n, req_pool_indices, seq_lens, seq_lens_cpu)
        return st

    def _check_meta(self, st: _RoleState, role: str, bs: int, n: int, req_pool_indices, seq_lens, seq_lens_cpu):
        """SELFSPEC_CHECK=1: the host plan inputs must describe exactly what the device kernels read."""
        dev_sl = seq_lens[:bs].to("cpu", torch.int64)
        host_sl = seq_lens_cpu[:bs].to(torch.int64)
        if not torch.equal(dev_sl, host_sl):
            raise RuntimeError(f"[selfspec-check] {role} bs={bs}: seq_lens_cpu {host_sl.tolist()} != device seq_lens {dev_sl.tolist()}")
        dev_ip = st.kv_indptr_dev[: bs + 1].cpu()
        if not torch.equal(dev_ip, st.kv_indptr_host[: bs + 1]):
            raise RuntimeError(f"[selfspec-check] {role} bs={bs}: device kv_indptr {dev_ip.tolist()} != host "
                               f"{st.kv_indptr_host[: bs + 1].tolist()}")
        rpi = req_pool_indices[:bs].long()
        ref = torch.cat([self.req_to_token[rpi[i], : int(host_sl[i]) + n] for i in range(bs)])
        if not torch.equal(st.kv_indices_dev[: st.total_kv].to(ref.dtype), ref):
            raise RuntimeError(f"[selfspec-check] {role} bs={bs}: device kv_indices != req_to_token rows")

    def _split_args(self, graph: bool):
        # a fixed split size under CUDA graphs changes split_kv / the plan-info layout vs the captured one -> eager only (warned
        # once at startup)
        fixed = self.fixed_split if (self.fixed_split > 0 and not graph and not self.no_split) else -1
        return fixed, self.no_split

    def _wait_pinned(self, st: _RoleState):
        # PrefillPlan writes the page-locked buffer, then cudaMemcpyAsync's it to the int workspace without a sync: the previous
        # plan's copy of this role must have run before the buffer is rewritten. cudaEventQuery first (non-blocking): in the spec-v1
        # loop a readback separates consecutive plans of one role, so the copy is always done and nothing blocks (an
        # unconditional cudaEventSynchronize would be a host sync between the draft and verify graph launches).
        # Only a still-pending copy is waited for, and counted; a pipelined loop would need ping-pong buffers instead.
        if st.plan_event_recorded and not st.plan_event.query():
            st.pinned_waits += 1
            if st.pinned_waits in (1, 10, 100, 1000):
                logger.warning(f"[selfspec-fastattn] {st.role}: re-plan waited for the previous plan's pinned-buffer copy "
                               f"({st.pinned_waits} times)")
            st.plan_event.synchronize()

    def _cpp_plan(self, st: _RoleState, w, bs: int):
        """flashinfer's PrefillPlan with the arguments flashinfer 0.6.7.post2's python plan() passes (fa2 backend,
        prefill.py:2021-2046)."""
        graph = w.is_cuda_graph_enabled
        fixed, no_split = self._split_args(graph)
        n = st.n
        self._wait_pinned(st)
        w._plan_info = w._cached_module.plan(
            w._float_workspace_buffer,
            w._int_workspace_buffer,
            w._pin_memory_int_workspace_buffer,
            st.qo(n)[0][: bs + 1],
            st.kv_indptr_host[: bs + 1],
            st.kvl_host[:bs],
            (w._max_total_num_rows or bs * n) if graph else bs * n,
            bs,
            self.num_qo_heads,
            self.num_kv_heads,
            1,
            graph,
            self.head_dim,
            self.head_dim,
            False,   # causal: unused by the fa2 plan
            -1,
            fixed,
            no_split,
            0,
        )
        st.plan_event.record()
        st.plan_event_recorded = True
        w._qo_indptr_last = bs * n
        w._batch_size = bs

    def _python_plan(self, st: _RoleState, w, bs: int):
        """flashinfer's own plan() (JIT module, dtypes, graph-mode _max_total_num_rows; reference plan info). Inputs are fresh
        copies so that a graph wrapper's plan() copies into its fixed buffers from distinct memory."""
        fixed, no_split = self._split_args(w.is_cuda_graph_enabled)
        self._wait_pinned(st)
        w.plan(
            st.qo(st.n)[0][: bs + 1].clone(),
            st.kv_indptr_host[: bs + 1].clone(),
            st.kv_indices_dev[: st.total_kv].clone(),
            self.ones[:bs].clone(),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            non_blocking=True,
            fixed_split_size=fixed if fixed > 0 else None,
            disable_split_kv=no_split,
        )
        st.plan_event.record()
        st.plan_event_recorded = True
        if w._backend != "fa2" or w._cached_module is None:
            raise RuntimeError(f"[selfspec-fastattn] wrapper resolved backend {w._backend} (fa2 required by _cpp_plan)")
        return list(w._plan_info)

    def _check(self, where, role, bs, w, ref):
        got = list(w._plan_info)
        if got != ref:
            raise RuntimeError(f"[selfspec-fastattn] {where} {role} bs={bs}: host plan info {got} != flashinfer plan() {ref}")
        self._once((where, role), f"[selfspec-fastattn] {where} {role} wrapper ready (bs={bs}: plan info == flashinfer plan(): "
                                  f"{got[:4]}.. split_kv={got[14]})")

    def _metadata(self, role: str, w, n: int):
        from sglang.srt.layers.attention.flashinfer_backend import PrefillMetadata

        draft = role == ROLE_DRAFT
        return PrefillMetadata([w], False, False, ss_causal=not draft, ss_seed_causal=draft, ss_K=n)

    def _plan_checked(self, st: _RoleState, w, role: str, bs: int, where: str):
        """_cpp_plan; with SELFSPEC_CHECK=1 first flashinfer's own plan() on the same inputs as the reference plan info."""
        ref = self._python_plan(st, w, bs) if self.check else None
        self._cpp_plan(st, w, bs)
        if ref is not None:
            self._check(where, role, bs, w, ref)
            self.n_checked += 1
            if self.n_checked in (1, 10, 100, 1000) or self.n_checked % 10000 == 0:
                logger.info(f"[selfspec-check] {self.n_checked} fast-path plans checked (metadata + plan info), 0 mismatches")

    def _share_int_ws(self, st: _RoleState, w):
        w._int_workspace_buffer = st.int_ws
        w._pin_memory_int_workspace_buffer = st.pinned_ws

    # ------------------------------------------------------------------ eager
    def plan_eager(self, role: str, forward_batch):
        bs = forward_batch.batch_size
        # the forward's own window length (K, or K + 1 for a SELFSPEC_TRAIN_MASKS=1 draft)
        n = int(getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", None) or self.K)
        st = self._meta(role, bs, n, forward_batch.req_pool_indices, forward_batch.seq_lens, forward_batch.seq_lens_cpu)
        w = st.eager_wrapper
        ref = None
        if w is None:
            w = self._Wrapper(self.backend.workspace_buffer, "NHD", backend=self.backend.prefill_backend)
            self._share_int_ws(st, w)
            ref = self._python_plan(st, w, bs)
            st.eager_wrapper = w
        w._qo_indptr_buf = st.qo(n)[1][: bs + 1]
        w._paged_kv_indptr_buf = st.kv_indptr_dev[: bs + 1]
        w._paged_kv_indices_buf = st.kv_indices_dev[: st.total_kv]
        w._paged_kv_last_page_len_buf = self.ones[:bs]
        w._custom_mask_buf = None
        w._mask_indptr_buf = None
        if ref is not None:
            self._cpp_plan(st, w, bs)
            self._check("eager", role, bs, w, ref)
        else:
            self._plan_checked(st, w, role, bs, "eager")
        return self._metadata(role, w, n)

    # ------------------------------------------------------------------ cuda graph
    def capture(self, role: str, bs: int, req_pool_indices, seq_lens):
        n = self.K   # graphs are captured at the server's window length (graph runner num_tokens_per_bs)
        st = self._meta(role, bs, n, req_pool_indices, seq_lens, seq_lens.cpu())   # capture only
        w = self._Wrapper(
            self.backend.workspace_buffer,
            "NHD",
            use_cuda_graph=True,
            backend=self.backend.prefill_backend,
            qo_indptr_buf=st.qo(n)[1][: bs + 1],
            paged_kv_indptr_buf=st.kv_indptr_dev[: bs + 1],
            paged_kv_indices_buf=st.kv_indices_dev,
            paged_kv_last_page_len_buf=self.ones[:bs],
        )
        self._share_int_ws(st, w)
        ref = self._python_plan(st, w, bs)
        self._cpp_plan(st, w, bs)
        self._check("capture", role, bs, w, ref)
        self.graph_wrappers[(bs, role)] = w
        self.capture_info[(bs, role)] = tuple(int(x) for x in w._plan_info)
        return self._metadata(role, w, n)

    def plan_replay(self, role: str, bs: int, req_pool_indices, seq_lens, seq_lens_cpu):
        w = self.graph_wrappers[(bs, role)]
        st = self._meta(role, bs, self.K, req_pool_indices, seq_lens, seq_lens_cpu)
        self._plan_checked(st, w, role, bs, "replay")
        self._check_capture_layout(w, role, bs)
        return self._metadata(role, w, self.K)

    def _check_capture_layout(self, w, role: str, bs: int):
        """the replay plan info must be the capture's (the graph's kernels read the int workspace at its offsets): 3 layout fields on
        every replay, all 15 on the first 64 replays of the wrapper, every 1024th, and always with SELFSPEC_CHECK=1."""
        key = (bs, role)
        cap = self.capture_info[key]
        pi = w._plan_info
        c = self.replay_count[key] = self.replay_count.get(key, 0) + 1
        full = self.check or c <= 64 or (c & 1023) == 0
        if full:
            got = tuple(pi)
            self.n_replay_layout_full += 1
        else:
            got = None
            if pi[0] != cap[0] or pi[3] != cap[3] or pi[14] != cap[14]:
                got = tuple(pi)
        if got is not None and got != cap:
            diff = {f: (a, b) for f, a, b in zip(PLAN_INFO_FIELDS, cap, got) if a != b}
            raise RuntimeError(f"[selfspec-fastattn] replay {role} bs={bs}: plan-info layout differs from the capture's {diff} "
                               f"(capture {cap}, replay {got})")
        self.n_replay_layout += 1
        if self.n_replay_layout in (1, 1000, 100000) or self.n_replay_layout % 1000000 == 0:
            logger.info(f"[selfspec-fastattn] {self.n_replay_layout} graph replays: plan-info layout == capture layout "
                        f"({self.n_replay_layout_full} with the full 15-field compare)")

    # ------------------------------------------------------------------ SELFSPEC_SHADOW
    def shadow_record(self, role: str, layer_id: int, bs: int, o_new, o_old):
        """bitwise compare of one attention call (fast path vs old custom-mask path); counts, logs the first mismatches."""
        self.shadow_calls[role] += 1
        if layer_id == 0:
            self.shadow_forwards[role] += 1
            self.shadow_bs_hist[bs] = self.shadow_bs_hist.get(bs, 0) + 1
        if not torch.equal(o_new, o_old):
            self.shadow_mismatch[role] += 1
            if self.shadow_mismatch[role] <= 20:
                d = (o_new.float() - o_old.float()).abs()
                rows = torch.nonzero(d.flatten(1).amax(1) > 0).flatten()
                logger.warning(f"[selfspec-shadow] MISMATCH {role} layer {layer_id} bs={bs}: {rows.numel()} of {o_new.shape[0]} rows "
                               f"differ (first {rows[:8].tolist()}), max |d| {float(d.max()):.3e}")
        n = sum(self.shadow_calls.values())
        if n in (1, 100) or n % 5000 == 0:
            logger.info(f"[selfspec-shadow] {n} attention calls compared: calls {self.shadow_calls} forwards {self.shadow_forwards} "
                        f"mismatches {self.shadow_mismatch} bs_hist {dict(sorted(self.shadow_bs_hist.items()))}")
