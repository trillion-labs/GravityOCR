"""Exact-M CUDA graphs for the GLM-OCR ViT (GlmOcrVisionModel), default OFF.

Flag: SS_GLM_VIT_GRAPH_MAXM=<max patches per image> (unset/0 = off; e.g. 2048 covers typical document crops).

Why: a GLM crop's ViT call is hundreds of eager launches plus a per-block cu_seqlens H2D copy and a per-block
`.max().item()` sync (FA3), so its host time is many times its GPU time.

What (all in this file; wired in by models/glm_ocr.py and the list dispatch in layers/attention/vision.py):
  * capture at the end of GlmOcrForConditionalGeneration.load_weights, under no_grad (see capture() for why not inference_mode),
    with the default dtype set to float32 (the serving context; load_weights runs inside set_default_torch_dtype(bf16)):
    every M = 4k <= MAXM (every patch count reachable with even h and w), largest first, into ONE shared pool.
    Static inputs are [:M] views of max-size buffers (pixel rows, cos, sin) plus a per-graph cu_seqlens = [0, M].
    A graph is keyed on M alone: the only (h, w)-dependent tensors are cos / sin, which are graph inputs.
  * replay only for one image with t == 1, M captured, x has M rows (FA3 backend and no DP encoder are checked at capture).
    The output is cloned right after the replay (graphs share one pool, so a later replay can overwrite it).
  * everything else (M > MAXM, several images, t > 1) runs the eager path with cu_seqlens passed once as the FA3 list
    [cu_gpu, max_seqlen] (no per-block H2D copy and no per-block .item() sync). SS_GLM_VIT_EAGER_LIST=0 keeps the
    original eager forward for those calls instead.
  * self-check (SS_GLM_VIT_GRAPH_CHECK, default 1): every M right after its capture (seeded random input, replay vs the
    ORIGINAL eager forward, torch.equal), then all M again in shuffled order after all captures, plus a few grids that
    differ from the capture grid and a few eager-list fallback shapes. A mismatching M is dropped (-> eager);
    a mismatching eager-list shape turns the eager-list fallback off.
  * startup line "[ss-vit-graph] ..." (graph count, seconds, GB, dropped M) and a usage line every 1000 calls.
"""
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch


def glm_vit_graph_maxm() -> int:
    try:
        return int(os.environ.get("SS_GLM_VIT_GRAPH_MAXM", "0") or 0)
    except ValueError:
        return 0


def _p(msg: str) -> None:  # print: the in-process Engine logs at level "error", so logger.info would be invisible
    print(f"[ss-vit-graph] {msg}", flush=True)


class GlmVitGraphRunner:
    D_IN_DEFAULT = 1176  # 3 * 2 * 14 * 14 pixel values per patch row (GLM-OCR)

    def __init__(self, vis, maxm: int):
        self.vis = vis
        self.maxm = int(maxm) - int(maxm) % 4
        self.graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self.eager_list_ok = os.environ.get("SS_GLM_VIT_EAGER_LIST", "1") == "1"
        self.check = os.environ.get("SS_GLM_VIT_GRAPH_CHECK", "1") == "1"
        self.n_replay = 0
        self.n_eager_list = 0
        self.n_eager_orig = 0
        self.dropped: List[int] = []
        self.X = self.C = self.S = None
        self.pool = None

    # ------------------------------------------------------------------ the forward body (== GlmOcrVisionModel.forward)
    def _body(self, x, cu_list, cos, sin):
        vis = self.vis
        x = vis.patch_embed(x)
        x = x.unsqueeze(1)
        for blk in vis.blocks:
            x = blk(x, cu_seqlens=cu_list, rotary_pos_emb_cos=cos, rotary_pos_emb_sin=sin)
        x = vis.post_layernorm(x)
        x = x.view(-1, vis.spatial_merge_size, vis.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = vis.downsample(x).view(-1, vis.out_hidden_size)
        x = vis.merger(x)
        return x

    def _rot(self, grid_thw):
        cos, sin, _ = self.vis.rot_pos_emb(grid_thw)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)

    def _eager_list(self, x, grid_thw):
        vis = self.vis
        x = x.to(device=vis.device, dtype=vis.dtype)  # patch_embed runs inside _body
        cos, sin = self._rot(grid_thw)
        g = grid_thw.cpu() if grid_thw.is_cuda else grid_thw
        cu = torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu = torch.cat([cu.new_zeros(1), cu])
        max_seqlen = int((cu[1:] - cu[:-1]).max())
        return self._body(x, [cu.to(vis.device), max_seqlen], cos, sin)

    def _eager_orig(self, x, grid_thw):
        return self.vis._ss_forward_orig(x, grid_thw)

    # ------------------------------------------------------------------ serving entry point
    def run(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        n = self.n_replay + self.n_eager_list + self.n_eager_orig
        if n and n % 1000 == 0:
            _p(f"usage: calls={n} replay={self.n_replay} eager_list={self.n_eager_list} eager_orig={self.n_eager_orig}")
        if self.graphs and grid_thw.shape[0] == 1:
            t, h, w = (int(v) for v in grid_thw[0].tolist())
            M = t * h * w
            if t == 1 and M in self.graphs and x.shape[0] == M:
                self.n_replay += 1
                return self._replay(x, grid_thw, M)
        if self.eager_list_ok:
            self.n_eager_list += 1
            return self._eager_list(x, grid_thw)
        self.n_eager_orig += 1
        return self._eager_orig(x, grid_thw)

    def _load_static(self, M, x, grid_thw):
        cos, sin, _ = self.vis.rot_pos_emb(grid_thw)
        d = cos.shape[-1]
        self.X[:M].copy_(x)
        self.C[:M, :d].copy_(cos)
        self.C[:M, d:].copy_(cos)
        self.S[:M, :d].copy_(sin)
        self.S[:M, d:].copy_(sin)

    def _replay(self, x, grid_thw, M):
        gr, out, _cu = self.graphs[M]
        self._load_static(M, x, grid_thw)
        gr.replay()
        return out.clone()

    # ------------------------------------------------------------------ startup capture + self-check
    def capture(self) -> None:
        vis = self.vis
        dev = vis.device
        from sglang.srt.layers.attention.vision import VisionFlash3Attention

        if not all(isinstance(b.attn.qkv_backend, VisionFlash3Attention) for b in vis.blocks):
            _p(f"OFF: ViT attention backend is {type(vis.blocks[0].attn.qkv_backend).__name__}, graphs need fa3")
            self.eager_list_ok = False
            return
        if self.maxm < 4:
            _p("OFF: MAXM < 4")
            return
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)  # = the serving context (load_weights runs under set_default_torch_dtype)
        try:
            # no_grad, NOT inference_mode: the first CUDA-graph capture of the process lazily creates the CUDA
            # generator's graph-state tensors; created under inference_mode they become inference tensors and the
            # LLM's CudaGraphRunner capture (outside inference_mode) then dies with "Inplace update to inference
            # tensor outside InferenceMode". The kernels do not depend on the mode.
            with torch.no_grad():
                self._capture_all(dev)
        finally:
            torch.set_default_dtype(prev_dtype)

    def _capture_all(self, dev):
        vis = self.vis
        t_all = time.perf_counter()
        d_in = vis.patch_embed.proj.weight.numel() // vis.patch_embed.hidden_size
        # warm the eager path once (compiles the dynamic-shape rotary kernel, cuBLAS handles)
        g0 = torch.tensor([[1, 26, 26]])
        for _ in range(2):
            vis._ss_forward_orig(torch.zeros(676, d_in, device=dev, dtype=vis.dtype), g0)
        torch.cuda.synchronize()
        cos0, _ = self._rot(torch.tensor([[1, 2, self.maxm // 2]]))
        rot_d = cos0.shape[-1]
        mem0 = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        self.X = torch.zeros(self.maxm, d_in, device=dev, dtype=vis.dtype)
        self.C = torch.zeros(self.maxm, rot_d, device=dev, dtype=cos0.dtype)
        self.S = torch.zeros(self.maxm, rot_d, device=dev, dtype=cos0.dtype)
        self.pool = torch.cuda.graph_pool_handle()
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        Ms = [4 * k for k in range(self.maxm // 4, 0, -1)]  # largest -> smallest
        refs: Dict[int, torch.Tensor] = {}
        t_cap = 0.0
        t_chk = 0.0
        for M in Ms:
            a = time.perf_counter()
            g = torch.tensor([[1, 2, M // 2]])
            cos, sin = self._rot(g)
            xs, cs, ss = self.X[:M], self.C[:M], self.S[:M]
            cs.copy_(cos)
            ss.copy_(sin)
            cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
            side.wait_stream(cur)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                self._body(xs, [cu, M], cs, ss)  # warm run of this M (Triton specialisation, cuBLAS heuristics)
                side.synchronize()
                # capture_begin/end directly: torch.cuda.graph() would add gc.collect() + empty_cache() per graph
                gr.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                try:
                    out = self._body(xs, [cu, M], cs, ss)
                finally:
                    gr.capture_end()
            cur.wait_stream(side)
            torch.cuda.synchronize()
            self.graphs[M] = (gr, out, cu)  # keep every graph + its output referenced (shared pool)
            t_cap += time.perf_counter() - a
            if self.check:
                a = time.perf_counter()
                rnd = self._rand(M, d_in, dev)
                ref = vis._ss_forward_orig(rnd, g)
                y = self._replay(rnd, g, M)
                if not torch.equal(y, ref):
                    self.dropped.append(M)
                else:
                    refs[M] = ref
                t_chk += time.perf_counter() - a
        for M in self.dropped:
            self.graphs.pop(M, None)
        n_shuf = n_grid = n_el = 0
        bad_shuf, bad_grid, bad_el = [], [], []
        if self.check:
            a = time.perf_counter()
            order = list(refs.keys())
            random.Random(1).shuffle(order)
            for M in order:  # arbitrary replay order after ALL captures (shared-pool aliasing)
                g = torch.tensor([[1, 2, M // 2]])
                y = self._replay(self._rand(M, d_in, dev), g, M)
                n_shuf += 1
                if not torch.equal(y, refs[M]):
                    bad_shuf.append(M)
            refs.clear()
            rng = random.Random(2)
            cand = []
            for M in (16, 144, 360, 576, 1024, 1536, 2048):
                if M in self.graphs:
                    hs = [h for h in range(4, 200, 2) if M % h == 0 and (M // h) % 2 == 0]
                    if hs:
                        h = rng.choice(hs)
                        cand.append((M, h, M // h))
            for (M, h, w) in cand:  # same M, grid != capture grid
                g = torch.tensor([[1, h, w]])
                rnd = self._rand(M, d_in, dev)
                ref = vis._ss_forward_orig(rnd, g)
                y = self._replay(rnd, g, M)
                n_grid += 1
                if not torch.equal(y, ref):
                    bad_grid.append((M, h, w))
            if self.eager_list_ok:
                for (h, w) in ((46, 46), (50, 60), (64, 64), (2, 1100)):  # M > MAXM (or not captured): eager-list fallback
                    g = torch.tensor([[1, h, w]])
                    rnd = self._rand(h * w, d_in, dev)
                    ref = vis._ss_forward_orig(rnd, g)
                    y = self._eager_list(rnd, g)
                    n_el += 1
                    if not torch.equal(y, ref):
                        bad_el.append((h, w))
                gm = torch.tensor([[1, 20, 30], [1, 16, 16]])  # two images in one call
                rnd = self._rand(856, d_in, dev)
                ref = vis._ss_forward_orig(rnd, gm)
                y = self._eager_list(rnd, gm)
                n_el += 1
                if not torch.equal(y, ref):
                    bad_el.append("2img")
                if bad_el:
                    self.eager_list_ok = False
            t_chk += time.perf_counter() - a
            for M in bad_shuf:
                self.graphs.pop(M, None)
                self.dropped.append(M)
            for (M, h, w) in bad_grid:
                self.graphs.pop(M, None)
                self.dropped.append(M)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem1 = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        _p(f"ON: {len(self.graphs)} graphs (M 4..{self.maxm}, step 4) in {time.perf_counter() - t_all:.1f} s "
           f"(capture {t_cap:.1f} s, self-check {t_chk:.1f} s, check={int(self.check)}), mem +{(mem1[0] - mem0[0]) / 1e9:.2f} GB alloc "
           f"/ +{(mem1[1] - mem0[1]) / 1e9:.2f} GB reserved; dropped M: {sorted(set(self.dropped))}; "
           f"shuffled {n_shuf} bad {bad_shuf}; other-grid {n_grid} bad {bad_grid}; "
           f"eager_list={'on' if self.eager_list_ok else 'off'} (checked {n_el}, bad {bad_el}); load {os.getloadavg()[0]:.0f}")

    def _rand(self, M, d_in, dev):
        gen = torch.Generator(device=dev).manual_seed(1000 + M)
        return torch.randn(M, d_in, device=dev, dtype=self.vis.dtype, generator=gen)
