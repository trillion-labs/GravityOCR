"""Forwards-per-token benchmark (single GPU, bs=1). Page set = VAL_PAGES (a save_to_disk dataset of
region crops with prompt/target); results depend on the page set, so always state which one.
This counts decoder forwards; it does not price a forward, so do not compare cached vs uncached
paths by tok/fwd alone (the uncached reference has a higher tok/fwd and is ~2x slower in wall time).

Rigorous forwards-per-token benchmark — answers "to generate N tokens, how many decoder
forwards on average?" for each decode method, on the SAME samples, counted identically.

Counts forwards via a forward-hook on text_model (fires once per decoder forward pass) — so
plain block-diffusion and AR-verified spec are measured on one consistent basis.
Also reports wall/sample (single-stream, UNCACHED path — note: not the deployment tps) and edit.

Methods on the dllm-AR model: (a) plain block-diffusion generate (thr0.9), (b) spec (ds=1, accept_conf=0).
Env: CKPT (hf checkpoint dir), VAL_PAGES (crop dataset), N (default 100). Single GPU.
"""
import os, json, time, difflib, sys, torch
os.environ.setdefault("NOSTRIP", "1")
# tools/ imports src/ modules; if the caller does not set PYTHONPATH this dies with
# ModuleNotFoundError: No module named 'models'.
# So the path is set up here, independent of the launcher.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from datasets import load_from_disk
from models.glm_ocr_dllm import GlmOcrBlockDiffusion

import random
CKPT = os.environ["CKPT"]; dev = "cuda"
N = int(os.environ.get("N", "100")); SEED = int(os.environ.get("SEED", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1")); SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
DRAFT_BD = os.environ.get("DRAFT_BD", "")  # K-sweep: override natcache draft window (default = model bd_size)
bd = json.load(open(f"{CKPT}/block_diffusion.json"))["bd_size"]
ds = load_from_disk(os.environ["VAL_PAGES"], keep_in_memory=False)   # save_to_disk dataset of region crops: image, prompt, target, task_type
# STRATIFY=1 -> draw N samples PER task_type.
#   Why: OmniDocBench regions are ~25x more text blocks than tables, so a random N=100 contains only
#   2-3 tables and a per-type breakdown is statistically meaningless. Stratifying gives N per type.
#   The sample differs, so absolute tok/s from a stratified run must NOT be compared with a random-N run;
#   use the stratified run for per-type breakdowns only.
if os.environ.get("STRATIFY", "0") == "1":
    import collections as _c, random as _r
    _by = _c.defaultdict(list)
    for _i, _t in enumerate(ds["task_type"]):
        _by[_t].append(_i)
    _r.seed(SEED)
    _sel = []
    for _t in sorted(_by):
        _pool = _by[_t]
        _sel += _r.sample(_pool, min(N, len(_pool)))
    _sel.sort()
    ds = ds.select(_sel)
    print(f"[bench-fwd] STRATIFY: " + str({k: v for k, v in sorted(_c.Counter(ds["task_type"]).items())}), flush=True)
random.seed(SEED)
# With STRATIFY the dataset is already reduced to N per type; sampling N again here would undo the
#   stratification, so the stratified run uses the whole ds.
if os.environ.get("STRATIFY", "0") == "1":
    allidx = list(range(ds.num_rows))
else:
    allidx = sorted(random.sample(range(ds.num_rows), min(N, ds.num_rows)))  # RANDOM (seeded) sample, not first-N
idxs = allidx[SHARD_ID::NUM_SHARDS]                                        # disjoint slice — each GPU times its own (no inter-GPU/NCCL in the measurement)
print(f"[bench-fwd] CKPT={CKPT} bd={bd} shard {SHARD_ID}/{NUM_SHARDS} -> {len(idxs)} of N={len(allidx)} RANDOM seed={SEED}", flush=True)

def nedit(a, b):
    a, b = a or "", b or ""
    if not a and not b: return 0.0
    return 1 - difflib.SequenceMatcher(None, a, b).ratio()

m = GlmOcrBlockDiffusion(model_id=CKPT, bd_size=bd, dtype=torch.bfloat16).to(dev).eval()
tok = m.tokenizer

cnt = {"n": 0}
def hook(*a): cnt["n"] += 1
# hook the FIRST decoder layer (fires once per forward pass) — robust whether the path calls
# text_model.forward() or iterates layers directly (_run_decoder_stack does the latter -> a
# text_model-level hook would NEVER fire for diffusion/spec). layers[0] always runs once/forward.
h = m.text_model.layers[0].register_forward_hook(lambda *a, **k: hook())

def ntok(txt): return len(tok(txt, add_special_tokens=False).input_ids)

agg = {}  # method -> dict of sums
# Per-sample ledger. Aggregates alone make type/length breakdowns impossible: tok/fwd depends
#   strongly on the page set because a round can commit at most as many tokens as the output has
#   left — short outputs cap acceptance. Showing that requires per-sample
#   (task_type, output tokens, forwards, wall).
#   The aggregate logic (agg) is unchanged; the ledger is additive.
PERSAMPLE = []
_cur_meta = {"task_type": None, "idx": None, "probe": None}
# PREFILL_PROBE=1 -> additionally time a max_new_tokens=1 request per arm to measure the fixed cost
#   (vision encode + prefill + 1 decode) per crop. decode-only tok/s = (toks-1)/(wall - wall_prefill).
#   NOTE: the probe warms the server-side prefix cache, so end-to-end numbers from the same run are optimistic.
PROBE = os.environ.get("PREFILL_PROBE", "0") == "1"
# ARMS (opt-in): run only the listed arms. Unset = all arms.
#   On a large crop set all 6 arms take half a day on one GPU, and the two slowest (plain_diffusion,
#   spec_ds1_ARverify: both uncached references) are not deployment paths.
#   e.g. ARMS=cached_diffusion,spec_prefixcache,spec_natcache
#   A partial run's json lacks the other arms; record the arm set when merging runs.
_ARMS = [s.strip() for s in os.environ.get("ARMS", "").split(",") if s.strip()]
def _want(name):
    return (not _ARMS) or any(a in name for a in _ARMS)
if _ARMS:
    print(f"[bench-fwd] ARMS filter: {_ARMS} (other arms skipped)", flush=True)
def add(method, fwd, toks, wall, edit, rounds=0):
    a = agg.setdefault(method, dict(fwd=0, toks=0, wall=0.0, edit=0.0, n=0, rounds=0))
    a["fwd"] += fwd; a["toks"] += toks; a["wall"] += wall; a["edit"] += edit; a["n"] += 1; a["rounds"] += rounds
    PERSAMPLE.append({"method": method, "task_type": _cur_meta["task_type"], "idx": _cur_meta["idx"],
                      "fwd": fwd, "toks": toks, "wall": wall, "edit": edit, "rounds": rounds,
                      "wall_prefill": _cur_meta.get("probe")})

for c, idx in enumerate(idxs):
    ex = ds[idx]; gt = ex["target"]; pr = ex.get("prompt", "Text Recognition:")
    # MAXNEW lifts the per-task cap (default 4096). Caps of 256/768 cut the long outputs where
    #   acceptance is highest and understate table/formula tok/fwd and tok/s.
    mnt = int(os.environ["MAXNEW"]) if os.environ.get("MAXNEW") else (
        256 if ex["task_type"] == "text" else (768 if ex["task_type"] == "table" else 256))
    _cur_meta["task_type"] = ex["task_type"]; _cur_meta["idx"] = int(idx)
    # (a) plain block-diffusion
    if _want("plain_diffusion"):
        cnt["n"] = 0; t0 = time.time()
        txt = m.generate(ex["image"], prompt=pr, max_new_tokens=mnt, threshold=0.9)
        add("plain_diffusion_thr0.9", cnt["n"], ntok(txt), time.time() - t0, nedit(txt, gt))
    # (a2) CACHED block-diffusion @0.9 — real block-causal KV cache (== eval.sh --fast deployment path),
    #      SAME pages/boundary as spec/AR so diffusion tps is comparable (NOT the uncached plain above).
    #      Score-independent: this is a speed bench; eval.sh scoring is untouched.
    import fast_sampling as _fscache
    # DIFF_THR: measure plain diffusion at several confidence thresholds so its tok/fwd can be matched
    #   to spec — i.e. compare errors at equal parallelism. For that both must be counted on the
    #   same boundary:
    #   forwards are counted by the text_model hook here, exactly as for spec (do not mix in trace-derived values).
    for _thr in ([float(x) for x in os.environ.get("DIFF_THR", "0.9").split(",")]
                 if _want("cached_diffusion") else []):
        cnt["n"] = 0; t0 = time.time()
        _ctrace = []          # committed (unmasked) tokens per forward
        ctxt = _fscache.generate_cached(m, ex["image"], prompt=pr, max_new_tokens=mnt, threshold=_thr,
                                        commit_trace=_ctrace)
        add(f"cached_diffusion_thr{_thr:g}", cnt["n"], ntok(ctxt), time.time() - t0, nedit(ctxt, gt))
        # Arm aligned with how other diffusion OCR systems report TPF: numerator = generated tokens actually
        #   emitted (ntok, no tail/think), denominator = forwards in which a commit happened (len(commit_trace)).
        if _ctrace:
            add(f"isoaxis_diffusion_thr{_thr:g}", len(_ctrace), ntok(ctxt),
                time.time() - t0, nedit(ctxt, gt))
    # (b) AR-verified spec, MTP-style 1-fwd draft
    if _want("spec_ds1_ARverify"):
        cnt["n"] = 0; t0 = time.time()
        stxt, st = m.generate_speculative(ex["image"], prompt=pr, max_new_tokens=mnt,
                                          accept_conf=0.0, draft_steps=1, return_stats=True)
        add("spec_ds1_ARverify", cnt["n"], ntok(stxt), time.time() - t0, nedit(stxt, gt), st.get("rounds", 0))
    # (c) prefixcache spec (current eval default: vision cached, response re-forwarded each round)
    import fast_sampling
    if _want("spec_prefixcache"):
        cnt["n"] = 0; t0 = time.time()
        ptxt, sp = fast_sampling.generate_speculative_prefixcache(m, ex["image"], prompt=pr, max_new_tokens=mnt,
                                          accept_conf=0.0, draft_steps=1, return_stats=True)
        add("spec_prefixcache_ds1", cnt["n"], ntok(ptxt), time.time() - t0, nedit(ptxt, gt), sp.get("rounds", 0))
    # (d) natcache spec (Nemotron-style O(N) token-causal incremental cache, 2 fwd/round)
    if _want("spec_natcache"):
        if PROBE:
            _p0 = time.time()
            fast_sampling.generate_speculative_natcache(m, ex["image"], prompt=pr, max_new_tokens=1,
                                                        accept_conf=0.0, draft_steps=1, return_stats=True,
                                                        draft_bd=(int(DRAFT_BD) if DRAFT_BD else None))
            _cur_meta["probe"] = time.time() - _p0
        cnt["n"] = 0; t0 = time.time()
        natxt, sn = fast_sampling.generate_speculative_natcache(m, ex["image"], prompt=pr, max_new_tokens=mnt,
                                          accept_conf=0.0, draft_steps=1, return_stats=True,
                                          draft_bd=(int(DRAFT_BD) if DRAFT_BD else None))
        add("spec_natcache_ds1", cnt["n"], ntok(natxt), time.time() - t0, nedit(natxt, gt), sn.get("rounds", 0))
    _cur_meta["probe"] = None
    if (c + 1) % 25 == 0:
        print(f"  ...{c+1}/{len(idxs)}", flush=True)
h.remove()

# native AR-greedy (GLM-OCR architecture, KV-cached) — the fair "pure AR" speed baseline (fwd/token ~1)
# Excluding this arm via ARMS skips loading the second model entirely (tens of seconds + VRAM).
if not _want("ar_greedy_native"):
    print("[bench-fwd] ar_greedy_native skipped (ARMS filter)", flush=True)
else:
    import infer_odb_ar as arlib
    from transformers import GlmOcrForConditionalGeneration, AutoProcessor
    proc2 = AutoProcessor.from_pretrained(CKPT)
    glm = GlmOcrForConditionalGeneration.from_pretrained(CKPT, dtype=torch.bfloat16).to(dev).eval()
    arlib.patch_vision_fast(glm)
    ca = {"n": 0}; ha = glm.register_forward_hook(lambda *a, **k: ca.__setitem__("n", ca["n"] + 1))
    for idx in idxs:
        ex = ds[idx]; gt = ex["target"]; pr = ex.get("prompt", "Text Recognition:")
        # MAXNEW lifts the per-task cap (default 4096). Caps of 256/768 cut the long outputs where
        #   acceptance is highest and understate table/formula tok/fwd and tok/s.
        mnt = int(os.environ["MAXNEW"]) if os.environ.get("MAXNEW") else (
            256 if ex["task_type"] == "text" else (768 if ex["task_type"] == "table" else 256))
        # Refresh the ledger label here as well.
        # AR is the denominator of the spec/AR ratio table; a wrong label invalidates every per-type comparison.
        _cur_meta["task_type"] = ex["task_type"]; _cur_meta["idx"] = int(idx)
        if PROBE:
            _p0 = time.time()
            arlib.generate_ar(proc2, glm, ex["image"], pr, 1, dev, 99999)
            _cur_meta["probe"] = time.time() - _p0
        ca["n"] = 0; t0 = time.time()
        txt = arlib.generate_ar(proc2, glm, ex["image"], pr, mnt, dev, 99999)
        add("ar_greedy_native", ca["n"], ntok(txt), time.time() - t0, nedit(txt, gt))
        _cur_meta["probe"] = None
    ha.remove()

print(f"\n=== shard {SHARD_ID}/{NUM_SHARDS} raw sums ({len(idxs)} samples, single-stream per-GPU, no inter-GPU in timing) ===")
print(f"{'method':<26}{'tok/fwd':>9}{'fwd/100tok':>12}{'resp_tok':>10}{'wall/s':>9}{'edit':>8}")
for method, a in agg.items():
    tokfwd = a["toks"] / max(1, a["fwd"]); fwd100 = 100 * a["fwd"] / max(1, a["toks"])
    print(f"{method:<26}{tokfwd:>9.2f}{fwd100:>12.1f}{a['toks']/a['n']:>10.1f}{a['wall']/a['n']:>9.2f}{a['edit']/a['n']:>8.4f}")
base = os.environ.get("OUT", "checks/probe/bench_forwards.json")
out = base.replace(".json", f"_shard{SHARD_ID}.json") if NUM_SHARDS > 1 else base
# Create the output directory up front so a missing directory cannot lose a finished benchmark
# at save time.
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump(agg, open(out, "w"), indent=1)
# the ledger goes to a separate file (the aggregate format is unchanged)
_ps = out.replace(".json", "_persample.json")
json.dump(PERSAMPLE, open(_ps, "w"), indent=1)
print(f"per-sample ledger: {len(PERSAMPLE)} rows -> {_ps}", flush=True)
print(f"[bench-fwd-done] wrote {out}")
