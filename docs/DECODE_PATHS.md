# Decode paths — what is cached, what is batch-safe, what each may claim

If a number's decode path is not in this table, do not trust it. Mirrored in the header of
`src/fast_sampling.py`.

## Block-diffusion checkpoint (`GlmOcrBlockDiffusion`)

| function | what is cached | batch | criterion | equivalence |
|---|---|---|---|---|
| `model.generate` | nothing — recomputes the full sequence every denoise step | bs 1 | confidence | the slow **reference** (correct by construction) |
| `fast_sampling.generate_cached` | full block-causal KV cache: prefix + all committed blocks; each new block forwards only `bd` tokens | bs 1 | confidence | cached diffusion |
| `fast_sampling_batched.generate_cached_batched` (`scripts/eval.sh` `--fast`) | same cache, batched (ragged prefixes left-padded) | bs > 1 | confidence | equal to bs 1 only up to floating point (near-tie argmax may flip) |
| `model.generate_speculative` | nothing (uncached spec reference) | bs 1 | diffusion draft + AR verify | reference spec |
| `fast_sampling.generate_speculative_prefixcache` (`--spec`) | prefix (vision) only; re-forwards the committed response each round, O(N²) | bs 1 | spec | **byte-identical to AR greedy** |
| `fast_sampling.generate_speculative_natcache` (`--spec_natcache`) | full committed cache, token-causal incremental, 2 forwards per round | bs 1 | spec | score equals AR greedy (bf16 cache drift can change a token on very long pages) |
| `generate_speculative_cached`, `generate_speculative_onecache` | (buggy) | — | — | **superseded / failed — never use** (`--spec_oldcached`, A/B only) |

"Prefix cache" = vision only = `prefixcache`. "Real cache" (committed tokens cached too) =
`generate_cached`, `generate_cached_batched`, `natcache`.

All three self-speculative paths run the same round (`full_draft=True`, the default): the draft window is the
boundary token plus `bd` masks and every mask logit is used, so the draft is `a0, d_1..d_bd` (the last mask's
logit predicts the token right after the block); the verify forward runs causally over
`[committed | a0, d_1..d_bd]` and, when the whole draft is accepted, also commits the verifier's bonus token —
up to `bd + 2` tokens per round. `full_draft=False` reproduces the earlier round (last mask logit unused, no
bonus, at most `bd` tokens). Output is identical either way; only tokens per forward changes (about +1–3%).
`SELFSPEC_FULLDRAFT` is the same switch in the SGLang worker (`docs/SGLANG_SELFSPEC_IMPL.md` §2).

## Can diffusion be scored with a larger batch? — no

- **AR**: `--batch_size 8` is verified equal to bs 1 → safe to scale.
- **Diffusion (`--fast` = `generate_cached_batched`)**: equal to bs 1 only up to floating point; a direct
  check (12 full-resolution pages) found 3 pages differing by a few characters under ragged left-padding.
  Keep a **fixed** batch size for every scored diffusion run (the released numbers use bs 8) and vary it
  only for speed-only measurements.

## How speed is measured

`scripts/eval.sh` launches both into the run folder's `speed.json`:
- **tokens/forward (accept efficiency)** — `tools/bench_forwards.py` counts decoder forwards with a
  forward hook on the first decoder layer (exact, hardware-independent), seeded random N pages, single
  GPU, `draft_steps=1`. Always `torch.cuda.synchronize()` around wall timing.
- **pages/s and tokens/s** — `src/measure_eval_perf.py` from a timed inference run (`WALL_SECONDS` printed
  by `slurm/eval_omnidocbench.slurm`).

A speed number is meaningless without (method, cached?, batch size, page set, synchronized?). Record all
of them. tok/fwd in particular is bounded by output length (a round cannot commit more tokens than the
output has left), so it is only comparable within one page set. A tok/fwd below 1 means the wrong decode
path was used for the checkpoint, not a result.
