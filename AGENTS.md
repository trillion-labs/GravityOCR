# AGENTS.md — how to run this repository

Operational reference for GravityOCR: environment, serving, in-process inference and evaluation, in
the order you would actually do them. Written so that a coding agent (Claude Code, Codex,
Cursor, …) or a person can follow it end to end; the README stays short and points here. Every command
below runs from the repository root. Deeper references live in `docs/` (start at `docs/00_INDEX.md`).

## 0. What is where

```
src/            model + decoders (models/glm_ocr_dllm.py, fast_sampling.py, fast_sampling_batched.py,
                infer_omnidocbench.py, infer_odb_ar.py, layout_postprocess.py)
scripts/        eval.sh (+ _lib.sh) — the evaluation launcher; it records every run into runs/<name>/
serve/          SGLang self-spec server (serve_sglang_ocr.sh) and the vLLM AR baseline (launch_vllm.sh, vllm_throughput.py)
patches/sglang/ the SGLang diff + the new worker file that implement self-speculative decoding
slurm/          the four slurm recipes: OmniDocBench inference (diffusion / AR), scoring, tok/fwd bench
tools/          6 measurement utilities: bench_forwards.py (tokens/forward), measure_tps_fair.py, http_ocr_throughput.py,
                precompute_odb_layout.py, score_breakdown.py, unimer_cdm.py
docs/           4 technical references — start at docs/00_INDEX.md
env/            pip freezes of the three environments used (sglang / vllm / scorer) + driver version
```

This repository runs the released model: inference, serving and evaluation. The training and RL
code, the data pipeline, side experiments, one-off analysis scripts and experiment logs are not
part of it; how the model was trained is summarized in `MODEL_CARD.md` and detailed in the paper.

## 1. Environment

Python 3.12, CUDA 12.8 driver (`env/DRIVER.txt`), H100. Two environment variables replace the
absolute paths of the original cluster wherever they appear in scripts or docs:

| variable | meaning | default |
|---|---|---|
| `DOCR_ROOT` | this repository's root | directory containing `scripts/` (slurm jobs: `$SLURM_SUBMIT_DIR`) |
| `HF_HOME` | Hugging Face cache | `~/.cache/huggingface` |
| `DOCR_WORKSPACE` | parent directory holding benchmarks / scorer checkouts. Appears as a literal `${DOCR_WORKSPACE}` placeholder in a few `tools/` and `docs/` examples — substitute your own paths there before use | — |
| `SGLANG_SRC` | your patched SGLang checkout (serving only) | — |
| `VAL_PAGES` | a `save_to_disk` dataset of region crops (`image, prompt, target, task_type`) for the speed tools (`tools/bench_forwards.py`, `tools/measure_tps_fair.py`, `tools/http_ocr_throughput.py`) | — |

```bash
uv sync                          # inference env (pyproject.toml, torch 2.9.1+cu128, transformers 5.x)
```

## 2. Serve (SGLang, self-speculative)

```bash
# SGLang v0.5.12 + our patch
git clone https://github.com/sgl-project/sglang $SGLANG_SRC && cd $SGLANG_SRC
git checkout $(cut -c1-40 $DOCR_ROOT/patches/sglang/sglang_BASE_COMMIT.txt)          # v0.5.12 base commit
git apply $DOCR_ROOT/patches/sglang/sglang_0.5.12_selfspec.patch
cp -r $DOCR_ROOT/patches/sglang/python .                                            # the new worker file
uv venv .venv-sglang --python 3.12 && uv pip install --python .venv-sglang/bin/python -r $DOCR_ROOT/env/sglang_venv_freeze.txt -e $SGLANG_SRC/python

# serve (MODE=spec = self-speculative, MODE=ar = plain autoregressive, same weights)
bash serve/serve_sglang_ocr.sh MODE=spec GPU=0 PORT=30500 CKPT=<path or trillionlabs/GravityOCR> CONTEXT=16384 VBIN=$PWD/.venv-sglang/bin
curl localhost:30500/v1/models          # OpenAI-compatible
```
Details, multi-GPU (`DP=8`), and the parity check (spec output ≡ AR output) are in `serve/SGLANG_SERVE.md`;
the implementation walkthrough is `docs/SGLANG_SELFSPEC_IMPL.md`. Concurrent-request throughput against
any OpenAI-compatible endpoint (ours, vLLM, …) with one identical client: `tools/http_ocr_throughput.py`.

## 3. In-process inference (no server)

```bash
python src/infer_omnidocbench.py --checkpoint <ckpt> --omnidocbench_gt OmniDocBench.json --images_dir <images> \
       --out_dir preds/ --max_long_side 99999 --spec_natcache      # self-spec; drop --spec_natcache for AR
```
`--spec` (prefix cache), `--spec_natcache` (incremental cache) and plain AR produce byte-identical text;
see `docs/DECODE_PATHS.md` for what each path caches. Self-speculation needs a checkpoint with a trained
AR path (`block_diffusion.json: ar_loss_weight > 0`).

## 4. Evaluate (OmniDocBench)

```bash
bash scripts/eval.sh MODEL=<ckpt dir> SCALE=full MODE=diffusion   # inference → official scoring → speed, one run folder
bash scripts/eval.sh MODEL=<ckpt dir> SCALE=full MODE=ar          # same pipeline through the plain AR path
```
Full resolution (`--max_long_side 99999`) is baked in. Prerequisites (symlinks are fine):
`$DOCR_ROOT/omnidocbench` → the [OmniDocBench](https://github.com/opendatalab/OmniDocBench) data
(`OmniDocBench.json` + `images/`), and `$DOCR_ROOT/scorer` → the OmniDocBench repository with its
scorer environment at `scorer/.venv-score` (`env/score_venv_freeze.txt`). Details: `docs/04_EVAL.md`.
Tokens-per-forward on a fixed crop set (`VAL_PAGES`): `tools/bench_forwards.py` (`slurm/bench_forwards.slurm`);
serving-side throughput: `tools/http_ocr_throughput.py` against the SGLang server. Speed numbers are only
comparable within one page set — always state it (`docs/DECODE_PATHS.md`).

## 5. Data

The training pool contains 12.3M region-level examples assembled from predominantly public data —
DocGenome, Docmatix, PubTables-1M, FinTabNet, SynthTabNet, PubTabNet, RVL-CDIP, DocLayNet, and the training
split of UniMER. Each example is a layout region cropped from a full page at native resolution, paired with a
text target: for full-page sources the regions come from the same PP-DocLayout-V3 detector used at inference,
and table- and formula-only sources are used as whole crops. Targets are transcriptions of each crop produced
by the base GLM-OCR, except for table crops from sources that ship cell-level annotations (about 39% of the
table stream), which use the original annotations converted to the evaluation markup convention. The three
streams — page text (6.49M), tables (2.90M), formulas (2.89M) — are predominantly English; the table and
formula streams are each subsampled once with a fixed seed to 2.16M for a 60/20/20 ratio, giving the 10.8M
globally shuffled training set.
