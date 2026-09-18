# Evaluation (OmniDocBench)

One command runs the whole pass — inference → official scoring → throughput / tokens-per-second /
tokens-per-forward — into a single self-contained run folder. Submit from a node that has `sbatch`.

## Prerequisites

- `$DOCR_ROOT/omnidocbench` → the OmniDocBench data (`OmniDocBench.json` + `images/`), from the
  [official repository](https://github.com/opendatalab/OmniDocBench).
- `$DOCR_ROOT/scorer` → a checkout of the OmniDocBench repository with its scorer environment at
  `scorer/.venv-score` (`env/score_venv_freeze.txt`; TEDS + CDM need the repository's extra
  dependencies and a TeX Live for CDM).
- `ODB_GT_200=<json>` only if you use `SCALE=200` (a fixed 200-page subset of the GT json for sweeps).

## One command

```bash
cd $DOCR_ROOT
bash scripts/eval.sh MODEL=<hf checkpoint dir> SCALE=full                 # diffusion / self-spec checkpoint
bash scripts/eval.sh MODE=ar MODEL=zai-org/GLM-OCR SCALE=full NAME=base   # plain AR (any HF checkpoint)
```

Options: `MODE=diffusion|ar` · `SCALE=full|200` (default full = the benchmark) · `THR=0.99` · `BS=8` ·
`LIMIT=0` · `NAME=<label>` · `STEP=<n>` · `TOKFWD=1|0` (also run the tok/fwd bench; needs `VAL_PAGES`).

`LIMIT=<n>` stops **inference** after n pages; the scorer still reads the whole ground-truth file, so the
missing pages count as empty and the score is meaningless as a benchmark. It is a plumbing smoke test
only, and the wrapper tags its run folder `-limit<n>` so it can never be mistaken for the real run. For a
genuine subset number use a ground-truth file that contains exactly those pages (`SCALE=200`
with `ODB_GT_200`).

## What it produces

`runs/<NAME>__s<STEP>__<MODE>__thr<THR>__<evalset>/`:

```
config.json    model, ckpt, step, mode, threshold, eval_set, gt, cmd, date
launch/        launch.txt (command + args + resolved env) + copies of the slurm scripts   <- reproducible
predictions/   *.md  (OmniDocBench prediction dir)
score.json     {overall, text_block, display_formula, table, reading_order, table_TEDS}
speed.json     {throughput_pg_s, tps_tok_s, pages, out_tokens, wall_s, tok_per_fwd, tok_per_fwd_by_method}
score_raw.json the raw scorer metric json (provenance) ;  score_config.yaml the scorer config used
STATUS.txt     one-glance summary
```

Three slurm jobs: **inference** (GPU, 16-way page-sharded) → **finalize** (CPU, `afterok`: official
scorer + `measure_eval_perf`) → **tok/fwd bench** (GPU, independent). When all finish:
`cat runs/<run>/score.json runs/<run>/speed.json`.

## Rules baked into the wrappers

- **Full resolution** `--max_long_side 99999`. The inference default (768) silently downscales pages.
- **Scale discipline.** `full` (1651 pages) is the headline number; `200` is for sweeps. The scorer GT
  always matches the inferred set. Never compare a 200-page number with a full one.
- **TEDS** only on `SCALE=200` in `score_finalize.slurm` — the full-set TEDS matcher is extremely slow on
  some pages; full-set table scoring there uses `Edit_dist`. The paper's official numbers (Overall =
  mean of text / table TEDS / formula CDM, page-averaged) come from the official OmniDocBench scorer run
  on the prediction directory with its default end-to-end config.
- **Idempotent.** The run folder name is deterministic for (model, step, mode, thr, scale); if it already
  holds a complete `score.json` + `speed.json` the command prints them and stops. `FORCE=1` re-runs.

## Standalone pieces

- Re-score an existing run: `sbatch --export=ALL,RD=runs/<run>,GT=<gt>,SCALE=<full|200>,INFER_LOG=<.out> slurm/score_finalize.slurm`
- tok/fwd bench only: `VAL_PAGES=<crop dataset> sbatch --export=ALL,CKPT=<ckpt>,N=100,RD=runs/<run> slurm/bench_forwards.slurm`

## Speculative variants of `infer_omnidocbench.py`

`--spec` (prefix cache), `--spec_natcache` (incremental cache), `--spec_uncached` (reference) all
produce AR-greedy output and differ only in speed; see `docs/DECODE_PATHS.md`.
