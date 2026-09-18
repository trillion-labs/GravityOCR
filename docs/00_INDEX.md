# docs — index

Start with the repository `README.md`. Commands assume the environment variables described in
README §Environment (`DOCR_ROOT`, `HF_HOME`, `VAL_PAGES`, `SGLANG_SRC`).

- `04_EVAL.md` — the OmniDocBench evaluation pass (`scripts/eval.sh`: inference → official scoring → speed)
- `09_CODE_MAP.md` — map of `src/` and the inference / eval data flow
- `DECODE_PATHS.md` — which generation path is cached / batched, and how speed is measured
- `SGLANG_SELFSPEC_IMPL.md` — how self-speculative block-diffusion decoding is implemented in SGLang
  (companion to `../patches/sglang/`)
- `../serve/SGLANG_SERVE.md` — serving the model with SGLang (spec and AR modes, parity check, throughput client)
