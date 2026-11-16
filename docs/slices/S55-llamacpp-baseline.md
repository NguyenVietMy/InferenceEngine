# S55 — llama.cpp baseline
**Phase:** F · **Target date:** Jan 10 · **Depends on:** S53

## Goal
Compare against the strongest batch-1 engine you can run on this laptop.

## Steps
1. Build llama.cpp with CUDA (`-DGGML_CUDA=ON`), outside this repo.
2. Convert the model: `convert_hf_to_gguf.py <model> --outtype bf16`. Use **bf16**, so precision matches yours.
3. Use `llama-bench` with `-p 128,512 -n 256 -r 20` (or the equivalent); parse its output into your CSV.
4. Note versions and the commit hash.

## Done when
- [ ] llama.cpp rows are in the CSV, with setup caveats noted.

## Notes
