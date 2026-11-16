# v1 Plan — 60 slices

**Cadence:** 1 slice/day, ~3 hrs each. Start **Tue Nov 17, 2026** → S60 lands **~Jan 15, 2027**. Jan 16–31 is buffer (holidays, bad days, slices that run long).
**Rule:** I hand-write all engine/kernel code. AI is for explaining, reviewing, and debugging hints only.

This plan supersedes the PRD's scope where they disagree (see "Scope decisions").
Each slice has a detail file in [`docs/slices/`](slices/) (`S01-*.md` … `S60-*.md`) with steps, done-when checks, pitfalls, resources, and a Notes section to fill in.

---

## Scope decisions (locked Nov 16, 2026)

| Topic | v1 decision | Cut / deferred |
|---|---|---|
| Finish line | Python ref → CUDA engine with own kernels → CLI | Quantization (M3), serving layer (M4) |
| Models | Llama 3.2 **1B-Instruct** + **3B-Instruct**, BF16 weights | 8B INT4 |
| CPU engine | **None.** Go straight from PyTorch to CUDA | PRD M1 |
| Numerics | BF16 weights, **FP32 activations + accumulation** | BF16 activations, FP8 |
| Kernels (own) | embedding, RMSNorm, RoPE, attention (+softmax), SwiGLU, GEMV (decode), tiled SIMT GEMM (prefill), argmax | Tensor-core GEMM (WMMA/mma.sync), FlashAttention prefill |
| KV cache | Contiguous, FP32 | Paged |
| Tokenizer | Python wrapper (HF tokenizer) ↔ C++ binary speaks token IDs | Native C++ BPE |
| Sampling | greedy, temperature, top-k, top-p; seeded | — |
| Tests | GoogleTest + raw `.bin` golden dumps from Python | pybind11 |
| Perf target | 1B decode batch 1 ≥ **50% peak BW (~115 tok/s)** | 70%+ |
| Comparison | HF transformers + llama.cpp | vLLM |
| Writing | README + 1 blog post | Posts #2, #3 |
| Study | Targeted, inside slices (only what the next slice needs) | Full Zero to Hero / PMPP |

**Model reference numbers**
- 1B: hidden 2048 · 16 layers · 32 q heads / 8 kv heads · head_dim 64 · MLP 8192 · vocab 128256 · tied embeddings.
- 3B: hidden 3072 · 28 layers · 24 q / 8 kv · head_dim 128 · MLP 8192 · tied embeddings.
- RoPE: θ = 500000, Llama-3 scaling (factor 32, low 1, high 4, orig ctx 8192).

---

## Phase A — Python reference (S01–S10) · target done Nov 26

- [ ] **S01 — Weights + HF baseline.** Get HF access, download 1B/3B-Instruct to `~/models` (outside the repo). `ref/hf_generate.py`: greedy, 50 tokens, fixed chat prompt.
  ✅ Coherent text printed; token IDs saved to `ref/golden/hf_greedy_1b.json`.
- [ ] **S02 — Study: architecture.** Karpathy "Let's build GPT" (the attention parts). `ref/list_tensors.py` prints every safetensors tensor + shape; `docs/notes/llama-arch.md` explains each one.
  ✅ You can explain every tensor in the 1B checkpoint, including why k_proj is 512×2048.
- [ ] **S03 — Ref: load + embedding + RMSNorm.** `ref/model.py`: load bf16 → fp32 (lossless) into a dict. Implement embedding + RMSNorm.
  ✅ Matches HF's modules on the same input, max abs diff < 1e-5.
- [ ] **S04 — Ref: RoPE with Llama-3 scaling.**
  ✅ Your inv_freq and rotated q/k match HF's rotary embedding, diff < 1e-5.
- [ ] **S05 — Ref: GQA attention, causal mask, no cache.**
  ✅ Layer-0 attention output matches HF's layer-0 self_attn on the same input, diff < 1e-4.
- [ ] **S06 — Ref: SwiGLU MLP + full forward → logits.**
  ✅ Logits vs HF (fp32) max abs diff < 1e-3 on the fixed prompt.
- [ ] **S07 — Ref: KV cache + greedy loop.**
  ✅ 50 greedy tokens identical to S01's IDs. *(M0 exit)*
- [ ] **S08 — Golden dump format + dumps.** `.bin` = magic, ndim, dims (int32), fp32 data. Writer + reader with a roundtrip test. Dump for 2 prompts: embed out; per layer: attn_norm, q/k/v post-RoPE, attn (pre-o_proj), attn block out, mlp out; final norm; logits.
  ✅ Roundtrip is exact; all files present under `ref/golden/` (gitignored if large).
- [ ] **S09 — Ref sampler.** `sample(logits, temp, top_k, top_p, u)` where `u` is a uniform in [0,1), which decouples sampling from the RNG. Dump ~50 (logits, params, u) → token cases.
  ✅ Cases cover greedy, temp only, top-k only, top-p only, and combined; edge cases include top_p=1, k=1, and ties.
- [ ] **S10 — 3B parity + tolerance study.** Run the ref on 3B. Compare HF bf16 vs fp32 logits so you know what "close" means.
  ✅ 3B greedy matches HF for 50 tokens; the tolerances you'll use in C++ are written down in `docs/notes/tolerances.md`.

## Phase B — C++/CUDA foundations (S11–S20) · target done Dec 6

- [ ] **S11 — CMake skeleton.** CUDA enabled, `CMAKE_CUDA_ARCHITECTURES 89`, targets: `engine` lib, `infer` exe, `tests` (GoogleTest via FetchContent). One vector-add kernel + one test.
  ✅ `cmake --build build && ctest --test-dir build` is green.
- [ ] **S12 — Study: PMPP ch. 2–3.** Write a `CUDA_CHECK` macro, a 2D matrix-add kernel, and cudaEvent timing.
  ✅ Tests pass; vector-add bandwidth at N=2^26 is measured and logged (expect >400 GB/s).
- [ ] **S13 — Tensor + device buffer.** An RAII `DeviceBuffer`. A `Tensor` with dtype {f32, bf16}, shape (≤4D), and a pointer; owning vs view.
  ✅ Unit tests for alloc/free and H2D→D2H roundtrip pass; `compute-sanitizer` is clean.
- [ ] **S14 — Golden loader + compare helper.** A C++ `.bin` reader. `max_abs_diff` / `max_rel_diff` helpers with readable failure messages.
  ✅ Loads every S08 file; the helper catches a deliberately corrupted value.
- [ ] **S15 — safetensors parser.** mmap the file, parse the 8-byte length + JSON header (nlohmann/json via FetchContent).
  ✅ Prints the same tensor list/dtypes/shapes/offsets as S02's script.
- [ ] **S16 — Config + weight upload.** Parse `config.json` into a struct. Upload all weights to the GPU as bf16 into a `Weights` struct with per-layer fields.
  ✅ 1B loads using ~2.5 GB (check nvidia-smi); embed row 0 downloaded back matches Python bit-for-bit.
- [ ] **S17 — Study: PMPP ch. 4–5 + reductions.** Sum-reduction kernel: naive → shared memory → warp shuffle.
  ✅ All 3 versions are correct vs CPU; GB/s for each is logged.
- [ ] **S18 — Kernel: embedding lookup** (bf16 table → fp32 rows).
  ✅ Matches golden embed out exactly.
- [ ] **S19 — Kernel: RMSNorm** (one block per row, shuffle reduction).
  ✅ Matches golden layer-0 attn_norm within tolerance.
- [ ] **S20 — Kernel: RoPE** (cos/sin table precomputed on the host with Llama-3 scaling).
  ✅ Applied to golden pre-RoPE q/k, it matches golden post-RoPE q/k.

## Phase C — Correct forward pass, naive kernels (S21–S32) · target done Dec 18

- [ ] **S21 — Naive matmul.** `Y[T,out] = X[T,in] · Wᵀ`, W bf16, one thread per output element.
  ✅ Matches a CPU reference for T=1 and T=7 on random data, including non-multiple-of-32 sizes.
- [ ] **S22 — QKV projection + RoPE wired.**
  ✅ Layer-0 q/k/v match golden.
- [ ] **S23 — KV cache.** Contiguous fp32 `[layer][pos][kv_head][head_dim]` for K and V; write at position.
  ✅ Write/readback test; sizing math for 1B/3B at 4096 ctx is documented.
- [ ] **S24 — Naive attention kernel.** QKᵀ/√d, causal mask, softmax, ·V, GQA head mapping. One block per (token, q_head).
  ✅ Layer-0 attn (pre-o_proj) matches golden for a multi-token prompt.
- [ ] **S25 — o_proj + residual add.**
  ✅ Layer-0 attention-block output matches golden.
- [ ] **S26 — SwiGLU kernel + MLP block + residual.**
  ✅ Layer-0 MLP out matches golden.
- [ ] **S27 — Full forward.** All layers, final norm, tied LM head.
  ✅ Every layer's output matches golden; logits within the S10 tolerance.
- [ ] **S28 — Activation arena.** Preallocate all activation/scratch buffers once.
  ✅ The forward does zero cudaMalloc calls (counter/assert); compute-sanitizer is clean on a full forward.
- [ ] **S29 — Prefill + decode loop with greedy argmax (host).** 🎯
  ✅ 50 tokens identical to HF golden IDs.
- [ ] **S30 — Tokenizer bridge.** `infer --model DIR` reads prompt IDs on stdin and streams one ID per line to stdout. `scripts/chat.py` applies the chat template, encodes, runs `infer`, and decodes incrementally.
  ✅ `python scripts/chat.py "What is the capital of France?"` streams a coherent answer.
- [ ] **S31 — Stop conditions.** EOS IDs from `generation_config.json`, `--max-tokens`, context-length guard.
  ✅ Stops cleanly at `<|eot_id|>`; hitting max ctx gives a clear error, not a crash.
- [ ] **S32 — 3B support.** No hard-coded 1B shapes (head_dim 128!).
  ✅ 3B greedy matches HF for 50 tokens.

## Phase D — Sampling + baseline (S33–S35) · target done Dec 21

- [ ] **S33 — C++ sampler: temperature + top-k** on host logits, `sample(logits, params, u)`.
  ✅ Passes every S09 case for those modes.
- [ ] **S34 — top-p + seeded RNG + CLI flags** (`--temp --top-k --top-p --seed`).
  ✅ All S09 cases pass; the same seed gives identical output across 3 runs.
- [ ] **S35 — Bench mode + naive baseline.** `infer --bench`: warm-up, then decode N tokens, median of 20, tok/s, % of peak BW (weight bytes / time/token), GPU clock from nvidia-smi.
  ✅ Naive 1B and 3B numbers are in `docs/log.md`. Tag `v0.1-correct`.

## Phase E — Make it fast (S36–S52) · target done Jan 7

- [ ] **S36 — Study: Nsight Systems + Compute** (check WSL support; fall back to nsys + cudaEvents if ncu misbehaves). Profile one decode step.
  ✅ Kernel-time breakdown table in `docs/log.md`.
- [ ] **S37 — Kernel microbench harness** (`kernels/bench`): run one kernel at real model shapes, report µs and GB/s.
  ✅ Naive matmul numbers at 2048→2048, 2048→8192, 8192→2048, 2048→128256.
- [ ] **S38 — GEMV v1:** one warp per output row, coalesced loads, shuffle reduce.
  ✅ Correct; ≥ 50% peak BW on 2048→8192.
- [ ] **S39 — GEMV v2:** 128-bit vectorized loads (8 bf16), multiple rows/block, x staged once.
  ✅ Correct; ≥ 75% peak BW on 2048→8192.
- [ ] **S40 — Use GEMV for T=1, matmul for T>1.**
  ✅ All golden + greedy tests pass; new tok/s logged.
- [ ] **S41 — Fuse weights:** concat q/k/v → one GEMV; gate/up → one GEMV.
  ✅ Tests pass; fewer launches per step (nsys); tok/s logged.
- [ ] **S42 — Decode attention v2:** split over sequence with online softmax; one block handles a kv head's whole q-group.
  ✅ Matches naive at seq len 1…4096 (random tests); microbench speedup at 1k/4k ctx logged.
- [ ] **S43 — Fused residual-add + RMSNorm.**
  ✅ Tests pass; tok/s logged.
- [ ] **S44 — GPU argmax** (reuse the S17 reduction). Greedy copies 4 bytes/token instead of 512 KB.
  ✅ Same tokens as host argmax; tok/s logged.
- [ ] **S45 — CUDA Graph for the decode step** (position/seq-len read from device memory).
  ✅ Graph and non-graph outputs are identical; launch gaps gone in nsys; tok/s logged.
- [ ] **S46 — Study: tiled matmul** (PMPP tiling chapter + Boehm's post, kernels 1–3). Add cuBLAS to the microbench only, as a reference.
  ✅ Table of naive vs cublasSgemm vs cuBLAS-BF16 GFLOPS at prefill shapes (T=512).
- [ ] **S47 — Tiled GEMM v1:** shared-memory tiles.
  ✅ Correct; GFLOPS logged.
- [ ] **S48 — Tiled GEMM v2:** 2D register blocking (each thread computes a TM×TN tile).
  ✅ Correct; ≥ 50% of cublasSgemm (fair SIMT comparison; the tensor-core number is reported separately).
- [ ] **S49 — Use the tiled GEMM for prefill.**
  ✅ Tests pass; TTFT for a 512-token prompt before/after is logged.
- [ ] **S50 — Long-context correctness.** ~1500-token prompt, 200 tokens of generation.
  ✅ Matches the fp32 ref for ≥100 tokens, or any divergence is at a near-tie (top-2 logit gap < tolerance) and explained.
- [ ] **S51 — Re-profile, kill the #1 remaining bottleneck.**
  ✅ A before/after breakdown is logged.
- [ ] **S52 — Perf target slice.**
  ✅ 1B decode ≥ 115 tok/s (≥ 50% peak BW). If already there: tune 3B. If not: a written explanation of where the time goes.

## Phase F — Benchmarks + writeup (S53–S60) · target done Jan 15

- [ ] **S53 — `bench/run.py`.** Fixed prompts (128/512 tokens in, 256 out), 20 runs, median/p90, clocks and power logged, CSV out.
  ✅ CSV rows for 1B and 3B.
- [ ] **S54 — HF transformers baseline** (bf16, same prompts, same harness).
  ✅ Rows in the same CSV.
- [ ] **S55 — llama.cpp baseline.** Build with CUDA, convert to GGUF BF16, run the same workload.
  ✅ Rows in the same CSV; any setup caveats are noted.
- [ ] **S56 — Plots.** Achieved GB/s per kernel vs peak (roofline-style), tok/s vs baselines, before/after kernel breakdown.
  ✅ PNGs in `docs/img/`.
- [ ] **S57 — Cleanup + fresh-clone test.** Dead code out, `--help`, consistent errors.
  ✅ Fresh clone in a temp dir → build → test → run, following only the README quickstart.
- [ ] **S58 — README.** Architecture, design decisions (why fp32 activations, why GEMV for decode, …), results table, limitations.
  ✅ Someone else could reproduce your numbers from it.
- [ ] **S59 — Blog post draft:** "From PyTorch to my own CUDA kernels: a Llama engine in 60 days".
  ✅ Complete draft with real numbers and plots.
- [ ] **S60 — Ship.** Final post, tag `v1.0`, fill the PRD resume bullets with real numbers.
  ✅ Public repo + post live.

---

## Post-v1 backlog (do NOT start before S60)
Tensor-core GEMM (WMMA → mma.sync) · INT4/INT8 quantization + 8B · paged KV + continuous batching + HTTP server · native C++ BPE · vLLM comparison · speculative decoding.
