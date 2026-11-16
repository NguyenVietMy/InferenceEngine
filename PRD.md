# PRD — LLM Inference & Serving Engine (working name: `milo-infer`)

**Owner:** Milo Nguyen · **Started:** Nov 2026 · **Target v1:** early Aug 2027
**Hardware:** RTX 4090 Laptop (Ada, sm_89, 16 GB, ~576 GB/s peak DRAM bandwidth)

---

## 1. Summary
A single-GPU LLM inference and serving engine for Llama-architecture models, written from scratch in C++/CUDA, exposing an OpenAI-compatible streaming API. Benchmarked against llama.cpp and vLLM on the same hardware.

**Why:** to learn and prove the full inference stack — kernels → engine → serving — for AI-infra / inference roles in the Summer 2028 cycle (applications open ~Aug 2027).

## 2. Goals
- G1. Correct: token-for-token match with HF `transformers` under greedy decoding.
- G2. Fast: decode reaches a high fraction of peak memory bandwidth with my own kernels.
- G3. Fits: Llama 3.1 8B runs in 16 GB via INT4 weight quantization.
- G4. Serves: many concurrent streaming requests with continuous batching + paged KV cache.
- G5. Explained: every design choice and number documented (README + blog posts).

## 3. Non-goals
Training/fine-tuning · multi-GPU / tensor parallel · non-Llama architectures · beating vLLM · production hardening (auth, multi-tenancy).

## 4. Supported models
| Model | Precision | Weights size (approx.) | Purpose |
|---|---|---|---|
| Llama 3.2 1B | BF16 | ~2.5 GB | Dev / correctness |
| Llama 3.2 3B | BF16 | ~6.5 GB | Main FP16-class benchmark |
| Llama 3.1 8B | INT4 (group-wise) | ~5 GB | Quantization showcase |

## 5. Functional requirements
- **FR1 Loader:** mmap safetensors, parse config, upload weights to GPU.
- **FR2 Model:** embedding, RMSNorm, RoPE (Llama 3 scaling), GQA attention, SwiGLU MLP, LM head.
- **FR3 KV cache:** contiguous first (M1–M3), paged block allocator (M4).
- **FR4 Sampling:** greedy, temperature, top-k, top-p; deterministic seed.
- **FR5 Tokenizer:** Python (tiktoken/HF) bridge first; optional native C++ BPE later.
- **FR6 Kernels (own implementations):** RMSNorm, RoPE, softmax, decode attention, GEMV, GEMM (tensor cores), INT4/INT8 dequant-fused GEMV.
- **FR7 Quantization:** offline converter script (safetensors → INT8 / INT4 group-wise, group size 128).
- **FR8 Server:** `POST /v1/chat/completions` and `/v1/completions`, SSE streaming, `max_tokens`, stop sequences.
- **FR9 Scheduler:** continuous batching (iteration-level), admission control, backpressure, request cancellation.
- **FR10 Observability:** Prometheus metrics — queue depth, batch size, TTFT, inter-token latency, tokens/sec, KV blocks used.
- **FR11 CLI:** `infer --model <path> --prompt "..."` for single-request runs.

## 6. Performance targets (estimates — revise after M1 baseline)
Batch-1 decode is memory-bound, so theoretical max ≈ peak bandwidth ÷ bytes of weights read per token.

| Metric | Theoretical | v1 target |
|---|---|---|
| 1B BF16 decode, batch 1 | ~230 tok/s | ≥ 70% of peak BW (~160 tok/s) |
| 3B BF16 decode, batch 1 | ~90 tok/s | ≥ 70% of peak BW (~60 tok/s) |
| 8B INT4 decode, batch 1 | ~115 tok/s | ≥ 60% of peak BW (~70 tok/s) |
| Tensor-core GEMM (prefill) | card peak | ≥ 60% of cuBLAS |
| INT4 quality | — | < 5% perplexity increase vs BF16 |
| Serving throughput | — | ≥ 3× naive static batching at equal p99 TTFT |

## 7. Architecture
```
client ──HTTP/SSE──> server/  (API, request queue, metrics)
                       │
                       ▼
                 scheduler/   (continuous batching, paged KV block allocator)
                       │
                       ▼
                   engine/    (loader, model graph, sampler, KV cache)
                       │
                       ▼
                  kernels/    (.cu: norm, rope, attention, gemv, gemm, dequant)
ref/   PyTorch reference + golden activation dumps
bench/ load tester, llama.cpp / vLLM comparison, plots
```

## 8. Milestones

### M0 — Python reference · Nov 11 → Dec 5, 2026
- [ ] Karpathy *Zero to Hero* through "Let's build GPT"
- [ ] Llama forward pass in plain PyTorch (~200 lines), loading HF weights
- [ ] Dump per-layer activations for fixed prompts as golden test data
- **Exit:** logits match HF `transformers` (max abs diff < 1e-3); greedy output identical for 50 tokens.

### M1 — C++ CPU engine · Dec 6, 2026 → Jan 24, 2027 (winter break)
- [ ] CMake project, tensor struct, mmap safetensors loader
- [ ] All layers + KV cache + sampler on CPU
- [ ] Unit tests per layer vs golden dumps
- **Exit:** coherent generation from 1B; all layer tests pass; CPU tok/s baseline recorded.

### M2 — CUDA engine + custom kernels · Jan 25 → Mar 28, 2027
- [ ] PMPP ch. 1–6 + GPU MODE lectures
- [ ] Port to GPU: naive kernels + cuBLAS baseline
- [ ] Replace with own kernels: RMSNorm, RoPE, softmax, decode attention, GEMV
- [ ] Tensor-core GEMM for prefill (WMMA → mma.sync)
- [ ] Nsight Systems/Compute profile for every kernel; micro-benchmarks in `kernels/bench`
- **Exit:** 1B and 3B run fully on own kernels; decode ≥ 60% peak BW; still matches golden outputs.
- **Blog post #1:** "Taking my matmul from X% to Y% of cuBLAS."

### M3 — Quantization · Mar 29 → Apr 30, 2027
- [ ] Offline INT8 and INT4 group-wise converter
- [ ] Dequant-fused INT4/INT8 GEMV kernels
- [ ] Perplexity eval script (small WikiText subset)
- **Exit:** 8B INT4 runs in 16 GB; speedup and perplexity delta measured.
- **Blog post #2:** "Fitting Llama 8B on a laptop GPU."

### M4 — Serving layer · May 1 → Jun 30, 2027
- [ ] HTTP server with SSE streaming, OpenAI-compatible schema
- [ ] Paged KV cache block allocator
- [ ] Continuous batching scheduler + admission control + cancellation
- [ ] Batched decode kernels (attention over paged KV)
- [ ] Prometheus metrics + Grafana dashboard
- **Exit:** 32+ concurrent streams served correctly; throughput ≥ 3× static batching at equal p99 TTFT.

### M5 — Benchmarks, polish, writeup · Jul 1 → Aug 1, 2027
- [ ] Load-test harness: TTFT, ITL, throughput vs concurrency, p50/p99
- [ ] Same-hardware comparison vs llama.cpp and vLLM
- [ ] Roofline plots; README with architecture, results, design decisions
- [ ] 1–2 merged PRs to vLLM / SGLang / llama.cpp (in parallel, from June)
- **Exit:** public repo, README, results table, resume v2 ready.
- **Blog post #3:** "Building continuous batching and paged attention from scratch."

### Stretch (only if ahead)
Speculative decoding · FlashAttention-style prefill · FP8 GEMM · prefix caching · native C++ tokenizer.

## 9. Testing strategy
- Every kernel: unit test vs PyTorch reference (tolerance per dtype) **before** optimizing.
- Layer-level and end-to-end golden tests in CI (CPU build runs anywhere).
- Server: concurrency tests (cancellation, OOM on KV blocks, stop sequences).
- Benchmarks: warm-up runs, median of ≥ 20 runs, laptop plugged in + performance mode, clock/power logged.

## 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Scope creep | Non-goals list; stretch items only after M5 exit |
| Can't explain own code in interviews | Write core engine/kernels/scheduler myself; AI for explanation and review only |
| Laptop thermal throttling skews numbers | Fixed power mode, log clocks, report median + variance |
| Semester workload | ~8–10 hrs/week; winter break and summer carry M1 and M4 |
| Falling behind | Each milestone is a shippable demo; v1 is still strong if it stops at M3 |

## 11. Weekly cadence
- Mon: plan the week's 2–3 tasks · Sat: commit benchmark numbers + short dev log in `docs/log.md`
- End of each milestone: tag a release, update README results table.

## 12. Target resume bullets (fill in real numbers)
- Built a C++/CUDA LLM inference engine from scratch for Llama 3 models; custom decode GEMV and attention kernels reach __% of peak memory bandwidth on an RTX 4090.
- Implemented INT4 group-wise weight quantization with fused dequantization, running Llama 3.1 8B in 16 GB at __ tok/s (__× over BF16) with __% perplexity increase.
- Designed an OpenAI-compatible serving layer with continuous batching and a paged KV cache, sustaining __ concurrent streams at p99 TTFT __ ms and __× the throughput of static batching.

## 13. References
Karpathy *Neural Networks: Zero to Hero* · *Programming Massively Parallel Processors* (PMPP) · GPU MODE lectures + Discord · Simon Boehm, "How to Optimize a CUDA Matmul Kernel" · vLLM paper (PagedAttention) · FlashAttention papers · llama.cpp and vLLM source code.
