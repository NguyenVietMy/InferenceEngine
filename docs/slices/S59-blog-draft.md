# S59 — Blog post draft
**Phase:** F · **Target date:** Jan 14 · **Depends on:** S58

## Goal
A complete draft: "From PyTorch to my own CUDA kernels: a Llama engine in 60 days."

## Suggested outline
1. Why build this; what batch-1 decode is bound by (bandwidth math).
2. Correctness first: the golden-dump approach.
3. The optimization ladder: naive → GEMV v1 → v2 → fusion → graphs. Numbers and the plot.
4. One deep dive: the GEMV or decode attention, with what surprised you.
5. Results vs HF and llama.cpp, with honest caveats.
6. What's next: tensor cores, INT4, serving.

## Done when
- [ ] A full draft with real numbers and plots embedded, readable start to finish.

## Notes
