# S58 — README
**Phase:** F · **Target date:** Jan 13 · **Depends on:** S57

## Goal
The README is the project's front page for recruiters and interviewers.

## Sections
1. One-paragraph pitch + a GIF or asciinema of `chat.py` streaming.
2. Results table (from the CSV) + the 3 plots.
3. Architecture: the diagram from PLAN/PRD, updated to what exists.
4. Design decisions: fp32 activations / bf16 weights; GEMV for decode vs GEMM for prefill; contiguous KV; CUDA graphs; the tokenizer split. One short paragraph each, with the trade-off.
5. Correctness: how golden tests work.
6. Limitations + the post-v1 backlog.

## Done when
- [ ] Someone else could reproduce your numbers from it.

## Notes
