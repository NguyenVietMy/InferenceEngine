# S10 — 3B parity + tolerance study
**Phase:** A · **Target date:** Nov 26 · **Depends on:** S09

## Goal
Prove your ref generalizes to 3B, and decide on numerical tolerances using data instead of guesses.

## Steps
1. Run your ref on 3B (head_dim 128, 28 layers, 24 heads). Fix anything that hard-coded 1B shapes.
2. Get the HF 3B greedy golden output (as in S01) and compare 50 tokens.
3. Tolerance study on 1B:
   - HF in fp32 vs HF in bf16: max abs diff of logits.
   - Your ref vs HF fp32: max abs diff of logits.
   - The top-1 vs top-2 logit gap across the 50 greedy steps. How close are the near-ties?
4. Write `docs/notes/tolerances.md`: per-op tolerances for C++ tests (elementwise ops, matmuls, full logits), and why.

## Done when
- [ ] 3B: 50 greedy tokens match HF.
- [ ] `tolerances.md` has concrete numbers with justification.
- [ ] 3B golden dumps are generated (re-run S08's script with a model flag).
- [ ] Phase A is done 🎉 Tag `m0-ref`.

## Pitfalls
- The 3B checkpoint is **sharded** (2 files + an index JSON). Your loader needs to handle that, here and in C++ (S32).
- 3B in fp32 is ~13 GB of RAM, so check WSL's memory limit (`.wslconfig`).

## Notes
