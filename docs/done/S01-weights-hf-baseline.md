# S01 — Weights + HF baseline
**Phase:** A (Python reference) · **Target date:** Nov 17 · **Depends on:** —

## Goal
Get the model weights onto disk and record what "correct output" looks like, straight from HF `transformers`.

## Why
Every later test eventually traces back to these token IDs. If they're wrong or not reproducible, everything downstream is built on sand.

## Steps
1. Request access to `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` on Hugging Face (gated; approval can take hours).
2. `hf auth login`, then `hf download <repo> --local-dir ~/models/<name>` for both. Keep weights **out of the repo**.
3. Write `ref/hf_generate.py`:
   - Load the 1B in **fp32** (`dtype=torch.float32`). fp32 is your source of truth for numerics.
   - Build one fixed chat prompt with `apply_chat_template(..., add_generation_prompt=True)`.
   - Greedy generate 50 new tokens (`do_sample=False`).
   - Print the decoded text.
   - Save **both** the prompt IDs and the generated IDs to `ref/golden/hf_greedy_1b.json`.
4. Run it twice and diff the JSON outputs.

## Done when
- [X] Coherent text is printed.
- [X] `ref/golden/hf_greedy_1b.json` has `prompt_ids` and `output_ids` (50).
- [X] Two runs produce byte-identical JSON.

## Pitfalls
- The Llama 3.2 chat template injects **today's date** into the system header. Either pass a fixed date to the template or always reuse the saved `prompt_ids`. Otherwise your golden output changes every day.
- Run on the GPU if you like, but fp32 on CPU is fine for 1B and more deterministic.

## Resources
- HF docs: `generate()`, chat templates.

## Notes
<!-- prompt used, time taken, anything surprising -->
