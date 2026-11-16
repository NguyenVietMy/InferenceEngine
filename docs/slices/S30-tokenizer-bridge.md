# S30 — Tokenizer bridge (CLI + Python wrapper)
**Phase:** C · **Target date:** Dec 16 · **Depends on:** S29

## Goal
Type text, see text, while the C++ binary speaks only token IDs.

## Design
- `infer --model DIR [--max-tokens N]`: reads prompt IDs from stdin (space-separated) and writes each generated ID to stdout **as soon as it's produced**, one per line, flushing each time.
- `scripts/chat.py "message"`: applies the chat template with the HF tokenizer, spawns `infer` via `subprocess`, pipes the IDs in, reads IDs line by line, and prints text incrementally.

## Steps
1. Parse arguments in `infer`. A tiny hand-rolled parser is fine.
2. Write the wrapper. Decode incrementally: decode the full ID list so far and print only the new suffix. Decoding a single ID breaks multi-byte UTF-8.
3. Send logs and errors from `infer` to **stderr** so they don't corrupt the ID stream.

## Done when
- [ ] `python scripts/chat.py "What is the capital of France?"` streams a sensible answer, word by word.

## Notes
