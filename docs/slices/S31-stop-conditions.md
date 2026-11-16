# S31 — Stop conditions
**Phase:** C · **Target date:** Dec 17 · **Depends on:** S30

## Goal
Generation stops for the right reasons and never crashes at the edges.

## Steps
1. Read `eos_token_id` from `generation_config.json`. It's a **list** for Llama 3.2 Instruct (it includes `<|eot_id|>` = 128009).
2. Stop on any EOS, on `--max-tokens`, or when `pos` would exceed `max_seq`. The last case gets a clear error message.
3. Reject an empty prompt or a prompt longer than max_seq with a clear error.

## Done when
- [ ] A chat answer ends cleanly at `<|eot_id|>` (not printed).
- [ ] A test with `--max-seq` set tiny produces an error message, not a segfault or sanitizer error.

## Notes
