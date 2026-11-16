# S32 — 3B support
**Phase:** C · **Target date:** Dec 18 · **Depends on:** S31

## Goal
The same engine runs the 3B model. This catches every hard-coded 1B assumption.

## Steps
1. Sharded loading: read `model.safetensors.index.json` and mmap each shard.
2. Grep your code for `64`, `2048`, `32`, `16`. Every shape must come from Config.
3. Kernels with head_dim-dependent block sizes must handle 128.
4. Run 3B golden layer tests and greedy generation.

## Done when
- [ ] 3B: all layer goldens pass; 50 greedy tokens match HF.
- [ ] Phase C is done 🎉 Tag it.

## Notes
