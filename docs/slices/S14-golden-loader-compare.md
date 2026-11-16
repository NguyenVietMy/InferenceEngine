# S14 — Golden loader + compare helper
**Phase:** B · **Target date:** Nov 30 · **Depends on:** S08, S13

## Goal
C++ can load any Python golden dump and report how far an output is from it.

## Steps
1. Write a `.bin` reader matching `ref/golden/FORMAT.md`: validate the magic, read dims, then read fp32 data into a host vector plus a shape.
2. Compare helper: returns max abs diff, max rel diff, and the **index of the worst element** with both values. That index is gold when debugging.
3. A GoogleTest-friendly assert that prints those details on failure.
4. Point tests at the golden directory via a CMake define or an env var. Don't hard-code absolute paths.

## Done when
- [ ] A test loads every file under `ref/golden/p0/` without error.
- [ ] A test deliberately corrupts one value and the helper reports the right index.

## Notes
