# S11 — CMake skeleton
**Phase:** B (C++/CUDA foundations) · **Target date:** Nov 27 · **Depends on:** —

## Goal
A build that compiles CUDA for your GPU and runs GoogleTest, before any real code exists.

## Steps
1. Top-level `CMakeLists.txt`: `project(... LANGUAGES CXX CUDA)`, C++17 (or 20), `CMAKE_CUDA_ARCHITECTURES 89`.
2. Targets:
   - `engine`: a static library (`engine/`, `kernels/`).
   - `infer`: an executable linking `engine`.
   - `tests`: GoogleTest via `FetchContent`, plus `gtest_discover_tests`.
3. Write one trivial kernel (vector add) in `kernels/` and one test that runs it and checks the result.
4. Add `build/` to `.gitignore`. Write a one-line build command in the README or `docs/log.md`.

## Done when
- [ ] `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build` passes.
- [ ] `nvcc --version` and the driver version are recorded in `docs/log.md`.

## Pitfalls
- If CMake can't find nvcc, set `CMAKE_CUDA_COMPILER` or put `/usr/local/cuda/bin` on PATH.
- `.cu` files compile with nvcc and `.cpp` files with g++. Keep CUDA-specific headers out of headers that `.cpp` files include, or guard them.

## Resources
- "An Introduction to Modern CMake" (cliutils.gitlab.io), CUDA section.

## Notes
