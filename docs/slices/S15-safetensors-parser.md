# S15 — safetensors parser
**Phase:** B · **Target date:** Dec 1 · **Depends on:** S11

## Goal
Read a `.safetensors` file in C++ without copying it into RAM.

## Format
- The first 8 bytes are a little-endian `uint64` N: the header length.
- The next N bytes are JSON: `{ name: { dtype, shape, data_offsets: [begin, end] }, "__metadata__": {...} }`.
- Offsets are relative to the start of the data section (byte 8 + N).

## Steps
1. `mmap` the file read-only.
2. Parse the header with nlohmann/json (FetchContent). This is plumbing, not learning, so using a library here is fine.
3. Build `name → {dtype, shape, const void* data, nbytes}`.
4. Check `end - begin == numel × sizeof(dtype)` for every tensor.
5. A small debug CLI or test prints the table.

## Done when
- [ ] The output matches `ref/list_tensors.py` (146 tensors for 1B, same shapes and dtypes).
- [ ] The size check passes for every tensor.

## Pitfalls
- Skip the `__metadata__` key.
- Keep the mmap alive as long as any pointer into it is in use.

## Notes
