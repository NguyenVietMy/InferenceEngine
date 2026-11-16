# S22 — QKV projection + RoPE, wired up
**Phase:** C · **Target date:** Dec 8 · **Depends on:** S20, S21

## Goal
Start the model's forward function: norm → q/k/v projections → RoPE, for layer 0.

## Steps
1. Create `engine/model.cu` (or similar), with a function that takes the input hidden states and a layer index and writes q, k, v.
2. Chain: RMSNorm → 3 matmuls → RoPE on q and k.
3. Test with golden embed out as input: compare q, k, v (post-RoPE) to golden layer 0.

## Done when
- [ ] q, k, and v all match golden within tolerance.

## Pitfalls
- Errors compound now. If q is off but norm output was fine, suspect the matmul layout (`W` vs `Wᵀ`).

## Notes
