## Notes
# ONE LAYER'S 9 TENSORS

| # | Tensor | Shape | Role |
|---|---|---|---|
| 1 | `input_layernorm.weight` | `[2048]` | Norm weights before attention, one per number |
| 2 | `self_attn.q_proj.weight` | `[2048, 2048]` | Makes the query: 32 heads × 64 |
| 3 | `self_attn.k_proj.weight` | `[512, 2048]` | Makes the key: 8 heads × 64, each shared by 4 query heads |
| 4 | `self_attn.v_proj.weight` | `[512, 2048]` | Makes the value: 8 heads × 64 |
| 5 | `self_attn.o_proj.weight` | `[2048, 2048]` | Mixes the 32 glued head outputs back into 2048 numbers |
| 6 | `post_attention_layernorm.weight` | `[2048]` | Norm weights before the MLP |
| 7 | `mlp.gate_proj.weight` | `[8192, 2048]` | Expand: the gate, passed through SiLU |
| 8 | `mlp.up_proj.weight` | `[8192, 2048]` | Expand: the content to pass through |
| 9 | `mlp.down_proj.weight` | `[2048, 8192]` | Shrink back to 2048 |

Outside the layers table

| Tensor | Shape | Role |
|---|---|---|
| `model.embed_tokens.weight` | `[128256, 2048]` | Token lookup at the start, and scoring at the end (tied, so no `lm_head.weight`) |
| `model.norm.weight` | `[2048]` | Final norm before scoring |

# HOW ATTENTION WORKS

- For each token in shape 1x2048, it is multiplied by the q_proj tensor with shape 2048x2048 resulting in 1x2048 (new numbers), then spliced into 32 slices
- Each previous tokens and also itself is multiplied by k_proj into keys, same idea, but sliced into 8 slices only.
- Slice k of query matching with slice k / 4 of key.
- Then we multiply each token by v_proj giving 8 value slices
- The query slice is dot-producted with each key slice → one score per token → turned into percentages (softmax).
- Glue the 32 results into 2048, then multiply by o_proj.
Then we blend the tokens' value slices with those percentages.

# RESIDUAL ADD
- after attention, we don't replace the token's 2048 numbers, instead, we add the attention results onto them
x = x + attention(x); (simplified)
x = x + mlp(x);
So each step only contributes an adjustment on top of what's already there, and the original information is never lost.

Picture a shared board that travels through 16 layers and every attention block and every MLP reads the shared board and writes notes on top of it, nobody erases it.

# THE NORM (RMSNorm)
Before attention and before MLP, the numbers are rescaled so they don't grow too big or shrink too small. After many additions, they could drift.
(RMS = Root mean squared) <-> squared each number, then add them together and average out and take square root

Steps, using [2, -2, 2, -2] as example
1. Square each number: -> 4,4,4,4
2. Take average: 4 + 4 + 4 + 4 = 16; 16/4 = 4
3. Take square root: sqrt(4 + epsilon) ~ 2 (so we dont divide by 0)
4. Divide each number by it: [1, -1, 1, -1]
5. Multiply each number by its own learned weight. That's the [2048] tensor input_layernorm.weight, one weight per number.

the norm's output goes into attention or MLP. the notepad x itself stays un-normalized
x = x + attention(norm1(x))
x = x + mlp(norm2(x))

# RoPE
The problem: attention compares query and key slices by dot product. Nothing in these numbers says where a token is. So "dog bites man" vs "man bites dog" would get the same scores. The model needs word ordering

The idea: rotate by position
   Take a pair of numbers, (x, y), and treat it as a vector, RoPE rotates that arrow by an angle that depends on the token's position:
   - Token at position 0: rotate 0°
   - Token at position 1: rotate 10°
   - Token at position 2: rotate 20°
   - ..and so on
Why rotation works?
   The angle between 2 vectors matter in a dot product only, so the angle between the 2 tokens with the same relative position will always be the same when placed those 2 in any position preserving the relative distance.
   This helps track which token is close and which token is far away
Only applies to queries and keys, not values
The 64-number slice is cut into 32 pairs that spin at different speeds: fast pairs for nearby words, slow pairs for far-apart words.

# Whole model end-to-end
1. Token ID (one integer, e.g. 12366)
        ↓
2. EMBEDDING: look up row 12366 of embed_tokens
        → x = 2048 numbers                    ← the "notepad"

3. REPEAT 16 TIMES (one layer each):

   ┌── ATTENTION BLOCK ─────────────────────────────┐
   │ a. h = norm(x)              input_layernorm    │
   │ b. q = h × q_proj  → 2048 = 32 slices of 64    │
   │    k = h × k_proj  →  512 =  8 slices of 64    │
   │    v = h × v_proj  →  512 =  8 slices of 64    │
   │ c. RoPE: rotate q and k by position            │
   │ d. save k, v (later tokens will look at them)  │
   │ e. each of 32 heads: score vs. all keys so far │
   │    → percentages → blend values → 64 numbers   │
   │ f. glue 32 × 64 = 2048, × o_proj               │
   │ g. x = x + result           ← residual add     │
   └────────────────────────────────────────────────┘

   ┌── MLP BLOCK ───────────────────────────────────┐
   │ a. h = norm(x)              post_attention_... │
   │ b. expand: gate_proj, up_proj → 8192 each      │
   │ c. filter: silu(gate) × up                     │
   │ d. shrink: × down_proj → 2048                  │
   │ e. x = x + result           ← residual add     │
   └────────────────────────────────────────────────┘

4. FINAL NORM: x = norm(x)
5. SCORES: each row of embed_tokens · x → 128,256 numbers, one per vocabulary token
           (weights are stored [out, in]: each output = one row · x)
6. ARGMAX: pick the biggest → next token ID