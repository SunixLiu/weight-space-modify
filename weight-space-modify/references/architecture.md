# Architecture Reference — Qwen3.5-0.8B (target of this skill)

## Model layout

`config.json` declares `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`.
It is a hybrid text/vision model; the vision tower (`model.visual.*`) is irrelevant for text-only work.

Text backbone (`model.language_model`):

- 24 decoder layers, `hidden_size` 1024, `intermediate_size` 3584, `vocab_size` 248320
- `layer_types` repeats `linear_attention ×3 + full_attention`; full attention at layers 3, 7, 11, 15, 19, 23
- full attention: 8 query heads, 2 KV heads, `head_dim` 256 (`q_proj` [4096,1024], `k/v_proj` [512,1024], `o_proj` [1024,2048])
- linear attention (GDN): 16 key/value heads × 128, conv kernel 4; `in_proj_qkv` [6144,1024], `in_proj_z` [2048,1024],
  `in_proj_a/b` [16,1024], `out_proj` [1024,2048], `conv1d` [6144,1,4], `A_log`/`dt_bias` [16], `norm` [128]
- MLP (all layers): `gate_proj`/`up_proj` [3584,1024], `down_proj` [1024,3584], SwiGLU
- `tie_word_embeddings: true` — no separate `lm_head.weight` entry (it views `embed_tokens.weight`)
- `mtp.*` tensors exist in the checkpoint (multi-token prediction) but the class ignores them — safe to leave alone
- rope theta 1e7, partial rotary 0.25, mrope interleaved sections [11,11,10]

## Tensor namespace

- `model.language_model.embed_tokens.weight` [248320, 1024]
- `model.language_model.layers.{i}.*` (see layout above)
- `model.language_model.norm.weight` [1024]
- `model.visual.*`, `mtp.*` — ignore for text surgery

## Direction spaces → which op to use

A probe captures, per token and per layer:

| slot | what it is | dim | usable op |
|---|---|---|---|
| `residual` | decoder-layer output (hidden stream) | 1024 | `project-out-output` on a tensor whose rows are hidden (e.g. `mlp.down_proj`, `self_attn.o_proj`, `linear_attn.out_proj`) |
| `mlp` | MLP block output | 1024 | `project-out-output` on `mlp.down_proj.weight` (W.shape[0]=1024) |
| `attn` | attention output projection result | 1024 | `project-out-output` on `o_proj` / `linear_attn.out_proj` |

`project-out` (input-space removal) instead needs a vector of length `W.shape[1]`
(1024 for projections, 3584 for `down_proj`); it is the right op when the surgeon wants to cut a
*input* feature direction feeding a matrix. `subtract` takes a full matrix (same shape) or a 1-D
vector of length `W.shape[1]` subtracted uniformly from every row.

Rule of thumb: `extract_direction --slot mlp` + `--op project-out-output --layer layernN.mlp.down_proj.weight`
is the canonical "push the loop attractor direction out of the MLP output" operation.

### Column-restricted edits (`--columns`)

`--op project-out-output --columns J` applies the same output-space removal to a subset of the
intermediate-neuron columns only:

```
W'[:, J] = W[:, J] - scale * v (vᵀ W[:, J]) / ‖v‖²
W'[:, ~J] = W[:, ~J]                      (bit-identical)
```

This is the form the reference project / paper used for its Qwen3.5-0.8B edit: 64 selected MLP
neurons per matrix (2 × 64 × 1024 = 131,072 scalar positions, ~130k after bf16 rounding) instead of a
full-width sweep. Pick the columns from evidence, not from norms — e.g. the intermediate neurons with
the largest activation on looping vs successful samples from `probe_activations`. `--columns` accepts
`12,40-63` or a `.npy/.pt/.json` file of indices (`.npy` of an `argpartition` result is the easiest).

Why it matters: a full-width `project-out-output` at scale 1.0 tangles every MLP row with the edit and
produced only noise in the layer scan; a column-restricted edit keeps 99%+ of the matrix untouched, so a
null result is informative and a positive result is attributable to those neurons.

## Loop-detection semantics (`run_eval_benchmark`)

- `period`: smallest p ≤ 48 such that the last ≤96 generated tokens are p-periodic → dead-loop fingerprint
  (`period: 1` = same token, `period: 5..20` = repeating sentence/step)
- `ngram_repeat`: an 8-gram occurring ≥3× within the last 256 generated tokens
- `truncated`: generation hit `max_new_tokens` without EOS — near-loop or too-slow signal, report separately
- `loop` = `period is not None or ngram_repeat is not None`

## weightspace view (`ws_core.py`)

The model is wrapped in a `weightspace.WeightSpace` built from the already-loaded torch module
(`from_torch`, not `from_hf` — see `docs/weightspace-integration.md`).

- **Edits are tracked.** `ws.edit(name)` snapshots the tensor; `ws.diff()` reports per tensor how many
  weights differ from the original, the L2 delta, and that delta as a fraction of the tensor norm.
  `checkpoint(label)` / `restore(label)` / `revert(name)` make trials reversible in-process.
- **bf16 quantization is the real constraint.** `TorchStore` mirrors bf16 weights as float32; edits reach the
  model only on `ws.sync()`. `ws_core.sync()` then `refresh()`es the mirror so `diff()` reflects the cast.
  Concretely: bf16 has 8 mantissa bits, so adding 1e-6 to a weight near 1.0 survives float32 and dies in
  bf16; a whole tensor edit at 0.4% relative norm is roughly the loss threshold. Always read
  `changed_params` from the JSON, never the nominal `delta_norm`.
- **Lazy inspection.** `SafetensorsDirStore` indexes headers only and materialises one float32 tensor at a
  time, so `inspect_weights.py --describe` gives spectrum/norms on a 0.8B model without loading it.
- **Sparse patches.** `save_patch` stores `<name>::idx` + `<name>::val` for changed positions only; applying
  it is `apply_patch` (see `patch_apply.py`). A patch is architecture-checkpoint independent, but not
  architecture independent.
- **Manifest / replay safety.** `apply_patch_and_save.py --out` writes a `.manifest.json` next to the
  patched model: source shard SHA-256s, the exact recipe (op/tensor/scale/columns/index/vector file) and
  the post-bf16 `changed_elements`. The same source hashes are stamped into the sparse patch
  (`__source_sha256__` key), and `patch_apply.py` verifies the target against them before applying —
  replaying a patch onto the wrong checkpoint is refused, not silently executed.

## Loading & environment gotchas

- Load weight tensors directly from every `*.safetensors` in the directory (`_common.load_model`):
  renamed shards, missing index files, and non-standard sharded names all load fine.
- transformers ≥5: `tokenizer.apply_chat_template(tokenize=True)` returns a `BatchEncoding`, not a list.
- transformers ≥5: `from_pretrained(device_map=...)` requires `accelerate`; the loader instantiates from
  config, loads the state dict, then `.to(device, bfloat16)`.
- GDN layers use reference PyTorch kernels unless `flash-linear-attention` + `causal-conv1d` are installed;
  expect ~20 s/question on CPU-ish fallback for a 0.8B model.
- The 27B surgeon (`Qwen3.8-27B`, llama.cpp) is a reasoning model: responses carry `reasoning_content`
  and `content`; give it ≥3000 `max_tokens` or `content` is empty.
