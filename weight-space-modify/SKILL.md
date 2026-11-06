---
name: weight-space-modify
description: Diagnose and fix mathematical dead-loops in small LLMs (repetition until the 32k-token cap, EOS never emitted, e.g. Qwen3.5-0.8B on MATH) by probing safetensors weights and applying minimal weight-space surgery, with a local 27B model acting as the surgeon at every decision point. Use when a small model loops on math benchmarks (MATH/GSM8K) and the fix must be weight edits, not retraining.
---

# Weight-space Modify

Python scripts are the scalpel and monitor; the local **27B surgeon** decides what to cut.
Never skip a surgeon review between phases, and never patch without a baseline.

## Environment defaults (overridable by flags)

- Python: `D:\01Projects\Weight-space-programing\.venv\Scripts\python.exe`
- Model: `E:\LLMModels\Qwen\qwen3.5-0.8b` (Qwen3.5-0.8B, hybrid GDN/attention)
- Benchmark: `data/math-test.jsonl` in this skill dir (MATH jsonl: `problem`/`answer`); override with `--data`
- Surgeon: `http://127.0.0.1:8001/v1`, model `Qwen3.8-27B` (reasoning model — needs big `--max-tokens`)
- Artifacts (eval/probe JSON, direction .pt, patched models): project root, never the skill dir.

Run scripts from the skill directory, e.g. `python scripts/run_eval_benchmark.py ...`.
On Windows set `PYTHONIOENCODING=utf-8` first. All scripts print one JSON document to
stdout (stderr carries warnings/progress).

## Tools

| Script | Purpose |
|---|---|
| `run_eval_benchmark.py --model D --data F [--limit N] [--max-new-tokens 512] [--temperature 0] [--repetition-penalty P]` | Accuracy + loop rate + truncation rate on a MATH jsonl. Loop = periodic tail (period ≤ 48) or an 8-gram repeated ≥3× in the last 256 tokens. `--repetition-penalty` (1.0 = off) is the decoding-side control knob: it works with greedy and measures how much of the loop rate is a decoding-level probability well rather than a weight-level attractor. |
| `inspect_weights.py --model D [--layer SUBSTR] [--all] [--top-k N] [--describe SUBSTR]` | Layer map (per-layer type + shapes) or per-tensor shape/mean/std/norm/min/max. `--describe` adds the structural view from `weightspace` (quantiles, row/column L2 norms, singular values, rank for 90% energy) and reads only safetensors headers, so the model is never loaded. |
| `probe_activations.py --model D --prompt T \| --prompt-file F [--layers all\|0,4,8] [--max-new-tokens 128] [--save-activations X.pt]` | Per-token residual norms, cos-to-previous-token, MLP stats (zero fraction, max), attention-output norms; flags `stuck` / `spike` / `saturated`. Also captures the 3584-dim SwiGLU intermediate (`mlp_mid`, per-neuron activations feeding `down_proj`) so individual columns can be selected from evidence. |
| `extract_direction.py --activations X.pt --layer N --slot residual\|mlp\|attn [--mode mean\|diff\|last] --out V.pt` | Builds the unit "loop direction" vector from probe captures. `mlp` slot = MLP output space (hidden size); `residual`/`attn` = hidden size. |
| `select_columns.py --loop X.pt [--loop Y.pt ...] --control Z.pt --layer N [--top-n 64] [--mode loop-abs\|delta] [--out-columns C.json]` | Ranks the 3584 intermediate neurons of a layer's `down_proj` by last-quarter activation in looping runs vs a successful run, and writes the top-N indices as a JSON list for `apply_patch_and_save.py --columns`. `--mode loop-abs` = paper form (largest activation in the loop state); `--mode delta` = largest loop-vs-control contrast. |
| `probe_logits.py --model D --prompt-file F [--max-new-tokens 512] [--top-k 5]` | Logit-level probe: per-step top-k token probabilities and EOS probability during greedy generation; flags steps inside a repeated 8-gram region ("loop steps") and reports loop-vs-nonloop confidence stats plus the dominant repeated n-gram. Answers "is the loop attractor in the final logits?" |
| `phase_probe.py --probes DIR --pattern 'probe_*.pt' [--layer N] [--slot residual\|mlp\|attn\|mlp_mid] [--max-period 48] [--out-dir D] [--ref V.pt] [--ref-pattern P]` | Offline phase-aligned analysis of saved probe .pt files (no GPU): estimates the loop period by autocorrelation, locates the loop start, bins activations by phase to find deviation/jump peaks ("trigger frame" candidates), and extracts a pooled "loop region vs pre-loop" direction. Use to test whether loops are strictly token-periodic before investing in phase-specific edits. |
| `apply_patch_and_save.py --model D --layer SUBSTR --op subtract\|project-out\|project-out-output\|add-row --vector V.pt [--scale S] [--index ROW] [--columns SPEC] [--out NEWDIR] [--patch-out P.npz] [--dry-run]` | Applies one minimal patch, saves a complete loadable model to NEWDIR. Edits go through `weightspace`, so the JSON reports `changed_params` / `tensors` **after the bf16 cast** — the only change count that matters. `project-out` removes an **input-space** direction (len = W.shape[1]); `project-out-output` suppresses an **output-space** direction (len = W.shape[0]); `subtract` takes a same-shape matrix or a 1-D column vector; `add-row` biases one row (with tied embeddings, `--layer embed_tokens.weight --index <eos_id>` raises the EOS logit). `--columns SPEC` (project-out-output only) restricts the edit to selected intermediate-neuron columns (`W'[:, J] = W[:, J] - scale·v(vᵀW[:,J])/‖v‖²`, every other column bit-identical) — SPEC is `12,40-63` or a `.npy/.pt/.json` of indices; this is the paper's "64 selected MLP neurons" form. When saving, writes a `.manifest.json` next to NEWDIR (source shard SHA-256 + recipe + changed_elements) and stamps the same source hashes into the sparse patch. |
| `patch_apply.py --model D --patch P.npz [--out DIR] [--dry-run]` | Replays a sparse patch (a few KB, indices + new values) onto a checkpoint of the same architecture; no surgery re-run, no full-model diff. Verifies the target's shard hashes against the patch's recorded source before applying — refuses on mismatch (`source_verified` in the JSON: true / false / null for un-stamped older patches). |
| `scan_layers.py --model D --prompts F --activations X.pt [--layers all\|0,5,15] [--scale S] [--max-new-tokens N]` | Layer-wise causal scan: applies the same intervention per layer on one loaded model and ranks layers by post-edit loop/EOS behavior. Every trial is a tracked edit reverted with `ws.revert()`. |
| `ws_agent.py --model D --mode fix\|read\|improve --task T [--prompt-file F] [--forward-budget N] [--max-steps N] [--out DIR] [--run-dir DIR]` | Agentic weight editing: the 27B surgeon gets the weights and a budget of forward passes (`forward_check`, `forward_generate`) instead of advice. Tools: list/describe/read tensors, Python over the weights, tracked `write_tensor`/`diff`/`checkpoint`/`restore`/`revert`. Ends with a sparse patch + loadable model dir only if weights actually changed. `--ping-llm` checks tool calling on the endpoint. |
| `consult_surgeon.py --phase diagnose\|probe\|strategy\|evaluate [--prompt P] [--context-file F ...]` | Calls the 27B surgeon with phase-specific system prompts; returns plain-text guidance. Still the cheapest option when the decision is "what to probe next", not "edit the weights". |
| `selfcheck_ws.py [--model DIR]` | Offline self-check of the `weightspace` bridge (tracking, patch round-trip, op parity with the old torch code, bf16 quantization, lazy store). Seconds, no model load — run it after touching `ws_core.py`. |

## The closed loop

1. **Baseline** — `run_eval_benchmark --limit 20` → save `eval_before.json`.
2. **Diagnose** — `consult_surgeon --phase diagnose --context-file eval_before.json`; the surgeon proposes
   mechanism hypotheses and a probing plan. Follow its layer ranges / metrics.
3. **Probe** — `probe_activations` on a prompt that actually loops (pick a `loop: true` sample tail from
   the baseline), with `--save-activations probe.pt`. Repeat with different `--layers` if the surgeon asks.
4. **Analyze probe** — `consult_surgeon --phase probe --context-file probe.json`; the surgeon names the
   candidate layers/directions.
5. **Strategy** — `consult_surgeon --phase strategy --context-file probe.json`; get the surgical plan
   (tensor substring, op, direction source, scale — start small, 0.1–0.5).
6. **Patch** — `extract_direction` (mode `diff` for the drift into the loop; `mean` for the loop state),
   optionally `select_columns` to pick the intermediate neurons from probe evidence, then
   `apply_patch_and_save --out patched_<n> --patch-out patched_<n>.npz [--columns C.json]`.
   Read `changed_params` in the JSON: if it is 0, the perturbation was below the bf16 quantization
   step and the run changed nothing.
7. **Re-evaluate** — `run_eval_benchmark --model patched_<n> --limit 20` → `eval_after.json`.
8. **Evaluate** — `consult_surgeon --phase evaluate --context-file eval_before.json --context-file eval_after.json`.
   If improved without collateral damage: keep and continue hardening. If collateral damage: revert to the
   previous model dir, narrow the layer range, reduce scale, or switch direction construction. Iterate.

## Agent mode (27B edits the weights itself)

Steps 5–6 can be replaced by one `ws_agent.py` run when the decision is "try something and measure it"
rather than "explain this probe to me":

```bash
python scripts/ws_agent.py --model E:\LLMModels\Qwen\qwen3.5-0.8b --mode fix \
    --task "Reduce the dead-loop rate without losing accuracy" \
    --prompt-file prompt_loop.txt --forward-budget 12 --max-steps 40 \
    --out patched_agent --run-dir runs/agent-1
```

- The surgeon only sees the weights through tools and only runs the network through `forward_check` /
  `forward_generate`, each costing 1 from `--forward-budget`.
- It edits through `checkpoint` + `write_tensor` / `W[...]`, so a dead-end edit is rolled back in-process.
- `--run-dir` writes `transcript.jsonl`, `result.json`, `patch.npz`; `--out` writes the model dir, and only
  if `changed_params > 0`.
- Always re-evaluate the resulting model with `run_eval_benchmark` before believing it: the agent optimizes
  against the few prompts it was given, which is exactly the overfitting trap this skill warns about.
- `--ping-llm` is a one-request handshake that checks llama.cpp tool calling before spending a real run.

## Hard rules

- Never write into the source model directory; every patch goes to a fresh `--out` dir.
- One patch, one re-evaluation. Never stack two patches before measuring.
- Weights are bf16 (~8 mantissa bits): the float32 edit you compute is **not** the edit that runs.
  Anything below ~0.4% of a tensor's norm rounds away. Trust `changed_params` / `tensors` in the JSON,
  not the nominal `delta_norm`; a zero there means "no-op, pick a bigger scale or a different direction".
- Prefer `--patch-out` + `patch_apply.py` over keeping full model copies: the sparse patch is a few KB and
  replays exactly the weights that changed, so the same edit can be re-applied to any checkpoint.
- Loop rate is the primary metric; accuracy must not regress. Truncation rate exposes near-loops.
- Scale is the safety window: a scale-1.0 `project-out` on `in_proj_a` produced a balanced
  3-fix/3-break (net zero) on 20 samples; start at 0.3–0.5 and widen only if the effect is one-sided.
- A change that fixes some samples and breaks others is *directionally real but too large* — do not
  dismiss it as noise; shrink the perturbation instead.
- Per-process load dominates wall time (~2 min for a 0.8B model on a shared GPU); measure
  `elapsed_s` inside the JSON (generation-only), not the process wall clock.
- During generation, the reported `period` (small integer) is the dead-loop fingerprint; `period: null`
  with `truncated: true` means "still reasoning / too slow", not necessarily looping.
- On this Qwen3.5 hybrid, loop runs show no NaN/variance blow-up (activation variance is *lower* than
  on success); the real fingerprint is the EOS logit trajectory: it rises before a successful stop and
  stays flat/falls during a loop. GDN `in_proj_a` gate means run ~+0.3 higher during loops.
- The 27B is a reasoning model: keep `--max-tokens` ≥ 3000 or `content` comes back empty.
- Speed: GDN layers fall back to reference PyTorch kernels without `flash-linear-attention` +
  `causal-conv1d` (installed in the venv improves throughput ~5–10×).

## Architecture reference

See [references/architecture.md](references/architecture.md) for tensor naming, layer layout,
which direction space maps to which tensor, and loop-detection semantics.
