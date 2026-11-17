# weight-space-modify

Diagnose and fix "math dead-loops" in small LLMs with **minimal weight-space surgery** — no retraining.

When a small model (e.g. Qwen3.5-0.8B) falls into infinite repetition on math benchmarks such as MATH/GSM8K — hitting the token cap, never emitting EOS — this skill provides the full closed loop: probe the safetensors weights → extract the loop direction vector → apply a minimal patch → re-evaluate to verify. A local 27B reasoning model acts as the "surgeon", deciding at every decision point; the Python scripts are the scalpel and the monitor.

> 中文说明见 [README.zh.md](README.zh.md)

## Getting started

**1. Install the skill.** Clone this repo and copy the `weight-space-modify/` directory into your agent's skills folder (e.g. `~/.claude/skills/`, or the skills dir of your DSH profile). It is self-contained: `SKILL.md` (agent-facing instructions) + `references/` + `scripts/`. The agent picks it up by name/description when a task matches — a small model looping on math that must be fixed with weight edits, not retraining.

**2. Set up the Python environment.**

```bash
python -m venv .venv
# Windows: .venv\Scripts\pip install ...   Linux/macOS: .venv/bin/pip install ...
pip install torch transformers numpy safetensors "weightspace==0.3.0"
# optional, ~5–10× faster GDN layers:
pip install flash-linear-attention causal-conv1d
```

**3. Prepare the inputs.**

- Target model: Qwen3.5-0.8B as a safetensors directory (standard HF layout)
- Benchmark: MATH jsonl — bundled at `weight-space-modify/data/math-test.jsonl` (500 problems), or your own file via `--data`
- Surgeon: a local 27B reasoning model behind an OpenAI-compatible endpoint with tool calling (e.g. a llama.cpp server), default `http://127.0.0.1:8001/v1`

**4. Use it.**

- *Agent-driven (recommended):* give your agent a task such as "Qwen3.5-0.8B dead-loops on MATH; reduce the loop rate with weight edits". It follows the closed loop in `SKILL.md`, consulting the 27B surgeon at each decision point.
- *Manual:* run the scripts from the skill directory, overriding the machine-specific defaults in `SKILL.md` via flags:

```bash
python scripts/run_eval_benchmark.py --model /path/to/qwen3.5-0.8b --data /path/to/math.jsonl --limit 20
```

All scripts print one JSON document to stdout (stderr carries progress). On Windows set `PYTHONIOENCODING=utf-8` first.

## The closed loop

1. **Baseline** — `run_eval_benchmark.py` on 20 problems → `eval_before.json` (accuracy / dead-loop rate / truncation rate)
2. **Diagnose** — `consult_surgeon.py --phase diagnose`: the 27B proposes mechanism hypotheses and a probing plan
3. **Probe** — `probe_activations.py` on a prompt that actually loops, capturing per-token residual/MLP/attention activations (`--save-activations`)
4. **Analyze** — `consult_surgeon.py --phase probe`: the 27B names candidate layers/directions
5. **Strategy** — `consult_surgeon.py --phase strategy`: the surgical plan (tensor, op, direction source, scale; start small, 0.1–0.5)
6. **Patch** — `extract_direction.py` (optionally `select_columns.py` to pick intermediate-neuron columns) → `apply_patch_and_save.py` saves a complete loadable model + a sparse patch (`.npz`, a few KB)
7. **Re-evaluate** — `run_eval_benchmark.py --model patched_x` → `eval_after.json`
8. **Evaluate** — `consult_surgeon.py --phase evaluate`: keep or revert, iterate

Agent mode: `ws_agent.py` lets the 27B read/write weights directly through tools (forward-pass budget), replacing steps 5–6.

## Layout

| Path | What it does |
|---|---|
| `SKILL.md` | Skill description and full tool list |
| `references/architecture.md` | Qwen3.5-0.8B tensor layout, direction space → op mapping |
| `data/math-test.jsonl` | Bundled MATH benchmark (500 problems) |
| `scripts/run_eval_benchmark.py` | MATH evaluation: accuracy / loop rate / truncation rate |
| `scripts/inspect_weights.py` | Layer map and tensor stats (reads safetensors headers only) |
| `scripts/probe_activations.py` | Per-token activation capture (residual/mlp/attn/mlp_mid) |
| `scripts/extract_direction.py` | Builds the unit "loop direction" vector from probe captures |
| `scripts/select_columns.py` | Ranks intermediate neurons by loop-vs-control contrast, selects columns |
| `scripts/probe_logits.py` | Logit-level probe: per-step top-k / EOS probabilities |
| `scripts/phase_probe.py` | Offline phase-aligned analysis of probe `.pt` files (no GPU) |
| `scripts/apply_patch_and_save.py` | Applies one minimal patch, saves full model + sparse patch |
| `scripts/patch_apply.py` | Replays a sparse patch onto a checkpoint of the same architecture |
| `scripts/scan_layers.py` | Layer-wise causal scan: ranks layers by post-edit behavior |
| `scripts/ws_agent.py` | Agentic weight editing (the 27B edits weights directly) |
| `scripts/consult_surgeon.py` | 27B surgeon consultation (four-phase system prompts) |
| `scripts/selfcheck_ws.py` | Offline self-check of the weightspace bridge |
| `scripts/ws_core.py` | weightspace bridge: tracking / rollback / bf16 accounting |
| `scripts/_common.py` | Model loading / chat template / dead-loop detection / answer extraction |

## Hard rules

- Never write into the source model directory; every patch goes to a fresh `--out` dir
- One patch, one re-evaluation; never stack two patches before measuring
- Weights are bf16 (~8 mantissa bits): trust `changed_params` in the JSON, not the nominal delta; 0 means "no-op"
- Prefer sparse patches (`--patch-out` + `patch_apply.py`) over keeping full model copies
- Loop rate is the primary metric; accuracy must not regress
- Start scales small (0.1–0.5); a change that fixes some samples and breaks others is *directionally real but too large* — shrink it, don't dismiss it as noise

## Verified results (Qwen3.5-0.8B, MATH, 20 problems)

Baseline: acc 10% / loop 50% / trunc 70%. After multiple single-edit closed loops, the best single edit is r7 (L20 GDN `A_log` head 2, scale 0.2, exactly one scalar changed): **loop 50% → 35%** (net −3: 4 fixed, 1 newly broken), acc unchanged at 10%. Honest boundary: accuracy never improved — this skill verifies that "the loop rate can be reduced by minimal weight edits", not the reference project's accuracy reproduction.
