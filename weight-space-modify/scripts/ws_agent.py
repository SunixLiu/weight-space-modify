#!/usr/bin/env python
"""Agentic weight editing: give the local 27B surgeon the weights, not just a transcript.

This is the `weightspace` session loop wired to this project:

  * the editing LLM sees a budgeted set of forward passes (`forward_check`,
    `forward_generate`) instead of being handed probe JSON by a human;
  * it can list/describe/read tensors, run Python statically over the weights,
    and edit them through tracked tools (`write_tensor`, `W[...]`, `diff`,
    `checkpoint`, `restore`, `revert`);
  * every edit is checkpointed, so a trial that does not help is rolled back by
    the same process that made it;
  * the run ends with a sparse patch + a complete loadable model directory.

The default LLM is the local llama.cpp endpoint (`Qwen3.8-27B`), which is a
reasoning model: tool calls are supported only with a tool-aware chat template,
so run `--ping-llm` once after starting the server.

Usage:
  python ws_agent.py --model DIR --task "..." --prompt-file prompt_loop.txt \
      --mode fix --forward-budget 12 --max-steps 40 --out patched_agent

  python ws_agent.py --model DIR --mode read --task "Which layers drive the loop?" \
      --prompt-file prompt_loop.txt --forward-budget 4

Read the transcript after the run: the rules (no training, no running the net
inside the sandbox) are enforced statistically, not mechanically.
"""
import argparse
import os
import sys

import weightspace as wsp

from _common import emit, gen_metrics
from ws_core import diff_dict, open_space, open_space_lazy, refresh, save_model, sync

DEFAULT_TASK = (
    "Reduce the dead-loop rate on MATH problems: the model keeps generating repetitive text until the "
    "token cap instead of emitting EOS. Diagnose the cause in the weights, then edit the weights to fix it. "
    "Loop rate is the primary metric; accuracy on problems it already answers must not regress."
)

ARCH_CONTEXT = """Model: Qwen3.5-0.8B (Qwen3_5ForConditionalGeneration), bf16, tied embeddings.
- 24 decoder layers; full attention at layers 3, 7, 11, 15, 19, 23, the rest are GDN linear attention.
- hidden size 1024, MLP width 3584, vocabulary 248320.
- tensor names: model.language_model.layers.{i}.{self_attn|linear_attn|mlp}.<proj>.weight
- embed_tokens.weight is tied with lm_head, so row <eos_id> is the EOS logit row.

Failure under study: semantic repetition loops (no periodic token cycle, no NaN, no
activation-variance blow-up). The known fingerprint is the EOS logit trajectory: it rises
before a successful stop and stays flat or falls during a loop.

Editing notes:
- Weights are bf16 (~8 mantissa bits). The sandbox shows float32 mirrors, but after the
  model is synced anything below ~0.4% of the tensor norm rounds away. Call `diff()` after
  every edit and trust only what it reports; prefer edits with a relative delta >= 0.005.
- The two useful directions from prior probing are the EOS row of embed_tokens and the
  GDN gate means (linear_attn.in_proj_a) in layers around 20, but the evidence is weak
  and net-zero so far: re-derive the cause rather than trusting that prior.
"""


def build_probes(space, eos_id: int, max_new_tokens: int = 1024):
    """Probes = the only way the editing LLM may run the network."""
    cap = max_new_tokens

    @wsp.probe(
        cost=1,
        parameters={
            "type": "object",
            "properties": {
                "prompts": {"type": "array", "items": {"type": "string"}, "description": "1-4 prompts"},
                "max_new_tokens": {"type": "integer", "default": 384},
            },
            "required": ["prompts"],
        },
    )
    def check(prompts, max_new_tokens: int = 384) -> list:
        """Generate greedily on up to 4 prompts with the current weights and report loop/EOS/truncation metrics.
        Lower loop, more eos_hit, larger first_repeat_at is better."""
        sync(space)  # replay edits into the model before spending the forward pass
        out = []
        for prompt in list(prompts)[:4]:
            m = gen_metrics(
                space.model, space.tokenizer, prompt, space.device, min(int(max_new_tokens), cap), eos_id
            )
            m["prompt"] = prompt[:160]
            out.append(m)
        return out

    @wsp.probe(
        cost=1,
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "max_new_tokens": {"type": "integer", "default": 256},
            },
            "required": ["prompt"],
        },
    )
    def generate(prompt, max_new_tokens: int = 256) -> dict:
        """Generate on one prompt with the current weights and return the tail of the text plus the same metrics
        (use this to see *what* the loop repeats, not just that it loops)."""
        sync(space)
        return gen_metrics(
            space.model,
            space.tokenizer,
            prompt,
            space.device,
            min(int(max_new_tokens), cap),
            eos_id,
            tail_tokens=400,
        )

    return [check, generate]


def build_llm(args):
    from weightspace import get_llm

    kw = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "thinking_style": None if args.thinking_style == "none" else args.thinking_style,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    return get_llm(args.llm, **kw)


def ping_llm(args) -> None:
    """One tool-call round trip against the editing LLM. Costs a single request."""
    ws = open_space_lazy(args.model)
    llm = build_llm(args)
    session = wsp.Session(ws, llm, mode="read", forward_budget=0, python=False, max_steps=1)
    specs = [t.spec() for t in session.tools.values()]
    resp = llm.complete(
        [
            {"role": "system", "content": "You are a weight-space programmer."},
            {
                "role": "user",
                "content": "Call list_tensors with pattern 'layers.20.mlp.down_proj' and nothing else.",
            },
        ],
        specs,
    )
    emit(
        {
            "llm": getattr(llm, "name", type(llm).__name__),
            "tool_calling_ok": bool(resp.tool_calls),
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in resp.tool_calls],
            "text": (resp.text or "")[:500],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="fix", choices=["read", "fix", "improve"])
    ap.add_argument("--task", default=None, help=f"Task text (default: the MATH dead-loop statement)")
    ap.add_argument("--task-file", default=None)
    ap.add_argument("--context-file", action="append", default=[], help="Extra context (repeatable)")
    ap.add_argument("--prompt-file", action="append", default=[], help="Known looping prompt(s) to show the LLM")
    ap.add_argument("--llm", default="openai:Qwen3.8-27B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--api-key", default="sk-local")
    ap.add_argument("--thinking-style", default="none", choices=["none", "chat_template", "reasoning_effort"])
    ap.add_argument("--max-tokens", type=int, default=4096, help="Editing-LLM reply cap (27B is a reasoning model)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--forward-budget", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="Cap for a single probe generation")
    ap.add_argument("--python", default="on", choices=["on", "off"], help="Give the LLM the weight sandbox")
    ap.add_argument("--out", default=None, help="Save the edited model here (only if weights changed)")
    ap.add_argument("--run-dir", default=None, help="Save transcript.jsonl/result.json/patch.npz here")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ping-llm", action="store_true", help="Check tool calling on the endpoint and exit")
    args = ap.parse_args()

    if args.ping_llm:
        ping_llm(args)
        return

    task = args.task
    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task = f.read().strip()
    task = task or DEFAULT_TASK

    context = ARCH_CONTEXT
    for path in args.context_file:
        with open(path, "r", encoding="utf-8") as f:
            context += f"\n\n## {os.path.basename(path)}\n{f.read().strip()}"
    if args.prompt_file:
        context += "\n\nKnown looping prompts (pass them to forward_check / forward_generate):"
        for path in args.prompt_file:
            with open(path, "r", encoding="utf-8") as f:
                context += f"\n--- {os.path.basename(path)} ---\n{f.read().strip()[:1500]}"

    space = open_space(args.model, args.device)
    eos_id = space.tokenizer.eos_token_id or space.model.config.eos_token_id
    probes = build_probes(space, eos_id, args.max_new_tokens)
    llm = build_llm(args)

    session = wsp.Session(
        space.ws,
        llm,
        mode=args.mode,
        context=context,
        probes=probes,
        forward_budget=args.forward_budget,
        max_steps=args.max_steps,
        python=args.python == "on",
        verbose=args.verbose,
    )

    print(f"[ws_agent] mode={args.mode} llm={llm.name} forward_budget={args.forward_budget}", file=sys.stderr, flush=True)
    try:
        res = session.run(task)
    except wsp.SessionError as e:
        res = e.result
        print(f"[ws_agent] run failed: {e}", file=sys.stderr, flush=True)

    # Make the report honest about the bf16 cast before diffing/saving.
    refresh(space.ws)
    report = diff_dict(space.ws)

    out_dir = None
    if args.out and report["changed_params"]:
        out_dir = save_model(space, args.out)

    run_dir = None
    if args.run_dir:
        res.save(args.run_dir)
        run_dir = os.path.abspath(args.run_dir)

    payload = {
        "mode": args.mode,
        "llm": llm.name,
        "answer": res.answer,
        "confidence": res.confidence,
        "stop_reason": res.stop_reason,
        "steps": res.steps,
        "forward_used": res.forward_used,
        "forward_budget": args.forward_budget,
        "changed_params": report["changed_params"],
        "fraction": report["fraction"],
        "tensors": report["tensors"],
        "out_dir": out_dir,
        "run_dir": run_dir,
        "tokens": res.tokens.to_dict(),
        "seconds": round(res.seconds, 1),
        "error": res.error,
    }
    if args.out and not report["changed_params"]:
        payload["note"] = "no weights changed, so no model directory was written"
    emit(payload)


if __name__ == "__main__":
    main()
