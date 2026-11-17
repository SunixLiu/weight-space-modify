#!/usr/bin/env python
"""Logit-level probe: what does the model's next-token distribution look like while it loops?

Answers the question "is the loop attractor in the final logits?" by recording, at every
generated step, the top-k token probabilities and the EOS probability, then comparing
steps that lie inside a repeated 8-gram region ("loop steps") against all other steps.

Usage:
  python probe_logits.py --model <dir> --prompt-file F [--max-new-tokens 512] \
      [--top-k 5] [--device auto]

Loop-step definition (same criterion as run_eval_benchmark's ngram_repeat):
  step t is a loop step if the 8-gram ending at t has already occurred >= 3 times
  in the generated sequence up to and including t.

Output (JSON on stdout):
  {model, prompt_file, n_generated, truncated, loop_detected, summary, per_step}
"""
import argparse
import json
import sys
import time

import torch

from _common import chat_input_ids, detect_ngram_repeat, detect_period, emit, load_model, load_tokenizer, pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True, help="Text file containing the prompt")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    device = pick_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = model.config.eos_token_id

    input_ids = chat_input_ids(tokenizer, prompt)
    t0 = time.time()
    gen = model.generate(
        torch.tensor([input_ids], device=device),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    tokens = gen.sequences[0].tolist()[len(input_ids):]
    scores = gen.scores  # tuple of [1, vocab] logits, one per generated step

    n = len(tokens)
    truncated = n == args.max_new_tokens and (not tokens or tokens[-1] != eos_id)
    period = detect_period(tokens)
    ngram = detect_ngram_repeat(tokens)

    # Per-step: top-k probabilities, EOS probability, loop-step flag.
    # Loop step: the 8-gram ending at t has occurred >= 3 times up to t.
    per_step = []
    seen = {}
    for t in range(n):
        probs = torch.softmax(scores[t][0].float(), dim=0)
        _, topk_idx = torch.topk(probs, args.top_k)
        topk_ids = topk_idx.tolist()
        top5 = []
        for i in topk_ids:
            top5.append([i, tokenizer.decode([i]), round(float(probs[i]), 6)])
        p_top1 = float(probs[tokens[t]])  # greedy: generated token is the argmax
        p_eos = float(probs[eos_id])

        if t >= 7:
            gram = tuple(tokens[t - 7 : t + 1])
            seen[gram] = seen.get(gram, 0) + 1
            in_loop = seen[gram] >= 3
        else:
            in_loop = False

        per_step.append(
            {
                "t": t,
                "token": tokens[t],
                "text": tokenizer.decode([tokens[t]]),
                "top1_prob": round(p_top1, 6),
                "eos_prob": round(p_eos, 6),
                "in_loop": in_loop,
                "top_k": top5,
            }
        )

    loop_steps = [s for s in per_step if s["in_loop"]]
    nonloop = [s for s in per_step if not s["in_loop"]]

    def stats(vals):
        if not vals:
            return None
        vals = sorted(vals)
        k = len(vals)
        return {
            "n": k,
            "mean": round(sum(vals) / k, 6),
            "median": round(vals[k // 2], 6),
            "p10": round(vals[max(0, int(k * 0.1))], 6),
            "p90": round(vals[min(k - 1, int(k * 0.9))], 6),
        }

    # Dominant repeated 8-gram and its text (what is actually being looped).
    counts = {}
    for t in range(7, n):
        gram = tuple(tokens[t - 7 : t + 1])
        counts[gram] = counts.get(gram, 0) + 1
    dominant = None
    if counts:
        g, c = max(counts.items(), key=lambda kv: kv[1])
        dominant = {"text": tokenizer.decode(list(g)), "count": c}

    emit(
        {
            "model": args.model,
            "prompt_file": args.prompt_file,
            "max_new_tokens": args.max_new_tokens,
            "n_generated": n,
            "truncated": truncated,
            "loop_detected": {"period": period, "ngram_repeat": list(ngram) if ngram else None},
            "summary": {
                "loop_steps": len(loop_steps),
                "nonloop_steps": len(nonloop),
                "top1_prob_loop": stats([s["top1_prob"] for s in loop_steps]),
                "top1_prob_nonloop": stats([s["top1_prob"] for s in nonloop]),
                "eos_prob_loop_mean": round(sum(s["eos_prob"] for s in loop_steps) / len(loop_steps), 6) if loop_steps else None,
                "eos_prob_nonloop_mean": round(sum(s["eos_prob"] for s in nonloop) / len(nonloop), 6) if nonloop else None,
                "sharp_frac_loop": round(sum(1 for s in loop_steps if s["top1_prob"] > 0.9) / len(loop_steps), 4) if loop_steps else None,
                "dominant_ngram": dominant,
            },
            "per_step": per_step,
        }
    )
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
