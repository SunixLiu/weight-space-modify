#!/usr/bin/env python
"""Run a MATH-style benchmark on a model and report accuracy + dead-loop rate.

Usage:
  python run_eval_benchmark.py --model <dir> --data <math.jsonl> \
      [--limit N] [--max-new-tokens 512] [--temperature 0.0] [--seed 0] [--device auto]

Data format: one JSON object per line with at least "problem" and "answer"
(hendrycks/MATH style; "solution" optional).

Output (JSON on stdout):
  {model, n, accuracy, loop_rate, truncated_rate, details: [
      {id, subject, level, correct, loop, period, ngram_repeat, truncated, tail}]}

Loop ("撞 token") detection on generated tokens only:
  - period: last 96 tokens are p-periodic for some p <= 48
  - ngram_repeat: an 8-gram appears >= 3 times in the last 256 tokens
  - truncated: hit max_new_tokens without EOS (reported separately)
"""
import argparse
import json
import random
import time

import torch

from _common import (
    answers_equal,
    chat_input_ids,
    detect_ngram_repeat,
    detect_period,
    emit,
    extract_answer,
    load_model,
    load_tokenizer,
    pick_device,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Model directory with config.json + safetensors")
    ap.add_argument("--data", required=True, help="MATH benchmark JSONL file")
    ap.add_argument("--limit", type=int, default=20, help="Max samples to run (default 20)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy/deterministic")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="Repetition penalty for generate() (1.0 = off); works with greedy too")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    random.seed(args.seed)
    device = pick_device(args.device)

    with open(args.data, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = model.config.eos_token_id

    details = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = s["problem"]
        input_ids = chat_input_ids(tokenizer, prompt)
        gen = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            return_dict_in_generate=True,
        )
        tokens = gen.sequences[0].tolist()[len(input_ids):]
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        truncated = len(tokens) == args.max_new_tokens and (not tokens or tokens[-1] != eos_id)
        period = detect_period(tokens)
        ngram = detect_ngram_repeat(tokens)
        loop = period is not None or ngram is not None

        gold = s.get("answer")
        pred = extract_answer(text)
        correct = answers_equal(pred, gold) if gold is not None else False

        details.append(
            {
                "id": s.get("unique_id", i),
                "subject": s.get("subject"),
                "level": s.get("level"),
                "correct": correct,
                "loop": loop,
                "period": period,
                "ngram_repeat": list(ngram) if ngram is not None else None,
                "truncated": truncated,
                "tail": text[-200:],
            }
        )
        print(f"[{i + 1}/{len(samples)}] correct={correct} loop={loop} period={period} "
              f"trunc={truncated} ({s.get('subject', '?')})", file=__import__("sys").stderr)

    n = len(details)
    emit(
        {
            "model": args.model,
            "n": n,
            "accuracy": sum(d["correct"] for d in details) / n if n else 0.0,
            "loop_rate": sum(d["loop"] for d in details) / n if n else 0.0,
            "truncated_rate": sum(d["truncated"] for d in details) / n if n else 0.0,
            "elapsed_s": round(time.time() - t0, 1),
            "details": details,
        }
    )


if __name__ == "__main__":
    main()
