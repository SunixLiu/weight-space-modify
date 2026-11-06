#!/usr/bin/env python
"""Layer-wise causal scan: find which layer's intervention changes loop behavior.

Loads the model ONCE, then for every candidate layer applies the same
minimal intervention (project-out-output of that layer's own loop direction
on mlp.down_proj), generates on a small set of known-looping prompts, and
records the loop metrics. The edit goes through `weightspace`, so each trial is a
tracked, reversible edit: `revert(name)` restores the exact original tensor and
the diff after `sync()` says how many weights actually survived the bf16 cast.
The result is a ranked list of layers that actually matter (causal, not
correlational).

Usage:
  python scan_layers.py --model DIR --prompts prompt_loop.txt [--prompts ...] \
      --activations probe_loop.pt [--layers all|0,5,15] [--scale 1.0] \
      --max-new-tokens 384 [--out scan.json]

Direction per layer comes from the same probe capture: mean of the `mlp` slot
over the looping trajectory (output space, hidden size). Layers whose slot was
not captured are skipped.
"""
import argparse
import json

import numpy as np

from _common import emit, gen_metrics
from ws_core import diff_dict, open_space, sync


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", action="append", required=True, help="Prompt file(s) that loop (repeatable)")
    ap.add_argument("--activations", required=True, help="probe .pt with all layers captured")
    ap.add_argument("--layers", default="all", help="'all' or comma list")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys
    import time

    import torch

    t0 = time.time()
    print(f"[scan] loading {args.model} on {args.device} ...", file=sys.stderr, flush=True)
    space = open_space(args.model, args.device)
    ws = space.ws
    eos_id = space.tokenizer.eos_token_id or space.model.config.eos_token_id
    print(f"[scan] model loaded in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    blob = torch.load(args.activations, map_location="cpu", weights_only=False)
    caps = blob["layers"]

    layer_idx = sorted(int(k) for k in caps if caps[k].get("mlp"))
    if args.layers != "all":
        want = {int(x) for x in args.layers.split(",") if x.strip()}
        layer_idx = [i for i in layer_idx if i in want]

    prompts = []
    for path in args.prompts:
        with open(path, "r", encoding="utf-8") as f:
            prompts.append(f.read().strip())

    name_map = {}
    for i in layer_idx:
        cand = [n for n in ws.names() if f"layers.{i}.mlp.down_proj.weight" in n]
        if len(cand) == 1:
            name_map[i] = cand[0]

    def run_all() -> list[dict]:
        return [
            gen_metrics(space.model, space.tokenizer, p, space.device, args.max_new_tokens, eos_id) for p in prompts
        ]

    t1 = time.time()
    baseline = run_all()
    print(f"baseline ({time.time() - t1:.1f}s): {json.dumps(baseline, ensure_ascii=False)}", file=sys.stderr, flush=True)

    results = []
    for i in layer_idx:
        if i not in name_map:
            continue
        name = name_map[i]
        v = torch.stack(caps[str(i)]["mlp"]).float().mean(dim=0)
        v = (v / (v.norm() + 1e-12)).numpy().astype(np.float32)
        W = np.array(ws[name], dtype=np.float32)
        W_new = W - args.scale * np.outer(v, v @ W) / float(np.dot(v, v))
        ws.set(name, W_new)
        sync(space)
        try:
            changed = diff_dict(ws)["tensors"]
            after = run_all()
        finally:
            ws.revert(name)
            sync(space)
        entry = {
            "layer": i,
            "tensor": name,
            "changed": changed,
            "baseline": baseline,
            "after": after,
            "loop_count_after": sum(m["loop"] for m in after),
            "eos_hits_after": sum(m["eos_hit"] for m in after),
            "first_repeat_shift": [
                (m["first_repeat_at"] or 10**6) - (b["first_repeat_at"] or 10**6)
                for m, b in zip(after, baseline)
            ],
        }
        results.append(entry)
        print(
            f"layer {i:2d}: loop={entry['loop_count_after']} eos={entry['eos_hits_after']} "
            f"repeat_shift={entry['first_repeat_shift']}",
            file=sys.stderr,
            flush=True,
        )

    results.sort(key=lambda r: (r["loop_count_after"], -r["eos_hits_after"]))
    out = {
        "model": args.model,
        "scale": args.scale,
        "baseline": baseline,
        "ranking": [r["layer"] for r in results],
        "layers": results,
    }
    emit(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"saved scan to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
