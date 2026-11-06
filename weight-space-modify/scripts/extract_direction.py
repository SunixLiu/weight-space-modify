#!/usr/bin/env python
"""Extract a "loop feature direction" vector from saved probe activations.

Usage:
  python extract_direction.py --activations probe.pt --layer 5 --slot mlp \
      [--mode mean|diff|last] [--out direction.pt]

Slots (as captured by probe_activations.py):
  residual: decoder layer output (hidden size)
  mlp:      MLP block output (intermediate size)
  attn:     attention output projection (hidden size)

Modes:
  mean: mean of the slot vectors across all generated tokens
  diff: mean(last quarter) - mean(first quarter) — the drift direction that
        builds up as the model enters the loop
  last: the final token's vector

Output: unit-normalized 1-D float32 tensor saved with torch.save, plus a JSON
summary on stderr. Feed the .pt file to apply_patch_and_save.py --vector.
"""
import argparse
import json
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True, help=".pt file saved by probe_activations.py --save-activations")
    ap.add_argument("--layer", required=True, help="Layer index as captured (e.g. 5)")
    ap.add_argument("--slot", required=True, choices=["residual", "mlp", "attn"])
    ap.add_argument("--mode", default="mean", choices=["mean", "diff", "last"])
    ap.add_argument("--out", required=True, help="Output .pt path for the direction vector")
    args = ap.parse_args()

    blob = torch.load(args.activations, map_location="cpu", weights_only=False)
    if args.layer not in blob["layers"]:
        raise SystemExit(f"layer {args.layer} not in saved activations; available: {list(blob['layers'])}")
    vecs = blob["layers"][args.layer][args.slot]
    if not vecs:
        raise SystemExit(f"no '{args.slot}' activations captured for layer {args.layer}")
    x = torch.stack(vecs).float()  # [steps, dim]

    if args.mode == "mean":
        v = x.mean(dim=0)
    elif args.mode == "last":
        v = x[-1]
    else:  # diff
        k = max(1, x.shape[0] // 4)
        v = x[-k:].mean(dim=0) - x[:k].mean(dim=0)

    v = v / (v.norm() + 1e-12)
    torch.save(v, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "layer": args.layer,
                "slot": args.slot,
                "mode": args.mode,
                "dim": int(v.numel()),
                "raw_norm": round(float(torch.stack(vecs).float().mean(dim=0).norm()), 4)
                if args.mode != "last" else None,
            }
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
