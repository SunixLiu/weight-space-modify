#!/usr/bin/env python
"""Inspect tensor shapes and statistics of a model's safetensors weights.

Usage:
  python inspect_weights.py --model <dir> [--layer SUBSTRING] [--all] [--top-k N]
  python inspect_weights.py --model <dir> --describe SUBSTRING

--layer filters tensor names by substring, e.g.:
  layers.5.mlp          -> all MLP tensors of layer 5
  mlp.down_proj         -> down_proj of every layer
  self_attn.q_proj      -> query projections of full-attention layers
  linear_attn.out_proj  -> GDN output projections

Without --layer/--all/--top-k/--describe: prints the layer map (one line per
decoder layer with its type and key tensor shapes).

--describe SUBSTRING gives the structural view from the `weightspace` package
(quantiles, row/column L2 norms, singular-value spectrum, rank for 90% energy)
for every matching tensor. It reads tensors lazily from the safetensors headers,
so it does not load or run the model.

Output (JSON on stdout): {tensors: [{name, shape, dtype, mean, std, norm, min, max}]}
or, for --describe: {tensors: [{name, description}]}
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open


def iter_tensors(model_dir: str):
    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def tensor_stats(t: torch.Tensor) -> dict:
    tf = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "mean": round(float(tf.mean()), 6),
        "std": round(float(tf.std()), 6),
        "norm": round(float(tf.norm()), 4),
        "min": round(float(tf.min()), 6),
        "max": round(float(tf.max()), 6),
    }


def layer_map(model_dir: str) -> list[dict]:
    """One entry per decoder layer: index, type, attention/mlp shapes."""
    layers: dict[int, dict] = {}
    for name, t in iter_tensors(model_dir):
        if ".layers." not in name or name.startswith("mtp."):
            continue
        idx = int(name.split(".layers.")[1].split(".")[0])
        entry = layers.setdefault(idx, {"index": idx, "type": None, "tensors": {}})
        short = name.split(f".layers.{idx}.")[-1]
        entry["tensors"][short] = list(t.shape)
        if "linear_attn" in short:
            entry["type"] = "linear_attention (GDN)"
        elif "self_attn" in short:
            entry["type"] = "full_attention"
    return [layers[k] for k in sorted(layers)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default=None, help="Substring filter on tensor names")
    ap.add_argument("--all", action="store_true", help="Stats for every tensor (slow for big models)")
    ap.add_argument("--top-k", type=int, default=0, help="Only the K largest tensors by norm")
    ap.add_argument(
        "--describe",
        default=None,
        metavar="SUBSTR",
        help="weightspace structural description (quantiles, norms, spectrum) of matching tensors",
    )
    args = ap.parse_args()

    if args.describe is not None:
        from _common import emit
        from ws_core import open_space_lazy

        ws = open_space_lazy(args.model)
        names = ws.names(args.describe)
        if not names:
            raise SystemExit(f"no tensor matches '{args.describe}'")
        emit(
            {
                "model": args.model,
                "filter": args.describe,
                "count": len(names),
                "tensors": [{"name": n, "description": ws.describe(n)} for n in names],
            }
        )
        return

    if not (args.layer or args.all or args.top_k):
        from _common import emit

        emit({"layer_map": layer_map(args.model)})
        return

    results = []
    for name, t in iter_tensors(args.model):
        if args.layer and args.layer not in name:
            continue
        results.append({"name": name, **tensor_stats(t)})

    if args.top_k > 0:
        results.sort(key=lambda r: r["norm"], reverse=True)
        results = results[: args.top_k]

    from _common import emit

    emit({"model": args.model, "filter": args.layer, "count": len(results), "tensors": results})


if __name__ == "__main__":
    main()
