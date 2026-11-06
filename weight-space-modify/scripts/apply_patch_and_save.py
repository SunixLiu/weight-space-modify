#!/usr/bin/env python
"""Apply a minimal weight patch to one tensor and save the patched model.

Usage:
  python apply_patch_and_save.py --model <dir> --layer <tensor-name-substring> \
      --op {subtract,project-out,project-out-output,add-row} --vector <file.pt|.npy> \
      [--scale 1.0] [--index ROW] [--out <output-dir>] [--patch-out patch.npz] \
      [--dry-run] [--device cpu]

Operations (W is the target weight matrix, shape [out, in]):
  subtract:           W' = W - scale * V
                      V may be 2-D (same shape as W) or 1-D of length in
                      (same subtraction applied to every row).
  project-out:        W' = W - scale * (W v) outer v / ||v||^2, len(v) == in
                      Removes input-space feature direction v from every row.
  project-out-output: W' = W - scale * (v outer (v^T W)) / ||v||^2, len(v) == out
                      Suppresses output-space feature direction v (e.g. a
                      "loop attractor" direction captured by probe_activations
                      at an MLP block output).
  add-row:            W'[index] = W[index] + scale * v, len(v) == in
                      Adds a bias to a single output row. With
                      tie_word_embeddings, --layer embed_tokens.weight and
                      --index <eos_id> raises the EOS logit uniformly.

--columns SPEC (project-out-output only) restricts the edit to a column subset:
  W'[:, J] = W[:, J] - scale * v (v^T W[:, J]) / ||v||^2, all other columns
bit-identical. This is the paper's "64 selected MLP neurons" form; without it
the op sweeps every column. SPEC is "12,40-63" or a .npy/.pt/.json file of
indices (e.g. the top-N intermediate neurons from a probe activation).

The edit goes through `weightspace` (see ws_core.py), so it is tracked: the JSON
reports how many weights really changed and by how much *after* the bf16 cast,
which is the only number that matters for what the model then does. `--patch-out`
writes a sparse .npz patch (indices + new values) that can be re-applied with
`patch_apply.py` without another full model diff.

The patched model is written to --out as a complete loadable directory
(config.json + safetensors + tokenizer). The original model is never modified.

Bookkeeping: when saving, a `.manifest.json` is written next to the patched model
recording the source shard SHA-256s, the exact recipe (op/tensor/scale/columns/
index/vector file) and how many elements actually changed; the same source hashes
are stamped into the sparse patch so `patch_apply.py` can verify its target before
replaying.

Output (JSON on stdout): {op, tensor, shape, scale, delta_norm, out_dir,
                          changed_params, fraction, tensors, patch, manifest, dry_run}
"""
import argparse
import os

import numpy as np

from _common import emit, pick_device
from ws_core import (
    apply_op,
    diff_dict,
    open_space,
    parse_columns,
    resolve_name,
    save_model,
    shard_hashes,
    stamp_patch_source,
    sync,
    write_manifest,
)


def load_vector(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    import torch

    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise SystemExit(f"--vector file must contain a single tensor (got {type(t)})")
    return t.float().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", required=True, help="Unique substring of the tensor name to patch")
    ap.add_argument("--op", required=True, choices=["subtract", "project-out", "project-out-output", "add-row"])
    ap.add_argument("--vector", required=True, help="Patch vector/matrix file (.pt or .npy)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--index", type=int, default=None, help="Row index for add-row")
    ap.add_argument(
        "--columns",
        default=None,
        metavar="SPEC",
        help="project-out-output only: column subset, e.g. '12,40-63' or a .npy/.pt/.json of indices",
    )
    ap.add_argument("--out", default=None, help="Output directory for the patched model")
    ap.add_argument("--patch-out", default=None, help="Also write a sparse .npz patch here")
    ap.add_argument("--dry-run", action="store_true", help="Report the edit without writing a model directory")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run is given")

    vector = load_vector(args.vector)
    columns = parse_columns(args.columns) if args.columns else None
    space = open_space(args.model, pick_device(args.device).type)
    name = resolve_name(space.ws, args.layer)
    applied = apply_op(space.ws, name, args.op, vector, args.scale, args.index, columns)

    sync(space)  # push into the module + re-read it, so the diff is post-quantization
    report = diff_dict(space.ws)

    source_shards = None
    if (not args.dry_run) or args.patch_out:
        source_shards = shard_hashes(args.model)

    patch_path = None
    if args.patch_out:
        space.ws.save_patch(args.patch_out)
        patch_path = os.path.abspath(args.patch_out)
        stamp_patch_source(patch_path, source_shards)

    out_dir = None
    manifest_path = None
    if not args.dry_run:
        out_dir = save_model(space, args.out)
        recipe = {
            "op": args.op,
            "tensor": name,
            "scale": args.scale,
            "columns": columns,
            "index": args.index,
            "vector_file": os.path.abspath(args.vector),
        }
        manifest_path = write_manifest(args.out, args.model, recipe, report["changed_params"], patch_path, source_shards)

    payload = {
        "op": args.op,
        "tensor": name,
        "shape": applied["shape"],
        "scale": args.scale,
        "delta_norm": applied["delta_norm"],
        "rel_delta": applied["rel_delta"],
        "columns": columns,
        "out_dir": out_dir,
        "changed_params": report["changed_params"],
        "fraction": report["fraction"],
        "tensors": report["tensors"],
        "patch": patch_path,
        "manifest": manifest_path,
        "dry_run": args.dry_run,
    }
    # A bf16 weight grid has ~8 mantissa bits: a delta this small mostly rounds away.
    if report["changed_params"] == 0:
        payload["warning"] = (
            "no weights survived the bf16 cast — the perturbation is below the quantization step; "
            "increase --scale or use a direction with a larger norm"
        )
    emit(payload)


if __name__ == "__main__":
    main()
