#!/usr/bin/env python
"""Apply a sparse weight patch (`.npz` from apply_patch_and_save --patch-out) to a model.

Usage:
  python patch_apply.py --model <dir> --patch patch.npz [--out <dir>] [--dry-run] [--device cpu]

The patch stores only flat indices and new values per tensor, so it is a few KB
instead of a full model copy. Patches written by `apply_patch_and_save.py` also
record the SHA-256 of the source shards they were cut from; before applying, this
script verifies that --model matches those hashes and refuses on a mismatch
(replaying a patch onto the wrong checkpoint would silently edit the wrong
weights). Older patches without recorded hashes are applied with a warning.

Without --out (or with --dry-run) nothing is written; the run only reports what
the patch would do.

Output (JSON on stdout): {patch, tensors_in_patch, out_dir, changed_params,
                          fraction, tensors, source_verified, dry_run}
"""
import argparse
import os

from _common import emit, pick_device
from ws_core import diff_dict, open_space, save_model, sync, verify_patch_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", required=True, help="Sparse .npz patch written by --patch-out")
    ap.add_argument("--out", default=None, help="Output directory for the patched model")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run is given")

    verification = verify_patch_source(args.patch, args.model)
    if verification["verified"] is False:
        raise SystemExit(
            "refusing to apply: --model is not the checkpoint this patch was cut from. "
            f"Mismatched shard(s): {', '.join(verification['mismatch'])}. "
            "Point --model at the source recorded in the patch (see its .manifest.json / JSON output)."
        )

    space = open_space(args.model, pick_device(args.device).type)
    space.ws.apply_patch(args.patch)
    sync(space)
    report = diff_dict(space.ws)

    out_dir = None
    if not args.dry_run:
        out_dir = save_model(space, args.out)

    payload = {
        "patch": os.path.abspath(args.patch),
        "tensors_in_patch": [t["name"] for t in report["tensors"]],
        "out_dir": out_dir,
        "changed_params": report["changed_params"],
        "fraction": report["fraction"],
        "tensors": report["tensors"],
        "source_verified": verification["verified"],
        "dry_run": args.dry_run,
    }
    if verification["verified"] is None:
        payload["warning"] = "patch carries no recorded source hash; skipping target verification"
    emit(payload)


if __name__ == "__main__":
    main()
