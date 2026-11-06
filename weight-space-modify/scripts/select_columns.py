#!/usr/bin/env python
"""Select intermediate-neuron columns for a column-restricted edit from probe evidence.

Usage:
  python select_columns.py --loop probe_loop.pt [--loop probe_loop2.pt ...] \
      --control probe_ctrl.pt --layer 20 [--top-n 64] [--mode loop-abs|delta] \
      [--out-columns cols.json]

Ranks the 3584 intermediate neurons (the input columns of `mlp.down_proj`) using
the `mlp_mid` activations captured by `probe_activations.py`:

  - loop_abs[j]: mean |h_j| over the last quarter of tokens, averaged across all
    --loop files — how strongly neuron j fires in the established loop state;
  - delta[j]:    loop_abs[j] - ctrl_abs[j] — how much more it fires in the loop
    than in a successful run.

--mode loop-abs picks the top-N by loop_abs (the reference project / paper form:
"the intermediate neurons with the largest activation on looping samples");
--mode delta picks the top-N by loop-vs-control contrast. The selected indices are
written to --out-columns as a JSON list, ready for `apply_patch_and_save.py
--columns`.

Output (JSON on stdout): {layer, mode, top_n, columns, stats} where stats give the
score of the selected columns and of the median unselected column, so you can see
whether the selection is actually separated from the rest of the matrix.
"""
import argparse
import json

import torch

from _common import emit


def last_quarter_abs_mean(path: str, layer: int) -> torch.Tensor:
    """Per-neuron mean |activation| over the last quarter of tokens in one probe file."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    vecs = blob["layers"][str(layer)]["mlp_mid"]
    if not vecs:
        raise SystemExit(f"no mlp_mid activations for layer {layer} in {path}")
    x = torch.stack(vecs).float()  # [steps, 3584]
    k = max(1, x.shape[0] // 4)
    return x[-k:].abs().mean(dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="append", required=True, help="Probe .pt file of a looping run (repeatable)")
    ap.add_argument("--control", required=True, help="Probe .pt file of a successful run")
    ap.add_argument("--layer", required=True, type=int, help="Layer index as captured by the probe")
    ap.add_argument("--top-n", type=int, default=64)
    ap.add_argument("--mode", default="loop-abs", choices=["loop-abs", "delta"])
    ap.add_argument("--out-columns", default=None, help="Write the selected indices here as a JSON list")
    args = ap.parse_args()

    loop_scores = [last_quarter_abs_mean(p, args.layer) for p in args.loop]
    ctrl_score = last_quarter_abs_mean(args.control, args.layer)
    loop_score = torch.stack(loop_scores).mean(dim=0)  # pooled across loop runs
    score = loop_score if args.mode == "loop-abs" else (loop_score - ctrl_score)

    n = int(score.numel())
    top_n = min(args.top_n, n)
    order = torch.argsort(score, descending=True)
    cols = [int(i) for i in order[:top_n]]
    selected = score[order[:top_n]]
    unselected = score[order[top_n:]]

    stats = {
        "n_neurons": n,
        "selected_min_score": round(float(selected.min()), 6),
        "selected_max_score": round(float(selected.max()), 6),
        "unselected_median_score": round(float(unselected.median()), 6) if unselected.numel() else None,
        "loop_abs_selected_mean": round(float(loop_score[torch.tensor(cols)].mean()), 6),
        "ctrl_abs_selected_mean": round(float(ctrl_score[torch.tensor(cols)].mean()), 6),
    }

    if args.out_columns:
        with open(args.out_columns, "w", encoding="utf-8") as f:
            json.dump(cols, f)

    emit({"layer": args.layer, "mode": args.mode, "top_n": top_n, "columns": cols, "stats": stats})


if __name__ == "__main__":
    main()
