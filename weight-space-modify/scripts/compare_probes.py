#!/usr/bin/env python
"""Compare two probe_activations reports (loop vs success) and rank suspect layers.

Usage:
  python compare_probes.py --loop probe_loop.json --control probe_ctrl.json [--out compare.json]

For every layer probed in both runs it computes loop/control deltas:
  - residual norm ratio, variance ratio, cos-prev delta
  - MLP zero-fraction delta, max-abs ratio
  - GDN gate mean delta (linear layers)
  - EOS logit drop, entropy plateau (run-level)

Layers are ranked by an anomaly score (higher = more structurally different
during the loop). Output (JSON on stdout) lists per-layer deltas + top_suspects.
"""
import argparse
import json
import math

from _common import emit


def _get(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def score_layer(loop: dict, ctrl: dict) -> dict:
    """Signed anomaly score for one layer; >0 means the loop run deviates."""
    lr = _get(loop, "residual_norm", "mean")
    cr = _get(ctrl, "residual_norm", "mean")
    lc = _get(loop, "cos_prev_mean")
    cc = _get(ctrl, "cos_prev_mean")
    lv = _get(loop, "variance", "residual", "ratio")
    cv = _get(ctrl, "variance", "residual", "ratio")
    lz = _get(loop, "mlp", "zero_frac")
    cz = _get(ctrl, "mlp", "zero_frac")
    lm = _get(loop, "mlp", "max_abs")
    cm = _get(ctrl, "mlp", "max_abs")
    lg = _get(loop, "gate", "mean")
    cg = _get(ctrl, "gate", "mean")

    d = {
        "layer": loop.get("index"),
        "type": loop.get("type"),
        "norm_ratio": round(lr / cr, 3) if lr and cr else None,
        "cos_prev_delta": round(lc - cc, 4) if lc is not None and cc is not None else None,
        "var_ratio_loop_vs_ctrl": round(lv / cv, 3) if lv and cv else None,
        "mlp_zero_delta": round(lz - cz, 4) if lz is not None and cz is not None else None,
        "mlp_maxabs_ratio": round(lm / cm, 3) if lm and cm else None,
        "gate_delta": round(lg - cg, 4) if lg is not None and cg is not None else None,
        "stuck_run_max": loop.get("stuck_run_max"),
        "anomalies": [a["kind"] for a in loop.get("anomalies", []) if a.get("layer") == loop.get("index")],
    }
    # weighted, log-scaled deviations; positive = loop-side excursion
    s = 0.0
    for key, w in (("norm_ratio", 1.0), ("var_ratio_loop_vs_ctrl", 0.7),
                   ("mlp_maxabs_ratio", 0.8)):
        v = d.get(key)
        if v:
            s += w * abs(math.log(v))
    if d["cos_prev_delta"] is not None:
        s += 2.0 * max(0.0, d["cos_prev_delta"])
    if d["gate_delta"] is not None:
        s += 2.0 * abs(d["gate_delta"])
    d["score"] = round(s, 3)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", required=True, help="probe JSON of the looping run")
    ap.add_argument("--control", required=True, help="probe JSON of the successful run")
    ap.add_argument("--out", default=None, help="Optional path to also save the JSON")
    args = ap.parse_args()

    loop = json.load(open(args.loop, encoding="utf-8"))
    ctrl = json.load(open(args.control, encoding="utf-8"))
    lp, cp = loop.get("per_layer", {}), ctrl.get("per_layer", {})

    rows = [score_layer(lp[k], cp[k]) for k in lp if k in cp]
    rows.sort(key=lambda r: r["score"], reverse=True)

    run_level = {
        "loop_steps": loop.get("steps"),
        "ctrl_steps": ctrl.get("steps"),
        "loop_eos": loop.get("logits", {}).get("eos"),
        "ctrl_eos": ctrl.get("logits", {}).get("eos"),
        "loop_eos_suppressed_steps": loop.get("logits", {}).get("eos_suppressed_steps"),
        "ctrl_eos_suppressed_steps": ctrl.get("logits", {}).get("eos_suppressed_steps"),
        "loop_entropy": loop.get("logits", {}).get("entropy"),
        "ctrl_entropy": ctrl.get("logits", {}).get("entropy"),
        "loop_anomalies": loop.get("anomalies", []),
        "ctrl_anomalies": ctrl.get("anomalies", []),
    }
    out = {
        "loop": args.loop,
        "control": args.control,
        "run_level": run_level,
        "top_suspects": [r["layer"] for r in rows[:6]],
        "per_layer": rows,
    }
    emit(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"saved comparison to {args.out}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
