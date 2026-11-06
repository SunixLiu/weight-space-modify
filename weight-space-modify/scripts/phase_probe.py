#!/usr/bin/env python
"""Phase-aligned analysis of saved probe activations (offline, no GPU).

Usage:
  python phase_probe.py --probes <dir> --pattern 'probe_r5_loop*.pt' \
      [--layer 15] [--slot mlp_mid] [--max-period 48] [--out-dir DIR] \
      [--ref REF.pt] [--ref-pattern 'v_l15_r5_*.pt']

For each probe .pt file (saved by probe_activations.py --save-activations):
  - estimate the loop period p* by autocorrelation of unit-normalized
    activations: score(p) = mean cos(x_t, x_{t+p}) over the second half,
    p in 1..max-period; p* = argmax
  - find the loop start t0: earliest step after which the p*-periodic match
    stays high (mean over a 96-step window > 0.9); fallback t0 = N/2
  - per-phase profile within the loop region: phase = (t - t0) mod p*;
    report phases with largest deviation from the pre-loop mean and
    largest jump from the previous phase ("trigger frame" candidates)
  - extract direction v = mean(loop region) - mean(pre-loop), unit-normalized

Pools directions across probes, optionally compares against reference
direction files (pooled --ref and/or per-probe --ref-pattern), and saves
per-probe + pooled direction .pt files to --out-dir.

Output: one JSON doc on stdout; progress on stderr.
"""
import argparse
import glob
import json
import os
import re

import torch


def load_trajectory(path, layer, slot):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    caps = blob["layers"][str(layer)]
    if slot not in caps or not caps[slot]:
        raise KeyError(f"slot {slot!r} missing for layer {layer} in {path}")
    return torch.stack(caps[slot]).float()


def estimate_period(U, max_period):
    """U: [N, D] unit rows. Returns (p*, score) or (None, best_score)."""
    N = U.shape[0]
    scores = {}
    for p in range(1, max_period + 1):
        if N - p < N // 2:
            continue
        a = U[N // 2: N - p]
        b = U[N // 2 + p: N]
        scores[p] = float((a * b).sum(dim=1).mean())
    if not scores:
        return None, 0.0
    p_star = max(scores, key=scores.get)
    return p_star, scores[p_star]


def find_loop_start(U, p, threshold=0.9, window=96):
    """Earliest t0 with mean cos(U[t], U[t+p]) > threshold over [t0, t0+window)."""
    N = U.shape[0]
    if p is None or N - p < window:
        return N // 2, True
    m = (U[: N - p] * U[p:]).sum(dim=1)
    for t0 in range(0, N - p - window + 1):
        if float(m[t0: t0 + window].mean()) > threshold:
            return t0, False
    return N // 2, True


def phase_profile(X, t0, p):
    """Per-phase means over the loop region; returns dict phase -> mean vector."""
    loop = X[t0:]
    phases = {}
    for i in range(loop.shape[0]):
        ph = i % p
        phases.setdefault(ph, []).append(loop[i])
    return {ph: torch.stack(v).mean(dim=0) for ph, v in phases.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", required=True, help="Directory containing probe .pt files")
    ap.add_argument("--pattern", default="*.pt", help="Glob pattern inside --probes")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--slot", default="mlp_mid", choices=["residual", "mlp", "attn", "mlp_mid"])
    ap.add_argument("--max-period", type=int, default=48)
    ap.add_argument("--out-dir", default=None, help="Save per-probe + pooled direction .pt files here")
    ap.add_argument("--ref", default=None, help="Reference pooled direction .pt for cosine comparison")
    ap.add_argument("--ref-pattern", default=None,
                    help="Per-probe reference pattern; number in probe filename selects the file")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.probes, args.pattern)))
    if not paths:
        raise SystemExit(f"no probe files match {args.probes}/{args.pattern}")

    ref_pooled = torch.load(args.ref, map_location="cpu") if args.ref else None
    ref_files = {}
    if args.ref_pattern:
        for f in glob.glob(os.path.join(args.probes, args.ref_pattern)):
            m = re.search(r"(\d+)\.pt$", os.path.basename(f))
            if m:
                ref_files[m.group(1)] = f

    probes_out = []
    dirs = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        X = load_trajectory(path, args.layer, args.slot)
        N, D = X.shape
        U = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
        p_star, score = estimate_period(U, args.max_period)
        t0, fallback = find_loop_start(U, p_star)
        pre = X[:t0].mean(dim=0) if t0 > 0 else torch.zeros(D)
        loop_mean = X[t0:].mean(dim=0)
        v = (loop_mean - pre)
        vn = v.norm()
        v_unit = v / vn.clamp_min(1e-8)
        dirs.append((stem, v_unit))

        entry = {
            "file": os.path.basename(path),
            "n_steps": N,
            "period": p_star,
            "period_score": round(score, 4),
            "loop_start": int(t0),
            "loop_len": int(N - t0),
            "fallback_loop_start": fallback,
            "dir_norm": round(float(vn), 4),
        }
        if p_star is not None and not fallback:
            prof = phase_profile(X, t0, p_star)
            dev = {ph: float(((mu - pre).norm() / pre.norm().clamp_min(1e-8))) for ph, mu in prof.items()}
            jump = {}
            for ph in prof:
                prev = prof[(ph - 1) % p_star]
                jump[ph] = float(((prof[ph] - prev).norm() / prev.norm().clamp_min(1e-8)))
            entry["phase_deviation_top3"] = [
                {"phase": ph, "rel_dev": round(x, 4)}
                for ph, x in sorted(dev.items(), key=lambda kv: -kv[1])[:3]
            ]
            entry["phase_jump_top3"] = [
                {"phase": ph, "rel_jump": round(x, 4)}
                for ph, x in sorted(jump.items(), key=lambda kv: -kv[1])[:3]
            ]
        probes_out.append(entry)
        print(f"{stem}: N={N} p*={p_star} score={score:.3f} t0={t0} fallback={fallback}",
              file=__import__("sys").stderr)

    pooled = torch.stack([d for _, d in dirs]).sum(dim=0)
    pooled = pooled / pooled.norm().clamp_min(1e-8)

    result = {
        "layer": args.layer,
        "slot": args.slot,
        "n_probes": len(probes_out),
        "probes": probes_out,
        "pooled_dir_norm": round(float(torch.stack([d for _, d in dirs]).sum(dim=0).norm()), 4),
    }

    if ref_pooled is not None:
        result["cos_pooled_vs_ref"] = round(float((pooled * ref_pooled).sum()), 4)
    per_ref = []
    for stem, v_unit in dirs:
        m = re.search(r"(\d+)$", stem)
        if m and m.group(1) in ref_files:
            rv = torch.load(ref_files[m.group(1)], map_location="cpu")
            per_ref.append({"probe": stem, "cos_vs_ref": round(float((v_unit * rv).sum()), 4)})
    if per_ref:
        result["per_probe_cos_vs_ref"] = per_ref

    ds = [d for _, d in dirs]
    if len(ds) > 1:
        cos = torch.stack(ds) @ torch.stack(ds).T
        iu = torch.triu_indices(len(ds), len(ds), 1)
        off = cos[iu[0], iu[1]]
        result["pairwise_cos"] = {
            "min": round(float(off.min()), 4),
            "max": round(float(off.max()), 4),
            "mean": round(float(off.mean()), 4),
        }

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for stem, v_unit in dirs:
            torch.save(v_unit, os.path.join(args.out_dir, f"v_phase_{stem}.pt"))
        torch.save(pooled, os.path.join(args.out_dir, "v_phase_pooled.pt"))
        print(f"saved directions to {args.out_dir}", file=__import__("sys").stderr)

    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
