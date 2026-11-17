#!/usr/bin/env python
"""Probe intermediate activations while decoding a prompt, and flag anomalies.

Usage:
  python probe_activations.py --model <dir> --prompt "..." \
      [--layers all] [--max-new-tokens 128] [--temperature 0.0] \
      [--save-activations OUT.pt] [--device auto]

Captures per generated token, for each selected decoder layer:
  - residual stream (decoder layer output): L2 norm, cosine vs previous token
  - MLP block output: mean, std, zero fraction (saturation), max |x|
  - MLP intermediate (SwiGLU activation feeding down_proj, 3584-dim): mean, std,
    zero fraction, max |x| — needed to select individual intermediate neurons
    for column-restricted edits (`apply_patch_and_save.py --columns`)
  - attention projection output (o_proj / linear_attn.out_proj): L2 norm

Anomaly flags:
  - stuck:    cos(residual_t, residual_{t-1}) > 0.995 for >= 8 consecutive tokens
  - spike:    residual norm > 4x the median norm of that layer
  - saturated: MLP zero fraction > 0.5

Output (JSON on stdout): summary + per-layer stats + top anomalies.
With --save-activations, full per-token tensors are saved to a .pt file for
deeper analysis (e.g. extracting feature directions for apply_patch_and_save).
"""
import argparse
import json

import torch

from _common import chat_input_ids, emit, load_model, load_tokenizer, pick_device


def find_decoder_layers(model):
    """Locate the decoder-layer ModuleList, whatever the nesting (e.g.
    Qwen3_5ForConditionalGeneration -> .model -> .language_model.layers)."""
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (name == "layers" or name.endswith(".layers")):
            candidates.append((name, mod))
    if not candidates:
        raise RuntimeError("could not locate decoder layers on the model")
    for name, mod in candidates:
        if "language_model" in name:
            return list(mod)
    return list(candidates[0][1])


def layer_type(layer) -> str:
    if hasattr(layer, "linear_attn"):
        return "linear_attention"
    if hasattr(layer, "self_attn"):
        return "full_attention"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=None, help="Prompt text (or use --prompt-file)")
    ap.add_argument("--prompt-file", default=None, help="File containing the prompt text")
    ap.add_argument("--layers", default="all", help="'all' or comma list, e.g. 0,4,8,12")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--save-activations", default=None, help="Save full trajectory to this .pt file")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.prompt is None and args.prompt_file is None:
        ap.error("provide --prompt or --prompt-file")
    prompt = args.prompt if args.prompt is not None else open(args.prompt_file, "r", encoding="utf-8").read().strip()

    device = pick_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device)
    layers = find_decoder_layers(model)

    if args.layers == "all":
        selected = list(range(len(layers)))
    else:
        selected = [int(x) for x in args.layers.split(",") if x.strip()]
        bad = [i for i in selected if i < 0 or i >= len(layers)]
        if bad:
            ap.error(f"layer index out of range (0..{len(layers) - 1}): {bad}")

    # Hook storage: layer_idx -> {"residual": [...], "mlp": [...], "mlp_mid": [...], "attn": [...], "gate": [...]}
    caps = {i: {"residual": [], "mlp": [], "mlp_mid": [], "attn": [], "gate": []} for i in selected}
    logit_stats = []

    def make_hook(i, slot):
        def hook(_module, _inputs, output):
            caps[i][slot].append(output[0, -1].detach().float())
        return hook

    handles = []
    for i in selected:
        layer = layers[i]
        handles.append(layer.register_forward_hook(make_hook(i, "residual")))
        if hasattr(layer, "mlp"):
            handles.append(layer.mlp.register_forward_hook(make_hook(i, "mlp")))
            # The SwiGLU intermediate (down_proj's input) is the 3584-dim
            # activation of the individual intermediate neurons.
            down_proj = getattr(layer.mlp, "down_proj", None)
            if down_proj is not None:
                # Factory (not a bare closure over the loop variable): each hook must
                # bind its own layer index, or every layer writes into the last one.
                def make_mid_hook(idx):
                    def hook(_module, inputs):
                        caps[idx]["mlp_mid"].append(inputs[0][0, -1].detach().float())

                    return hook

                handles.append(down_proj.register_forward_pre_hook(make_mid_hook(i)))
        attn_proj = getattr(getattr(layer, "linear_attn", None), "out_proj", None) or getattr(
            getattr(layer, "self_attn", None), "o_proj", None
        )
        if attn_proj is not None:
            handles.append(attn_proj.register_forward_hook(make_hook(i, "attn")))
        # GDN gate proxy: the in_proj_a output drives the decay gate.
        in_proj_a = getattr(getattr(layer, "linear_attn", None), "in_proj_a", None)
        if in_proj_a is not None:
            handles.append(in_proj_a.register_forward_hook(make_hook(i, "gate")))

    input_ids = torch.tensor([chat_input_ids(tokenizer, prompt)], device=device)
    eos_id = tokenizer.eos_token_id or model.config.eos_token_id
    greedy = args.temperature <= 0.0

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            out = model(input_ids)
            logits = out.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            ent = float(-(probs * torch.log(probs + 1e-12)).sum())
            eos_logit = float(logits[eos_id]) if eos_id is not None else None
            eos_rank = int((logits > logits[eos_id]).sum()) if eos_id is not None else None
            top1_logit = float(torch.topk(logits, 1).values[0])
            top5 = torch.topk(logits, 5).indices.tolist()
            logit_stats.append(
                {
                    "token": None,  # filled below
                    "eos_logit": round(eos_logit, 3) if eos_logit is not None else None,
                    "eos_rank": eos_rank,
                    "top1_logit": round(top1_logit, 3),
                    "entropy": round(ent, 3),
                    "top5": top5,
                }
            )
            if greedy:
                token = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                token = int(torch.multinomial(probs, 1))
            logit_stats[-1]["token"] = token
            input_ids = torch.cat([input_ids, torch.tensor([[token]], device=device)], dim=1)
            if token == eos_id:
                break

    for h in handles:
        h.remove()

    n_steps = len(caps[selected[0]]["residual"])
    for i in selected:
        if caps[i]["mlp_mid"] and len(caps[i]["mlp_mid"]) != n_steps:
            raise RuntimeError(
                f"layer {i}: mlp_mid has {len(caps[i]['mlp_mid'])} entries but residual has {n_steps}; "
                "hook wiring is wrong (each layer must capture its own intermediate)"
            )
    per_layer = {}
    anomalies = []
    for i in selected:
        res = torch.stack(caps[i]["residual"]) if caps[i]["residual"] else None
        mlp = torch.stack(caps[i]["mlp"]) if caps[i]["mlp"] else None
        attn = torch.stack(caps[i]["attn"]) if caps[i]["attn"] else None

        entry = {"index": i, "type": layer_type(layers[i]), "steps": n_steps}
        # structural integrity: NaN / Inf / extreme values
        for slot, tensor in (("residual", res), ("mlp", mlp), ("attn", attn)):
            if tensor is None:
                continue
            entry.setdefault("integrity", {})[slot] = {
                "nan": int(torch.isnan(tensor).sum()),
                "inf": int(torch.isinf(tensor).sum()),
                "max_abs": round(float(tensor.abs().max()), 3),
            }
            if entry["integrity"][slot]["nan"] or entry["integrity"][slot]["inf"]:
                anomalies.append({"layer": i, "kind": "nan_inf",
                                  "detail": f"{slot}: nan={entry['integrity'][slot]['nan']} inf={entry['integrity'][slot]['inf']}"})
        # variance explosion: last quarter vs first quarter of the trajectory
        for slot, tensor in (("residual", res), ("mlp", mlp)):
            if tensor is None or tensor.shape[0] < 8:
                continue
            q = max(2, tensor.shape[0] // 4)
            v_first = float(tensor[:q].var(dim=0).mean())
            v_last = float(tensor[-q:].var(dim=0).mean())
            ratio = v_last / (v_first + 1e-9)
            entry.setdefault("variance", {})[slot] = {
                "first_q": round(v_first, 6),
                "last_q": round(v_last, 6),
                "ratio": round(ratio, 2),
            }
            if ratio > 10.0:
                anomalies.append({"layer": i, "kind": "variance_explosion",
                                  "detail": f"{slot}: variance x{ratio:.1f} (first→last quarter)"})
        if res is not None:
            norms = res.norm(dim=1)
            med = float(norms.median())
            entry["residual_norm"] = {"mean": round(float(norms.mean()), 3), "median": round(med, 3),
                                      "max": round(float(norms.max()), 3)}
            # cosine vs previous token
            cos = []
            for t in range(1, len(norms)):
                a, b = res[t - 1], res[t]
                cos.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)))
            entry["cos_prev_mean"] = round(sum(cos) / len(cos), 5) if cos else None
            # stuck detection
            run = best = 0
            for c in cos:
                run = run + 1 if c > 0.995 else 0
                best = max(best, run)
            entry["stuck_run_max"] = best
            if best >= 8:
                anomalies.append({"layer": i, "kind": "stuck", "detail": f"cos>0.995 for {best} consecutive tokens"})
            # spike detection
            spikes = [t for t in range(len(norms)) if norms[t] > 4 * med]
            entry["spike_tokens"] = spikes[:20]
            if spikes:
                anomalies.append({"layer": i, "kind": "spike", "detail": f"{len(spikes)} tokens with norm > 4x median"})
        if mlp is not None:
            zero_frac = float((mlp.abs() < 1e-6).float().mean())
            entry["mlp"] = {"mean": round(float(mlp.mean()), 5), "std": round(float(mlp.std()), 5),
                           "zero_frac": round(zero_frac, 4), "max_abs": round(float(mlp.abs().max()), 3)}
            if zero_frac > 0.5:
                anomalies.append({"layer": i, "kind": "saturated", "detail": f"MLP zero fraction {zero_frac:.2f}"})
        mlp_mid = torch.stack(caps[i]["mlp_mid"]) if caps[i]["mlp_mid"] else None
        if mlp_mid is not None:
            entry["mlp_mid"] = {"dim": int(mlp_mid.shape[1]), "mean": round(float(mlp_mid.mean()), 5),
                               "std": round(float(mlp_mid.std()), 5),
                               "zero_frac": round(float((mlp_mid.abs() < 1e-6).float().mean()), 4),
                               "max_abs": round(float(mlp_mid.abs().max()), 3)}
        if caps[i]["gate"]:
            g = torch.stack(caps[i]["gate"]).mean(dim=1)
            entry["gate"] = {"mean": round(float(g.mean()), 4), "first_quarter": round(float(g[: max(1, len(g) // 4)].mean()), 4),
                             "last_quarter": round(float(g[-max(1, len(g) // 4):].mean()), 4)}
        per_layer[str(i)] = entry

    # EOS / entropy trajectory (H1/H3 evidence)
    eos_series = [s["eos_logit"] for s in logit_stats if s["eos_logit"] is not None]
    ent_series = [s["entropy"] for s in logit_stats]
    logits_summary = {"steps": len(logit_stats)}
    if eos_series:
        # how often EOS is buried far below the top-1 token
        suppressed = sum(
            1 for s in logit_stats
            if s["eos_logit"] is not None and (
                (s["eos_rank"] is not None and s["eos_rank"] > 100)
                or s["eos_logit"] < s["top1_logit"] - 10.0
            )
        )
        logits_summary["eos_suppressed_steps"] = suppressed
        logits_summary["eos"] = {
            "mean": round(sum(eos_series) / len(eos_series), 3),
            "min": round(min(eos_series), 3),
            "max": round(max(eos_series), 3),
            "first_quarter_mean": round(sum(eos_series[: max(1, len(eos_series) // 4)]) / max(1, len(eos_series) // 4), 3),
            "last_quarter_mean": round(sum(eos_series[-max(1, len(eos_series) // 4):]) / max(1, len(eos_series) // 4), 3),
        }
        if logits_summary["eos"]["last_quarter_mean"] < logits_summary["eos"]["first_quarter_mean"] - 1.0:
            anomalies.append({"layer": "lm_head", "kind": "eos_suppressed",
                              "detail": f"EOS logit fell from {logits_summary['eos']['first_quarter_mean']} to {logits_summary['eos']['last_quarter_mean']}"})
    if ent_series:
        logits_summary["entropy"] = {
            "mean": round(sum(ent_series) / len(ent_series), 3),
            "first_quarter_mean": round(sum(ent_series[: max(1, len(ent_series) // 4)]) / max(1, len(ent_series) // 4), 3),
            "last_quarter_mean": round(sum(ent_series[-max(1, len(ent_series) // 4):]) / max(1, len(ent_series) // 4), 3),
        }
        if logits_summary["entropy"]["last_quarter_mean"] >= logits_summary["entropy"]["first_quarter_mean"]:
            anomalies.append({"layer": "lm_head", "kind": "entropy_plateau",
                              "detail": "next-token entropy did not decline in the last quarter"})
    # decode top-5 tokens at start / mid / end for human reading
    picks = sorted({0, len(logit_stats) // 2, len(logit_stats) - 1})
    logits_summary["top5_probe"] = [
        {"step": p, "tokens": [tokenizer.decode([t]) for t in logit_stats[p]["top5"]]}
        for p in picks if 0 <= p < len(logit_stats)
    ]

    result = {
        "model": args.model,
        "prompt_chars": len(prompt),
        "steps": n_steps,
        "layers_probed": selected,
        "per_layer": per_layer,
        "logits": logits_summary,
        "anomalies": anomalies,
    }
    emit(result)

    if args.save_activations:
        torch.save(
            {"prompt": prompt, "layers": {str(i): caps[i] for i in selected}},
            args.save_activations,
        )
        print(f"saved activations to {args.save_activations}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
