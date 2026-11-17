#!/usr/bin/env python
"""Probe the full-attention layers' attention weights during greedy decoding.

Usage:
  python probe_attn_state.py --model <dir> --prompt-file P.txt [P2.txt ...] \
      [--max-new-tokens 512] [--layers all] [--out-dir D] [--device auto]

Runs ONE manual incremental greedy decode per prompt with an explicit
DynamicCache and eager attention (output_attentions=True). After each
generated token, the attention ROW of the new query position over all
previous positions is captured for every selected full-attention layer:
shape [H, T].

This answers "where does the model attend while looping": per step we know
the full distribution over prompt + previously generated tokens, per head.

Per prompt it saves a trajectory .pt (per-layer list of [H, t] tensors +
token list) and prints one combined JSON summary on stdout.
"""
import argparse
import os
import sys

import torch

from _common import (
    chat_input_ids,
    detect_ngram_repeat,
    detect_period,
    emit,
    first_repeat,
    load_tokenizer,
    pick_device,
)


def load_model_eager(model_path: str, device: torch.device):
    """Like _common.load_model but forces the eager attention implementation so
    output_attentions=True returns real attention matrices."""
    import glob
    import json

    import transformers
    from safetensors import safe_open
    from transformers import AutoConfig

    with open(f"{model_path}/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cls = getattr(transformers, cfg["architectures"][0])
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    model = cls(config)

    state = {}
    for path in sorted(glob.glob(f"{model_path}/*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if getattr(config, "tie_word_embeddings", False):
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        print(
            f"warning: {len(missing)} tensors missing from files (randomly initialized): "
            + ", ".join(missing[:5]),
            file=sys.stderr,
        )
    model.to(device, torch.bfloat16)
    model.eval()
    return model


def find_decoder_layers(model):
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


def mem_snapshot():
    """(sys_free_GiB, proc_working_set_MiB) via Win32 APIs."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
        ]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("dwJobType", ctypes.c_ulong),
            ("dwThrottleStatus", ctypes.c_ulong),
            ("PageFaults", ctypes.c_ulonglong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPageFile", ctypes.c_size_t),
            ("QuotaPageFile", ctypes.c_size_t),
            ("QuotaVirtualSpace", ctypes.c_size_t),
            ("QuotaPeakVirtualSpace", ctypes.c_size_t),
            ("QuotaReserved", ctypes.c_size_t),
            ("QuotaPeakReserved", ctypes.c_size_t),
            ("VirtualDeleted", ctypes.c_size_t),
        ]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    sys_free_gb = st.ullAvailPhys / 1e9
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.dwSize = ctypes.sizeof(pmc)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(pmc),
        ctypes.sizeof(pmc),
    )
    proc_mb = pmc.WorkingSetSize / 1e6
    return sys_free_gb, proc_mb


def probe_one(model, tokenizer, prompt: str, selected, attn_idx, device, max_new_tokens):
    """One incremental greedy decode; returns (result_dict, caps)."""
    from transformers.cache_utils import DynamicCache

    caps = {i: {"rows": []} for i in selected}
    cache = DynamicCache(config=model.config)
    prompt_ids = chat_input_ids(tokenizer, prompt)
    eos_id = tokenizer.eos_token_id or model.config.eos_token_id
    tokens = []

    with torch.no_grad():
        # Prefill: process the whole prompt once (seq_len > 1 path).
        out = model(
            torch.tensor([prompt_ids], device=device),
            use_cache=True,
            past_key_values=cache,
            output_attentions=True,
        )
        token = int(torch.argmax(out.logits[0, -1].float()))
        tokens.append(token)

        # Decode: pass ONLY the new single token each step (seq_len == 1 path),
        # so the cache is updated incrementally and memory stays O(T).
        cur = torch.tensor([[token]], device=device)
        for step in range(max_new_tokens - 1):
            out = model(cur, use_cache=True, past_key_values=cache, output_attentions=True)
            token = int(torch.argmax(out.logits[0, -1].float()))
            tokens.append(token)
            # The attention of the query just processed (position len(prompt)+step)
            # over all previous positions: [1, H, q_len, T] -> last row [H, T].
            for i in selected:
                attn = out.attentions[attn_idx[i]]
                if attn is None:
                    raise RuntimeError(f"no attentions returned for layer {i}")
                caps[i]["rows"].append(attn[0, :, -1, :].detach().float())
            cur = torch.tensor([[token]], device=device)
            if step % 32 == 31:
                sys_free_gb, proc_mb = mem_snapshot()
                alloc_mb = torch.cuda.memory_allocated(device) / 1e6
                reserved_mb = torch.cuda.memory_reserved(device) / 1e6
                print(
                    f"step={step + 1} proc_ws={proc_mb:.0f}MiB sys_free={sys_free_gb:.1f}GiB "
                    f"cuda_alloc={alloc_mb:.0f}MiB cuda_reserved={reserved_mb:.0f}MiB",
                    file=sys.stderr,
                )
            if token == eos_id:
                break

    result = {
        "prompt_chars": len(prompt),
        "prompt_tokens": len(prompt_ids),
        "n_tokens": len(tokens),
        "eos_hit": bool(tokens) and tokens[-1] == eos_id,
        "truncated": len(tokens) == max_new_tokens and (not tokens or tokens[-1] != eos_id),
        "loop": detect_period(tokens) is not None or detect_ngram_repeat(tokens) is not None,
        "period": detect_period(tokens),
        "first_repeat_at": first_repeat(tokens),
    }
    return result, caps, tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompt-file",
        required=True,
        nargs="+",
        help="One or more files containing prompt texts (model loaded once)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--layers", default="all", help="'all' (full-attention layers only) or comma list")
    ap.add_argument("--out-dir", default=None, help="Save trajectory here")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = pick_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model_eager(args.model, device)
    layers = find_decoder_layers(model)

    full_layers = [i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")]
    if args.layers == "all":
        selected = full_layers
    else:
        selected = [int(x) for x in args.layers.split(",") if x.strip()]
        bad = [i for i in selected if i not in full_layers]
        if bad:
            ap.error(f"layers {bad} are not full-attention layers (full: {full_layers})")

    # out.attentions is a tuple with one entry per FULL-attention layer only
    # (GDN layers return none), ordered as full_layers.
    attn_idx = {layer_i: k for k, layer_i in enumerate(full_layers)}

    results = []
    for prompt_file in args.prompt_file:
        prompt = open(prompt_file, "r", encoding="utf-8").read().strip()
        stem = os.path.splitext(os.path.basename(prompt_file))[0]
        print(f"probing {prompt_file} ...", file=sys.stderr)
        result, caps, tokens = probe_one(model, tokenizer, prompt, selected, attn_idx, device, args.max_new_tokens)
        result["prompt_file"] = prompt_file
        results.append(result)

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(
                {
                    "prompt": prompt,
                    "tokens": tokens,
                    "prompt_len": result["prompt_tokens"],
                    "layers": {str(i): {"attn_rows": caps[i]["rows"]} for i in selected},
                },
                os.path.join(args.out_dir, f"attn_{stem}.pt"),
            )
            print(f"saved trajectory to {os.path.join(args.out_dir, 'attn_' + stem + '.pt')}", file=sys.stderr)

    emit({"model": args.model, "full_layers": full_layers, "probed": selected, "prompts": results})


if __name__ == "__main__":
    main()
