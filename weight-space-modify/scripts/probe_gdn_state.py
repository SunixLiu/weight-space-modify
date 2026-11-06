#!/usr/bin/env python
"""Probe the GDN (gated delta network) recurrent state during greedy decoding.

Usage:
  python probe_gdn_state.py --model <dir> --prompt-file P.txt \
      [--max-new-tokens 512] [--layers all] [--out-dir D] [--device auto]

Runs ONE manual incremental greedy decode with an explicit DynamicCache; after
each generated token the per-layer recurrent state S_t is read directly from the
cache (`cache.layers[i].recurrent_states[0]`, shape [1, H, d_k, d_v]). All
selected layers are captured in the same single decode pass.

Per GDN layer and per step it records:
  - state_norm:     Frobenius norm of S_t
  - head_norm_mean: mean over heads of each head's Frobenius norm
  - decay:          PER-HEAD vector [16] of exp(g_t), g = -exp(A_log) * softplus(a + dt_bias)
                    (hook on in_proj_a; the fraction of previous state retained)
  - beta:           PER-HEAD vector [16] of sigmoid(b) (hook on in_proj_b; delta-update strength)
  - a_log / dt_bias: the layer's per-head gate parameters, saved for offline analysis

Output: one JSON doc on stdout (per-prompt summary); with --out-dir, a
trajectory file `gdn_<stem>.pt` for offline analysis.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

from _common import chat_input_ids, emit, load_model, load_tokenizer, pick_device


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True, help="File containing the prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--layers", default="all", help="'all' (GDN layers only) or comma list")
    ap.add_argument("--out-dir", default=None, help="Save trajectory here")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    prompt = open(args.prompt_file, "r", encoding="utf-8").read().strip()
    device = pick_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device)
    layers = find_decoder_layers(model)

    gdn_layers = [i for i, layer in enumerate(layers) if hasattr(layer, "linear_attn")]
    if args.layers == "all":
        selected = gdn_layers
    else:
        selected = [int(x) for x in args.layers.split(",") if x.strip()]
        bad = [i for i in selected if i not in gdn_layers]
        if bad:
            ap.error(f"layers {bad} are not GDN layers (GDN: {gdn_layers})")

    from transformers.cache_utils import DynamicCache

    # Per-layer capture buffers + hooks, all active during the single decode pass.
    caps = {i: {"state_norm": [], "head_norm_mean": [], "decay": [], "beta": []} for i in selected}
    handles = []
    for i in selected:
        layer = layers[i]
        a_log = layer.linear_attn.A_log.detach().float()
        dt_bias = layer.linear_attn.dt_bias.detach().float()
        caps[i]["a_log"] = a_log
        caps[i]["dt_bias"] = dt_bias

        def make_a_hook(idx):
            def hook(_module, _inputs, output):
                a = output[0, -1].detach().float()
                g = -a_log.exp() * F.softplus(a + dt_bias)
                caps[idx]["decay"].append(torch.exp(g))  # per-head [16]
            return hook

        def make_b_hook(idx):
            def hook(_module, _inputs, output):
                b = output[0, -1].detach().float()
                caps[idx]["beta"].append(torch.sigmoid(b))  # per-head [16]
            return hook

        handles.append(layer.linear_attn.in_proj_a.register_forward_hook(make_a_hook(i)))
        handles.append(layer.linear_attn.in_proj_b.register_forward_hook(make_b_hook(i)))

    stem = os.path.splitext(os.path.basename(args.prompt_file))[0]
    cache = DynamicCache(config=model.config)
    prompt_ids = chat_input_ids(tokenizer, prompt)
    eos_id = tokenizer.eos_token_id or model.config.eos_token_id
    tokens = []

    with torch.no_grad():
        # Prefill: process the whole prompt once (seq_len > 1 path).
        out = model(torch.tensor([prompt_ids], device=device), use_cache=True, past_key_values=cache)
        token = int(torch.argmax(out.logits[0, -1].float()))
        tokens.append(token)
        for i in selected:
            state = cache.layers[i].recurrent_states[0]
            if state is not None:
                s = state.float()
                caps[i]["state_norm"].append(s.norm())
                caps[i]["head_norm_mean"].append(
                    s.reshape(-1, s.shape[-2], s.shape[-1]).norm(dim=(1, 2)).mean()
                )

        # Decode: pass ONLY the new single token each step (seq_len == 1 path),
        # so the cache is updated incrementally and memory stays O(T).
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
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

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

        def mem_snapshot():
            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            sys_free_gb = st.ullAvailPhys / 1e9
            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.dwSize = ctypes.sizeof(pmc)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc),
                ctypes.sizeof(pmc),
            )
            proc_mb = pmc.WorkingSetSize / 1e6
            return sys_free_gb, proc_mb

        cur = torch.tensor([[token]], device=device)
        for step in range(args.max_new_tokens - 1):
            out = model(cur, use_cache=True, past_key_values=cache)
            token = int(torch.argmax(out.logits[0, -1].float()))
            tokens.append(token)
            for i in selected:
                state = cache.layers[i].recurrent_states[0]
                if state is not None:
                    s = state.float()
                    caps[i]["state_norm"].append(s.norm())
                    caps[i]["head_norm_mean"].append(
                        s.reshape(-1, s.shape[-2], s.shape[-1]).norm(dim=(1, 2)).mean()
                    )
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

    for h in handles:
        h.remove()

    result = {"model": args.model, "prompt_chars": len(prompt), "gdn_layers": gdn_layers, "probed": selected}
    for i in selected:
        c = caps[i]
        n = len(c["state_norm"])
        entry = {"steps": n}
        if n:
            sn = torch.stack(c["state_norm"], dim=0)
            hn = torch.stack(c["head_norm_mean"], dim=0)
            dec = torch.stack(c["decay"], dim=0)  # [T, 16]
            bet = torch.stack(c["beta"], dim=0)  # [T, 16]
            q = max(1, n // 4)
            entry["state_norm"] = {
                "first_q_mean": round(float(sn[:q].mean()), 4),
                "last_q_mean": round(float(sn[-q:].mean()), 4),
                "growth_last_over_first": round(float(sn[-q:].mean() / (sn[:q].mean() + 1e-9)), 3),
                "max": round(float(sn.max()), 4),
            }
            entry["head_norm_mean"] = {
                "first_q_mean": round(float(hn[:q].mean()), 4),
                "last_q_mean": round(float(hn[-q:].mean()), 4),
            }
            entry["decay"] = {
                "first_q_mean": round(float(dec[:q].mean()), 5),
                "last_q_mean": round(float(dec[-q:].mean()), 5),
            }
            entry["beta"] = {
                "first_q_mean": round(float(bet[:q].mean()), 5),
                "last_q_mean": round(float(bet[-q:].mean()), 5),
            }
        result[str(i)] = entry

    emit(result)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(
            {
                "prompt": prompt,
                "tokens": tokens,
                "layers": {
                    str(i): {
                        "state_norm": torch.tensor(caps[i]["state_norm"]),
                        "head_norm_mean": torch.tensor(caps[i]["head_norm_mean"]),
                        "decay": torch.stack(caps[i]["decay"], dim=0),  # [T, 16]
                        "beta": torch.stack(caps[i]["beta"], dim=0),  # [T, 16]
                        "a_log": caps[i]["a_log"],
                        "dt_bias": caps[i]["dt_bias"],
                    }
                    for i in selected
                },
            },
            os.path.join(args.out_dir, f"gdn_{stem}.pt"),
        )
        print(f"saved trajectory to {os.path.join(args.out_dir, 'gdn_' + stem + '.pt')}", file=sys.stderr)


if __name__ == "__main__":
    main()
