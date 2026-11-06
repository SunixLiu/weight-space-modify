"""Shared helpers for weight-space-modify scripts.

Loads a HuggingFace model directory (config.json + safetensors + tokenizer)
with the architecture class named in config.json, so hybrid/multimodal
models (e.g. Qwen3_5ForConditionalGeneration) load without special-casing.
"""
import json
import sys

import torch
from transformers import AutoTokenizer


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device: torch.device):
    """Load the model using the architecture class from config.json.

    Tensors are read directly from every *.safetensors file in the directory,
    so renamed shards / missing index files do not break loading.
    """
    import glob

    import transformers
    from safetensors import safe_open
    from transformers import AutoConfig

    with open(f"{model_path}/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cls = getattr(transformers, cfg["architectures"][0])
    config = AutoConfig.from_pretrained(model_path)
    model = cls(config)

    state = {}
    for path in sorted(glob.glob(f"{model_path}/*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tied embeddings: lm_head.weight is a view of embed_tokens.weight and has
    # no separate file entry — not actually missing.
    if getattr(config, "tie_word_embeddings", False):
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        print(f"warning: {len(missing)} tensors missing from files (randomly initialized): "
              + ", ".join(missing[:5]), file=sys.stderr)
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected tensors ignored: "
              + ", ".join(unexpected[:5]), file=sys.stderr)

    model.to(device, torch.bfloat16)
    model.eval()
    return model


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path)


def chat_input_ids(tokenizer, prompt: str) -> list[int]:
    """Tokenize a single user turn via the model's chat template."""
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, (list, tuple)):  # BatchEncoding / dict in transformers 5.x
        ids = ids["input_ids"]
    return list(ids)


def detect_period(tokens: list[int], max_p: int = 48, min_len: int = 32) -> int | None:
    """Smallest period p such that the last L tokens are p-periodic.

    Returns p or None. A dead loop shows up as a short period (1..~16).
    """
    n = len(tokens)
    if n < min_len:
        return None
    L = min(n, 96)
    start = n - L
    for p in range(1, max_p + 1):
        if L < 2 * p + 8:
            continue
        if all(tokens[i] == tokens[i - p] for i in range(start + p, n)):
            return p
    return None


def detect_ngram_repeat(tokens: list[int], ngram: int = 8, min_count: int = 3, window: int = 256) -> tuple | None:
    """Most frequent n-gram in the last `window` tokens, if it repeats >= min_count."""
    from collections import Counter

    t = tokens[-window:]
    if len(t) < ngram + min_count:
        return None
    c = Counter(tuple(t[i : i + ngram]) for i in range(len(t) - ngram + 1))
    gram, count = c.most_common(1)[0]
    return gram if count >= min_count else None


def first_repeat(tokens: list[int], ngram: int = 8, count: int = 3, window: int = 256) -> int | None:
    """Earliest index at which an n-gram has appeared `count` times (loop onset)."""
    seen: dict = {}
    for i in range(len(tokens) - ngram + 1):
        w = tokens[max(0, i - window + ngram) : i + ngram]
        g = tuple(w[-ngram:])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= count:
            return i
    return None


def gen_metrics(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 384,
    eos_id: int | None = None,
    tail_tokens: int = 0,
) -> dict:
    """Greedy generation on one prompt plus the loop/stop fingerprint.

    loop:      periodic tail (period <= 48) or an 8-gram repeated >= 3x
    eos_hit:   the run stopped by itself (good)
    truncated: hit the token cap without EOS (near-loop / too slow)
    first_repeat_at: where repetition starts; later is better
    tail_tokens: when > 0, also decode the last N generated tokens into `text_tail`
    """
    ids = chat_input_ids(tokenizer, prompt)
    if eos_id is None:
        eos_id = tokenizer.eos_token_id or model.config.eos_token_id
    with torch.no_grad():
        seq = model.generate(
            torch.tensor([ids], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    tokens = seq[0].tolist()[len(ids) :]
    period = detect_period(tokens)
    ngram = detect_ngram_repeat(tokens)
    out = {
        "n_tokens": len(tokens),
        "eos_hit": bool(tokens) and tokens[-1] == eos_id,
        "truncated": len(tokens) == max_new_tokens and (not tokens or tokens[-1] != eos_id),
        "loop": period is not None or ngram is not None,
        "period": period,
        "first_repeat_at": first_repeat(tokens),
    }
    if tail_tokens:
        out["text_tail"] = tokenizer.decode(tokens[-tail_tokens:])
    return out


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} content with brace matching, or None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    j = i
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return text[i : j - 1]


def extract_answer(text: str) -> str | None:
    """Predicted final answer: \\boxed{...} first, then common fallbacks."""
    import re

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    # "**Answer:** ...", "Answer: ...", "the answer is ..." — last occurrence
    m = None
    for pat in (r"\*\*Answer:\*\*\s*([^\n]+)", r"(?i)\banswer\s*[:：]\s*([^\n]+)",
                r"(?i)\bthe answer is\s*([^\n.]+)"):
        for m in re.finditer(pat, text):
            pass
        if m is not None:
            cand = m.group(1).strip()
            # prefer a trailing $...$ math segment if present
            mm = re.findall(r"\$([^$]+)\$", cand)
            return mm[-1].strip() if mm else cand
    # last inline math segment
    math = re.findall(r"\$([^$]+)\$", text)
    return math[-1].strip() if math else None


def answers_equal(pred: str | None, gold: str) -> bool:
    """Compare a predicted answer with the ground truth (string, then numeric)."""
    if pred is None:
        return False
    p = " ".join(pred.split())
    g = " ".join(gold.split())
    if p == g:
        return True
    try:
        return float(p) == float(g)
    except ValueError:
        return False


def emit(payload: dict) -> None:
    """Print the result as JSON on stdout (agent-readable contract)."""
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
