"""Bridge between this skill and the `weightspace` package.

Why a bridge instead of calling `WeightSpace.from_hf` directly: the package
loads models with `AutoModelForCausalLM.from_pretrained`
(`weightspace.adapters.load_hf`), which does not resolve this hybrid
Qwen3.5 architecture. We already have a loader that does (`_common.load_model`),
so we hand the loaded module to `WeightSpace.from_torch`.

What the package buys us, on top of `_common`:

* every edit is snapshotted, so `diff()` reports exactly which weights moved
  (count, L2 delta, % of tensor norm) instead of a single delta_norm;
* `checkpoint` / `restore` / `revert` make trials reversible in-process;
* sparse patches (`save_patch` / `apply_patch`) replace full model copies;
* `describe()` gives quantiles, row/col norms and a spectrum with no bespoke code;
* `Session` / `Probe` drives the editing LLM with a hard forward-pass budget.

Two caveats handled here:

1. `TorchStore` mirrors bf16 (and CUDA) weights as float32; edits only reach the
   model on `sync()`. Any code that runs the model must sync first — `sync()` and
   the probes in `ws_agent.py` do.
2. After `sync()` the float32 mirror still holds the pre-quantization values, so
   `diff()` would overstate what survived the bf16 cast. `refresh()` re-reads the
   touched tensors from the model, so `diff()` is honest about the edit that
   actually runs. This matters: bf16 has ~8 mantissa bits, so a perturbation
   below ~0.4% of the weight magnitude silently rounds away.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from weightspace import WeightSpace
from weightspace.space import Store

# safetensors reports dtypes as short strings ("F32", "BF16", ...); normalize them
# to the torch names the rest of the skill prints.
_DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def _dtype_name(raw) -> str:
    text = str(raw).replace("torch.", "")
    return _DTYPE_NAMES.get(text.upper(), text)


class SafetensorsDirStore(Store):
    """Lazy read-only store over every ``*.safetensors`` file in a model directory.

    Heads are indexed from the file headers (`get_slice`), and a tensor is
    materialised as float32 only when it is read, so `describe` / `read` on a
    0.8B model costs one tensor of RAM instead of a full fp32 copy of the model.
    `flush` is a no-op: this store is for inspection, not for saving.
    """

    def __init__(self, model_dir: str, cache_bytes: int = 256 << 20):
        from safetensors import safe_open

        self.model_dir = model_dir
        self._handles: dict[str, object] = {}
        self._meta: dict[str, tuple[str, tuple[int, ...], str]] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._cache_bytes = cache_bytes
        self._cached_bytes = 0

        paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not paths:
            raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
        for path in paths:
            handle = safe_open(path, framework="pt")
            self._handles[path] = handle
            for key in handle.keys():
                sl = handle.get_slice(key)
                self._meta[key] = (path, tuple(sl.get_shape()), _dtype_name(sl.get_dtype()))
        if not self._meta:
            raise ValueError(f"{model_dir} contains no tensors")

    # Store interface ---------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self._meta)

    def shape(self, name: str) -> tuple[int, ...]:
        return self._meta[name][1]

    def dtype(self, name: str) -> str:
        return self._meta[name][2]

    def get(self, name: str) -> np.ndarray:
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        import torch

        path = self._meta[name][0]
        tensor = self._handles[path].get_tensor(name)
        arr = tensor.to(torch.float32).numpy()
        if arr.nbytes <= self._cache_bytes:
            self._cache[name] = arr
            self._cached_bytes += arr.nbytes
            while self._cached_bytes > self._cache_bytes and len(self._cache) > 1:
                old, evicted = next(iter(self._cache.items()))
                self._cache.pop(old)
                self._cached_bytes -= evicted.nbytes
        return arr

    def flush(self, names: list[str]) -> None:
        """Read-only store: nothing to write back (see `save_model` for saving)."""

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()
        self._cache.clear()
        self._cached_bytes = 0


@dataclass
class Space:
    """A loaded model plus its tracked weight space."""

    ws: WeightSpace
    model: object | None = None
    tokenizer: object | None = None
    path: str = ""
    device: object | None = None


def open_space(model_path: str, device: str = "auto") -> Space:
    """Load the model the skill's way, then wrap it in a `WeightSpace`."""
    from _common import load_model, load_tokenizer, pick_device

    dev = pick_device(device)
    model = load_model(model_path, dev)
    tokenizer = load_tokenizer(model_path)
    ws = WeightSpace.from_torch(model)
    return Space(ws=ws, model=model, tokenizer=tokenizer, path=model_path, device=dev)


def open_space_lazy(model_path: str) -> WeightSpace:
    """Read-only `WeightSpace` over a model directory, without loading the model."""
    return WeightSpace(SafetensorsDirStore(model_path))


def resolve_name(ws: WeightSpace, needle: str) -> str:
    """Unique tensor name containing `needle` (the old `find_tensor` contract)."""
    matches = ws.names(needle)
    # keep duplicates out while preserving order
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise SystemExit(f"no tensor matches '{needle}'. Try inspect_weights.py --layer to list candidates.")
    if len(matches) > 1:
        raise SystemExit(f"'{needle}' is ambiguous ({len(matches)} matches): " + "; ".join(matches[:8]))
    return matches[0]


def refresh(ws: WeightSpace) -> None:
    """Re-read touched tensors from the model so `diff()` sees post-quantization values.

    TorchStore keeps a float32 mirror for bf16 / CUDA tensors and writes it back on
    `sync()`; without this the mirror still holds values that bf16 rounding may have
    discarded, and `diff()` would claim edits that never reached the model.
    Tensors held as a shared CPU array need no refresh: the mirror *is* the parameter.
    """
    store = ws._store  # noqa: SLF001 - the adapter exposes no public hook for this
    tensors = getattr(store, "_t", None)
    arrays = getattr(store, "_arr", None)
    mirrored = getattr(store, "_mirrored", None)
    torch = getattr(store, "_torch", None)
    if not tensors or arrays is None or mirrored is None or torch is None:
        return
    for name in ws.touched():
        if name not in mirrored:
            continue
        param = tensors[name]
        arrays[name] = param.detach().to("cpu", torch.float32).numpy().copy()


def sync(space: Space) -> None:
    """Push edits into the model, then make the mirror agree with the bf16 weights."""
    space.ws.sync()
    refresh(space.ws)


def parse_columns(spec: str) -> list[int]:
    """Parse a column selection for `project-out-output --columns`.

    Accepted forms:
      * inline list with inclusive ranges: ``"12,40-63"``
      * a ``.npy`` / ``.pt`` / ``.json`` file holding indices (e.g. the top-N
        intermediate neurons picked from a probe activation)

    Duplicates are dropped, order preserved. Bounds are checked against the
    tensor in `apply_op`, because only the target knows W.shape[1].
    """
    if os.path.isfile(spec):
        if spec.endswith(".npy"):
            raw = np.load(spec)
        elif spec.endswith((".pt", ".pth")):
            import torch

            obj = torch.load(spec, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                obj = obj.tolist()
            elif isinstance(obj, dict):
                raise SystemExit(f"--columns file {spec} holds a dict; expected a list of indices")
            raw = obj
        elif spec.endswith(".json"):
            import json

            with open(spec, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raise SystemExit(f"unsupported --columns file type: {spec} (use .npy/.pt/.json)")
        cols = np.asarray(raw).reshape(-1).tolist()
    else:
        cols = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                cols.extend(range(int(lo), int(hi) + 1))
            else:
                cols.append(int(part))

    cols = [int(c) for c in cols]
    if any(c < 0 for c in cols):
        raise SystemExit("--columns indices must be >= 0")
    return list(dict.fromkeys(cols))


def apply_op(
    ws: WeightSpace,
    name: str,
    op: str,
    vector,
    scale: float = 1.0,
    index: int | None = None,
    columns: list[int] | None = None,
) -> dict:
    """Apply one of the skill's ops to a tensor through the tracked weight space.

    numpy port of the torch code that used to live in `apply_patch_and_save.py`;
    semantics are unchanged (W is [out, in]):

      subtract           W' = W - scale * V   (V: 2-D same shape, or 1-D of length in;
                                              for a 1-D tensor W, V is element-wise)
      project-out        W' = W - scale * (W v) outer v / ||v||^2      len(v) == in
      project-out-output W' = W - scale * v outer (v^T W) / ||v||^2    len(v) == out
      add-row            W'[index] = W[index] + scale * v             len(v) == in

    `columns` (project-out-output only) restricts the edit to the selected
    intermediate-neuron columns J: W'[:, J] = W[:, J] - scale * v (v^T W[:, J]) / ||v||^2,
    leaving every other column bit-identical. This is the paper's edit form
    ("64 selected MLP neurons"), as opposed to the full-width sweep you get
    without it.
    """
    if columns is not None and op != "project-out-output":
        raise SystemExit("--columns only applies to --op project-out-output")
    arr = ws.edit(name)  # live float32 mirror, edit tracked from here on
    W = np.array(arr, dtype=np.float32)  # working copy
    v = np.asarray(vector, dtype=np.float32)

    if op == "subtract":
        if W.ndim == 1:
            # 1-D tensor (e.g. per-head A_log / dt_bias): element-wise subtract
            if v.ndim != 1 or v.size != W.size:
                raise SystemExit(f"1-D tensor needs a 1-D vector of length {W.size} (got {v.shape})")
            W_new = W - scale * v
        elif v.ndim == 1:
            if v.size != W.shape[1]:
                raise SystemExit(f"1-D vector must have length W.shape[1]={W.shape[1]} (got {v.size})")
            W_new = W - scale * v
        elif v.shape != W.shape:
            raise SystemExit(f"shape mismatch: W {tuple(W.shape)} vs V {tuple(v.shape)}")
        else:
            W_new = W - scale * v
    elif op == "project-out":
        if v.ndim != 1 or v.size != W.shape[1]:
            raise SystemExit(
                f"project-out needs a 1-D vector of length W.shape[1]={W.shape[1]}, got shape {tuple(v.shape)}"
            )
        W_new = W - scale * np.outer(W @ v, v) / float(np.dot(v, v))
    elif op == "project-out-output":
        if v.ndim != 1 or v.size != W.shape[0]:
            raise SystemExit(
                f"project-out-output needs a 1-D vector of length W.shape[0]={W.shape[0]}, got shape {tuple(v.shape)}"
            )
        vv = float(np.dot(v, v))
        if columns is None:
            W_new = W - scale * np.outer(v, v @ W) / vv
        else:
            if not columns:
                raise SystemExit("--columns selected no columns")
            cols = np.asarray(columns, dtype=np.int64)
            if cols.min() < 0 or cols.max() >= W.shape[1]:
                raise SystemExit(f"--columns out of range: W has {W.shape[1]} columns, got [{cols.min()}, {cols.max()}]")
            W_new = W.copy()
            sub = W[:, cols]
            W_new[:, cols] = sub - scale * np.outer(v, v @ sub) / vv
    elif op == "add-row":
        if index is None:
            raise SystemExit("add-row needs --index")
        if v.ndim != 1 or v.size != W.shape[1]:
            raise SystemExit(f"add-row vector must have length W.shape[1]={W.shape[1]}, got shape {tuple(v.shape)}")
        W_new = W.copy()
        W_new[index] = W_new[index] + scale * v
    else:
        raise SystemExit(f"unknown op {op!r}")

    arr[...] = W_new.astype(arr.dtype)
    delta = W_new - W
    return {
        "tensor": name,
        "op": op,
        "shape": list(W.shape),
        "dtype": str(arr.dtype),
        "scale": float(scale),
        "columns": len(columns) if columns is not None else None,
        "delta_norm": round(float(np.linalg.norm(delta)), 6),
        "delta_max_abs": round(float(np.abs(delta).max()) if delta.size else 0.0, 6),
        "rel_delta": round(float(np.linalg.norm(delta) / (np.linalg.norm(W) + 1e-30)), 6),
    }


def diff_dict(ws: WeightSpace) -> dict:
    """JSON-friendly version of `ws.diff()`."""
    d = ws.diff()
    return {
        "total_params": d.total_params,
        "changed_params": d.changed_params,
        "fraction": round(d.fraction, 8),
        "tensors": [
            {
                "name": t.name,
                "numel": t.numel,
                "changed": t.changed,
                "l2_delta": round(t.l2_delta, 6),
                "max_abs_delta": round(t.max_abs_delta, 6),
                "rel_delta": round(t.rel_delta, 6),
            }
            for t in d.tensors
        ],
    }


def save_model(space: Space, out_dir: str) -> str:
    """Sync edits into the module, then write a complete loadable HF directory."""
    if space.model is None:
        raise SystemExit("this space has no model module; open it with open_space() to save")
    sync(space)
    os.makedirs(out_dir, exist_ok=True)
    space.model.save_pretrained(out_dir)
    if space.tokenizer is not None:
        space.tokenizer.save_pretrained(out_dir)
    return os.path.abspath(out_dir)


# Manifest / replay verification -----------------------------------------------------

MANIFEST_NAME = ".manifest.json"
PATCH_SOURCE_KEY = "__source_sha256__"


def shard_hashes(model_dir: str) -> dict[str, str]:
    """SHA-256 of every ``*.safetensors`` file in the directory, keyed by file name."""
    paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
    out: dict[str, str] = {}
    for path in paths:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.basename(path)] = h.hexdigest()
    return out


def write_manifest(
    out_dir: str,
    source_dir: str,
    recipe: dict,
    changed_elements: int,
    patch_path: str | None = None,
    shards: dict[str, str] | None = None,
) -> str:
    """Persist a self-describing manifest next to the patched model.

    Records what the patched directory is: which source shards it was cut from
    (SHA-256), the exact edit recipe, and how many elements actually changed after
    the bf16 cast — the same bookkeeping the reference project keeps in its
    ``.manifest.json``. Costs one extra read pass over the shards (~1.7 GB here).
    """
    manifest = {
        "format": "weight-space-modify/manifest@1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"path": os.path.abspath(source_dir), "shards": shards or shard_hashes(source_dir)},
        "recipe": recipe,
        "changed_elements": int(changed_elements),
        "patch": os.path.abspath(patch_path) if patch_path else None,
    }
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path


def read_manifest(model_dir: str) -> dict | None:
    """Load the manifest next to a patched model directory (None if absent)."""
    path = os.path.join(model_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stamp_patch_source(patch_path: str, shards: dict[str, str]) -> None:
    """Embed the source shard hashes into a sparse patch so replay can verify its target.

    `WeightSpace.apply_patch` skips keys that do not end in ``::idx``, so an extra
    metadata key is safe.
    """
    with np.load(patch_path) as f:
        data = {k: f[k] for k in f.files}
    data[PATCH_SOURCE_KEY] = np.asarray(json.dumps(shards))
    np.savez_compressed(patch_path, **data)


def patch_source_hashes(patch_path: str) -> dict[str, str] | None:
    """Source-shard hashes recorded inside a sparse patch (None for older patches)."""
    with np.load(patch_path) as f:
        if PATCH_SOURCE_KEY not in f.files:
            return None
        return json.loads(f[PATCH_SOURCE_KEY].item())


def verify_patch_source(patch_path: str, model_dir: str) -> dict:
    """Compare the patch's recorded source hashes against the target model's shards.

    Returns ``{"recorded", "target", "verified", "mismatch"}``; ``verified`` is None
    when the patch carries no recorded hashes (older patch), True/False otherwise.
    A mismatch means the target is not the checkpoint the patch was cut from —
    replaying it would silently edit the wrong weights.
    """
    recorded = patch_source_hashes(patch_path)
    target = shard_hashes(model_dir)
    if recorded is None:
        return {"recorded": None, "target": target, "verified": None, "mismatch": []}
    mismatch = sorted(name for name in set(recorded) | set(target) if recorded.get(name) != target.get(name))
    return {"recorded": recorded, "target": target, "verified": not mismatch, "mismatch": mismatch}
