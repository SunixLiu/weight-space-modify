#!/usr/bin/env python
"""Offline self-check for the weightspace bridge (ws_core.py). Runs in seconds, no model load.

Checks:
  1. tracked edits: diff counts, checkpoint/restore, revert;
  2. sparse patch round-trip onto a fresh copy of the weights;
  3. the numpy op ports match the old torch implementation;
  4. `refresh()` makes diff() honest about the bf16 cast (a lost edit reports 0);
  5. the lazy safetensors store reads headers/shapes/describe without loading the model;
  6. manifest + patch source-hash verification (shard hashes, manifest round-trip,
     stamping, verify pass / changed-shard / missing-shard).

Usage:
  python selfcheck_ws.py [--model DIR]     # DIR defaults to the skill's model path
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from weightspace import WeightSpace

from ws_core import SafetensorsDirStore, apply_op, diff_dict, open_space_lazy, parse_columns, refresh, resolve_name

DEFAULT_MODEL = r"E:\LLMModels\Qwen\qwen3.5-0.8b"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def torch_ref(W, v, op, scale, index=None):
    W, v = W.float(), v.float()
    if op == "subtract":
        if v.dim() == 1:
            return W - scale * v
        return W - scale * v
    if op == "project-out":
        return W - scale * torch.outer(W @ v, v) / float(torch.dot(v, v))
    if op == "project-out-output":
        return W - scale * torch.outer(v, v @ W) / float(torch.dot(v, v))
    if op == "add-row":
        out = W.clone()
        out[index] = out[index] + scale * v
        return out
    raise ValueError(op)


def test_tracking() -> None:
    print("1. tracked edits / checkpoints / patches")
    rng = np.random.default_rng(0)
    ws = WeightSpace.from_numpy({"w": rng.standard_normal((8, 4)).astype(np.float32), "b": np.zeros(4, np.float32)})
    ws.edit("w")[0, 0] = 5.0
    d = ws.diff()
    check("one tensor changed", d.changed_params == 1 and d.tensors[0].name == "w")
    check("diff fraction is per-parameter", abs(d.fraction - 1 / 36) < 1e-9, f"got {d.fraction}")
    ws.checkpoint("after")
    ws.edit("b")[1] = 2.0
    check("second edit tracked", ws.diff().changed_params == 2)
    ws.restore("after")
    check("restore drops the second edit", ws.diff().changed_params == 1)
    ws.revert()
    check("revert restores originals", ws.diff().changed_params == 0)


def test_patch_roundtrip() -> None:
    print("2. sparse patch round-trip")
    rng = np.random.default_rng(1)
    base = {"w": rng.standard_normal((16, 8)).astype(np.float32)}
    # from_numpy aliases the array it is given, so hand it a copy.
    ws = WeightSpace.from_numpy({"w": base["w"].copy()})
    ws.edit("w")[2, 3] = 42.0
    ws.edit("w")[7, 1] = -1.0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "p.npz")
        ws.save_patch(path)
        target = WeightSpace.from_numpy({"w": base["w"].copy()})
        target.apply_patch(path)
        diff = target.diff()
        check("patch changes exactly the edited weights", diff.changed_params == 2, str(diff))
        check("patch reproduces values", float(target["w"][2, 3]) == 42.0)
        check("patch is sparse", os.path.getsize(path) < 4096, f"{os.path.getsize(path)} bytes")


def test_ops() -> None:
    print("3. numpy ops match the torch implementation")
    rng = np.random.default_rng(2)
    W = rng.standard_normal((16, 8)).astype(np.float32)
    cases = [
        ("project-out", rng.standard_normal(8).astype(np.float32), 0.7, None),
        ("project-out-output", rng.standard_normal(16).astype(np.float32), 0.5, None),
        ("subtract", rng.standard_normal(8).astype(np.float32), 0.3, None),
        ("subtract", rng.standard_normal((16, 8)).astype(np.float32), 0.9, None),
        ("add-row", rng.standard_normal(8).astype(np.float32), 0.4, 3),
    ]
    for op, v, scale, index in cases:
        ws = WeightSpace.from_numpy({"w": W.copy()})
        apply_op(ws, "w", op, v, scale, index)
        got = np.array(ws["w"], dtype=np.float32)
        want = torch_ref(torch.from_numpy(W), torch.from_numpy(v), op, scale, index).numpy()
        err = float(np.abs(got - want).max())
        check(f"{op} scale={scale} matches", err < 1e-5, f"max err {err:g}")
    ws = WeightSpace.from_numpy({"w": W.copy()})
    try:
        apply_op(ws, "w", "project-out", rng.standard_normal(16).astype(np.float32))
        check("project-out rejects a wrong-length vector", False)
    except SystemExit:
        check("project-out rejects a wrong-length vector", True)


def test_columns() -> None:
    print("3b. --columns selection (paper form: only selected MLP neurons)")
    rng = np.random.default_rng(3)
    W = rng.standard_normal((16, 32)).astype(np.float32)
    v = rng.standard_normal(16).astype(np.float32)
    cols = [0, 1, 2, 5, 9, 17, 31]

    ws = WeightSpace.from_numpy({"w": W.copy()})
    apply_op(ws, "w", "project-out-output", v, 0.5, None, cols)
    got = np.array(ws["w"], dtype=np.float32)

    scale, vv = 0.5, float(np.dot(v, v))
    sub = torch.from_numpy(W)[:, cols]
    want = torch.from_numpy(W).clone()
    want[:, cols] = sub - scale * torch.outer(torch.from_numpy(v), torch.from_numpy(v) @ sub) / vv
    err = float(np.abs(got - want.numpy()).max())
    check("selected columns match the torch form", err < 1e-5, f"max err {err:g}")

    other = np.setdiff1d(np.arange(W.shape[1]), cols)
    check("unselected columns are bit-identical", np.array_equal(got[:, other], W[:, other]))
    check("selected columns did change", not np.array_equal(got[:, cols], W[:, cols]))

    check("inline spec parses ranges and dedupes", parse_columns("1,3-5,5") == [1, 3, 4, 5], str(parse_columns("1,3-5,5")))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cols.npy")
        np.save(path, np.array([7, 2, 7, 11]))
        check("indices file parses and dedupes", parse_columns(path) == [7, 2, 11])
        jpath = os.path.join(td, "cols.json")
        with open(jpath, "w") as f:
            f.write("[4, 6]")
        check("json indices parse", parse_columns(jpath) == [4, 6])

    ws = WeightSpace.from_numpy({"w": W.copy()})
    try:
        apply_op(ws, "w", "project-out-output", v, 1.0, None, [32])
        check("out-of-range column is rejected", False)
    except SystemExit as e:
        check("out-of-range column is rejected", "out of range" in str(e))
    try:
        apply_op(ws, "w", "project-out", rng.standard_normal(32).astype(np.float32), 1.0, None, [1])
        check("--columns on another op is rejected", False)
    except SystemExit as e:
        check("--columns on another op is rejected", "only applies" in str(e))


def test_bf16_refresh() -> None:
    print("4. refresh() reports what survived the bf16 cast")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))

    model = M()
    ws = WeightSpace.from_torch(model)
    before = float(model.w[0, 0])
    ws.edit("w")[0, 0] = before + 1.0
    ws.sync()
    refresh(ws)
    check("a real edit survives", ws.diff().changed_params == 1, str(ws.diff()))
    check("model holds the new value", abs(float(model.w[0, 0]) - (before + 1.0)) < 0.05)

    ws.commit()  # the synced weights become the new baseline
    ws.edit("w")[1, 1] = float(ws._store.get("w")[1, 1]) + 1e-6  # fp32 keeps it, bf16 does not
    ws.sync()
    refresh(ws)
    check("a sub-quantization edit reports zero change", ws.diff().changed_params == 0, str(ws.diff()))


def test_manifest() -> None:
    print("6. manifest + patch source-hash verification")
    import hashlib

    from safetensors.torch import save_file

    from ws_core import patch_source_hashes, shard_hashes, stamp_patch_source, verify_patch_source, write_manifest

    a = torch.randn(4, 4).to(torch.bfloat16)
    b = torch.randn(3, 3).to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        os.makedirs(src)
        save_file({"w": a}, os.path.join(src, "model-00001.safetensors"))
        save_file({"v": b}, os.path.join(src, "model-00002.safetensors"))

        want = {}
        for fn in sorted(os.listdir(src)):
            h = hashlib.sha256()
            with open(os.path.join(src, fn), "rb") as f:
                h.update(f.read())
            want[fn] = h.hexdigest()
        check("shard_hashes matches an independent hashlib pass", shard_hashes(src) == want)

        out = os.path.join(td, "out")
        os.makedirs(out)
        recipe = {"op": "project-out-output", "tensor": "w", "scale": 0.5, "columns": [1, 2], "index": None, "vector_file": "v.pt"}
        mpath = write_manifest(out, src, recipe, 7, shards=want)
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        check(
            "manifest records source shards + recipe + changed_elements",
            m["source"]["shards"] == want and m["recipe"] == recipe and m["changed_elements"] == 7,
        )

        ws = WeightSpace.from_numpy({"w": a.float().numpy()})
        ws.edit("w")[0, 0] = 1.0
        ppath = os.path.join(td, "p.npz")
        ws.save_patch(ppath)
        check("un-stamped patch records no source hashes", patch_source_hashes(ppath) is None)
        stamp_patch_source(ppath, want)
        check("stamped patch records the source hashes", patch_source_hashes(ppath) == want)
        target = WeightSpace.from_numpy({"w": a.float().numpy()})
        target.apply_patch(ppath)
        check("stamped patch still applies", target.diff().changed_params == 1)

        v = verify_patch_source(ppath, src)
        check("verification passes on the right source", v["verified"] is True)

        other = os.path.join(td, "other")
        os.makedirs(other)
        save_file({"w": torch.randn(4, 4).to(torch.bfloat16)}, os.path.join(other, "model-00001.safetensors"))
        save_file({"v": b}, os.path.join(other, "model-00002.safetensors"))
        v2 = verify_patch_source(ppath, other)
        check("verification flags a changed shard", v2["verified"] is False and "model-00001.safetensors" in v2["mismatch"])

        partial = os.path.join(td, "partial")
        os.makedirs(partial)
        save_file({"w": a}, os.path.join(partial, "model-00001.safetensors"))
        v3 = verify_patch_source(ppath, partial)
        check("verification flags a missing shard", v3["verified"] is False and "model-00002.safetensors" in v3["mismatch"])


def test_lazy_store(model_dir: str) -> None:
    print("5. lazy safetensors store")
    if not os.path.isdir(model_dir):
        print(f"  [skip] model dir not found: {model_dir}")
        return
    ws = open_space_lazy(model_dir)
    names = ws.names()
    check("tensor count matches the index", len(names) == 488, f"got {len(names)}")
    name = "model.language_model.layers.15.mlp.down_proj.weight"
    check("shape available from the header", ws.shape(name) == (1024, 3584), str(ws.shape(name)))
    check("dtype available from the header", ws.dtype(name) == "bfloat16", ws.dtype(name))
    text = ws.describe(name)
    check("describe returns stats + spectrum", "singular values" in text and "row L2 norms" in text)
    check("describe is cached by content", text == ws.describe(name))
    try:
        resolve_name(ws, "mlp.down_proj")
        check("ambiguous name is rejected", False)
    except SystemExit as e:
        check("ambiguous name is rejected", "ambiguous" in str(e))
    check("unique name resolves", resolve_name(ws, "layers.15.mlp.down_proj.weight").endswith(name.split("model.")[-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    test_tracking()
    test_patch_roundtrip()
    test_ops()
    test_columns()
    test_bf16_refresh()
    test_manifest()
    if not args.skip_model:
        test_lazy_store(args.model)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
