#!/usr/bin/env python
"""Consult the local 27B "surgeon" model at a decision point of the loop.

The Python scripts are the scalpel and monitor; this call is the surgeon's
brain. It talks to the local llama.cpp OpenAI-compatible server and returns
plain text guidance that the agent then executes.

Usage:
  python consult_surgeon.py --phase diagnose [--prompt "..."]
  python consult_surgeon.py --phase probe   --context-file probe.json
  python consult_surgeon.py --phase strategy --context-file probe.json --context-file weights.json
  python consult_surgeon.py --phase evaluate --context-file eval_after.json --context-file eval_before.json
  [--base-url http://127.0.0.1:8001/v1] [--model Qwen3.8-27B]
  [--temperature 0.2] [--max-tokens 1500]

Phases and what the surgeon is asked to produce:
  diagnose: mechanism hypotheses + a concrete probing plan
  probe:    which layers / feature directions are loop candidates
  strategy: surgical plan (layer, op, direction, scale) in executable terms
  evaluate: verdict + next iteration or stop
"""
import argparse
import json
import sys
import urllib.request

SYSTEM = {
    "diagnose": (
        "你是主治外科医生：精通 PyTorch、Transformer 内部架构与神经网络机械可解释性。"
        "背景：我们的小模型（0.8B）在 MATH 推理时会陷入数学死循环（撞 token，不输出 EOS，"
        "重复同样的推导），目标是通过权重空间的微创手术修复，而不是重新训练。"
        "你根据评测现象分析死循环的机制假设（哪些层/哪些特征方向可能把模型锁在循环里），"
        "并给出下一步的排查计划：给 probe_activations 脚本什么参数（探哪些层、抓什么激活、"
        "关注什么指标）。输出简洁的行动计划，不要写代码全文。"
    ),
    "probe": (
        "你是主治外科医生。下面是 probe_activations 脚本抓到的中间层激活统计与异常标记。"
        "请判断：哪些层（尤其是 MLP / 注意力输出）、哪些 token 区间的激活异常最可能是死循环的"
        "根因候选；如何从这些激活中构造'循环特征方向'向量（例如死循环 token 的激活均值，"
        "或循环前后 token 激活的差向量）。给出候选列表和构造向量的具体做法。"
    ),
    "strategy": (
        "你是主治外科医生。基于探查数据，给出权重空间手术方案：选择哪一层、选择哪种操作"
        "（project-out 特征方向剥离，或 subtract 矩阵减法）、如何得到方向向量、缩放系数取多少"
        "（从小开始，如 0.1~1.0，避免误伤其他能力）。输出必须可执行：层名（tensor 子串）、"
        "op、向量来源、scale。同时说明这次手术预期改善什么、可能误伤什么。"
    ),
    "evaluate": (
        "你是主治外科医生。对比手术前后的评测结果（正确率、死循环率、截断率、各题型表现）。"
        "判断：改善是否真实、是否误伤其他能力（如指令遵循）、是否值得继续。给出结论：收工，"
        "或进入下一轮迭代（并给出下一轮的调整方向，如缩小改动层范围、减小 scale、换方向向量）。"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=list(SYSTEM))
    ap.add_argument("--prompt", default=None, help="Free-form question (in addition to context files)")
    ap.add_argument("--prompt-file", default=None, help="Read the question from a file")
    ap.add_argument("--context-file", action="append", default=[],
                    help="Artifact to attach as context (repeatable: eval JSON, probe JSON, ...)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4096, help="Reasoning models need headroom")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    parts = []
    if args.prompt_file:
        parts.append(open(args.prompt_file, "r", encoding="utf-8").read())
    if args.prompt:
        parts.append(args.prompt)
    for path in args.context_file:
        body = open(path, "r", encoding="utf-8").read()
        parts.append(f"--- {path} ---\n{body}")
    if not parts:
        ap.error("provide --prompt/--prompt-file and/or --context-file")

    messages = [
        {"role": "system", "content": SYSTEM[args.phase]},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    payload = json.dumps(
        {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text:
        # Reasoning models may spend all tokens on reasoning_content.
        text = (msg.get("reasoning_content") or "").strip()
        if text:
            print("[surgeon] content empty; returning reasoning_content", file=sys.stderr)
    if not text:
        print("[surgeon] ERROR: model returned no text (raise --max-tokens)", file=sys.stderr)
        sys.exit(2)
    usage = data.get("usage", {})
    print(f"[surgeon] phase={args.phase} tokens={usage.get('total_tokens', '?')}", file=sys.stderr)
    print(text)


if __name__ == "__main__":
    main()
