# weight-space-modify

用小模型权重空间的**微创手术**诊断并修复"数学死循环"——不重训练。

当小模型（如 Qwen3.5-0.8B）在 MATH/GSM8K 等数学基准上陷入无限重复——撞 token 上限、永不发 EOS——本技能提供完整闭环：探测 safetensors 权重 → 提取循环方向向量 → 施加最小补丁 → 重新评测验证。本地 27B 推理模型充当"主治外科医生"，在每个决策点介入；Python 脚本是手术刀和监视器。

> English version: [README.md](README.md)

## 上手指南（Getting started）

**1. 安装技能。** clone 本仓库，把 `weight-space-modify/` 目录拷进你的 agent 的 skills 目录（如 `~/.claude/skills/`，或 DSH profile 的 skills 目录）。它是自包含的：`SKILL.md`（面向 agent 的说明）+ `references/` + `scripts/`。当任务匹配时（小模型在数学上死循环、必须用权重编辑而非重训练来修），agent 会按名字/描述自动选用它。

**2. 准备 Python 环境。**

```bash
python -m venv .venv
# Windows: .venv\Scripts\pip install ...   Linux/macOS: .venv/bin/pip install ...
pip install torch transformers numpy safetensors "weightspace==0.3.0"
# 可选，GDN 层提速约 5–10×：
pip install flash-linear-attention causal-conv1d
```

**3. 准备输入。**

- 目标模型：Qwen3.5-0.8B 的 safetensors 目录（标准 HF 布局）
- 基准：MATH jsonl——已随技能打包在 `weight-space-modify/data/math-test.jsonl`（500 题），也可用 `--data` 换成自己的文件
- 外科医生：本地 27B 推理模型的 OpenAI 兼容端点（需支持 tool calling，如 llama.cpp server），默认 `http://127.0.0.1:8001/v1`

**4. 使用。**

- *agent 驱动（推荐）*：给 agent 一个任务，例如"Qwen3.5-0.8B 在 MATH 上死循环；用权重编辑降低循环率"。它会按 `SKILL.md` 的闭环执行，在每个决策点咨询 27B 外科医生。
- *手动*：在技能目录下运行脚本，用 flag 覆盖 `SKILL.md` 里的本机默认路径：

```bash
python scripts/run_eval_benchmark.py --model /path/to/qwen3.5-0.8b --data /path/to/math.jsonl --limit 20
```

所有脚本向 stdout 输出一个 JSON 文档（stderr 承载进度）。Windows 先设置 `PYTHONIOENCODING=utf-8`。

## 闭环流程

1. **基线** — `run_eval_benchmark.py` 跑 20 题 → `eval_before.json`（准确率 / 死循环率 / 截断率）
2. **诊断** — `consult_surgeon.py --phase diagnose`：27B 提出机制假设与探针计划
3. **探针** — `probe_activations.py` 在真正循环的 prompt 上逐 token 捕获 residual/MLP/attention 激活（`--save-activations`）
4. **分析** — `consult_surgeon.py --phase probe`：27B 点名候选层/方向
5. **方案** — `consult_surgeon.py --phase strategy`：手术方案（张量、op、方向来源、scale；从小开始 0.1–0.5）
6. **打补丁** — `extract_direction.py`（可选 `select_columns.py` 选中间神经元列）→ `apply_patch_and_save.py` 保存完整可加载模型 + 稀疏补丁（`.npz`，几 KB）
7. **复评** — `run_eval_benchmark.py --model patched_x` → `eval_after.json`
8. **复盘** — `consult_surgeon.py --phase evaluate`：保留或回滚，迭代

Agent 模式：`ws_agent.py` 让 27B 通过工具直接读写权重（forward-pass 预算制），替代第 5–6 步。

## 目录结构

| 路径 | 说明 |
|---|---|
| `SKILL.md` | 技能说明与完整工具清单 |
| `references/architecture.md` | Qwen3.5-0.8B 张量布局、方向空间 → op 映射 |
| `data/math-test.jsonl` | 随包 MATH 基准数据集（500 题） |
| `scripts/run_eval_benchmark.py` | MATH 评测：准确率 / 循环率 / 截断率 |
| `scripts/inspect_weights.py` | 层地图与张量统计（只读 safetensors 头） |
| `scripts/probe_activations.py` | 逐 token 激活捕获（residual/mlp/attn/mlp_mid） |
| `scripts/extract_direction.py` | 从探针捕获构建单位"循环方向"向量 |
| `scripts/select_columns.py` | 按 loop-vs-control 对比给中间神经元排名、选列 |
| `scripts/probe_logits.py` | logit 级探针：逐步 top-k / EOS 概率 |
| `scripts/phase_probe.py` | 离线相位对齐分析探针 `.pt`（无需 GPU） |
| `scripts/apply_patch_and_save.py` | 施加一次最小补丁，保存完整模型 + 稀疏补丁 |
| `scripts/patch_apply.py` | 把稀疏补丁重放到同架构 checkpoint |
| `scripts/scan_layers.py` | 逐层因果扫描：按编辑后行为给层排名 |
| `scripts/ws_agent.py` | agentic 权重编辑（27B 直接改权重） |
| `scripts/consult_surgeon.py` | 27B 外科医生咨询（四阶段 system prompt） |
| `scripts/selfcheck_ws.py` | weightspace 桥离线自检 |
| `scripts/ws_core.py` | weightspace 桥：追踪 / 回滚 / bf16 记账 |
| `scripts/_common.py` | 模型加载 / chat 模板 / 死循环检测 / 答案提取 |

## 关键原则

- 永不写入源模型目录；每个补丁都存到新的 `--out` 目录
- 一次补丁、一次复评；测量前绝不叠加两个补丁
- 权重是 bf16（约 8 位尾数）：信 JSON 里的 `changed_params`，不信名义 delta；为 0 即"no-op"
- 优先稀疏补丁（`--patch-out` + `patch_apply.py`），而不是保留完整模型副本
- 循环率是主指标；准确率不得回退
- scale 从小开始（0.1–0.5）；"修了部分、坏了部分"说明方向真实但幅度太大——缩小扰动，而不是当作噪声

## 已验证结果（Qwen3.5-0.8B，MATH 20 题）

基线：acc 10% / loop 50% / trunc 70%。多轮单刀闭环后，最佳单编辑为 r7（L20 GDN `A_log` head 2，scale 0.2，恰好 1 个标量变化）：**loop 50% → 35%**（净 −3：修 4 新开 1），acc 保持 10%。诚实边界：全程 acc 未提升——本技能验证的是"循环率可被最小权重编辑降低"，而非参考项目的准确率复现。
