# Agent Ranking

面向**本地部署、无网络环境**的大语言模型量化评测框架。通过 **OpenAI 兼容 API** 接入 vLLM、TGI、SGLang、Ollama 等推理服务，对 30B–100B 级模型进行推理、代码、逻辑、准确率、速度、多轮对话和 Agent 能力的系统化评测与排名。

---

## 功能特性

- **OpenAI Compatible API**：统一适配层，配置即可切换模型 endpoint
- **分级运行策略**：按 `small / medium / large / xlarge` 自动调整并发与超时
- **六大评测套件**：`reasoning` `logic` `accuracy` `code` `dialogue` `agent`
- **可扩展题库**：JSONL 格式，按 profile 组织（`smoke` / `standard` / `benchmark-light` / `benchmark-full`）
- **可扩展判分器**：规则判分、代码沙箱、本地 LLM Judge
- **速度基准**：流式 TTFT、tokens/s
- **批量排名**：多模型一次评测，自动生成 `ranking.json`
- **能力探测**：启动前自动检测 chat / stream / tools / json_mode

---

## 目录结构

```
agent_ranking/
├── cli.py                      # 命令行入口
├── configs/
│   ├── models.yaml             # 模型 API 配置
│   ├── tiers.yaml              # 30B-100B 分级策略
│   ├── judge.yaml              # Judge 模型（可选）
│   ├── profiles.yaml           # 评测 profile
│   └── weights.yaml            # 综合分权重
├── datasets/
│   └── smoke/                  # 冒烟题库（约 20 题）
│       ├── reasoning.jsonl
│       ├── logic.jsonl
│       ├── accuracy.jsonl
│       ├── code.jsonl
│       ├── dialogue.jsonl
│       └── agent.jsonl
├── agent_ranking/
│   ├── adapters/               # API 适配层
│   ├── core/                   # 配置、类型、执行器
│   ├── judges/                 # 判分器
│   ├── runners/                # 评测执行
│   ├── suites/                 # 可扩展评测套件
│   ├── tools/                  # Agent mock 工具
│   └── reports/                # 报告生成
└── reports/                    # 默认输出目录（运行后生成）
```

---

## 快速开始

### 1. 安装依赖

```bash
cd agent_ranking
pip install -r requirements.txt

# 或以可编辑模式安装（推荐）
pip install -e .
```

### 2. 启动模型 API 服务

以 vLLM 为例（OpenAI 兼容）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your-model \
  --served-model-name qwen3-72b-instruct \
  --port 8000
```

其他常见地址：

| 部署方式 | base_url |
|----------|----------|
| vLLM | `http://localhost:8000/v1` |
| TGI | `http://localhost:8080/v1` |
| SGLang | `http://localhost:30000/v1` |
| Ollama | `http://localhost:11434/v1` |

### 3. 配置模型

编辑 `configs/models.yaml`：

```yaml
models:
  qwen3-72b:
    adapter: openai_compat
    base_url: "http://localhost:8000/v1"
    api_key: "EMPTY"
    model_name: "qwen3-72b-instruct"   # 与 --served-model-name 一致
    tier: large                         # medium / large / xlarge
    supports_tools: true
    max_context: 32768
```

**tier 参考：**

| 参数量 | 建议 tier | 并发 |
|--------|-----------|------|
| 30B–40B | `medium` | 4 |
| 50B–70B | `large` | 2 |
| 80B–100B+ | `xlarge` | 1（串行） |

### 4. 探测模型能力

```bash
python cli.py probe --model qwen3-72b
```

输出包含：`chat`、`streaming`、`tools`、`json_mode` 是否可用，以及会跳过哪些套件。

### 5. 运行冒烟评测

```bash
python cli.py run --model qwen3-72b --profile smoke
```

报告输出到 `reports/qwen3-72b/`：
- `report.json` — 完整结构化数据
- `report.html` — 可视化报告

### 6. 批量对比排名

```bash
python cli.py run-multi --models qwen3-72b,deepseek-33b --profile smoke
```

生成 `reports/ranking.json`。

### 7. 单独测速

```bash
python cli.py benchmark-speed --model qwen3-72b --runs 5
```

---

## CLI 命令参考

| 命令 | 说明 |
|------|------|
| `list-models` | 列出已配置模型 |
| `list-suites` | 列出评测套件 |
| `probe -m <model>` | 探测 API 能力 |
| `run -m <model> -p <profile>` | 运行评测 |
| `run-multi -m a,b,c` | 多模型批量评测 |
| `benchmark-speed -m <model>` | 速度基准 |
| `run-suite -s <suite> -p <jsonl> -m <model>` | 自定义题库单套件评测 |

**常用参数：**

```bash
python cli.py run \
  --model qwen3-72b \
  --profile smoke \
  --output reports \
  --no-speed              # 跳过速度测试
  --dataset-profile smoke # 指定题库目录
```

安装后也可使用入口命令：

```bash
agent-rank run --model qwen3-72b --profile smoke
```

---

## 评测套件说明

### reasoning — 推理

多步数学、常识链式推理。判分方式：规则（数值、包含、精确匹配）。

```json
{
  "id": "reasoning_001",
  "prompt": "小明有 5 个苹果，又买了 3 个，吃掉了 2 个。他现在有几个苹果？",
  "checks": [{"type": "numeric", "value": 6}]
}
```

### logic — 逻辑

演绎推理、真假判断、约束题。

```json
{
  "id": "logic_001",
  "prompt": "前提：如果下雨，地面会湿。地面是湿的。结论：一定下雨了。是否正确？",
  "checks": [{"type": "contains", "value": "不正确"}]
}
```

### accuracy — 准确率

知识问答、选择题、指令遵循。

```json
{
  "id": "accuracy_004",
  "prompt": "以下哪个是质数？A. 9  B. 15  C. 17  D. 21。只回答选项字母。",
  "checks": [{"type": "choice", "value": "C"}]
}
```

开放题可启用 LLM Judge（见下文）。

### code — 代码

生成 Python 函数，沙箱执行单元测试。

```json
{
  "id": "code_001",
  "prompt": "编写 Python 函数 solve(n)，判断 n 是否为偶数。",
  "entry": "solve",
  "tests": [
    {"input": {"n": 4}, "expected": true},
    {"input": {"n": 7}, "expected": false}
  ]
}
```

### dialogue — 多轮对话

多轮剧本，检测记忆、纠错、连贯性。

```json
{
  "id": "dialogue_002",
  "system": "用户纠正你时要接受纠正。",
  "turns": [
    {"user": "我最喜欢的颜色是蓝色。"},
    {"user": "我最喜欢的颜色是什么？", "check": {"type": "contains", "value": "蓝"}},
    {"user": "不对，是红色。"},
    {"user": "我最喜欢的颜色是什么？", "check": {"type": "contains", "value": "红"}}
  ]
}
```

### agent — Agent 工具调用

本地 mock 工具（calculator、search_docs、read_file、list_files），完全离线。

```json
{
  "id": "agent_001",
  "task": "请计算 123 × 456，使用 calculator 工具。",
  "expected_tools": ["calculator"],
  "expected_answer": "56088",
  "checks": [
    {"type": "tool_called", "tool": "calculator"},
    {"type": "answer_contains", "value": "56088"}
  ]
}
```

Agent 题可附带本地虚拟环境：

```json
{
  "docs": {"deploy_guide": "推荐使用 vLLM 部署..."},
  "files": {"/data/config.txt": "server_port=8080"}
}
```

---

## 内置题库一览

| Profile | 题量 | 说明 | 数据来源 |
|---------|------|------|----------|
| `smoke` | ~20 | 冒烟测试，最快 | 自研 |
| `standard` | ~48 | 中等难度自研题 | 自研 |
| `benchmark-light` | ~128 | 通用基准轻量版 | GSM8K、MMLU、C-Eval、HumanEval、MBPP + 自研 |
| `benchmark-full` | ~974 | 通用基准完整版 | GSM8K×500、HumanEval 全量 164、MBPP×80 等 |

```bash
# 通用基准（轻量，约 1-3 小时，视模型而定）
python cli.py run --model openrouter --profile benchmark-light

# 通用基准（完整，建议过夜跑）
python cli.py run --model openrouter --profile benchmark-full
```

各 profile 元数据见 `datasets/<profile>/meta.json`。

### 重建通用基准题库

项目已内置生成好的 `datasets/benchmark-*`。若需重新生成：

```bash
python scripts/fetch_benchmark_cache.py   # 下载 GSM8K/HumanEval/MBPP 到 scripts/cache/
python scripts/build_benchmarks.py        # 生成 benchmark-light / benchmark-full
```

## 扩展题库

### 添加新题目

1. 在 `datasets/<profile>/` 下编辑或新建 JSONL 文件
2. 每行一题，必须包含 `id` 字段
3. 文件名 = 套件名（如 `reasoning.jsonl`）

```bash
# 创建扩展题库
mkdir -p datasets/standard
cp datasets/smoke/*.jsonl datasets/standard/
# 追加更多题目到 standard 目录
```

### 使用自定义题库

```bash
python cli.py run --model qwen3-72b --profile standard --dataset-profile standard
```

或单独跑某个套件：

```bash
python cli.py run-suite \
  --suite reasoning \
  --path /path/to/my_reasoning.jsonl \
  --model qwen3-72b
```

---

## 扩展评测方式

### 1. 新增判分 check 类型

编辑 `agent_ranking/judges/rule_judge.py`，在 `_run_check` 中添加新 `type`。

### 2. 启用 LLM Judge

编辑 `configs/judge.yaml`：

```yaml
judge:
  enabled: true
  base_url: "http://localhost:8001/v1"
  model_name: "qwen2.5-7b-instruct"
  tier: small
```

题目中指定 `"judge": "llm"` 并附带 `rubric` / `reference`：

```json
{
  "id": "accuracy_open_001",
  "prompt": "解释什么是 RAG？",
  "judge": "llm",
  "rubric": "回答应包含检索、增强、生成三个概念",
  "reference": "RAG 是 Retrieval-Augmented Generation...",
  "pass_threshold": 0.6
}
```

### 3. 注册自定义评测套件

```python
# my_custom_suite.py
from agent_ranking.suites.base import BenchmarkSuite
from agent_ranking.suites.registry import register_suite

class MySuite(BenchmarkSuite):
    name = "my_suite"

    def run_item(self, client, item, **kwargs):
        # 自定义调用与判分逻辑
        ...

register_suite("my_suite", MySuite)
```

在 `configs/profiles.yaml` 中加入 `my_suite`，并创建 `datasets/smoke/my_suite.jsonl`。

### 4. 调整综合分权重

编辑 `configs/weights.yaml`：

```yaml
weights:
  reasoning: 0.20
  code: 0.25
  agent: 0.20
  speed: 0.10
  # ...
```

---

## 支持的 check 类型

| type | 说明 | 示例 |
|------|------|------|
| `exact` | 精确匹配 | `"value": "北京"` |
| `contains` | 包含子串 | `"value": "56088"` |
| `not_contains` | 不包含 | `"value": "不知道"` |
| `regex` | 正则匹配 | `"pattern": "答案[是为]\\s*C"` |
| `any_of` | 包含任一 | `"values": ["北京", "Beijing"]` |
| `numeric` | 数值匹配 | `"value": 6, "tolerance": 0.01` |
| `choice` | 选择题 | `"value": "C"` |
| `session_contains` | 对话历史包含 | 用于多轮记忆 |

Agent 专用 check：

| type | 说明 |
|------|------|
| `tool_called` | 是否调用了指定工具 |
| `answer_contains` | 最终回答包含 |
| `max_steps` | 步数不超过限制 |

---

## 配置参考

### models.yaml

```yaml
models:
  my-model:
    adapter: openai_compat          # 目前仅支持此适配器
    base_url: "http://host:port/v1"
    api_key: "EMPTY"                # 或 ${ENV_VAR}
    model_name: "served-model-name"
    tier: large
    supports_tools: true
    max_context: 32768
    extra_headers: {}               # 可选自定义头
```

### tiers.yaml

控制并发、超时、默认生成长度：

```yaml
tiers:
  xlarge:
    max_concurrency: 1
    request_timeout_sec: 600
    default_max_tokens: 4096
    retry_count: 3
```

### profiles.yaml

```yaml
profiles:
  smoke:
    description: "冒烟测试"
    suites: [reasoning, logic, accuracy, code, dialogue, agent]
```

---

## 报告解读

`report.json` 核心字段：

```json
{
  "model": "qwen3-72b",
  "composite_score": 71.2,
  "suites": [
    {
      "suite": "reasoning",
      "total": 4,
      "passed": 3,
      "avg_score": 0.75,
      "avg_latency_ms": 2340
    }
  ],
  "speed": {
    "ttft_ms": 340,
    "tokens_per_sec": 28.5
  }
}
```

- **avg_score**：0–1，该套件平均得分
- **composite_score**：加权综合分（0–100）
- **passed/total**：达到 `pass_threshold` 的题数

---

## 多模型部署示例

```yaml
# configs/models.yaml
models:
  qwen3-32b:
    base_url: "http://10.0.1.10:8000/v1"
    model_name: "qwen3-32b-instruct"
    tier: medium

  qwen3-72b:
    base_url: "http://10.0.1.11:8000/v1"
    model_name: "qwen3-72b-instruct"
    tier: large

  judge:
    base_url: "http://10.0.1.20:8001/v1"
    model_name: "qwen2.5-7b-instruct"
    tier: small
```

```bash
python cli.py run-multi --models qwen3-32b,qwen3-72b --profile smoke
```

---

## 常见问题

### Q: 连接失败 / health check failed

1. 确认 API 服务已启动：`curl http://localhost:8000/v1/models`
2. 确认 `model_name` 与服务的 `--served-model-name` 一致
3. 检查防火墙与 `base_url` 地址

### Q: Agent 套件被跳过

模型不支持 tool calling。可：
- 换用支持 function calling 的模型/instruct 版本
- 或将 `supports_tools: false` 并在报告中接受跳过

### Q: 代码题超时

默认单题执行超时 10 秒。检查模型是否生成了有效 Python 代码块（\`\`\`python ... \`\`\`）。

### Q: 100B 模型评测太慢

- 使用 `--profile smoke` 日常筛查
- 在 `tiers.yaml` 设 `xlarge.max_concurrency: 1`
- 使用 `--no-speed` 跳过速度测试

### Q: 如何完全离线运行

1. 预下载所有模型权重到本地
2. 本地启动 vLLM 等服务
3. 题库、mock 工具、代码沙箱均不依赖网络
4. 仅 LLM Judge 需要额外部署一个小模型（可选）

---

## 开发说明

### 运行测试（无需真实模型）

```bash
python -c "from agent_ranking.suites.registry import list_suites; print(list_suites())"
python cli.py list-models
python cli.py list-suites
```

### 代码风格

- Python 3.10+
- 套件通过 `BenchmarkSuite` 基类扩展
- 判分器通过 `RuleJudge` / `CodeJudge` / `LLMJudge` 扩展
- 新 adapter 实现 `ModelClient` 协议后注册到 `create_client`

---

## 路线图

- [ ] 异步并发 batch runner（提升吞吐）
- [x] 内置通用基准 benchmark-light / benchmark-full
- [ ] 扩大 MMLU/C-Eval 官方全量导入
- [ ] 长上下文 needle-in-haystack 套件
- [ ] 评测结果历史对比与趋势图
- [ ] Gradio 可视化面板

---

## License

MIT
