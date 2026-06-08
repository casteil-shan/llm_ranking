# 自定义 Agent 接入完整指南

本文档说明如何将**你自己的 Agent** 接入 `agent_ranking` 评测框架，涵盖三种典型架构、配置方法、题库编写、工具扩展与故障排查。

---

## 目录

1. [先搞清楚：你要接的是什么](#1-先搞清楚你要接的是什么)
2. [架构总览](#2-架构总览)
3. [方式一：Agent 已封装为 API（推荐入门）](#3-方式一agent-已封装为-api推荐入门)
4. [方式二：扩展工具（LLM + Tool Calling）](#4-方式二扩展工具llm--tool-calling)
5. [方式三：完全自定义 Agent 逻辑](#5-方式三完全自定义-agent-逻辑)
6. [Agent 题库编写规范](#6-agent-题库编写规范)
7. [判分规则与 checks 类型](#7-判分规则与-checks-类型)
8. [配置 Profile 与权重](#8-配置-profile-与权重)
9. [完整工作流示例](#9-完整工作流示例)
10. [常见问题](#10-常见问题)

---

## 1. 先搞清楚：你要接的是什么

| 你的 Agent 形态 | 典型技术栈 | 推荐接入方式 |
|----------------|-----------|-------------|
| **黑盒 API 服务** | 自研后端、Dify、FastGPT 网关 | [方式一](#3-方式一agent-已封装为-api推荐入门) |
| **LLM + 工具调用** | vLLM function calling、OpenAI tools | [方式二](#4-方式二扩展工具llm--tool-calling) |
| **完整 Agent 框架** | LangGraph、CrewAI、AutoGen、自研 ReAct | [方式三](#5-方式三完全自定义-agent-逻辑) |

**核心原则**：评测框架负责 **出题 → 调用被测对象 → 判分 → 出报告**；你的 Agent 负责 **理解任务并执行**。

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                     agent_ranking CLI                        │
│                  python cli.py run --model xxx               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   BenchmarkRunner                            │
│   按 profile 加载题库 → 逐套件执行 → 汇总报告                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    reasoning/        dialogue/          agent/  ← 你的 Agent 主要在这里
    logic/            accuracy/          code/
          │                │                │
          └────────────────┼────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              ModelClient（OpenAI 兼容适配层）                   │
│   chat() / stream_chat() / tools                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   你的 Agent API                    自定义 BenchmarkSuite
   (configs/models.yaml)             (直接调 agent.run())
```

**关键扩展点：**

| 扩展点 | 文件 | 作用 |
|--------|------|------|
| 模型/Agent endpoint | `configs/models.yaml` | 配置 API 地址 |
| 工具注册 | `agent_ranking/tools/mock_registry.py` | 添加 mock/真实工具 |
| Agent 评测循环 | `agent_ranking/suites/builtin/agent.py` | 默认 tool-calling 循环 |
| 自定义套件 | 继承 `BenchmarkSuite` + `register_suite()` | 完全接管调用逻辑 |
| 题库 | `datasets/<profile>/agent.jsonl` | 定义任务与预期 |
| 评测方案 | `configs/profiles.yaml` | 选择跑哪些套件 |

---

## 3. 方式一：Agent 已封装为 API（推荐入门）

适用于：你的 Agent 已经是一个 HTTP 服务，内部自行处理规划、记忆、工具调用，对外提供统一对话接口。

### 3.1 前提条件

你的服务需满足以下**至少一条**：

- 提供 **OpenAI 兼容** 的 `POST /v1/chat/completions`
- 或能被一层薄网关转成该格式

### 3.2 配置 models.yaml

```yaml
# configs/models.yaml
models:
  my-agent:
    adapter: openai_compat
    base_url: "http://127.0.0.1:9000/v1"       # 你的 Agent 服务
    api_key: "EMPTY"                              # 或 ${MY_AGENT_KEY}
    model_name: "my-agent"                        # 与服务端 model 名一致
    tier: medium                                  # 30B-40B 用 medium；大模型用 large/xlarge
    supports_tools: false                         # 若 Agent 内部处理工具，外部不可见则设 false
    max_context: 32768
    extra_headers: {}                             # 可选自定义头
```

**tier 参考：**

| 规模 | tier | 并发 |
|------|------|------|
| 30B–40B | `medium` | 4 |
| 50B–70B | `large` | 2 |
| 80B–100B+ | `xlarge` | 1 |

### 3.3 探测能力

```bash
python cli.py probe --model my-agent
```

关注输出：

```json
{
  "healthy": true,
  "capabilities": {
    "chat": true,
    "tools": false
  },
  "skipped_suites": ["agent"]
}
```

- 若 `tools: false`，内置 `agent` 套件（依赖外部 tool calling）会被跳过
- 你仍可测 `reasoning` / `code` / `dialogue` 等**对话类**能力
- 若你的 Agent 内部有工具但 API 不暴露，建议用 [方式三](#5-方式三完全自定义-agent-逻辑) 写自定义套件

### 3.4 运行评测

```bash
# 快速
python cli.py run --model my-agent --profile smoke

# 通用基准
python cli.py run --model my-agent --profile benchmark-light

# 只看某个套件
python cli.py run-suite --suite dialogue \
  --path datasets/smoke/dialogue.jsonl \
  --model my-agent
```

### 3.5 网关示例（FastAPI 包装你的 Agent）

若你的 Agent 是 Python 函数，可快速包一层 OpenAI 兼容 API：

```python
# my_agent_server.py（示例，非本项目代码）
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.0
    max_tokens: int = 512

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_msg = req.messages[-1]["content"]
    answer = my_agent.run(user_msg)   # ← 你的 Agent
    return {
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
```

启动后把 `base_url` 指向 `http://127.0.0.1:8000/v1` 即可。

---

## 4. 方式二：扩展工具（LLM + Tool Calling）

适用于：被测对象是**支持 OpenAI function calling 的 LLM**，评测框架负责工具循环，你需要添加自己的业务工具。

### 4.1 内置工具一览

当前 `MockToolRegistry` 提供 4 个离线 mock 工具：

| 工具名 | 功能 |
|--------|------|
| `calculator` | 安全数学计算 |
| `search_docs` | 搜索题目预置的 `docs` 字典 |
| `read_file` | 读取题目预置的 `files` 虚拟文件系统 |
| `list_files` | 列出虚拟目录文件 |

### 4.2 添加自定义工具

编辑 `agent_ranking/tools/mock_registry.py`：

```python
# 1. 在 __init__ 中注册 handler
self._handlers = {
    "calculator": self._calculator,
    "search_docs": self._search_docs,
    "read_file": self._read_file,
    "list_files": self._list_files,
    "query_db": self._query_db,          # ← 新增
}

# 2. 在 get_tool_definitions() 中添加 schema
{
    "type": "function",
    "function": {
        "name": "query_db",
        "description": "查询业务数据库",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL 查询语句"}
            },
            "required": ["sql"],
        },
    },
},

# 3. 实现 handler
def _query_db(self, args: dict) -> str:
    sql = args.get("sql", "")
    # 接你的真实逻辑，或 mock 返回
    return json.dumps({"rows": [{"id": 1, "name": "test"}]})
```

### 4.3 接真实外部服务（注意安全）

```python
def _call_my_api(self, args: dict) -> str:
    import httpx
    resp = httpx.post("http://internal-api/query", json=args, timeout=10)
    return resp.text
```

建议：
- 评测环境用 **mock 数据**，避免污染生产
- 设置超时，防止 Agent 卡死
- 返回 JSON 字符串，便于模型解析

### 4.4 配套题库

```json
{
  "id": "agent_custom_001",
  "suite": "agent",
  "task": "查询用户表中 id=1 的用户名，使用 query_db 工具。",
  "checks": [
    {"type": "tool_called", "tool": "query_db"},
    {"type": "answer_contains", "value": "test"}
  ]
}
```

### 4.5 运行

```bash
python cli.py run-suite \
  --suite agent \
  --path datasets/my_agent/agent.jsonl \
  --model qwen3-72b
```

---

## 5. 方式三：完全自定义 Agent 逻辑

适用于：你的 Agent 有独立的规划循环、记忆、多 Agent 协作，**不走标准 OpenAI tool calling**。

### 5.1 实现自定义 BenchmarkSuite

参考 `examples/custom_agent_suite.py`（项目已附带完整示例）：

```python
from agent_ranking.core.types import EvalResult
from agent_ranking.suites.base import BenchmarkSuite
from agent_ranking.suites.registry import register_suite


class CustomAgentSuite(BenchmarkSuite):
    """直接调用你自己的 Agent，不依赖 OpenAI tool calling。"""

    name = "custom_agent"

    def __init__(self, dataset_path=None, agent_factory=None, **kwargs):
        super().__init__(dataset_path)
        self.agent_factory = agent_factory or (lambda: MyAgent())

    def run_item(self, client, item, **kwargs) -> EvalResult:
        agent = self.agent_factory()
        task = item.get("task") or item.get("prompt", "")

        # 调用你的 Agent（示例接口）
        result = agent.run(
            task=task,
            context=item.get("context", {}),
            max_steps=item.get("max_steps", 10),
        )
        # result 期望: {"answer": str, "steps": list, "tools_used": list}

        score, passed, detail = self._score(item, result)
        return EvalResult(
            item_id=item["id"],
            suite=self.name,
            score=score,
            passed=passed,
            detail=detail,
            response=result.get("answer", ""),
            latency_ms=result.get("latency_ms", 0),
        )

    def _score(self, item, result):
        checks = item.get("checks", [])
        scores = []
        tools_used = result.get("tools_used", [])
        answer = result.get("answer", "")

        for check in checks:
            t = check.get("type")
            if t == "tool_called":
                scores.append(1.0 if check["tool"] in tools_used else 0.0)
            elif t == "answer_contains":
                scores.append(1.0 if check["value"] in answer else 0.0)

        if item.get("expected_answer"):
            scores.append(1.0 if item["expected_answer"] in answer else 0.0)

        final = sum(scores) / len(scores) if scores else 0.5
        return final, final >= item.get("pass_threshold", 0.6), {"result": result}


# 注册套件
register_suite("custom_agent", CustomAgentSuite)
```

### 5.2 注册到评测流程

**方法 A：在 CLI 启动前 import**

```python
# run_my_agent.py
import examples.custom_agent_suite  # 触发 register_suite
from cli import main
main()
```

**方法 B：修改 `configs/profiles.yaml`**

```yaml
profiles:
  my-agent-eval:
    description: "评测我的 Agent"
    suites:
      - custom_agent
    dataset_profile: my_agent
```

并创建 `datasets/my_agent/custom_agent.jsonl`。

### 5.3 你的 Agent 只需实现统一接口

```python
class MyAgent:
    def run(self, task: str, context: dict = None, max_steps: int = 10) -> dict:
        """
        返回格式建议：
        {
            "answer": "最终回答文本",
            "tools_used": ["calculator", "search"],
            "steps": [...],
            "latency_ms": 1234.5,
        }
        """
        ...
```

### 5.4 LangGraph / LangChain 接入示例

```python
from langgraph.prebuilt import create_react_agent

class LangGraphAgent:
    def __init__(self, llm, tools):
        self.graph = create_react_agent(llm, tools)

    def run(self, task, **kwargs):
        import time
        t0 = time.monotonic()
        state = self.graph.invoke({"messages": [("user", task)]})
        answer = state["messages"][-1].content
        return {
            "answer": answer,
            "tools_used": self._extract_tools(state),
            "latency_ms": (time.monotonic() - t0) * 1000,
        }
```

在 `CustomAgentSuite` 里实例化 `LangGraphAgent` 替代 `MyAgent` 即可。

---

## 6. Agent 题库编写规范

### 6.1 文件位置

```
datasets/
├── smoke/agent.jsonl           # 内置
├── my_agent/custom_agent.jsonl # 你的自定义题
```

每行一题，JSON 格式。

### 6.2 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | ✅ | 唯一标识，如 `agent_001` |
| `suite` | ✅ | 套件名，默认 `agent` 或 `custom_agent` |
| `task` | ✅ | 给 Agent 的任务描述 |
| `system` | ❌ | 系统提示词 |
| `docs` | ❌ | 虚拟文档库 `{"doc_id": "content"}` |
| `files` | ❌ | 虚拟文件系统 `{"/path": "content"}` |
| `expected_tools` | ❌ | 预期调用的工具列表 |
| `expected_answer` | ❌ | 预期最终答案包含的文本 |
| `checks` | ❌ | 判分规则数组 |
| `pass_threshold` | ❌ | 通过阈值，默认 0.6 |
| `max_tokens` | ❌ | 单次生成长度 |
| `lang` | ❌ | 标注语言 `en` / `zh` |

### 6.3 示例题库

**单工具：**

```json
{
  "id": "agent_001",
  "suite": "agent",
  "lang": "zh",
  "task": "用 calculator 计算 256 × 128",
  "expected_tools": ["calculator"],
  "expected_answer": "32768",
  "checks": [
    {"type": "tool_called", "tool": "calculator"},
    {"type": "answer_contains", "value": "32768"}
  ]
}
```

**多工具链：**

```json
{
  "id": "agent_002",
  "suite": "agent",
  "lang": "zh",
  "task": "先搜索「部署」相关文档，再计算 1024/16",
  "docs": {"guide": "推荐使用 vLLM 部署，默认端口 8000"},
  "expected_tools": ["search_docs", "calculator"],
  "checks": [
    {"type": "tool_called", "tool": "search_docs"},
    {"type": "tool_called", "tool": "calculator"},
    {"type": "answer_contains", "value": "64"},
    {"type": "max_steps", "value": 5}
  ]
}
```

**失败恢复：**

```json
{
  "id": "agent_003",
  "suite": "agent",
  "system": "工具失败时应换策略，不要编造结果。",
  "task": "读取 /secret/key.txt；若失败则列出 /secret 目录并读取存在的文件。",
  "files": {"/secret/token.txt": "api_key=sk-abc"},
  "checks": [
    {"type": "tool_called", "tool": "list_files"},
    {"type": "answer_contains", "value": "sk-abc"}
  ]
}
```

---

## 7. 判分规则与 checks 类型

### Agent 专用 checks

| type | 参数 | 说明 |
|------|------|------|
| `tool_called` | `tool`: 工具名 | 是否调用了指定工具 |
| `answer_contains` | `value`: 子串 | 最终回答是否包含 |
| `max_steps` | `value`: 整数 | 步数是否不超过限制 |

### 通用 checks（dialogue/accuracy 等也可用）

| type | 说明 |
|------|------|
| `exact` | 精确匹配 |
| `contains` | 包含子串 |
| `numeric` | 数值匹配（支持 `tolerance`） |
| `choice` | 选择题 A/B/C/D |
| `regex` | 正则匹配 |

### 综合得分

```
Agent分 = mean(各 check 得分)
通过   = Agent分 >= pass_threshold（默认 0.6）
```

---

## 8. 配置 Profile 与权重

### 8.1 创建专属 profile

```yaml
# configs/profiles.yaml
profiles:
  my-agent-smoke:
    description: "我的 Agent 冒烟测试"
    suites:
      - custom_agent    # 或 agent
      - dialogue
    dataset_profile: my_agent

  my-agent-full:
    description: "我的 Agent + 通用基准"
    suites:
      - custom_agent
      - reasoning
      - code
      - dialogue
    dataset_profile: benchmark-light
```

### 8.2 调整权重

```yaml
# configs/weights.yaml
weights:
  custom_agent: 0.30   # 你的 Agent 权重加大
  reasoning: 0.15
  code: 0.15
  dialogue: 0.15
  agent: 0.10
  speed: 0.10
```

### 8.3 运行

```bash
python cli.py run --model my-agent --profile my-agent-smoke
```

---

## 9. 完整工作流示例

以「自研 Agent + 自定义套件 + 自定义题库」为例：

```bash
# ① 安装
pip install -e .

# ② 实现并注册自定义套件（见 examples/custom_agent_suite.py）
# ③ 创建题库
mkdir -p datasets/my_agent
cp datasets/smoke/agent.jsonl datasets/my_agent/custom_agent.jsonl
# 编辑题目...

# ④ 配置 profile（configs/profiles.yaml）
# ⑤ 若 Agent 是 API，配置 configs/models.yaml

# ⑥ 探测
python cli.py probe --model my-agent

# ⑦ 单套件调试
python cli.py run-suite \
  --suite custom_agent \
  --path datasets/my_agent/custom_agent.jsonl \
  --model my-agent

# ⑧ 完整评测
python cli.py run --model my-agent --profile my-agent-smoke

# ⑨ 查看报告
open reports/my-agent/report.html
```

---

## 10. 常见问题

### Q: probe 显示 tools=false，agent 套件被跳过？

**原因**：模型/API 不支持 OpenAI function calling。

**处理**：
- 换支持 tools 的 instruct 模型
- 或用 [方式三](#5-方式三完全自定义-agent-逻辑) 写 `CustomAgentSuite`，不依赖外部 tool calling

### Q: 我的 Agent 内部有工具，但 API 不暴露 tool_calls？

用 **方式三**：在你的 Suite 里调 `agent.run()`，由你的 Agent 返回 `tools_used` 列表，框架据此判分。

### Q: 能否同时评测「裸 LLM」和「Agent」？

可以，配置两个 model：

```yaml
models:
  raw-llm:
    base_url: "http://localhost:8000/v1"
    model_name: "qwen3-72b"
  my-agent:
    base_url: "http://localhost:9000/v1"
    model_name: "my-agent"
```

```bash
python cli.py run-multi --models raw-llm,my-agent --profile benchmark-light
```

### Q: 如何只测 Agent，不测推理/代码？

```yaml
profiles:
  agent-only:
    suites: [agent, dialogue]
    dataset_profile: smoke
```

### Q: 评测很慢怎么办？

- 大模型设 `tier: xlarge`（串行，避免 OOM）
- 用 `--no-speed` 跳过速度测试
- 先用 `smoke` 或 `benchmark-light`

### Q: 能否接 MCP 工具？

可以。两种方式：
1. 在 `MockToolRegistry` 里封装 MCP 调用
2. 在自定义 Suite 里由你的 Agent 自行处理 MCP，评测框架只判最终 `answer` 和 `tools_used`

---

## 附录：三种方式选型速查

```
你的 Agent 是 HTTP 服务？
  └─ 是 → 方式一（配 models.yaml）
  └─ 否 → 是 LLM + 标准 tool calling？
           └─ 是 → 方式二（扩展 MockToolRegistry）
           └─ 否 → 方式三（自定义 BenchmarkSuite）
```

---

更多基础用法见 [README.md](../README.md)。
