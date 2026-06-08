"""
自定义 Agent 接入示例

用法：
  1. 实现 MyAgent.run() 接口
  2. python examples/run_custom_agent.py

或在其他入口文件开头 import 本模块以注册套件：
  import examples.custom_agent_suite
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from agent_ranking.core.types import EvalResult
from agent_ranking.suites.base import BenchmarkSuite
from agent_ranking.suites.registry import register_suite


class AgentProtocol(Protocol):
    """你的 Agent 需实现的接口。"""

    def run(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
        max_steps: int = 10,
    ) -> dict[str, Any]:
        """
        返回:
            answer: str       最终回答
            tools_used: list  使用过的工具名
            steps: list       中间步骤（可选）
            latency_ms: float 耗时（可选）
        """
        ...


class MyAgent:
    """示例 Agent：替换为你自己的实现。"""

    def run(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
        max_steps: int = 10,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        # TODO: 替换为你的 Agent 逻辑
        # 示例：直接 echo
        answer = f"已处理任务: {task}"
        return {
            "answer": answer,
            "tools_used": [],
            "steps": [{"action": "echo", "result": answer}],
            "latency_ms": (time.monotonic() - t0) * 1000,
        }


class CustomAgentSuite(BenchmarkSuite):
    """不依赖 OpenAI tool calling，直接调用自定义 Agent。"""

    name = "custom_agent"

    def __init__(self, dataset_path=None, agent_factory=None, **kwargs):
        super().__init__(dataset_path)
        self.agent_factory = agent_factory or (lambda: MyAgent())

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        agent = self.agent_factory()
        task = item.get("task") or item.get("prompt", "")

        result = agent.run(
            task=task,
            context={
                "docs": item.get("docs", {}),
                "files": item.get("files", {}),
            },
            max_steps=item.get("max_steps", 10),
        )

        score, passed, detail = self._evaluate(item, result)
        return EvalResult(
            item_id=item["id"],
            suite=self.name,
            score=score,
            passed=passed,
            detail=detail,
            response=result.get("answer", ""),
            latency_ms=result.get("latency_ms", 0),
        )

    def _evaluate(self, item: dict, result: dict) -> tuple[float, bool, dict]:
        checks = item.get("checks", [])
        expected_tools = item.get("expected_tools", [])
        expected_answer = item.get("expected_answer")
        tools_used = result.get("tools_used", [])
        answer = result.get("answer", "")
        scores: list[float] = []

        if expected_tools:
            tool_score = sum(1 for t in expected_tools if t in tools_used) / len(expected_tools)
            scores.append(tool_score)

        if expected_answer:
            scores.append(1.0 if str(expected_answer) in answer else 0.0)

        for check in checks:
            ctype = check.get("type")
            if ctype == "tool_called":
                scores.append(1.0 if check.get("tool") in tools_used else 0.0)
            elif ctype == "answer_contains":
                scores.append(1.0 if check.get("value", "") in answer else 0.0)
            elif ctype == "max_steps":
                steps = result.get("steps", [])
                scores.append(1.0 if len(steps) <= check.get("value", 10) else 0.0)

        if not scores:
            scores.append(0.5)

        final_score = sum(scores) / len(scores)
        passed = final_score >= item.get("pass_threshold", 0.6)
        return final_score, passed, {"agent_result": result, "tools_used": tools_used}


# 注册到全局套件表
register_suite("custom_agent", CustomAgentSuite)
