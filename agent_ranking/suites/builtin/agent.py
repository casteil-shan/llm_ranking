from __future__ import annotations

import json
from typing import Any

from agent_ranking.core.types import EvalResult, ToolCall
from agent_ranking.suites.base import BenchmarkSuite
from agent_ranking.tools.mock_registry import MockToolRegistry


class AgentSuite(BenchmarkSuite):
    name = "agent"
    judge_type = "agent"

    def __init__(self, dataset_path=None, **kwargs):
        super().__init__(dataset_path)
        self.max_steps = kwargs.get("max_steps", 6)

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        registry = MockToolRegistry(
            docs=item.get("docs", {}),
            files=item.get("files", {}),
        )
        tools = registry.get_tool_definitions()
        task = item.get("task") or item.get("prompt", "")
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": item.get(
                    "system",
                    "你是任务执行 Agent。使用提供的工具完成任务，按需多步调用。",
                ),
            },
            {"role": "user", "content": task},
        ]

        steps_log = []
        total_latency = 0.0
        final_answer = ""

        for step in range(self.max_steps):
            try:
                resp = client.chat(
                    messages,
                    tools=tools,
                    temperature=0.0,
                    max_tokens=item.get("max_tokens", 512),
                )
            except Exception:
                # 不支持 tools 时回退到文本模式
                resp = client.chat(messages, temperature=0.0, max_tokens=512)

            total_latency += resp.latency_ms
            final_answer = resp.content

            if resp.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in resp.tool_calls
                    ],
                })
                for tc in resp.tool_calls:
                    result = registry.execute(tc.name, tc.arguments)
                    steps_log.append({
                        "step": step,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result": result,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                steps_log.append({"step": step, "final": resp.content})
                break

        score, passed, detail = self._evaluate_agent(item, steps_log, final_answer)
        return EvalResult(
            item_id=item["id"],
            suite=self.name,
            score=score,
            passed=passed,
            detail=detail,
            response=final_answer,
            latency_ms=total_latency,
        )

    def _evaluate_agent(
        self,
        item: dict[str, Any],
        steps_log: list[dict],
        final_answer: str,
    ) -> tuple[float, bool, dict]:
        expected_tools = item.get("expected_tools", [])
        expected_answer = item.get("expected_answer")
        checks = item.get("checks", [])

        scores = []
        details: dict[str, Any] = {"steps": steps_log}

        # 工具调用检查
        used_tools = [s["tool"] for s in steps_log if "tool" in s]
        if expected_tools:
            tool_score = sum(1 for t in expected_tools if t in used_tools) / len(expected_tools)
            scores.append(tool_score)
            details["tool_score"] = tool_score

        # 最终答案检查
        if expected_answer:
            answer_ok = str(expected_answer) in final_answer or str(expected_answer) in json.dumps(steps_log)
            scores.append(1.0 if answer_ok else 0.0)
            details["answer_ok"] = answer_ok

        # 自定义 checks
        for check in checks:
            ctype = check.get("type")
            if ctype == "tool_called":
                ok = check.get("tool") in used_tools
                scores.append(1.0 if ok else 0.0)
            elif ctype == "answer_contains":
                ok = check.get("value", "") in final_answer
                scores.append(1.0 if ok else 0.0)
            elif ctype == "max_steps":
                ok = len(steps_log) <= check.get("value", 10)
                scores.append(1.0 if ok else 0.0)

        if not scores:
            # 无预期时，有工具调用且未报错即通过
            scores.append(1.0 if used_tools else 0.5)

        final_score = sum(scores) / len(scores)
        passed = final_score >= item.get("pass_threshold", 0.6)
        return final_score, passed, details
