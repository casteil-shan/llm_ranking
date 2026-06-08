from __future__ import annotations

import json
from typing import Any

from agent_ranking.adapters.base import ModelClient


def probe_model(client: ModelClient) -> dict[str, Any]:
    """探测模型 API 能力，决定可运行哪些评测套件。"""
    report: dict[str, Any] = {
        "model": client.model_name,
        "healthy": False,
        "capabilities": {},
        "skipped_suites": [],
        "warnings": [],
    }

    report["healthy"] = client.health_check()
    if not report["healthy"]:
        report["warnings"].append("health_check failed")
        return report

    info = client.get_model_info()
    report["model_info"] = {
        "model_name": info.model_name,
        "max_context": info.max_context,
        "supports_tools": info.supports_tools,
        "tier": info.tier,
    }

    report["capabilities"]["chat"] = _test_chat(client)
    report["capabilities"]["streaming"] = _test_streaming(client)
    report["capabilities"]["tools"] = _test_tools(client, info.supports_tools)
    report["capabilities"]["json_mode"] = _test_json_mode(client)

    if not report["capabilities"]["tools"]:
        report["skipped_suites"].append("agent")
        report["warnings"].append("Tool calling not supported; agent suite will be skipped.")

    return report


def _test_chat(client: ModelClient) -> bool:
    try:
        resp = client.chat([{"role": "user", "content": "回复 OK"}], max_tokens=8)
        return bool(resp.content)
    except Exception:
        return False


def _test_streaming(client: ModelClient) -> bool:
    try:
        stream = client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=8)
        for _ in stream:
            return True
        return False
    except Exception:
        return False


def _test_tools(client: ModelClient, declared_support: bool) -> bool:
    if not declared_support:
        return False
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]
    try:
        resp = client.chat(
            [{"role": "user", "content": "计算 2+3，使用 calculator 工具"}],
            tools=tools,
            max_tokens=128,
        )
        if resp.tool_calls:
            return resp.tool_calls[0].name == "calculator"
        # 部分模型在文本中返回 JSON
        return "calculator" in resp.content.lower() or "2" in resp.content
    except Exception:
        return False


def _test_json_mode(client: ModelClient) -> bool:
    try:
        resp = client.chat(
            [{"role": "user", "content": '返回 JSON: {"status": "ok"}'}],
            response_format={"type": "json_object"},
            max_tokens=32,
        )
        json.loads(resp.content)
        return True
    except Exception:
        return False
