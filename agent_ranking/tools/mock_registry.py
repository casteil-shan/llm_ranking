from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any, Callable


class MockToolRegistry:
    """Agent 评测用的本地模拟工具，完全离线。"""

    def __init__(self, docs: dict[str, str] | None = None, files: dict[str, str] | None = None):
        self.docs = docs or {}
        self.files = files or {}
        self._handlers: dict[str, Callable[[dict], str]] = {
            "calculator": self._calculator,
            "search_docs": self._search_docs,
            "read_file": self._read_file,
            "list_files": self._list_files,
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "计算数学表达式，支持 + - * / 和括号",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "在本地文档库中搜索关键词",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取沙箱虚拟文件系统中的文件",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "列出沙箱目录下的文件",
                    "parameters": {
                        "type": "object",
                        "properties": {"directory": {"type": "string"}},
                        "required": [],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: str | dict) -> str:
        if name not in self._handlers:
            return json.dumps({"error": f"unknown tool: {name}"})
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {"raw": arguments}
        else:
            args = arguments
        try:
            return self._handlers[name](args)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _calculator(self, args: dict) -> str:
        expr = args.get("expression", "")
        result = _safe_eval(expr)
        return json.dumps({"result": result})

    def _search_docs(self, args: dict) -> str:
        query = args.get("query", "").lower()
        hits = []
        for doc_id, content in self.docs.items():
            if query in content.lower() or query in doc_id.lower():
                hits.append({"id": doc_id, "snippet": content[:200]})
        return json.dumps({"results": hits})

    def _read_file(self, args: dict) -> str:
        path = args.get("path", "")
        if path not in self.files:
            return json.dumps({"error": f"file not found: {path}"})
        return json.dumps({"content": self.files[path]})

    def _list_files(self, args: dict) -> str:
        directory = args.get("directory", "/")
        files = [p for p in self.files if p.startswith(directory)]
        return json.dumps({"files": files})


def _safe_eval(expr: str) -> float:
    """安全计算简单数学表达式。"""
    expr = expr.strip()
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError("unsupported operator")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return allowed_ops[ast.USub](_eval(node.operand))
        raise ValueError("unsupported expression")

    # 仅允许数字和运算符
    if not re.match(r"^[\d\s+\-*/().]+$", expr):
        raise ValueError("invalid characters in expression")
    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree))
