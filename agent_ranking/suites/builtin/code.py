from __future__ import annotations

from typing import Any

from agent_ranking.core.types import EvalResult
from agent_ranking.judges.code_judge import CodeJudge
from agent_ranking.suites.base import BenchmarkSuite


class CodeSuite(BenchmarkSuite):
    name = "code"
    judge_type = "code"

    def __init__(self, dataset_path=None, judge=None, **kwargs):
        super().__init__(dataset_path)
        self.judge = judge or CodeJudge()

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        prompt = item.get("prompt") or item.get("question", "")
        system = item.get(
            "system",
            "你是编程助手。请用 Python 编写代码，将代码放在 ```python 代码块中。",
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat(messages, temperature=0.0, max_tokens=item.get("max_tokens", 1024))
        result = self.judge.evaluate(item, resp.content)
        result.latency_ms = resp.latency_ms
        result.response = resp.content
        return result
