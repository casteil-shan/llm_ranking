from __future__ import annotations

from typing import Any

from agent_ranking.core.types import EvalResult
from agent_ranking.judges.rule_judge import RuleJudge
from agent_ranking.suites.base import BenchmarkSuite


class ReasoningSuite(BenchmarkSuite):
    name = "reasoning"
    judge_type = "rule"

    def __init__(self, dataset_path=None, judge=None, **kwargs):
        super().__init__(dataset_path)
        self.judge = judge or RuleJudge()

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        prompt = item.get("prompt") or item.get("question", "")
        system = item.get("system", "你是一个精确的推理助手。请一步步思考，最后给出明确答案。")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat(messages, temperature=0.0, max_tokens=item.get("max_tokens", 512))
        result = self.judge.evaluate(item, resp.content)
        result.latency_ms = resp.latency_ms
        result.response = resp.content
        return result
