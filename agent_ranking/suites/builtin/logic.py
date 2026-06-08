from __future__ import annotations

from typing import Any

from agent_ranking.core.types import EvalResult
from agent_ranking.judges.rule_judge import RuleJudge
from agent_ranking.suites.base import BenchmarkSuite


class LogicSuite(BenchmarkSuite):
    name = "logic"
    judge_type = "rule"

    def __init__(self, dataset_path=None, judge=None, **kwargs):
        super().__init__(dataset_path)
        self.judge = judge or RuleJudge()

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        prompt = item.get("prompt") or item.get("question", "")
        system = item.get("system", "你是逻辑推理专家。分析前提与结论，给出最终判断。")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat(messages, temperature=0.0, max_tokens=item.get("max_tokens", 256))
        result = self.judge.evaluate(item, resp.content)
        result.latency_ms = resp.latency_ms
        result.response = resp.content
        return result
