from __future__ import annotations

from typing import Any

from agent_ranking.core.types import EvalResult
from agent_ranking.judges.llm_judge import LLMJudge
from agent_ranking.judges.rule_judge import RuleJudge
from agent_ranking.suites.base import BenchmarkSuite


class AccuracySuite(BenchmarkSuite):
    name = "accuracy"
    judge_type = "rule"

    def __init__(self, dataset_path=None, judge=None, llm_judge=None, **kwargs):
        super().__init__(dataset_path)
        self.rule_judge = judge or RuleJudge()
        self.llm_judge = llm_judge

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        prompt = item.get("prompt") or item.get("question", "")
        messages = [{"role": "user", "content": prompt}]
        if item.get("system"):
            messages.insert(0, {"role": "system", "content": item["system"]})

        resp = client.chat(messages, temperature=0.0, max_tokens=item.get("max_tokens", 512))

        judge_mode = item.get("judge", "rule")
        if judge_mode == "llm" and self.llm_judge:
            result = self.llm_judge.evaluate(item, resp.content)
        else:
            result = self.rule_judge.evaluate(item, resp.content)

        result.latency_ms = resp.latency_ms
        result.response = resp.content
        return result
