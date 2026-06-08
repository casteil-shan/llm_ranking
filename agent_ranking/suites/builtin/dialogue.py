from __future__ import annotations

from typing import Any

from agent_ranking.core.types import EvalResult
from agent_ranking.judges.llm_judge import LLMJudge
from agent_ranking.judges.rule_judge import RuleJudge
from agent_ranking.suites.base import BenchmarkSuite


class DialogueSuite(BenchmarkSuite):
    name = "dialogue"
    judge_type = "rule"

    def __init__(self, dataset_path=None, judge=None, llm_judge=None, **kwargs):
        super().__init__(dataset_path)
        self.rule_judge = judge or RuleJudge()
        self.llm_judge = llm_judge

    def run_item(self, client, item: dict[str, Any], **kwargs) -> EvalResult:
        turns = item.get("turns", [])
        session: list[dict[str, str]] = []
        if item.get("system"):
            session.append({"role": "system", "content": item["system"]})

        last_response = ""
        total_latency = 0.0
        turn_results = []

        for i, turn in enumerate(turns):
            user_msg = turn.get("user", "")
            session.append({"role": "user", "content": user_msg})
            resp = client.chat(session, temperature=0.0, max_tokens=item.get("max_tokens", 512))
            last_response = resp.content
            total_latency += resp.latency_ms
            session.append({"role": "assistant", "content": resp.content})

            check = turn.get("check")
            if check:
                mini_item = {
                    "id": f"{item['id']}_turn{i}",
                    "suite": self.name,
                    "checks": [check] if isinstance(check, dict) else check,
                }
                tr = self.rule_judge.evaluate(mini_item, resp.content, session=session)
                turn_results.append(tr)

        # 综合评分
        if turn_results:
            score = sum(t.score for t in turn_results) / len(turn_results)
            passed = all(t.passed for t in turn_results)
        elif item.get("rubric") and self.llm_judge:
            result = self.llm_judge.evaluate(item, last_response)
            score, passed = result.score, result.passed
        else:
            score, passed = 1.0, True  # 无检查项的对话剧本记为通过

        return EvalResult(
            item_id=item["id"],
            suite=self.name,
            score=score,
            passed=passed,
            detail={"turn_results": [t.detail for t in turn_results]},
            response=last_response,
            latency_ms=total_latency,
        )
