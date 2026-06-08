from __future__ import annotations

import json
import re
from typing import Any

from agent_ranking.adapters.openai_compat import OpenAICompatClient
from agent_ranking.core.types import EvalResult


class LLMJudge:
    """使用本地 Judge 模型对开放题打分。"""

    name = "llm"

    def __init__(self, client: OpenAICompatClient | None):
        self.client = client

    def evaluate(self, item: dict[str, Any], response: str, **kwargs: Any) -> EvalResult:
        if not self.client:
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "unknown"),
                score=0.0,
                passed=False,
                detail={"reason": "judge model not configured"},
                response=response,
            )

        rubric = item.get("rubric", "回答是否正确、完整")
        reference = item.get("reference", "")
        prompt = f"""你是一个严格的评测裁判。请根据评分标准对模型回答打分。

【题目】{item.get('question', item.get('prompt', ''))}
【参考答案】{reference}
【评分标准】{rubric}
【模型回答】{response}

请以 JSON 格式回复，包含 score (0-1 浮点数) 和 reason (简短说明)：
{{"score": 0.8, "reason": "..."}}"""

        try:
            resp = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.content)
            score = float(data.get("score", 0))
            score = max(0.0, min(1.0, score))
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "unknown"),
                score=score,
                passed=score >= item.get("pass_threshold", 0.6),
                detail={"reason": data.get("reason", ""), "judge": "llm"},
                response=response,
            )
        except Exception:
            # 回退：从文本中提取分数
            score = self._extract_score_fallback(response)
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "unknown"),
                score=score,
                passed=score >= item.get("pass_threshold", 0.6),
                detail={"judge": "llm_fallback"},
                response=response,
            )

    def _extract_score_fallback(self, text: str) -> float:
        match = re.search(r"score[\"']?\s*[:=]\s*([0-9.]+)", text, re.IGNORECASE)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.0
