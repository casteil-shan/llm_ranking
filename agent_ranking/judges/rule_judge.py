from __future__ import annotations

import re
from typing import Any

from agent_ranking.core.types import EvalResult


class RuleJudge:
    """基于规则的判分器，支持多种 check 类型。"""

    name = "rule"

    def evaluate(self, item: dict[str, Any], response: str, **kwargs: Any) -> EvalResult:
        checks = item.get("checks") or []
        if not checks and item.get("answer") is not None:
            checks = [{"type": "exact", "value": item["answer"]}]

        if not checks:
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "unknown"),
                score=0.0,
                passed=False,
                detail={"reason": "no checks defined"},
                response=response,
            )

        passed_count = 0
        details = []
        for check in checks:
            ok, detail = self._run_check(check, response, kwargs.get("session"))
            details.append({"check": check, "passed": ok, "detail": detail})
            if ok:
                passed_count += 1

        score = passed_count / len(checks)
        return EvalResult(
            item_id=item["id"],
            suite=item.get("suite", "unknown"),
            score=score,
            passed=score >= (item.get("pass_threshold", 1.0)),
            detail={"checks": details},
            response=response,
        )

    def _run_check(
        self,
        check: dict[str, Any],
        response: str,
        session: list[dict] | None = None,
    ) -> tuple[bool, str]:
        ctype = check.get("type", "exact")
        value = check.get("value")
        response_norm = response.strip()

        if ctype == "exact":
            return response_norm == str(value).strip(), f"expected exact: {value}"

        if ctype == "contains":
            return str(value) in response, f"expected contains: {value}"

        if ctype == "not_contains":
            return str(value) not in response, f"expected not contains: {value}"

        if ctype == "regex":
            pattern = check.get("pattern", value)
            return bool(re.search(pattern, response, re.IGNORECASE)), f"regex: {pattern}"

        if ctype == "any_of":
            options = check.get("values", [])
            return any(str(o) in response for o in options), f"any_of: {options}"

        if ctype == "numeric":
            expected = float(value)
            numbers = re.findall(r"-?\d+\.?\d*", response)
            if not numbers:
                return False, "no number found"
            actual = float(numbers[-1])
            tolerance = float(check.get("tolerance", 0.01))
            return abs(actual - expected) <= tolerance, f"expected {expected}, got {actual}"

        if ctype == "choice":
            # 选择题：A/B/C/D
            letter = str(value).upper()
            patterns = [
                rf"\b{letter}\b",
                rf"答案[是为：:]\s*{letter}",
                rf"选[项择][是为：:]\s*{letter}",
            ]
            matched = any(re.search(p, response, re.IGNORECASE) for p in patterns)
            return matched, f"expected choice {letter}"

        if ctype == "session_contains" and session:
            # 检查历史中是否包含某信息（对话记忆题）
            history = " ".join(
                m.get("content", "") for m in session if m.get("role") == "assistant"
            )
            return str(value) in history or str(value) in response, f"session contains {value}"

        return False, f"unknown check type: {ctype}"
