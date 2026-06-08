from __future__ import annotations

from typing import Any, Protocol

from agent_ranking.core.types import EvalResult


class Judge(Protocol):
    name: str

    def evaluate(self, item: dict[str, Any], response: str, **kwargs: Any) -> EvalResult: ...
