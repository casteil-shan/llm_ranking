from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent_ranking.adapters.base import ModelClient
from agent_ranking.core.types import EvalResult, SuiteSummary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    if not path.exists():
        return items
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class BenchmarkSuite(ABC):
    """可扩展评测套件基类。"""

    name: str = "base"
    judge_type: str = "rule"

    def __init__(self, dataset_path: Path | None = None):
        self.dataset_path = dataset_path
        self.items: list[dict[str, Any]] = []
        if dataset_path:
            self.items = load_jsonl(dataset_path)
            for item in self.items:
                item.setdefault("suite", self.name)

    @abstractmethod
    def run_item(
        self,
        client: ModelClient,
        item: dict[str, Any],
        **kwargs: Any,
    ) -> EvalResult: ...

    def run_all(self, client: ModelClient, **kwargs: Any) -> SuiteSummary:
        results: list[EvalResult] = []
        for item in self.items:
            try:
                result = self.run_item(client, item, **kwargs)
                results.append(result)
            except Exception as exc:
                results.append(
                    EvalResult(
                        item_id=item.get("id", "unknown"),
                        suite=self.name,
                        score=0.0,
                        passed=False,
                        error=str(exc),
                    )
                )

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / total if total else 0.0
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return SuiteSummary(
            suite=self.name,
            total=total,
            passed=passed,
            avg_score=avg_score,
            avg_latency_ms=avg_latency,
            results=results,
        )
