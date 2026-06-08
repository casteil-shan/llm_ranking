from __future__ import annotations

from typing import Any

from agent_ranking.adapters.base import ModelClient
from agent_ranking.core.types import SpeedMetrics


class SpeedProfiler:
    def __init__(self, client: ModelClient, tier_config: dict[str, Any] | None = None):
        self.client = client
        self.tier_config = tier_config or {}
        self.runs = self.tier_config.get("speed_runs", 3)
        self.max_tokens = self.tier_config.get("default_max_tokens", 128)

    def run(
        self,
        prompt: str = "请用一句话介绍人工智能。",
        max_tokens: int | None = None,
    ) -> SpeedMetrics:
        return self.client.benchmark_speed(
            prompt=prompt,
            max_tokens=max_tokens or min(self.max_tokens, 128),
            runs=self.runs,
        )
