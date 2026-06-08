from __future__ import annotations

from typing import Any, AsyncIterator, Protocol

from agent_ranking.core.types import ModelInfo, ModelResponse, SpeedMetrics


class ModelClient(Protocol):
    model_name: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> ModelResponse: ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> ModelResponse: ...

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> Any: ...

    def health_check(self) -> bool: ...

    def get_model_info(self) -> ModelInfo: ...

    def benchmark_speed(
        self,
        prompt: str = "请用一句话介绍人工智能。",
        max_tokens: int = 128,
        runs: int = 3,
    ) -> SpeedMetrics: ...
