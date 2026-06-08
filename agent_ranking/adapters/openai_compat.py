from __future__ import annotations

import json
import statistics
import time
from typing import Any

from openai import OpenAI

from agent_ranking.core.types import ModelInfo, ModelResponse, SpeedMetrics, TokenUsage, ToolCall


class OpenAICompatClient:
    """OpenAI 兼容 API 客户端，适配 vLLM / TGI / SGLang / Ollama 等。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.base_url = config["base_url"]
        self.api_key = config.get("api_key") or "EMPTY"
        self.tier = config.get("tier", "large")
        self.supports_tools = config.get("supports_tools", True)
        self.max_context = config.get("max_context", 32768)
        self.default_max_tokens = config.get("default_max_tokens")

        extra_headers = config.get("extra_headers") or {}
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=extra_headers if extra_headers else None,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if stop:
            kwargs["stop"] = stop

        start = time.monotonic()
        response = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in choice.message.tool_calls
            ]

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=latency_ms,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> ModelResponse:
        return self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ):
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            try:
                self.complete("ping", max_tokens=4)
                return True
            except Exception:
                return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name=self.model_name,
            max_context=self.max_context,
            supports_tools=self.supports_tools,
            tier=self.tier,
        )

    def benchmark_speed(
        self,
        prompt: str = "请用一句话介绍人工智能。",
        max_tokens: int = 128,
        runs: int = 3,
    ) -> SpeedMetrics:
        metrics: list[SpeedMetrics] = []
        for _ in range(runs):
            metrics.append(self._single_speed_run(prompt, max_tokens))
        return SpeedMetrics(
            ttft_ms=_median([m.ttft_ms for m in metrics if m.ttft_ms is not None]),
            total_ms=statistics.median([m.total_ms for m in metrics]),
            tokens_per_sec=statistics.median([m.tokens_per_sec for m in metrics]),
            completion_tokens=int(statistics.median([m.completion_tokens for m in metrics])),
            prompt_tokens=int(statistics.median([m.prompt_tokens for m in metrics])),
        )

    def _single_speed_run(self, prompt: str, max_tokens: int) -> SpeedMetrics:
        messages = [{"role": "user", "content": prompt}]
        start = time.monotonic()
        ttft: float | None = None
        completion_tokens = 0
        prompt_tokens = 0

        stream = self.stream_chat(messages, max_tokens=max_tokens)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = (time.monotonic() - start) * 1000
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens

        total_ms = (time.monotonic() - start) * 1000
        total_sec = total_ms / 1000
        tps = completion_tokens / total_sec if total_sec > 0 else 0.0

        return SpeedMetrics(
            ttft_ms=ttft,
            total_ms=total_ms,
            tokens_per_sec=tps,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )


def create_client(config: dict[str, Any]) -> OpenAICompatClient:
    adapter = config.get("adapter", "openai_compat")
    if adapter != "openai_compat":
        raise ValueError(f"Unsupported adapter: {adapter}")
    return OpenAICompatClient(config)


def _median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.median(clean)
