from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    model_name: str
    max_context: int
    supports_tools: bool
    tier: str


@dataclass
class SpeedMetrics:
    ttft_ms: float | None
    total_ms: float
    tokens_per_sec: float
    completion_tokens: int
    prompt_tokens: int = 0


@dataclass
class EvalResult:
    item_id: str
    suite: str
    score: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    response: str = ""
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class SuiteSummary:
    suite: str
    total: int
    passed: int
    avg_score: float
    avg_latency_ms: float
    results: list[EvalResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    model: str
    profile: str
    suites: list[SuiteSummary]
    speed: SpeedMetrics | None = None
    composite_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
