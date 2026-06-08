from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from agent_ranking.suites.base import BenchmarkSuite
from agent_ranking.suites.builtin.accuracy import AccuracySuite
from agent_ranking.suites.builtin.agent import AgentSuite
from agent_ranking.suites.builtin.code import CodeSuite
from agent_ranking.suites.builtin.dialogue import DialogueSuite
from agent_ranking.suites.builtin.logic import LogicSuite
from agent_ranking.suites.builtin.reasoning import ReasoningSuite

_SUITE_REGISTRY: dict[str, Type[BenchmarkSuite]] = {
    "reasoning": ReasoningSuite,
    "logic": LogicSuite,
    "accuracy": AccuracySuite,
    "code": CodeSuite,
    "dialogue": DialogueSuite,
    "agent": AgentSuite,
}


def register_suite(name: str, suite_cls: Type[BenchmarkSuite]) -> None:
    """注册自定义评测套件。"""
    _SUITE_REGISTRY[name] = suite_cls


def get_suite(name: str, dataset_path: Path | None = None, **kwargs: Any) -> BenchmarkSuite:
    if name not in _SUITE_REGISTRY:
        available = ", ".join(_SUITE_REGISTRY.keys())
        raise KeyError(f"Suite '{name}' not found. Available: {available}")
    return _SUITE_REGISTRY[name](dataset_path=dataset_path, **kwargs)


def list_suites() -> list[str]:
    return list(_SUITE_REGISTRY.keys())
