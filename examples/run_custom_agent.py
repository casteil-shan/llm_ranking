#!/usr/bin/env python3
"""运行自定义 Agent 套件评测的示例入口。"""

import sys
from pathlib import Path

# 注册 custom_agent 套件
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import examples.custom_agent_suite  # noqa: F401, E402

from agent_ranking.core.config import ConfigLoader
from agent_ranking.runners.benchmark_runner import BenchmarkRunner
from agent_ranking.suites.registry import get_suite
from rich.console import Console

console = Console()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    dataset = Path(__file__).resolve().parents[1] / "datasets" / "smoke" / "agent.jsonl"

    runner = BenchmarkRunner()
    client = runner.create_client(model_name)

    # 使用自定义套件，但 client 仅用于兼容接口（实际调用 MyAgent）
    suite = get_suite("custom_agent", dataset_path=dataset)
    summary = suite.run_all(client)

    console.print(
        f"[bold]{summary.suite}[/bold]: "
        f"{summary.passed}/{summary.total} passed, "
        f"avg={summary.avg_score:.2%}"
    )


if __name__ == "__main__":
    main()
