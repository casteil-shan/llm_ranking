from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent_ranking.adapters.base import ModelClient
from agent_ranking.adapters.capability_probe import probe_model
from agent_ranking.adapters.openai_compat import create_client
from agent_ranking.core.config import ConfigLoader
from agent_ranking.core.types import BenchmarkReport, SuiteSummary
from agent_ranking.judges.llm_judge import LLMJudge
from agent_ranking.reports.generator import ReportGenerator
from agent_ranking.runners.speed_profiler import SpeedProfiler
from agent_ranking.suites.registry import get_suite, list_suites

console = Console()


class BenchmarkRunner:
    def __init__(self, root: Path | None = None):
        self.config = ConfigLoader(root)
        self.report_gen = ReportGenerator()

    def create_client(self, model_name: str) -> ModelClient:
        model_cfg = self.config.get_model_config(model_name)
        return create_client(model_cfg)

    def create_judge(self) -> LLMJudge | None:
        judge_cfg = self.config.load_judge()
        if not judge_cfg.get("judge", {}).get("enabled"):
            return None
        j = judge_cfg["judge"]
        tier = self.config.get_tier_config(j.get("tier", "small"))
        client = create_client({**j, **{"default_max_tokens": tier.get("default_max_tokens", 256)}})
        return LLMJudge(client)

    def probe(self, model_name: str) -> dict[str, Any]:
        client = self.create_client(model_name)
        return probe_model(client)

    def run_profile(
        self,
        model_name: str,
        profile: str = "smoke",
        output_dir: Path | None = None,
        include_speed: bool = True,
        dataset_profile: str | None = None,
    ) -> BenchmarkReport:
        model_cfg = self.config.get_model_config(model_name)
        tier_cfg = self.config.get_tier_config(model_cfg.get("tier", "large"))
        profile_cfg = self.config.get_profile(profile)
        ds_profile = dataset_profile or profile_cfg.get("dataset_profile") or profile
        if not self.config.dataset_path("reasoning", ds_profile).parent.exists():
            ds_profile = "smoke"

        client = self.create_client(model_name)
        llm_judge = self.create_judge()

        probe = probe_model(client)
        if not probe["healthy"]:
            raise RuntimeError(f"Model '{model_name}' health check failed.")

        skipped = set(probe.get("skipped_suites", []))
        suite_names = [s for s in profile_cfg.get("suites", list_suites()) if s not in skipped]

        summaries: list[SuiteSummary] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for suite_name in suite_names:
                task = progress.add_task(f"Running {suite_name}...", total=None)
                path = self.config.dataset_path(suite_name, ds_profile)
                kwargs: dict[str, Any] = {}
                if suite_name in ("accuracy", "dialogue"):
                    kwargs["llm_judge"] = llm_judge
                suite = get_suite(suite_name, dataset_path=path, **kwargs)
                if not suite.items:
                    console.print(f"[yellow]Skip {suite_name}: no items in {path}[/yellow]")
                    progress.remove_task(task)
                    continue
                summary = suite.run_all(client)
                summaries.append(summary)
                progress.remove_task(task)
                console.print(
                    f"  [green]{suite_name}[/green]: "
                    f"{summary.passed}/{summary.total} passed, "
                    f"avg={summary.avg_score:.2%}"
                )

        speed = None
        if include_speed:
            profiler = SpeedProfiler(client, tier_cfg)
            speed = profiler.run()
            console.print(
                f"  [blue]speed[/blue]: TTFT={speed.ttft_ms}ms, "
                f"TPS={speed.tokens_per_sec:.1f}"
            )

        weights = self.config.load_weights().get("weights", {})
        composite = self._compute_composite(summaries, speed, weights)

        report = BenchmarkReport(
            model=model_name,
            profile=profile,
            suites=summaries,
            speed=speed,
            composite_score=composite,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "probe": probe,
                "tier": model_cfg.get("tier"),
                "dataset_profile": ds_profile,
            },
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.report_gen.save(report, output_dir)

        return report

    def run_multi(
        self,
        model_names: list[str],
        profile: str = "smoke",
        output_dir: Path | None = None,
    ) -> list[BenchmarkReport]:
        reports = []
        for name in model_names:
            console.print(f"\n[bold]Evaluating {name}[/bold]")
            out = output_dir / name if output_dir else None
            reports.append(self.run_profile(name, profile=profile, output_dir=out))
        if output_dir and len(reports) > 1:
            self.report_gen.save_ranking(reports, output_dir)
        return reports

    def _compute_composite(
        self,
        summaries: list[SuiteSummary],
        speed,
        weights: dict[str, float],
    ) -> float:
        scores: dict[str, float] = {s.suite: s.avg_score * 100 for s in summaries}
        if speed and speed.tokens_per_sec > 0:
            # 速度分：以 50 tokens/s 为满分参考
            scores["speed"] = min(100.0, speed.tokens_per_sec / 50.0 * 100)

        total_weight = 0.0
        weighted = 0.0
        for key, w in weights.items():
            if key in scores:
                weighted += scores[key] * w
                total_weight += w
        return weighted / total_weight if total_weight else 0.0
