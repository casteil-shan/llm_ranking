#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agent_ranking.core.config import ConfigLoader, ROOT_DIR
from agent_ranking.runners.benchmark_runner import BenchmarkRunner
from agent_ranking.suites.registry import list_suites

console = Console()


@click.group()
@click.option("--root", type=click.Path(exists=True, file_okay=False), default=str(ROOT_DIR))
@click.pass_context
def cli(ctx, root):
    """Agent Ranking - 本地大模型量化评测框架"""
    ctx.ensure_object(dict)
    ctx.obj["root"] = Path(root)
    ctx.obj["runner"] = BenchmarkRunner(ctx.obj["root"])


@cli.command()
@click.pass_context
def list_models(ctx):
    """列出 configs/models.yaml 中配置的模型"""
    cfg = ConfigLoader(ctx.obj["root"])
    models = cfg.load_models().get("models", {})
    table = Table(title="Configured Models")
    table.add_column("Name")
    table.add_column("Base URL")
    table.add_column("Tier")
    for name, m in models.items():
        table.add_row(name, m.get("base_url", ""), m.get("tier", ""))
    console.print(table)


@cli.command("list-suites")
@click.pass_context
def list_suites_cmd(ctx):
    """列出可用评测套件"""
    table = Table(title="Benchmark Suites")
    table.add_column("Suite")
    for s in list_suites():
        table.add_row(s)
    console.print(table)


@cli.command()
@click.option("--model", "-m", required=True, help="models.yaml 中的模型名")
@click.pass_context
def probe(ctx, model):
    """探测模型 API 能力"""
    runner: BenchmarkRunner = ctx.obj["runner"]
    report = runner.probe(model)
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--model", "-m", required=True, help="待测模型名")
@click.option("--profile", "-p", default="smoke", help="评测 profile: smoke/fast/standard")
@click.option("--output", "-o", type=click.Path(), default="reports", help="报告输出目录")
@click.option("--no-speed", is_flag=True, help="跳过速度评测")
@click.option("--dataset-profile", default=None, help="题库目录名，默认与 profile 相同")
@click.pass_context
def run(ctx, model, profile, output, no_speed, dataset_profile):
    """运行评测"""
    runner: BenchmarkRunner = ctx.obj["runner"]
    out = Path(output) / model
    report = runner.run_profile(
        model,
        profile=profile,
        output_dir=out,
        include_speed=not no_speed,
        dataset_profile=dataset_profile,
    )
    console.print(f"\n[bold green]Done![/bold green] Composite: {report.composite_score:.1f}")
    console.print(f"Report: {out / 'report.html'}")


@cli.command("run-multi")
@click.option("--models", "-m", required=True, help="逗号分隔的模型名列表")
@click.option("--profile", "-p", default="smoke")
@click.option("--output", "-o", type=click.Path(), default="reports")
@click.pass_context
def run_multi(ctx, models, profile, output):
    """批量评测多个模型并生成排名"""
    runner: BenchmarkRunner = ctx.obj["runner"]
    names = [n.strip() for n in models.split(",")]
    out = Path(output)
    reports = runner.run_multi(names, profile=profile, output_dir=out)
    console.print("\n[bold]Ranking:[/bold]")
    for i, r in enumerate(sorted(reports, key=lambda x: x.composite_score, reverse=True)):
        console.print(f"  {i+1}. {r.model}: {r.composite_score:.1f}")


@cli.command()
@click.option("--model", "-m", required=True)
@click.option("--runs", default=3, help="测试轮数")
@click.pass_context
def benchmark_speed(ctx, model, runs):
    """单独运行速度基准测试"""
    from agent_ranking.runners.speed_profiler import SpeedProfiler

    runner: BenchmarkRunner = ctx.obj["runner"]
    cfg = ConfigLoader(ctx.obj["root"])
    model_cfg = cfg.get_model_config(model)
    tier_cfg = cfg.get_tier_config(model_cfg.get("tier", "large"))
    tier_cfg["speed_runs"] = runs
    client = runner.create_client(model)
    metrics = SpeedProfiler(client, tier_cfg).run()
    console.print_json(json.dumps({
        "ttft_ms": metrics.ttft_ms,
        "total_ms": metrics.total_ms,
        "tokens_per_sec": metrics.tokens_per_sec,
        "completion_tokens": metrics.completion_tokens,
    }, indent=2))


@cli.command()
@click.option("--suite", "-s", required=True, help="套件名")
@click.option("--path", "-p", required=True, type=click.Path(exists=True), help="JSONL 题库路径")
@click.option("--model", "-m", required=True)
@click.pass_context
def run_suite(ctx, suite, path, model):
    """对单个套件运行自定义题库"""
    from agent_ranking.suites.registry import get_suite

    runner: BenchmarkRunner = ctx.obj["runner"]
    client = runner.create_client(model)
    llm_judge = runner.create_judge()
    kwargs = {}
    if suite in ("accuracy", "dialogue"):
        kwargs["llm_judge"] = llm_judge
    s = get_suite(suite, dataset_path=Path(path), **kwargs)
    summary = s.run_all(client)
    console.print(f"{summary.suite}: {summary.passed}/{summary.total}, avg={summary.avg_score:.2%}")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
