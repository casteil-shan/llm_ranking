from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from agent_ranking.core.types import BenchmarkReport


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>Agent Ranking Report - {{ model }}</title>
  <style>
    body { font-family: -apple-system, sans-serif; margin: 2rem; background: #f8f9fa; }
    h1 { color: #1a1a2e; }
    .card { background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 0.6rem; text-align: left; border-bottom: 1px solid #eee; }
    .score { font-size: 2rem; font-weight: bold; color: #4361ee; }
    .pass { color: #2d6a4f; } .fail { color: #d00000; }
  </style>
</head>
<body>
  <h1>评测报告: {{ model }}</h1>
  <div class="card">
    <p>Profile: <b>{{ profile }}</b> | 时间: {{ timestamp }}</p>
    <p class="score">综合分: {{ "%.1f"|format(composite) }}</p>
    {% if speed %}
    <p>速度: TTFT {{ speed.ttft_ms }}ms | {{ "%.1f"|format(speed.tokens_per_sec) }} tokens/s</p>
    {% endif %}
  </div>
  <div class="card">
    <h2>各维度得分</h2>
    <table>
      <tr><th>套件</th><th>通过</th><th>平均分</th><th>延迟(ms)</th></tr>
      {% for s in suites %}
      <tr>
        <td>{{ s.suite }}</td>
        <td>{{ s.passed }}/{{ s.total }}</td>
        <td>{{ "%.1f%%"|format(s.avg_score * 100) }}</td>
        <td>{{ "%.0f"|format(s.avg_latency_ms) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  <div class="card">
    <h2>失败样例</h2>
    <ul>
    {% for f in failures %}
      <li><b>{{ f.item_id }}</b> ({{ f.suite }}): score={{ "%.2f"|format(f.score) }}</li>
    {% endfor %}
    {% if not failures %}<li>无</li>{% endif %}
    </ul>
  </div>
</body>
</html>"""


class ReportGenerator:
    def save(self, report: BenchmarkReport, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        data = self._to_dict(report)
        (output_dir / "report.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        failures = [
            r for s in report.suites for r in s.results if not r.passed
        ][:20]

        html = Template(HTML_TEMPLATE).render(
            model=report.model,
            profile=report.profile,
            timestamp=report.metadata.get("timestamp", ""),
            composite=report.composite_score,
            speed=report.speed,
            suites=report.suites,
            failures=failures,
        )
        (output_dir / "report.html").write_text(html, encoding="utf-8")

    def save_ranking(self, reports: list[BenchmarkReport], output_dir: Path) -> None:
        ranking = sorted(reports, key=lambda r: r.composite_score, reverse=True)
        data = {
            "rankings": [
                {
                    "rank": i + 1,
                    "model": r.model,
                    "composite": r.composite_score,
                    "suites": {s.suite: s.avg_score for s in r.suites},
                }
                for i, r in enumerate(ranking)
            ]
        }
        (output_dir / "ranking.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _to_dict(self, report: BenchmarkReport) -> dict:
        return {
            "model": report.model,
            "profile": report.profile,
            "composite_score": report.composite_score,
            "metadata": report.metadata,
            "speed": {
                "ttft_ms": report.speed.ttft_ms,
                "total_ms": report.speed.total_ms,
                "tokens_per_sec": report.speed.tokens_per_sec,
            } if report.speed else None,
            "suites": [
                {
                    "suite": s.suite,
                    "total": s.total,
                    "passed": s.passed,
                    "avg_score": s.avg_score,
                    "avg_latency_ms": s.avg_latency_ms,
                    "results": [
                        {
                            "item_id": r.item_id,
                            "score": r.score,
                            "passed": r.passed,
                            "latency_ms": r.latency_ms,
                            "error": r.error,
                        }
                        for r in s.results
                    ],
                }
                for s in report.suites
            ],
        }
