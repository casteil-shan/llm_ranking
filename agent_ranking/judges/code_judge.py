from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent_ranking.core.types import EvalResult


class CodeJudge:
    """代码题判分：提取代码块并运行测试用例。"""

    name = "code"

    def evaluate(self, item: dict[str, Any], response: str, **kwargs: Any) -> EvalResult:
        code = self._extract_code(response, item.get("language", "python"))
        if not code:
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "code"),
                score=0.0,
                passed=False,
                detail={"reason": "no code block found"},
                response=response,
            )

        test_script = item.get("test_script")
        if test_script:
            passed, details = self._run_humaneval(code, test_script, item.get("entry", "solve"))
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "code"),
                score=1.0 if passed else 0.0,
                passed=passed,
                detail={"tests": details, "code": code, "mode": "humaneval"},
                response=response,
            )

        tests = item.get("tests", [])
        if not tests:
            # 无测试时做简单语法检查
            ok = self._syntax_check(code)
            return EvalResult(
                item_id=item["id"],
                suite=item.get("suite", "code"),
                score=1.0 if ok else 0.0,
                passed=ok,
                detail={"syntax_only": True},
                response=response,
            )

        passed, details = self._run_tests(code, tests, item.get("entry", "solve"))
        score = sum(1 for d in details if d["passed"]) / len(details)
        return EvalResult(
            item_id=item["id"],
            suite=item.get("suite", "code"),
            score=score,
            passed=passed,
            detail={"tests": details, "code": code},
            response=response,
        )

    def _extract_code(self, response: str, language: str) -> str | None:
        pattern = rf"```(?:{language})?\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # 无代码块时，若响应像代码则直接使用
        if "def " in response or "class " in response:
            return response.strip()
        return None

    def _syntax_check(self, code: str) -> bool:
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _run_tests(self, code: str, tests: list[dict], entry: str) -> tuple[bool, list[dict]]:
        details = []
        all_passed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "solution.py"
            test_script = Path(tmpdir) / "run_tests.py"
            script.write_text(code, encoding="utf-8")

            test_code = self._build_test_runner(entry, tests)
            test_script.write_text(test_code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [sys.executable, str(test_script)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=tmpdir,
                )
                for line in result.stdout.splitlines():
                    if line.startswith("TEST:"):
                        parts = line.split(":", 2)
                        details.append({
                            "name": parts[1],
                            "passed": parts[2] == "PASS",
                            "output": "",
                        })
                if not details:
                    all_passed = result.returncode == 0
                    details.append({
                        "name": "all",
                        "passed": all_passed,
                        "output": result.stderr[:500],
                    })
                else:
                    all_passed = all(d["passed"] for d in details)
            except subprocess.TimeoutExpired:
                all_passed = False
                details.append({"name": "timeout", "passed": False, "output": "timeout"})

        return all_passed, details

    def _run_humaneval(self, code: str, test_script: str, entry: str) -> tuple[bool, list[dict]]:
        details = []
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = Path(tmpdir) / "run_humaneval.py"
            runner.write_text(
                f"{code}\n\n{test_script}\n\ncheck({entry})\nprint('TEST:0:PASS')\n",
                encoding="utf-8",
            )
            try:
                result = subprocess.run(
                    [sys.executable, str(runner)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=tmpdir,
                )
                passed = result.returncode == 0
                details.append({
                    "name": "humaneval",
                    "passed": passed,
                    "output": (result.stderr or result.stdout)[:500],
                })
            except subprocess.TimeoutExpired:
                details.append({"name": "timeout", "passed": False, "output": "timeout"})
                passed = False
        return passed, details

    def _build_test_runner(self, entry: str, tests: list[dict]) -> str:
        lines = [
            "from solution import *",
            "",
        ]
        for i, test in enumerate(tests):
            args = repr(test.get("input", {}))
            expected = repr(test.get("expected"))
            lines.append("try:")
            lines.append(f"    result = {entry}(**{args})")
            lines.append(f"    assert result == {expected}, f'got {{result}}'")
            lines.append(f"    print('TEST:{i}:PASS')")
            lines.append("except Exception:")
            lines.append(f"    print('TEST:{i}:FAIL')")
        return "\n".join(lines)
