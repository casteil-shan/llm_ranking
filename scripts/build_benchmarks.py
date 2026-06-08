#!/usr/bin/env python3
"""从本地缓存 + 种子数据构建 benchmark-light / benchmark-full 题库。

数据来源:
  - GSM8K (scripts/cache/gsm8k_test.jsonl)
  - HumanEval (scripts/cache/HumanEval.jsonl)
  - MBPP (scripts/cache/mbpp.json)
  - MMLU 英文种子 (scripts/benchmark_seeds/mmlu_en.jsonl)
  - C-Eval 中文种子 (scripts/benchmark_seeds/ceval_zh.jsonl)
  - 自研 logic / dialogue / agent 题

首次使用请先运行: python scripts/fetch_benchmark_cache.py
然后: python scripts/build_benchmarks.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_benchmarks_static as static  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "scripts" / "cache"
SEEDS = ROOT / "scripts" / "benchmark_seeds"
DATASETS = ROOT / "datasets"

RNG = random.Random(42)

PROFILE_CONFIG = {
    "benchmark-light": {
        "gsm8k": 20,
        "mmlu": 30,
        "ceval": 20,
        "humaneval": 20,
        "mbpp": 10,
        "logic": 12,
        "dialogue": 8,
        "agent": 8,
    },
    "benchmark-full": {
        "gsm8k": 500,
        "mmlu": 100,
        "ceval": 60,
        "humaneval": 164,
        "mbpp": 80,
        "logic": 30,
        "dialogue": 20,
        "agent": 20,
    },
}


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_gsm8k_number(answer: str) -> float | int:
    text = answer.replace(",", "")
    if "####" in text:
        text = text.split("####")[-1].strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        raise ValueError("no number")
    val = nums[-1]
    return int(val) if "." not in val else float(val)


def load_gsm8k(limit: int) -> list[dict]:
    path = CACHE / "gsm8k_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/fetch_benchmark_cache.py")
    rows = load_jsonl(path)
    RNG.shuffle(rows)
    items = []
    for row in rows:
        if len(items) >= limit:
            break
        try:
            value = extract_gsm8k_number(row["answer"])
        except ValueError:
            continue
        items.append({
            "id": f"gsm8k_{len(items):04d}",
            "suite": "reasoning",
            "source": "gsm8k",
            "lang": "en",
            "prompt": row["question"],
            "checks": [{"type": "numeric", "value": value, "tolerance": max(0.01, abs(value) * 0.001)}],
        })
    return items


def _mmlu_row_to_item(row: dict, idx: int) -> dict:
    choices = row["choices"]
    letters = "ABCD"
    options = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return {
        "id": f"mmlu_{idx:04d}",
        "suite": "accuracy",
        "source": "mmlu",
        "lang": "en",
        "subject": row.get("subject", ""),
        "prompt": f"{row['question']}\n{options}\nAnswer with the option letter only.",
        "checks": [{"type": "choice", "value": row["answer"]}],
    }


def _ceval_row_to_item(row: dict, idx: int) -> dict:
    letters = "ABCD"
    options = "\n".join(f"{letters[i]}. {row[letters[i]]}" for i in range(4))
    return {
        "id": f"ceval_{idx:04d}",
        "suite": "accuracy",
        "source": "ceval",
        "lang": "zh",
        "subject": row.get("subject", ""),
        "prompt": f"{row['question']}\n{options}\n请只回答选项字母。",
        "checks": [{"type": "choice", "value": row["answer"].upper()}],
    }


def load_mmlu_seed(limit: int) -> list[dict]:
    rows = load_jsonl(SEEDS / "mmlu_en.jsonl") + static.ACCURACY_EN_EXTRA
    RNG.shuffle(rows)
    items = []
    for row in rows[:limit]:
        items.append(_mmlu_row_to_item(row, len(items)))
    return items


def load_ceval_seed(limit: int) -> list[dict]:
    rows = load_jsonl(SEEDS / "ceval_zh.jsonl") + static.ACCURACY_ZH_EXTRA
    RNG.shuffle(rows)
    items = []
    for row in rows[:limit]:
        items.append(_ceval_row_to_item(row, len(items)))
    return items


def load_humaneval(limit: int) -> list[dict]:
    path = CACHE / "HumanEval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/fetch_benchmark_cache.py")
    rows = load_jsonl(path)
    items = []
    for row in rows[:limit]:
        items.append({
            "id": f"humaneval_{row['task_id'].replace('/', '_')}",
            "suite": "code",
            "source": "humaneval",
            "lang": "en",
            "prompt": (
                "Complete the following Python function. Return code in a ```python block.\n\n"
                f"{row['prompt']}"
            ),
            "entry": row["entry_point"],
            "test_script": row["test"],
            "language": "python",
        })
    return items


def load_mbpp(limit: int) -> list[dict]:
    path = CACHE / "mbpp.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/fetch_benchmark_cache.py")
    rows = json.loads(path.read_text(encoding="utf-8"))
    RNG.shuffle(rows)
    items = []
    for row in rows:
        if len(items) >= limit:
            break
        code = row.get("code", "")
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if not match:
            continue
        entry = match.group(1)
        test_list = row.get("test_list", [])
        if not test_list:
            continue
        lines = ["def check(candidate):"]
        for t in test_list:
            lines.append(f"    {t}")
        items.append({
            "id": f"mbpp_{row.get('task_id', len(items)):04d}",
            "suite": "code",
            "source": "mbpp",
            "lang": "en",
            "prompt": (
                f"Write a Python function `{entry}` to solve the task. "
                f"Return code in a ```python block.\n\n{row['prompt']}"
            ),
            "entry": entry,
            "test_script": "\n".join(lines),
            "language": "python",
        })
    return items


def static_items(pool: list[dict], limit: int, suite: str) -> list[dict]:
    return [{"suite": suite, **raw} for raw in pool[:limit]]


def build_profile(name: str, cfg: dict) -> dict[str, int]:
    print(f"\n=== Building {name} ===")
    counts = {}

    reasoning = load_gsm8k(cfg["gsm8k"])
    write_jsonl(DATASETS / name / "reasoning.jsonl", reasoning)
    counts["reasoning"] = len(reasoning)

    accuracy = load_mmlu_seed(cfg["mmlu"]) + load_ceval_seed(cfg["ceval"])
    write_jsonl(DATASETS / name / "accuracy.jsonl", accuracy)
    counts["accuracy"] = len(accuracy)

    code = load_humaneval(cfg["humaneval"]) + load_mbpp(cfg["mbpp"])
    write_jsonl(DATASETS / name / "code.jsonl", code)
    counts["code"] = len(code)

    logic = static_items(static.STATIC_LOGIC, cfg["logic"], "logic")
    write_jsonl(DATASETS / name / "logic.jsonl", logic)
    counts["logic"] = len(logic)

    dialogue = static_items(static.STATIC_DIALOGUE, cfg["dialogue"], "dialogue")
    write_jsonl(DATASETS / name / "dialogue.jsonl", dialogue)
    counts["dialogue"] = len(dialogue)

    agent = static_items(static.STATIC_AGENT, cfg["agent"], "agent")
    write_jsonl(DATASETS / name / "agent.jsonl", agent)
    counts["agent"] = len(agent)

    meta = {
        "profile": name,
        "sources": {
            "reasoning": "GSM8K",
            "accuracy": "MMLU (en seed) + C-Eval (zh seed)",
            "code": "HumanEval + MBPP",
            "logic": "curated",
            "dialogue": "curated",
            "agent": "curated",
        },
        "counts": counts,
        "total": sum(counts.values()),
    }
    (DATASETS / name / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return counts


def main():
    for profile, cfg in PROFILE_CONFIG.items():
        build_profile(profile, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
