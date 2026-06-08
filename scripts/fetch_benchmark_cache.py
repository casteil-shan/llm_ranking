#!/usr/bin/env python3
"""下载公开基准数据到 scripts/cache/（仅需运行一次，支持离线构建）。"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "scripts" / "cache"

URLS = {
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "HumanEval.jsonl.gz": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    "mbpp.json": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
}


def download(name: str, url: str) -> None:
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {name} (exists)")
        return
    print(f"download {name} ...")
    data = urllib.request.urlopen(url, timeout=120).read()
    if name.endswith(".gz"):
        text = gzip.open(io.BytesIO(data), "rt").read()
        (CACHE / name.replace(".gz", "")).write_text(text, encoding="utf-8")
        print(f"  -> {name.replace('.gz', '')} ({len(text)} bytes)")
    else:
        dest.write_bytes(data)
        print(f"  -> {name} ({len(data)} bytes)")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        try:
            download(name, url)
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
    print("Done. Run: python scripts/build_benchmarks.py")


if __name__ == "__main__":
    main()
