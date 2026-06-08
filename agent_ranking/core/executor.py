from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

from agent_ranking.adapters.base import ModelClient

T = TypeVar("T")


class TieredExecutor:
    """按 tier 限制并发与超时的请求执行器。"""

    def __init__(self, client: ModelClient, tier_config: dict):
        self.client = client
        self.max_concurrency = tier_config.get("max_concurrency", 2)
        self.timeout = tier_config.get("request_timeout_sec", 300)
        self.retry_count = tier_config.get("retry_count", 1)
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run(self, coro_factory: Callable[[], Awaitable[T]]) -> T:
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                async with self._get_semaphore():
                    return await asyncio.wait_for(coro_factory(), timeout=self.timeout)
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    await asyncio.sleep(min(2 ** attempt, 10))
        raise last_error  # type: ignore[misc]

    async def run_sync_in_executor(self, fn: Callable[[], T]) -> T:
        loop = asyncio.get_event_loop()
        start = time.monotonic()
        result = await loop.run_in_executor(None, fn)
        _ = time.monotonic() - start
        return result
