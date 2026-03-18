"""Fixture: async/await patterns, context managers, generators.

Tests:
- async def correctly identified as functions
- await calls resolve to the correct target
- async context managers (__aenter__/__aexit__)
- async generators (async for)
- sync generators (yield)
- Chained awaits (await obj.method())
"""

import asyncio
from typing import AsyncIterator


class AsyncDB:
    """Async context manager."""

    async def __aenter__(self) -> "AsyncDB":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def query(self, sql: str) -> list[dict]:
        return [{"id": 1}]

    async def stream_rows(self, sql: str) -> AsyncIterator[dict]:
        """Async generator."""
        results = await self.query(sql)
        for row in results:
            yield row


async def fetch_data(db: AsyncDB) -> list[dict]:
    """Uses async context manager and chained awaits."""
    async with db:
        results = await db.query("SELECT *")
        return results


async def stream_all(db: AsyncDB) -> list[dict]:
    """Consumes async generator."""
    rows = []
    async for row in db.stream_rows("SELECT *"):
        rows.append(row)
    return rows


async def parallel_fetch() -> list:
    """asyncio.gather with multiple coroutines."""
    db = AsyncDB()
    results = await asyncio.gather(
        fetch_data(db),
        stream_all(db),
    )
    return results


def sync_generator(n: int):
    """Sync generator — yield, not return."""
    for i in range(n):
        yield i * 2


def consume_generator() -> list:
    """Consumes sync generator."""
    return list(sync_generator(10))


# ── Async type resolution stress tests ──────────────────────────────

class Connection:
    """Returned by AsyncDB.__aenter__ — tests async-with as-variable typing."""

    def execute(self, sql: str) -> list:
        return []


class AsyncPool:
    """Async context manager that returns a DIFFERENT type from __aenter__."""

    async def __aenter__(self) -> Connection:
        return Connection()

    async def __aexit__(self, *args) -> None:
        pass


async def get_db() -> AsyncDB:
    """Async function returning a project type."""
    return AsyncDB()


async def await_return_type() -> list[dict]:
    """await get_db() should propagate AsyncDB type through the await."""
    db = await get_db()        # db should be AsyncDB
    return await db.query("x") # should resolve AsyncDB.query


async def async_with_as_var() -> list:
    """async with pool as conn — conn should be Connection (from __aenter__)."""
    pool = AsyncPool()
    async with pool as conn:
        return conn.execute("SELECT 1")  # should resolve Connection.execute


async def async_for_typing(db: AsyncDB) -> list[dict]:
    """async for row in db.stream_rows() — row is dict (external, won't resolve).
    But db.stream_rows should resolve to AsyncDB.stream_rows."""
    rows = []
    async for row in db.stream_rows("SELECT *"):
        rows.append(row)
    return rows
