"""Fixture: nested closures inside async functions, mirroring real MCP patterns.

Tests that the parser correctly resolves calls from:
- An async function to closures defined in its enclosing scope
- An async function to other async functions via await
- Closures calling functions from a different scope
- await loop.run_in_executor(None, fn) — fn is the real call target
"""

import asyncio
from typing import Any


class GraphBuilder:
    def build(self, path: str) -> dict:
        return {"path": path, "nodes": 10}

    def reset(self) -> None:
        pass


class QueryRunner:
    def __init__(self, graph: GraphBuilder) -> None:
        self.graph = graph

    def stats(self) -> dict:
        return {"total": 10}

    def reset_filter(self) -> None:
        pass


def make_server() -> dict:
    """Factory that defines closures and an async handler — mirrors create_mcp_server."""
    _state: dict[str, Any] = {"ready": False, "graph": None, "engine": None}

    def _build(path: str) -> tuple:
        """Sync closure — called via run_in_executor from async handler."""
        g = GraphBuilder()
        result = g.build(path)
        e = QueryRunner(g)
        return g, e, result

    async def _check() -> bool:
        """Async closure — called via await from handler."""
        return _state["ready"]

    async def handler(action: str, args: dict) -> str:
        """Async handler that calls closures — mirrors call_tool."""
        ready = await _check()

        if action == "switch":
            loop = asyncio.get_running_loop()

            def _do_switch():
                """Nested closure inside async handler — called via executor."""
                g = _state.get("graph")
                if g is None:
                    g = GraphBuilder()
                return g.build(args["path"]), g

            result_dict, graph = await loop.run_in_executor(None, _do_switch)
            engine = QueryRunner(graph)
            engine.reset_filter()
            _state["graph"] = graph
            _state["engine"] = engine
            _state["ready"] = True
            return str(result_dict)

        if not ready:
            loop = asyncio.get_running_loop()
            graph, engine, result = await loop.run_in_executor(None, _build, ".")
            _state["graph"] = graph
            _state["engine"] = engine
            _state["ready"] = True

        engine = _state["engine"]
        return str(engine.stats())

    return {"handler": handler, "check": _check, "build": _build}
