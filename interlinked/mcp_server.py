"""MCP Server for Interlinked — exposes the full view API as MCP tools.

This allows Windsurf, Claude Desktop, or any MCP-compatible client to
drive the Interlinked visualization directly.

Usage:
    interlinked mcp ./my_project                    # stdio transport
    interlinked mcp ./my_project --port 8421        # SSE transport

MCP config example (for .windsurf/mcp_config.json or claude_desktop_config.json):
{
  "mcpServers": {
    "interlinked": {
      "command": "/path/to/.venv/bin/interlinked",
      "args": ["mcp", "/path/to/project"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from interlinked.analyzer.parser import parse_project
from interlinked.analyzer.graph import CodeGraph
from interlinked.analyzer.dead_code import detect_dead_code
from interlinked.commander.query import QueryEngine


def _deferred_background_work(graph: CodeGraph, engine: QueryEngine, project_path: str) -> None:
    """Run similarity + embeddings in a SINGLE background thread (non-blocking).

    Mirrors the REST API pattern: never spawn competing CPU threads that
    fight over the GIL with the main event loop. Everything heavy runs
    sequentially in one thread.
    """
    import threading

    def _work():
        # 1. Similarity fingerprinting (CPU-bound)
        try:
            from interlinked.analyzer.similarity import analyze_similarity
            analyze_similarity(graph)
        except Exception:
            pass

        # 2. Embedding build (spawns its own single thread internally via build_async)
        try:
            from interlinked.visualizer.server import _start_embedding_build
            _start_embedding_build(graph, project_path, engine)
        except Exception:
            pass

    threading.Thread(target=_work, daemon=True, name="mcp-background").start()


def build_graph(project_path: str) -> tuple[CodeGraph, QueryEngine]:
    """Parse a project and build the graph + query engine.

    Returns immediately after parsing + dead code detection.
    Similarity and embeddings are NOT started here — they are deferred
    to avoid GIL contention with the caller.
    """
    nodes, edges = parse_project(project_path)
    graph = CodeGraph()
    graph.build_from(nodes, edges)
    detect_dead_code(graph)

    engine = QueryEngine(graph)
    return graph, engine


def create_mcp_server(project_path: str) -> Server:
    """Create an MCP server with all Interlinked tools.

    Graph building is deferred until the first tool call so the MCP
    handshake responds immediately (large projects can take 10+ seconds
    to parse, which exceeds Windsurf's startup timeout).
    """
    # Lazy state — built on first tool call (only if no web server is running)
    _state: dict[str, Any] = {"graph": None, "engine": None, "ready": False}

    async def _check_server(port: int = 8420) -> str | None:
        """Check if the web visualizer is running right now. Returns base URL or None."""
        url = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{url}/api/stats", timeout=1.0)
                if r.status_code == 200:
                    return url
        except Exception:
            pass
        return None

    async def _ensure_ready() -> tuple[CodeGraph, QueryEngine]:
        if not _state["ready"]:
            import asyncio
            print(f"Analyzing {project_path} ...", file=sys.stderr)
            loop = asyncio.get_running_loop()
            graph, engine = await loop.run_in_executor(None, build_graph, project_path)
            _state["graph"] = graph
            _state["engine"] = engine
            _state["ready"] = True
            # Deferred: similarity + embeddings in single background thread
            # Scheduled AFTER build returns so the tool response goes out first
            _deferred_background_work(graph, engine, project_path)
        return _state["graph"], _state["engine"]

    # Pick up API key from env if available
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    server = Server("interlinked")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="interlinked_stats",
                description="Get summary statistics about the analyzed Python project: node counts, edge counts, dead code count.",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="interlinked_isolate",
                description="Isolate a module, class, or function and show it plus everything that connects to it. This is the primary command for exploring a codebase. The target can be a partial name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Name or partial name of the module/class/function to isolate"},
                        "level": {"type": "string", "enum": ["module", "class", "function", "variable", "all"], "description": "Zoom level", "default": "function"},
                        "depth": {"type": "integer", "description": "How many hops of connections to show", "default": 3},
                        "edge_types": {"type": "array", "items": {"type": "string"}, "description": "Optional: only follow these edge types (calls, imports, inherits, reads, writes)"},
                    },
                    "required": ["target"],
                },
            ),
            Tool(
                name="interlinked_zoom",
                description="Set the zoom level of the visualization: 'module' (high-level), 'class' (mid-level), or 'function' (detailed).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["module", "class", "function", "variable", "all"]},
                    },
                    "required": ["level"],
                },
            ),
            Tool(
                name="interlinked_focus",
                description="Focus the visualization on a specific node and its neighborhood within N hops.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Qualified name of the node to focus on"},
                        "depth": {"type": "integer", "description": "Number of hops to show", "default": 2},
                    },
                    "required": ["node_id"],
                },
            ),
            Tool(
                name="interlinked_query",
                description="Run a structured query against the codebase graph. Supports: 'callers of X', 'callees of X', 'dead functions', 'dead functions in <scope>', 'functions in <scope>', 'classes in <scope>', 'modules in <scope>', 'external calls in <scope>', 'imports of X', 'functions returning <type>', or any search term for fuzzy name matching. The 'in <scope>' suffix filters by qualified name prefix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Query expression"},
                    },
                    "required": ["expression"],
                },
            ),
            Tool(
                name="interlinked_trace_variable",
                description="Trace a variable's path through reads and writes across the codebase.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "Variable name to trace"},
                        "origin": {"type": "string", "description": "Optional: qualified name of the origin scope"},
                    },
                    "required": ["variable"],
                },
            ),
            Tool(
                name="interlinked_propose_function",
                description="Add a hypothetical function to the graph to visualize where it would connect. Shown in green to distinguish from real code.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Function name"},
                        "module": {"type": "string", "description": "Module to place it in"},
                        "calls": {"type": "array", "items": {"type": "string"}, "description": "Functions this would call"},
                        "called_by": {"type": "array", "items": {"type": "string"}, "description": "Functions that would call this"},
                    },
                    "required": ["name", "module"],
                },
            ),
            Tool(
                name="interlinked_find_duplicates",
                description="Find symbols with similar structure, signatures, or logic paths — potential duplicated functionality. Returns groups of similar symbols with similarity scores. Highlights results in the graph UI.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number", "description": "Similarity threshold 0.0-1.0 (default 0.6). Lower = more results.", "default": 0.6},
                        "scope": {"type": "string", "description": "Optional: limit search to symbols under this prefix, e.g. 'analyzer' or 'commander.query'"},
                        "kind": {"type": "string", "enum": ["function", "method", "class"], "description": "Optional: filter by symbol type — 'function', 'method', or 'class'. Omit for all."},
                    },
                    "required": [],
                },
            ),
            Tool(
                name="interlinked_similar_to",
                description="Find functions/classes structurally similar to a given symbol. Useful for finding duplicated or redundant functionality.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Name or partial name of the symbol to compare against"},
                        "threshold": {"type": "number", "description": "Similarity threshold 0.0-1.0", "default": 0.5},
                    },
                    "required": ["target"],
                },
            ),
            Tool(
                name="interlinked_get_context",
                description="Get rich context about a symbol: source code, docstring, signature, comments, connections, and structural fingerprint. Useful for understanding what a function does before comparing.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Name or partial name of the symbol"},
                    },
                    "required": ["target"],
                },
            ),
            Tool(
                name="interlinked_command",
                description="Execute a raw Python command against the view API. For advanced usage. The `view` object is the QueryEngine and `graph` is the CodeGraph.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Python expression to evaluate, e.g. view.isolate('analyzer')"},
                    },
                    "required": ["command"],
                },
            ),
            Tool(
                name="interlinked_switch_project",
                description="Switch to analyzing a different Python project. Re-parses the new project and rebuilds the entire graph.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative path to the Python project root"},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="interlinked_edges_between",
                description="List all edges (calls, imports, etc.) from one module scope to another, or to everything outside the scope. Returns edges grouped by target module. Essential for module isolation checks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_scope": {"type": "string", "description": "Qualified name prefix of source, e.g. 'engine.rules.resolver'"},
                        "target_scope": {"type": "string", "description": "Optional: qualified name prefix of target to filter to, e.g. 'engine.systems'. Omit to see ALL outgoing edges."},
                        "edge_types": {"type": "array", "items": {"type": "string"}, "description": "Optional: edge types to include (calls, imports, inherits, reads, writes). Default: all."},
                    },
                    "required": ["source_scope"],
                },
            ),
            Tool(
                name="interlinked_reachable",
                description="Check if there is any path from source to target following specific edge types (default: calls only). Use for purity contracts and isolation verification. Returns the shortest path if reachable.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Qualified name of the source symbol"},
                        "target": {"type": "string", "description": "Qualified name (or partial) of the target symbol"},
                        "edge_types": {"type": "array", "items": {"type": "string"}, "description": "Edge types to traverse (default: ['calls'])"},
                        "max_depth": {"type": "integer", "description": "Maximum path length (default: 20)", "default": 20},
                    },
                    "required": ["source", "target"],
                },
            ),
            Tool(
                name="interlinked_reset",
                description="Reset all filters, focus, and highlights back to module-level overview.",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="interlinked_ui_status",
                description="Check if the Interlinked web UI server is running. Returns the URL if running, or instructions to start it. Use this before trying to push visual results to the user.",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="interlinked_set_context",
                description="Push an explanation message to the UI. The message appears as a context banner over the graph, explaining what the user is looking at and why. Use this after running queries to explain results visually.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "what": {"type": "string", "description": "What is being shown in the visualization"},
                        "why": {"type": "string", "description": "Why this view was chosen / what question it answers"},
                        "where": {"type": "string", "description": "Which part of the codebase (scope/modules/symbols)"},
                    },
                    "required": ["what"],
                },
            ),
            Tool(
                name="interlinked_start_ui",
                description="Start the Interlinked web visualization server if it is not already running. Returns the URL to open in the browser. The server runs as a background process and persists after this call.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "port": {"type": "integer", "description": "Port to run on (default: 8420)", "default": 8420},
                    },
                    "required": [],
                },
            ),
            Tool(
                name="interlinked_set_api_key",
                description="Set the Anthropic API key for the built-in chat pilot. This is stored in memory only, not written to disk.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "Anthropic API key (sk-ant-...)"},
                    },
                    "required": ["api_key"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        nonlocal api_key

        try:
            # Handle ui_status first — it doesn't need graph or server proxy
            if name == "interlinked_ui_status":
                server_url = await _check_server()
                if server_url:
                    result = json.dumps({
                        "running": True,
                        "url": server_url,
                        "message": f"Interlinked UI is running at {server_url}. All visualization commands will be displayed there.",
                    })
                else:
                    result = json.dumps({
                        "running": False,
                        "start_tool": "interlinked_start_ui",
                        "message": "Interlinked UI is not running. Call interlinked_start_ui to launch it, or the user can run: interlinked analyze " + str(project_path),
                    })
                return [TextContent(type="text", text=result)]

            # Handle start_ui — needs to spawn the server process
            if name == "interlinked_start_ui":
                import asyncio
                port = arguments.get("port", 8420)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, _start_ui, project_path, port)
                return [TextContent(type="text", text=result)]

            # Try proxying through the running web server first
            server_url = await _check_server()
            if server_url:
                result = await _async_dispatch_via_server(name, arguments, server_url)
                if result is not None:
                    return [TextContent(type="text", text=str(result))]

            # switch_project: skip _ensure_ready (would double-parse), rebuild directly
            import asyncio
            loop = asyncio.get_running_loop()

            if name == "interlinked_switch_project":
                from interlinked.visualizer.server import _rebuild_graph

                def _do_switch():
                    g = _state.get("graph")
                    if g is None:
                        g = CodeGraph()
                    return _rebuild_graph(arguments["path"], g, run_similarity=False), g

                result_dict, graph = await loop.run_in_executor(None, _do_switch)
                engine = _state.get("engine")
                if engine is None:
                    engine = QueryEngine(graph)
                engine.reset_filter()
                _state["graph"] = graph
                _state["engine"] = engine
                _state["ready"] = True
                # Deferred: single background thread AFTER response sent
                _deferred_background_work(graph, engine, arguments["path"])
                return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]

            # All other tools: ensure graph is built first
            graph, engine = await _ensure_ready()
            result = await loop.run_in_executor(
                None, _dispatch_tool, name, arguments, engine, graph, api_key
            )
            if name == "interlinked_set_api_key":
                api_key = arguments.get("api_key", "")
                os.environ["ANTHROPIC_API_KEY"] = api_key
                result = f"API key {'set' if api_key else 'cleared'}."
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


def _start_ui(project_path: str, port: int = 8420) -> str:
    """Spawn the Interlinked web server as a background process.

    Returns JSON with the URL and status. If already running, returns immediately.
    """
    import subprocess
    import time

    url = f"http://127.0.0.1:{port}"

    # Already running?
    try:
        r = httpx.get(f"{url}/api/stats", timeout=1.0)
        if r.status_code == 200:
            return json.dumps({
                "status": "already_running",
                "url": url,
                "message": f"Interlinked UI is already running at {url}",
            })
    except Exception:
        pass

    # Spawn as a detached background process
    cmd = [
        sys.executable, "-c",
        "import uvicorn; "
        "from interlinked.visualizer.server import create_app, _rebuild_graph; "
        "from interlinked.analyzer.graph import CodeGraph; "
        "graph = CodeGraph(); "
        f"_rebuild_graph({project_path!r}, graph); "
        f"app = create_app(graph, initial_path={project_path!r}); "
        f"uvicorn.run(app, host='0.0.0.0', port={port}, log_level='warning')",
    ]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to start server: {e}"})

    # Wait for it to come up (up to 15s for large projects)
    for _ in range(30):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{url}/api/stats", timeout=1.0)
            if r.status_code == 200:
                return json.dumps({
                    "status": "started",
                    "url": url,
                    "message": f"Interlinked UI started at {url}",
                })
        except Exception:
            pass

    return json.dumps({
        "status": "timeout",
        "url": url,
        "message": f"Server process spawned but not responding yet at {url}. It may still be parsing the project — try interlinked_ui_status in a few seconds.",
    })


def _dispatch_via_server(name: str, args: dict[str, Any], server_url: str) -> str | None:
    """Proxy an MCP tool call through the running web server's REST API.

    Returns the result string, or None if the tool can't be proxied
    (falls back to direct mode).
    """
    # Map MCP tool names to REST endpoints and request bodies
    _TOOL_TO_ENDPOINT: dict[str, tuple[str, str, dict]] = {
        # (method, path, body)
        "interlinked_stats":       ("GET",  "/api/stats", {}),
        "interlinked_isolate":     ("POST", "/api/isolate", {
            "target": args.get("target", ""),
            "level": args.get("level", "function"),
            "depth": args.get("depth", 3),
            "edge_types": args.get("edge_types"),
        }),
        "interlinked_zoom":        ("POST", "/api/zoom", {"level": args.get("level", "module")}),
        "interlinked_focus":       ("POST", "/api/focus", {
            "node_id": args.get("node_id", ""),
            "depth": args.get("depth", 2),
        }),
        "interlinked_query":       ("POST", "/api/query", {"expression": args.get("expression", "")}),
        "interlinked_trace_variable": ("POST", "/api/trace_variable", {
            "variable": args.get("variable", ""),
            "origin": args.get("origin"),
        }),
        "interlinked_propose_function": ("POST", "/api/propose", {
            "name": args.get("name", ""),
            "module": args.get("module", ""),
            "calls": args.get("calls"),
            "called_by": args.get("called_by"),
        }),
        "interlinked_find_duplicates": ("POST", "/api/find_duplicates", {
            "threshold": args.get("threshold", 0.6),
            "scope": args.get("scope"),
            "kind": args.get("kind"),
        }),
        "interlinked_similar_to":  ("POST", "/api/similar_to", {
            "target": args.get("target", ""),
            "threshold": args.get("threshold", 0.5),
        }),
        "interlinked_get_context": ("POST", "/api/get_context", {"target": args.get("target", "")}),
        "interlinked_command":     ("POST", "/api/command", {"command": args.get("command", "")}),
        "interlinked_switch_project": ("POST", "/api/switch_project", {"path": args.get("path", "")}),
        "interlinked_edges_between": ("POST", "/api/edges_between", {
            "source_scope": args.get("source_scope", ""),
            "target_scope": args.get("target_scope"),
            "edge_types": args.get("edge_types"),
        }),
        "interlinked_reachable":   ("POST", "/api/reachable", {
            "source": args.get("source", ""),
            "target": args.get("target", ""),
            "edge_types": args.get("edge_types"),
            "max_depth": args.get("max_depth", 20),
        }),
        "interlinked_reset":       ("POST", "/api/reset", {}),
        "interlinked_set_context": ("POST", "/api/set_context", {
            "what": args.get("what", ""),
            "why": args.get("why", ""),
            "where": args.get("where", ""),
        }),
    }

    if name not in _TOOL_TO_ENDPOINT:
        return None

    method, path, body = _TOOL_TO_ENDPOINT[name]
    url = f"{server_url}{path}"

    try:
        if method == "GET":
            r = httpx.get(url, timeout=30.0)
        else:
            timeout = 120.0 if "switch_project" in path else 30.0
            r = httpx.post(url, json=body, timeout=timeout)
        data = r.json()

        # Extract the most useful part of the response for the LLM
        if "result" in data:
            result = data["result"]
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        if "results" in data:
            results = data["results"]
            if len(results) > 20:
                return f"Found {len(results)} results. Showing first 20:\n" + json.dumps(results[:20], indent=2)
            return json.dumps(results, indent=2)
        if "error" in data:
            return f"Error: {data['error']}"
        return json.dumps(data, indent=2)
    except Exception as e:
        print(f"Server proxy failed for {name}: {e}", file=sys.stderr)
        return None


async def _async_dispatch_via_server(name: str, args: dict[str, Any], server_url: str) -> str | None:
    """Async proxy — uses httpx.AsyncClient so the MCP event loop isn't blocked."""
    # Reuse the same endpoint mapping
    _TOOL_TO_ENDPOINT: dict[str, tuple[str, str, dict]] = {
        "interlinked_stats":       ("GET",  "/api/stats", {}),
        "interlinked_isolate":     ("POST", "/api/isolate", {
            "target": args.get("target", ""),
            "level": args.get("level", "function"),
            "depth": args.get("depth", 3),
            "edge_types": args.get("edge_types"),
        }),
        "interlinked_zoom":        ("POST", "/api/zoom", {"level": args.get("level", "module")}),
        "interlinked_focus":       ("POST", "/api/focus", {
            "node_id": args.get("node_id", ""),
            "depth": args.get("depth", 2),
        }),
        "interlinked_query":       ("POST", "/api/query", {"expression": args.get("expression", "")}),
        "interlinked_trace_variable": ("POST", "/api/trace_variable", {
            "variable": args.get("variable", ""),
            "origin": args.get("origin"),
        }),
        "interlinked_propose_function": ("POST", "/api/propose", {
            "name": args.get("name", ""),
            "module": args.get("module", ""),
            "calls": args.get("calls"),
            "called_by": args.get("called_by"),
        }),
        "interlinked_find_duplicates": ("POST", "/api/find_duplicates", {
            "threshold": args.get("threshold", 0.6),
            "scope": args.get("scope"),
            "kind": args.get("kind"),
        }),
        "interlinked_similar_to":  ("POST", "/api/similar_to", {
            "target": args.get("target", ""),
            "threshold": args.get("threshold", 0.5),
        }),
        "interlinked_get_context": ("POST", "/api/get_context", {"target": args.get("target", "")}),
        "interlinked_command":     ("POST", "/api/command", {"command": args.get("command", "")}),
        "interlinked_switch_project": ("POST", "/api/switch_project", {"path": args.get("path", "")}),
        "interlinked_edges_between": ("POST", "/api/edges_between", {
            "source_scope": args.get("source_scope", ""),
            "target_scope": args.get("target_scope"),
            "edge_types": args.get("edge_types"),
        }),
        "interlinked_reachable":   ("POST", "/api/reachable", {
            "source": args.get("source", ""),
            "target": args.get("target", ""),
            "edge_types": args.get("edge_types"),
            "max_depth": args.get("max_depth", 20),
        }),
        "interlinked_reset":       ("POST", "/api/reset", {}),
        "interlinked_set_context": ("POST", "/api/set_context", {
            "what": args.get("what", ""),
            "why": args.get("why", ""),
            "where": args.get("where", ""),
        }),
    }

    if name not in _TOOL_TO_ENDPOINT:
        return None

    method, path, body = _TOOL_TO_ENDPOINT[name]
    url = f"{server_url}{path}"

    try:
        async with httpx.AsyncClient() as client:
            timeout = 120.0 if "switch_project" in path else 30.0
            if method == "GET":
                r = await client.get(url, timeout=timeout)
            else:
                r = await client.post(url, json=body, timeout=timeout)
        data = r.json()

        if "result" in data:
            result = data["result"]
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        if "results" in data:
            results = data["results"]
            if len(results) > 20:
                return f"Found {len(results)} results. Showing first 20:\n" + json.dumps(results[:20], indent=2)
            return json.dumps(results, indent=2)
        if "error" in data:
            return f"Error: {data['error']}"
        return json.dumps(data, indent=2)
    except Exception as e:
        print(f"Async server proxy failed for {name}: {e}", file=sys.stderr)
        return None


def _dispatch_tool(
    name: str, args: dict[str, Any],
    engine: QueryEngine, graph: CodeGraph, api_key: str,
) -> str:
    """Route a tool call to the appropriate engine method."""

    if name == "interlinked_stats":
        stats = engine.stats()
        return json.dumps(stats, indent=2)

    elif name == "interlinked_isolate":
        return engine.isolate(
            target=args["target"],
            level=args.get("level", "function"),
            depth=args.get("depth", 3),
            edge_types=args.get("edge_types"),
        )

    elif name == "interlinked_zoom":
        return engine.zoom(args["level"])

    elif name == "interlinked_focus":
        return engine.focus(args["node_id"], depth=args.get("depth", 2))

    elif name == "interlinked_query":
        results = engine.query(args["expression"])
        if len(results) > 20:
            summary = f"Found {len(results)} results. Showing first 20:\n"
            return summary + json.dumps(results[:20], indent=2)
        return json.dumps(results, indent=2)

    elif name == "interlinked_trace_variable":
        return engine.trace_variable(args["variable"], args.get("origin"))

    elif name == "interlinked_propose_function":
        return engine.propose_function(
            name=args["name"], module=args["module"],
            calls=args.get("calls"), called_by=args.get("called_by"),
        )

    elif name == "interlinked_find_duplicates":
        threshold = args.get("threshold", 0.6)
        scope = args.get("scope")
        kind = args.get("kind")
        return engine.find_duplicates(threshold=threshold, scope=scope, kind=kind)

    elif name == "interlinked_similar_to":
        return engine.similar_to(args["target"], threshold=args.get("threshold", 0.5))

    elif name == "interlinked_get_context":
        return engine.get_context(args["target"])

    elif name == "interlinked_command":
        import signal

        cmd = args["command"]
        local_ns: dict[str, Any] = {"view": engine, "graph": graph}

        def _timeout_handler(signum, frame):
            raise TimeoutError("Command execution timed out (5s limit)")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)
        try:
            try:
                result = eval(cmd, {"__builtins__": {}}, local_ns)
            except SyntaxError:
                exec(cmd, {"__builtins__": {}}, local_ns)
                result = "OK"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), indent=2)
        return str(result)

    elif name == "interlinked_switch_project":
        from interlinked.visualizer.server import _rebuild_graph
        result = _rebuild_graph(args["path"], graph, run_similarity=False)
        engine.reset_filter()
        # Deferred: similarity + embeddings in a single background thread
        # (matches REST API: never run competing CPU threads)
        _deferred_background_work(graph, engine, args["path"])
        return json.dumps(result, indent=2)

    elif name == "interlinked_edges_between":
        return engine.edges_between(
            source_scope=args["source_scope"],
            target_scope=args.get("target_scope"),
            edge_types=args.get("edge_types"),
        )

    elif name == "interlinked_reachable":
        return engine.reachable(
            source=args["source"],
            target=args["target"],
            edge_types=args.get("edge_types"),
            max_depth=args.get("max_depth", 20),
        )

    elif name == "interlinked_reset":
        return engine.reset_filter()

    elif name == "interlinked_set_context":
        from interlinked.models import ViewContext
        engine.state.context = ViewContext(
            what=args.get("what", ""),
            why=args.get("why", ""),
            where=args.get("where", ""),
            source="mcp",
        )
        engine._notify()
        return "Context updated on UI."

    elif name == "interlinked_set_api_key":
        return ""  # handled in call_tool

    return f"Unknown tool: {name}"


async def run_mcp_stdio(project_path: str) -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server(project_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
