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


def build_graph(project_path: str) -> tuple[CodeGraph, QueryEngine]:
    """Parse a project and build the graph + query engine."""
    nodes, edges = parse_project(project_path)
    graph = CodeGraph()
    graph.build_from(nodes, edges)
    detect_dead_code(graph)

    # Run similarity analysis if available
    try:
        from interlinked.analyzer.similarity import analyze_similarity
        analyze_similarity(graph)
    except Exception:
        pass

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

    def _check_server(port: int = 8420) -> str | None:
        """Check if the web visualizer is running right now. Returns base URL or None."""
        url = f"http://127.0.0.1:{port}"
        try:
            r = httpx.get(f"{url}/api/stats", timeout=1.0)
            if r.status_code == 200:
                return url
        except Exception:
            pass
        return None

    def _ensure_ready() -> tuple[CodeGraph, QueryEngine]:
        if not _state["ready"]:
            print(f"Analyzing {project_path} ...", file=sys.stderr)
            graph, engine = build_graph(project_path)
            _state["graph"] = graph
            _state["engine"] = engine
            _state["ready"] = True
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
                description="Find functions/methods with similar structure, signatures, or logic paths — potential duplicated functionality. Returns groups of similar symbols with similarity scores.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number", "description": "Similarity threshold 0.0-1.0 (default 0.6)", "default": 0.6},
                        "scope": {"type": "string", "description": "Optional: limit search to symbols under this prefix"},
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
                description="Reset all filters, focus, and highlights back to the default full-graph view.",
                inputSchema={"type": "object", "properties": {}, "required": []},
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
            # Try proxying through the running web server first
            server_url = _check_server()
            if server_url:
                result = _dispatch_via_server(name, arguments, server_url)
                if result is not None:
                    return [TextContent(type="text", text=str(result))]

            # Fall back to direct graph mode
            graph, engine = _ensure_ready()
            result = _dispatch_tool(name, arguments, engine, graph, api_key)
            if name == "interlinked_set_api_key":
                api_key = arguments.get("api_key", "")
                os.environ["ANTHROPIC_API_KEY"] = api_key
                result = f"API key {'set' if api_key else 'cleared'}."
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


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
    }

    if name not in _TOOL_TO_ENDPOINT:
        return None

    method, path, body = _TOOL_TO_ENDPOINT[name]
    url = f"{server_url}{path}"

    try:
        if method == "GET":
            r = httpx.get(url, timeout=30.0)
        else:
            r = httpx.post(url, json=body, timeout=30.0)
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
        return engine.find_duplicates(threshold=threshold, scope=scope)

    elif name == "interlinked_similar_to":
        return engine.similar_to(args["target"], threshold=args.get("threshold", 0.5))

    elif name == "interlinked_get_context":
        return engine.get_context(args["target"])

    elif name == "interlinked_command":
        cmd = args["command"]
        local_ns: dict[str, Any] = {"view": engine, "graph": graph}
        try:
            result = eval(cmd, {"__builtins__": {}}, local_ns)
        except SyntaxError:
            exec(cmd, {"__builtins__": {}}, local_ns)
            result = "OK"
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), indent=2)
        return str(result)

    elif name == "interlinked_switch_project":
        from interlinked.visualizer.server import _rebuild_graph
        result = _rebuild_graph(args["path"], graph)
        engine.reset_filter()
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

    elif name == "interlinked_set_api_key":
        return ""  # handled in call_tool

    return f"Unknown tool: {name}"


async def run_mcp_stdio(project_path: str) -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server(project_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
