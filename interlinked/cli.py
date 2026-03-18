"""CLI entry point — `interlinked analyze ./project` or `interlinked repl ./project`."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="interlinked",
        description="Interlinked — A Python program topology explorer",
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyze (default: launch web UI) ─────────────────────────
    analyze_p = sub.add_parser("analyze", help="Analyze a project and launch the web UI")
    analyze_p.add_argument("path", type=str, help="Path to the Python project root")
    analyze_p.add_argument("--port", type=int, default=8420, help="Port for the web server")
    analyze_p.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    analyze_p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # ── repl ─────────────────────────────────────────────────────
    repl_p = sub.add_parser("repl", help="Analyze and drop into interactive REPL")
    repl_p.add_argument("path", type=str, help="Path to the Python project root")

    # ── stats ────────────────────────────────────────────────────
    stats_p = sub.add_parser("stats", help="Print project statistics and exit")
    stats_p.add_argument("path", type=str, help="Path to the Python project root")

    # ── mcp ───────────────────────────────────────────────────────
    mcp_p = sub.add_parser("mcp", help="Run as an MCP server (stdio transport) for Windsurf/Claude Desktop")
    mcp_p.add_argument("path", type=str, nargs="?", default=".", help="Path to the Python project root (default: current directory)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    project_path = Path(args.path).resolve()
    if not project_path.exists():
        print(f"Error: path '{project_path}' does not exist.")
        sys.exit(1)

    if args.command == "mcp":
        _run_mcp(project_path)
        return

    # Build the graph
    graph = _build_graph(project_path)

    if args.command == "analyze":
        _run_server(graph, args.host, args.port, not args.no_browser, project_path=str(project_path))
    elif args.command == "repl":
        _run_repl(graph)
    elif args.command == "stats":
        _print_stats(graph)


def _build_graph(project_path: Path):
    """Parse the project and build the CodeGraph."""
    from interlinked.analyzer.parser import parse_project
    from interlinked.analyzer.graph import CodeGraph
    from interlinked.analyzer.dead_code import detect_dead_code

    print(f"Analyzing {project_path} ...")
    nodes, edges = parse_project(project_path)
    print(f"  Found {len(nodes)} symbols, {len(edges)} relationships")

    graph = CodeGraph()
    graph.build_from(nodes, edges)

    dead_ids = detect_dead_code(graph)
    print(f"  Detected {len(dead_ids)} potentially dead symbols")

    # Structural fingerprinting for similarity/duplicate detection
    try:
        from interlinked.analyzer.similarity import analyze_similarity
        analyze_similarity(graph)
        print(f"  Computed structural fingerprints for similarity analysis")
    except Exception as e:
        print(f"  Warning: similarity analysis failed: {e}")

    return graph


def _run_server(graph, host: str, port: int, open_browser: bool, project_path: str = "") -> None:
    """Start the FastAPI web server."""
    import uvicorn
    from interlinked.visualizer.server import create_app

    app = create_app(graph, initial_path=project_path)
    url = f"http://{host}:{port}"
    print(f"\n  Interlinked running at {url}")
    print(f"  Press Ctrl+C to stop\n")

    if open_browser:
        webbrowser.open(url)

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _run_repl(graph) -> None:
    """Start the interactive REPL."""
    from interlinked.commander.repl import InterlinkedREPL
    repl = InterlinkedREPL(graph)
    repl.start()


def _run_mcp(project_path: Path) -> None:
    """Run as an MCP server over stdio."""
    import asyncio
    from interlinked.mcp_server import run_mcp_stdio
    asyncio.run(run_mcp_stdio(str(project_path)))


def _print_stats(graph) -> None:
    """Print statistics and exit."""
    from interlinked.commander.query import QueryEngine
    engine = QueryEngine(graph)
    stats = engine.stats()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║           INTERLINKED — Project Stats            ║")
    print("╚══════════════════════════════════════════════════╝\n")
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label:.<30} {value}")
    print()


if __name__ == "__main__":
    main()
