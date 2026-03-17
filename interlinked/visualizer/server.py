"""FastAPI server — serves the frontend and provides REST + SSE APIs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from interlinked.analyzer.graph import CodeGraph
from interlinked.commander.query import QueryEngine
from interlinked.commander.llm import LLMAdapter, get_system_prompt
from interlinked.visualizer.layouts import compute_layout
from interlinked.models import ViewState, NodeData, EdgeData

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


def _rebuild_graph(project_path: str, graph: CodeGraph) -> dict:
    """Re-parse a project and rebuild the graph in-place. Returns stats."""
    from interlinked.analyzer.parser import parse_project
    from interlinked.analyzer.dead_code import detect_dead_code
    from interlinked.analyzer.similarity import analyze_similarity

    path = Path(project_path).resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    nodes, edges = parse_project(str(path))
    graph.build_from(nodes, edges)
    dead = detect_dead_code(graph)
    try:
        analyze_similarity(graph)
    except Exception:
        pass
    return {"path": str(path), "nodes": len(nodes), "edges": len(edges), "dead": len(dead)}


def create_app(graph: CodeGraph, initial_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Interlinked", version="0.1.0")
    engine = QueryEngine(graph)

    llm = LLMAdapter(engine)
    app_state = {"project_path": initial_path or ""}
    sse_queues: list[asyncio.Queue] = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Broadcast view changes to all SSE clients ────────────────

    def _layout_from_snap(snap: dict) -> dict:
        """Compute layout from the snapshot's visible nodes/edges only."""
        snap_nodes = [NodeData(**n) for n in snap.get("nodes", [])]
        snap_edges = [EdgeData(**e) for e in snap.get("edges", [])]
        return compute_layout(snap_nodes, snap_edges)

    def _snapshot_with_layout() -> dict:
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return snap

    def on_view_change(snapshot: dict) -> None:
        snapshot["layout"] = _layout_from_snap(snapshot)
        msg = json.dumps({"type": "snapshot", "data": snapshot})
        dead: list[asyncio.Queue] = []
        for q in sse_queues:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            sse_queues.remove(q)

    engine.on_change(on_view_change)

    # ── REST endpoints ───────────────────────────────────────────

    @app.get("/api/project")
    async def get_project() -> JSONResponse:
        return JSONResponse({"path": app_state["project_path"]})

    @app.post("/api/switch_project")
    async def switch_project(body: dict) -> JSONResponse:
        """Switch to a different project. Re-parses and rebuilds the graph."""
        project_path = body.get("path", "")
        if not project_path:
            return JSONResponse({"error": "No path provided"}, status_code=400)
        try:
            result = _rebuild_graph(project_path, graph)
            app_state["project_path"] = result["path"]
            engine.reset_filter()
            from interlinked.models import ViewContext
            engine.state.context = ViewContext(
                what=f"Switched to project: {Path(result['path']).name}",
                why=f"{result['nodes']} symbols, {result['edges']} edges, {result['dead']} dead",
                where=result["path"],
                source="system",
            )
            engine._notify()
            return JSONResponse({"result": f"Switched to {result['path']}", **result})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.get("/api/snapshot")
    async def get_snapshot() -> JSONResponse:
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse(content=snap)

    @app.get("/api/stats")
    async def get_stats() -> JSONResponse:
        return JSONResponse(content=engine.stats())

    @app.get("/api/health")
    async def get_health() -> JSONResponse:
        return JSONResponse(content=json.loads(engine.health()))

    @app.post("/api/command")
    async def run_command(body: dict) -> JSONResponse:
        """Execute a command string against the QueryEngine.

        Body: {"command": "view.zoom('module')"}
        """
        cmd = body.get("command", "")
        if not cmd:
            return JSONResponse({"error": "No command provided"}, status_code=400)

        # Security: only allow access to the engine object
        local_ns: dict[str, Any] = {"view": engine, "graph": graph}
        try:
            # Try as expression first
            try:
                result = eval(cmd, {"__builtins__": {}}, local_ns)
            except SyntaxError:
                exec(cmd, {"__builtins__": {}}, local_ns)
                result = "OK"

            # Serialize result
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            elif isinstance(result, list) and result and hasattr(result[0], "model_dump"):
                result = [r.model_dump() for r in result]

            return JSONResponse({"result": result})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/nl")
    async def natural_language(body: dict) -> JSONResponse:
        """Natural language command."""
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        result = engine.nl(text)
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse({"result": result, "snapshot": snap})

    @app.post("/api/zoom")
    async def set_zoom(body: dict) -> JSONResponse:
        level = body.get("level", "module")
        result = engine.zoom(level)
        return JSONResponse({"result": result})

    @app.post("/api/edge_types")
    async def set_edge_types(body: dict) -> JSONResponse:
        edge_types = body.get("edge_types", [])
        result = engine.set_edge_types(edge_types)
        return JSONResponse({"result": result})

    @app.post("/api/focus")
    async def set_focus(body: dict) -> JSONResponse:
        node_id = body.get("node_id", "")
        depth = body.get("depth", 2)
        result = engine.focus(node_id, depth)
        return JSONResponse({"result": result})

    @app.post("/api/query")
    async def run_query(body: dict) -> JSONResponse:
        expr = body.get("expression", "")
        results = engine.query(expr)
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse({"results": results, "snapshot": snap})

    @app.post("/api/propose")
    async def propose_function(body: dict) -> JSONResponse:
        result = engine.propose_function(
            name=body.get("name", ""),
            module=body.get("module", ""),
            calls=body.get("calls"),
            called_by=body.get("called_by"),
            signature=body.get("signature"),
            color=body.get("color"),
        )
        return JSONResponse({"result": result})

    @app.post("/api/clear_proposed")
    async def clear_proposed() -> JSONResponse:
        result = engine.clear_proposed()
        return JSONResponse({"result": result})

    @app.post("/api/isolate")
    async def isolate_target(body: dict) -> JSONResponse:
        target = body.get("target", "")
        level = body.get("level", "function")
        depth = body.get("depth", 3)
        edge_types = body.get("edge_types")
        result = engine.isolate(target, level=level, depth=depth, edge_types=edge_types)
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse({"result": result, "snapshot": snap})

    @app.post("/api/find_duplicates")
    async def find_duplicates(body: dict) -> JSONResponse:
        threshold = body.get("threshold", 0.6)
        scope = body.get("scope")
        result = engine.find_duplicates(threshold=threshold, scope=scope)
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse({"result": result, "snapshot": snap})

    @app.post("/api/similar_to")
    async def similar_to(body: dict) -> JSONResponse:
        target = body.get("target", "")
        threshold = body.get("threshold", 0.5)
        result = engine.similar_to(target, threshold=threshold)
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse({"result": result, "snapshot": snap})

    @app.post("/api/get_context")
    async def get_context(body: dict) -> JSONResponse:
        target = body.get("target", "")
        result = engine.get_context(target)
        return JSONResponse({"result": result})

    @app.post("/api/reset")
    async def reset_filters() -> JSONResponse:
        result = engine.reset_filter()
        return JSONResponse({"result": result})

    @app.post("/api/trace_variable")
    async def trace_variable(body: dict) -> JSONResponse:
        var_name = body.get("variable", "")
        origin = body.get("origin")
        result = engine.trace_variable(var_name, origin)
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        return JSONResponse({"result": result, "snapshot": snap})

    # ── LLM chat + settings ──────────────────────────────────────

    @app.post("/api/chat")
    async def chat(body: dict) -> JSONResponse:
        """Send a natural language message through the LLM adapter.

        Body: {"message": "show me the analyzer module and everything that connects to it"}
        Returns: {"explanation": str, "commands_run": [...], "results": [...], "snapshot": {...}}
        """
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "No message provided"}, status_code=400)

        result = await llm.chat(message)

        # Set natural language context so the UI knows what it's looking at
        from interlinked.models import ViewContext
        explanation = result.get("explanation", "")
        commands_run = result.get("commands_run", [])
        # Derive "where" from highlighted nodes
        highlighted = engine.state.highlighted_node_ids
        if highlighted:
            # Get the common scope prefix
            parts = [h.rsplit(".", 1)[0] for h in highlighted[:10] if "." in h]
            if parts:
                common = parts[0]
                for p in parts[1:]:
                    while not p.startswith(common) and "." in common:
                        common = common.rsplit(".", 1)[0]
                where = common if common else ", ".join(h.split(".")[-1] for h in highlighted[:5])
            else:
                where = ", ".join(h.split(".")[-1] for h in highlighted[:5])
        else:
            where = ""

        engine.state.context = ViewContext(
            what=explanation,
            why=message,
            where=where,
            source="llm",
        )
        engine._notify()

        # Always return fresh snapshot after commands have executed
        snap = engine.snapshot()
        snap["layout"] = _layout_from_snap(snap)
        result["snapshot"] = snap

        return JSONResponse(result)

    @app.get("/api/system-prompt")
    async def system_prompt() -> JSONResponse:
        """Return the system prompt that teaches an LLM how to drive the view.

        Any external LLM agent can GET this to learn the full API.
        """
        return JSONResponse({"prompt": get_system_prompt(engine)})

    @app.get("/api/settings")
    async def get_settings() -> JSONResponse:
        return JSONResponse({
            "has_api_key": llm.is_configured,
            "model": llm.model,
        })

    @app.post("/api/settings")
    async def update_settings(body: dict) -> JSONResponse:
        if "api_key" in body:
            llm.set_api_key(body["api_key"])
        if "model" in body:
            llm.set_model(body["model"])
        return JSONResponse({
            "has_api_key": llm.is_configured,
            "model": llm.model,
        })

    @app.post("/api/chat/clear")
    async def clear_chat() -> JSONResponse:
        llm.clear_history()
        return JSONResponse({"result": "Chat history cleared."})

    # ── SSE for live updates ─────────────────────────────────────

    @app.get("/api/events")
    async def sse_events(request: Request) -> StreamingResponse:
        """Server-Sent Events stream. Pushes snapshot updates to all connected clients."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        sse_queues.append(q)

        async def event_generator():
            # Send initial snapshot
            snap = _snapshot_with_layout()
            yield f"data: {json.dumps({'type': 'snapshot', 'data': snap})}\n\n"
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        # Keepalive
                        yield ": keepalive\n\n"
            finally:
                if q in sse_queues:
                    sse_queues.remove(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Serve frontend (SPA fallback) ────────────────────────────

    @app.get("/")
    async def serve_index() -> HTMLResponse:
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text())
        # Fallback: serve the embedded single-file frontend
        return HTMLResponse(content=_get_embedded_frontend())

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    return app


def _get_embedded_frontend() -> str:
    """Return the single-file embedded frontend HTML."""
    frontend_file = Path(__file__).parent / "frontend" / "index.html"
    if frontend_file.exists():
        return frontend_file.read_text()
    return "<html><body><h1>Interlinked</h1><p>Frontend not found.</p></body></html>"
