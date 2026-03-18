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
from interlinked.models import ViewState, NodeData, EdgeData, GraphDelta

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


def _rebuild_graph(project_path: str, graph: CodeGraph, run_similarity: bool = True) -> dict:
    """Re-parse a project and rebuild the graph in-place. Returns stats.

    Similarity analysis is optional and CPU-heavy (~3s for large projects).
    When called from switch_project, it's skipped to avoid blocking — the
    startup event rebuilds embeddings/similarity in the background instead.
    """
    from interlinked.analyzer.parser import parse_project
    from interlinked.analyzer.dead_code import detect_dead_code

    path = Path(project_path).resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    nodes, edges = parse_project(str(path))
    graph.build_from(nodes, edges)
    dead = detect_dead_code(graph)

    if run_similarity:
        try:
            from interlinked.analyzer.similarity import analyze_similarity
            analyze_similarity(graph)
        except Exception:
            pass

    return {"path": str(path), "nodes": len(nodes), "edges": len(edges), "dead": len(dead)}


def _compute_snapshot_delta(
    old_snap: dict | None, new_snap: dict
) -> dict:
    """Compute a delta between two snapshots.

    Returns a delta dict if the diff is worthwhile, or a full snapshot
    flagged with full_snapshot=True if more than 60% of the data changed.
    """
    if old_snap is None:
        return {"type": "snapshot", "data": new_snap}

    old_node_ids = {n["id"] for n in old_snap.get("nodes", [])}
    new_node_ids = {n["id"] for n in new_snap.get("nodes", [])}
    old_nodes_by_id = {n["id"]: n for n in old_snap.get("nodes", [])}
    new_nodes_by_id = {n["id"]: n for n in new_snap.get("nodes", [])}

    removed_ids = old_node_ids - new_node_ids
    added_ids = new_node_ids - old_node_ids
    common_ids = old_node_ids & new_node_ids

    # Check for updated nodes (highlight state, dead status changes, etc.)
    updated_nodes = []
    for nid in common_ids:
        old_n = old_nodes_by_id[nid]
        new_n = new_nodes_by_id[nid]
        if old_n != new_n:
            updated_nodes.append(new_n)

    # Edge diff
    def _edge_key(e: dict) -> tuple:
        return (e["source"], e["target"], e["edge_type"])

    old_edge_keys = {_edge_key(e) for e in old_snap.get("edges", [])}
    new_edge_keys = {_edge_key(e) for e in new_snap.get("edges", [])}
    new_edges_by_key = {_edge_key(e): e for e in new_snap.get("edges", [])}
    old_edges_by_key = {_edge_key(e): e for e in old_snap.get("edges", [])}

    added_edge_keys = new_edge_keys - old_edge_keys
    removed_edge_keys = old_edge_keys - new_edge_keys

    # Layout diff — only send positions that changed
    old_layout = old_snap.get("layout", {})
    new_layout = new_snap.get("layout", {})
    layout_updates = {}
    for nid, pos in new_layout.items():
        old_pos = old_layout.get(nid)
        if old_pos != pos:
            layout_updates[nid] = pos

    total_changes = len(removed_ids) + len(added_ids) + len(updated_nodes) + len(added_edge_keys) + len(removed_edge_keys)
    total_size = max(len(new_node_ids) + len(new_edge_keys), 1)

    # If more than 60% changed, just send a full snapshot
    if total_changes > total_size * 0.6:
        return {"type": "snapshot", "data": new_snap}

    # Build delta
    delta = {
        "type": "delta",
        "data": {
            "added_nodes": [new_nodes_by_id[nid] for nid in added_ids],
            "removed_node_ids": list(removed_ids),
            "updated_nodes": updated_nodes,
            "added_edges": [new_edges_by_key[k] for k in added_edge_keys],
            "removed_edges": [old_edges_by_key[k] for k in removed_edge_keys],
            "view": new_snap.get("view", {}),
            "layout_updates": layout_updates,
        },
    }
    return delta


class _SSEClient:
    """Tracks per-client state for delta computation."""
    __slots__ = ("queue", "last_snapshot")

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.last_snapshot: dict | None = None


def _collect_function_sources(graph: CodeGraph) -> list[dict[str, Any]]:
    """Collect function/method source code for embedding."""
    from interlinked.models import SymbolType
    results = []
    for node in graph.all_nodes(include_proposed=False):
        if node.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            continue
        source = None
        if node.file_path and node.line_start and node.line_end:
            try:
                lines = Path(node.file_path).read_text(encoding="utf-8", errors="replace").splitlines()
                start = max(0, node.line_start - 1)
                end = min(len(lines), node.line_end)
                source = "\n".join(lines[start:end])
            except Exception:
                pass
        if source:
            results.append({"id": node.id, "source": source})
    return results


# Per-project embedding cache — survives switch_project round-trips
_embedding_cache: dict[str, Any] = {}


def _start_embedding_build(graph: CodeGraph, project_path: str, engine: QueryEngine, on_progress=None):
    """Start the embedding index build in the background (if deps available).

    Caches EmbeddingIndex per resolved project path so switching back to a
    previously-loaded project reuses the warm index (model already loaded,
    vectors already in memory) instead of re-scanning the DB.
    """
    from interlinked.analyzer.embeddings import EmbeddingIndex, is_available
    if not is_available():
        return None

    resolved = str(Path(project_path).resolve())
    cached = _embedding_cache.get(resolved)

    if cached is not None:
        if on_progress:
            cached.set_progress_callback(on_progress)
        engine._embedding_index = cached

        if cached.status == "ready":
            # Warm index — just delta-update for any changed functions
            functions = _collect_function_sources(graph)
            import threading
            threading.Thread(
                target=cached.update_functions,
                args=(functions,),
                daemon=True,
                name="embedding-delta",
            ).start()
        # else: build still in progress — reuse it, don't spawn a duplicate

        return cached

    emb_index = EmbeddingIndex(project_path)
    if on_progress:
        emb_index.set_progress_callback(on_progress)
    engine._embedding_index = emb_index
    _embedding_cache[resolved] = emb_index

    functions = _collect_function_sources(graph)
    emb_index.build_async(functions)
    return emb_index


def create_app(graph: CodeGraph, initial_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Interlinked", version="0.1.0")
    engine = QueryEngine(graph)

    llm = LLMAdapter(engine)
    app_state: dict[str, Any] = {"project_path": initial_path or "", "embedding_index": None}
    sse_clients: list[_SSEClient] = []

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
        dead: list[_SSEClient] = []
        for client in sse_clients:
            try:
                delta = _compute_snapshot_delta(client.last_snapshot, snapshot)
                client.queue.put_nowait(json.dumps(delta))
                client.last_snapshot = snapshot
            except Exception:
                dead.append(client)
        for client in dead:
            sse_clients.remove(client)

    engine.on_change(on_view_change)

    # ── Embedding progress callback (pushes via SSE) ─────────────

    def _on_embedding_progress(status_dict: dict) -> None:
        """Push embedding build progress to all SSE clients.

        Called from the background embedding thread — must use
        call_soon_threadsafe so we never touch asyncio.Queue or the
        shared sse_clients list from outside the event loop thread.
        """
        loop = app_state.get("_loop")
        if loop is None or loop.is_closed():
            return
        msg = json.dumps({"type": "embedding_status", "data": status_dict})

        def _push() -> None:
            dead: list[_SSEClient] = []
            for client in sse_clients:
                try:
                    client.queue.put_nowait(msg)
                except Exception:
                    dead.append(client)
            for client in dead:
                sse_clients.remove(client)

        try:
            loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass  # loop already closed

    # ── REST endpoints ───────────────────────────────────────────

    @app.get("/api/project")
    async def get_project() -> JSONResponse:
        return JSONResponse({"path": app_state["project_path"]})

    @app.get("/api/embedding_status")
    async def embedding_status() -> JSONResponse:
        emb = app_state.get("embedding_index")
        if emb is None:
            return JSONResponse({"status": "idle", "progress": 0, "total": 0, "completed": 0, "device": "none", "model": "", "vector_count": 0})
        return JSONResponse(emb.status_dict())

    @app.post("/api/switch_project")
    async def switch_project(body: dict) -> JSONResponse:
        """Switch to a different project. Re-parses and rebuilds the graph."""
        import asyncio, time, logging
        logger = logging.getLogger("uvicorn")
        project_path = body.get("path", "")
        if not project_path:
            return JSONResponse({"error": "No path provided"}, status_code=400)
        try:
            t0 = time.monotonic()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: _rebuild_graph(project_path, graph, run_similarity=False)
            )
            t1 = time.monotonic()
            logger.warning(f"switch_project rebuild: {t1-t0:.3f}s")
            app_state["project_path"] = result["path"]
            engine.reset_filter()
            from interlinked.models import ViewContext
            engine.state.context = ViewContext(
                what=f"Switched to project: {Path(result['path']).name}",
                why=f"{result['nodes']} symbols, {result['edges']} edges, {result['dead']} dead",
                where=result["path"],
                source="system",
            )
            t2 = time.monotonic()
            logger.warning(f"switch_project state: {t2-t1:.3f}s")
            engine._notify()
            t3 = time.monotonic()
            logger.warning(f"switch_project notify: {t3-t2:.3f}s")
            _start_watcher()
            t4 = time.monotonic()
            logger.warning(f"switch_project watcher: {t4-t3:.3f}s  total: {t4-t0:.3f}s")
            return JSONResponse({"result": f"Switched to {result['path']}", **result})
        except Exception as e:
            logger.warning(f"switch_project ERROR: {e}")
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
        kind = body.get("kind")
        result = engine.find_duplicates(threshold=threshold, scope=scope, kind=kind)
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

    @app.post("/api/set_context")
    async def set_context(body: dict) -> JSONResponse:
        """Push an explanation context to the UI from MCP or external agents."""
        from interlinked.models import ViewContext
        engine.state.context = ViewContext(
            what=body.get("what", ""),
            why=body.get("why", ""),
            where=body.get("where", ""),
            source="mcp",
        )
        engine._notify()
        return JSONResponse({"result": "Context updated."})

    @app.post("/api/edges_between")
    async def edges_between(body: dict) -> JSONResponse:
        result = engine.edges_between(
            source_scope=body.get("source_scope", ""),
            target_scope=body.get("target_scope"),
            edge_types=body.get("edge_types"),
        )
        return JSONResponse(content=json.loads(result))

    @app.post("/api/reachable")
    async def reachable(body: dict) -> JSONResponse:
        result = engine.reachable(
            source=body.get("source", ""),
            target=body.get("target", ""),
            edge_types=body.get("edge_types"),
            max_depth=body.get("max_depth", 20),
        )
        return JSONResponse(content=json.loads(result))

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
        """Server-Sent Events stream. Pushes delta updates to all connected clients."""
        client = _SSEClient()
        sse_clients.append(client)

        async def event_generator():
            # Send initial full snapshot
            snap = _snapshot_with_layout()
            client.last_snapshot = snap
            yield f"data: {json.dumps({'type': 'snapshot', 'data': snap})}\n\n"
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(client.queue.get(), timeout=30)
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        # Keepalive
                        yield ": keepalive\n\n"
            finally:
                if client in sse_clients:
                    sse_clients.remove(client)

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

    # ── File watcher for live graph updates ──────────────────────

    _watcher_task: dict[str, Any] = {"task": None}

    async def _watch_project(project_path: str) -> None:
        """Watch for .py file changes and incrementally update the graph."""
        from watchfiles import awatch, Change
        from interlinked.analyzer.parser import parse_file, path_to_module
        from interlinked.analyzer.dead_code import detect_dead_code
        from interlinked.models import ViewContext

        root = Path(project_path).resolve()
        loop = asyncio.get_running_loop()

        def _apply_changes(changes_list):
            """Synchronous graph mutations — runs in executor thread."""
            for change_type, file_path_str in changes_list:
                file_path = Path(file_path_str)
                try:
                    rel_path = file_path.relative_to(root)
                except ValueError:
                    continue

                module_qname = path_to_module(rel_path)

                if change_type == Change.deleted:
                    graph.remove_file(module_qname)
                else:
                    existing_ids = {n.id for n in graph.all_nodes()}
                    type_idx: dict[str, str] = {}
                    for n in graph.all_nodes():
                        if n.symbol_type.value in ("module", "class"):
                            type_idx[n.name] = n.id
                    new_nodes, new_edges = parse_file(
                        file_path, module_qname,
                        existing_node_ids=existing_ids,
                        existing_type_index=type_idx,
                    )
                    graph.update_file(module_qname, new_nodes, new_edges)

            detect_dead_code(graph)

            try:
                from interlinked.analyzer.similarity import analyze_similarity
                analyze_similarity(graph)
            except Exception:
                pass

            emb = app_state.get("embedding_index")
            if emb and emb.status == "ready":
                try:
                    changed_funcs = _collect_function_sources(graph)
                    emb.update_functions(changed_funcs)
                except Exception:
                    pass

        try:
            async for changes in awatch(
                root,
                watch_filter=lambda change, path: path.endswith(".py"),
                debounce=500,
            ):
                await loop.run_in_executor(None, _apply_changes, list(changes))

                engine.state.context = ViewContext(
                    what="Live update: files changed on disk",
                    why=f"{len(changes)} file(s) modified",
                    where=app_state["project_path"],
                    source="system",
                )
                engine._notify()
        except asyncio.CancelledError:
            pass

    def _start_watcher() -> None:
        """Start (or restart) the file watcher background task."""
        if _watcher_task["task"] is not None:
            _watcher_task["task"].cancel()
        project_path = app_state.get("project_path", "")
        if project_path:
            _watcher_task["task"] = asyncio.create_task(
                _watch_project(project_path)
            )

    @app.on_event("startup")
    async def _on_startup() -> None:
        import asyncio
        app_state["_loop"] = asyncio.get_running_loop()
        _start_watcher()
        # Start embedding build in background (if torch/transformers installed)
        project_path = app_state.get("project_path", "")
        if project_path:
            try:
                emb = _start_embedding_build(graph, project_path, engine, on_progress=_on_embedding_progress)
                app_state["embedding_index"] = emb
            except Exception:
                pass

    return app


def _get_embedded_frontend() -> str:
    """Return the legacy single-file D3 frontend as fallback."""
    legacy = Path(__file__).parent / "frontend" / "index.html.d3-legacy"
    if legacy.exists():
        return legacy.read_text()
    return "<html><body><h1>Interlinked</h1><p>Frontend not found. Run <code>npm run build</code> in the frontend directory.</p></body></html>"
