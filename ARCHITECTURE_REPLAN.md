# Interlinked Performance Architecture Replan

## Problem Statement

The current architecture will die on large codebases (60k+ nodes, 200k+ edges) because:

1. **Full-graph snapshots** — Every `_notify()` serializes the entire filtered graph as JSON and pushes it via SSE to all connected browsers
2. **Full rebuild on any change** — `parse_project()` walks every `.py` file from scratch; no incremental update path
3. **SVG rendering** — D3 SVG cannot handle 10k+ DOM elements without choking the browser
4. **O(n) edge filtering** — `_filter_edges()` iterates all 200k edges on every snapshot to find the visible subset
5. **Server-computed layout** — `compute_layout()` runs `nx.spring_layout` server-side on every snapshot, which is O(n²) per iteration
6. **No delta protocol** — SSE sends full state every time, no concept of "what changed"

## Design Constraints

- **The human MUST be able to view the full graph if they want to** — no hard caps that cripple functionality
- **WebGL is mandatory** — D3 SVG is out; we need GPU-accelerated rendering
- **MCP cannot block on server start** — spawning uvicorn from the MCP thread deadlocks; the MCP must detect/inform, not spawn
- **LLM and human workflows must both be first-class** — the system is MCP-driven but the human UI must be independently powerful
- **NetworkX stays authoritative** — no moving the graph engine to the frontend

## Architecture Changes

### Phase 1: Backend Performance (No Frontend Changes Required)

#### 1A. Incremental File Parsing (`parse_file()` + graph delta)

**Current:** `parse_project(root)` walks every `.py` file, builds all nodes/edges from scratch, then `graph.build_from(nodes, edges)` replaces the entire graph.

**New:** Add `parse_file(file_path, module_qname)` that parses a single file and returns its nodes + raw edges. Add `CodeGraph.update_file(module_qname, new_nodes, new_edges)` that:

1. Removes all nodes/edges where `node.qualified_name.startswith(module_qname)` or `edge.source.startswith(module_qname)`
2. Adds the new nodes/edges
3. Re-runs edge resolution for the new edges only (using the existing name index)
4. Re-runs dead code detection incrementally (only affected nodes)
5. Emits a delta event (not a full snapshot)

**File changes:**
- `analyzer/parser.py` — Extract `_extract_from_module()` as public `parse_file()`, add `parse_project_incremental()` that wraps it
- `analyzer/graph.py` — Add `CodeGraph.update_file()`, `CodeGraph.remove_file()`, add edge indexes
- `analyzer/dead_code.py` — Add `detect_dead_code_incremental(graph, affected_node_ids)` that only re-checks affected subgraph
- `visualizer/server.py` — Add file watcher endpoint or integrate `watchfiles`

**Impact:** Re-parsing a single changed file on a 60k-node project goes from ~10s (full rebuild) to ~50ms (single file parse + graph patch).

#### 1B. Edge Index on CodeGraph

**Current:** `_filter_edges()` does `for e in self.all_edges()` — iterates ALL edges.

**New:** Maintain indexed edge structures:
```python
self._edges_by_source: dict[str, list[EdgeData]] = {}
self._edges_by_target: dict[str, list[EdgeData]] = {}
self._edges_by_type: dict[EdgeType, list[EdgeData]] = {}
```

Populated during `add_edge()`, updated during `update_file()`.

`_filter_edges(view, node_ids)` becomes:
```python
for nid in node_ids:
    for e in self._edges_by_source.get(nid, []):
        if e.target in node_ids and e.edge_type in view.visible_edge_types:
            yield e
```

**Impact:** Edge filtering goes from O(E) to O(V × avg_degree) where V is visible nodes. For a 500-node focused view on a 200k-edge graph: ~2500 lookups instead of 200k.

#### 1C. SSE Delta Protocol

**Current:** `on_view_change(snapshot)` sends the entire snapshot JSON to every SSE client.

**New:** Compute the diff between previous and current state, send only:
```json
{
  "type": "delta",
  "added_nodes": [...],
  "removed_node_ids": [...],
  "updated_nodes": [...],
  "added_edges": [...],
  "removed_edges": [...],
  "view_state": { ... },
  "layout_updates": { "node_id": {"x": ..., "y": ...}, ... }
}
```

Each SSE client tracks its last-sent snapshot hash. On reconnect or first connect, send full snapshot. After that, deltas only.

**Fall back to full snapshot** if delta is larger than 60% of the full snapshot (i.e., a major view change like switching projects).

**File changes:**
- `models.py` — Add `GraphDelta` model
- `analyzer/graph.py` — Add `snapshot_delta(old_view, new_view)` method
- `visualizer/server.py` — Track per-client state, send deltas

#### 1D. Default View State

**Current:** `ViewState` defaults to `zoom_level="function"`, empty `visible_node_ids` (= show everything).

**New:** Default to `zoom_level="module"`. The initial SSE snapshot sends only module-level nodes + inter-module edges. This is typically 20-200 nodes even on a 60k-node project.

The LLM/user then drills down with `isolate()`, `focus()`, `zoom()`. The full graph is always available — you just don't start there.

**File changes:**
- `models.py` — Change `ViewState.zoom_level` default from `"function"` to `"module"`
- `commander/query.py` — Change `reset_filter()` to reset to module zoom, not function zoom

### Phase 2: WebGL Frontend

#### 2A. Renderer Selection

Replace D3 SVG with a WebGL-based graph renderer. Options evaluated:

| Library | Pros | Cons |
|---------|------|------|
| **Sigma.js v2** | Purpose-built for large graphs, WebGL, 100k+ nodes proven, good API | Opinionated layout |
| **Pixi.js + custom** | Fast 2D WebGL, full control | Have to build graph primitives |
| **deck.gl** | Massive scale (millions of points), WebGL2 | Overkill, heavy dependency |
| **regl + custom** | Minimal, fast, full control | Have to build everything |

**Recommendation: Sigma.js v2** — it's specifically designed for large graph visualization in WebGL, handles 100k+ nodes natively, has built-in node/edge programs, camera controls, zoom, hover, selection, and can be driven programmatically. It integrates with graphology (a graph data structure library) which can mirror the backend graph state.

The frontend would be a **React + Sigma.js** application (replacing the current single-file React + D3 app).

#### 2B. Frontend Architecture

```
frontend/
├── src/
│   ├── App.tsx                  # Root layout (header, sidebar, graph, panels)
│   ├── graph/
│   │   ├── GraphCanvas.tsx      # Sigma.js WebGL canvas
│   │   ├── graphStore.ts        # Graphology instance + delta application
│   │   ├── nodePrograms.ts      # Custom WebGL node shapes (hexagon, diamond, circle, etc.)
│   │   ├── edgePrograms.ts      # Custom edge rendering (typed colors, dash patterns)
│   │   └── cameraController.ts  # Zoom-to-fit, focus-on-node, smooth transitions
│   ├── state/
│   │   ├── sseClient.ts         # SSE connection + delta application
│   │   ├── viewState.ts         # Mirror of backend ViewState
│   │   └── selectionState.ts    # What's selected/hovered
│   ├── panels/
│   │   ├── Sidebar.tsx          # Node list, search
│   │   ├── Inspector.tsx        # Node detail panel
│   │   ├── ChatPanel.tsx        # LLM chat
│   │   └── ContextBanner.tsx    # Explanation overlay
│   └── ui/
│       ├── Header.tsx           # Stats, zoom controls, edge type toggles
│       ├── CommandBar.tsx        # Python/NL command input
│       └── StatusBar.tsx         # Connection status, radio player
├── package.json
├── vite.config.ts
└── index.html
```

**Key design decisions:**
- **Graphology** as the client-side graph data structure — it's what Sigma.js expects, and it supports efficient add/remove/update operations for delta application
- **Custom node programs** for shape differentiation (hexagon=module, diamond=class, circle=function, etc.)
- **Layout computed client-side by Sigma.js / graphology-layout-forceatlas2** — ForceAtlas2 in WebGL is fast even at 60k nodes; eliminates server layout computation entirely
- **The frontend graph is a projection** — it holds only what the backend says is visible, plus layout positions. The backend remains authoritative.

#### 2C. Delta Application on Frontend

The `sseClient.ts` receives deltas and applies them to the graphology instance:

```typescript
function applyDelta(graph: Graph, delta: GraphDelta) {
  // Remove nodes (cascades to edges)
  for (const id of delta.removed_node_ids) {
    if (graph.hasNode(id)) graph.dropNode(id);
  }
  // Add new nodes
  for (const node of delta.added_nodes) {
    graph.addNode(node.id, { ...node, x: layout[node.id]?.x, y: layout[node.id]?.y });
  }
  // Update existing nodes (highlight state, dead status, etc.)
  for (const node of delta.updated_nodes) {
    if (graph.hasNode(node.id)) graph.mergeNodeAttributes(node.id, node);
  }
  // Add/remove edges similarly
  // Apply view state (highlights, trace roles, dimming)
}
```

Sigma.js automatically re-renders only the changed portions of the WebGL buffer.

#### 2D. Interaction Model

- **Click node** → select, show in inspector
- **Double-click node** → `isolate(node_id)` via REST, backend pushes delta
- **Right-click node** → context menu: "Show callers", "Show callees", "Impact of", "Trace", "Expand neighborhood"
- **Scroll zoom** → Sigma.js camera zoom (smooth, GPU-accelerated)
- **Drag** → pan the camera
- **Shift+drag** → box select
- **Ctrl+click** → add to selection
- **Search box** → fuzzy node search, camera animates to result
- **Hover** → show tooltip with node info, highlight direct connections

### Phase 3: MCP Integration

#### 3A. `interlinked_ui_status` Tool

```python
Tool(
    name="interlinked_ui_status",
    description="Check if the Interlinked web UI server is running. Returns the URL if running, or instructions to start it.",
    inputSchema={"type": "object", "properties": {}, "required": []},
)
```

Implementation: call `_check_server()`, return either:
- `{"running": true, "url": "http://127.0.0.1:8420"}` 
- `{"running": false, "start_command": "interlinked analyze /path/to/project --port 8420"}`

The LLM can then tell the user to run the command in another terminal, or use the tool results headlessly.

#### 3B. `interlinked_set_context` Tool

```python
Tool(
    name="interlinked_set_context",
    description="Push an explanation message to the UI. The message appears as a context banner over the graph, explaining what the user is looking at and why.",
    inputSchema={
        "type": "object",
        "properties": {
            "what": {"type": "string", "description": "What is being shown"},
            "why": {"type": "string", "description": "Why this view was chosen"},
            "where": {"type": "string", "description": "Which part of the codebase"},
        },
        "required": ["what"],
    },
)
```

When proxied through the server, this sets `engine.state.context` and triggers `_notify()`, which pushes the context update via SSE delta to the frontend.

#### 3C. `interlinked_command` Restrictions

- **Remove from MCP tool list** (keep in REST API for power users)
- OR: Add 5-second execution timeout + restrict to read-only operations (no `exec`, no `graph._g` mutations)

### Phase 4: File Watching + Live Graph Updates

#### 4A. File Watcher Integration

Add `watchfiles` to dependencies. The server watches the project directory:

```python
async def _file_watcher(project_path: str, graph: CodeGraph, engine: QueryEngine):
    async for changes in awatch(project_path, watch_filter=PythonFilter()):
        for change_type, file_path in changes:
            rel_path = Path(file_path).relative_to(project_path)
            module_qname = _path_to_module(rel_path)
            if change_type in (Change.added, Change.modified):
                new_nodes, new_edges = parse_file(file_path, module_qname)
                graph.update_file(module_qname, new_nodes, new_edges)
            elif change_type == Change.deleted:
                graph.remove_file(module_qname)
            # Re-run incremental dead code detection
            detect_dead_code_incremental(graph, affected_ids)
            # Emit delta to all SSE clients
            engine._notify_delta()
```

**Impact:** When a developer saves a file, the graph updates in ~100ms. The frontend receives a delta and smoothly animates the change. No full rebuild.

#### 4B. MCP-Triggered Re-parse

The `interlinked_switch_project` tool currently calls `_rebuild_graph()` which does a full rebuild. After Phase 1A, it should use `parse_project_incremental()` which:
1. Walks all `.py` files
2. Computes file content hashes
3. Only re-parses files whose hash changed
4. Applies deltas per changed file

## Performance Targets

| Operation | Current | Target |
|-----------|---------|--------|
| Initial parse (60k nodes) | ~10s | ~10s (unavoidable first parse) |
| Single file re-parse | ~10s (full rebuild) | ~100ms |
| Snapshot generation (500 visible nodes) | ~500ms (full edge scan) | ~5ms (indexed) |
| SSE message size (500 nodes) | ~2MB (full JSON) | ~50KB (delta) |
| Frontend render (60k nodes visible) | 💀 (SVG DOM death) | ~60fps (WebGL) |
| Frontend render (500 focused nodes) | ~200ms (D3 force) | ~16ms (Sigma.js) |
| Layout computation | ~2s server-side | ~0ms (client-side ForceAtlas2, continuous) |

## Implementation Order

1. **1D. Default view** — 15 min, immediate safety valve
2. **1B. Edge index** — 1-2 hours, biggest backend perf win
3. **1A. Incremental parsing** — 3-4 hours, enables live updates
4. **1C. SSE delta protocol** — 2-3 hours, backend side
5. **3A+3B. MCP tools** — 1 hour
6. **2A-2D. WebGL frontend** — this is the big one, 2-3 days
7. **4A. File watching** — 1-2 hours (builds on 1A + 1C)

## Dependencies to Add

```toml
dependencies = [
    # existing...
    "watchfiles>=0.21",  # async file watching
]
```

Frontend (new `package.json`):
```json
{
  "dependencies": {
    "sigma": "^2.4",
    "graphology": "^0.25",
    "graphology-layout-forceatlas2": "^0.10",
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "vite": "^5",
    "@vitejs/plugin-react": "^4",
    "typescript": "^5"
  }
}
```

## What We Are NOT Changing

- **NetworkX as the authoritative graph engine** — stays
- **The QueryEngine API surface** — stays (isolate, trace, impact, etc.)
- **The MCP tool names and schemas** — stays (backward compatible)
- **The REST API endpoints** — stays (add new ones, don't break old ones)
- **The analysis pipeline** (parser → graph → dead code → similarity) — stays, just made incremental
- **The Blade Runner aesthetic** — stays obviously
