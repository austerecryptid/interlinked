"""QueryEngine — the DSL that LLMs and humans use to manipulate the view."""

from __future__ import annotations

import json
import re
from typing import Any

from interlinked.analyzer.graph import CodeGraph
from interlinked.models import (
    EdgeType, SymbolType, ViewState, ViewContext, ColorScheme, NodeData, EdgeData,
)


class QueryEngine:
    """Provides a high-level API for querying and manipulating the code graph.

    This is the object exposed as `view` in the REPL.
    An LLM emits Python calls against this API to control the visualization.
    """

    def __init__(self, graph: CodeGraph) -> None:
        self.graph = graph
        self.state = ViewState()
        self._change_callbacks: list[Any] = []

    def on_change(self, callback: Any) -> None:
        """Register a callback fired whenever the view state changes."""
        self._change_callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._change_callbacks:
            cb(self.snapshot())

    # ── Zoom / Focus ─────────────────────────────────────────────────

    def zoom(self, level: str) -> str:
        """Set zoom level: 'module', 'class', 'function', 'variable', or 'all'."""
        valid = ("module", "class", "function", "variable", "all")
        if level not in valid:
            return f"Invalid zoom level: {level}. Use one of: {', '.join(valid)}."
        self.state.zoom_level = level
        self._notify()
        return f"Zoom set to {level} level."

    def focus(self, node_id: str, depth: int = 2) -> str:
        """Focus the view on a specific node and its neighborhood."""
        node = self.graph.get_node(node_id)
        if not node:
            # Try fuzzy match
            matches = [
                n for n in self.graph.all_nodes()
                if node_id.lower() in n.qualified_name.lower()
            ]
            if len(matches) == 1:
                node = matches[0]
                node_id = node.id
            elif matches:
                names = [m.qualified_name for m in matches[:10]]
                return f"Ambiguous. Did you mean one of: {', '.join(names)}?"
            else:
                return f"Node '{node_id}' not found."

        self.state.focus_node = node_id
        self.state.focus_depth = depth
        self._notify()
        return f"Focused on {node_id} (depth={depth})."

    def unfocus(self) -> str:
        """Remove focus — show the full graph at current zoom level."""
        self.state.focus_node = None
        self._notify()
        return "Focus cleared."

    # ── Isolate (the core LLM use-case) ──────────────────────────────

    def isolate(
        self,
        target: str,
        level: str = "function",
        depth: int = 3,
        edge_types: list[str] | None = None,
    ) -> str:
        """Isolate a module/class/service and show it + everything that connects to it.

        This is the primary command an LLM uses to walk a human through the codebase.

        Args:
            target: Name or partial name of the module/class/function to isolate.
            level: Zoom level — 'module', 'class', or 'function'.
            depth: How many hops of connections to show.
            edge_types: Which relationship types to follow (default: all).

        Examples:
            view.isolate('analyzer.parser')                    # show parser and connections
            view.isolate('CodeGraph', level='function')        # show all methods + what calls them
            view.isolate('visualizer', level='module', depth=1)# show module + direct dependencies
            view.isolate('QueryEngine', level='function', depth=2, edge_types=['calls'])
        """
        # Resolve the target name
        node = self.graph.get_node(target)
        if not node:
            matches = [
                n for n in self.graph.all_nodes()
                if target.lower() in n.qualified_name.lower()
            ]
            if not matches:
                return f"No symbol matching '{target}' found."
            if len(matches) == 1:
                node = matches[0]
            else:
                # Pick the broadest match (shortest qualified name = higher-level symbol)
                node = min(matches, key=lambda n: len(n.qualified_name))

        # Resolve edge types
        et_filter = None
        if edge_types:
            et_filter = []
            for et_str in edge_types:
                try:
                    et_filter.append(EdgeType(et_str))
                except ValueError:
                    pass

        # Get all descendants (things contained within the target)
        all_nodes = self.graph.all_nodes()
        internal_ids = {
            n.id for n in all_nodes
            if n.qualified_name.startswith(node.qualified_name)
        }

        # Get the subgraph around the target — includes external connections
        sub_nodes, sub_edges = self.graph.subgraph_around(
            node.id, depth=depth, edge_types=et_filter
        )
        sub_ids = {n.id for n in sub_nodes}

        # Combine: everything internal + everything in the neighborhood
        visible_ids = internal_ids | sub_ids

        # Set view state
        self.state.zoom_level = level
        self.state.focus_node = None  # We use visible_node_ids instead
        self.state.visible_node_ids = list(visible_ids)
        self.state.highlighted_node_ids = list(internal_ids)  # Highlight the target
        if et_filter:
            self.state.visible_edge_types = et_filter
        else:
            self.state.visible_edge_types = list(EdgeType)

        self._notify()

        n_internal = len(internal_ids)
        n_external = len(visible_ids - internal_ids)
        return (
            f"Isolated '{node.qualified_name}' at {level} level: "
            f"{n_internal} internal symbols, {n_external} external connections (depth={depth})."
        )

    def show(self, target: str, level: str = "function", depth: int = 2) -> str:
        """Shorthand for isolate — show me this thing and what connects to it."""
        return self.isolate(target, level=level, depth=depth)

    # ── Filtering ────────────────────────────────────────────────────

    def filter(
        self,
        edge_type: str | None = None,
        symbol_type: str | None = None,
        min_depth: int | None = None,
        max_depth: int | None = None,
        name_pattern: str | None = None,
    ) -> str:
        """Filter the visible graph by various criteria."""
        if edge_type:
            try:
                et = EdgeType(edge_type)
                self.state.visible_edge_types = [et]
            except ValueError:
                return f"Unknown edge type: {edge_type}. Options: {[e.value for e in EdgeType]}"

        if name_pattern:
            regex = re.compile(name_pattern, re.IGNORECASE)
            matching = [
                n for n in self.graph.all_nodes()
                if regex.search(n.qualified_name) or regex.search(n.name)
            ]
            self.state.visible_node_ids = [n.id for n in matching]
        else:
            self.state.visible_node_ids = []

        self._notify()
        parts = []
        if edge_type:
            parts.append(f"edge_type={edge_type}")
        if name_pattern:
            parts.append(f"name_pattern={name_pattern}")
        return f"Filter applied: {', '.join(parts) if parts else 'reset'}."

    def set_edge_types(self, edge_types: list[str]) -> str:
        """Set which edge types are visible. Pass list of edge type strings."""
        valid = []
        for et in edge_types:
            try:
                valid.append(EdgeType(et))
            except ValueError:
                pass
        self.state.visible_edge_types = valid if valid else list(EdgeType)
        self._notify()
        return f"Visible edge types: {[e.value for e in self.state.visible_edge_types]}"

    def reset_filter(self) -> str:
        """Reset all filters to module-level overview."""
        self.state.zoom_level = "module"
        self.state.visible_node_ids = []
        self.state.visible_edge_types = list(EdgeType)
        self.state.focus_node = None
        self.state.filter_expression = None
        self.state.highlighted_node_ids = []
        self.state.trace_node_roles = {}
        self.state.trace_edge_roles = {}
        self._notify()
        return "Reset to module-level overview."

    # ── Queries ──────────────────────────────────────────────────────

    def query(self, expression: str) -> list[dict]:
        """Run a structured query against the graph.

        Examples:
            view.query("functions returning List[str]")
            view.query("dead functions")
            view.query("uncalled")
            view.query("callers of MyClass.my_method")
            view.query("callees of main")
            view.query("imports of analyzer.parser")
            view.query("parameters of parse_project")
            view.query("returns of build_from")
            view.query("external calls in analyzer.parser")
        """
        expr = expression.lower().strip()
        results: list[NodeData] = []

        if expr.startswith("callers of"):
            target = expression.split("callers of", 1)[1].strip()
            results = self.graph.callers_of(target)

        elif expr.startswith("callees of"):
            target = expression.split("callees of", 1)[1].strip()
            results = self.graph.callees_of(target)

        elif expr.startswith("parameters of") or expr.startswith("params of"):
            target = expression.split("of", 1)[1].strip()
            node = self._resolve_node(target)
            if node:
                G = self.graph._g
                if node.id in G:
                    param_ids = [
                        v for _, v, d in G.out_edges(node.id, data=True)
                        if d.get("edge_type") == "contains"
                        and self.graph.get_node(v)
                        and self.graph.get_node(v).symbol_type == SymbolType.PARAMETER
                    ]
                    results = [self.graph.get_node(p) for p in param_ids if self.graph.get_node(p)]

        elif expr.startswith("returns of"):
            target = expression.split("returns of", 1)[1].strip()
            node = self._resolve_node(target)
            if node:
                edges = self.graph.edges_from(node.id, EdgeType.RETURNS)
                ret_nodes = [self.graph.get_node(e.target) for e in edges if self.graph.get_node(e.target)]
                results = ret_nodes

        elif expr.startswith("external calls"):
            # "external calls in X" or "external calls of X"
            rest = expr.split("external calls", 1)[1].strip()
            rest = rest.lstrip("in ").lstrip("of ").strip()
            node_ids = {n.id for n in self.graph.all_nodes()}
            G = self.graph._g
            if rest:
                # Scope to a module/class/function
                scope_node = self._resolve_node(rest)
                scope_prefix = scope_node.qualified_name if scope_node else rest
                ext_calls = []
                for e in self.graph.all_edges():
                    if e.edge_type != EdgeType.CALLS:
                        continue
                    if e.target in node_ids:
                        continue
                    if e.source.startswith(scope_prefix):
                        ext_calls.append({"source": e.source, "target": e.target, "line": e.line})
                self.state.highlighted_node_ids = list({c["source"] for c in ext_calls})
                self._notify()
                return ext_calls
            else:
                # All external calls
                ext_calls = [
                    {"source": e.source, "target": e.target, "line": e.line}
                    for e in self.graph.all_edges()
                    if e.edge_type == EdgeType.CALLS and e.target not in node_ids
                ]
                return ext_calls

        elif expr.startswith("functions returning"):
            type_hint = expression.split("returning", 1)[1].strip()
            results = self.graph.functions_returning(type_hint)

        elif "dead" in expr or "uncalled" in expr:
            # Support scoped dead queries: "dead functions in engine.rules"
            scope = None
            scope_match = re.search(r'(?:dead|uncalled)\s+\w+\s+in\s+(\S+)', expression, re.IGNORECASE)
            if scope_match:
                scope = scope_match.group(1)
            # Support type filter: "dead functions" vs "dead classes" vs just "dead"
            type_filter: set[SymbolType] | None = None
            if "function" in expr or "method" in expr:
                type_filter = {SymbolType.FUNCTION, SymbolType.METHOD}
            elif "class" in expr:
                type_filter = {SymbolType.CLASS}
            elif "module" in expr:
                type_filter = {SymbolType.MODULE}
            elif "variable" in expr:
                type_filter = {SymbolType.VARIABLE}
            results = [
                n for n in self.graph.all_nodes()
                if n.is_dead
                and (type_filter is None or n.symbol_type in type_filter)
                and (scope is None or n.qualified_name.startswith(scope))
            ]

        elif expr.startswith("imports of"):
            target = expression.split("imports of", 1)[1].strip()
            edges = self.graph.edges_from(target, EdgeType.IMPORTS)
            return [e.model_dump() for e in edges]

        elif expr.startswith("modules"):
            scope = self._parse_scope(expression, "modules")
            results = self.graph.nodes_by_type(SymbolType.MODULE)
            if scope:
                results = [n for n in results if n.qualified_name.startswith(scope)]

        elif expr.startswith("classes"):
            scope = self._parse_scope(expression, "classes")
            results = self.graph.nodes_by_type(SymbolType.CLASS)
            if scope:
                results = [n for n in results if n.qualified_name.startswith(scope)]

        elif expr.startswith("functions") or expr.startswith("methods"):
            scope = self._parse_scope(expression, "functions") or self._parse_scope(expression, "methods")
            results = (
                self.graph.nodes_by_type(SymbolType.FUNCTION)
                + self.graph.nodes_by_type(SymbolType.METHOD)
            )
            if scope:
                results = [n for n in results if n.qualified_name.startswith(scope)]

        elif expr.startswith("parameters") or expr.startswith("variables"):
            scope = self._parse_scope(expression, "parameters") or self._parse_scope(expression, "variables")
            results = [
                n for n in self.graph.all_nodes()
                if n.symbol_type == (SymbolType.PARAMETER if "param" in expr else SymbolType.VARIABLE)
            ]
            if scope:
                results = [n for n in results if n.qualified_name.startswith(scope)]

        else:
            # Fuzzy name search
            results = [
                n for n in self.graph.all_nodes()
                if expr in n.qualified_name.lower() or expr in n.name.lower()
            ]

        # Highlight results in the view
        self.state.highlighted_node_ids = [n.id for n in results]

        # Build WHAT / WHERE / WHY context banner
        if results:
            types = {}
            for r in results:
                t = r.symbol_type.value if hasattr(r.symbol_type, 'value') else str(r.symbol_type)
                types[t] = types.get(t, 0) + 1
            type_str = ", ".join(f"{v} {k}{'s' if v != 1 else ''}" for k, v in sorted(types.items(), key=lambda x: -x[1]))
            sample = [r.name for r in results[:5]]
            sample_str = ", ".join(sample) + ("..." if len(results) > 5 else "")

            # Determine modules involved
            modules = sorted({r.qualified_name.split(".")[0] for r in results})
            where_str = ", ".join(modules[:6]) + ("..." if len(modules) > 6 else "")

            self.state.context = ViewContext(
                what=f"Query '{expression}' — {len(results)} results ({type_str})",
                why=f"Matching symbols: {sample_str}",
                where=f"Across: {where_str}",
                source="query",
            )
        else:
            self.state.context = ViewContext(
                what=f"Query '{expression}' — no results",
                why="No symbols matched this query",
                where="",
                source="query",
            )

        self._notify()

        return [r.model_dump() for r in results]

    def trace_variable(self, var_name: str, origin: str | None = None) -> str:
        """Trace a variable's path through reads/writes and highlight it.

        Nodes are color-coded by role: origin (first write), mutator (subsequent writes),
        passthrough (reads then passes along), destination (terminal reader).
        Edges are colored by type: write (mutation) vs read (non-mutating).
        """
        nodes, edges, node_roles, edge_roles = self.graph.trace_variable(var_name, origin)
        self.state.highlighted_node_ids = [n.id for n in nodes]
        self.state.trace_node_roles = node_roles
        self.state.trace_edge_roles = edge_roles

        role_counts = {}
        for r in node_roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1
        role_str = ", ".join(f"{v} {k}" for k, v in role_counts.items())

        origins = [nid.split('.')[-1] for nid, r in node_roles.items() if r == 'origin']
        dests = [nid.split('.')[-1] for nid, r in node_roles.items() if r == 'destination']
        self.state.context = ViewContext(
            what=f"Data flow trace of '{var_name}' — {len(nodes)} symbols, {role_str}",
            why=f"Tracking how '{var_name}' is written, read, and passed through the codebase",
            where=f"Origins: {', '.join(origins[:5])}. Destinations: {', '.join(dests[:5])}",
            source="trace",
        )
        self._notify()
        return f"Traced '{var_name}': {len(nodes)} nodes, {len(edges)} edges. Roles: {role_str}."

    # ── Tracing (Phase 1c) ─────────────────────────────────────────

    def trace_function(self, name: str) -> str:
        """Trace a function's full call chain — everything that calls it and everything it calls.

        The target function is green (origin), callers are yellow (passthrough),
        callees are white (destination). Edges show direction of calls.
        """
        node = self._resolve_node(name)
        if not node:
            return f"No symbol matching '{name}' found."

        nodes, edges, node_roles, edge_roles = self.graph.trace_function(node.id)
        self.state.highlighted_node_ids = [n.id for n in nodes]
        self.state.trace_node_roles = node_roles
        self.state.trace_edge_roles = edge_roles

        role_counts = {}
        for r in node_roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1
        role_str = ", ".join(f"{v} {k}" for k, v in role_counts.items())

        callers = [nid.split('.')[-1] for nid, r in node_roles.items() if r == 'passthrough']
        callees = [nid.split('.')[-1] for nid, r in node_roles.items() if r == 'destination']
        self.state.context = ViewContext(
            what=f"Call chain of '{node.name}' — {len(nodes)} symbols in chain",
            why=f"Full upstream callers and downstream callees of '{node.qualified_name}'",
            where=f"Callers: {', '.join(callers[:5])}. Callees: {', '.join(callees[:5])}",
            source="trace",
        )
        self._notify()
        return f"Traced '{node.qualified_name}': {len(nodes)} nodes, {len(edges)} edges. {role_str}."

    def trace_call_chain(self, source: str, target: str, max_depth: int = 8) -> str:
        """Find all call paths from source to target function.

        Shows every route through the call graph from A to B.
        Source is green (origin), target is white (destination),
        intermediates are yellow (passthrough).
        """
        src_node = self._resolve_node(source)
        tgt_node = self._resolve_node(target)
        if not src_node:
            return f"Source '{source}' not found."
        if not tgt_node:
            return f"Target '{target}' not found."

        nodes, edges, node_roles, edge_roles = self.graph.trace_call_chain(
            src_node.id, tgt_node.id, max_depth=max_depth
        )
        if not nodes:
            return f"No call path found between '{src_node.qualified_name}' and '{tgt_node.qualified_name}'."

        self.state.highlighted_node_ids = [n.id for n in nodes]
        self.state.trace_node_roles = node_roles
        self.state.trace_edge_roles = edge_roles
        self._notify()
        return (
            f"Found call chain: {len(nodes)} nodes, {len(edges)} edges "
            f"between '{src_node.qualified_name}' and '{tgt_node.qualified_name}'."
        )

    # ── Impact & Dependency (Phase 2) ────────────────────────────────

    def impact_of(self, name: str) -> str:
        """Show everything downstream — if I change this, what's affected?

        Highlights the blast radius of changing a symbol.
        """
        node = self._resolve_node(name)
        if not node:
            return f"No symbol matching '{name}' found."

        affected = self.graph.impact_of(node.id)
        all_ids = [node.id] + list(affected)
        self.state.highlighted_node_ids = all_ids
        self.state.trace_node_roles = {node.id: "origin"}
        self.state.trace_node_roles.update({nid: "destination" for nid in affected})
        self.state.trace_edge_roles = {}
        affected_names = [nid.split('.')[-1] for nid in list(affected)[:5]]
        self.state.context = ViewContext(
            what=f"Blast radius of '{node.name}' — {len(affected)} downstream symbols affected",
            why=f"Everything that would break if '{node.qualified_name}' changes",
            where=f"Affected: {', '.join(affected_names)}{'...' if len(affected) > 5 else ''}",
            source="trace",
        )
        self._notify()
        return f"Impact of '{node.qualified_name}': {len(affected)} downstream symbols affected."

    def depends_on(self, name: str) -> str:
        """Show everything upstream — what does this symbol depend on?

        Highlights all dependencies feeding into this symbol.
        """
        node = self._resolve_node(name)
        if not node:
            return f"No symbol matching '{name}' found."

        deps = self.graph.feeds_into(node.id)
        all_ids = [node.id] + list(deps)
        self.state.highlighted_node_ids = all_ids
        self.state.trace_node_roles = {node.id: "destination"}
        self.state.trace_node_roles.update({nid: "origin" for nid in deps})
        self.state.trace_edge_roles = {}
        dep_names = [nid.split('.')[-1] for nid in list(deps)[:5]]
        self.state.context = ViewContext(
            what=f"Dependencies of '{node.name}' — {len(deps)} upstream symbols",
            why=f"Everything '{node.qualified_name}' depends on",
            where=f"Depends on: {', '.join(dep_names)}{'...' if len(deps) > 5 else ''}",
            source="trace",
        )
        self._notify()
        return f"'{node.qualified_name}' depends on {len(deps)} upstream symbols."

    def path_between(self, source: str, target: str) -> str:
        """Show the shortest dependency chain between two symbols."""
        src_node = self._resolve_node(source)
        tgt_node = self._resolve_node(target)
        if not src_node:
            return f"Source '{source}' not found."
        if not tgt_node:
            return f"Target '{target}' not found."

        path = self.graph.path_between(src_node.id, tgt_node.id)
        if not path:
            return f"No path between '{src_node.qualified_name}' and '{tgt_node.qualified_name}'."

        self.state.highlighted_node_ids = list(path)
        self.state.trace_node_roles = {
            path[0]: "origin",
            path[-1]: "destination",
        }
        for nid in path[1:-1]:
            self.state.trace_node_roles[nid] = "passthrough"
        self.state.trace_edge_roles = {}
        self._notify()

        short_path = " → ".join(n.split(".")[-1] for n in path)
        return f"Shortest path ({len(path)} hops): {short_path}"

    def all_paths(self, source: str, target: str, max_depth: int = 8) -> str:
        """Show every route between two symbols."""
        src_node = self._resolve_node(source)
        tgt_node = self._resolve_node(target)
        if not src_node:
            return f"Source '{source}' not found."
        if not tgt_node:
            return f"Target '{target}' not found."

        paths = self.graph.all_paths_between(src_node.id, tgt_node.id, max_depth=max_depth)
        if not paths:
            return f"No paths between '{src_node.qualified_name}' and '{tgt_node.qualified_name}'."

        all_nodes: set[str] = set()
        for p in paths:
            all_nodes.update(p)

        self.state.highlighted_node_ids = list(all_nodes)
        self.state.trace_node_roles = {src_node.id: "origin", tgt_node.id: "destination"}
        for nid in all_nodes - {src_node.id, tgt_node.id}:
            self.state.trace_node_roles[nid] = "passthrough"
        self.state.trace_edge_roles = {}
        self._notify()
        return f"Found {len(paths)} paths between '{src_node.qualified_name}' and '{tgt_node.qualified_name}', {len(all_nodes)} nodes involved."

    # ── Architecture Health (Phase 3) ────────────────────────────────

    def find_cycles(self) -> str:
        """Find and highlight circular dependencies."""
        cycles = self.graph.find_cycles()
        if not cycles:
            return "No circular dependencies found."

        all_ids: set[str] = set()
        for cycle in cycles:
            all_ids.update(cycle)

        self.state.highlighted_node_ids = list(all_ids)
        self.state.trace_node_roles = {nid: "mutator" for nid in all_ids}
        self.state.trace_edge_roles = {}
        self._notify()
        return f"Found {len(cycles)} circular dependencies involving {len(all_ids)} symbols."

    def critical_nodes(self, top_n: int = 20) -> str:
        """Highlight the most important symbols by PageRank.

        These are the nodes that, if removed, would most disrupt the codebase.
        """
        ranked = self.graph.critical_nodes(top_n=top_n)
        if not ranked:
            return "Could not compute critical nodes."

        ids = [nid for nid, _ in ranked]
        self.state.highlighted_node_ids = ids
        self.state.trace_node_roles = {}
        self.state.trace_edge_roles = {}
        self._notify()

        top5 = ", ".join(f"{nid.split('.')[-1]} ({score:.4f})" for nid, score in ranked[:5])
        return f"Top {len(ranked)} critical nodes. Top 5: {top5}"

    def bottlenecks(self, top_n: int = 20) -> str:
        """Highlight bottleneck nodes — everything flows through these.

        High betweenness centrality = coupling hotspot.
        """
        ranked = self.graph.bottlenecks(top_n=top_n)
        if not ranked:
            return "Could not compute bottlenecks."

        ids = [nid for nid, score in ranked if score > 0]
        self.state.highlighted_node_ids = ids
        self.state.trace_node_roles = {nid: "mutator" for nid in ids}
        self.state.trace_edge_roles = {}
        self._notify()

        top5 = ", ".join(f"{nid.split('.')[-1]} ({score:.4f})" for nid, score in ranked[:5])
        return f"Top {len(ids)} bottlenecks. Top 5: {top5}"

    def coupling(self, module_a: str, module_b: str) -> str:
        """Show coupling between two modules — all cross-module edges."""
        result = self.graph.coupling_between(module_a, module_b)
        if result["edge_count"] == 0:
            return f"No direct coupling between '{module_a}' and '{module_b}'."

        all_ids: set[str] = set()
        for e in result["edges"]:
            all_ids.add(e["source"])
            all_ids.add(e["target"])

        self.state.highlighted_node_ids = list(all_ids)
        self.state.trace_node_roles = {}
        self.state.trace_edge_roles = {}
        self._notify()
        return f"Coupling between '{module_a}' and '{module_b}': {result['edge_count']} cross-module edges."

    def health(self) -> str:
        """Full architecture health report."""
        cycles = self.graph.find_cycles()
        critical = self.graph.critical_nodes(top_n=5)
        bn = self.graph.bottlenecks(top_n=5)
        coupled = self.graph.most_coupled(top_n=5)
        clusters = self.graph.find_clusters()
        circ = self.graph.circular_clusters()
        dead = self.graph.truly_dead()
        total = self.graph.node_count

        # Resolution quality and external dependency summary
        all_nodes = self.graph.all_nodes(include_proposed=False)
        all_edges = self.graph.all_edges(include_proposed=False)
        node_ids = {n.id for n in all_nodes}
        rw_edges = [e for e in all_edges if e.edge_type in (EdgeType.READS, EdgeType.WRITES)]
        rw_resolved = sum(1 for e in rw_edges if e.target in node_ids)
        ext_calls = [e for e in all_edges if e.edge_type == EdgeType.CALLS and e.target not in node_ids]
        ext_targets = {}
        for e in ext_calls:
            root = e.target.split(".")[0]
            ext_targets[root] = ext_targets.get(root, 0) + 1

        report = {
            "total_nodes": total,
            "total_edges": self.graph.edge_count,
            "data_flow_resolution": f"{rw_resolved}/{len(rw_edges)} ({round(rw_resolved / max(len(rw_edges), 1) * 100)}%)",
            "external_calls": len(ext_calls),
            "external_dependencies": dict(sorted(ext_targets.items(), key=lambda x: -x[1])[:10]),
            "circular_dependencies": len(cycles),
            "circular_clusters": len(circ),
            "disconnected_clusters": len(clusters),
            "truly_dead_symbols": len(dead),
            "dead_pct": round(len(dead) / max(total, 1) * 100, 1),
            "top_critical": [{"name": n, "score": round(s, 4)} for n, s in critical],
            "top_bottlenecks": [{"name": n, "score": round(s, 4)} for n, s in bn],
            "top_coupled": [{"name": n, "degree": d} for n, d in coupled],
        }
        return json.dumps(report, indent=2)

    # ── Helper ───────────────────────────────────────────────────────

    def _parse_scope(self, expression: str, keyword: str) -> str | None:
        """Extract scope from 'keyword in scope' pattern, e.g. 'functions in engine.rules'."""
        m = re.search(rf'{keyword}\s+in\s+(\S+)', expression, re.IGNORECASE)
        return m.group(1) if m else None

    def _resolve_node(self, name: str) -> NodeData | None:
        """Resolve a name to a node, with fuzzy matching."""
        node = self.graph.get_node(name)
        if node:
            return node
        matches = [
            n for n in self.graph.all_nodes()
            if name.lower() in n.qualified_name.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return min(matches, key=lambda n: len(n.qualified_name))
        return None

    # ── Proposals (hypotheticals) ────────────────────────────────────

    def propose_function(
        self,
        name: str,
        module: str,
        calls: list[str] | None = None,
        called_by: list[str] | None = None,
        signature: str | None = None,
        color: str | None = None,
    ) -> str:
        """Add a hypothetical function to see where it would connect."""
        node = self.graph.propose_function(
            name=name, module=module,
            calls=calls, called_by=called_by,
            signature=signature,
        )
        if color:
            self.state.colors.proposed = color
        self._notify()
        return f"Proposed function '{node.qualified_name}' added to graph."

    def clear_proposed(self) -> str:
        """Remove all proposed/hypothetical elements."""
        self.graph.clear_proposed()
        self._notify()
        return "All proposed elements cleared."

    # ── Similarity / Duplication ──────────────────────────────────────

    def find_duplicates(self, threshold: float = 0.6, scope: str | None = None, kind: str | None = None) -> str:
        """Find groups of structurally similar symbols — potential duplicated functionality.

        Args:
            threshold: Similarity threshold 0.0-1.0 (default 0.6). Lower = more results.
            scope: Optional prefix to limit search, e.g. "analyzer" or "commander.query".
            kind: Filter by type — "function", "method", "class", or None for all.

        Returns JSON with groups of similar symbols, sorted by similarity score.
        """
        from interlinked.analyzer.similarity import find_duplicate_groups
        emb_index = getattr(self, '_embedding_index', None)
        groups = find_duplicate_groups(self.graph, threshold=threshold, scope=scope, kind=kind, embedding_index=emb_index)

        # Switch zoom level to match the kind being searched
        _KIND_TO_ZOOM = {
            "function": "function", "functions": "function",
            "method": "function",   "methods": "function",
            "class": "class",       "classes": "class",
        }
        if kind:
            self.state.zoom_level = _KIND_TO_ZOOM.get(kind.lower(), "function")
        else:
            self.state.zoom_level = "function"

        # Highlight all members of duplicate groups in the view
        all_ids = []
        for group in groups:
            for member in group["members"]:
                all_ids.append(member["id"])
        self.state.highlighted_node_ids = all_ids

        _KIND_PLURAL = {"class": "classes", "function": "functions", "method": "methods"}
        kind_label = _KIND_PLURAL.get(kind, "functions/methods") if kind else "functions/methods"
        scope_label = f" in {scope}" if scope else ""
        self.state.context = ViewContext(
            what=f"Duplicate {kind_label}{scope_label} (threshold {threshold})",
            why=f"Found {len(groups)} groups, {len(all_ids)} symbols with similar structure",
            where="Similarity analysis (minhash + TED + WL kernel" + (" + embeddings" if emb_index and emb_index.is_ready else "") + ")",
            source="similarity",
        )
        self._notify()

        emb_note = ""
        if emb_index and emb_index.status == "building":
            emb_note = f" (Embedding index building: {int(emb_index.progress * 100)}% — results will improve when complete)"
        elif emb_index and emb_index.status == "idle":
            emb_note = " (Install interlinked[similarity] for Type 4 semantic detection)"

        if not groups:
            return json.dumps({"message": f"No duplicate groups found at threshold {threshold}.{emb_note}", "groups": []})
        return json.dumps({"message": f"Found {len(groups)} groups of similar symbols.{emb_note}", "groups": groups}, indent=2)

    def similar_to(self, target: str, threshold: float = 0.5, top: int = 10) -> str:
        """Find functions/classes structurally similar to a given symbol.

        Args:
            target: Name or partial name of the symbol to compare against.
            threshold: Similarity threshold 0.0-1.0.
            top: Max results to highlight (default 10). Pass 0 for unlimited.
        """
        from interlinked.analyzer.similarity import find_similar_to

        # Resolve target
        node = self.graph.get_node(target)
        if not node:
            matches = [
                n for n in self.graph.all_nodes()
                if target.lower() in n.qualified_name.lower()
            ]
            if not matches:
                return json.dumps({"error": f"No symbol matching '{target}' found."})
            node = min(matches, key=lambda n: len(n.qualified_name))

        emb_index = getattr(self, '_embedding_index', None)
        results = find_similar_to(self.graph, node.id, threshold=threshold, embedding_index=emb_index)

        # Cap results for UI clarity (already sorted by score descending)
        if top > 0:
            results = results[:top]

        # Switch zoom level to match the target's symbol type
        _TYPE_TO_ZOOM = {
            SymbolType.MODULE: "module",
            SymbolType.CLASS: "class",
            SymbolType.FUNCTION: "function",
            SymbolType.METHOD: "function",
            SymbolType.VARIABLE: "variable",
            SymbolType.PARAMETER: "variable",
        }
        self.state.zoom_level = _TYPE_TO_ZOOM.get(node.symbol_type, "function")

        # Highlight similar symbols
        self.state.highlighted_node_ids = [node.id] + [r["id"] for r in results]
        top_score = f"{results[0]['similarity']:.0%}" if results else "n/a"
        self.state.context = ViewContext(
            what=f"Symbols similar to '{node.name}'",
            why=f"{len(results)} matches (top similarity: {top_score}, threshold: {threshold})",
            where=f"{node.qualified_name} ({node.file_path})",
            source="similarity",
        )
        self._notify()

        if not results:
            return json.dumps({"message": f"No symbols similar to '{node.qualified_name}' at threshold {threshold}.", "results": []})
        return json.dumps({
            "message": f"Found {len(results)} symbols similar to '{node.qualified_name}'.",
            "target": node.qualified_name,
            "results": results,
        }, indent=2)

    def get_context(self, target: str) -> str:
        """Get rich context for a symbol: source, docstring, comments, connections, fingerprint.

        Args:
            target: Name or partial name of the symbol.
        """
        from interlinked.analyzer.similarity import get_rich_context

        node = self.graph.get_node(target)
        if not node:
            matches = [
                n for n in self.graph.all_nodes()
                if target.lower() in n.qualified_name.lower()
            ]
            if not matches:
                return json.dumps({"error": f"No symbol matching '{target}' found."})
            node = min(matches, key=lambda n: len(n.qualified_name))

        context = get_rich_context(self.graph, node)
        return json.dumps(context, indent=2, default=str)

    # ── Cross-module edge enumeration ────────────────────────────────

    def edges_between(
        self,
        source_scope: str,
        target_scope: str | None = None,
        edge_types: list[str] | None = None,
    ) -> str:
        """List all edges from one module scope to another (or to everything external).

        Args:
            source_scope: Qualified name prefix of the source module, e.g. "engine.rules.resolver".
            target_scope: Optional target prefix. If omitted, shows all edges leaving source_scope.
            edge_types: Optional filter, e.g. ["calls", "imports"]. Default: all types.

        Returns JSON with edges grouped by target module.
        """
        et_filter = None
        if edge_types:
            et_filter = set()
            for et_str in edge_types:
                try:
                    et_filter.add(EdgeType(et_str))
                except ValueError:
                    pass

        all_edges = self.graph.all_edges(include_proposed=False)
        source_nodes = {n.id for n in self.graph.all_nodes() if n.qualified_name.startswith(source_scope)}

        edges_out: list[dict] = []
        for e in all_edges:
            if e.source not in source_nodes:
                continue
            if et_filter and e.edge_type not in et_filter:
                continue
            # Skip edges within the source scope (internal)
            target_node = self.graph.get_node(e.target)
            target_qname = target_node.qualified_name if target_node else e.target
            if target_qname.startswith(source_scope):
                continue
            # If target_scope specified, filter to it
            if target_scope and not target_qname.startswith(target_scope):
                continue
            edges_out.append({
                "source": e.source,
                "target": e.target,
                "edge_type": e.edge_type.value,
                "line": e.line,
            })

        # Group by target module (first two dotted components of target)
        by_module: dict[str, list[dict]] = {}
        for edge in edges_out:
            tgt = edge["target"]
            parts = tgt.split(".")
            mod = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            by_module.setdefault(mod, []).append(edge)

        result = {
            "source_scope": source_scope,
            "target_scope": target_scope or "(all external)",
            "total_edges": len(edges_out),
            "by_target_module": {
                mod: {"count": len(edges), "edges": edges}
                for mod, edges in sorted(by_module.items(), key=lambda x: -len(x[1]))
            },
        }

        # Highlight source nodes in view
        self.state.highlighted_node_ids = list({e["source"] for e in edges_out})
        self._notify()

        return json.dumps(result, indent=2)

    # ── Reachability ─────────────────────────────────────────────────

    def reachable(
        self,
        source: str,
        target: str,
        edge_types: list[str] | None = None,
        max_depth: int = 20,
    ) -> str:
        """Check if target is reachable from source via specific edge types.

        By default follows only 'calls' edges — use for purity/isolation checks.
        Returns the shortest path if reachable, or a clear "not reachable" message.

        Args:
            source: Qualified name of the source symbol.
            target: Qualified name (or partial) of the target symbol.
            edge_types: Edge types to traverse (default: ["calls"]).
            max_depth: Maximum path length to search.
        """
        import networkx as nx

        src_node = self._resolve_node(source)
        tgt_node = self._resolve_node(target)
        if not src_node:
            return json.dumps({"reachable": False, "error": f"Source '{source}' not found."})
        if not tgt_node:
            return json.dumps({"reachable": False, "error": f"Target '{target}' not found."})

        # Build edge-type-filtered subgraph
        et_filter = set()
        for et_str in (edge_types or ["calls"]):
            try:
                et_filter.add(EdgeType(et_str).value)
            except ValueError:
                pass
        if not et_filter:
            et_filter = {"calls"}

        sub_edges = [
            (u, v) for u, v, d in self.graph._g.edges(data=True)
            if d.get("edge_type") in et_filter
        ]
        sub_g = nx.DiGraph()
        sub_g.add_edges_from(sub_edges)

        try:
            path = nx.shortest_path(sub_g, src_node.id, tgt_node.id)
            if len(path) - 1 > max_depth:
                return json.dumps({"reachable": False, "reason": f"Path exists but exceeds max_depth={max_depth}."})

            # Highlight the path
            self.state.highlighted_node_ids = list(path)
            self.state.trace_node_roles = {path[0]: "origin", path[-1]: "destination"}
            for nid in path[1:-1]:
                self.state.trace_node_roles[nid] = "passthrough"
            self.state.trace_edge_roles = {}
            self._notify()

            short_path = " → ".join(n.split(".")[-1] for n in path)
            return json.dumps({
                "reachable": True,
                "path_length": len(path) - 1,
                "path": list(path),
                "short_path": short_path,
            }, indent=2)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return json.dumps({
                "reachable": False,
                "source": src_node.qualified_name,
                "target": tgt_node.qualified_name,
                "edge_types": list(et_filter),
            })

    # ── Display settings ─────────────────────────────────────────────

    @property
    def colors(self) -> ColorScheme:
        return self.state.colors

    @colors.setter
    def colors(self, scheme: ColorScheme) -> None:
        self.state.colors = scheme
        self._notify()

    def set_color(self, key: str, value: str) -> str:
        """Set a specific color in the scheme, e.g. set_color('dead_link', '#ff0000')."""
        if hasattr(self.state.colors, key):
            setattr(self.state.colors, key, value)
            self._notify()
            return f"Color '{key}' set to {value}."
        return f"Unknown color key: {key}. Available: {list(ColorScheme.model_fields.keys())}"

    def show_dead(self, visible: bool = True) -> str:
        self.state.show_dead = visible
        self._notify()
        return f"Dead code visibility: {visible}."

    def show_proposed(self, visible: bool = True) -> str:
        self.state.show_proposed = visible
        self._notify()
        return f"Proposed elements visibility: {visible}."

    # ── Natural language (parsed to commands) ────────────────────────

    def nl(self, text: str) -> str:
        """Natural-language command parser.

        Maps common phrasings to structured commands.
        This is designed so an LLM can simply describe what it wants to show
        and the view updates accordingly.
        """
        t = text.lower().strip()

        # ── Isolate / show me commands (primary LLM use-case) ────────
        isolate_match = re.search(
            r'(?:isolate|show\s+(?:me\s+)?|display|examine|look\s+at)\s+'
            r'(?:the\s+)?(.+?)(?:\s+at\s+(?:the\s+)?(module|class|function|variable)\s+level)?'
            r'(?:\s+(?:with\s+)?depth\s*(?:=|of)?\s*(\d+))?$',
            t
        )
        if isolate_match:
            target = isolate_match.group(1).strip().strip("'\"")
            level = isolate_match.group(2) or "function"
            depth = int(isolate_match.group(3)) if isolate_match.group(3) else 3
            # Clean up target — remove trailing "and connections" etc.
            target = re.sub(r'\s+and\s+(?:its\s+)?(?:connections|everything|dependencies).*', '', target)
            target = re.sub(r'\s+(?:module|class|service|component)$', '', target)
            return self.isolate(target, level=level, depth=depth)

        # ── Impact / blast radius ────────────────────────────────────
        impact_match = re.search(
            r'(?:impact|blast\s+radius|what\s+(?:happens|breaks|changes)\s+if\s+(?:I\s+)?(?:change|modify|remove|delete))\s+'
            r'(?:of\s+)?["\']?(\S+)["\']?', t
        )
        if impact_match:
            return self.impact_of(impact_match.group(1))

        # ── Dependencies / what feeds into ───────────────────────────
        dep_match = re.search(
            r'(?:what\s+does\s+(.+?)\s+depend\s+on|dependencies?\s+(?:of|for)\s+(.+?)(?:\s|$)|depends\s+on\s+(.+?)(?:\s|$))', t
        )
        if dep_match:
            target = (dep_match.group(1) or dep_match.group(2) or dep_match.group(3)).strip().strip("'\"")
            return self.depends_on(target)

        # ── Path between / how does X connect to Y ──────────────────
        path_match = re.search(
            r'(?:path|route|connection|how\s+does\s+.+?\s+connect)\s+'
            r'(?:between|from)\s+["\']?(\S+?)["\']?\s+(?:and|to)\s+["\']?(\S+)["\']?', t
        )
        if path_match:
            return self.path_between(path_match.group(1), path_match.group(2))

        # ── Cycles / circular dependencies ───────────────────────────
        if "cycle" in t or "circular" in t:
            return self.find_cycles()

        # ── Critical / important nodes ───────────────────────────────
        if "critical" in t or ("most" in t and "important" in t):
            return self.critical_nodes()

        # ── Bottlenecks ──────────────────────────────────────────────
        if "bottleneck" in t or "coupling hotspot" in t:
            return self.bottlenecks()

        # ── Health check ─────────────────────────────────────────────
        if "health" in t and ("check" in t or "report" in t or t == "health"):
            return self.health()

        # ── Coupling between modules ─────────────────────────────────
        coupling_match = re.search(
            r'coupling\s+(?:between\s+)?["\']?(\S+?)["\']?\s+(?:and|to|with)\s+["\']?(\S+)["\']?', t
        )
        if coupling_match:
            return self.coupling(coupling_match.group(1), coupling_match.group(2))

        # ── Dead / uncalled ──────────────────────────────────────────
        if "uncalled" in t or "never called" in t or ("dead" in t and "code" in t):
            results = self.query("dead functions")
            return f"Found {len(results)} dead/uncalled functions. Highlighted in view."

        # ── Tracing ──────────────────────────────────────────────────
        if "full path" in t or "trace" in t:
            words = text.split()
            # trace function X
            for i, w in enumerate(words):
                if w.lower() == "function" and i + 1 < len(words):
                    return self.trace_function(words[i + 1].strip("'\""))
            # trace call chain from X to Y
            chain_match = re.search(r'call\s+chain\s+(?:from\s+)?["\']?(\S+?)["\']?\s+(?:to)\s+["\']?(\S+)["\']?', t)
            if chain_match:
                return self.trace_call_chain(chain_match.group(1), chain_match.group(2))
            # trace variable X
            for i, w in enumerate(words):
                if w.lower() in ("variable", "var"):
                    if i + 1 < len(words):
                        return self.trace_variable(words[i + 1].strip("'\""))
            match = re.search(r"['\"](\w+)['\"]", text)
            if match:
                return self.trace_variable(match.group(1))
            return "Could not determine what to trace. Use: view.trace_variable('name'), view.trace_function('name'), or view.trace_call_chain('from', 'to')"

        # ── Zoom ─────────────────────────────────────────────────────
        if "zoom" in t:
            for level in ("module", "class", "function"):
                if level in t:
                    return self.zoom(level)

        # ── Focus ────────────────────────────────────────────────────
        if "focus" in t:
            match = re.search(r"focus\s+(?:on\s+)?(\S+)", t)
            if match:
                return self.focus(match.group(1))

        # ── Return type search ───────────────────────────────────────
        if "return" in t:
            match = re.search(r"return(?:s|ing)?\s+(.+)", t)
            if match:
                results = self.query(f"functions returning {match.group(1)}")
                return f"Found {len(results)} functions. Highlighted."

        # ── Reset ────────────────────────────────────────────────────
        if "reset" in t or "clear" in t:
            return self.reset_filter()

        # ── Fallback: treat as search ────────────────────────────────
        results = self.query(text)
        return f"Search for '{text}': {len(results)} results. Highlighted in view."

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Get the current graph snapshot as a dict (for JSON serialization)."""
        return self.graph.snapshot(self.state).model_dump()

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        all_nodes = self.graph.all_nodes(include_proposed=False)
        all_edges = self.graph.all_edges(include_proposed=False)
        node_ids = {n.id for n in all_nodes}

        # Count external calls (CALLS edges to non-project targets)
        ext_calls = sum(
            1 for e in all_edges
            if e.edge_type == EdgeType.CALLS and e.target not in node_ids
        )

        return {
            "total_nodes": len(all_nodes),
            "modules": len([n for n in all_nodes if n.symbol_type == SymbolType.MODULE]),
            "classes": len([n for n in all_nodes if n.symbol_type == SymbolType.CLASS]),
            "functions": len([n for n in all_nodes if n.symbol_type == SymbolType.FUNCTION]),
            "methods": len([n for n in all_nodes if n.symbol_type == SymbolType.METHOD]),
            "variables": len([n for n in all_nodes if n.symbol_type == SymbolType.VARIABLE]),
            "parameters": len([n for n in all_nodes if n.symbol_type == SymbolType.PARAMETER]),
            "dead_nodes": len([n for n in all_nodes if n.is_dead]),
            "total_edges": self.graph.edge_count,
            "external_calls": ext_calls,
        }
