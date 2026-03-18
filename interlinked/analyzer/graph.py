"""CodeGraph — a NetworkX-backed directed multigraph of Python symbols."""

from __future__ import annotations

from typing import Any, Iterator

import networkx as nx

from interlinked.models import (
    NodeData, EdgeData, EdgeType, SymbolType,
    GraphSnapshot, ViewState, ColorScheme,
)


# Method names so common on builtins (dict, list, set, str, etc.) that resolving
# them to project symbols by bare-name matching is almost always a false positive.
# e.g. `op_dict.items()` should NOT resolve to `ActionCost.items`.
# Checked against the LAST component of dotted targets (e.g. "effect.get" → "get").
_BUILTIN_METHOD_NAMES: frozenset[str] = frozenset({
    # dict
    "items", "keys", "values", "get", "pop", "update", "setdefault",
    "clear", "copy",
    # list / sequence
    "append", "extend", "insert", "remove", "sort", "reverse",
    "count", "index",
    # set
    "add", "discard", "union", "intersection", "difference",
    "issubset", "issuperset",
    # str
    "strip", "split", "join", "replace", "startswith", "endswith",
    "lower", "upper", "format", "encode", "decode",
    # general / logging
    "close", "read", "write", "flush", "seek", "tell",
    "warning", "error", "info", "debug", "exception", "critical",
})


class CodeGraph:
    """The core graph structure representing an entire Python project.

    Wraps a NetworkX MultiDiGraph with typed node/edge accessors
    and query methods used by the commander layer.
    """

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._node_data: dict[str, NodeData] = {}
        self._proposed_nodes: dict[str, NodeData] = {}
        self._proposed_edges: list[EdgeData] = []
        # Edge indexes for fast filtered lookups
        self._edges_by_source: dict[str, list[EdgeData]] = {}
        self._edges_by_target: dict[str, list[EdgeData]] = {}
        self._edge_set: set[tuple[str, str, str]] = set()  # (src, tgt, type) dedup

    # ── Construction ─────────────────────────────────────────────────

    def add_node(self, node: NodeData) -> None:
        bucket = self._proposed_nodes if node.is_proposed else self._node_data
        bucket[node.id] = node
        self._g.add_node(node.id, **node.model_dump())

    def add_edge(self, edge: EdgeData) -> None:
        if edge.is_proposed:
            self._proposed_edges.append(edge)
        self._g.add_edge(
            edge.source, edge.target,
            key=edge.edge_type.value,
            **edge.model_dump(),
        )
        # Maintain edge indexes
        dedup_key = (edge.source, edge.target, edge.edge_type.value)
        if dedup_key not in self._edge_set:
            self._edge_set.add(dedup_key)
            self._edges_by_source.setdefault(edge.source, []).append(edge)
            self._edges_by_target.setdefault(edge.target, []).append(edge)

    def _clear(self) -> None:
        """Clear all graph state for a full rebuild."""
        self._g.clear()
        self._node_data.clear()
        self._proposed_nodes.clear()
        self._proposed_edges.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()
        self._edge_set.clear()

    def _remove_edge(self, edge: EdgeData) -> None:
        """Remove an edge from the graph and all indexes."""
        dedup_key = (edge.source, edge.target, edge.edge_type.value)
        self._edge_set.discard(dedup_key)
        # Remove from source index
        src_list = self._edges_by_source.get(edge.source)
        if src_list:
            self._edges_by_source[edge.source] = [
                e for e in src_list
                if not (e.source == edge.source and e.target == edge.target and e.edge_type == edge.edge_type)
            ]
        # Remove from target index
        tgt_list = self._edges_by_target.get(edge.target)
        if tgt_list:
            self._edges_by_target[edge.target] = [
                e for e in tgt_list
                if not (e.source == edge.source and e.target == edge.target and e.edge_type == edge.edge_type)
            ]
        # Remove from NetworkX
        if self._g.has_edge(edge.source, edge.target, key=edge.edge_type.value):
            self._g.remove_edge(edge.source, edge.target, key=edge.edge_type.value)

    def remove_file(self, module_qname: str) -> dict[str, list]:
        """Remove all nodes and edges belonging to a module. Returns removed IDs.

        This removes:
        - All nodes whose qualified_name starts with module_qname
        - All edges sourced from or targeting those nodes
        - All edges sourced from or targeting module_qname itself
        """
        prefix = module_qname + "."
        removed_node_ids: list[str] = []
        removed_edges: list[EdgeData] = []

        # Find nodes to remove
        to_remove = [
            nid for nid in self._node_data
            if nid == module_qname or nid.startswith(prefix)
        ]
        removed_ids_set = set(to_remove)

        # Find edges to remove (any edge touching a removed node)
        for nid in to_remove:
            for e in list(self._edges_by_source.get(nid, [])):
                removed_edges.append(e)
                self._remove_edge(e)
            for e in list(self._edges_by_target.get(nid, [])):
                if (e.source, e.target, e.edge_type.value) in self._edge_set:
                    removed_edges.append(e)
                    self._remove_edge(e)

        # Remove nodes
        for nid in to_remove:
            del self._node_data[nid]
            if nid in self._g:
                self._g.remove_node(nid)
            removed_node_ids.append(nid)

        return {"removed_nodes": removed_node_ids, "removed_edges": removed_edges}

    def update_file(
        self,
        module_qname: str,
        new_nodes: list[NodeData],
        new_edges: list[EdgeData],
    ) -> dict[str, list]:
        """Incrementally update the graph for a single changed file.

        1. Remove all old nodes/edges for this module
        2. Add new nodes/edges
        3. Resolve new edges using the full name index

        Returns a delta dict with added/removed node IDs.
        """
        # Step 1: Remove old data for this module
        removed = self.remove_file(module_qname)

        # Step 2: Add new nodes
        for n in new_nodes:
            self.add_node(n)

        # Step 3: Build name index from ALL current nodes for edge resolution
        all_nodes = self.all_nodes(include_proposed=False)
        name_index: dict[str, set[str]] = {}
        for n in all_nodes:
            name_index.setdefault(n.name, set()).add(n.id)
            parts = n.qualified_name.split(".")
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:])
                name_index.setdefault(suffix, set()).add(n.id)

        node_ids = {n.id for n in all_nodes}

        # Step 4: Resolve and add new edges (skip external references)
        added_edges: list[EdgeData] = []
        for e in new_edges:
            resolved = self._resolve_edge(e, node_ids, name_index)
            if resolved.source in node_ids and resolved.target in node_ids:
                self.add_edge(resolved)
                added_edges.append(resolved)

        return {
            "removed_nodes": removed["removed_nodes"],
            "added_nodes": [n.id for n in new_nodes],
            "removed_edges": removed["removed_edges"],
            "added_edges": added_edges,
        }

    def build_from(self, nodes: list[NodeData], edges: list[EdgeData]) -> None:
        """Populate from parser output, resolving short names to qualified IDs."""
        self._clear()
        for n in nodes:
            self.add_node(n)

        # Build a lookup: short name -> set of qualified IDs
        # Sets prevent duplicates from suffix indexing (which caused
        # _resolve_edge to see len>1 for single-node names and bail).
        name_index: dict[str, set[str]] = {}
        for n in nodes:
            name_index.setdefault(n.name, set()).add(n.id)
            # Also index by qualified_name suffix fragments
            # e.g. "graph.CodeGraph" for "analyzer.graph.CodeGraph"
            parts = n.qualified_name.split(".")
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:])
                name_index.setdefault(suffix, set()).add(n.id)

        node_ids = {n.id for n in nodes}

        for e in edges:
            resolved = self._resolve_edge(e, node_ids, name_index)
            # Source must be a known project node. Targets may be
            # unresolved for CALLS/READS (inference gaps on untyped
            # variables), but structural edges need both endpoints.
            if resolved.source not in node_ids:
                continue
            self.add_edge(resolved)

    @staticmethod
    def _resolve_edge(
        edge: EdgeData,
        node_ids: set[str],
        name_index: dict[str, set[str]],
    ) -> EdgeData:
        """Try to resolve unqualified source/target names to known node IDs."""
        source = edge.source
        target = edge.target

        if source not in node_ids:
            candidates = name_index.get(source, set())
            if len(candidates) == 1:
                source = next(iter(candidates))

        if target not in node_ids:
            # Check the last dotted component against builtin method names.
            # Catches both bare "items" and dotted "effect.get", "args.items".
            leaf = target.rsplit(".", 1)[-1]
            if leaf in _BUILTIN_METHOD_NAMES:
                return edge

            # Dotted targets like "effect.get" or "logger.warning" where the
            # root is NOT a known node are local-variable method calls —
            # resolving them through the name index produces false positives.
            if "." in target:
                root = target.split(".", 1)[0]
                # If the root isn't itself a project node, it's a local var
                if root not in node_ids and root not in name_index:
                    return edge

            candidates = name_index.get(target, set())
            if len(candidates) == 1:
                target = next(iter(candidates))
            elif len(candidates) > 1:
                # Exclude self-calls (bare `process()` != `self.process()`)
                filtered = candidates - {source}
                if not filtered:
                    filtered = candidates

                # Prefer candidates sharing the longest common prefix with
                # the source.  This ensures a call from engine.rules.resolver
                # to _resolve_entity_ref picks resolver's own definition
                # over engine.rules.field_paths._resolve_entity_ref.
                src_parts = source.split(".")

                def _common_prefix_len(c: str) -> int:
                    c_parts = c.split(".")
                    n = 0
                    for a, b in zip(src_parts, c_parts):
                        if a == b:
                            n += 1
                        else:
                            break
                    return n

                best = max(filtered, key=lambda c: (_common_prefix_len(c), -c.count(".")))
                target = best

        if source == edge.source and target == edge.target:
            return edge

        return EdgeData(
            source=source,
            target=target,
            edge_type=edge.edge_type,
            is_dead=edge.is_dead,
            is_proposed=edge.is_proposed,
            line=edge.line,
            metadata=edge.metadata,
        )

    # ── Node access ──────────────────────────────────────────────────

    def get_node(self, node_id: str) -> NodeData | None:
        return self._node_data.get(node_id) or self._proposed_nodes.get(node_id)

    def all_nodes(self, include_proposed: bool = True) -> list[NodeData]:
        nodes = list(self._node_data.values())
        if include_proposed:
            nodes.extend(self._proposed_nodes.values())
        return nodes

    def nodes_by_type(self, sym_type: SymbolType) -> list[NodeData]:
        return [n for n in self._node_data.values() if n.symbol_type == sym_type]

    @property
    def node_count(self) -> int:
        return len(self._node_data) + len(self._proposed_nodes)

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    # ── Edge access ──────────────────────────────────────────────────

    def all_edges(self, include_proposed: bool = True) -> list[EdgeData]:
        edges: list[EdgeData] = []
        seen = set()
        for u, v, data in self._g.edges(data=True):
            key = (u, v, data.get("edge_type"))
            if key not in seen:
                seen.add(key)
                ed = EdgeData(**{k: data[k] for k in EdgeData.model_fields if k in data})
                if not include_proposed and ed.is_proposed:
                    continue
                edges.append(ed)
        return edges

    def edges_from(self, node_id: str, edge_type: EdgeType | None = None) -> list[EdgeData]:
        if node_id not in self._g:
            return []
        result = []
        for _, v, data in self._g.out_edges(node_id, data=True):
            if edge_type and data.get("edge_type") != edge_type.value:
                continue
            result.append(EdgeData(**{k: data[k] for k in EdgeData.model_fields if k in data}))
        return result

    def edges_to(self, node_id: str, edge_type: EdgeType | None = None) -> list[EdgeData]:
        if node_id not in self._g:
            return []
        result = []
        for u, _, data in self._g.in_edges(node_id, data=True):
            if edge_type and data.get("edge_type") != edge_type.value:
                continue
            result.append(EdgeData(**{k: data[k] for k in EdgeData.model_fields if k in data}))
        return result

    # ── Queries ──────────────────────────────────────────────────────

    def callers_of(self, node_id: str) -> list[NodeData]:
        """Who calls this function?"""
        return [
            self._node_data[e.source]
            for e in self.edges_to(node_id, EdgeType.CALLS)
            if e.source in self._node_data
        ]

    def callees_of(self, node_id: str) -> list[NodeData]:
        """What does this function call?"""
        return [
            self._node_data[e.target]
            for e in self.edges_from(node_id, EdgeType.CALLS)
            if e.target in self._node_data
        ]

    def subgraph_around(
        self, node_id: str, depth: int = 2, edge_types: list[EdgeType] | None = None
    ) -> tuple[list[NodeData], list[EdgeData]]:
        """BFS expansion around a node up to `depth` hops."""
        if node_id not in self._g:
            return [], []

        visited: set[str] = set()
        frontier: set[str] = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for _, v, data in self._g.out_edges(nid, data=True):
                    if edge_types and data.get("edge_type") not in [et.value for et in edge_types]:
                        continue
                    next_frontier.add(v)
                for u, _, data in self._g.in_edges(nid, data=True):
                    if edge_types and data.get("edge_type") not in [et.value for et in edge_types]:
                        continue
                    next_frontier.add(u)
            frontier = next_frontier - visited

        visited |= frontier
        nodes = [self.get_node(nid) for nid in visited if self.get_node(nid)]
        edges = [
            e for e in self.all_edges()
            if e.source in visited and e.target in visited
        ]
        return nodes, edges  # type: ignore[return-value]

    def trace_variable(self, var_name: str, origin: str | None = None) -> tuple[list[NodeData], list[EdgeData], dict[str, str], dict[str, str]]:
        """Trace a variable's read/write path using real graph pathfinding.

        Uses nx.ancestors/descendants to find the full data-flow picture,
        then nx.all_simple_paths to find actual connecting paths between
        writers and readers.

        Returns (nodes, edges, node_roles, edge_roles) where edge_roles
        are keyed by ``"src_id|tgt_id"`` using real node IDs.
        """
        writers: set[str] = set()
        readers: set[str] = set()
        var_targets: set[str] = set()

        # 1. Find all functions that read/write this variable
        for e in self.all_edges():
            if e.edge_type in (EdgeType.READS, EdgeType.WRITES):
                target_name = e.target.split(".")[-1] if "." in e.target else e.target
                if target_name == var_name:
                    if origin and not e.source.startswith(origin):
                        continue
                    var_targets.add(e.target)
                    if e.edge_type == EdgeType.WRITES:
                        writers.add(e.source)
                    else:
                        readers.add(e.source)

        trace_func_ids = writers | readers
        if not trace_func_ids:
            return [], [], {}, {}

        # 2. Use NetworkX to find paths between writers and readers
        #    This gives us the actual intermediate nodes in the call chain
        path_nodes: set[str] = set(trace_func_ids)
        path_nodes |= var_targets

        # Use flow-only subgraph (no CONTAINS/INHERITS) for pathfinding — much faster
        flow_graph = self._g.edge_subgraph(
            [(u, v, k) for u, v, k in self._g.edges(keys=True)
             if k not in ("contains", "inherits")]
        )

        for w in writers:
            for r in readers:
                if w == r:
                    continue
                for src, tgt in [(w, r), (r, w)]:
                    if src in flow_graph and tgt in flow_graph:
                        try:
                            for path in nx.all_simple_paths(flow_graph, src, tgt, cutoff=5):
                                path_nodes.update(path)
                        except nx.NetworkXError:
                            pass

        # Also add ancestors/descendants of each writer/reader within the trace
        for nid in list(trace_func_ids):
            if nid not in flow_graph:
                continue
            try:
                anc = nx.ancestors(flow_graph, nid)
                path_nodes |= (anc & trace_func_ids)
            except nx.NetworkXError:
                pass
            try:
                desc = nx.descendants(flow_graph, nid)
                path_nodes |= (desc & trace_func_ids)
            except nx.NetworkXError:
                pass

        # 3. Collect ALL edges between path participants
        relevant_edges: list[EdgeData] = []
        edge_roles: dict[str, str] = {}

        for e in self.all_edges():
            if e.edge_type == EdgeType.CONTAINS:
                continue
            src_in = e.source in path_nodes
            tgt_in = e.target in path_nodes
            tgt_is_var = e.target in var_targets

            if e.edge_type in (EdgeType.READS, EdgeType.WRITES) and src_in and tgt_is_var:
                relevant_edges.append(e)
                key = f"{e.source}|{e.target}"
                edge_roles[key] = "write" if e.edge_type == EdgeType.WRITES else "read"
            elif src_in and tgt_in:
                relevant_edges.append(e)
                key = f"{e.source}|{e.target}"
                if key not in edge_roles:
                    if e.edge_type == EdgeType.WRITES:
                        edge_roles[key] = "write"
                    elif e.edge_type == EdgeType.READS:
                        edge_roles[key] = "read"
                    else:
                        edge_roles[key] = "flow"

        # 4. Classify node roles
        node_roles: dict[str, str] = {}
        for vid in var_targets:
            node_roles[vid] = "origin"

        writer_list = sorted(writers, key=lambda w: min(
            (e.line or 9999 for e in relevant_edges
             if e.source == w and e.edge_type == EdgeType.WRITES), default=9999
        ))

        for nid in path_nodes - var_targets:
            if nid in writers and nid in readers:
                node_roles[nid] = "mutator"
            elif nid in writers:
                if writer_list and nid == writer_list[0]:
                    node_roles[nid] = "origin"
                else:
                    node_roles[nid] = "mutator"
            elif nid in readers:
                node_roles[nid] = "destination"
            else:
                # Intermediate node on a path (not a direct reader/writer)
                node_roles[nid] = "passthrough"

        # Upgrade pure readers to passthrough if they connect to other trace nodes
        for nid in readers - writers:
            if nid not in self._g:
                continue
            has_outgoing = any(
                e.target in path_nodes and e.target != nid
                for e in self.edges_from(nid, EdgeType.CALLS)
            )
            if has_outgoing:
                node_roles[nid] = "passthrough"

        nodes = [self.get_node(nid) for nid in path_nodes if self.get_node(nid)]
        return nodes, relevant_edges, node_roles, edge_roles  # type: ignore[return-value]

    # ── Tracing & Pathfinding (Phase 1c) ─────────────────────────────

    def trace_function(self, node_id: str) -> tuple[list[NodeData], list[EdgeData], dict[str, str], dict[str, str]]:
        """Trace a function's call chain — everything that calls it and everything it calls.

        Uses nx.ancestors and nx.descendants on calls-only subgraph.
        """
        if node_id not in self._g:
            return [], [], {}, {}

        # Build a calls-only view
        calls_edges = {
            (e.source, e.target) for e in self.all_edges()
            if e.edge_type == EdgeType.CALLS
        }
        calls_graph = self._g.edge_subgraph(
            [(u, v, k) for u, v, k in self._g.edges(keys=True) if k == "calls"]
        )

        path_nodes: set[str] = {node_id}
        try:
            path_nodes |= nx.ancestors(calls_graph, node_id)
        except nx.NetworkXError:
            pass
        try:
            path_nodes |= nx.descendants(calls_graph, node_id)
        except nx.NetworkXError:
            pass

        # Collect edges between participants
        relevant_edges: list[EdgeData] = []
        edge_roles: dict[str, str] = {}
        for e in self.all_edges():
            if e.edge_type == EdgeType.CONTAINS:
                continue
            if e.source in path_nodes and e.target in path_nodes:
                relevant_edges.append(e)
                key = f"{e.source}|{e.target}"
                if e.edge_type == EdgeType.CALLS:
                    edge_roles[key] = "flow"
                elif e.edge_type == EdgeType.READS:
                    edge_roles[key] = "read"
                elif e.edge_type == EdgeType.WRITES:
                    edge_roles[key] = "write"
                else:
                    edge_roles.setdefault(key, "flow")

        # Classify roles: the target is origin, callers are upstream, callees are downstream
        node_roles: dict[str, str] = {}
        try:
            upstream = nx.ancestors(calls_graph, node_id)
        except nx.NetworkXError:
            upstream = set()
        try:
            downstream = nx.descendants(calls_graph, node_id)
        except nx.NetworkXError:
            downstream = set()

        for nid in path_nodes:
            if nid == node_id:
                node_roles[nid] = "origin"
            elif nid in upstream and nid in downstream:
                node_roles[nid] = "mutator"  # in a cycle with target
            elif nid in upstream:
                node_roles[nid] = "passthrough"  # callers
            elif nid in downstream:
                node_roles[nid] = "destination"  # callees
            else:
                node_roles[nid] = "passthrough"

        nodes = [self.get_node(nid) for nid in path_nodes if self.get_node(nid)]
        return nodes, relevant_edges, node_roles, edge_roles  # type: ignore[return-value]

    def trace_call_chain(self, source: str, target: str, max_depth: int = 8) -> tuple[list[NodeData], list[EdgeData], dict[str, str], dict[str, str]]:
        """Find all call paths from source to target.

        Uses nx.all_simple_paths on calls-only subgraph.
        """
        if source not in self._g or target not in self._g:
            return [], [], {}, {}

        calls_graph = self._g.edge_subgraph(
            [(u, v, k) for u, v, k in self._g.edges(keys=True) if k == "calls"]
        )

        path_nodes: set[str] = set()
        try:
            for path in nx.all_simple_paths(calls_graph, source, target, cutoff=max_depth):
                path_nodes.update(path)
        except nx.NetworkXError:
            pass

        if not path_nodes:
            # Try reverse direction
            try:
                for path in nx.all_simple_paths(calls_graph, target, source, cutoff=max_depth):
                    path_nodes.update(path)
            except nx.NetworkXError:
                pass

        if not path_nodes:
            return [], [], {}, {}

        relevant_edges: list[EdgeData] = []
        edge_roles: dict[str, str] = {}
        for e in self.all_edges():
            if e.edge_type == EdgeType.CONTAINS:
                continue
            if e.source in path_nodes and e.target in path_nodes:
                relevant_edges.append(e)
                key = f"{e.source}|{e.target}"
                edge_roles.setdefault(key, "flow")

        node_roles: dict[str, str] = {}
        for nid in path_nodes:
            if nid == source:
                node_roles[nid] = "origin"
            elif nid == target:
                node_roles[nid] = "destination"
            else:
                node_roles[nid] = "passthrough"

        nodes = [self.get_node(nid) for nid in path_nodes if self.get_node(nid)]
        return nodes, relevant_edges, node_roles, edge_roles  # type: ignore[return-value]

    # ── Impact & Dependency (Phase 2) ────────────────────────────────

    def impact_of(self, node_id: str) -> set[str]:
        """Everything downstream — if I change this node, what's affected?

        Uses edge-subgraph excluding CONTAINS so we only follow
        real data/control flow (calls, reads, writes, returns).
        """
        if node_id not in self._g:
            return set()
        flow_graph = self._g.edge_subgraph(
            [(u, v, k) for u, v, k in self._g.edges(keys=True)
             if k not in ("contains", "inherits")]
        )
        if node_id not in flow_graph:
            return set()
        return nx.descendants(flow_graph, node_id)

    def feeds_into(self, node_id: str) -> set[str]:
        """Everything upstream — what does this node depend on?"""
        if node_id not in self._g:
            return set()
        return nx.ancestors(self._g, node_id)

    def path_between(self, source: str, target: str) -> list[str]:
        """Shortest dependency chain from source to target."""
        try:
            return nx.shortest_path(self._g, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Try reverse
            try:
                return nx.shortest_path(self._g, target, source)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

    def all_paths_between(self, source: str, target: str, max_depth: int = 8) -> list[list[str]]:
        """Every route from source to target."""
        paths: list[list[str]] = []
        try:
            paths.extend(nx.all_simple_paths(self._g, source, target, cutoff=max_depth))
        except (nx.NetworkXError, nx.NodeNotFound):
            pass
        if not paths:
            try:
                paths.extend(nx.all_simple_paths(self._g, target, source, cutoff=max_depth))
            except (nx.NetworkXError, nx.NodeNotFound):
                pass
        return paths

    def are_connected(self, a: str, b: str) -> bool:
        """Can data/control flow from a to b (or b to a)?"""
        try:
            return nx.has_path(self._g, a, b) or nx.has_path(self._g, b, a)
        except nx.NodeNotFound:
            return False

    # ── Architecture Health (Phase 3) ────────────────────────────────

    def find_cycles(self) -> list[list[str]]:
        """Find circular dependencies (via calls and imports edges only)."""
        calls_graph = self._g.edge_subgraph(
            [(u, v, k) for u, v, k in self._g.edges(keys=True)
             if k in ("calls", "imports")]
        )
        try:
            return list(nx.simple_cycles(calls_graph))
        except nx.NetworkXError:
            return []

    def has_circular_deps(self) -> bool:
        """Quick check: are there any circular dependencies?"""
        calls_graph = self._g.edge_subgraph(
            [(u, v, k) for u, v, k in self._g.edges(keys=True)
             if k in ("calls", "imports")]
        )
        return not nx.is_directed_acyclic_graph(calls_graph)

    def critical_nodes(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Most important nodes by PageRank."""
        try:
            scores = nx.pagerank(self._g)
            return sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        except (nx.NetworkXError, ImportError, ModuleNotFoundError):
            return []

    def bottlenecks(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Nodes everything flows through — high betweenness centrality."""
        try:
            scores = nx.betweenness_centrality(self._g)
            return sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        except (nx.NetworkXError, ImportError, ModuleNotFoundError):
            return []

    def most_coupled(self, top_n: int = 20) -> list[tuple[str, int]]:
        """Nodes with highest fan-in + fan-out."""
        return sorted(
            ((n, self._g.in_degree(n) + self._g.out_degree(n)) for n in self._g.nodes()),
            key=lambda x: -x[1],
        )[:top_n]

    def find_clusters(self) -> list[set[str]]:
        """Isolated groups of code — weakly connected components."""
        return [c for c in nx.weakly_connected_components(self._g)]

    def circular_clusters(self) -> list[set[str]]:
        """Groups of mutually dependent code — strongly connected components > 1."""
        return [c for c in nx.strongly_connected_components(self._g) if len(c) > 1]

    def dependency_layers(self) -> list[list[str]]:
        """Topological ordering — natural dependency layers."""
        try:
            condensed = nx.condensation(self._g)
            return [list(gen) for gen in nx.topological_generations(condensed)]
        except nx.NetworkXError:
            return []

    def coupling_between(self, module_a: str, module_b: str) -> dict[str, Any]:
        """How tightly coupled are two modules? Cross-module edges."""
        a_nodes = {n for n in self._g.nodes() if n.startswith(module_a)}
        b_nodes = {n for n in self._g.nodes() if n.startswith(module_b)}
        cross: list[dict[str, str]] = []
        for e in self.all_edges():
            if e.edge_type == EdgeType.CONTAINS:
                continue
            a_to_b = e.source in a_nodes and e.target in b_nodes
            b_to_a = e.source in b_nodes and e.target in a_nodes
            if a_to_b or b_to_a:
                cross.append({"source": e.source, "target": e.target, "type": e.edge_type.value})
        return {"edge_count": len(cross), "edges": cross}

    # ── Enhanced Dead Code (Phase 4) ─────────────────────────────────

    def truly_dead(self, entry_points: list[str] | None = None) -> list[str]:
        """Find code unreachable from any entry point via graph reachability."""
        if entry_points is None:
            entry_points = [n.id for n in self.all_nodes() if n.symbol_type == SymbolType.MODULE]
        reachable: set[str] = set()
        for ep in entry_points:
            if ep in self._g:
                reachable |= nx.descendants(self._g, ep) | {ep}
        return [n.id for n in self.all_nodes(include_proposed=False) if n.id not in reachable]

    def functions_returning(self, type_hint: str) -> list[NodeData]:
        """Find functions whose return annotation matches a string."""
        results = []
        for n in self._node_data.values():
            if n.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
                if n.signature and f"-> {type_hint}" in (n.metadata.get("return_annotation", "")):
                    results.append(n)
                # Fallback: check signature metadata
                if n.metadata.get("return_annotation") == type_hint:
                    results.append(n)
        return results

    def unreachable_from(self, entry_point: str) -> list[NodeData]:
        """Find all nodes NOT reachable from a given entry point via calls."""
        if entry_point not in self._g:
            return []
        reachable = set(nx.descendants(self._g, entry_point)) | {entry_point}
        return [
            n for n in self._node_data.values()
            if n.id not in reachable
            and n.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)
        ]

    # ── Proposed / Hypothetical ──────────────────────────────────────

    def propose_function(
        self,
        name: str,
        module: str,
        calls: list[str] | None = None,
        called_by: list[str] | None = None,
        signature: str | None = None,
    ) -> NodeData:
        """Add a hypothetical function to the graph."""
        qname = f"{module}.{name}"
        node = NodeData(
            id=qname,
            name=name,
            qualified_name=qname,
            symbol_type=SymbolType.FUNCTION,
            is_proposed=True,
            signature=signature or f"def {name}(...)",
        )
        self.add_node(node)

        for callee in (calls or []):
            edge = EdgeData(
                source=qname, target=callee,
                edge_type=EdgeType.CALLS, is_proposed=True,
            )
            self.add_edge(edge)

        for caller in (called_by or []):
            edge = EdgeData(
                source=caller, target=qname,
                edge_type=EdgeType.CALLS, is_proposed=True,
            )
            self.add_edge(edge)

        return node

    def clear_proposed(self) -> None:
        """Remove all hypothetical nodes and edges."""
        for nid in list(self._proposed_nodes.keys()):
            self._g.remove_node(nid)
        self._proposed_nodes.clear()
        self._proposed_edges.clear()

    # ── Snapshot for frontend ────────────────────────────────────────

    def snapshot(self, view: ViewState | None = None) -> GraphSnapshot:
        """Generate a filtered snapshot based on the current ViewState."""
        if view is None:
            view = ViewState()

        nodes = self._filter_nodes(view)
        node_ids = {n.id for n in nodes}
        edges = self._filter_edges(view, node_ids)

        # Remap highlights and trace roles to visible ancestors at the current zoom level
        if view.highlighted_node_ids:
            remapped: set[str] = set()
            remapped_node_roles: dict[str, str] = {}
            remapped_edge_roles: dict[str, str] = {}
            # Priority for role merging when multiple children collapse into one ancestor
            role_priority = {"origin": 0, "mutator": 1, "passthrough": 2, "destination": 3}

            for hid in view.highlighted_node_ids:
                if hid in node_ids:
                    remapped.add(hid)
                    if hid in view.trace_node_roles:
                        self._merge_role(remapped_node_roles, hid, view.trace_node_roles[hid], role_priority)
                else:
                    ancestor = self._ancestor_at_zoom(hid, node_ids)
                    if ancestor:
                        remapped.add(ancestor)
                        if hid in view.trace_node_roles:
                            self._merge_role(remapped_node_roles, ancestor, view.trace_node_roles[hid], role_priority)

            # Remap edge roles to ancestor edges
            for ekey, role in view.trace_edge_roles.items():
                src, tgt = ekey.split("|", 1)
                new_src = src if src in node_ids else (self._ancestor_at_zoom(src, node_ids) or src)
                new_tgt = tgt if tgt in node_ids else (self._ancestor_at_zoom(tgt, node_ids) or tgt)
                new_key = f"{new_src}|{new_tgt}"
                # Write takes priority over read
                if new_key not in remapped_edge_roles or role == "write":
                    remapped_edge_roles[new_key] = role

            # If trace was active but no roles survived remapping (e.g. traced
            # symbols don't exist inside any class at class-zoom), preserve the
            # original roles so the frontend legend still shows trace sections.
            final_node_roles = remapped_node_roles if remapped_node_roles else view.trace_node_roles
            final_edge_roles = remapped_edge_roles if remapped_edge_roles else view.trace_edge_roles

            view = view.model_copy(update={
                "highlighted_node_ids": list(remapped),
                "trace_node_roles": final_node_roles,
                "trace_edge_roles": final_edge_roles,
            })

        return GraphSnapshot(nodes=nodes, edges=edges, view=view)

    @staticmethod
    def _merge_role(roles: dict[str, str], node_id: str, role: str, priority: dict[str, int]) -> None:
        """Merge a role into the dict, keeping the highest-priority (lowest number) role."""
        if node_id not in roles or priority.get(role, 99) < priority.get(roles[node_id], 99):
            roles[node_id] = role

    def _filter_nodes(self, view: ViewState) -> list[NodeData]:
        all_nodes = self.all_nodes(include_proposed=view.show_proposed)

        # Zoom level filter — each level shows ONLY its symbol types
        type_filter: set[SymbolType] = set()
        if view.zoom_level == "module":
            type_filter = {SymbolType.MODULE}
        elif view.zoom_level == "class":
            type_filter = {SymbolType.CLASS}
        elif view.zoom_level == "function":
            type_filter = {SymbolType.FUNCTION, SymbolType.METHOD}
        elif view.zoom_level == "variable":
            type_filter = {SymbolType.VARIABLE, SymbolType.PARAMETER}
        elif view.zoom_level == "all":
            type_filter = set(SymbolType)
        else:
            type_filter = {SymbolType.FUNCTION, SymbolType.METHOD}

        nodes = [n for n in all_nodes if n.symbol_type in type_filter]

        if not view.show_dead:
            nodes = [n for n in nodes if not n.is_dead]

        # Focus node — only show subgraph around it
        if view.focus_node:
            sub_nodes, _ = self.subgraph_around(
                view.focus_node, depth=view.focus_depth
            )
            sub_ids = {n.id for n in sub_nodes}
            nodes = [n for n in nodes if n.id in sub_ids]

        # Explicit visible list overrides
        if view.visible_node_ids:
            vis = set(view.visible_node_ids)
            nodes = [n for n in nodes if n.id in vis]

        return nodes

    def _ancestor_at_zoom(self, node_id: str, visible_ids: set[str]) -> str | None:
        """Walk up the containment hierarchy to find a visible ancestor."""
        current = node_id
        while current:
            if current in visible_ids:
                return current
            # Go up one level: "a.b.c.d" -> "a.b.c"
            if "." in current:
                current = current.rsplit(".", 1)[0]
            else:
                break
        return None

    def _filter_edges(self, view: ViewState, node_ids: set[str]) -> list[EdgeData]:
        # Use edge index: only scan edges originating from or targeting visible nodes
        # This is O(V * avg_degree) instead of O(E) — critical for 200k+ edge graphs
        visible_etypes = set(view.visible_edge_types)

        # Collect candidate edges from the index (edges touching visible nodes)
        # We also need edges from non-visible nodes that aggregate UP to visible ancestors
        candidate_edges: list[EdgeData] = []
        seen_sources: set[str] = set()

        # Edges from visible nodes
        for nid in node_ids:
            for e in self._edges_by_source.get(nid, []):
                if not view.show_proposed and e.is_proposed:
                    continue
                if not view.show_dead and e.is_dead:
                    continue
                candidate_edges.append(e)
            seen_sources.add(nid)

        # For coarser zoom levels, we also need edges from descendant nodes
        # that aggregate up to visible ancestors. Collect edges from all nodes
        # that have a visible ancestor (i.e., nodes whose qualified_name starts
        # with a visible node's qualified_name).
        # We use the _edges_by_source index keyed on all source nodes.
        if view.zoom_level in ("module", "class"):
            # Build prefix set for fast startswith checks
            prefixes = sorted(node_ids)
            for src_id in self._edges_by_source:
                if src_id in seen_sources:
                    continue
                # Check if this source has a visible ancestor
                ancestor = self._ancestor_at_zoom(src_id, node_ids)
                if ancestor:
                    for e in self._edges_by_source[src_id]:
                        if not view.show_proposed and e.is_proposed:
                            continue
                        if not view.show_dead and e.is_dead:
                            continue
                        candidate_edges.append(e)

        # Aggregate edges up to visible ancestors (dedup by src+tgt+type)
        aggregated: dict[tuple[str, str, str], EdgeData] = {}
        for e in candidate_edges:
            if e.edge_type not in visible_etypes:
                continue
            # Skip containment edges at module/class zoom — they're implicit
            if e.edge_type == EdgeType.CONTAINS:
                if e.source in node_ids and e.target in node_ids:
                    aggregated[(e.source, e.target, e.edge_type.value)] = e
                continue

            src = e.source if e.source in node_ids else self._ancestor_at_zoom(e.source, node_ids)
            tgt = e.target if e.target in node_ids else self._ancestor_at_zoom(e.target, node_ids)

            if src and tgt and src != tgt:
                key = (src, tgt, e.edge_type.value)
                if key not in aggregated:
                    aggregated[key] = EdgeData(
                        source=src,
                        target=tgt,
                        edge_type=e.edge_type,
                        is_dead=e.is_dead,
                        is_proposed=e.is_proposed,
                    )

        return list(aggregated.values())
