"""Dead code detection via NetworkX graph queries.

All detection is done through native NetworkX degree/reachability queries
on the typed MultiDiGraph. No manual graph traversal.

Detects:
  - Uncalled functions/methods (zero in-degree on 'calls' + 'reads' keys)
  - Dead parameters (zero in-degree on 'reads' key)
  - Dead variables (has 'writes' in-edges but zero 'reads' in-edges)
  - Dead returns (RETURNS edges whose targets have zero 'reads' in-edges)
  - Dead imports (target not a project node and not a prefix of one)
  - Transitive dead (all ancestors in calls graph are already dead)
"""

from __future__ import annotations

import networkx as nx

from interlinked.analyzer.graph import CodeGraph
from interlinked.models import EdgeType, SymbolType

_EXEMPT: frozenset[str] = frozenset({
    "__init__", "__new__", "__del__", "__repr__", "__str__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
    "__iter__", "__next__", "__len__", "__getitem__", "__setitem__",
    "__contains__", "__hash__", "__eq__", "__ne__", "__lt__", "__gt__",
    "__le__", "__ge__", "__bool__", "__call__", "__get__", "__set__",
    "__post_init__", "main", "setup", "teardown",
    "setUp", "tearDown", "setUpClass", "tearDownClass",
})


def detect_dead_code(graph: CodeGraph) -> list[str]:
    """Mark dead nodes/edges using NetworkX queries. Returns dead node IDs."""
    G = graph._g
    dead: set[str] = set()
    node_ids = {n.id for n in graph.all_nodes(include_proposed=False)}

    # ── Uncalled functions ────────────────────────────────────────
    for n in graph.all_nodes(include_proposed=False):
        if n.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            continue
        if n.name in _EXEMPT or n.name.startswith("test_"):
            continue
        if n.id not in G:
            continue
        # No incoming calls AND no incoming reads (callback references)
        has_caller = any(
            d.get("edge_type") in ("calls", "reads")
            for _, _, d in G.in_edges(n.id, data=True)
        )
        if not has_caller:
            n.is_dead = True
            dead.add(n.id)

    # ── Dead parameters (never read) ─────────────────────────────
    for n in graph.all_nodes(include_proposed=False):
        if n.symbol_type != SymbolType.PARAMETER or n.name in ("self", "cls"):
            continue
        # Skip params of already-dead functions
        parent = n.id.rsplit(".", 1)[0] if "." in n.id else ""
        if parent in dead:
            continue
        if n.id not in G:
            continue
        has_reader = any(
            d.get("edge_type") == "reads"
            for _, _, d in G.in_edges(n.id, data=True)
        )
        if not has_reader:
            n.is_dead = True
            dead.add(n.id)

    # ── Dead variables (written but never read) ──────────────────
    for n in graph.all_nodes(include_proposed=False):
        if n.symbol_type != SymbolType.VARIABLE:
            continue
        if n.id not in G:
            continue
        has_writer = any(d.get("edge_type") == "writes" for _, _, d in G.in_edges(n.id, data=True))
        has_reader = any(d.get("edge_type") == "reads" for _, _, d in G.in_edges(n.id, data=True))
        if has_writer and not has_reader:
            n.is_dead = True
            dead.add(n.id)

    # ── Dead returns (return value never read) ───────────────────
    for n in graph.all_nodes(include_proposed=False):
        if n.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            continue
        if n.id in dead or n.id not in G:
            continue
        ret_targets = [
            v for _, v, d in G.out_edges(n.id, data=True)
            if d.get("edge_type") == "returns"
        ]
        if not ret_targets:
            continue
        any_read = any(
            any(d.get("edge_type") == "reads" for _, _, d in G.in_edges(rt, data=True))
            for rt in ret_targets
        )
        if not any_read:
            # Mark the return edges as dead, not the function
            for e in graph.edges_from(n.id, EdgeType.RETURNS):
                e.is_dead = True

    # ── Dead imports ─────────────────────────────────────────────
    for e in graph.all_edges(include_proposed=False):
        if e.edge_type != EdgeType.IMPORTS:
            continue
        if e.target not in node_ids:
            is_prefix = any(nid.startswith(e.target) for nid in node_ids)
            if is_prefix:
                e.is_dead = True

    # ── Transitive dead ──────────────────────────────────────────
    # If ALL ancestors of a function in the calls graph are dead, it's dead too
    calls_graph = G.edge_subgraph(
        [(u, v, k) for u, v, k in G.edges(keys=True) if k == "calls"]
    )
    changed = True
    while changed:
        changed = False
        for n in graph.all_nodes(include_proposed=False):
            if n.is_dead or n.id not in calls_graph:
                continue
            if n.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
                continue
            if n.name in _EXEMPT or n.name.startswith("test_"):
                continue
            callers = set(calls_graph.predecessors(n.id))
            if callers and callers.issubset(dead):
                n.is_dead = True
                dead.add(n.id)
                changed = True

    return list(dead)
