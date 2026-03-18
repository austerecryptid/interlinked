"""Dead code detection via forward reachability from production entry points.

A symbol is dead if no production execution path can reach it.  This is
computed by building a calls-graph (including decorator registrations and
module/class-level calls captured by the parser) and doing a forward BFS
from every production entry point.

Entry points:
  - Module nodes (their scope-level code runs on import)
  - Dunder methods (called by the Python runtime)
  - ``main``, ``setup``/``teardown`` and similar framework hooks
  - Decorated functions (the parser emits calls edges from decorator → fn)

*Test-only reachability does not count.* If a symbol is reachable only from
``test_*`` functions, it is still flagged as dead — tests exercise code,
they don't make it production-live.

Additionally detects:
  - Dead variables (written but never read)
  - Dead return edges (return value never consumed)
  - Dead imports (target not a project node and not a prefix of one)
"""

from __future__ import annotations

from collections import deque

from interlinked.analyzer.graph import CodeGraph
from interlinked.models import EdgeType, SymbolType

_EXEMPT_NAMES: frozenset[str] = frozenset({
    "__init__", "__new__", "__del__", "__repr__", "__str__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
    "__iter__", "__next__", "__len__", "__getitem__", "__setitem__",
    "__contains__", "__hash__", "__eq__", "__ne__", "__lt__", "__gt__",
    "__le__", "__ge__", "__bool__", "__call__", "__get__", "__set__",
    "__post_init__", "main", "setup", "teardown",
    "setUp", "tearDown", "setUpClass", "tearDownClass",
})


def detect_dead_code(graph: CodeGraph) -> list[str]:
    """Mark dead nodes/edges using forward reachability. Returns dead node IDs."""
    G = graph._g
    all_nodes = graph.all_nodes(include_proposed=False)
    node_ids = {n.id for n in all_nodes}

    # ── Build adjacency maps ─────────────────────────────────────
    # calls/reads: execution flow — "X invokes or references Y"
    call_fwd: dict[str, set[str]] = {}
    # contains: structural — "X's scope defines Y"
    contains_fwd: dict[str, set[str]] = {}
    # inherits: class → base class
    inherits_targets: dict[str, set[str]] = {}

    for u, v, d in G.edges(data=True):
        etype = d.get("edge_type")
        if etype in ("calls", "reads"):
            call_fwd.setdefault(u, set()).add(v)
        elif etype == "contains":
            contains_fwd.setdefault(u, set()).add(v)
        elif etype == "inherits":
            inherits_targets.setdefault(u, set()).add(v)

    # Which nodes are classes?
    class_ids = {n.id for n in all_nodes if n.symbol_type == SymbolType.CLASS}

    # Classes that inherit from serializable bases (Pydantic BaseModel,
    # dataclasses, etc.). Their fields are implicitly read by framework
    # serialization machinery — model_dump(), asdict(), etc.
    _SERIALIZABLE_BASES = {"BaseModel", "Model", "Schema"}
    serializable_class_ids: set[str] = set()
    for cls_id in class_ids:
        for base in inherits_targets.get(cls_id, ()):
            base_short = base.rsplit(".", 1)[-1] if "." in base else base
            if base_short in _SERIALIZABLE_BASES:
                serializable_class_ids.add(cls_id)

    # ── Identify production entry points ──────────────────────────
    # Modules are roots — their scope-level code runs on import.
    # Dunder methods and framework hooks are implicitly invoked.
    entry_points: set[str] = set()
    for n in all_nodes:
        if n.symbol_type == SymbolType.MODULE:
            entry_points.add(n.id)
        elif n.name in _EXEMPT_NAMES:
            entry_points.add(n.id)

    # ── Forward BFS from production entry points ──────────────────
    # When we reach a node, follow its calls/reads edges.
    # When we reach a CLASS, also follow its contains edges — instantiating
    # a class makes all its methods callable (handles visitor pattern,
    # framework base classes, etc.).
    reachable: set[str] = set()
    queue: deque[str] = deque(entry_points)
    while queue:
        nid = queue.popleft()
        if nid in reachable:
            continue
        reachable.add(nid)
        # Follow execution edges
        for target in call_fwd.get(nid, ()):
            if target not in reachable:
                queue.append(target)
        # If this is a class, reaching it means its methods are callable
        if nid in class_ids:
            for child in contains_fwd.get(nid, ()):
                if child not in reachable:
                    queue.append(child)

    # ── Mark unreachable functions/methods as dead ─────────────────
    dead: set[str] = set()
    for n in all_nodes:
        if n.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            continue
        # Test functions are not dead — they're tests
        if n.name.startswith("test_"):
            continue
        # Exempt names are never dead
        if n.name in _EXEMPT_NAMES:
            continue
        # If not reachable from any production entry point → dead
        if n.id not in reachable:
            n.is_dead = True
            dead.add(n.id)

    # ── Dead parameters (parent is dead, or never read) ───────────
    for n in all_nodes:
        if n.symbol_type != SymbolType.PARAMETER or n.name in ("self", "cls"):
            continue
        parent = n.id.rsplit(".", 1)[0] if "." in n.id else ""
        if parent in dead:
            n.is_dead = True
            dead.add(n.id)
            continue
        # If the parent function has incoming 'reads' edges, it's being
        # referenced as a value (passed as callback, stored in a variable,
        # etc.). Its parameters are contract-obligated by whatever receives it.
        if parent in G:
            has_reads_edge = any(
                d.get("edge_type") == "reads"
                for _, _, d in G.in_edges(parent, data=True)
            )
            if has_reads_edge:
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

    # ── Dead variables (written but never read) ───────────────────
    for n in all_nodes:
        if n.symbol_type != SymbolType.VARIABLE:
            continue
        if n.id not in G:
            continue
        # If this variable is a field on a serializable class (BaseModel etc.),
        # it's implicitly read by framework serialization machinery.
        parent = n.id.rsplit(".", 1)[0] if "." in n.id else ""
        if parent in serializable_class_ids:
            continue
        has_writer = any(d.get("edge_type") == "writes" for _, _, d in G.in_edges(n.id, data=True))
        has_reader = any(d.get("edge_type") == "reads" for _, _, d in G.in_edges(n.id, data=True))
        if has_writer and not has_reader:
            n.is_dead = True
            dead.add(n.id)

    # ── Dead returns (return value never read) ────────────────────
    for n in all_nodes:
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
            for e in graph.edges_from(n.id, EdgeType.RETURNS):
                e.is_dead = True

    # ── Dead imports ──────────────────────────────────────────────
    for e in graph.all_edges(include_proposed=False):
        if e.edge_type != EdgeType.IMPORTS:
            continue
        if e.target not in node_ids:
            is_prefix = any(nid.startswith(e.target) for nid in node_ids)
            if is_prefix:
                e.is_dead = True

    return list(dead)
