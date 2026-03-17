"""Similarity analysis — structural fingerprinting and duplicate detection.

Uses NetworkX graph algorithms (Jaccard coefficient on neighbor sets) combined
with AST structural features to detect similar/duplicate code.

Detects:
- Functions/methods with similar call patterns (Jaccard on callees)
- Similar read/write patterns (Jaccard on data-flow neighbors)
- Similar structural shape (AST node type distribution, nesting depth, control flow)
- Potential duplicated logic paths

Clustering uses nx.connected_components on a similarity threshold graph.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from interlinked.analyzer.graph import CodeGraph
from interlinked.models import NodeData, EdgeData, EdgeType, SymbolType


@dataclass
class StructuralFingerprint:
    """A normalized feature vector describing the shape of a code symbol."""
    node_id: str
    name: str
    qualified_name: str
    symbol_type: SymbolType
    # Structural features
    arg_count: int = 0
    arg_names: tuple[str, ...] = ()
    return_annotation: str = ""
    line_count: int = 0
    # AST shape
    ast_node_counts: dict[str, int] = field(default_factory=dict)
    max_nesting_depth: int = 0
    has_loops: bool = False
    has_conditionals: bool = False
    has_try_except: bool = False
    has_yield: bool = False
    has_await: bool = False
    # Graph shape
    callees: frozenset[str] = frozenset()
    callers: frozenset[str] = frozenset()
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    # Source context
    docstring: str = ""
    source_snippet: str = ""


def analyze_similarity(graph: CodeGraph) -> None:
    """Compute fingerprints for all functions/methods and store them on the nodes."""
    all_nodes = graph.all_nodes(include_proposed=False)
    functions = [
        n for n in all_nodes
        if n.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)
    ]

    for node in functions:
        fp = _compute_fingerprint(node, graph)
        node.metadata["fingerprint"] = _fingerprint_to_dict(fp)

    # Also fingerprint classes by their method signatures + shape
    classes = [n for n in all_nodes if n.symbol_type == SymbolType.CLASS]
    for node in classes:
        fp = _compute_class_fingerprint(node, graph)
        node.metadata["fingerprint"] = _fingerprint_to_dict(fp)


def find_duplicate_groups(
    graph: CodeGraph,
    threshold: float = 0.6,
    scope: str | None = None,
) -> list[dict]:
    """Find groups of structurally similar functions.

    Uses nx.connected_components on a similarity threshold graph to cluster.
    Returns a list of groups, each containing similar symbols with scores.
    """
    all_nodes = graph.all_nodes(include_proposed=False)
    targets = [
        n for n in all_nodes
        if n.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)
        and n.metadata.get("fingerprint")
    ]

    if scope:
        targets = [n for n in targets if n.qualified_name.startswith(scope)]

    # Pairwise comparison — build a similarity graph
    sim_graph = nx.Graph()
    pairs: dict[tuple[str, str], float] = {}
    for i, a in enumerate(targets):
        for j in range(i + 1, len(targets)):
            b = targets[j]
            score = _similarity_score(
                a.metadata["fingerprint"],
                b.metadata["fingerprint"],
            )
            if score >= threshold:
                sim_graph.add_edge(a.id, b.id, weight=score)
                pairs[(a.id, b.id)] = score

    # Cluster using nx.connected_components
    result = []
    for component in nx.connected_components(sim_graph):
        if len(component) < 2:
            continue
        members = []
        for nid in component:
            node = graph.get_node(nid)
            if node:
                members.append({
                    "id": node.id,
                    "name": node.name,
                    "qualified_name": node.qualified_name,
                    "file": node.file_path,
                    "lines": f"{node.line_start}-{node.line_end}",
                    "signature": node.signature or "",
                    "docstring": (node.docstring or "")[:200],
                })
        if len(members) >= 2:
            group_scores = [
                s for (a, b), s in pairs.items()
                if a in component and b in component
            ]
            avg_score = sum(group_scores) / len(group_scores) if group_scores else 0
            result.append({
                "similarity": round(avg_score, 3),
                "count": len(members),
                "members": members,
            })

    result.sort(key=lambda g: g["similarity"], reverse=True)
    return result


def find_similar_to(
    graph: CodeGraph,
    target_id: str,
    threshold: float = 0.5,
) -> list[dict]:
    """Find symbols similar to a specific target."""
    target_node = graph.get_node(target_id)
    if not target_node or not target_node.metadata.get("fingerprint"):
        return []

    target_fp = target_node.metadata["fingerprint"]
    all_nodes = graph.all_nodes(include_proposed=False)

    results = []
    for node in all_nodes:
        if node.id == target_id:
            continue
        if not node.metadata.get("fingerprint"):
            continue

        score = _similarity_score(target_fp, node.metadata["fingerprint"])
        if score >= threshold:
            results.append({
                "id": node.id,
                "name": node.name,
                "qualified_name": node.qualified_name,
                "symbol_type": node.symbol_type.value,
                "similarity": round(score, 3),
                "file": node.file_path,
                "signature": node.signature or "",
                "docstring": (node.docstring or "")[:200],
            })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


def get_rich_context(graph: CodeGraph, node: NodeData) -> dict:
    """Get rich context for a symbol: source, docstring, connections, fingerprint."""
    context: dict[str, Any] = {
        "id": node.id,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "symbol_type": node.symbol_type.value,
        "file": node.file_path,
        "lines": f"{node.line_start}-{node.line_end}" if node.line_start else None,
        "signature": node.signature,
        "docstring": node.docstring,
        "is_dead": node.is_dead,
    }

    # Source snippet
    if node.file_path and node.line_start and node.line_end:
        try:
            lines = Path(node.file_path).read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, node.line_start - 1)
            end = min(len(lines), node.line_end)
            context["source"] = "\n".join(lines[start:end])
        except Exception:
            context["source"] = None
    else:
        context["source"] = None

    # Comments above the function (look for comment block just before line_start)
    if node.file_path and node.line_start:
        try:
            lines = Path(node.file_path).read_text(encoding="utf-8", errors="replace").splitlines()
            comments = []
            i = node.line_start - 2  # 0-indexed, line before
            while i >= 0 and lines[i].strip().startswith("#"):
                comments.insert(0, lines[i].strip())
                i -= 1
            context["preceding_comments"] = "\n".join(comments) if comments else None
        except Exception:
            context["preceding_comments"] = None
    else:
        context["preceding_comments"] = None

    # Connections
    callers = graph.callers_of(node.id)
    callees = graph.callees_of(node.id)
    context["callers"] = [{"id": n.id, "name": n.name} for n in callers[:20]]
    context["callees"] = [{"id": n.id, "name": n.name} for n in callees[:20]]

    # Fingerprint
    context["fingerprint"] = node.metadata.get("fingerprint")

    return context


# ── Internal: fingerprint computation ────────────────────────────────

def _compute_fingerprint(node: NodeData, graph: CodeGraph) -> StructuralFingerprint:
    """Compute a structural fingerprint for a function/method."""
    fp = StructuralFingerprint(
        node_id=node.id,
        name=node.name,
        qualified_name=node.qualified_name,
        symbol_type=node.symbol_type,
        docstring=node.docstring or "",
    )

    # Parse the source to get AST shape
    if node.file_path and node.line_start and node.line_end:
        try:
            source = Path(node.file_path).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=node.file_path)
            func_node = _find_ast_node(tree, node.line_start)
            if func_node:
                _analyze_ast_shape(func_node, fp)
        except Exception:
            pass

    # Graph-based features — use resolved qualified names for accurate comparison
    G = graph._g
    if node.id in G:
        fp.callees = frozenset(
            v for _, v, d in G.out_edges(node.id, data=True)
            if d.get("edge_type") == "calls"
        )
        fp.callers = frozenset(
            u for u, _, d in G.in_edges(node.id, data=True)
            if d.get("edge_type") == "calls"
        )
        fp.reads = frozenset(
            v for _, v, d in G.out_edges(node.id, data=True)
            if d.get("edge_type") == "reads"
        )
        fp.writes = frozenset(
            v for _, v, d in G.out_edges(node.id, data=True)
            if d.get("edge_type") == "writes"
        )

    # Use PARAMETER child nodes from the graph (richer than re-parsing AST)
    param_nodes = [
        graph.get_node(v)
        for _, v, d in G.out_edges(node.id, data=True)
        if d.get("edge_type") == "contains"
        and graph.get_node(v)
        and graph.get_node(v).symbol_type == SymbolType.PARAMETER
    ] if node.id in G else []
    if param_nodes:
        fp.arg_count = len([p for p in param_nodes if p.name not in ("self", "cls")])
        fp.arg_names = tuple(p.name for p in param_nodes if p.name not in ("self", "cls"))

    # Line count
    if node.line_start and node.line_end:
        fp.line_count = node.line_end - node.line_start + 1

    # Source snippet for context
    if node.file_path and node.line_start and node.line_end:
        try:
            lines = Path(node.file_path).read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, node.line_start - 1)
            end = min(len(lines), min(node.line_end, node.line_start + 30))
            fp.source_snippet = "\n".join(lines[start:end])
        except Exception:
            pass

    return fp


def _compute_class_fingerprint(node: NodeData, graph: CodeGraph) -> StructuralFingerprint:
    """Compute a fingerprint for a class based on its methods and structure."""
    fp = StructuralFingerprint(
        node_id=node.id,
        name=node.name,
        qualified_name=node.qualified_name,
        symbol_type=node.symbol_type,
        docstring=node.docstring or "",
    )

    # Get method names and count
    methods = [
        e.target for e in graph.edges_from(node.id, EdgeType.CONTAINS)
        if graph.get_node(e.target) and
        graph.get_node(e.target).symbol_type == SymbolType.METHOD
    ]
    fp.arg_count = len(methods)
    fp.arg_names = tuple(sorted(m.split(".")[-1] for m in methods))

    # Aggregate callees/callers across all methods
    all_callees: set[str] = set()
    all_callers: set[str] = set()
    for mid in methods:
        for e in graph.edges_from(mid, EdgeType.CALLS):
            all_callees.add(e.target.split(".")[-1])
        for e in graph.edges_to(mid, EdgeType.CALLS):
            all_callers.add(e.source.split(".")[-1])

    fp.callees = frozenset(all_callees)
    fp.callers = frozenset(all_callers)

    if node.line_start and node.line_end:
        fp.line_count = node.line_end - node.line_start + 1

    return fp


def _find_ast_node(tree: ast.Module, target_line: int) -> ast.AST | None:
    """Find the AST node at a specific line number."""
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and node.lineno == target_line:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node
    return None


def _analyze_ast_shape(node: ast.AST, fp: StructuralFingerprint) -> None:
    """Analyze the AST shape of a function."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        fp.arg_count = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        fp.arg_names = tuple(a.arg for a in args.args if a.arg != "self")
        if node.returns:
            fp.return_annotation = ast.dump(node.returns)
        fp.has_await = isinstance(node, ast.AsyncFunctionDef)

    # Count AST node types and detect patterns
    node_counts: Counter[str] = Counter()
    max_depth = [0]

    def _walk_depth(n: ast.AST, depth: int) -> None:
        node_counts[type(n).__name__] += 1
        max_depth[0] = max(max_depth[0], depth)

        if isinstance(n, (ast.For, ast.While, ast.AsyncFor)):
            fp.has_loops = True
        if isinstance(n, (ast.If, ast.IfExp)):
            fp.has_conditionals = True
        if isinstance(n, (ast.Try, ast.ExceptHandler)):
            fp.has_try_except = True
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            fp.has_yield = True
        if isinstance(n, (ast.Await,)):
            fp.has_await = True

        for child in ast.iter_child_nodes(n):
            _walk_depth(child, depth + 1)

    _walk_depth(node, 0)
    fp.ast_node_counts = dict(node_counts)
    fp.max_nesting_depth = max_depth[0]


# ── Internal: similarity scoring ─────────────────────────────────────

def _similarity_score(fp_a: dict, fp_b: dict) -> float:
    """Compute similarity between two fingerprint dicts. Returns 0.0-1.0."""
    scores: list[tuple[float, float]] = []  # (score, weight)

    # Argument pattern similarity
    args_a = set(fp_a.get("arg_names", []))
    args_b = set(fp_b.get("arg_names", []))
    if args_a or args_b:
        arg_sim = len(args_a & args_b) / max(len(args_a | args_b), 1)
        scores.append((arg_sim, 2.0))

    # Arg count similarity
    ac_a = fp_a.get("arg_count", 0)
    ac_b = fp_b.get("arg_count", 0)
    if ac_a + ac_b > 0:
        scores.append((1.0 - abs(ac_a - ac_b) / max(ac_a + ac_b, 1), 1.0))

    # Line count similarity
    lc_a = fp_a.get("line_count", 0)
    lc_b = fp_b.get("line_count", 0)
    if lc_a > 0 and lc_b > 0:
        scores.append((1.0 - abs(lc_a - lc_b) / max(lc_a, lc_b), 1.0))

    # AST shape similarity (cosine similarity of node type counts)
    ast_a = fp_a.get("ast_node_counts", {})
    ast_b = fp_b.get("ast_node_counts", {})
    if ast_a and ast_b:
        ast_sim = _cosine_similarity(ast_a, ast_b)
        scores.append((ast_sim, 3.0))  # Heavy weight — this is the shape

    # Control flow pattern match
    flow_features = ["has_loops", "has_conditionals", "has_try_except", "has_yield", "has_await"]
    flow_match = sum(1 for f in flow_features if fp_a.get(f) == fp_b.get(f))
    scores.append((flow_match / len(flow_features), 1.5))

    # Callee overlap (what they call)
    callees_a = set(fp_a.get("callees", []))
    callees_b = set(fp_b.get("callees", []))
    if callees_a or callees_b:
        callee_sim = len(callees_a & callees_b) / max(len(callees_a | callees_b), 1)
        scores.append((callee_sim, 2.5))  # Strong signal

    # Read/write variable overlap
    reads_a = set(fp_a.get("reads", []))
    reads_b = set(fp_b.get("reads", []))
    if reads_a or reads_b:
        read_sim = len(reads_a & reads_b) / max(len(reads_a | reads_b), 1)
        scores.append((read_sim, 1.5))

    # Nesting depth similarity
    nd_a = fp_a.get("max_nesting_depth", 0)
    nd_b = fp_b.get("max_nesting_depth", 0)
    if nd_a > 0 or nd_b > 0:
        scores.append((1.0 - abs(nd_a - nd_b) / max(nd_a, nd_b, 1), 0.5))

    if not scores:
        return 0.0

    total_weight = sum(w for _, w in scores)
    weighted_sum = sum(s * w for s, w in scores)
    return weighted_sum / total_weight


def _cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity between two sparse vectors."""
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _fingerprint_to_dict(fp: StructuralFingerprint) -> dict:
    """Convert a fingerprint to a serializable dict."""
    return {
        "arg_count": fp.arg_count,
        "arg_names": list(fp.arg_names),
        "return_annotation": fp.return_annotation,
        "line_count": fp.line_count,
        "ast_node_counts": fp.ast_node_counts,
        "max_nesting_depth": fp.max_nesting_depth,
        "has_loops": fp.has_loops,
        "has_conditionals": fp.has_conditionals,
        "has_try_except": fp.has_try_except,
        "has_yield": fp.has_yield,
        "has_await": fp.has_await,
        "callees": list(fp.callees),
        "callers": list(fp.callers),
        "reads": list(fp.reads),
        "writes": list(fp.writes),
    }
