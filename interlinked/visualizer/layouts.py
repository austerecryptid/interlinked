"""Graph layout algorithms — compute node positions for the frontend.

Uses a pure-Python force-directed layout so numpy is not required.
"""

from __future__ import annotations

import math
import random
from typing import Any

from interlinked.models import NodeData, EdgeData, SymbolType


def _circular_layout(nodes: list[NodeData]) -> dict[str, tuple[float, float]]:
    """Arrange nodes in a circle."""
    n = len(nodes)
    if n == 0:
        return {}
    pos: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        pos[node.id] = (math.cos(angle), math.sin(angle))
    return pos


def compute_layout(
    nodes: list[NodeData],
    edges: list[EdgeData],
    algorithm: str = "force",
    width: float = 1200,
    height: float = 800,
) -> dict[str, dict[str, float]]:
    """Compute x,y positions for each node.

    Returns: {node_id: {"x": float, "y": float}}
    """
    if not nodes:
        return {}

    if algorithm == "hierarchical":
        pos = _hierarchical_layout(nodes, edges)
    elif algorithm == "circular":
        pos = _circular_layout(nodes)
    else:
        pos = _force_layout(nodes, edges)

    # Scale to canvas dimensions
    result: dict[str, dict[str, float]] = {}
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max_x - min_x or 1
        range_y = max_y - min_y or 1
        margin = 80

        for nid, (x, y) in pos.items():
            result[nid] = {
                "x": margin + ((x - min_x) / range_x) * (width - 2 * margin),
                "y": margin + ((y - min_y) / range_y) * (height - 2 * margin),
            }

    return result


def _force_layout(
    nodes: list[NodeData],
    edges: list[EdgeData],
) -> dict[str, tuple[float, float]]:
    """Force-directed layout using networkx (numpy-accelerated)."""
    import networkx as nx

    node_ids = [n.id for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}
    if n == 1:
        return {node_ids[0]: (0.0, 0.0)}

    id_set = set(node_ids)

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for e in edges:
        if e.source in id_set and e.target in id_set:
            G.add_edge(e.source, e.target)

    # Scale iterations: fewer for larger graphs (still fast with numpy)
    iters = 50 if n < 1000 else 20 if n < 10000 else 10
    k = math.sqrt(4.0 / max(n, 1))

    pos = nx.spring_layout(G, k=k, iterations=iters, seed=42)
    return {nid: (float(xy[0]), float(xy[1])) for nid, xy in pos.items()}


def _hierarchical_layout(
    nodes: list[NodeData], edges: list[EdgeData]
) -> dict[str, tuple[float, float]]:
    """Simple hierarchical layout: modules at top, classes below, functions at bottom."""
    layers: dict[SymbolType, list[str]] = {
        SymbolType.MODULE: [],
        SymbolType.CLASS: [],
        SymbolType.FUNCTION: [],
        SymbolType.METHOD: [],
        SymbolType.VARIABLE: [],
    }

    for n in nodes:
        layers[n.symbol_type].append(n.id)

    pos: dict[str, tuple[float, float]] = {}
    layer_order = [
        SymbolType.MODULE,
        SymbolType.CLASS,
        SymbolType.FUNCTION,
        SymbolType.METHOD,
        SymbolType.VARIABLE,
    ]

    y = 0.0
    for sym_type in layer_order:
        ids = layers[sym_type]
        if not ids:
            continue
        for i, nid in enumerate(ids):
            x = (i + 1) / (len(ids) + 1)
            pos[nid] = (x, y)
        y += 1.0

    return pos
