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
        pos = _force_layout(nodes, edges, iterations=80)

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
    iterations: int = 80,
) -> dict[str, tuple[float, float]]:
    """Pure-Python Fruchterman-Reingold force-directed layout."""
    rng = random.Random(42)
    node_ids = [n.id for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}
    if n == 1:
        return {node_ids[0]: (0.0, 0.0)}

    idx = {nid: i for i, nid in enumerate(node_ids)}
    id_set = set(node_ids)

    # Initial random positions
    px = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    py = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # Build adjacency
    adj: list[list[int]] = [[] for _ in range(n)]
    for e in edges:
        if e.source in id_set and e.target in id_set:
            si, ti = idx[e.source], idx[e.target]
            adj[si].append(ti)
            adj[ti].append(si)

    area = 4.0 * n
    k = math.sqrt(area / max(n, 1))
    temp = 1.0

    for iteration in range(iterations):
        # Repulsive forces between all pairs
        dx = [0.0] * n
        dy = [0.0] * n

        for i in range(n):
            for j in range(i + 1, n):
                ddx = px[i] - px[j]
                ddy = py[i] - py[j]
                dist = math.sqrt(ddx * ddx + ddy * ddy) or 0.001
                force = (k * k) / dist
                fx = (ddx / dist) * force
                fy = (ddy / dist) * force
                dx[i] += fx
                dy[i] += fy
                dx[j] -= fx
                dy[j] -= fy

        # Attractive forces along edges
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                ddx = px[i] - px[j]
                ddy = py[i] - py[j]
                dist = math.sqrt(ddx * ddx + ddy * ddy) or 0.001
                force = (dist * dist) / k
                fx = (ddx / dist) * force
                fy = (ddy / dist) * force
                dx[i] -= fx
                dy[i] -= fy
                dx[j] += fx
                dy[j] += fy

        # Apply with temperature
        for i in range(n):
            disp = math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]) or 0.001
            scale = min(disp, temp) / disp
            px[i] += dx[i] * scale
            py[i] += dy[i] * scale

        temp *= 0.95  # cooling

    return {node_ids[i]: (px[i], py[i]) for i in range(n)}


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
