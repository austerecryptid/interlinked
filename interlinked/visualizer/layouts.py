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
    width: float | None = None,
    height: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compute x,y positions for each node.

    Canvas scales with sqrt(node count) so spacing stays consistent
    whether you have 20 nodes or 65,000.

    Returns: {node_id: {"x": float, "y": float}}
    """
    n = len(nodes)
    scale = max(1.0, math.sqrt(n)) * 2  # generous: ~6x at 10 nodes, ~32x at 250, ~510x at 65k
    if width is None:
        width = 1200 * scale
    if height is None:
        height = 800 * scale
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
    """Fast scatter layout — O(n). Real layout is done by FA2 on the frontend."""
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0].id: (0.0, 0.0)}

    # Place nodes in a spiral so they start spread out and non-overlapping.
    # FA2 on the frontend will refine from here.
    pos: dict[str, tuple[float, float]] = {}
    golden_angle = math.pi * (3 - math.sqrt(5))  # ~137.5°
    for i, node in enumerate(nodes):
        r = math.sqrt(i + 1) / math.sqrt(n)  # radius grows with sqrt
        theta = i * golden_angle
        pos[node.id] = (r * math.cos(theta), r * math.sin(theta))
    return pos


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
