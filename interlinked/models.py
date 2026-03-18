"""Shared data models for Interlinked."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class SymbolType(str, enum.Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    PARAMETER = "parameter"


class EdgeType(str, enum.Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    READS = "reads"
    WRITES = "writes"
    CONTAINS = "contains"  # module contains class, class contains method, etc.
    PROPOSED = "proposed"
    RETURNS = "returns"  # function returns value to caller


class NodeData(BaseModel):
    id: str
    name: str
    qualified_name: str
    symbol_type: SymbolType
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    docstring: str | None = None
    signature: str | None = None
    is_dead: bool = False
    is_proposed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeData(BaseModel):
    source: str
    target: str
    edge_type: EdgeType
    is_dead: bool = False
    is_proposed: bool = False
    line: int | None = None  # line where the reference occurs
    metadata: dict[str, Any] = Field(default_factory=dict)


class ColorScheme(BaseModel):
    healthy: str = "#4a90d9"
    dead_link: str = "#e74c3c"
    proposed: str = "#2ecc71"
    highlighted: str = "#f1c40f"
    contains: str = "#95a5a6"
    inherits: str = "#9b59b6"
    imports: str = "#3498db"
    calls: str = "#e67e22"
    reads: str = "#1abc9c"
    writes: str = "#e91e63"
    module_bg: str = "#2c3e50"
    class_bg: str = "#34495e"
    function_bg: str = "#4a6fa5"
    variable_bg: str = "#7f8c8d"


class ViewContext(BaseModel):
    """Natural language context for what the user is currently looking at."""
    what: str = ""     # What is being shown
    why: str = ""      # Why this view was chosen
    where: str = ""    # Which part of the codebase (scope/modules/symbols)
    source: str = ""   # Who set this context: "llm", "command", "trace", ""


class ViewState(BaseModel):
    """Current state of the visualization — what the user sees."""
    zoom_level: str = "module"  # module, class, function
    focus_node: str | None = None
    focus_depth: int = 2
    visible_node_ids: list[str] = Field(default_factory=list)
    visible_edge_types: list[EdgeType] = Field(
        default_factory=lambda: list(EdgeType)
    )
    highlighted_node_ids: list[str] = Field(default_factory=list)
    highlighted_edge_ids: list[tuple[str, str]] = Field(default_factory=list)
    # Trace roles: node_id -> "origin" | "mutator" | "passthrough" | "destination"
    trace_node_roles: dict[str, str] = Field(default_factory=dict)
    # Trace edge roles: "src|tgt" -> "write" | "read"
    trace_edge_roles: dict[str, str] = Field(default_factory=dict)
    colors: ColorScheme = Field(default_factory=ColorScheme)
    show_dead: bool = True
    show_proposed: bool = True
    filter_expression: str | None = None
    context: ViewContext = Field(default_factory=ViewContext)


class GraphDelta(BaseModel):
    """Diff between two graph states — sent via SSE instead of full snapshots."""
    added_nodes: list[NodeData] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)
    updated_nodes: list[NodeData] = Field(default_factory=list)
    added_edges: list[EdgeData] = Field(default_factory=list)
    removed_edges: list[EdgeData] = Field(default_factory=list)
    view: ViewState
    layout_updates: dict[str, dict[str, float]] = Field(default_factory=dict)
    # If true, the client should discard its state and use this as a full snapshot
    full_snapshot: bool = False


class GraphSnapshot(BaseModel):
    """Serializable snapshot of the graph for the frontend."""
    nodes: list[NodeData]
    edges: list[EdgeData]
    view: ViewState
