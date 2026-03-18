"""Generic tree-sitter CST walker.

Walks a tree-sitter CST using a LanguageAdapter to emit NodeData/EdgeData
in the same format as the Python ast-based parser. This enables all
downstream systems (CodeGraph, QueryEngine, dead code detection,
similarity, visualization) to work with any language.

The walker performs two passes:
  1. **Structure pass** — cursor-based walk to extract scopes (functions,
     classes, methods) and build the CONTAINS tree.
  2. **Edge pass** — S-expression queries to extract CALLS, IMPORTS, and
     INHERITS edges within each scope.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from interlinked.models import EdgeData, EdgeType, NodeData, SymbolType

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

from interlinked.analyzer.treesitter.adapter import LanguageAdapter

# Map adapter symbol type strings to our SymbolType enum
_SYMBOL_MAP = {
    "function": SymbolType.FUNCTION,
    "method": SymbolType.METHOD,
    "class": SymbolType.CLASS,
    "module": SymbolType.MODULE,
    "variable": SymbolType.VARIABLE,
}


def parse_project_treesitter(
    root: str | Path,
    adapter: LanguageAdapter,
) -> tuple[list[NodeData], list[EdgeData]]:
    """Walk a project directory and extract symbols and edges using tree-sitter.

    This is the multi-language equivalent of ``parser.parse_project``.
    It produces the same NodeData/EdgeData output, so it plugs directly
    into CodeGraph.build_from().

    Args:
        root: Project root directory.
        adapter: Language adapter defining how to parse this language.

    Returns:
        (nodes, edges) ready for CodeGraph.build_from().
    """
    root = Path(root).resolve()
    nodes: list[NodeData] = []
    edges: list[EdgeData] = []

    for ext in adapter.extensions:
        for source_file in sorted(root.rglob(f"*{ext}")):
            # Skip excluded directories
            try:
                rel = source_file.relative_to(root)
            except ValueError:
                continue
            if any(adapter.skip_directory(part) for part in rel.parts[:-1]):
                continue

            file_nodes, file_edges = parse_file_treesitter(
                source_file, root, adapter,
            )
            nodes.extend(file_nodes)
            edges.extend(file_edges)

    # Optional: resolve unqualified targets via stack graphs
    try:
        from interlinked.analyzer.treesitter.resolver import is_available, resolve_edges
        if is_available():
            edges = resolve_edges(str(root), adapter.name, nodes, edges)
    except ImportError:
        pass

    return nodes, edges


def parse_file_treesitter(
    file_path: str | Path,
    project_root: str | Path,
    adapter: LanguageAdapter,
) -> tuple[list[NodeData], list[EdgeData]]:
    """Parse a single file using tree-sitter and a language adapter.

    Args:
        file_path: Absolute path to the source file.
        project_root: Project root for computing module names.
        adapter: Language adapter.

    Returns:
        (nodes, edges) for this file.
    """
    from tree_sitter import Parser

    file_path = Path(file_path)
    project_root = Path(project_root)

    try:
        source = file_path.read_bytes()
    except OSError:
        return [], []

    rel_path = str(file_path.relative_to(project_root))
    module_qname = adapter.module_name_from_path(rel_path)

    # Parse with tree-sitter
    language = adapter.grammar()
    parser = Parser(language)
    tree = parser.parse(source)

    source_text = source.decode("utf-8", errors="replace")
    line_count = source_text.count("\n") + 1

    nodes: list[NodeData] = []
    edges: list[EdgeData] = []
    node_ids: set[str] = set()

    # Module node
    nodes.append(NodeData(
        id=module_qname,
        name=module_qname.rsplit(".", 1)[-1],
        qualified_name=module_qname,
        symbol_type=SymbolType.MODULE,
        file_path=str(file_path),
        line_start=1,
        line_end=line_count,
        docstring=_extract_docstring(tree.root_node, adapter),
    ))
    node_ids.add(module_qname)

    # Pass 1: Structure — walk CST to find scope-creating nodes
    _walk_structure(
        tree.root_node, module_qname, str(file_path),
        adapter, nodes, edges, node_ids,
    )

    # Pass 2: Edges — run queries to extract calls, imports, inheritance
    _extract_edges(
        tree, module_qname, str(file_path), source_text,
        adapter, language, nodes, edges, node_ids,
    )

    return nodes, edges


# ---------------------------------------------------------------------------
# Pass 1: Structure walk
# ---------------------------------------------------------------------------

def _walk_structure(
    root: Node,
    module_qname: str,
    file_path: str,
    adapter: LanguageAdapter,
    nodes: list[NodeData],
    edges: list[EdgeData],
    node_ids: set[str],
) -> None:
    """Recursively walk the CST to extract scope-creating definitions."""
    scope_rules = adapter.scope_rules

    # Stack of (node, parent_qname) pairs for iterative DFS
    stack: list[tuple[Node, str]] = [(root, module_qname)]

    while stack:
        node, parent_qname = stack.pop()

        node_type = node.type
        if node_type in scope_rules:
            rule = scope_rules[node_type]

            # Extract the name
            name_node = node.child_by_field_name(rule.name_field)
            if not name_node:
                # Push children with same parent scope
                for child in reversed(node.children):
                    stack.append((child, parent_qname))
                continue

            name = adapter.extract_name(name_node)
            if not name:
                for child in reversed(node.children):
                    stack.append((child, parent_qname))
                continue

            qname = f"{parent_qname}.{name}"

            # Determine symbol type — promote to METHOD if inside a class
            sym_type_str = rule.symbol_type
            if rule.is_method_if_nested and _is_class_scope(parent_qname, nodes):
                sym_type_str = "method"

            sym_type = _SYMBOL_MAP.get(sym_type_str, SymbolType.FUNCTION)

            # Extract signature (first line of the definition)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            signature = _extract_signature(node, adapter)

            if qname not in node_ids:
                nodes.append(NodeData(
                    id=qname,
                    name=name,
                    qualified_name=qname,
                    symbol_type=sym_type,
                    file_path=file_path,
                    line_start=start_line,
                    line_end=end_line,
                    docstring=_extract_docstring(node, adapter),
                    signature=signature,
                ))
                node_ids.add(qname)

            # CONTAINS edge
            edges.append(EdgeData(
                source=parent_qname,
                target=qname,
                edge_type=EdgeType.CONTAINS,
                line=start_line,
            ))

            # Children of this scope use it as parent
            for child in reversed(node.children):
                stack.append((child, qname))
        else:
            # Non-scope node — children inherit parent scope
            for child in reversed(node.children):
                stack.append((child, parent_qname))


def _is_class_scope(qname: str, nodes: list[NodeData]) -> bool:
    """Check if qname refers to a CLASS node."""
    return any(n.id == qname and n.symbol_type == SymbolType.CLASS for n in nodes)


# ---------------------------------------------------------------------------
# Pass 2: Edge extraction via queries
# ---------------------------------------------------------------------------

def _extract_edges(
    tree: Tree,
    module_qname: str,
    file_path: str,
    source_text: str,
    adapter: LanguageAdapter,
    language: object,
    nodes: list[NodeData],
    edges: list[EdgeData],
    node_ids: set[str],
) -> None:
    """Run S-expression queries to extract CALLS, IMPORTS, INHERITS edges."""
    from tree_sitter import Query, QueryCursor

    # Build a position-to-scope lookup for assigning edges to their enclosing scope
    scope_ranges = _build_scope_ranges(nodes, file_path)

    # ── CALLS ────────────────────────────────────────────────────────
    if adapter.call_query:
        try:
            query = Query(language, adapter.call_query)
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            callee_nodes = captures.get("callee", [])
            for callee_node in callee_nodes:
                callee_name = adapter.extract_callee(callee_node)
                if not callee_name:
                    continue
                line = callee_node.start_point[0] + 1
                caller = _scope_at(line, scope_ranges, module_qname)
                edges.append(EdgeData(
                    source=caller,
                    target=callee_name,
                    edge_type=EdgeType.CALLS,
                    line=line,
                ))
        except Exception:
            pass  # Query syntax errors for a specific language — skip gracefully

    # ── IMPORTS ──────────────────────────────────────────────────────
    if adapter.import_query:
        try:
            query = Query(language, adapter.import_query)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            for _pattern_idx, match in matches:
                source_nodes = match.get("source", [])
                name_nodes = match.get("name", [])
                source_node = source_nodes[0] if source_nodes else None
                name_node = name_nodes[0] if name_nodes else None
                if source_node:
                    target = adapter.extract_import_target(source_node, name_node)
                    line = source_node.start_point[0] + 1
                    edges.append(EdgeData(
                        source=module_qname,
                        target=target,
                        edge_type=EdgeType.IMPORTS,
                        line=line,
                    ))
        except Exception:
            pass

    # ── INHERITS ─────────────────────────────────────────────────────
    if adapter.inheritance_query:
        try:
            query = Query(language, adapter.inheritance_query)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            for _pattern_idx, match in matches:
                class_nodes = match.get("class_name", [])
                base_nodes = match.get("base", [])
                if class_nodes and base_nodes:
                    class_name = adapter.extract_name(class_nodes[0])
                    for base_node in base_nodes:
                        base_name = adapter.extract_name(base_node)
                        if class_name and base_name:
                            line = base_node.start_point[0] + 1
                            # Find the qualified class name
                            class_qname = _find_qname(class_name, nodes, module_qname)
                            edges.append(EdgeData(
                                source=class_qname,
                                target=base_name,
                                edge_type=EdgeType.INHERITS,
                                line=line,
                            ))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scope_ranges(
    nodes: list[NodeData], file_path: str,
) -> list[tuple[int, int, str]]:
    """Build sorted list of (start_line, end_line, qname) for scope lookup.

    Sorted by start_line descending so the innermost scope is found first.
    """
    ranges = []
    for n in nodes:
        if n.file_path == file_path and n.symbol_type in (
            SymbolType.FUNCTION, SymbolType.METHOD, SymbolType.CLASS,
        ):
            ranges.append((n.line_start, n.line_end, n.id))
    # Sort by start_line descending — innermost scope first
    ranges.sort(key=lambda r: -r[0])
    return ranges


def _scope_at(line: int, scope_ranges: list[tuple[int, int, str]], default: str) -> str:
    """Find the innermost scope containing the given line number."""
    for start, end, qname in scope_ranges:
        if start <= line <= end:
            return qname
    return default


def _find_qname(name: str, nodes: list[NodeData], module_qname: str) -> str:
    """Find the fully qualified name for a short name within a module."""
    candidate = f"{module_qname}.{name}"
    for n in nodes:
        if n.id == candidate:
            return candidate
    # Fallback: return module-qualified
    return candidate


def _extract_docstring(node: Node, adapter: LanguageAdapter) -> str | None:
    """Try to extract a docstring from the first child of a scope node.

    Works for languages with string-literal docstrings (Python, Julia).
    Returns None for languages that use comment-based docs.
    """
    body = node.child_by_field_name("body")
    target = body if body else node

    for child in target.children:
        if child.type in ("expression_statement",):
            for inner in child.children:
                if inner.type in ("string", "string_literal"):
                    text = inner.text.decode("utf-8", errors="replace") if inner.text else ""
                    # Strip quotes
                    for quote in ('"""', "'''", '"', "'"):
                        if text.startswith(quote) and text.endswith(quote):
                            return text[len(quote):-len(quote)].strip()
                    return text
            break
        # Skip comments and decorators at the top
        if child.type not in ("comment", "decorator", "decorated_definition"):
            break

    return None


def _extract_signature(node: Node, adapter: LanguageAdapter) -> str | None:
    """Extract a function/class signature from the CST node.

    Returns the text from the start of the definition to the colon/brace.
    """
    # Get text of the first line
    if node.text:
        first_line = node.text.decode("utf-8", errors="replace").split("\n")[0]
        return first_line.rstrip()
    return None
