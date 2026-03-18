"""Language adapter interface for tree-sitter based parsing.

Each supported language provides a thin adapter that maps tree-sitter CST
node types to interlinked's NodeData/EdgeData models. The adapter defines:

- Which file extensions the language handles
- How to load the tree-sitter grammar
- Which CST node types create scopes (functions, classes, methods)
- S-expression queries for extracting calls, imports, and inheritance
- Name extraction helpers for language-specific AST shapes

The generic walker in walker.py uses these adapters to parse any language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Language, Node


@dataclass
class ScopeRule:
    """Maps a tree-sitter node type to an interlinked symbol type.

    Attributes:
        symbol_type: The interlinked SymbolType string ("function", "method", "class").
        name_field: The tree-sitter field name containing the symbol's name.
            Usually "name" for most languages.
        is_method_if_nested: If True, this scope becomes a METHOD when nested
            inside a CLASS scope (e.g. function_definition inside class_definition).
    """
    symbol_type: str  # "function", "method", "class"
    name_field: str = "name"
    is_method_if_nested: bool = False


class LanguageAdapter(ABC):
    """Abstract base class for tree-sitter language adapters.

    Subclasses must implement the abstract methods/properties. The generic
    walker calls these to extract structure and edges from the CST.

    Minimal adapter example (pseudocode)::

        class GoAdapter(LanguageAdapter):
            name = "go"
            extensions = (".go",)
            scope_rules = {
                "function_declaration": ScopeRule("function"),
                "method_declaration": ScopeRule("method"),
                "type_declaration": ScopeRule("class"),
            }
            ...
    """

    # ── Identity ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short language identifier, e.g. 'python', 'typescript', 'go'."""

    @property
    @abstractmethod
    def extensions(self) -> tuple[str, ...]:
        """File extensions handled by this adapter, e.g. ('.ts', '.tsx')."""

    # ── Grammar ──────────────────────────────────────────────────────

    @abstractmethod
    def grammar(self) -> Language:
        """Return the tree-sitter Language object for this language.

        Implementations should import the grammar package lazily::

            def grammar(self) -> Language:
                import tree_sitter_python as tspython
                from tree_sitter import Language
                return Language(tspython.language())
        """

    # ── Scope rules ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def scope_rules(self) -> dict[str, ScopeRule]:
        """Map tree-sitter node types to scope-creating symbol types.

        The walker uses this to identify function/class/method definitions
        and build the scope tree (CONTAINS edges).

        Example::

            {
                "function_definition": ScopeRule("function", is_method_if_nested=True),
                "class_definition": ScopeRule("class"),
            }
        """

    # ── Queries ──────────────────────────────────────────────────────
    # S-expression queries run against the CST to extract edges.
    # Each query should capture nodes with specific @names that the
    # walker knows how to process.

    @property
    @abstractmethod
    def call_query(self) -> str:
        """S-expression query capturing call expressions.

        Must capture:
            @callee — the node containing the function/method being called

        Example (Python)::

            (call function: (_) @callee)
        """

    @property
    @abstractmethod
    def import_query(self) -> str:
        """S-expression query capturing import statements.

        Must capture:
            @source — the module being imported from
            @name — the specific name being imported (optional)
            @alias — the local alias (optional)

        Return empty string if the language has no import system.
        """

    @property
    def inheritance_query(self) -> str:
        """S-expression query capturing class inheritance.

        Must capture:
            @class_name — the class being defined
            @base — each base class

        Return empty string if not applicable. Default: empty.
        """
        return ""

    # ── Name extraction ──────────────────────────────────────────────

    def extract_name(self, node: Node) -> str:
        """Extract a human-readable name from a CST node.

        Default: return the node's text decoded as UTF-8.
        Override for languages where the name requires processing
        (e.g. stripping type parameters, handling destructuring).
        """
        return node.text.decode("utf-8") if node.text else ""

    def extract_callee(self, node: Node) -> str:
        """Extract the callee name from a call expression capture.

        Default: return the node's text. Override for languages where
        call targets need special handling (e.g. method chains,
        optional chaining, generic type arguments).
        """
        return node.text.decode("utf-8") if node.text else ""

    def extract_import_target(self, source_node: Node, name_node: Node | None) -> str:
        """Build the full import target from captured nodes.

        Args:
            source_node: The module/package being imported from.
            name_node: The specific name imported (None for whole-module imports).

        Returns:
            Dotted qualified name of the import target.
        """
        source = source_node.text.decode("utf-8") if source_node.text else ""
        if name_node:
            name = name_node.text.decode("utf-8") if name_node.text else ""
            return f"{source}.{name}" if source else name
        return source

    # ── Module naming ────────────────────────────────────────────────

    @abstractmethod
    def module_name_from_path(self, rel_path: str) -> str:
        """Convert a relative file path to a dotted module/package name.

        Example (Python): 'interlinked/analyzer/parser.py' -> 'interlinked.analyzer.parser'
        Example (TypeScript): 'src/utils/graph.ts' -> 'src.utils.graph'
        Example (Go): 'cmd/server/main.go' -> 'cmd.server.main'
        """

    # ── Optional hooks ───────────────────────────────────────────────

    def skip_directory(self, dirname: str) -> bool:
        """Return True if this directory should be skipped during project walk.

        Default skips common non-source directories.
        """
        return dirname in {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".tox", ".nox", ".eggs",
            ".mypy_cache", ".ruff_cache", ".pytest_cache",
        }

    def is_test_file(self, rel_path: str) -> bool:
        """Return True if this file is a test file.

        Default: filename starts with 'test_' or ends with '_test'.
        Override for language-specific conventions (e.g. *.spec.ts).
        """
        from pathlib import Path
        stem = Path(rel_path).stem
        return stem.startswith("test_") or stem.endswith("_test")
