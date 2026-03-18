"""Tree-sitter Python language adapter.

Note: For Python projects, the ast-based parser in analyzer/parser.py is
preferred — it provides deep type inference, alias tracking, and generator
element typing that tree-sitter cannot. This adapter exists for completeness
and for mixed-language projects where Python files need to coexist with
other languages in the same graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from interlinked.analyzer.treesitter.adapter import LanguageAdapter, ScopeRule
from interlinked.analyzer.treesitter import registry

if TYPE_CHECKING:
    from tree_sitter import Language, Node


class PythonAdapter(LanguageAdapter):

    @property
    def name(self) -> str:
        return "python"

    @property
    def extensions(self) -> tuple[str, ...]:
        return (".py",)

    def grammar(self) -> Language:
        import tree_sitter_python as tspython
        from tree_sitter import Language
        return Language(tspython.language())

    @property
    def scope_rules(self) -> dict[str, ScopeRule]:
        return {
            "function_definition": ScopeRule("function", is_method_if_nested=True),
            "class_definition": ScopeRule("class"),
        }

    @property
    def call_query(self) -> str:
        return "(call function: (_) @callee)"

    @property
    def import_query(self) -> str:
        return """
        (import_from_statement
            module_name: (dotted_name) @source
            name: (dotted_name) @name)
        (import_statement
            name: (dotted_name) @source)
        """

    @property
    def inheritance_query(self) -> str:
        return """
        (class_definition
            name: (identifier) @class_name
            superclasses: (argument_list
                (identifier) @base))
        """

    def extract_callee(self, node: Node) -> str:
        # call function can be identifier, attribute, or subscript
        text = node.text.decode("utf-8") if node.text else ""
        return text

    def module_name_from_path(self, rel_path: str) -> str:
        from pathlib import Path
        parts = list(Path(rel_path).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts) if parts else "__root__"

    def skip_directory(self, dirname: str) -> bool:
        return dirname in {
            ".git", "__pycache__", "node_modules",
            ".venv", "venv", "env", ".env",
            "dist", "build", ".build",
            ".tox", ".nox", ".eggs",
            ".mypy_cache", ".ruff_cache", ".pytest_cache",
        }

    def is_test_file(self, rel_path: str) -> bool:
        from pathlib import Path
        stem = Path(rel_path).stem
        return stem.startswith("test_") or stem.endswith("_test")


# Auto-register on import
registry.register(PythonAdapter())
