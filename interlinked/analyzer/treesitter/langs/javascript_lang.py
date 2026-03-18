"""Tree-sitter JavaScript and TypeScript language adapter.

Handles .js, .jsx, .ts, .tsx files. Uses tree-sitter-javascript for JS/JSX
and tree-sitter-typescript for TS/TSX. Both share the same adapter logic
since TypeScript's grammar is a superset of JavaScript's.

Extracts:
- Functions (function declarations, arrow functions, function expressions)
- Classes (class declarations, class expressions)
- Methods (method definitions inside classes)
- Calls (call expressions, new expressions)
- Imports (ES6 import statements, require() calls)
- Inheritance (extends clauses)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from interlinked.analyzer.treesitter.adapter import LanguageAdapter, ScopeRule
from interlinked.analyzer.treesitter import registry

if TYPE_CHECKING:
    from tree_sitter import Language, Node


class JavaScriptAdapter(LanguageAdapter):

    @property
    def name(self) -> str:
        return "javascript"

    @property
    def extensions(self) -> tuple[str, ...]:
        return (".js", ".jsx", ".mjs", ".cjs")

    def grammar(self) -> Language:
        import tree_sitter_javascript as tsjs
        from tree_sitter import Language
        return Language(tsjs.language())

    @property
    def scope_rules(self) -> dict[str, ScopeRule]:
        return {
            "function_declaration": ScopeRule("function"),
            "generator_function_declaration": ScopeRule("function"),
            "class_declaration": ScopeRule("class"),
            "method_definition": ScopeRule("method"),
        }

    @property
    def call_query(self) -> str:
        return """
        (call_expression
            function: (_) @callee)
        (new_expression
            constructor: (_) @callee)
        """

    @property
    def import_query(self) -> str:
        # ES6: import { foo } from 'bar'
        # CommonJS: require('bar') handled as a call, not here
        return """
        (import_statement
            source: (string) @source)
        """

    @property
    def inheritance_query(self) -> str:
        return """
        (class_declaration
            name: (identifier) @class_name
            (class_heritage
                (extends_clause
                    value: (_) @base)))
        """

    def extract_name(self, node: Node) -> str:
        text = node.text.decode("utf-8") if node.text else ""
        return text

    def extract_callee(self, node: Node) -> str:
        text = node.text.decode("utf-8") if node.text else ""
        # Strip optional chaining: foo?.bar() -> foo.bar
        return text.replace("?.", ".")

    def extract_import_target(self, source_node: Node, name_node: Node | None) -> str:
        # Import source is a string literal — strip quotes
        text = source_node.text.decode("utf-8") if source_node.text else ""
        for quote in ('"', "'", '`'):
            text = text.strip(quote)
        # Convert relative paths to dotted names: ./utils/graph -> utils.graph
        text = text.lstrip("./")
        return text.replace("/", ".")

    def module_name_from_path(self, rel_path: str) -> str:
        from pathlib import Path
        p = Path(rel_path)
        # Strip all JS/TS extensions
        stem = p.stem
        if stem == "index":
            # src/utils/index.ts -> src.utils
            parts = list(p.parent.parts)
        else:
            parts = list(p.parent.parts) + [stem]
        return ".".join(parts) if parts else "__root__"

    def skip_directory(self, dirname: str) -> bool:
        return dirname in {
            ".git", "node_modules", "dist", "build", ".next",
            ".nuxt", ".svelte-kit", "coverage", ".turbo",
            "__pycache__", ".venv", "venv",
        }

    def is_test_file(self, rel_path: str) -> bool:
        from pathlib import Path
        stem = Path(rel_path).stem
        return (
            stem.endswith(".test")
            or stem.endswith(".spec")
            or stem.startswith("test_")
            or stem.endswith("_test")
        )


class TypeScriptAdapter(LanguageAdapter):

    @property
    def name(self) -> str:
        return "typescript"

    @property
    def extensions(self) -> tuple[str, ...]:
        return (".ts", ".tsx")

    def grammar(self) -> Language:
        import tree_sitter_typescript as tsts
        from tree_sitter import Language
        # tree-sitter-typescript provides both typescript and tsx
        return Language(tsts.language_typescript())

    @property
    def scope_rules(self) -> dict[str, ScopeRule]:
        return {
            "function_declaration": ScopeRule("function"),
            "generator_function_declaration": ScopeRule("function"),
            "class_declaration": ScopeRule("class"),
            "method_definition": ScopeRule("method"),
            # TypeScript-specific
            "interface_declaration": ScopeRule("class"),
            "type_alias_declaration": ScopeRule("class"),
            "enum_declaration": ScopeRule("class"),
        }

    @property
    def call_query(self) -> str:
        return """
        (call_expression
            function: (_) @callee)
        (new_expression
            constructor: (_) @callee)
        """

    @property
    def import_query(self) -> str:
        return """
        (import_statement
            source: (string) @source)
        """

    @property
    def inheritance_query(self) -> str:
        return """
        (class_declaration
            name: (type_identifier) @class_name
            (class_heritage
                (extends_clause
                    value: (_) @base)))
        (class_declaration
            name: (type_identifier) @class_name
            (class_heritage
                (implements_clause
                    (type_identifier) @base)))
        """

    def extract_name(self, node: Node) -> str:
        text = node.text.decode("utf-8") if node.text else ""
        # Strip generic type params: MyClass<T> -> MyClass
        if "<" in text:
            text = text[:text.index("<")]
        return text

    def extract_callee(self, node: Node) -> str:
        text = node.text.decode("utf-8") if node.text else ""
        # Strip optional chaining and type assertions
        text = text.replace("?.", ".")
        # Strip generic call args: foo<T>() -> foo
        if "<" in text:
            text = text[:text.index("<")]
        return text

    def extract_import_target(self, source_node: Node, name_node: Node | None) -> str:
        text = source_node.text.decode("utf-8") if source_node.text else ""
        for quote in ('"', "'", '`'):
            text = text.strip(quote)
        text = text.lstrip("./")
        return text.replace("/", ".")

    def module_name_from_path(self, rel_path: str) -> str:
        from pathlib import Path
        p = Path(rel_path)
        stem = p.stem
        if stem == "index":
            parts = list(p.parent.parts)
        else:
            parts = list(p.parent.parts) + [stem]
        return ".".join(parts) if parts else "__root__"

    def skip_directory(self, dirname: str) -> bool:
        return dirname in {
            ".git", "node_modules", "dist", "build", ".next",
            ".nuxt", ".svelte-kit", "coverage", ".turbo",
            "__pycache__", ".venv", "venv",
        }

    def is_test_file(self, rel_path: str) -> bool:
        from pathlib import Path
        stem = Path(rel_path).stem
        return (
            stem.endswith(".test")
            or stem.endswith(".spec")
            or stem.startswith("test_")
            or stem.endswith("_test")
        )


# Auto-register on import
try:
    registry.register(JavaScriptAdapter())
except Exception:
    pass

try:
    registry.register(TypeScriptAdapter())
except Exception:
    pass
