"""Tree-sitter based multi-language parser.

Provides a generic CST walker that uses per-language adapters to emit
the same NodeData/EdgeData models as the Python ast-based parser.
This enables interlinked to analyze any language with a tree-sitter grammar.
"""

from interlinked.analyzer.treesitter.adapter import LanguageAdapter
from interlinked.analyzer.treesitter.walker import parse_file_treesitter, parse_project_treesitter

__all__ = ["LanguageAdapter", "parse_file_treesitter", "parse_project_treesitter"]
