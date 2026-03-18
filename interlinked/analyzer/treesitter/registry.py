"""Language adapter registry.

Discovers and manages available tree-sitter language adapters.
Auto-detects the language of a project by scanning file extensions.
"""

from __future__ import annotations

from pathlib import Path

from interlinked.analyzer.treesitter.adapter import LanguageAdapter

# Global registry of adapters, keyed by language name
_ADAPTERS: dict[str, LanguageAdapter] = {}

# Extension -> adapter lookup, built lazily
_EXT_MAP: dict[str, LanguageAdapter] = {}


def register(adapter: LanguageAdapter) -> None:
    """Register a language adapter."""
    _ADAPTERS[adapter.name] = adapter
    for ext in adapter.extensions:
        _EXT_MAP[ext] = adapter


def get_adapter(name: str) -> LanguageAdapter | None:
    """Get a registered adapter by language name."""
    return _ADAPTERS.get(name)


def adapter_for_file(path: str | Path) -> LanguageAdapter | None:
    """Get the adapter for a file based on its extension."""
    ext = Path(path).suffix
    return _EXT_MAP.get(ext)


def available_languages() -> list[str]:
    """List all registered language names."""
    return sorted(_ADAPTERS.keys())


def detect_languages(root: str | Path) -> list[LanguageAdapter]:
    """Detect which languages are present in a project directory.

    Scans file extensions in the project and returns adapters for
    any registered languages found.
    """
    root = Path(root)
    found: set[str] = set()
    adapters: list[LanguageAdapter] = []

    for path in root.rglob("*"):
        if path.is_file():
            adapter = adapter_for_file(path)
            if adapter and adapter.name not in found:
                found.add(adapter.name)
                adapters.append(adapter)

    return adapters


def _auto_register_builtins() -> None:
    """Register built-in adapters if their grammar packages are available."""
    # Adapters are registered lazily when their modules are imported.
    # This function attempts to import all built-in adapter modules.
    import importlib
    _BUILTIN_ADAPTERS = [
        "interlinked.analyzer.treesitter.langs.python_lang",
        "interlinked.analyzer.treesitter.langs.javascript_lang",
    ]
    for module_name in _BUILTIN_ADAPTERS:
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            pass  # Grammar not installed — adapter not available


# Auto-register on import
_auto_register_builtins()
