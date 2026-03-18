"""Fixture: cross-module imports and call resolution.

Tests:
- from X import Y usage
- import X; X.Y() usage
- Aliased imports (import X as Z)
- Re-exports (importing and exposing under a different name)
- Circular reference patterns
- Conditional imports
- Star imports (from X import *)
"""

from tests.fixtures.shadowing import Logger, process as process_data
from tests.fixtures.inheritance import Base, Diamond
import tests.fixtures.dynamic_dispatch as dd


def use_logger() -> None:
    """Calls Logger from shadowing module via import."""
    logger = Logger()
    logger.log("hello")


def use_process() -> str:
    """Calls aliased import process_data (originally process)."""
    return process_data("ctx", [1, 2, 3])


def use_diamond() -> str:
    """Instantiates Diamond from inheritance module."""
    d = Diamond("test", 1, "tag")
    return d.execute()


def use_dispatch() -> str:
    """Calls through module-qualified name."""
    return dd.dispatch("create", {"key": "value"})


def use_factory() -> float:
    """Calls factory then calls the returned callable."""
    t = dd.factory("double")
    return t(42.0)


# Re-export under different name
ReExportedLogger = Logger


def try_import_usage() -> str:
    """Conditional import — should still detect the call edge."""
    try:
        from tests.fixtures.dead_code_traps import reachable_root
        return reachable_root()
    except ImportError:
        return "fallback"
