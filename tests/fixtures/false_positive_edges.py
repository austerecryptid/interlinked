"""Fixture: false positive edge regression tests.

Tests that local-variable method calls, builtin method calls on dotted
targets, and same-name cross-module functions resolve correctly.

Patterns that MUST NOT create false edges:
- effect.get("key")       — dict.get(), not a project function
- logger.warning("msg")   — logging.warning(), not a project symbol
- ops.extend(new_ops)     — list.extend(), not a project function
- ref.startswith("$")     — str.startswith(), not a project function
- args.items()            — dict.items(), not a project function
- collected.append(x)     — list.append(), not a project function

Patterns that MUST resolve to the LOCAL definition:
- _helper() inside module_a should resolve to module_a._helper,
  NOT module_b._helper
"""
import logging

logger = logging.getLogger(__name__)


# ── Two modules with same-name private helper ────────────────────────

def _shared_helper(x: int) -> int:
    """This name also exists in dead_code_traps or other fixtures."""
    return x + 1


def calls_shared_helper() -> int:
    """Should call THIS module's _shared_helper, not another module's."""
    return _shared_helper(10)


# ── Local variable method calls that must NOT resolve ────────────────

def dict_method_calls(data: dict) -> list:
    """All of these are dict/list/str builtins on local variables."""
    effect = {"type": "damage", "value": 10}
    val = effect.get("value", 0)

    ops: list = []
    ops.extend([1, 2, 3])
    ops.append(val)

    args = {"a": 1, "b": 2}
    for k, v in args.items():
        ops.append(v)

    ref = "some_string"
    if ref.startswith("$"):
        parts = ref.split(".")
        ref = ".".join(parts[1:])

    collected: list = []
    collected.append(ref)

    return collected


def logging_calls() -> None:
    """logger.warning/error/info are logging methods, not project symbols."""
    logger.warning("something happened")
    logger.error("something bad")
    logger.info("something informational")
    logger.debug("debug info")


class ServiceWithMethods:
    """A class whose method names DON'T collide with builtins."""

    def process(self, data: dict) -> str:
        return str(data)

    def validate(self, value: int) -> bool:
        return value > 0


def uses_service() -> str:
    """self.process() and self.validate() should resolve to ServiceWithMethods."""
    svc = ServiceWithMethods()
    if svc.validate(42):
        return svc.process({"key": "val"})
    return ""
