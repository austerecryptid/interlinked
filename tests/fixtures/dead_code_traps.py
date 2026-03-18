"""Fixture: dead code detection traps.

Tests:
- Truly dead functions (never called anywhere)
- Functions that LOOK dead but are called via:
  - Decorator references
  - Callback registration
  - __all__ exports
  - String-based dispatch
  - __init_subclass__ / metaclass hooks
- Functions reachable only through a chain of calls
- Module-level code that invokes functions (not dead)
- Conditional imports that might hide liveness
"""

__all__ = ["exported_but_never_called_directly"]


def exported_but_never_called_directly() -> str:
    """In __all__ — should NOT be marked dead."""
    return "I'm exported"


def _truly_dead_helper() -> str:
    """Never called by anything. Should be dead."""
    return "nobody calls me"


def _also_dead(x: int) -> int:
    """Also never called. Dead."""
    return x * 2


def reachable_root() -> str:
    """Called from module level — alive."""
    return _reachable_step1()


def _reachable_step1() -> str:
    """Only called by reachable_root — alive via chain."""
    return _reachable_step2()


def _reachable_step2() -> str:
    """Only called by _reachable_step1 — alive via chain."""
    return "deep"


def _callback_target(event: dict) -> None:
    """Registered as a callback — alive."""
    print(event)


def register_callbacks() -> dict:
    """Registers _callback_target — makes it alive."""
    return {"on_event": _callback_target}


class _DeadClass:
    """Never instantiated or referenced. Dead."""

    def method(self) -> None:
        pass


class AliveClass:
    """Instantiated below — alive."""

    def work(self) -> str:
        return "working"

    def _internal(self) -> str:
        """Called by work? No — but called by external_caller. Alive."""
        return "internal"


def external_caller() -> str:
    """Calls AliveClass._internal via instance."""
    obj = AliveClass()
    return obj._internal()


# Module-level execution — makes reachable_root alive
_result = reachable_root()
_callbacks = register_callbacks()
_instance = AliveClass()
_ext = external_caller()
