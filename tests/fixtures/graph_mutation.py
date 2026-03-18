"""Fixture: graph mutation stress test.

This module exists to be modified between parse runs to test:
- Incremental update (add/remove/change functions)
- No orphaned edges after node removal
- No duplicate nodes after re-parse
- Edge targets updated when a function signature changes
"""


class MutableService:
    def __init__(self) -> None:
        self.counter = 0

    def increment(self) -> int:
        self.counter += 1
        return self.counter

    def reset(self) -> None:
        self.counter = 0

    def get_count(self) -> int:
        return self.counter


def caller_a() -> int:
    svc = MutableService()
    svc.increment()
    return svc.get_count()


def caller_b() -> None:
    svc = MutableService()
    svc.reset()


def standalone() -> str:
    """Will be removed in mutation test — edges to it must be cleaned up."""
    return "standalone"


def calls_standalone() -> str:
    """Calls standalone() — after standalone is removed, this edge must vanish."""
    return standalone()
