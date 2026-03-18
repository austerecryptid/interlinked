"""Fixture: dynamic dispatch, decorators, callables, and indirect calls.

Tests:
- Decorated functions (decorator should be a call edge)
- functools.wraps preserving identity
- Callable objects (__call__)
- Factory functions returning instances
- Higher-order functions (functions as arguments)
- Conditional function assignment
- Dict-based dispatch tables
"""

import functools
from typing import Callable


def timer(func):
    """Decorator — parser should see timer() calling func()."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def retry(n: int):
    """Parameterized decorator — two levels of wrapping."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    pass
            raise RuntimeError("exhausted retries")
        return wrapper
    return decorator


class Transformer:
    """Callable object — instances can be called like functions."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, value: float) -> float:
        return self._apply(value)

    def _apply(self, value: float) -> float:
        return value * self.scale


@timer
def compute(x: int) -> int:
    return x * 2


@retry(3)
def fetch(url: str) -> str:
    return f"data from {url}"


def factory(kind: str) -> Transformer:
    """Factory returning a callable instance."""
    if kind == "double":
        return Transformer(2.0)
    return Transformer(1.0)


def higher_order(fn: Callable[[int], int], values: list[int]) -> list[int]:
    """Takes a function as argument and calls it."""
    return [fn(v) for v in values]


def dispatch(action: str, data: dict) -> str:
    """Dict-based dispatch table — all handlers should be call targets."""
    handlers = {
        "create": _handle_create,
        "update": _handle_update,
        "delete": _handle_delete,
    }
    handler = handlers.get(action)
    if handler is None:
        return "unknown"
    return handler(data)


def _handle_create(data: dict) -> str:
    return f"created {data}"


def _handle_update(data: dict) -> str:
    return f"updated {data}"


def _handle_delete(data: dict) -> str:
    return f"deleted {data}"


def conditional_assignment(flag: bool) -> Callable:
    """Function reference assigned conditionally."""
    if flag:
        fn = _handle_create
    else:
        fn = _handle_delete
    return fn
