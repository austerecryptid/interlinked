"""Fixture: Python language features that stress-test static analysis.

Covers:
- Context managers (sync and async, with-as variable typing)
- Generators and generator delegation (yield from)
- Decorators (wrapping, parameterized, stacked)
- Dataclasses (generated __init__, field access)
- Comprehension scoping
- Walrus operator (:=)
- Exception handling (except X as e, raise from)
- Unpacking assignments
- Class methods and static methods
- Slots
- String-based __all__ with re-exports
"""

from dataclasses import dataclass, field
from typing import Iterator, Generator


# ── Context managers ─────────────────────────────────────────────────

class SyncConnection:
    def execute(self, sql: str) -> list:
        return []

    def close(self) -> None:
        pass


class SyncPool:
    """Sync context manager returning a DIFFERENT type."""

    def __enter__(self) -> SyncConnection:
        return SyncConnection()

    def __exit__(self, *args) -> None:
        pass


def use_sync_with() -> list:
    """with pool as conn — conn should be typed as SyncConnection."""
    pool = SyncPool()
    with pool as conn:
        return conn.execute("SELECT 1")


def use_sync_with_inline() -> list:
    """with SyncPool() as conn — constructor inline."""
    with SyncPool() as conn:
        return conn.execute("SELECT 1")


class SelfReturningCM:
    """Context manager that returns self — common pattern."""

    def __enter__(self) -> "SelfReturningCM":
        return self

    def __exit__(self, *args) -> None:
        pass

    def do_work(self) -> str:
        return "work"


def use_self_returning_cm() -> str:
    """with SelfReturningCM() as obj — obj is same type."""
    with SelfReturningCM() as obj:
        return obj.do_work()


# ── Generators ───────────────────────────────────────────────────────

class Item:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def process(self) -> str:
        return f"{self.name}={self.value}"


def item_generator(n: int) -> Generator[Item, None, None]:
    """Yields Item instances — consumer should know element type."""
    for i in range(n):
        yield Item(f"item_{i}", i)


def consume_typed_generator() -> list[str]:
    """for item in item_generator(): item.process() should resolve."""
    results = []
    for item in item_generator(10):
        results.append(item.process())
    return results


def delegating_generator() -> Generator[Item, None, None]:
    """yield from — delegates to item_generator."""
    yield from item_generator(5)


def consume_delegated() -> list[str]:
    """Consuming a yield-from generator — same element type."""
    return [item.process() for item in delegating_generator()]


def generator_with_send() -> Generator[int, str, None]:
    """Generator that receives values via send()."""
    value = yield 0
    while value:
        value = yield len(value)


# ── Decorators ───────────────────────────────────────────────────────

class Registry:
    """Class-based decorator — registers functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, callable] = {}

    def register(self, name: str):
        """Parameterized decorator."""
        def decorator(fn):
            self._handlers[name] = fn
            return fn
        return decorator

    def get_handler(self, name: str):
        return self._handlers.get(name)


registry = Registry()


@registry.register("create")
def handle_create(data: dict) -> str:
    return f"created {data}"


@registry.register("delete")
def handle_delete(data: dict) -> str:
    return f"deleted {data}"


def stacked_decorators_target() -> str:
    """Called by dispatching through registry — alive via decorator registration."""
    return registry.get_handler("create")({"key": "val"})


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class Server:
    config: Config
    name: str = "default"

    def address(self) -> str:
        """Accesses self.config.host — chained dataclass attribute."""
        return f"{self.config.host}:{self.config.port}"

    def is_debug(self) -> bool:
        return self.config.debug


def use_dataclass() -> str:
    """Instantiate dataclass, call method, chain attributes."""
    cfg = Config(host="0.0.0.0", port=9090)
    srv = Server(config=cfg, name="test")
    return srv.address()


def access_dataclass_fields(cfg: Config) -> str:
    """Direct field access on typed parameter."""
    return f"{cfg.host}:{cfg.port}"


# ── Exception handling ───────────────────────────────────────────────

class AppError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def describe(self) -> str:
        return f"[{self.code}] {self.message}"


class NotFoundError(AppError):
    def __init__(self, resource: str) -> None:
        super().__init__(404, f"{resource} not found")
        self.resource = resource


def handle_errors() -> str:
    """except AppError as e — e should be typed as AppError."""
    try:
        raise NotFoundError("widget")
    except AppError as e:
        return e.describe()
    except Exception:
        return "unknown"


def raise_from_chain() -> None:
    """raise X from Y — both should be recognized."""
    try:
        raise NotFoundError("thing")
    except AppError as original:
        raise AppError(500, "internal") from original


# ── Unpacking ────────────────────────────────────────────────────────

def tuple_return() -> tuple[Config, Server]:
    """Returns a tuple of project types."""
    cfg = Config()
    srv = Server(config=cfg)
    return cfg, srv


def unpack_tuple() -> str:
    """cfg, srv = tuple_return() — both should get correct types."""
    cfg, srv = tuple_return()
    return srv.address()


# ── Class methods and static methods ─────────────────────────────────

class Factory:
    @classmethod
    def create(cls, name: str) -> "Factory":
        return cls(name)

    @staticmethod
    def default() -> "Factory":
        return Factory("default")

    def __init__(self, name: str) -> None:
        self.name = name

    def describe(self) -> str:
        return f"Factory({self.name})"


def use_classmethod() -> str:
    """Factory.create() returns Factory — should resolve describe()."""
    f = Factory.create("test")
    return f.describe()


def use_staticmethod() -> str:
    """Factory.default() returns Factory — should resolve describe()."""
    f = Factory.default()
    return f.describe()


# ── Walrus operator ──────────────────────────────────────────────────

def walrus_in_while(items: list[Item]) -> list[str]:
    """while (item := next(iter)) is not None — item should be typed."""
    it = iter(items)
    results = []
    while (item := next(it, None)) is not None:
        results.append(item.process())
    return results


def walrus_in_if(items: list[Item]) -> str:
    """if (first := items[0]) — first should be typed."""
    if items and (first := items[0]):
        return first.process()
    return ""


# ── Comprehension edge cases ────────────────────────────────────────

def nested_comprehension(servers: list[Server]) -> list[str]:
    """Nested comprehension — inner variable typing."""
    return [
        tag
        for srv in servers
        for tag in srv.config.tags
    ]


def dict_comprehension(items: list[Item]) -> dict[str, int]:
    """Dict comprehension — key/value from typed elements."""
    return {item.name: item.value for item in items}


def filtered_comprehension(items: list[Item]) -> list[str]:
    """Comprehension with if-clause accessing typed element."""
    return [item.process() for item in items if item.value > 0]
