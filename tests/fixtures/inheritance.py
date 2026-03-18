"""Fixture: inheritance edge cases.

Tests:
- Diamond inheritance (A -> B, C -> D)
- Super() calls resolving correctly
- Method override detection
- Abstract methods via ABC
- Mixin classes
- __init_subclass__ hook
- Property vs method distinction
"""

from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def execute(self) -> str:
        ...

    def describe(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class Mixin:
    """Mixin that adds logging — no __init__."""

    def log(self, msg: str) -> None:
        print(f"[{self.__class__.__name__}] {msg}")


class Left(Base, Mixin):
    def execute(self) -> str:
        self.log("left execute")
        return f"left-{self.name}"


class Right(Base):
    def __init__(self, name: str, priority: int) -> None:
        super().__init__(name)
        self.priority = priority

    def execute(self) -> str:
        return f"right-{self.name}-{self.priority}"


class Diamond(Left, Right):
    """Diamond: Diamond -> Left -> Right -> Base, plus Mixin from Left."""

    def __init__(self, name: str, priority: int, tag: str) -> None:
        super().__init__(name, priority)
        self.tag = tag

    def execute(self) -> str:
        base = super().execute()  # MRO: Left.execute
        return f"{base}:{self.tag}"


class AutoRegister:
    """Uses __init_subclass__ to track subclasses."""

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        AutoRegister._registry[cls.__name__] = cls


class PluginA(AutoRegister):
    def run(self) -> None:
        pass


class PluginB(AutoRegister):
    def run(self) -> None:
        pass


class WithProperty:
    def __init__(self, x: int) -> None:
        self._x = x

    @property
    def value(self) -> int:
        """Property — should be a variable/attribute, not a call target."""
        return self._x

    @value.setter
    def value(self, v: int) -> None:
        self._x = v

    def double(self) -> int:
        return self.value * 2  # reads property, not a call
