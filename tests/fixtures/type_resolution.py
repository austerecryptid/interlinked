"""Fixture: type resolution edge cases.

Tests:
- Chained attribute access (obj.attr.method())
- Typed parameters resolving method calls
- Return type propagation (x = factory(); x.method())
- Assignment propagation (a = b; a.method() should resolve same as b.method())
- For-loop variable typing (for item in collection: item.method())
- Comprehension variable scope
- Ternary/conditional assignment types
- Optional/Union types
"""


class Engine:
    def __init__(self) -> None:
        self.state = EngineState()

    def run(self) -> "Result":
        self.state.activate()
        return Result(True)

    def get_state(self) -> "EngineState":
        return self.state


class EngineState:
    def __init__(self) -> None:
        self.active = False

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False


class Result:
    def __init__(self, success: bool) -> None:
        self.success = success

    def describe(self) -> str:
        return f"success={self.success}"


def chained_access(engine: Engine) -> None:
    """engine.state.activate() — two-level chain resolution."""
    engine.state.activate()
    engine.state.deactivate()


def return_type_propagation(engine: Engine) -> str:
    """r = engine.run() returns Result; r.describe() should resolve."""
    r = engine.run()
    return r.describe()


def get_then_call(engine: Engine) -> None:
    """s = engine.get_state() returns EngineState; s.activate() should resolve."""
    s = engine.get_state()
    s.activate()


def assignment_propagation(engine: Engine) -> None:
    """a = engine; b = a; b.run() — should still resolve to Engine.run."""
    a = engine
    b = a
    b.run()


def loop_variable_typing(engines: list[Engine]) -> None:
    """for e in engines: e.run() — e should be typed as Engine."""
    for e in engines:
        e.run()
    # Comprehension
    results = [e.run() for e in engines]


def conditional_typing(flag: bool) -> str:
    """Type depends on branch — both branches produce Result."""
    if flag:
        r = Result(True)
    else:
        r = Result(False)
    return r.describe()


class Container:
    """Tests self.attr.method() chains within a class."""

    def __init__(self) -> None:
        self.engine = Engine()
        self.items: list[Result] = []

    def process(self) -> str:
        """self.engine.run() — resolve through instance attribute type."""
        result = self.engine.run()
        self.items.append(result)
        return result.describe()

    def summarize(self) -> list[str]:
        """Iterate self.items (typed list[Result]), call .describe()."""
        return [item.describe() for item in self.items]
