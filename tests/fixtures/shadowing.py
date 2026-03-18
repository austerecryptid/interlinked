"""Fixture: name shadowing, nested closures, reuse of names across scopes.

Tests that the parser correctly distinguishes:
- A module-level function vs a local with the same name
- A parameter shadowing a module-level class
- Nested closures capturing variables from enclosing scope
- A method with the same name as a module-level function
"""


class Logger:
    """Class that will be shadowed by a parameter name."""

    def log(self, msg: str) -> None:
        print(msg)


def process(Logger: str, data: list) -> str:
    """Parameter 'Logger' shadows the class Logger above."""

    def _inner(x):
        """Closure capturing 'Logger' from enclosing scope (the parameter, not the class)."""
        return f"{Logger}: {x}"

    results = []
    for item in data:
        results.append(_inner(item))
    return ", ".join(results)


def log(msg: str) -> None:
    """Module-level function with same name as Logger.log method."""
    print(f"[MODULE] {msg}")


class Processor:
    def process(self, data: list) -> str:
        """Method with same name as module-level process() function."""
        return process("Processor", data)

    def run(self) -> None:
        log("starting")           # should resolve to module-level log()
        result = self.process([])  # should resolve to Processor.process()
        log(result)                # should resolve to module-level log()


def nested_closures(x: int) -> callable:
    """Three levels of closure nesting."""
    def level1(y: int):
        def level2(z: int):
            return x + y + z      # x, y captured from enclosing scopes
        return level2
    return level1
