# TODO

## Tech Debt

- [ ] **Scope-aware edge emission in parser**: Replace `_BUILTIN_METHOD_NAMES` blocklist with proper scope tracking in Pass 1. Tag each call edge root as local/parameter vs import/module-scoped. This eliminates the need for a hand-maintained method name list and catches all localvar.method() false positives generically.

- [ ] **Move `_BUILTIN_METHOD_NAMES` to shared module**: `parser.py` currently imports it from `graph.py`, which is a layering violation (parser → graph dependency). Move to `models.py` or a new `constants.py`.

- [ ] **Scope-aware name index**: `_resolve_edge` does O(n·m) prefix matching when multiple candidates share a bare name. A scope-aware index (keyed by module prefix) would make ambiguous resolution O(1) and architecturally cleaner.
