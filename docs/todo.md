# Analyzer TODO

## Known Inference Gaps

### Walrus operator (`:=`) type propagation
The parser sees calls on walrus-assigned variables (e.g. `item.process()`) but
doesn't resolve them to qualified targets because `NamedExpr` isn't handled as
an assignment in the type inferencer. Full resolution would require:

1. Treat `ast.NamedExpr` as an assignment in `_TypeInferencer.collect_types`
2. Track builtin return types (`next(it)` → element type of iterator,
   `items[0]` → element type of list)

Low priority — walrus appears in <0.1% of production Python code. The realistic
use case (`if m := re.match(...)`) involves stdlib types that wouldn't resolve
to project nodes anyway.

### Import alias tracking in `build_from` (incremental updates)
Import aliases (`from X import Y as Z`) are resolved in `parse_project` by
injecting into the name_index. The incremental `parse_file` → `build_from`
path does not yet carry alias information, so aliased names may not resolve
during incremental updates. Fix: propagate alias map through `parse_file`
return value and inject into `build_from`'s name_index.
