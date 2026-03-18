"""AST-based parser that extracts symbols and relationships from Python source.

Architecture:
  Pass 1 (_SymbolVisitor) -- Extracts all nodes and **raw** edges. Edge targets
      are the literal names from the AST (e.g. "self.state.zoom_level", "n.id",
      "graph"). No resolution happens here.
  Pass 2 (_TypeInferencer) -- Collects type annotations from the AST, then
      resolves every raw edge target using the full type map. Handles self/cls,
      dotted attribute chains, typed loop variables, assignment propagation.
  Pass 3 (Structural inference) -- For any unresolved dotted name like "n.id",
      builds a reverse index (field_name -> classes), intersects all field
      accesses for a variable, and infers its type from the unique match.
  Pass 4 (Progressive truncation + drop) -- For any remaining unresolved edge,
      progressively strips attrs from the right until a known node is hit.
      If nothing resolves, the edge is external and is dropped.
  Pass 5 (CodeGraph.build_from) -- Final short-name to qualified-name resolution
      for bare names (cross-module calls).

No hardcoded external module or builtin method lists. Resolution is entirely
dynamic: if a target resolves to a project node, it's kept; otherwise it's
progressively truncated or dropped.

Extracts:
  Nodes -- modules, classes, functions/methods, variables (module/class/instance
          scope), parameters, local variables (function scope).
  Edges -- contains, calls, imports, inherits, reads, writes, returns.
"""

from __future__ import annotations

import ast
import builtins
import warnings
from pathlib import Path
from typing import Any

from interlinked.models import NodeData, EdgeData, SymbolType, EdgeType

# Python builtins we should never create nodes/edges for
_BUILTINS: frozenset[str] = frozenset(dir(builtins)) | frozenset({
    "None", "True", "False", "__name__", "__file__", "__doc__",
    "__all__", "__spec__", "__loader__", "__package__", "__builtins__",
})



def parse_file(
    file_path: str | Path,
    module_qname: str,
    existing_node_ids: set[str] | None = None,
    existing_type_index: dict[str, str] | None = None,
) -> tuple[list[NodeData], list[EdgeData]]:
    """Parse a single Python file and return its nodes + resolved edges.

    This is the incremental counterpart to parse_project(). It parses one file
    and resolves edges using optional context from the existing graph.

    Args:
        file_path: Absolute path to the .py file.
        module_qname: Dotted module name (e.g. "analyzer.graph").
        existing_node_ids: Node IDs from the rest of the graph (for edge resolution).
        existing_type_index: Type index from the rest of the graph (class/module short name -> qname).

    Returns:
        (nodes, edges) — fully resolved for this file, ready for CodeGraph.update_file().
    """
    file_path = Path(file_path)
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return [], []

    # Pass 1: extract symbols and raw edges
    nodes, edges = _extract_from_module(tree, source, module_qname, str(file_path))

    # Build combined node ID set and type index
    node_ids = {n.id for n in nodes}
    if existing_node_ids:
        node_ids |= existing_node_ids

    type_index: dict[str, str] = dict(existing_type_index) if existing_type_index else {}
    for n in nodes:
        if n.symbol_type in (SymbolType.CLASS, SymbolType.MODULE):
            type_index[n.name] = n.id
            parts = n.qualified_name.split(".")
            for i in range(len(parts)):
                suffix = ".".join(parts[i:])
                type_index.setdefault(suffix, n.id)

    # Pass 2: type inference
    inferencer = _TypeInferencer(type_index, node_ids)
    inferencer.collect_types(tree, module_qname)

    # Pass 3: structural type inference
    inferencer.infer_structural_types(edges)

    # Build name index for resolution
    name_index: dict[str, list[str]] = {}
    all_node_ids_for_index = node_ids  # includes existing
    for n in nodes:
        name_index.setdefault(n.name, []).append(n.id)
        parts = n.qualified_name.split(".")
        for i in range(1, len(parts)):
            suffix = ".".join(parts[i:])
            name_index.setdefault(suffix, []).append(n.id)

    # Pass 4: resolve edges
    resolved_edges: list[EdgeData] = []
    for e in edges:
        if e.edge_type not in (EdgeType.READS, EdgeType.WRITES, EdgeType.CALLS, EdgeType.RETURNS):
            resolved_edges.append(e)
            continue

        if e.edge_type == EdgeType.CALLS:
            raw_target = e.target
            callee_root = raw_target.split(".")[0]
            if callee_root in _BUILTINS:
                continue
            resolved = inferencer.resolve(raw_target, e.source)
            if resolved and resolved in node_ids:
                resolved_edges.append(EdgeData(
                    source=e.source, target=resolved,
                    edge_type=e.edge_type, is_dead=e.is_dead,
                    is_proposed=e.is_proposed, line=e.line,
                    metadata=e.metadata,
                ))
            else:
                resolved_edges.append(e)
            continue

        raw_target = e.target
        resolved = inferencer.resolve(raw_target, e.source)
        if resolved is None:
            continue

        if resolved in node_ids:
            if resolved != raw_target:
                e = EdgeData(
                    source=e.source, target=resolved,
                    edge_type=e.edge_type, is_dead=e.is_dead,
                    is_proposed=e.is_proposed, line=e.line,
                    metadata=e.metadata,
                )
            resolved_edges.append(e)
            continue

        if "." in resolved:
            parts = resolved.split(".")
            found = False
            for i in range(len(parts), 0, -1):
                candidate = ".".join(parts[:i])
                if candidate in node_ids:
                    resolved_edges.append(EdgeData(
                        source=e.source, target=candidate,
                        edge_type=e.edge_type, is_dead=e.is_dead,
                        is_proposed=e.is_proposed, line=e.line,
                        metadata=e.metadata,
                    ))
                    found = True
                    break
            if found:
                continue

        if "." not in resolved:
            if resolved in name_index:
                resolved_edges.append(EdgeData(
                    source=e.source, target=resolved,
                    edge_type=e.edge_type, is_dead=e.is_dead,
                    is_proposed=e.is_proposed, line=e.line,
                    metadata=e.metadata,
                ))

    return nodes, resolved_edges


def path_to_module(rel_path: Path) -> str:
    """Convert a relative file path to a dotted module name. Public API."""
    return _path_to_module(rel_path)


def parse_project(root: str | Path) -> tuple[list[NodeData], list[EdgeData]]:
    """Walk a Python project directory and extract all symbols and edges."""
    root = Path(root).resolve()
    nodes: list[NodeData] = []
    edges: list[EdgeData] = []

    py_files = sorted(root.rglob("*.py"))

    # Pass 1: extract all symbols and raw (unresolved) edges
    trees: list[tuple[ast.Module, str, str]] = []
    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        rel_path = py_file.relative_to(root)
        module_qname = _path_to_module(rel_path)
        trees.append((tree, module_qname, str(py_file)))

        file_nodes, file_edges = _extract_from_module(
            tree, source, module_qname, str(py_file)
        )
        nodes.extend(file_nodes)
        edges.extend(file_edges)

    # Pass 2: type inference from annotations
    node_ids = {n.id for n in nodes}

    # Build class/type short name -> qualified name index
    type_index: dict[str, str] = {}
    for n in nodes:
        if n.symbol_type in (SymbolType.CLASS, SymbolType.MODULE):
            type_index[n.name] = n.id
            parts = n.qualified_name.split(".")
            for i in range(len(parts)):
                suffix = ".".join(parts[i:])
                type_index.setdefault(suffix, n.id)

    inferencer = _TypeInferencer(type_index, node_ids)
    for tree, module_qname, _fp in trees:
        inferencer.collect_types(tree, module_qname)

    # Pass 3: structural type inference — infer types from field access patterns
    inferencer.infer_structural_types(edges)

    # Build name_index for bare-name filtering (same index build_from uses)
    name_index: dict[str, list[str]] = {}
    for n in nodes:
        name_index.setdefault(n.name, []).append(n.id)
        parts = n.qualified_name.split(".")
        for i in range(1, len(parts)):
            suffix = ".".join(parts[i:])
            name_index.setdefault(suffix, []).append(n.id)

    # Pass 4: resolve all data-flow edges, progressive truncation, drop external
    #
    # Edge type handling:
    #   CALLS / IMPORTS — ALWAYS keep. Even unresolved external calls like
    #     nx.all_simple_paths() are critical for auditing what the code does.
    #   READS / WRITES — resolve or drop. External attribute accesses like
    #     node.lineno (ast module) are not useful; project data flow must resolve.
    #   RETURNS — resolve or drop (same as reads/writes).
    #   CONTAINS / INHERITS — pass through unchanged.
    resolved_edges: list[EdgeData] = []
    for e in edges:
        # Structural edges — always keep
        if e.edge_type not in (EdgeType.READS, EdgeType.WRITES, EdgeType.CALLS, EdgeType.RETURNS):
            resolved_edges.append(e)
            continue

        # CALLS — always keep (external calls are audit-critical), but filter builtins
        if e.edge_type == EdgeType.CALLS:
            raw_target = e.target
            # Filter builtin function calls (len, str, isinstance, etc.)
            callee_root = raw_target.split(".")[0]
            if callee_root in _BUILTINS:
                continue
            resolved = inferencer.resolve(raw_target, e.source)
            if resolved and resolved in node_ids:
                resolved_edges.append(EdgeData(
                    source=e.source, target=resolved,
                    edge_type=e.edge_type, is_dead=e.is_dead,
                    is_proposed=e.is_proposed, line=e.line,
                    metadata=e.metadata,
                ))
            else:
                # Keep the raw call target — external library calls visible to auditors
                resolved_edges.append(e)
            continue

        # READS / WRITES / RETURNS — resolve, truncate, or drop
        raw_target = e.target
        resolved = inferencer.resolve(raw_target, e.source)
        if resolved is None:
            continue  # filtered out (builtin like len, str, True)

        # If resolved is a known node, keep
        if resolved in node_ids:
            if resolved != raw_target:
                e = EdgeData(
                    source=e.source, target=resolved,
                    edge_type=e.edge_type, is_dead=e.is_dead,
                    is_proposed=e.is_proposed, line=e.line,
                    metadata=e.metadata,
                )
            resolved_edges.append(e)
            continue

        # Progressive truncation: strip from right until we hit a known node
        if "." in resolved:
            parts = resolved.split(".")
            found = False
            for i in range(len(parts), 0, -1):
                candidate = ".".join(parts[:i])
                if candidate in node_ids:
                    resolved_edges.append(EdgeData(
                        source=e.source, target=candidate,
                        edge_type=e.edge_type, is_dead=e.is_dead,
                        is_proposed=e.is_proposed, line=e.line,
                        metadata=e.metadata,
                    ))
                    found = True
                    break
            if found:
                continue

        # Bare names — keep only if they match a project symbol short name
        if "." not in resolved:
            if resolved in name_index:
                resolved_edges.append(EdgeData(
                    source=e.source, target=resolved,
                    edge_type=e.edge_type, is_dead=e.is_dead,
                    is_proposed=e.is_proposed, line=e.line,
                    metadata=e.metadata,
                ))
            continue

        # Dotted read/write that didn't resolve — external, drop it

    return nodes, resolved_edges


def _path_to_module(rel_path: Path) -> str:
    """Convert a relative file path to a dotted module name."""
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts) if parts else "__root__"


def _extract_from_module(
    tree: ast.Module,
    source: str,
    module_qname: str,
    file_path: str,
) -> tuple[list[NodeData], list[EdgeData]]:
    """Extract nodes and edges from a single parsed module."""
    nodes: list[NodeData] = []
    edges: list[EdgeData] = []

    mod_docstring = ast.get_docstring(tree)
    nodes.append(NodeData(
        id=module_qname,
        name=module_qname.split(".")[-1],
        qualified_name=module_qname,
        symbol_type=SymbolType.MODULE,
        file_path=file_path,
        line_start=1,
        line_end=len(source.splitlines()),
        docstring=mod_docstring,
    ))

    visitor = _SymbolVisitor(module_qname, file_path, nodes, edges)
    visitor.visit(tree)

    # Extract module-level calls (statements that run on import, outside any
    # function/class body). This captures things like app.add_middleware(...),
    # Field(...) in Pydantic models, etc.
    visitor._extract_scope_level_calls(tree, module_qname)

    return nodes, edges


# ---------------------------------------------------------------------------
# Pass 1: AST visitor -- node creation + raw edge emission
# ---------------------------------------------------------------------------

class _SymbolVisitor(ast.NodeVisitor):
    """Walks the AST, creates graph nodes, and emits raw (unresolved) edges.

    Edge targets are the literal AST names: "self.state", "n.id", "graph".
    Resolution is deferred entirely to pass 2 (_TypeInferencer).
    """

    def __init__(
        self,
        module_qname: str,
        file_path: str,
        nodes: list[NodeData],
        edges: list[EdgeData],
    ):
        self._module = module_qname
        self._file = file_path
        self._nodes = nodes
        self._edges = edges
        self._scope_stack: list[str] = [module_qname]
        self._node_ids: set[str] = set()

    @property
    def _current_scope(self) -> str:
        return self._scope_stack[-1]

    def _add_node(self, node: NodeData) -> None:
        if node.id not in self._node_ids:
            self._nodes.append(node)
            self._node_ids.add(node.id)

    def _is_inside_class(self) -> bool:
        return any(
            n.symbol_type == SymbolType.CLASS
            for n in self._nodes
            if n.id == self._current_scope
        )

    def _class_scope(self) -> str | None:
        for scope in reversed(self._scope_stack):
            if any(n.id == scope and n.symbol_type == SymbolType.CLASS for n in self._nodes):
                return scope
        return None

    # -- Classes -----------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = f"{self._current_scope}.{node.name}"
        self._add_node(NodeData(
            id=qname, name=node.name, qualified_name=qname,
            symbol_type=SymbolType.CLASS, file_path=self._file,
            line_start=node.lineno, line_end=node.end_lineno,
            docstring=ast.get_docstring(node),
        ))
        self._edges.append(EdgeData(
            source=self._current_scope, target=qname,
            edge_type=EdgeType.CONTAINS, line=node.lineno,
        ))
        for base in node.bases:
            base_name = _name_from_node(base)
            if base_name:
                self._edges.append(EdgeData(
                    source=qname, target=base_name,
                    edge_type=EdgeType.INHERITS, line=node.lineno,
                ))
        # Class decorators are calls from the enclosing scope
        for deco in node.decorator_list:
            deco_name = _name_from_node(deco.func if isinstance(deco, ast.Call) else deco)
            if deco_name:
                self._edges.append(EdgeData(
                    source=self._current_scope, target=deco_name,
                    edge_type=EdgeType.CALLS, line=deco.lineno,
                ))
        self._scope_stack.append(qname)
        # Extract calls and variable access at class body scope
        self._extract_scope_level_calls(node, qname)
        self.generic_visit(node)
        self._scope_stack.pop()

    # -- Functions / Methods -----------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_funcdef(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_funcdef(node)

    def _handle_funcdef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qname = f"{self._current_scope}.{node.name}"
        is_method = self._is_inside_class()
        sym_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION
        sig = _signature_from_funcdef(node)

        self._add_node(NodeData(
            id=qname, name=node.name, qualified_name=qname,
            symbol_type=sym_type, file_path=self._file,
            line_start=node.lineno, line_end=node.end_lineno,
            docstring=ast.get_docstring(node), signature=sig,
        ))
        self._edges.append(EdgeData(
            source=self._current_scope, target=qname,
            edge_type=EdgeType.CONTAINS, line=node.lineno,
        ))

        # Decorators are calls — @app.post("/x") calls app.post, and the
        # decorated function is the implicit argument. This means the
        # enclosing scope calls the decorator, and the decorator calls the
        # function being defined (making it reachable).
        for deco in node.decorator_list:
            deco_name = _name_from_node(deco.func if isinstance(deco, ast.Call) else deco)
            if deco_name:
                # Enclosing scope calls the decorator
                self._edges.append(EdgeData(
                    source=self._current_scope, target=deco_name,
                    edge_type=EdgeType.CALLS, line=deco.lineno,
                ))
                # Decorator calls (registers) the decorated function
                self._edges.append(EdgeData(
                    source=deco_name, target=qname,
                    edge_type=EdgeType.CALLS, line=deco.lineno,
                ))

        self._extract_parameters(node, qname)

        if node.name == "__init__" and is_method:
            self._extract_instance_attrs(node)

        self._scope_stack.append(qname)
        self._extract_calls(node, qname)
        self._extract_variable_access(node, qname)
        self._extract_returns(node, qname)
        self.generic_visit(node)
        self._scope_stack.pop()

    # -- Imports -----------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._edges.append(EdgeData(
                source=self._module, target=alias.name,
                edge_type=EdgeType.IMPORTS, line=node.lineno,
            ))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        for alias in node.names:
            target = f"{base}.{alias.name}" if base else alias.name
            self._edges.append(EdgeData(
                source=self._module, target=target,
                edge_type=EdgeType.IMPORTS, line=node.lineno,
            ))

    # -- Assignments at module / class scope --------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._current_scope == self._module or self._is_inside_class():
            for target in node.targets:
                for name in _assigned_names(target):
                    qname = f"{self._current_scope}.{name}"
                    self._add_node(NodeData(
                        id=qname, name=name, qualified_name=qname,
                        symbol_type=SymbolType.VARIABLE, file_path=self._file,
                        line_start=node.lineno, line_end=node.end_lineno,
                    ))
                    self._edges.append(EdgeData(
                        source=self._current_scope, target=qname,
                        edge_type=EdgeType.CONTAINS, line=node.lineno,
                    ))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._current_scope == self._module or self._is_inside_class():
            if node.target:
                name = _name_from_node(node.target)
                if name and "." not in name:
                    qname = f"{self._current_scope}.{name}"
                    self._add_node(NodeData(
                        id=qname, name=name, qualified_name=qname,
                        symbol_type=SymbolType.VARIABLE, file_path=self._file,
                        line_start=node.lineno, line_end=node.end_lineno,
                    ))
                    self._edges.append(EdgeData(
                        source=self._current_scope, target=qname,
                        edge_type=EdgeType.CONTAINS, line=node.lineno,
                    ))
        self.generic_visit(node)

    # -- Scope-level calls (module / class body) --------------------------

    def _extract_scope_level_calls(self, scope_node: ast.AST, scope_qname: str) -> None:
        """Extract calls and reads from module-level or class-level statements.

        These are statements that execute when the module imports or the class
        body runs — NOT inside function/method bodies. This captures things like
        module-level function calls, class-level assignments with calls, etc.
        We only look at direct children of the scope, not nested function bodies.
        """
        body = getattr(scope_node, "body", [])
        for stmt in body:
            # Skip function/class defs — their bodies don't run at scope level
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # Walk this statement for calls
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    callee = _name_from_node(node.func)
                    if callee:
                        self._edges.append(EdgeData(
                            source=scope_qname, target=callee,
                            edge_type=EdgeType.CALLS,
                            line=getattr(node, "lineno", None),
                        ))

    # -- Internal helpers --------------------------------------------------

    def _extract_instance_attrs(self, init_node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Create VARIABLE nodes for self.X = ... and self.X: T = ... in __init__."""
        class_scope = self._class_scope()
        if not class_scope:
            return
        for node in ast.walk(init_node):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.target:
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == "self":
                        attr_qname = f"{class_scope}.{target.attr}"
                        self._add_node(NodeData(
                            id=attr_qname, name=target.attr,
                            qualified_name=attr_qname,
                            symbol_type=SymbolType.VARIABLE,
                            file_path=self._file,
                            line_start=node.lineno,
                            line_end=getattr(node, "end_lineno", None),
                        ))
                        self._edges.append(EdgeData(
                            source=class_scope, target=attr_qname,
                            edge_type=EdgeType.CONTAINS, line=node.lineno,
                        ))

    def _extract_parameters(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, func_qname: str) -> None:
        args = func_node.args
        all_args: list[ast.arg] = (
            args.posonlyargs + args.args + args.kwonlyargs
        )
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        for arg in all_args:
            if arg.arg in ("self", "cls"):
                continue
            param_qname = f"{func_qname}.{arg.arg}"
            self._add_node(NodeData(
                id=param_qname, name=arg.arg, qualified_name=param_qname,
                symbol_type=SymbolType.PARAMETER, file_path=self._file,
                line_start=func_node.lineno,
            ))
            self._edges.append(EdgeData(
                source=func_qname, target=param_qname,
                edge_type=EdgeType.CONTAINS, line=func_node.lineno,
            ))

    def _extract_calls(self, func_node: ast.AST, caller_qname: str) -> None:
        """Emit raw CALLS edges. Targets are unresolved (e.g. 'self.add_node')."""
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            callee = _name_from_node(node.func)
            if not callee:
                continue

            arg_names: list[str] = []
            for arg in node.args:
                aname = _name_from_node(arg)
                if aname:
                    arg_names.append(aname)
            kwarg_names: dict[str, str] = {}
            for kw in node.keywords:
                if kw.arg:
                    vname = _name_from_node(kw.value)
                    if vname:
                        kwarg_names[kw.arg] = vname

            metadata: dict[str, Any] = {}
            if arg_names:
                metadata["args"] = arg_names
            if kwarg_names:
                metadata["kwargs"] = kwarg_names

            self._edges.append(EdgeData(
                source=caller_qname, target=callee,
                edge_type=EdgeType.CALLS,
                line=getattr(node, "lineno", None),
                metadata=metadata,
            ))

    def _extract_variable_access(self, func_node: ast.AST, scope_qname: str) -> None:
        """Emit raw READS/WRITES edges. Targets are unresolved."""
        param_names: set[str] = set()
        local_names: set[str] = set()

        if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (func_node.args.posonlyargs + func_node.args.args +
                        func_node.args.kwonlyargs):
                if arg.arg not in ("self", "cls"):
                    param_names.add(arg.arg)
            if func_node.args.vararg:
                param_names.add(func_node.args.vararg.arg)
            if func_node.args.kwarg:
                param_names.add(func_node.args.kwarg.arg)

        # Collect local assignment targets
        # Skip _ everywhere — Python's universal "intentional discard" convention
        _SKIP = {"self", "cls", "_"} | _BUILTINS
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    for name in _assigned_names(t):
                        if name not in _SKIP:
                            local_names.add(name)
            elif isinstance(node, ast.AugAssign):
                name = _name_from_node(node.target)
                if name and name not in _SKIP and "." not in name:
                    local_names.add(name)
            elif isinstance(node, ast.AnnAssign) and node.value and node.target:
                name = _name_from_node(node.target)
                if name and name not in _SKIP and "." not in name:
                    local_names.add(name)
            # For-loop targets
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name) and node.target.id not in _SKIP:
                    local_names.add(node.target.id)
                elif isinstance(node.target, (ast.Tuple, ast.List)):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name) and elt.id not in _SKIP:
                            local_names.add(elt.id)

            # Comprehension loop variables (listcomp, setcomp, genexpr, dictcomp)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in node.generators:
                    if isinstance(gen.target, ast.Name) and gen.target.id not in _SKIP:
                        local_names.add(gen.target.id)
                    elif isinstance(gen.target, (ast.Tuple, ast.List)):
                        for elt in gen.target.elts:
                            if isinstance(elt, ast.Name) and elt.id not in _SKIP:
                                local_names.add(elt.id)

        # Create local variable nodes (not params -- those already exist)
        for lname in local_names:
            if lname not in param_names:
                lvar_qname = f"{scope_qname}.{lname}"
                self._add_node(NodeData(
                    id=lvar_qname, name=lname, qualified_name=lvar_qname,
                    symbol_type=SymbolType.VARIABLE, file_path=self._file,
                    line_start=getattr(func_node, "lineno", None),
                ))
                self._edges.append(EdgeData(
                    source=scope_qname, target=lvar_qname,
                    edge_type=EdgeType.CONTAINS,
                    line=getattr(func_node, "lineno", None),
                ))

        known_locals = param_names | local_names

        # Emit raw reads/writes
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._emit_raw_write(target, scope_qname, known_locals, node.lineno)
            elif isinstance(node, ast.AugAssign):
                self._emit_raw_write(node.target, scope_qname, known_locals, node.lineno)
                # AugAssign also reads
                name = _name_from_node(node.target)
                if name:
                    raw = self._raw_target(name, scope_qname, known_locals)
                    if raw:
                        self._edges.append(EdgeData(
                            source=scope_qname, target=raw,
                            edge_type=EdgeType.READS, line=node.lineno,
                        ))
            elif isinstance(node, ast.AnnAssign) and node.value and node.target:
                self._emit_raw_write(node.target, scope_qname, known_locals, node.lineno)

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in _BUILTINS or node.id in ("self", "cls"):
                    continue
                raw = self._raw_target(node.id, scope_qname, known_locals)
                if raw:
                    self._edges.append(EdgeData(
                        source=scope_qname, target=raw,
                        edge_type=EdgeType.READS, line=node.lineno,
                    ))

            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                dotted = _name_from_node(node)
                if dotted and dotted.split(".")[0] not in _BUILTINS:
                    self._edges.append(EdgeData(
                        source=scope_qname, target=dotted,
                        edge_type=EdgeType.READS, line=node.lineno,
                    ))

    def _emit_raw_write(self, target: ast.AST, scope_qname: str, known_locals: set[str], lineno: int) -> None:
        if isinstance(target, ast.Name):
            if target.id in _BUILTINS or target.id in ("self", "cls"):
                return
            raw = self._raw_target(target.id, scope_qname, known_locals)
            if raw:
                self._edges.append(EdgeData(
                    source=scope_qname, target=raw,
                    edge_type=EdgeType.WRITES, line=lineno,
                ))
        elif isinstance(target, ast.Attribute):
            dotted = _name_from_node(target)
            if dotted and dotted.split(".")[0] not in _BUILTINS:
                self._edges.append(EdgeData(
                    source=scope_qname, target=dotted,
                    edge_type=EdgeType.WRITES, line=lineno,
                ))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._emit_raw_write(elt, scope_qname, known_locals, lineno)

    @staticmethod
    def _raw_target(name: str, scope_qname: str, known_locals: set[str]) -> str | None:
        """Return the raw edge target for a bare name.

        Known locals/params get scope-qualified so they match their node IDs.
        Everything else is left bare for the resolver.
        """
        if name in _BUILTINS:
            return None
        if name in known_locals:
            return f"{scope_qname}.{name}"
        return name

    def _extract_returns(self, func_node: ast.AST, func_qname: str) -> None:
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value:
                ret_name = _name_from_node(node.value)
                if ret_name:
                    self._edges.append(EdgeData(
                        source=func_qname, target=ret_name,
                        edge_type=EdgeType.RETURNS, line=node.lineno,
                    ))



# ---------------------------------------------------------------------------
# Pass 2: Type inference and unified resolution
# ---------------------------------------------------------------------------

class _TypeInferencer:
    """Collects type annotations and resolves ALL edge targets.

    Single point of resolution for every name pattern:
      - "self.X"           -> Class.X  (via scope)
      - "self.X.Y"         -> resolve X's type, then Type.Y
      - "n.id"             -> look up n's type, then NodeData.id
      - "graph.all_nodes"  -> look up graph's type, then CodeGraph.all_nodes
      - bare "result"      -> already scope-qualified by visitor
      - "ast.Name"         -> filtered (external)
    """

    def __init__(self, type_index: dict[str, str], node_ids: set[str]) -> None:
        self._type_index = type_index
        self._node_ids = node_ids
        # (func_qname, var_name) -> class_qname
        self._var_types: dict[tuple[str, str], str] = {}
        # func_qname -> return type annotation AST
        self._return_types: dict[str, ast.AST] = {}

    # -- Annotation helpers ------------------------------------------------

    def _resolve_annotation(self, ann: ast.AST) -> str | None:
        """Extract the resolved class qname from a type annotation."""
        if isinstance(ann, ast.Name):
            return self._type_index.get(ann.id)
        if isinstance(ann, ast.Attribute):
            dotted = _name_from_node(ann)
            return self._type_index.get(dotted) if dotted else None
        if isinstance(ann, ast.Subscript):
            return self._resolve_annotation(ann.value)
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            return self._resolve_annotation(ann.left) or self._resolve_annotation(ann.right)
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            return self._type_index.get(ann.value)
        return None

    def _resolve_subscript_inner(self, ann: ast.AST) -> str | None:
        """For list[NodeData] or set[X], resolve the element type."""
        if isinstance(ann, ast.Subscript):
            sl = ann.slice
            if isinstance(sl, ast.Name):
                return self._type_index.get(sl.id)
            if isinstance(sl, ast.Attribute):
                dotted = _name_from_node(sl)
                return self._type_index.get(dotted) if dotted else None
            # dict[K, V] -- return V for .values() iteration
            if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                return self._resolve_annotation(sl.elts[-1])
        # Handle X | None wrapping
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            return self._resolve_subscript_inner(ann.left) or self._resolve_subscript_inner(ann.right)
        return None

    # -- Type collection ---------------------------------------------------

    def collect_types(self, tree: ast.Module, module_qname: str) -> None:
        """Walk an AST and collect all type information.

        Two sub-passes per module:
          1. Collect ALL return type annotations (so method call assignments
             can look up return types regardless of definition order).
          2. Collect param types, local types, assignments, for-loop types.
        """
        # Precompute func node id -> qualified name in one walk (O(n) not O(n²))
        func_qnames = self._build_func_qname_map(tree, module_qname)

        # Sub-pass 1: return types for every function in this module
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fq = func_qnames.get(id(node))
                if fq and node.returns:
                    self._return_types[fq] = node.returns

        # Sub-pass 2: everything else
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_qname = func_qnames.get(id(node))
            if not func_qname:
                continue

            # Parameter annotations
            param_annotations: dict[str, ast.AST] = {}
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
                if arg.arg in ("self", "cls") or not arg.annotation:
                    continue
                param_annotations[arg.arg] = arg.annotation
                resolved = self._resolve_annotation(arg.annotation)
                if resolved:
                    self._var_types[(func_qname, arg.arg)] = resolved

            # Local annotations and constructor assignments
            local_annotations: dict[str, ast.AST] = {}
            for child in ast.walk(node):
                if isinstance(child, ast.AnnAssign) and child.target:
                    name = _name_from_node(child.target)
                    if not name or name in ("self", "cls"):
                        continue
                    if name.startswith("self."):
                        attr = name.split(".", 1)[1]
                        resolved = self._resolve_annotation(child.annotation)
                        if resolved:
                            self._var_types[(func_qname, attr)] = resolved
                    elif "." not in name:
                        local_annotations[name] = child.annotation
                        resolved = self._resolve_annotation(child.annotation)
                        if resolved:
                            self._var_types[(func_qname, name)] = resolved

                if isinstance(child, ast.Assign) and len(child.targets) == 1:
                    target = child.targets[0]
                    if isinstance(child.value, ast.Call):
                        callee = _name_from_node(child.value.func)
                        if callee:
                            # Case A: constructor call — x = CodeGraph()
                            resolved = self._type_index.get(callee)
                            if resolved:
                                if isinstance(target, ast.Name):
                                    self._var_types[(func_qname, target.id)] = resolved
                                elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                    if target.value.id in ("self", "cls"):
                                        self._var_types[(func_qname, target.attr)] = resolved

                            # Case B: method call — x = obj.method()
                            elif "." in callee and isinstance(target, ast.Name):
                                obj_name, method = callee.rsplit(".", 1)
                                cls = self._resolve_var_type(obj_name, func_qname)
                                if cls:
                                    method_qname = f"{cls}.{method}"
                                    ret_ann = self._return_types.get(method_qname)
                                    if ret_ann:
                                        ret_type = self._resolve_annotation(ret_ann)
                                        if ret_type:
                                            self._var_types[(func_qname, target.id)] = ret_type
                                        local_annotations[target.id] = ret_ann

                    # Case C: assignment type propagation
                    # self.x = param  or  x = other_typed_var
                    elif isinstance(child.value, ast.Name):
                        rhs_name = child.value.id
                        rhs_type = self._resolve_var_type(rhs_name, func_qname)
                        if not rhs_type:
                            # Check param annotations directly
                            rhs_ann = param_annotations.get(rhs_name)
                            if rhs_ann:
                                rhs_type = self._resolve_annotation(rhs_ann)
                                # Also store raw annotation for subscript extraction
                                if isinstance(target, ast.Name):
                                    local_annotations[target.id] = rhs_ann
                        if rhs_type:
                            if isinstance(target, ast.Name):
                                self._var_types[(func_qname, target.id)] = rhs_type
                            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                if target.value.id in ("self", "cls"):
                                    self._var_types[(func_qname, target.attr)] = rhs_type

            # For-loop and comprehension element type inference
            for child in ast.walk(node):
                if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
                    elem_type = self._infer_iter_element_type(
                        child.iter, func_qname, local_annotations, param_annotations,
                    )
                    if elem_type:
                        self._var_types[(func_qname, child.target.id)] = elem_type

                elif isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                    for gen in child.generators:
                        if isinstance(gen.target, ast.Name):
                            elem_type = self._infer_iter_element_type(
                                gen.iter, func_qname, local_annotations, param_annotations,
                            )
                            if elem_type:
                                self._var_types[(func_qname, gen.target.id)] = elem_type

    def _infer_iter_element_type(
        self,
        it: ast.AST,
        func_qname: str,
        local_annotations: dict[str, ast.AST],
        param_annotations: dict[str, ast.AST],
    ) -> str | None:
        """Infer element type from an iterator expression."""

        # Case 1: for x in local_var -- check annotations for list[X]
        if isinstance(it, ast.Name):
            for ann_map in (local_annotations, param_annotations):
                ann = ann_map.get(it.id)
                if ann:
                    # Strip X | None
                    actual = ann
                    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
                        actual = ann.left
                    inner = self._resolve_subscript_inner(actual)
                    if inner:
                        return inner

        # Case 2: for x in obj.method() -- resolve obj type, look up method return
        if isinstance(it, ast.Call):
            callee = _name_from_node(it.func)
            if callee and "." in callee:
                obj_name, method = callee.rsplit(".", 1)
                cls = self._resolve_var_type(obj_name, func_qname)
                if cls:
                    method_qname = f"{cls}.{method}"
                    ret_ann = self._return_types.get(method_qname)
                    if ret_ann:
                        inner = self._resolve_subscript_inner(ret_ann)
                        if inner:
                            return inner

        # Case 3: for x in obj.values() on dict[K, V]
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):
            if it.func.attr == "values":
                obj = _name_from_node(it.func.value)
                if obj:
                    for ann_map in (local_annotations, param_annotations):
                        ann = ann_map.get(obj)
                        if ann and isinstance(ann, ast.Subscript):
                            sl = ann.slice
                            if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                                return self._resolve_annotation(sl.elts[-1])

        return None

    def _resolve_var_type(self, name: str, func_qname: str) -> str | None:
        """Look up a variable's type, handling 'self'/'cls', dotted chains, and scope walking.

        Handles:
          - 'self' / 'cls' -> enclosing class
          - 'graph' -> param/local type lookup
          - 'self.graph' -> resolve self to class, then look up 'graph' attr type
        """
        if name in ("self", "cls"):
            parts = func_qname.split(".")
            for i in range(len(parts) - 1, 0, -1):
                candidate = ".".join(parts[:i])
                if candidate in self._type_index.values():
                    return candidate
            return None

        # Handle dotted names: self.graph, self._node_data, etc.
        if "." in name:
            parts = name.split(".")
            root_type = self._resolve_var_type(parts[0], func_qname)
            if root_type:
                # Walk the chain resolving each attribute's type
                current_type = root_type
                for attr in parts[1:]:
                    attr_type = self._find_attr_type(current_type, attr)
                    if attr_type:
                        current_type = attr_type
                    else:
                        return None
                return current_type
            return None

        scope = func_qname
        while scope:
            key = (scope, name)
            if key in self._var_types:
                return self._var_types[key]
            if "." in scope:
                scope = scope.rsplit(".", 1)[0]
            else:
                break
        return None

    # -- Structural type inference -----------------------------------------

    def infer_structural_types(self, edges: list[EdgeData]) -> None:
        """Infer variable types from field access patterns (reverse index).

        For untyped variable 'n' where we see n.id, n.symbol_type, n.qualified_name,
        build a reverse index field_name -> {classes that have that field as a child
        node}, intersect all accessed fields, and if there's a unique class match,
        type 'n' as that class.
        """
        # Build reverse index: field_name -> set of parent class qnames
        field_to_classes: dict[str, set[str]] = {}
        for nid in self._node_ids:
            parts = nid.rsplit(".", 1)
            if len(parts) == 2:
                parent, field = parts
                # Only consider class children (not module-level or function locals)
                if parent in self._node_ids:
                    field_to_classes.setdefault(field, set()).add(parent)

        # Collect field accesses per (scope, variable) for untyped variables
        # An edge target like "n.id" from scope S means variable 'n' in S accesses field 'id'
        var_fields: dict[tuple[str, str], set[str]] = {}  # (scope, var) -> {field1, field2, ...}
        for e in edges:
            if e.edge_type not in (EdgeType.READS, EdgeType.WRITES):
                continue
            t = e.target
            if "." not in t:
                continue
            parts = t.split(".")
            if parts[0] in ("self", "cls"):
                continue
            var_name = parts[0]
            field = parts[1]  # first attribute access
            key = (e.source, var_name)
            # Only care about variables we haven't already typed
            if key not in self._var_types:
                var_fields.setdefault(key, set()).add(field)

        # Intersect: for each untyped var, find classes that have ALL its accessed fields
        for (scope, var_name), fields in var_fields.items():
            if not fields:
                continue
            # Find classes that have ALL these fields as children
            candidate_classes: set[str] | None = None
            for field in fields:
                classes_with_field = field_to_classes.get(field)
                if classes_with_field is None:
                    candidate_classes = set()
                    break
                if candidate_classes is None:
                    candidate_classes = set(classes_with_field)
                else:
                    candidate_classes &= classes_with_field
                if not candidate_classes:
                    break

            if candidate_classes and len(candidate_classes) == 1:
                cls = next(iter(candidate_classes))
                self._var_types[(scope, var_name)] = cls

    # -- Unified resolution ------------------------------------------------

    def resolve(self, target: str, source: str) -> str | None:
        """Resolve a raw edge target to a qualified node ID.

        Returns None if the target should be filtered out (builtin).
        Returns the resolved target (may be progressively truncated).
        """
        # Already a known node
        if target in self._node_ids:
            return target

        # Filter builtins
        root = target.split(".")[0]
        if root in _BUILTINS:
            return None

        # Bare name (no dots) -- leave for build_from
        if "." not in target:
            return target

        # Dotted name -- resolve through type system
        parts = target.split(".")
        first = parts[0]

        # Handle self.X / cls.X chains
        if first in ("self", "cls"):
            cls_qname = self._resolve_var_type(first, source)
            if cls_qname:
                resolved = self._resolve_attr_chain(cls_qname, parts[1:])
                if resolved:
                    return resolved
            return target

        # Handle typed variable chains: graph.all_nodes, n.id, etc.
        var_type = self._resolve_var_type(first, source)
        if var_type:
            resolved = self._resolve_attr_chain(var_type, parts[1:])
            if resolved:
                return resolved

        # Untyped root -- return raw for progressive truncation later
        return target

    def _resolve_attr_chain(self, class_qname: str, attrs: list[str]) -> str | None:
        """Resolve an attribute chain against a known class.

        Returns the deepest resolvable node ID, or None if nothing resolves.
        Uses progressive truncation: tries full chain first, then strips from
        the right until a known node is found.

        Example: class_qname='CodeGraph', attrs=['_edges', 'append']
          1. Try 'CodeGraph._edges.append' -- not a node
          2. Try 'CodeGraph._edges' -- IS a node -> return it
        """
        if not attrs:
            return class_qname if class_qname in self._node_ids else None

        # Try full chain first
        full = class_qname + "." + ".".join(attrs)
        if full in self._node_ids:
            return full

        # Try type-based resolution for deeper chains
        first = attrs[0]
        first_resolved = f"{class_qname}.{first}"

        if len(attrs) > 1:
            attr_type = self._find_attr_type(class_qname, first)
            if attr_type:
                deeper = self._resolve_attr_chain(attr_type, attrs[1:])
                if deeper and deeper in self._node_ids:
                    return deeper

        # Progressive truncation: strip from right until we hit a node
        for i in range(len(attrs), 0, -1):
            candidate = class_qname + "." + ".".join(attrs[:i])
            if candidate in self._node_ids:
                return candidate

        # The class itself is a node
        if class_qname in self._node_ids:
            return class_qname

        return first_resolved  # best-effort

    def _find_attr_type(self, class_qname: str, attr_name: str) -> str | None:
        """Find the type of a class attribute via __init__ or class-level scope."""
        for scope in (f"{class_qname}.__init__", class_qname):
            key = (scope, attr_name)
            if key in self._var_types:
                return self._var_types[key]
        return None

    @staticmethod
    def _build_func_qname_map(tree: ast.Module, module_qname: str) -> dict[int, str]:
        """Single-pass precomputation: map func node id() -> qualified name.

        Replaces the O(n²) _find_func_qname which did a full AST walk per function.
        """
        result: dict[int, str] = {}
        class _Mapper(ast.NodeVisitor):
            def __init__(self):
                self.stack = [module_qname]
            def visit_ClassDef(self, node):
                self.stack.append(f"{self.stack[-1]}.{node.name}")
                self.generic_visit(node)
                self.stack.pop()
            def visit_FunctionDef(self, node):
                qname = f"{self.stack[-1]}.{node.name}"
                result[id(node)] = qname
                self.stack.append(qname)
                self.generic_visit(node)
                self.stack.pop()
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
        _Mapper().visit(tree)
        return result


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _name_from_node(node: ast.AST) -> str | None:
    """Extract a dotted name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name_from_node(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _assigned_names(target: ast.AST) -> list[str]:
    """Get flat list of names from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_assigned_names(elt))
        return names
    return []


def _signature_from_funcdef(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a human-readable signature string."""
    args = node.args
    parts: list[str] = []

    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")

    n_defaults = len(args.defaults)
    n_regular = len(args.args)
    for i, a in enumerate(args.args):
        default_idx = i - (n_regular - n_defaults)
        if default_idx >= 0:
            parts.append(f"{a.arg}=...")
        else:
            parts.append(a.arg)

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    for i, a in enumerate(args.kwonlyargs):
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            parts.append(f"{a.arg}=...")
        else:
            parts.append(a.arg)

    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)})"
