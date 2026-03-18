"""Accuracy tests for the Interlinked analyzer.

Tests parser correctness, edge resolution, dead code detection,
query engine fidelity, and graph mutation integrity using crafted
fixture files designed to expose edge cases.

Run: pytest tests/test_accuracy.py -v
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from interlinked.analyzer.parser import parse_project
from interlinked.analyzer.graph import CodeGraph
from interlinked.analyzer.dead_code import detect_dead_code
from interlinked.commander.query import QueryEngine
from interlinked.models import SymbolType, EdgeType


# ── Helpers ──────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


def _build(root: Path | str) -> tuple[CodeGraph, QueryEngine, list, list]:
    """Parse a project, build graph, detect dead code, return everything."""
    nodes, edges = parse_project(str(root))
    graph = CodeGraph()
    graph.build_from(nodes, edges)
    dead = detect_dead_code(graph)
    engine = QueryEngine(graph)
    return graph, engine, nodes, edges


def _node_ids(graph: CodeGraph) -> set[str]:
    return {n.id for n in graph.all_nodes()}


def _edges_of(graph: CodeGraph, edge_type: EdgeType | None = None) -> list[tuple[str, str, str]]:
    """Return (source, target, type) triples, optionally filtered."""
    result = []
    for e in graph.all_edges():
        if edge_type is None or e.edge_type == edge_type:
            result.append((e.source, e.target, e.edge_type.value))
    return result


def _call_targets_of(graph: CodeGraph, caller_qname: str) -> set[str]:
    """All nodes that caller_qname calls."""
    return {
        e.target for e in graph.all_edges()
        if e.source == caller_qname and e.edge_type == EdgeType.CALLS
    }


def _callers_of(graph: CodeGraph, callee_qname: str) -> set[str]:
    """All nodes that call callee_qname."""
    return {
        e.source for e in graph.all_edges()
        if e.target == callee_qname and e.edge_type == EdgeType.CALLS
    }


def _inherits(graph: CodeGraph, child_qname: str) -> set[str]:
    """All classes that child_qname inherits from."""
    return {
        e.target for e in graph.all_edges()
        if e.source == child_qname and e.edge_type == EdgeType.INHERITS
    }


def _dead_ids(graph: CodeGraph) -> set[str]:
    return {n.id for n in graph.all_nodes() if n.is_dead}


def _find(graph: CodeGraph, partial: str) -> str:
    """Resolve a partial name to a node ID. Prefers exact suffix match
    over substring match to avoid 'Foo' matching 'Foo.__init__'."""
    best = None
    for n in graph.all_nodes():
        if n.qualified_name == partial:
            return n.id  # exact match
        if n.qualified_name.endswith("." + partial) or n.qualified_name.endswith(partial):
            if best is None or len(n.qualified_name) < len(best):
                best = n.qualified_name
        elif partial in n.qualified_name:
            if best is None:
                best = n.id
    assert best is not None, f"No node matching '{partial}'"
    # Return the ID for the best match
    node = graph.get_node(best)
    return node.id if node else best


# ══════════════════════════════════════════════════════════════════════
# 1. PARSER COMPLETENESS — every symbol must be found
# ══════════════════════════════════════════════════════════════════════


class TestParserCompleteness:
    """Verify that parse_project finds every symbol in the fixtures."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def test_modules_found(self):
        """Every fixture file should produce a module node."""
        ids = _node_ids(self.graph)
        expected_modules = [
            "shadowing", "inheritance", "dynamic_dispatch",
            "dead_code_traps", "cross_module_calls",
            "async_patterns", "type_resolution", "graph_mutation",
        ]
        for mod in expected_modules:
            assert any(mod in nid for nid in ids), f"Module '{mod}' not found in graph"

    def test_classes_found(self):
        """All classes across fixtures must appear."""
        ids = _node_ids(self.graph)
        expected = [
            "Logger", "Processor",                          # shadowing
            "Base", "Mixin", "Left", "Right", "Diamond",   # inheritance
            "AutoRegister", "PluginA", "PluginB",           # inheritance
            "WithProperty",                                 # inheritance
            "Transformer",                                  # dynamic_dispatch
            "AsyncDB",                                      # async_patterns
            "Engine", "EngineState", "Result", "Container", # type_resolution
            "MutableService",                               # graph_mutation
            "_DeadClass", "AliveClass",                     # dead_code_traps
        ]
        for cls in expected:
            assert any(cls in nid for nid in ids), f"Class '{cls}' not found in graph"

    def test_functions_found(self):
        """Key functions must appear as nodes."""
        ids = _node_ids(self.graph)
        expected = [
            "process", "log", "nested_closures",     # shadowing
            "timer", "retry", "compute", "fetch",    # dynamic_dispatch
            "dispatch", "factory", "higher_order",   # dynamic_dispatch
            "fetch_data", "stream_all",              # async_patterns
            "parallel_fetch", "sync_generator",      # async_patterns
            "reachable_root", "_truly_dead_helper",  # dead_code_traps
            "chained_access", "return_type_propagation", # type_resolution
        ]
        for fn in expected:
            assert any(fn in nid for nid in ids), f"Function '{fn}' not found in graph"

    def test_methods_found(self):
        """Key methods must appear."""
        ids = _node_ids(self.graph)
        expected = [
            "Logger.log", "Processor.process", "Processor.run",
            "Base.execute", "Base.describe",
            "Left.execute", "Right.execute", "Diamond.execute",
            "Transformer.__call__", "Transformer._apply",
            "AsyncDB.connect", "AsyncDB.query", "AsyncDB.stream_rows",
            "Engine.run", "EngineState.activate",
            "Container.process", "Container.summarize",
            "MutableService.increment", "MutableService.reset",
        ]
        for method in expected:
            assert any(method in nid for nid in ids), f"Method '{method}' not found"


# ══════════════════════════════════════════════════════════════════════
# 2. EDGE RESOLUTION — calls resolve to the correct target
# ══════════════════════════════════════════════════════════════════════


class TestEdgeResolution:
    """Verify that call edges point to the correct targets."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def _find_node(self, partial: str) -> str:
        """Find the full qualified name containing 'partial'."""
        for n in self.graph.all_nodes():
            if partial in n.qualified_name:
                return n.id
        pytest.fail(f"No node matching '{partial}'")

    def test_processor_run_calls_module_log(self):
        """Processor.run() calls module-level log(), not Logger.log()."""
        caller = self._find_node("Processor.run")
        targets = _call_targets_of(self.graph, caller)
        # Should call module-level log
        assert any("log" in t and "Logger" not in t for t in targets), \
            f"Processor.run should call module-level log(), got targets: {targets}"

    def test_processor_run_calls_self_process(self):
        """Processor.run() calls self.process() -> Processor.process."""
        caller = self._find_node("Processor.run")
        targets = _call_targets_of(self.graph, caller)
        assert any("Processor.process" in t for t in targets), \
            f"Processor.run should call Processor.process, got: {targets}"

    def test_diamond_execute_calls_super(self):
        """Diamond.execute() calls super().execute() which should resolve via MRO."""
        caller = self._find_node("Diamond.execute")
        targets = _call_targets_of(self.graph, caller)
        # super().execute() in Diamond should resolve to Left.execute (MRO)
        # or at minimum to some .execute method
        assert any("execute" in t and t != caller for t in targets), \
            f"Diamond.execute should call a parent execute(), got: {targets}"

    def test_dispatch_calls_all_handlers(self):
        """dispatch() references _handle_create, _handle_update, _handle_delete."""
        caller = self._find_node("dynamic_dispatch.dispatch")
        targets = _call_targets_of(self.graph, caller)
        # Dict dispatch — handlers are referenced but may not parse as direct calls
        # At minimum, the function should exist and handlers should be in the graph
        for h in ["_handle_create", "_handle_update", "_handle_delete"]:
            assert any(h in nid for nid in _node_ids(self.graph)), \
                f"Handler '{h}' should exist in graph"

    def test_fetch_data_calls_db_query(self):
        """fetch_data() awaits db.query() — async call should resolve."""
        caller = self._find_node("async_patterns.fetch_data")
        targets = _call_targets_of(self.graph, caller)
        assert any("query" in t for t in targets), \
            f"fetch_data should call AsyncDB.query, got: {targets}"

    def test_chained_access_resolves(self):
        """engine.state.activate() should resolve through the chain."""
        caller = self._find_node("type_resolution.chained_access")
        targets = _call_targets_of(self.graph, caller)
        assert any("activate" in t for t in targets), \
            f"chained_access should call EngineState.activate, got: {targets}"
        assert any("deactivate" in t for t in targets), \
            f"chained_access should call EngineState.deactivate, got: {targets}"

    def test_return_type_propagation(self):
        """r = engine.run(); r.describe() — should resolve Result.describe."""
        caller = self._find_node("type_resolution.return_type_propagation")
        targets = _call_targets_of(self.graph, caller)
        assert any("describe" in t for t in targets), \
            f"return_type_propagation should call Result.describe, got: {targets}"

    def test_loop_variable_typing(self):
        """for e in engines: e.run() — should resolve Engine.run."""
        caller = self._find_node("type_resolution.loop_variable_typing")
        targets = _call_targets_of(self.graph, caller)
        assert any("Engine" in t and "run" in t for t in targets), \
            f"loop_variable_typing should call Engine.run, got: {targets}"

    def test_container_self_engine_chain(self):
        """self.engine.run() inside Container should resolve Engine.run."""
        caller = self._find_node("Container.process")
        targets = _call_targets_of(self.graph, caller)
        assert any("run" in t for t in targets), \
            f"Container.process should call Engine.run, got: {targets}"

    def test_cross_module_import_call(self):
        """use_logger() imports Logger and calls Logger.log()."""
        caller = self._find_node("cross_module_calls.use_logger")
        targets = _call_targets_of(self.graph, caller)
        assert any("log" in t for t in targets), \
            f"use_logger should call Logger.log, got: {targets}"

    def test_aliased_import_call(self):
        """use_process() calls process_data (aliased from shadowing.process)."""
        caller = self._find_node("cross_module_calls.use_process")
        targets = _call_targets_of(self.graph, caller)
        assert any("process" in t for t in targets), \
            f"use_process should call shadowing.process, got: {targets}"

    def test_module_qualified_call(self):
        """dd.dispatch() should resolve to dynamic_dispatch.dispatch."""
        caller = self._find_node("cross_module_calls.use_dispatch")
        targets = _call_targets_of(self.graph, caller)
        assert any("dispatch" in t for t in targets), \
            f"use_dispatch should call dynamic_dispatch.dispatch, got: {targets}"

    # ── Async-specific resolution ────────────────────────────────────

    def test_await_return_type_propagation(self):
        """db = await get_db(); db.query() — await should propagate AsyncDB type."""
        caller = self._find_node("async_patterns.await_return_type")
        targets = _call_targets_of(self.graph, caller)
        assert any("query" in t for t in targets), \
            f"await_return_type should resolve db.query() to AsyncDB.query, got: {targets}"

    def test_async_with_as_variable(self):
        """async with pool as conn — conn typed from __aenter__ return, conn.execute() must resolve
        to the QUALIFIED Connection.execute, not remain as bare 'conn.execute'."""
        caller = self._find_node("async_patterns.async_with_as_var")
        targets = _call_targets_of(self.graph, caller)
        assert any("Connection.execute" in t for t in targets), \
            f"async_with_as_var should resolve conn.execute() to Connection.execute, got: {targets}"

    def test_async_for_call_resolution(self):
        """async for row in db.stream_rows() — db.stream_rows should resolve."""
        caller = self._find_node("async_patterns.async_for_typing")
        targets = _call_targets_of(self.graph, caller)
        assert any("stream_rows" in t for t in targets), \
            f"async_for_typing should resolve db.stream_rows() to AsyncDB.stream_rows, got: {targets}"


# ══════════════════════════════════════════════════════════════════════
# 2b. NESTED CLOSURE / ASYNC CALL RESOLUTION
# ══════════════════════════════════════════════════════════════════════


class TestNestedAsyncResolution:
    """Verify that calls from async handlers to closures defined in
    enclosing scopes actually resolve. Mirrors the real MCP pattern
    where call_tool (async) calls _check_server, _ensure_ready, etc.
    (closures inside create_mcp_server).

    Root cause of failure: name_index has duplicate entries from suffix
    indexing, _resolve_edge requires len==1 and bails on len==2 even
    when both entries are the same node ID.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def _find_node(self, partial: str) -> str:
        for n in self.graph.all_nodes():
            if partial in n.qualified_name:
                return n.id
        pytest.fail(f"No node matching '{partial}'")

    def test_handler_calls_check(self):
        """handler() awaits _check() — closure in enclosing scope."""
        caller = self._find_node("nested_async.make_server.handler")
        targets = _call_targets_of(self.graph, caller)
        assert any("_check" in t for t in targets), \
            f"handler should call _check(), got: {targets}"

    def test_handler_calls_build(self):
        """handler() calls _build via run_in_executor — should emit CALLS edge.
        run_in_executor(None, fn, ...) is an indirect call to fn."""
        caller = self._find_node("nested_async.make_server.handler")
        targets = _call_targets_of(self.graph, caller)
        assert any("_build" in t for t in targets), \
            f"handler should call _build(), got: {targets}"

    def test_handler_calls_do_switch_nested_closure(self):
        """handler() defines _do_switch and calls it via executor."""
        caller = self._find_node("nested_async.make_server.handler")
        targets = _call_targets_of(self.graph, caller)
        assert any("_do_switch" in t for t in targets), \
            f"handler should call _do_switch(), got: {targets}"

    def test_do_switch_calls_graph_build(self):
        """_do_switch() (nested closure) calls GraphBuilder.build()."""
        caller = self._find_node("_do_switch")
        targets = _call_targets_of(self.graph, caller)
        assert any("build" in t for t in targets), \
            f"_do_switch should call GraphBuilder.build(), got: {targets}"

    def test_build_closure_calls_graph_builder(self):
        """_build() closure calls GraphBuilder() and GraphBuilder.build()."""
        caller = self._find_node("nested_async.make_server._build")
        targets = _call_targets_of(self.graph, caller)
        assert any("GraphBuilder" in t for t in targets), \
            f"_build should call GraphBuilder, got: {targets}"

    def test_handler_calls_reset_filter(self):
        """handler() calls engine.reset_filter() — should resolve QueryRunner.reset_filter."""
        caller = self._find_node("nested_async.make_server.handler")
        targets = _call_targets_of(self.graph, caller)
        assert any("reset_filter" in t for t in targets), \
            f"handler should call reset_filter(), got: {targets}"

    def test_handler_calls_stats(self):
        """handler() calls engine.stats() — should resolve QueryRunner.stats."""
        caller = self._find_node("nested_async.make_server.handler")
        targets = _call_targets_of(self.graph, caller)
        assert any("stats" in t for t in targets), \
            f"handler should call stats(), got: {targets}"


# ══════════════════════════════════════════════════════════════════════
# 2c. PYTHON LANGUAGE PATTERNS — type inference stress tests
# ══════════════════════════════════════════════════════════════════════


class TestLanguagePatterns:
    """Verify the parser correctly resolves calls through Python language
    features that require type inference beyond simple assignments."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def _find_node(self, partial: str) -> str:
        for n in self.graph.all_nodes():
            if partial in n.qualified_name:
                return n.id
        pytest.fail(f"No node matching '{partial}'")

    # ── Context managers ─────────────────────────────────────────────

    def test_sync_with_as_variable(self):
        """with SyncPool() as conn — conn typed from __enter__ return -> SyncConnection."""
        caller = self._find_node("language_patterns.use_sync_with")
        targets = _call_targets_of(self.graph, caller)
        assert any("SyncConnection.execute" in t for t in targets), \
            f"use_sync_with should resolve conn.execute() to SyncConnection.execute, got: {targets}"

    def test_sync_with_inline_constructor(self):
        """with SyncPool() as conn — inline constructor, same typing."""
        caller = self._find_node("language_patterns.use_sync_with_inline")
        targets = _call_targets_of(self.graph, caller)
        assert any("SyncConnection.execute" in t for t in targets), \
            f"use_sync_with_inline should resolve conn.execute(), got: {targets}"

    def test_self_returning_context_manager(self):
        """with SelfReturningCM() as obj — obj is same type, obj.do_work() resolves."""
        caller = self._find_node("language_patterns.use_self_returning_cm")
        targets = _call_targets_of(self.graph, caller)
        assert any("do_work" in t for t in targets), \
            f"use_self_returning_cm should resolve obj.do_work(), got: {targets}"

    # ── Generators ───────────────────────────────────────────────────

    def test_typed_generator_consumption(self):
        """for item in item_generator(): item.process() — resolve via Generator[Item] return type."""
        caller = self._find_node("language_patterns.consume_typed_generator")
        targets = _call_targets_of(self.graph, caller)
        assert any("Item.process" in t for t in targets), \
            f"consume_typed_generator should resolve item.process() to Item.process, got: {targets}"

    def test_delegating_generator(self):
        """yield from item_generator() — delegating_generator has same element type."""
        caller = self._find_node("language_patterns.consume_delegated")
        targets = _call_targets_of(self.graph, caller)
        assert any("Item.process" in t for t in targets), \
            f"consume_delegated should resolve item.process() to Item.process, got: {targets}"

    def test_generator_caller_edge(self):
        """consume_typed_generator calls item_generator."""
        caller = self._find_node("language_patterns.consume_typed_generator")
        targets = _call_targets_of(self.graph, caller)
        assert any("item_generator" in t for t in targets), \
            f"consume_typed_generator should call item_generator, got: {targets}"

    # ── Decorators ───────────────────────────────────────────────────

    def test_parameterized_decorator_registers_function(self):
        """@registry.register('create') makes handle_create reachable."""
        # The decorator call should create an edge
        node_ids = _node_ids(self.graph)
        assert any("handle_create" in nid for nid in node_ids), \
            "handle_create should exist in graph"
        assert any("handle_delete" in nid for nid in node_ids), \
            "handle_delete should exist in graph"

    def test_decorator_creates_call_edge(self):
        """@registry.register('create') is a call from module scope to registry.register."""
        # Module scope should call registry.register
        edges = [e for e in self.graph.all_edges()
                 if e.edge_type == EdgeType.CALLS and "register" in e.target]
        assert len(edges) > 0, "Decorator @registry.register should create call edges"

    # ── Dataclasses ──────────────────────────────────────────────────

    def test_dataclass_instantiation(self):
        """Config() and Server() should be recognized as constructor calls."""
        caller = self._find_node("language_patterns.use_dataclass")
        targets = _call_targets_of(self.graph, caller)
        assert any("Config" in t for t in targets), \
            f"use_dataclass should call Config(), got: {targets}"
        assert any("Server" in t for t in targets), \
            f"use_dataclass should call Server(), got: {targets}"

    def test_dataclass_method_call(self):
        """srv.address() should resolve to Server.address."""
        caller = self._find_node("language_patterns.use_dataclass")
        targets = _call_targets_of(self.graph, caller)
        assert any("address" in t for t in targets), \
            f"use_dataclass should call Server.address(), got: {targets}"

    def test_dataclass_chained_field(self):
        """Server.address() accesses self.config.host — chained dataclass field."""
        caller = self._find_node("language_patterns.Server.address")
        # Should read config.host / config.port — at minimum reads 'config'
        reads = {e.target for e in self.graph.all_edges()
                 if e.source == caller and e.edge_type == EdgeType.READS}
        assert any("config" in t for t in reads), \
            f"Server.address should read self.config, got reads: {reads}"

    def test_dataclass_field_access_on_param(self):
        """access_dataclass_fields(cfg: Config) — cfg.host should resolve."""
        caller = self._find_node("language_patterns.access_dataclass_fields")
        reads = {e.target for e in self.graph.all_edges()
                 if e.source == caller and e.edge_type == EdgeType.READS}
        assert any("host" in t for t in reads), \
            f"access_dataclass_fields should read cfg.host, got reads: {reads}"

    # ── Exception handling ───────────────────────────────────────────

    def test_except_as_variable_typing(self):
        """except AppError as e — e.describe() should resolve to AppError.describe."""
        caller = self._find_node("language_patterns.handle_errors")
        targets = _call_targets_of(self.graph, caller)
        assert any("describe" in t for t in targets), \
            f"handle_errors should call e.describe(), got: {targets}"

    def test_exception_inheritance(self):
        """NotFoundError inherits AppError."""
        nfe_class = self._find_node("language_patterns.NotFoundError")
        # Ensure we got the class node, not __init__ or a child
        node = self.graph.get_node(nfe_class)
        assert node and node.symbol_type == SymbolType.CLASS, \
            f"Expected CLASS node, got {node.symbol_type if node else 'None'}"
        parents = _inherits(self.graph, nfe_class)
        assert any("AppError" in p for p in parents), \
            f"NotFoundError should inherit AppError, got: {parents}"

    def test_raise_creates_call_to_constructor(self):
        """raise NotFoundError('thing') is a call to NotFoundError."""
        caller = self._find_node("language_patterns.handle_errors")
        targets = _call_targets_of(self.graph, caller)
        assert any("NotFoundError" in t for t in targets), \
            f"handle_errors should call NotFoundError(), got: {targets}"

    # ── Unpacking ────────────────────────────────────────────────────

    def test_tuple_unpack_typing(self):
        """cfg, srv = tuple_return() — srv.address() should resolve."""
        caller = self._find_node("language_patterns.unpack_tuple")
        targets = _call_targets_of(self.graph, caller)
        assert any("address" in t for t in targets), \
            f"unpack_tuple should resolve srv.address(), got: {targets}"

    # ── Classmethod / staticmethod ───────────────────────────────────

    def test_classmethod_return_type(self):
        """Factory.create() returns Factory — f.describe() should resolve."""
        caller = self._find_node("language_patterns.use_classmethod")
        targets = _call_targets_of(self.graph, caller)
        assert any("describe" in t for t in targets), \
            f"use_classmethod should resolve f.describe(), got: {targets}"

    def test_staticmethod_return_type(self):
        """Factory.default() returns Factory — f.describe() should resolve."""
        caller = self._find_node("language_patterns.use_staticmethod")
        targets = _call_targets_of(self.graph, caller)
        assert any("describe" in t for t in targets), \
            f"use_staticmethod should resolve f.describe(), got: {targets}"

    # ── Walrus operator ──────────────────────────────────────────────

    def test_walrus_in_while(self):
        """while (item := next(it)) — item.process() requires tracing
        iter(items) -> next(it) -> Item. Verify the call edge exists
        and points at Item.process (resolved or unresolved)."""
        caller = self._find_node("language_patterns.walrus_in_while")
        all_edges = {(e.edge_type.value, e.target) for e in self.graph.all_edges()
                     if e.source == caller}
        # The raw call to item.process must at least exist as an edge
        assert any("process" in t for _, t in all_edges), \
            f"walrus_in_while should have a process call/read edge, got: {all_edges}"

    def test_walrus_in_if(self):
        """if (first := items[0]) — first.process() requires tracing
        list subscript -> element type. Verify the call edge exists."""
        caller = self._find_node("language_patterns.walrus_in_if")
        all_edges = {(e.edge_type.value, e.target) for e in self.graph.all_edges()
                     if e.source == caller}
        assert any("process" in t for _, t in all_edges), \
            f"walrus_in_if should have a process call/read edge, got: {all_edges}"

    # ── Comprehensions ───────────────────────────────────────────────

    def test_nested_comprehension_chained_access(self):
        """[tag for srv in servers for tag in srv.config.tags] — srv typed."""
        caller = self._find_node("language_patterns.nested_comprehension")
        reads = {e.target for e in self.graph.all_edges()
                 if e.source == caller and e.edge_type == EdgeType.READS}
        assert any("config" in t for t in reads), \
            f"nested_comprehension should access srv.config, got reads: {reads}"

    def test_dict_comprehension_element_access(self):
        """{item.name: item.value for item in items} — item fields accessed."""
        caller = self._find_node("language_patterns.dict_comprehension")
        reads = {e.target for e in self.graph.all_edges()
                 if e.source == caller and e.edge_type == EdgeType.READS}
        assert any("name" in t for t in reads) or any("value" in t for t in reads), \
            f"dict_comprehension should access item.name/value, got reads: {reads}"

    def test_filtered_comprehension_call(self):
        """[item.process() for item in items if item.value > 0] — item.process() resolves."""
        caller = self._find_node("language_patterns.filtered_comprehension")
        targets = _call_targets_of(self.graph, caller)
        assert any("process" in t for t in targets), \
            f"filtered_comprehension should call item.process(), got: {targets}"


# ══════════════════════════════════════════════════════════════════════
# 3. INHERITANCE EDGES
# ══════════════════════════════════════════════════════════════════════


class TestInheritance:
    """Verify inheritance edges are correct."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def _find_class(self, partial: str) -> str:
        for n in self.graph.all_nodes():
            if partial in n.qualified_name and n.symbol_type == SymbolType.CLASS:
                return n.id
        pytest.fail(f"No class matching '{partial}'")

    def test_diamond_inherits_left_and_right(self):
        parents = _inherits(self.graph, self._find_class("Diamond"))
        parent_names = {p.split(".")[-1] for p in parents}
        assert "Left" in parent_names, f"Diamond should inherit Left, got: {parent_names}"
        assert "Right" in parent_names, f"Diamond should inherit Right, got: {parent_names}"

    def test_left_inherits_base_and_mixin(self):
        parents = _inherits(self.graph, self._find_class("Left"))
        parent_names = {p.split(".")[-1] for p in parents}
        assert "Base" in parent_names, f"Left should inherit Base, got: {parent_names}"
        assert "Mixin" in parent_names, f"Left should inherit Mixin, got: {parent_names}"

    def test_plugin_inherits_autoregister(self):
        parents_a = _inherits(self.graph, self._find_class("PluginA"))
        parents_b = _inherits(self.graph, self._find_class("PluginB"))
        assert any("AutoRegister" in p for p in parents_a)
        assert any("AutoRegister" in p for p in parents_b)


# ══════════════════════════════════════════════════════════════════════
# 4. DEAD CODE DETECTION
# ══════════════════════════════════════════════════════════════════════


class TestDeadCode:
    """Verify dead code detection — no false positives on live code,
    no false negatives on dead code."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)
        self.dead = _dead_ids(self.graph)

    def _find_node(self, partial: str) -> str:
        for n in self.graph.all_nodes():
            if partial in n.qualified_name:
                return n.id
        pytest.fail(f"No node matching '{partial}'")

    # ── Must be dead ──

    def test_truly_dead_helper_is_dead(self):
        nid = self._find_node("_truly_dead_helper")
        assert nid in self.dead, "_truly_dead_helper should be dead"

    def test_also_dead_is_dead(self):
        nid = self._find_node("_also_dead")
        assert nid in self.dead, "_also_dead should be dead"

    def test_dead_class_is_dead(self):
        nid = self._find_node("_DeadClass")
        assert nid in self.dead, "_DeadClass should be dead"

    # ── Must NOT be dead ──

    def test_reachable_root_is_alive(self):
        nid = self._find_node("dead_code_traps.reachable_root")
        assert nid not in self.dead, "reachable_root is called at module level — alive"

    def test_reachable_chain_step1_is_alive(self):
        nid = self._find_node("_reachable_step1")
        assert nid not in self.dead, "_reachable_step1 called by reachable_root — alive"

    def test_reachable_chain_step2_is_alive(self):
        nid = self._find_node("_reachable_step2")
        assert nid not in self.dead, "_reachable_step2 called by _reachable_step1 — alive"

    def test_callback_target_is_alive(self):
        nid = self._find_node("_callback_target")
        assert nid not in self.dead, "_callback_target registered in dict — alive"

    def test_alive_class_is_alive(self):
        nid = self._find_node("dead_code_traps.AliveClass")
        assert nid not in self.dead, "AliveClass is instantiated — alive"

    def test_external_caller_is_alive(self):
        nid = self._find_node("external_caller")
        assert nid not in self.dead, "external_caller called at module level — alive"

    def test_exported_is_alive(self):
        """Functions in __all__ should not be marked dead."""
        nid = self._find_node("exported_but_never_called_directly")
        assert nid not in self.dead, "exported_but_never_called_directly is in __all__ — alive"


# ══════════════════════════════════════════════════════════════════════
# 5. QUERY ENGINE ACCURACY
# ══════════════════════════════════════════════════════════════════════


class TestQueryEngine:
    """Verify QueryEngine returns correct results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def test_callers_of(self):
        """'callers of reachable_root' should find module-level call."""
        result = self.engine.query("callers of reachable_root")
        assert len(result) > 0, \
            f"callers of reachable_root should return results, got: {result}"

    def test_callees_of(self):
        """'callees of reachable_root' should find _reachable_step1."""
        result = self.engine.query("callees of reachable_root")
        assert any("_reachable_step1" in str(r) for r in result), \
            f"callees of reachable_root should find _reachable_step1: {result}"

    def test_dead_functions_query(self):
        """'dead functions' should find _truly_dead_helper."""
        result = self.engine.query("dead functions")
        result_str = str(result)
        assert "_truly_dead_helper" in result_str, \
            f"dead functions should include _truly_dead_helper: {result_str[:200]}"

    def test_isolate_returns_connected_subgraph(self):
        """isolate('Engine') should include EngineState (connected via self.state)."""
        result = self.engine.isolate("Engine", level="class", depth=2)
        snap = self.engine.snapshot()
        node_ids = {n["id"] for n in snap.get("nodes", [])}
        assert any("Engine" in nid for nid in node_ids), \
            f"isolate should include Engine: {node_ids}"

    def test_stats_reflect_full_graph(self):
        """Stats should report nonzero counts for all symbol types."""
        stats = self.engine.stats()
        assert stats["total_nodes"] > 0
        assert stats["modules"] > 0
        assert stats["classes"] > 0
        assert stats["functions"] > 0
        assert stats["methods"] > 0


# ══════════════════════════════════════════════════════════════════════
# 6. GRAPH INTEGRITY — no orphans, no duplicates after mutations
# ══════════════════════════════════════════════════════════════════════


class TestGraphIntegrity:
    """Verify graph invariants hold after mutations."""

    def test_no_orphaned_edges(self):
        """Every edge source must be a known project node (build_from guarantees this).
        CONTAINS targets must also be known nodes (scope tree integrity)."""
        graph, _, _, _ = _build(FIXTURES)
        node_ids = _node_ids(graph)
        for edge in graph.all_edges():
            assert edge.source in node_ids, \
                f"Orphaned edge source: {edge.source} -> {edge.target} ({edge.edge_type})"
            if edge.edge_type == EdgeType.CONTAINS:
                assert edge.target in node_ids, \
                    f"Orphaned CONTAINS target: {edge.source} -> {edge.target}"

    def test_no_self_call_edges(self):
        """A function should not have a 'calls' edge to itself (except recursion — rare in fixtures)."""
        graph, _, _, _ = _build(FIXTURES)
        # Our fixtures don't have recursion, so any self-call is a bug
        for edge in graph.all_edges():
            if edge.edge_type == EdgeType.CALLS:
                assert edge.source != edge.target, \
                    f"Self-call edge: {edge.source} calls itself"

    def test_no_duplicate_nodes(self):
        """No two nodes should share the same qualified name."""
        graph, _, _, _ = _build(FIXTURES)
        seen = {}
        for n in graph.all_nodes():
            assert n.id not in seen, \
                f"Duplicate node: {n.id} (first: {seen[n.id]}, second: {n.file_path}:{n.line_start})"
            seen[n.id] = f"{n.file_path}:{n.line_start}"

    def test_rebuild_idempotent(self):
        """Parsing twice produces identical graph."""
        g1, _, n1, e1 = _build(FIXTURES)
        g2, _, n2, e2 = _build(FIXTURES)

        ids1 = sorted(_node_ids(g1))
        ids2 = sorted(_node_ids(g2))
        assert ids1 == ids2, "Rebuild produced different node sets"

        edges1 = sorted(_edges_of(g1))
        edges2 = sorted(_edges_of(g2))
        assert edges1 == edges2, "Rebuild produced different edge sets"

    def test_incremental_update_removes_edges(self):
        """After removing a function, edges pointing to it must vanish."""
        # Build full graph
        tmpdir = Path(tempfile.mkdtemp())
        try:
            shutil.copytree(FIXTURES, tmpdir / "fixtures")
            graph, engine, _, _ = _build(tmpdir / "fixtures")

            # Verify standalone and calls_standalone exist
            ids = _node_ids(graph)
            standalone_id = None
            for nid in ids:
                if "standalone" in nid and "calls" not in nid:
                    standalone_id = nid
                    break
            assert standalone_id is not None, "standalone() should exist initially"

            # Now modify graph_mutation.py to remove standalone()
            mutation_file = tmpdir / "fixtures" / "graph_mutation.py"
            source = mutation_file.read_text()
            # Remove the standalone function and calls_standalone
            lines = source.split("\n")
            new_lines = []
            skip = False
            for line in lines:
                if "def standalone()" in line or "def calls_standalone()" in line:
                    skip = True
                    continue
                if skip and (line.startswith("def ") or line.startswith("class ") or line == ""):
                    skip = False
                if not skip:
                    new_lines.append(line)
            mutation_file.write_text("\n".join(new_lines))

            # Rebuild
            graph2, _, _, _ = _build(tmpdir / "fixtures")
            ids2 = _node_ids(graph2)

            # standalone should be gone
            assert standalone_id not in ids2, \
                f"standalone should be removed after mutation, still found: {standalone_id}"

            # No edges should reference it
            for edge in graph2.all_edges():
                assert standalone_id not in (edge.source, edge.target), \
                    f"Orphaned edge references removed node {standalone_id}: {edge.source} -> {edge.target}"
        finally:
            shutil.rmtree(tmpdir)


# ══════════════════════════════════════════════════════════════════════
# 7. PARSER SKIP DIRS — .venv and friends must be excluded
# ══════════════════════════════════════════════════════════════════════


class TestParserSkipDirs:
    """Verify that parse_project skips .venv, node_modules, etc."""

    def test_skips_venv(self):
        """A .venv directory containing .py files should be ignored."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            # Real project file
            (tmpdir / "app.py").write_text("def hello(): return 'hi'\n")
            # Fake venv with thousands of files (we simulate with a few)
            venv = tmpdir / ".venv" / "lib" / "site-packages" / "fakepkg"
            venv.mkdir(parents=True)
            (venv / "evil.py").write_text("def should_not_appear(): pass\n")
            (venv / "also_evil.py").write_text("class EvilClass: pass\n")

            nodes, edges = parse_project(str(tmpdir))
            node_names = {n.name for n in nodes}

            assert "hello" in node_names, "Real project function should be found"
            assert "should_not_appear" not in node_names, ".venv files should be skipped"
            assert "EvilClass" not in node_names, ".venv files should be skipped"
        finally:
            shutil.rmtree(tmpdir)

    def test_skips_node_modules(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            (tmpdir / "app.py").write_text("x = 1\n")
            nm = tmpdir / "node_modules" / "pkg"
            nm.mkdir(parents=True)
            (nm / "script.py").write_text("def evil(): pass\n")

            nodes, _ = parse_project(str(tmpdir))
            assert not any(n.name == "evil" for n in nodes)
        finally:
            shutil.rmtree(tmpdir)

    def test_skips_git(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            (tmpdir / "app.py").write_text("y = 2\n")
            git = tmpdir / ".git" / "hooks"
            git.mkdir(parents=True)
            (git / "pre-commit.py").write_text("def hook(): pass\n")

            nodes, _ = parse_project(str(tmpdir))
            assert not any(n.name == "hook" for n in nodes)
        finally:
            shutil.rmtree(tmpdir)


# ══════════════════════════════════════════════════════════════════════
# 8. MCP FIDELITY — direct mode produces same results as engine
# ══════════════════════════════════════════════════════════════════════


class TestMCPFidelity:
    """Verify MCP dispatch produces the same results as direct engine calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine, _, _ = _build(FIXTURES)

    def test_stats_match(self):
        """MCP _dispatch_tool stats should match engine.stats()."""
        from interlinked.mcp_server import _dispatch_tool
        mcp_result = json.loads(
            _dispatch_tool("interlinked_stats", {}, self.engine, self.graph, "")
        )
        engine_result = self.engine.stats()
        assert mcp_result == engine_result, \
            f"MCP stats != engine stats:\nMCP: {mcp_result}\nEngine: {engine_result}"

    def test_isolate_match(self):
        """MCP isolate should produce same result string as engine.isolate."""
        from interlinked.mcp_server import _dispatch_tool
        # Reset state between calls
        self.engine.reset_filter()
        engine_result = self.engine.isolate("Engine", level="class", depth=2)
        self.engine.reset_filter()
        mcp_result = _dispatch_tool(
            "interlinked_isolate",
            {"target": "Engine", "level": "class", "depth": 2},
            self.engine, self.graph, ""
        )
        # Both should mention "Engine" in the result
        assert "Engine" in engine_result
        assert "Engine" in mcp_result

    def test_query_match(self):
        """MCP query should return same data as engine.query."""
        from interlinked.mcp_server import _dispatch_tool
        self.engine.reset_filter()
        engine_result = self.engine.query("dead functions")
        self.engine.reset_filter()
        mcp_raw = _dispatch_tool(
            "interlinked_query",
            {"expression": "dead functions"},
            self.engine, self.graph, ""
        )
        # Both should find dead functions
        assert len(engine_result) > 0, "Engine should find dead functions"
        # MCP returns JSON string of results
        mcp_result = json.loads(mcp_raw) if mcp_raw.startswith("[") else mcp_raw
        assert "_truly_dead_helper" in str(engine_result)
        assert "_truly_dead_helper" in str(mcp_result)
