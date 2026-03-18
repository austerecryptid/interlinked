"""Tests that every query type returns FULL untruncated results.

No result set should ever be silently capped or truncated.
Every query path in QueryEngine.query() and every MCP dispatch path
must return all matching results.

Run: pytest tests/test_query_completeness.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlinked.analyzer.parser import parse_project
from interlinked.analyzer.graph import CodeGraph
from interlinked.analyzer.dead_code import detect_dead_code
from interlinked.commander.query import QueryEngine, _slim_node_dict
from interlinked.models import SymbolType, EdgeType, NodeData


FIXTURES = Path(__file__).parent / "fixtures"


def _build() -> tuple[CodeGraph, QueryEngine]:
    nodes, edges = parse_project(str(FIXTURES))
    graph = CodeGraph()
    graph.build_from(nodes, edges)
    detect_dead_code(graph)
    engine = QueryEngine(graph)
    return graph, engine


# ══════════════════════════════════════════════════════════════════════
# 1. QUERY COMPLETENESS — every query type returns all results
# ══════════════════════════════════════════════════════════════════════


class TestQueryCompleteness:
    """Every query type must return the full result set, never truncated."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine = _build()

    def _count_by_type(self, sym_type: SymbolType) -> int:
        return len(self.graph.nodes_by_type(sym_type))

    def _count_dead(self, type_filter: set[SymbolType] | None = None) -> int:
        return len([
            n for n in self.graph.all_nodes()
            if n.is_dead
            and (type_filter is None or n.symbol_type in type_filter)
        ])

    # ── "functions" / "methods" ──────────────────────────────────

    def test_functions_query_returns_all(self):
        results = self.engine.query("functions")
        expected = (
            self._count_by_type(SymbolType.FUNCTION)
            + self._count_by_type(SymbolType.METHOD)
        )
        assert len(results) == expected, \
            f"'functions' query returned {len(results)}, expected {expected}"

    def test_methods_query_returns_all(self):
        results = self.engine.query("methods")
        expected = (
            self._count_by_type(SymbolType.FUNCTION)
            + self._count_by_type(SymbolType.METHOD)
        )
        assert len(results) == expected, \
            f"'methods' query returned {len(results)}, expected {expected}"

    # ── "classes" ────────────────────────────────────────────────

    def test_classes_query_returns_all(self):
        results = self.engine.query("classes")
        expected = self._count_by_type(SymbolType.CLASS)
        assert len(results) == expected, \
            f"'classes' query returned {len(results)}, expected {expected}"

    # ── "modules" ────────────────────────────────────────────────

    def test_modules_query_returns_all(self):
        results = self.engine.query("modules")
        expected = self._count_by_type(SymbolType.MODULE)
        assert len(results) == expected, \
            f"'modules' query returned {len(results)}, expected {expected}"

    # ── "variables" ──────────────────────────────────────────────

    def test_variables_query_returns_all(self):
        results = self.engine.query("variables")
        expected = len([
            n for n in self.graph.all_nodes()
            if n.symbol_type == SymbolType.VARIABLE
        ])
        assert len(results) == expected, \
            f"'variables' query returned {len(results)}, expected {expected}"

    # ── "parameters" ─────────────────────────────────────────────

    def test_parameters_query_returns_all(self):
        results = self.engine.query("parameters")
        expected = len([
            n for n in self.graph.all_nodes()
            if n.symbol_type == SymbolType.PARAMETER
        ])
        assert len(results) == expected, \
            f"'parameters' query returned {len(results)}, expected {expected}"

    # ── "dead functions" ─────────────────────────────────────────

    def test_dead_functions_returns_all(self):
        results = self.engine.query("dead functions")
        expected = self._count_dead({SymbolType.FUNCTION, SymbolType.METHOD})
        assert len(results) == expected, \
            f"'dead functions' returned {len(results)}, expected {expected}"

    def test_dead_returns_all(self):
        results = self.engine.query("dead")
        expected = self._count_dead()
        assert len(results) == expected, \
            f"'dead' returned {len(results)}, expected {expected}"

    # ── "callers of X" ───────────────────────────────────────────

    def test_callers_of_returns_all(self):
        # Pick a node that has callers
        for node in self.graph.all_nodes():
            callers = self.graph.callers_of(node.id)
            if len(callers) >= 2:
                results = self.engine.query(f"callers of {node.qualified_name}")
                assert len(results) == len(callers), \
                    f"'callers of {node.name}' returned {len(results)}, expected {len(callers)}"
                return
        pytest.skip("No node with >=2 callers found in fixtures")

    # ── "callees of X" ───────────────────────────────────────────

    def test_callees_of_returns_all(self):
        for node in self.graph.all_nodes():
            callees = self.graph.callees_of(node.id)
            if len(callees) >= 2:
                results = self.engine.query(f"callees of {node.qualified_name}")
                assert len(results) == len(callees), \
                    f"'callees of {node.name}' returned {len(results)}, expected {len(callees)}"
                return
        pytest.skip("No node with >=2 callees found in fixtures")

    # ── "parameters of X" ────────────────────────────────────────

    def test_parameters_of_returns_all(self):
        for node in self.graph.all_nodes():
            if node.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
                continue
            G = self.graph._g
            if node.id not in G:
                continue
            param_ids = [
                v for _, v, d in G.out_edges(node.id, data=True)
                if d.get("edge_type") == "contains"
                and self.graph.get_node(v)
                and self.graph.get_node(v).symbol_type == SymbolType.PARAMETER
            ]
            if len(param_ids) >= 2:
                results = self.engine.query(f"parameters of {node.qualified_name}")
                assert len(results) == len(param_ids), \
                    f"'parameters of {node.name}' returned {len(results)}, expected {len(param_ids)}"
                return
        pytest.skip("No function with >=2 params found in fixtures")

    # ── "returns of X" ───────────────────────────────────────────

    def test_returns_of_returns_all(self):
        for node in self.graph.all_nodes():
            edges = self.graph.edges_from(node.id, EdgeType.RETURNS)
            ret_nodes = [self.graph.get_node(e.target) for e in edges if self.graph.get_node(e.target)]
            if len(ret_nodes) >= 1:
                results = self.engine.query(f"returns of {node.qualified_name}")
                assert len(results) == len(ret_nodes), \
                    f"'returns of {node.name}' returned {len(results)}, expected {len(ret_nodes)}"
                return
        pytest.skip("No function with return edges found in fixtures")

    # ── "imports of X" ───────────────────────────────────────────

    def test_imports_of_returns_all(self):
        for node in self.graph.all_nodes():
            if node.symbol_type != SymbolType.MODULE:
                continue
            edges = self.graph.edges_from(node.id, EdgeType.IMPORTS)
            if len(edges) >= 1:
                results = self.engine.query(f"imports of {node.qualified_name}")
                assert len(results) == len(edges), \
                    f"'imports of {node.name}' returned {len(results)}, expected {len(edges)}"
                return
        pytest.skip("No module with imports found in fixtures")

    # ── "external calls" ─────────────────────────────────────────

    def test_external_calls_returns_all(self):
        node_ids = {n.id for n in self.graph.all_nodes()}
        expected = [
            e for e in self.graph.all_edges()
            if e.edge_type == EdgeType.CALLS and e.target not in node_ids
        ]
        results = self.engine.query("external calls")
        assert len(results) == len(expected), \
            f"'external calls' returned {len(results)}, expected {len(expected)}"

    # ── scoped queries ───────────────────────────────────────────

    def test_scoped_functions_returns_all(self):
        # Find a module with multiple functions
        modules = self.graph.nodes_by_type(SymbolType.MODULE)
        for mod in modules:
            scope = mod.qualified_name
            expected = [
                n for n in self.graph.all_nodes()
                if n.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)
                and n.qualified_name.startswith(scope)
            ]
            if len(expected) >= 3:
                results = self.engine.query(f"functions in {scope}")
                assert len(results) == len(expected), \
                    f"'functions in {scope}' returned {len(results)}, expected {len(expected)}"
                return
        pytest.skip("No module with >=3 functions found")

    def test_scoped_classes_returns_all(self):
        modules = self.graph.nodes_by_type(SymbolType.MODULE)
        for mod in modules:
            scope = mod.qualified_name
            expected = [
                n for n in self.graph.nodes_by_type(SymbolType.CLASS)
                if n.qualified_name.startswith(scope)
            ]
            if len(expected) >= 2:
                results = self.engine.query(f"classes in {scope}")
                assert len(results) == len(expected), \
                    f"'classes in {scope}' returned {len(results)}, expected {len(expected)}"
                return
        pytest.skip("No module with >=2 classes found")

    # ── fuzzy name search ────────────────────────────────────────

    def test_fuzzy_search_returns_all(self):
        # Search for a common substring
        term = "process"
        expected = [
            n for n in self.graph.all_nodes()
            if term in n.qualified_name.lower() or term in n.name.lower()
        ]
        results = self.engine.query(term)
        assert len(results) == len(expected), \
            f"fuzzy search '{term}' returned {len(results)}, expected {len(expected)}"


# ══════════════════════════════════════════════════════════════════════
# 2. MCP DISPATCH — no truncation in the MCP layer
# ══════════════════════════════════════════════════════════════════════


class TestMCPNoTruncation:
    """MCP _dispatch_tool must never truncate query results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine = _build()

    def _mcp_query(self, expression: str) -> list:
        from interlinked.mcp_server import _dispatch_tool
        self.engine.reset_filter()
        raw = _dispatch_tool(
            "interlinked_query",
            {"expression": expression},
            self.engine, self.graph, ""
        )
        return json.loads(raw)

    def test_mcp_functions_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("functions")
        mcp_results = self._mcp_query("functions")
        assert len(mcp_results) == len(engine_results), \
            f"MCP truncated: {len(mcp_results)} vs engine {len(engine_results)}"

    def test_mcp_classes_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("classes")
        mcp_results = self._mcp_query("classes")
        assert len(mcp_results) == len(engine_results)

    def test_mcp_modules_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("modules")
        mcp_results = self._mcp_query("modules")
        assert len(mcp_results) == len(engine_results)

    def test_mcp_dead_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("dead")
        mcp_results = self._mcp_query("dead")
        assert len(mcp_results) == len(engine_results)

    def test_mcp_variables_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("variables")
        mcp_results = self._mcp_query("variables")
        assert len(mcp_results) == len(engine_results)

    def test_mcp_parameters_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("parameters")
        mcp_results = self._mcp_query("parameters")
        assert len(mcp_results) == len(engine_results)

    def test_mcp_external_calls_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("external calls")
        mcp_results = self._mcp_query("external calls")
        assert len(mcp_results) == len(engine_results)

    def test_mcp_callers_not_truncated(self):
        # Find a node with callers
        for node in self.graph.all_nodes():
            callers = self.graph.callers_of(node.id)
            if len(callers) >= 2:
                self.engine.reset_filter()
                engine_results = self.engine.query(f"callers of {node.qualified_name}")
                mcp_results = self._mcp_query(f"callers of {node.qualified_name}")
                assert len(mcp_results) == len(engine_results), \
                    f"MCP truncated callers: {len(mcp_results)} vs {len(engine_results)}"
                return
        pytest.skip("No node with >=2 callers")

    def test_mcp_callees_not_truncated(self):
        for node in self.graph.all_nodes():
            callees = self.graph.callees_of(node.id)
            if len(callees) >= 2:
                self.engine.reset_filter()
                engine_results = self.engine.query(f"callees of {node.qualified_name}")
                mcp_results = self._mcp_query(f"callees of {node.qualified_name}")
                assert len(mcp_results) == len(engine_results), \
                    f"MCP truncated callees: {len(mcp_results)} vs {len(engine_results)}"
                return
        pytest.skip("No node with >=2 callees")

    def test_mcp_fuzzy_search_not_truncated(self):
        self.engine.reset_filter()
        engine_results = self.engine.query("process")
        mcp_results = self._mcp_query("process")
        assert len(mcp_results) == len(engine_results)


# ══════════════════════════════════════════════════════════════════════
# 3. SLIM NODE — fingerprint stripping doesn't lose essential fields
# ══════════════════════════════════════════════════════════════════════


class TestSlimNode:
    """_slim_node_dict strips heavy fields but preserves everything else."""

    def test_no_fingerprint_passthrough(self):
        d = {"id": "a.b", "metadata": {}}
        assert _slim_node_dict(d) == d

    def test_strips_heavy_keys(self):
        d = {
            "id": "a.b",
            "metadata": {
                "fingerprint": {
                    "arg_count": 3,
                    "has_loops": True,
                    "ast_tree": ("FunctionDef", (("Return", ()),)),
                    "minhash": tuple(range(100)),
                    "ast_node_counts": {"FunctionDef": 1, "Return": 1},
                    "source_snippet": "def foo(): return 1",
                    "callees": ["bar"],
                    "callers": ["baz"],
                },
            },
        }
        result = _slim_node_dict(d)
        fp = result["metadata"]["fingerprint"]
        assert "ast_tree" not in fp
        assert "minhash" not in fp
        assert "ast_node_counts" not in fp
        assert "source_snippet" not in fp
        # Preserved fields
        assert fp["arg_count"] == 3
        assert fp["has_loops"] is True
        assert fp["callees"] == ["bar"]
        assert fp["callers"] == ["baz"]

    def test_does_not_mutate_original(self):
        d = {
            "id": "a.b",
            "metadata": {
                "fingerprint": {
                    "arg_count": 3,
                    "minhash": (1, 2, 3),
                },
            },
        }
        _slim_node_dict(d)
        assert "minhash" in d["metadata"]["fingerprint"], \
            "_slim_node_dict must not mutate the original dict"

    def test_snapshot_nodes_are_slim(self):
        """engine.snapshot() nodes must not contain heavy fingerprint fields."""
        graph, engine = _build()
        # Run similarity to populate fingerprints
        try:
            from interlinked.analyzer.similarity import analyze_similarity
            analyze_similarity(graph)
        except Exception:
            pytest.skip("similarity module not available")

        snap = engine.snapshot()
        heavy_keys = {"ast_tree", "minhash", "ast_node_counts", "source_snippet"}
        for node in snap["nodes"]:
            fp = node.get("metadata", {}).get("fingerprint")
            if fp:
                found = heavy_keys & set(fp.keys())
                assert not found, \
                    f"Node {node['id']} snapshot contains heavy keys: {found}"

    def test_query_results_are_slim(self):
        """engine.query() results must not contain heavy fingerprint fields."""
        graph, engine = _build()
        try:
            from interlinked.analyzer.similarity import analyze_similarity
            analyze_similarity(graph)
        except Exception:
            pytest.skip("similarity module not available")

        results = engine.query("functions")
        heavy_keys = {"ast_tree", "minhash", "ast_node_counts", "source_snippet"}
        for node in results:
            fp = node.get("metadata", {}).get("fingerprint")
            if fp:
                found = heavy_keys & set(fp.keys())
                assert not found, \
                    f"Query result {node['id']} contains heavy keys: {found}"


# ══════════════════════════════════════════════════════════════════════
# 4. GET_CONTEXT — callers/callees not truncated
# ══════════════════════════════════════════════════════════════════════


class TestGetContextCompleteness:
    """get_context must return full callers/callees lists."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.engine = _build()

    def test_callers_not_capped(self):
        for node in self.graph.all_nodes():
            callers = self.graph.callers_of(node.id)
            if len(callers) >= 2:
                ctx = self.engine.get_context(node.qualified_name)
                ctx_data = json.loads(ctx) if isinstance(ctx, str) else ctx
                assert len(ctx_data["callers"]) == len(callers), \
                    f"get_context callers capped: {len(ctx_data['callers'])} vs {len(callers)}"
                return
        pytest.skip("No node with >=2 callers")

    def test_callees_not_capped(self):
        for node in self.graph.all_nodes():
            callees = self.graph.callees_of(node.id)
            if len(callees) >= 2:
                ctx = self.engine.get_context(node.qualified_name)
                ctx_data = json.loads(ctx) if isinstance(ctx, str) else ctx
                assert len(ctx_data["callees"]) == len(callees), \
                    f"get_context callees capped: {len(ctx_data['callees'])} vs {len(callees)}"
                return
        pytest.skip("No node with >=2 callees")

    def test_fingerprint_slim_in_context(self):
        """get_context fingerprint must not include heavy fields."""
        try:
            from interlinked.analyzer.similarity import analyze_similarity
            analyze_similarity(self.graph)
        except Exception:
            pytest.skip("similarity module not available")

        for node in self.graph.all_nodes():
            if node.metadata.get("fingerprint"):
                ctx = self.engine.get_context(node.qualified_name)
                ctx_data = json.loads(ctx) if isinstance(ctx, str) else ctx
                fp = ctx_data.get("fingerprint")
                if fp:
                    heavy = {"ast_tree", "minhash", "ast_node_counts", "source_snippet"}
                    found = heavy & set(fp.keys())
                    assert not found, \
                        f"get_context fingerprint has heavy keys: {found}"
                return
        pytest.skip("No node with fingerprint")
