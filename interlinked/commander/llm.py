"""LLM adapter — translates natural language into view commands via Claude (or any LLM).

The key idea: the LLM gets a system prompt describing the full view API + current graph stats,
then returns JSON with both a human-readable explanation and executable Python commands.

This module is used by:
1. The /api/chat endpoint (browser chat box → Claude → commands → view updates)
2. Any external LLM agent hitting /api/command directly (e.g. Cascade)

If no API key is configured, commands pass through to a basic keyword fallback.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any

import httpx

from interlinked.commander.query import QueryEngine
from interlinked.models import SymbolType, EdgeType


def get_system_prompt(engine: QueryEngine) -> str:
    """Build the system prompt that teaches an LLM how to drive Interlinked.

    This is also served at GET /api/system-prompt so any external agent
    can read it and know how to control the view.
    """
    stats = engine.stats()
    all_nodes = engine.graph.all_nodes(include_proposed=False)
    modules = sorted(set(n.qualified_name for n in all_nodes if n.symbol_type == SymbolType.MODULE))
    classes = sorted(set(n.qualified_name for n in all_nodes if n.symbol_type == SymbolType.CLASS))

    return f"""\
You are the pilot of INTERLINKED, a Python codebase topology explorer.
The user is looking at an interactive graph visualization of a Python project in their browser.
You control what they see by emitting Python commands against the `view` object.

## Current Project Stats
- Modules: {stats['modules']} | Classes: {stats['classes']} | Functions: {stats['functions']} | Methods: {stats['methods']}
- Variables: {stats['variables']} | Parameters: {stats['parameters']} | External calls: {stats['external_calls']}
- Dead/unreachable: {stats['dead_nodes']} | Total edges: {stats['total_edges']}

## Known Modules
{chr(10).join(f'  - {m}' for m in modules)}

## Known Classes
{chr(10).join(f'  - {c}' for c in classes)}

## The `view` API

### Navigation
- `view.zoom(level)` — Set zoom: "all", "module", "class", "function", or "variable" (variable includes parameters)
- `view.focus(node_id, depth=2)` — Focus on a node and its neighborhood
- `view.unfocus()` — Clear focus, show full graph
- `view.isolate(target, level="function", depth=3, edge_types=None)` — **PRIMARY COMMAND**: Isolate a module/class/service and show it + everything connecting to it. `target` can be a partial name. `edge_types` is optional list like ["calls", "imports"].
- `view.show(target, level="function", depth=2)` — Shorthand for isolate
- `view.reset_filter()` — Reset everything to default view

### Queries
- `view.query("dead functions")` — Find dead/uncalled code (highlights results)
- `view.query("dead functions in <scope>")` — Dead functions scoped to a module prefix, e.g. `"dead functions in engine.rules.ir_exec"`
- `view.query("dead classes in <scope>")` — Dead classes in a scope
- `view.query("callers of <name>")` — Who calls this?
- `view.query("callees of <name>")` — What does this call?
- `view.query("parameters of <name>")` — Show function parameters
- `view.query("returns of <name>")` — Show what a function returns (project symbols)
- `view.query("external calls in <name>")` — Show external library calls from a module/class/function
- `view.query("external calls")` — All external calls in the project
- `view.query("functions in <scope>")` — List all functions/methods under a module prefix, e.g. `"functions in engine.rules"`
- `view.query("classes in <scope>")` — List all classes under a module prefix
- `view.query("modules")` / `view.query("classes")` / `view.query("functions")` / `view.query("parameters")` / `view.query("variables")`
- `view.query("<search_term>")` — Fuzzy name search

### Tracing (powered by NetworkX graph pathfinding)
- `view.trace_variable(var_name, origin=None)` — Trace a variable's data flow: who writes it, who reads it, and the paths between them.
- `view.trace_function(name)` — Trace a function's full call chain: everything that calls it (upstream) and everything it calls (downstream).
- `view.trace_call_chain(source, target, max_depth=8)` — Find all call paths from one function to another.

### Impact & Dependency Analysis
- `view.impact_of(name)` — **Blast radius**: highlight everything downstream affected by changing this symbol. Only follows data/control flow (calls, reads, writes, returns), not containment.
- `view.depends_on(name)` — **Upstream dependencies**: highlight everything this symbol depends on.
- `view.path_between(source, target)` — Shortest dependency chain between two symbols.
- `view.all_paths(source, target, max_depth=8)` — Every route between two symbols.

### Cross-Module Edge Enumeration
- `view.edges_between(source_scope, target_scope=None, edge_types=None)` — List all edges leaving a module scope, grouped by target module. Essential for module isolation checks.
  - `view.edges_between("engine.rules.resolver")` — ALL outgoing edges from resolver
  - `view.edges_between("engine.rules.resolver", target_scope="engine.systems")` — Only edges going to engine.systems
  - `view.edges_between("engine.rules", edge_types=["imports"])` — Only import edges
  - `view.edges_between("engine.rules", edge_types=["calls", "imports"])` — Calls + imports

### Reachability (Purity / Isolation Checks)
- `view.reachable(source, target, edge_types=None, max_depth=20)` — Check if target is reachable from source via specific edge types. Default: follows only call edges.
  - `view.reachable("resolve_effects", "world.set_component")` — Is there ANY call path? Returns path if yes.
  - `view.reachable("resolver.resolve", "db.commit", edge_types=["calls", "imports"])` — Check calls + imports
  - Returns JSON: `{{"reachable": true/false, "path": [...], "short_path": "A → B → C"}}`

### Architecture Health
- `view.find_cycles()` — Find and highlight circular dependencies (calls/imports).
- `view.critical_nodes(top_n=20)` — Most important symbols by PageRank.
- `view.bottlenecks(top_n=20)` — Coupling hotspots (betweenness centrality).
- `view.coupling(module_a, module_b)` — Show all cross-module edges.
- `view.health()` — Full architecture health report (JSON): cycles, dead code %, data flow resolution %, external dependencies, bottlenecks, critical nodes.

### Similarity & Duplicates
- `view.find_duplicates(threshold=0.6, scope=None)` — Find groups of structurally similar functions.
- `view.similar_to(target, threshold=0.5)` — Find functions similar to a specific symbol.
- `view.get_context(target)` — Rich context for a symbol: source, connections, fingerprint.

### Filtering
- `view.filter(edge_type="calls")` — Show only one edge type
- `view.set_edge_types(["calls", "reads", "writes"])` — Toggle which edge types are visible. Useful for decluttering.
- `view.filter(name_pattern="regex_pattern")` — Filter by name
- `view.show_dead(True/False)` — Toggle dead code visibility
- `view.show_proposed(True/False)` — Toggle hypothetical elements

### Hypotheticals
- `view.propose_function(name, module, calls=[...], called_by=[...])` — Add a hypothetical function (shown in green)
- `view.clear_proposed()` — Remove all hypothetical elements

### Display
- `view.set_color(key, hex_value)` — Change a color.
- `view.stats()` — Summary statistics (includes parameter count, external call count)

### Edge Types
calls, imports, inherits, contains, reads, writes, returns, proposed

### Node Types
module, class, function, method, variable, parameter

## Response Format
Respond with a JSON object:
```json
{{
  "explanation": "Human-readable description of what you're showing and why. Describe what the user is seeing and key observations.",
  "commands": ["view.isolate('analyzer.parser', level='function', depth=2)"]
}}
```

The `commands` array contains Python expressions executed against the `view` object.
The `explanation` will be shown to the user in the chat panel — it MUST describe what the visualization is now showing.

## Guidelines
- Use `view.isolate()` as your go-to for "show me X" requests.
- Use `view.set_edge_types(...)` to declutter when there are too many edges. For example, `view.set_edge_types(["calls"])` to show only call relationships.
- Trace/impact/dependency commands preserve the current zoom level. The graph automatically remaps highlights to visible ancestors at coarser zoom levels. Use `view.zoom("module")` to see trace results aggregated to modules, or `view.zoom("class")` for classes. This is powerful for large projects — "trace this variable at module level" shows which modules are involved without rendering thousands of nodes.
- When the user asks about architecture or structure, start at module level then drill down.
- When showing dead code, explain what it means and whether it might be intentional. Dead code now includes dead parameters (never read), dead variables (written but never read), and dead returns (return value never used).
- Node IDs are dotted qualified names like "analyzer.graph.CodeGraph.build_from"
- Partial names work for isolate/focus — "CodeGraph" will match "analyzer.graph.CodeGraph"
- Always explain what the user is looking at in your explanation.
"""


class LLMAdapter:
    """Bridges natural language to view commands via an LLM API."""

    def __init__(self, engine: QueryEngine) -> None:
        self.engine = engine
        self.api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
        self.model: str = "claude-sonnet-4-20250514"
        self.conversation: list[dict] = []
        self.max_history: int = 20

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str) -> None:
        self.api_key = key

    def set_model(self, model: str) -> None:
        self.model = model

    def clear_history(self) -> None:
        self.conversation.clear()

    async def chat(self, user_message: str) -> dict:
        """Process a user message: send to Claude, execute returned commands, return result.

        Returns: {"explanation": str, "commands_run": list[str], "results": list[str], "error": str|None}
        """
        if not self.api_key:
            return {
                "explanation": "No API key configured. Set ANTHROPIC_API_KEY environment variable or use the settings panel. You can also drive the view directly via the command bar with Python: view.isolate('target')",
                "commands_run": [],
                "results": [],
                "error": "no_api_key",
            }

        # Add user message to conversation
        self.conversation.append({"role": "user", "content": user_message})

        # Trim history
        if len(self.conversation) > self.max_history:
            self.conversation = self.conversation[-self.max_history:]

        # Call Claude
        system_prompt = get_system_prompt(self.engine)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 1024,
                        "system": system_prompt,
                        "messages": self.conversation,
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            error_msg = f"Claude API error: {e.response.status_code}"
            try:
                error_body = e.response.json()
                error_msg += f" — {error_body.get('error', {}).get('message', '')}"
            except Exception:
                pass
            return {
                "explanation": error_msg,
                "commands_run": [],
                "results": [],
                "error": error_msg,
            }
        except Exception as e:
            return {
                "explanation": f"Failed to reach Claude API: {e}",
                "commands_run": [],
                "results": [],
                "error": str(e),
            }

        # Extract response text
        assistant_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                assistant_text += block["text"]

        # Add assistant response to conversation
        self.conversation.append({"role": "assistant", "content": assistant_text})

        # Parse the JSON response
        explanation, commands = self._parse_response(assistant_text)

        # Execute commands
        results = []
        for cmd in commands:
            try:
                result = self._execute_command(cmd)
                results.append(f"{cmd} → {result}")
            except Exception as e:
                results.append(f"{cmd} → ERROR: {e}")

        return {
            "explanation": explanation,
            "commands_run": commands,
            "results": results,
            "error": None,
        }

    def _parse_response(self, text: str) -> tuple[str, list[str]]:
        """Extract explanation and commands from Claude's response."""
        # Try to find JSON block
        json_match = None
        # Look for ```json ... ``` blocks
        import re
        json_block = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if json_block:
            try:
                json_match = json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing the whole text as JSON
        if not json_match:
            try:
                json_match = json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object in the text
        if not json_match:
            brace_match = re.search(r'\{[^{}]*"explanation"[^{}]*\}', text, re.DOTALL)
            if brace_match:
                try:
                    json_match = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass

        if json_match and isinstance(json_match, dict):
            explanation = json_match.get("explanation", text)
            commands = json_match.get("commands", [])
            if isinstance(commands, str):
                commands = [commands]
            return explanation, commands

        # Fallback: treat the whole text as explanation, look for view.* commands inline
        commands = re.findall(r'(view\.\w+\([^)]*\))', text)
        return text, commands

    def _execute_command(self, cmd: str) -> str:
        """Execute a single view command string."""
        local_ns: dict[str, Any] = {"view": self.engine, "graph": self.engine.graph}
        try:
            result = eval(cmd, {"__builtins__": {}}, local_ns)
        except SyntaxError:
            exec(cmd, {"__builtins__": {}}, local_ns)
            result = "OK"
        if hasattr(result, "model_dump"):
            return str(result.model_dump())
        return str(result)
