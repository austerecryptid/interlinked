"""Interactive REPL for controlling the visualization."""

from __future__ import annotations

import code
import sys
from pathlib import Path

from interlinked.analyzer.graph import CodeGraph
from interlinked.commander.query import QueryEngine


class InterlinkedREPL:
    """Drops into an interactive Python REPL with the QueryEngine as `view`."""

    def __init__(self, graph: CodeGraph) -> None:
        self.graph = graph
        self.view = QueryEngine(graph)

    def start(self) -> None:
        """Launch the interactive REPL."""
        banner = (
            "\n"
            "╔══════════════════════════════════════════════════╗\n"
            "║           INTERLINKED — Topology Explorer        ║\n"
            "╚══════════════════════════════════════════════════╝\n"
            "\n"
            "  Available objects:\n"
            "    view    — QueryEngine (main control interface)\n"
            "    graph   — CodeGraph (raw graph access)\n"
            "\n"
            "  Quick start:\n"
            "    view.stats()                          # summary\n"
            "    view.zoom('module')                   # zoom level\n"
            "    view.focus('my_module')               # focus on node\n"
            "    view.query('dead functions')          # find dead code\n"
            "    view.trace_variable('config')         # trace a var\n"
            "    view.nl('show me uncalled functions') # natural language\n"
            "\n"
            "  Type help(view) for full API documentation.\n"
        )

        local_ns = {
            "view": self.view,
            "graph": self.graph,
            "QueryEngine": QueryEngine,
            "CodeGraph": CodeGraph,
        }

        code.interact(banner=banner, local=local_ns, exitmsg="Goodbye.")
