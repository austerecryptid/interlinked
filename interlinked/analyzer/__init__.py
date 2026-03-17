"""Analyzer — static analysis and graph extraction from Python source."""

from interlinked.analyzer.parser import parse_project
from interlinked.analyzer.graph import CodeGraph
from interlinked.analyzer.dead_code import detect_dead_code

__all__ = ["parse_project", "CodeGraph", "detect_dead_code"]
