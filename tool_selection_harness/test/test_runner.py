"""Unit tests for the benchmark runner orchestration layer.

Uses a deterministic fake retriever and a canned fake selector (no network,
no embeddings) to verify the retrieve -> select loop, results dict contents,
JSON round-trip, and console summary table. Defensive benchmarking only.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from tool_selection_harness.core import (
    BenchmarkRunner,
    SelectionResult,
    ToolDocument,
    ToolLibrary,
)

DOC_A = ToolDocument("tool_a", "description of tool a")
DOC_B = ToolDocument("tool_b", "description of tool b")
DOC_C = ToolDocument("tool_c", "description of tool c")
DOC_T = ToolDocument("tool_t", "test document variant")


class FakeRetriever:
    """Deterministic retriever: returns ranked tool names per query."""

    def __init__(self, rankings: Dict[str, List[str]]) -> None:
        self.rankings = rankings
        self.top_k_calls = 0

    def top_k(self, query: str, library: ToolLibrary, k: int, metric: str = "cosine"):
        self.top_k_calls += 1
        by_name = {doc.tool_name: doc for doc in library.documents}
        names = self.rankings[query][:k]
        return [(by_name[name], 1.0 - i * 0.1) for i, name in enumerate(names)]


class FakeSelector:
    """Returns canned SelectionResult objects keyed by query."""

    def __init__(self, outcomes: Dict[str, SelectionResult]) -> None:
        self.outcomes = outcomes
        self.select_calls = 0

    def select(self, query: str, candidates: List[ToolDocument]) -> SelectionResult:
        self.select_calls += 1
        return self.outcomes[query]


@pytest.fixture
def library() -> ToolLibrary:
    return ToolLibrary(documents=[DOC_A, DOC_B, DOC_C])


@pytest.fixture
def queries() -> List[Tuple[str, str]]:
    return [("q1", "tool_a"), ("q2", "tool_b"), ("q3", "tool_c")]


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever(
        {
            "q1": ["tool_a", "tool_b"],
            "q2": ["tool_b", "tool_c"],
            "q3": ["tool_c", "tool_a"],
        }
    )


@pytest.fixture
def selector() -> FakeSelector:
    return FakeSelector(
        {
            "q1": SelectionResult("tool_a", '{"select_tool": "tool_a"}', "success"),
            "q2": SelectionResult(None, "not json at all", "invalid_json"),
            "q3": SelectionResult("tool_b", '{"select_tool": "tool_b"}', "success"),
        }
    )