"""Benchmark orchestration: retrieve -> select loop over evaluation queries.

:class:`BenchmarkRunner` wires together a :class:`ToolLibrary`, a
:class:`Retriever`, and a :class:`Selector` to produce evaluation records and
aggregate metrics for tool-selection robustness experiments. It is defensive
research tooling: the runner measures selection/retrieval behavior on
baseline libraries and, optionally, libraries augmented with a single test
document (e.g., an injected or variant tool description) for a given eval
pass. It does not generate attacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_selection_harness.core.metrics import (
    EvalRecord,
    compute_all,
    status_breakdown,
)
from tool_selection_harness.core.retriever import Retriever
from tool_selection_harness.core.selector import Selector
from tool_selection_harness.core.tool_document import ToolDocument
from tool_selection_harness.core.tool_library import ToolLibrary

_FLOAT_METRICS = (
    "accuracy",
    "hit_rate_at_k",
    "target_selection_rate",
    "target_retrieval_rate",
)

_STATUS_ORDER = ("success", "invalid_json", "unknown_tool", "refused")


class BenchmarkRunner:
    """Run a full retrieval + selection benchmark and aggregate results.

    Args:
        library: The baseline tool library under evaluation.
        retriever: Embedding-based retriever used for candidate ranking.
        selector: LLM-backed selector used to pick one tool.
        queries: Evaluation pairs of ``(query, expected_tool_name)``.
        k: Number of top retrieval candidates to pass to the selector.
        test_document: Optional test document to inject into a copy of the
            library for this eval pass. When ``None``, the run is a baseline
            pass; when provided, the same queries are re-run against the
            modified library and target-* metrics are computed.
    """

    def __init__(
        self,
        library: ToolLibrary,
        retriever: Retriever,
        selector: Selector,
        queries: List[Tuple[str, str]],
        k: int,
        test_document: Optional[ToolDocument] = None,
    ) -> None:
        self.library = library
        self.retriever = retriever
        self.selector = selector
        self.queries = queries
        self.k = k
        self.test_document = test_document
        self._last_results: Optional[Dict[str, Any]] = None

    # -- execution ---------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the retrieve -> select loop and aggregate results.

        Returns:
            A results dict with ``config`` (run parameters), ``metrics``
            (all metrics from :mod:`core.metrics` plus selector status
            counts), and ``records`` (serialized :class:`EvalRecord`
            objects, one per query).
        """
        active_library = (
            self.library.inject(self.test_document)
            if self.test_document is not None
            else self.library
        )

        records: List[EvalRecord] = []
        for query, expected_tool in self.queries:
            ranked = self.retriever.top_k(query, active_library, self.k)
            retrieved_docs = [doc for doc, _ in ranked]
            selection = self.selector.select(query, retrieved_docs)
            records.append(
                EvalRecord(
                    query=query,
                    retrieved_docs=retrieved_docs,
