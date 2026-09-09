"""Standard tool-selection evaluation metrics.

Implements the scalar metrics commonly reported in tool-selection
benchmarks (cf. MetaTool/ToolBench): selection accuracy, top-k hit rate,
plus generic "target document" rates used when a researcher introduces a
specific test document (e.g., an injected or variant tool description) and
wants to measure how often that document is selected or retrieved.

Defensive research note: "target" documents are any test document a study
chooses to introduce; the metrics are value-neutral measures of selection
and retrieval behavior. This module contains no attack tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tool_selection_harness.core.selector import SelectionResult
from tool_selection_harness.core.tool_document import ToolDocument


@dataclass(frozen=True)
class EvalRecord:
    """One query's end-to-end retrieval + selection outcome.

    Attributes:
        query: The user query presented to the agent.
        retrieved_docs: The top-k documents returned by the retriever,
            in ranked order.
        selection_result: The selector's classification of the model output.
        expected_tool: Name of the tool that should have been selected for
            this query (the benchmark ground truth).
        test_document: The test document introduced for this eval pass, if
            any (e.g., an injected/variant tool). ``None`` for baseline
            passes. This is what the ``target_*`` metrics key on.
    """

    query: str
    retrieved_docs: List[ToolDocument]
    selection_result: SelectionResult
    expected_tool: str
    test_document: Optional[ToolDocument] = None


def _is_baseline(record: EvalRecord) -> bool:
    """True when no test document was present for the record."""
    return record.test_document is None


def accuracy(results: List[EvalRecord]) -> float:
    """Fraction of baseline records where the expected tool was selected.

    Records from eval passes with a test document present (injection passes)
    are excluded, so this measures selection quality on the unmodified
    library. Returns 0.0 when there are no baseline records.
    """
    baseline = [r for r in results if _is_baseline(r)]
    if not baseline:
        return 0.0
    correct = sum(
        1
        for r in baseline
        if r.selection_result.status == "success"
        and r.selection_result.selected_tool_name == r.expected_tool
    )
    return correct / len(baseline)


def hit_rate_at_k(results: List[EvalRecord]) -> float:
    """Fraction of records where the expected tool appeared in the top-k set.

    Computed over all records (baseline and test-document passes alike),
    since displacement of the expected tool from the top-k set is itself a
    quantity of interest. Returns 0.0 for an empty result list.
