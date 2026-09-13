"""Unit tests for the tool-selection evaluation metrics.

Handcrafted :class:`EvalRecord` objects (no retriever/selector needed) are
used to verify each metric's numerator/denominator semantics, including the
baseline-vs-test-document filtering. Defensive benchmarking metrics only.
"""

from typing import List

import pytest

from tool_selection_harness.core import (
    EvalRecord,
    SelectionResult,
    ToolDocument,
    accuracy,
    compute_all,
    hit_rate_at_k,
    status_breakdown,
    target_retrieval_rate,
    target_selection_rate,
)

DOC_A = ToolDocument("tool_a", "description of tool a")
DOC_B = ToolDocument("tool_b", "description of tool b")
DOC_T = ToolDocument("tool_t", "variant document under test")

SUCCESS_A = SelectionResult("tool_a", '{"select_tool": "tool_a"}', "success")
SUCCESS_B = SelectionResult("tool_b", '{"select_tool": "tool_b"}', "success")
SUCCESS_T = SelectionResult("tool_t", '{"select_tool": "tool_t"}', "success")
INVALID = SelectionResult(None, "not json", "invalid_json")
UNKNOWN = SelectionResult(None, '{"select_tool": "tool_z"}', "unknown_tool")


def make_record(
    retrieved: List[ToolDocument],
    selection: SelectionResult,
    expected: str,
    test_doc: ToolDocument | None = None,
) -> EvalRecord:
    return EvalRecord(
        query="q",
        retrieved_docs=retrieved,
        selection_result=selection,
        expected_tool=expected,
        test_document=test_doc,
    )