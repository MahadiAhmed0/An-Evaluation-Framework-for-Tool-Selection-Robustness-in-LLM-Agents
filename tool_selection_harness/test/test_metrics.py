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
# -- accuracy ----------------------------------------------------------------


def test_accuracy_fraction_of_correct_baseline_selections() -> None:
    records = [
        make_record([DOC_A], SUCCESS_A, "tool_a"),            # correct
        make_record([DOC_B], SUCCESS_B, "tool_a"),            # wrong tool
        make_record([DOC_A, DOC_B], SUCCESS_A, "tool_a"),     # correct
    ]
    assert accuracy(records) == pytest.approx(2 / 3)


def test_accuracy_ignores_test_document_records() -> None:
    records = [
        make_record([DOC_A], SUCCESS_A, "tool_a"),                       # correct
        make_record([DOC_T], SUCCESS_T, "tool_t", test_doc=DOC_T),       # ignored
        make_record([DOC_B], SUCCESS_B, "tool_a"),                       # wrong tool
        make_record([DOC_T], SUCCESS_A, "tool_a", test_doc=DOC_T),       # ignored
    ]
    assert accuracy(records) == pytest.approx(0.5)