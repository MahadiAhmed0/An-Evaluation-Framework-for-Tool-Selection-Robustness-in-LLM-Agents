"""Unit tests for the tool-selection robustness harness data structures.

These tests verify the defensive benchmarking plumbing: library
construction, copy-on-write semantics (especially for ``inject``), and JSON
persistence. No attack generation is performed or tested here.
"""

from pathlib import Path

import pytest

from tool_selection_harness.core import ToolDocument, ToolLibrary


@pytest.fixture
def doc_a() -> ToolDocument:
    return ToolDocument("tool_a", "Description of tool A")


@pytest.fixture
def doc_b() -> ToolDocument:
    return ToolDocument("tool_b", "Description of tool B", metadata={"cat": "x"})


# -- ToolDocument ---------------------------------------------------------


def test_document_fields(doc_a: ToolDocument) -> None:
    assert doc_a.tool_name == "tool_a"
    assert doc_a.tool_description == "Description of tool A"
    assert doc_a.metadata == {}


def test_document_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ToolDocument("", "desc")


def test_document_rejects_blank_description() -> None:
    with pytest.raises(ValueError):
        ToolDocument("tool_x", "   ")


def test_document_is_frozen(doc_a: ToolDocument) -> None:
    with pytest.raises(Exception):
        doc_a.tool_name = "mutated"  # type: ignore[misc]