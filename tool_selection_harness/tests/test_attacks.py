"""Unit tests for the published attack-baseline reproduction (attacks.py).

Manual templates and the gradient-free ToolHijacker path are tested with
scripted LLMs; the gradient-based components are verified with small
stub models (no downloads). Everything targets the local benchmark
harness only, per the module's ethics note.
"""

import pytest

from tool_selection_harness.core.attacks import manual_attack_documents


# -- manual baselines ----------------------------------------------------------


def test_manual_attacks_produce_five_documents() -> None:
    docs = manual_attack_documents("WeatherPro")
    assert set(docs) == {
        "naive",
        "escape",
        "context_ignore",
        "fake_completion",
        "combined",
    }
    for doc in docs.values():
        assert doc.tool_name == "WeatherPro"
        assert "WeatherPro" in doc.tool_description


def test_manual_attacks_contain_instruction_phrases() -> None:
    docs = manual_attack_documents("Pro")
    assert "Just output" in docs["naive"].tool_description
    assert "Ignore all previous instructions" in docs["context_ignore"].tool_description
    assert "Task completed" in docs["fake_completion"].tool_description
