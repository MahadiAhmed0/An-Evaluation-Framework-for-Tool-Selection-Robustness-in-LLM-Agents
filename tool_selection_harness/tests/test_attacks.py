"""Unit tests for the published attack-baseline reproduction (attacks.py).

Manual templates and the gradient-free ToolHijacker path are tested with
scripted LLMs; the gradient-based components are verified with small
stub models (no downloads). Everything targets the local benchmark
harness only, per the module's ethics note.
"""

from typing import List

import pytest

from tool_selection_harness.core.attacks import (
    generate_retrieval_sequence,
    manual_attack_documents,
)


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


# -- gradient-free R ------------------------------------------------------------


def test_generate_retrieval_sequence_uses_prompt() -> None:
    prompts: List[str] = []

    def llm_call(prompt: str) -> str:
        prompts.append(prompt)
        return "Provides weather information for any city."

    result = generate_retrieval_sequence(
        ["What is the weather in Paris?"], num_words=30, llm_call=llm_call
    )
    assert result == "Provides weather information for any city."
    assert "What is the weather in Paris?" in prompts[0]
    assert "functionality description" in prompts[0]


def test_generate_retrieval_sequence_rejects_empty_queries() -> None:
    with pytest.raises(ValueError):
        generate_retrieval_sequence([], 10, lambda p: "x")
