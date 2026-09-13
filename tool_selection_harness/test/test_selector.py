"""Unit tests for the two-step tool selector.

Uses a canned ``llm_call`` stub (no network) to verify prompt rendering,
lenient JSON parsing, candidate validation, and status classification of
model outputs. Defensive benchmarking only: these tests exercise the
selection harness, not attack generation.
"""

from typing import Callable, List

import pytest

from tool_selection_harness.core import SelectionResult, Selector, ToolDocument

QUERY = "What is the weather in Paris today?"


@pytest.fixture
def candidates() -> List[ToolDocument]:
    return [
        ToolDocument("weather_tool", "Fetch current weather conditions"),
        ToolDocument("calendar_tool", "Add calendar event"),
        ToolDocument("translate_text", "Translate text between languages"),
    ]


class CannedLLM:
    """Records prompts and returns a fixed canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def select_with(
    response: str, candidates: List[ToolDocument]
) -> tuple[SelectionResult, CannedLLM]:
    llm = CannedLLM(response)
    selector = Selector(llm_call=llm)
    result = selector.select(QUERY, candidates)
    return result, llm