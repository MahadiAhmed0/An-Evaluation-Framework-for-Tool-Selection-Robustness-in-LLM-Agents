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

# -- success cases ----------------------------------------------------------


def test_valid_json_is_success(candidates: List[ToolDocument]) -> None:
    result, _ = select_with('{"select_tool": "weather_tool"}', candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "weather_tool"
    assert result.raw_output == '{"select_tool": "weather_tool"}'


def test_fenced_json_is_success(candidates: List[ToolDocument]) -> None:
    response = '```json\n{"select_tool": "calendar_tool"}\n```'
    result, _ = select_with(response, candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "calendar_tool"


def test_trailing_text_after_json_is_success(candidates: List[ToolDocument]) -> None:
    response = '{"select_tool": "translate_text"} Hope this helps!'
    result, _ = select_with(response, candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "translate_text"


def test_whitespace_around_json_is_success(candidates: List[ToolDocument]) -> None:
    response = '\n\n  {"select_tool": "weather_tool"}  \n'
    result, _ = select_with(response, candidates)
    assert result.status == "success"
    assert result.selected_tool_name == "weather_tool"