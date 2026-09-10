"""Unit tests for the synthetic data generators.

Uses canned ``llm_call`` responses to verify prompt construction and
tolerant parsing of JSON lists, fenced output, and line-based fallbacks.
No network access; no attack content.
"""

from typing import Callable, List

import pytest

from tool_selection_harness.core import ToolDocument
from tool_selection_harness.core.generators import (
    generate_task_descriptions,
    generate_tool_documents,
)


class CannedLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


# -- generate_task_descriptions ----------------------------------------------


def test_task_descriptions_parse_json_list() -> None:
    llm = CannedLLM('["check the weather in Paris", "is it raining in London?"]')
    queries = generate_task_descriptions("weather", 5, llm)
    assert queries == ["check the weather in Paris", "is it raining in London?"]


def test_task_descriptions_parse_fenced_json() -> None:
    llm = CannedLLM('```json\n["query one", "query two"]\n```')
    queries = generate_task_descriptions("weather", 2, llm)
    assert queries == ["query one", "query two"]


def test_task_descriptions_fallback_to_numbered_lines() -> None:
    llm = CannedLLM("1. first query\n2) second query\n- third query")
    queries = generate_task_descriptions("weather", 3, llm)
    assert queries == ["first query", "second query", "third query"]


def test_task_descriptions_strip_quotes_from_lines() -> None:
    llm = CannedLLM('"quoted query"\n"another quoted"')
    queries = generate_task_descriptions("weather", 2, llm)
    assert queries == ["quoted query", "another quoted"]


def test_task_descriptions_capped_at_num() -> None:
    llm = CannedLLM('["a", "b", "c", "d"]')
    queries = generate_task_descriptions("weather", 2, llm)
    assert queries == ["a", "b"]


def test_task_descriptions_unparseable_output_raises() -> None:
    llm = CannedLLM("\n\n  \n")
    with pytest.raises(ValueError):
        generate_task_descriptions("weather", 3, llm)


def test_task_descriptions_non_positive_num_raises() -> None:
    with pytest.raises(ValueError):
        generate_task_descriptions("weather", 0, lambda p: "[]")


def test_task_descriptions_prompt_mentions_task_and_count() -> None:
    llm = CannedLLM('["q"]')
    generate_task_descriptions("checking the weather in a city", 7, llm)
    prompt = llm.prompts[0]
    assert "checking the weather in a city" in prompt
    assert "7" in prompt
