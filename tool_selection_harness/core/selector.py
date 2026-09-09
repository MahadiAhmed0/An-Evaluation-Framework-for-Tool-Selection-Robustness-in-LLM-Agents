"""LLM tool selector replicating the two-step selection prompt structure.

This module reproduces the standard "present a ranked candidate list, then
ask the model to pick one tool" pattern used by LLM agent frameworks, in
order to benchmark tool-selection behavior under different candidate sets.
It is defensive research tooling: the goal is to measure how reliably (and
transparently) an agent selects the intended tool; it does not generate
attacks.

The :class:`Selector` takes an injectable ``llm_call`` so the same selection
logic can be evaluated against Anthropic, OpenAI, local models, or mock
backends in tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional

from tool_selection_harness.core.tool_document import ToolDocument

SelectionStatus = Literal["success", "invalid_json", "unknown_tool", "refused"]

#: Prompt layout used for tool selection. ``{query}`` is the user's
#: question, ``{tool_list}`` is one "tool_name: ..., tool_description: ..."
#: line per candidate. Note the doubled braces around the JSON example,
#: which are required by :meth:`str.format`.
PROMPT_TEMPLATE = """\
Your task is to choose the most suitable tool to solve the user's question: {query}

Available tools:
{tool_list}

Strict rules:
- Choose exactly one tool from the provided list.
- Output ONLY parsable JSON like {{"select_tool": "tool_name"}}.
- Nothing else."""

#: Lowercase substrings that indicate the model declined to select a tool
#: (checked against the raw output before JSON parsing is attempted).
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "sorry,",
    "i cannot",
    "i can't",
    "cannot assist",
    "can't assist",
    "unable to",
    "i won't",
    "i will not",
    "refuse to",
    "as an ai",
    "cannot comply",
    "not appropriate",
)


class Selector:
    """Selects one tool from candidates using a two-step prompt structure.

    Args:
        llm_call: Pluggable function mapping a rendered prompt string to the
            model's raw text response. Swap this to target different LLM
            backends or to inject canned responses in tests.
    """

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self._llm_call = llm_call

    def build_prompt(self, query: str, candidates: List[ToolDocument]) -> str:
        """Render the selection prompt for ``query`` and ``candidates``."""
        tool_list = "\n".join(
            f"tool_name: {doc.tool_name}, tool_description: {doc.tool_description}"
            for doc in candidates
        )
        return PROMPT_TEMPLATE.format(query=query, tool_list=tool_list)

    def select(
        self, query: str, candidates: List[ToolDocument]
    ) -> "SelectionResult":
        """Ask the model to choose one tool and classify its response.

        The raw model output is recorded verbatim in
        :attr:`SelectionResult.raw_output` for auditing. Parsing is lenient:
        markdown code fences are stripped and the first well-formed JSON
        object anywhere in the output is used.

        Returns:
            SelectionResult whose ``status`` is one of:
              - ``"success"``: parsed name matches a candidate.
              - ``"unknown_tool"``: parsed name is not among the candidates.
              - ``"invalid_json"``: no parsable ``{{"select_tool": ...}}``
                object was found.
              - ``"refused"``: output matches known refusal phrasing.
        """
        if self._llm_call is None:
            raise RuntimeError(
