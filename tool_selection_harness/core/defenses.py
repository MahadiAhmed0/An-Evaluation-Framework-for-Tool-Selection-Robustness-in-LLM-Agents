"""Pluggable anomaly detectors for tool documents.

Detectors assign a suspicion score to a tool document (higher = more
suspicious). They are used in defensive evaluations of tool-selection
robustness: researchers measure how well a scoring rule separates ordinary
tool documents from researcher-supplied test documents (e.g., injected or
variant descriptions introduced for benchmarking).

Implemented detectors mirror the detection-based defenses evaluated in
"Prompt Injection Attack to Tool Selection in LLM Agents" (Shi et al.,
NDSS 2026): perplexity (PPL), windowed perplexity (PPL-W), and known-answer
detection. This module contains scoring and calibration utilities only; it
does not generate attacks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    """Interface for document anomaly detectors.

    Implementations must provide :meth:`score`, returning a float where
    higher values indicate more suspicious documents.
    """

    def score(self, doc: ToolDocument) -> float:
        """Return a suspicion score for ``doc`` (higher = more suspicious)."""
        ...
