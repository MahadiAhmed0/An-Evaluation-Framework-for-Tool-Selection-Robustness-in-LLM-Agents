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


class _LocalLMDetectorBase:
    """Shared lazy loading + per-token NLL for local causal-LM detectors."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _load(self):
        """Load tokenizer + model on first use (lazy import)."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "This detector requires the 'transformers' and 'torch' "
                    "packages. Install with `pip install transformers torch`."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model
