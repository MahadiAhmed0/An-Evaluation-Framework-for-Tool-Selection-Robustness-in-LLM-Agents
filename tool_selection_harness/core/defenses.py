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

import numpy as np


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

    def _token_nlls(self, text: str) -> np.ndarray:
        """Per-token negative log-likelihood of ``text`` under the LM."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "This detector requires 'torch'. Install with "
                "`pip install torch`."
            ) from exc

        tokenizer, model = self._load()
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if input_ids.shape[-1] < 2:
            return np.zeros(0, dtype=np.float64)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
        )
        return nll.detach().cpu().numpy().astype(np.float64)


class PerplexityDetector(_LocalLMDetectorBase):
    """Score documents by description perplexity (PPL) under a local LM.

    Computes the average token negative log-likelihood (NLL) of the
    ``tool_description`` under a causal language model (default: gpt2).
    Unusual or incoherent descriptions tend to receive higher NLL, making
    this a simple baseline anomaly score (cf. Jain et al., the paper's PPL
    detection). The model is loaded lazily on the first call.
    """

    def score(self, doc: ToolDocument) -> float:
        """Return mean token NLL of ``doc.tool_description``."""
        nlls = self._token_nlls(doc.tool_description)
        if nlls.size == 0:
            return 0.0
        return float(nlls.mean())
