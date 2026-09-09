"""Unit tests for detector interfaces and threshold calibration.

The LM-based detector scoring paths are tested with stubbed
tokenizer/model objects (no model download); the threshold classifier and
edge cases are tested with numeric scores. Defensive benchmarking
utilities only.
"""

import math
from typing import List

import numpy as np
import pytest

from tool_selection_harness.core import ToolDocument
from tool_selection_harness.core.defenses import (
    Detector,
    KnownAnswerDetector,
    PerplexityDetector,
    PerplexityWindowedDetector,
    ThresholdClassifier,
    _threshold_for_fpr,
)

torch = pytest.importorskip("torch")


class StubTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor(
            [[float(len(token)) for token in text.split()]], dtype=torch.long
        )
        return {"input_ids": ids}


class StubModel:
    """Logits shaped like a real LM (one distribution per input token,
    predicting the next token) so the per-token NLL of each predicted
    token equals that token's id value (see _token_nlls consumption)."""

    VOCAB = 8

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def eval(self):
        return self

    def __call__(self, input_ids=None, labels=None, **kwargs):
        length = input_ids.shape[-1]
        logits = torch.zeros(1, length, self.VOCAB)
        for position in range(length - 1):
            label = int(input_ids[0, position + 1])
            nll = float(input_ids[0, position + 1])
            logits[0, position, label] = math.log(
                (self.VOCAB - 1) / (math.exp(nll) - 1)
            )
        loss = None
        if labels is not None:
            loss = labels.float().mean()
        return type("Outputs", (), {"logits": logits, "loss": loss})()


@pytest.fixture
def stub_transformers(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        AutoTokenizer=StubTokenizer, AutoModelForCausalLM=StubModel
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    return fake
