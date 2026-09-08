"""Tool library model for the tool-selection robustness harness.

:class:`ToolLibrary` is an immutable-by-convention container of
:class:`ToolDocument` objects. All mutating-style operations return a new
library instead of modifying the receiver, which lets researchers snapshot
libraries before and after modifications when running controlled
tool-selection robustness experiments.

Purpose note: this module supports defensive research and benchmarking of
tool retrieval/selection robustness in LLM agents. It provides the data
plumbing for experiments (e.g., measuring how selection accuracy degrades
under modified tool registries); it does not generate exploits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tool_selection_harness.core.tool_document import ToolDocument


@dataclass
class ToolLibrary:
    """An ordered collection of tool documents available to an LLM agent.

    Methods that would conventionally mutate the library (``add``,
    ``remove``, ``inject``) instead return a new :class:`ToolLibrary`
    instance, leaving the original untouched. This copy-on-write behavior is
    deliberate: evaluation experiments frequently need to compare agent
    behavior against a pristine baseline library and one or more modified
    libraries simultaneously.
    """

    documents: list[ToolDocument] = field(default_factory=list)

    # -- construction helpers -------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "ToolLibrary":
        """Construct a library from a JSON file on disk.

        Args:
            path: Path to a JSON file containing a list of tool document
                objects, each with ``tool_name`` and ``tool_description``
                keys (and optionally ``metadata``).
        """
        data = _read_json(Path(path))
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list of tool documents in {path}, "
                f"got {type(data).__name__}"
            )
        documents = [_document_from_mapping(item) for item in data]
        return cls(documents=documents)

    # -- queries --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __contains__(self, name: str) -> bool:
        return any(doc.tool_name == name for doc in self.documents)

    def get(self, name: str) -> ToolDocument | None:
        """Return the document with ``name``, or ``None`` if absent."""
        for doc in self.documents:
            if doc.tool_name == name:
                return doc
        return None

    def names(self) -> list[str]:
        """Return the tool names in library order."""
        return [doc.tool_name for doc in self.documents]

    # -- copy-on-write operations ----------------------------------------

    def add(self, doc: ToolDocument) -> "ToolLibrary":
        """Return a new library with ``doc`` appended.

        The receiver is left unchanged. If a document with the same
        ``tool_name`` already exists, the new document is appended after it
