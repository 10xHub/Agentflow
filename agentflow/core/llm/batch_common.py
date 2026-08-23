"""Shared types for provider batch helpers.

Both batch helpers expose the same surface — ``add`` / ``submit`` / ``status`` /
``wait`` / ``results`` — and return the same :class:`BatchResult`, so switching
providers does not mean relearning the interface. The provider-specific parts
(inline requests vs a JSONL upload, different status vocabularies) stay hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BatchResult:
    """One entry from a completed batch, normalised across providers."""

    custom_id: str
    status: str
    """``succeeded``, ``errored``, ``canceled``, or ``expired``."""
    text: str = ""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: Any = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"
