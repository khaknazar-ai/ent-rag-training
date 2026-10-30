"""Require actual synthetic evidence without fabricating missing metadata."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def require_evidence(value: T | None, message: str) -> T:
    """Keep absent required evidence a diagnostic failure, never a default."""
    if value is None:
        raise RuntimeError(message)
    return value
