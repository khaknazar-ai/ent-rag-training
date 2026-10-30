"""Minimal byte-stream surfaces; ownership stays with the caller."""

from typing import Protocol


class BinaryReader(Protocol):
    """Read bytes from the current position without requiring seeking."""

    def read(self, size: int = -1, /) -> bytes: ...


class BinaryWriter(Protocol):
    """Write bytes; internal upload buffers remain standard owned spools."""

    def write(self, data: bytes, /) -> int: ...
