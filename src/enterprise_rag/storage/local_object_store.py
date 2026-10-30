"""A small streamed local object-store adapter for the Docker-free profile."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from enterprise_rag.domain.documents.ids import sha256_stream
from enterprise_rag.streams import BinaryReader


class DuplicateObject(FileExistsError):
    """Raised when a key already contains different bytes."""


class ObjectTooLarge(ValueError):
    """Raised when a streamed object exceeds the configured byte limit."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    size: int
    sha256: str
    already_exists: bool = False


class LocalObjectStore:
    """Store objects under a controlled root without allowing path traversal."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        if (
            not key
            or "\x00" in key
            or "\\" in key
            or Path(key).is_absolute()
            or PureWindowsPath(key).drive
            or PureWindowsPath(key).root
        ):
            raise ValueError("Object key must be a non-empty relative path")
        parts = key.split("/")
        reserved_names = {
            "aux",
            "con",
            "nul",
            "prn",
            *(f"com{index}" for index in range(1, 10)),
            *(f"lpt{index}" for index in range(1, 10)),
        }
        if any(not part or part in {".", ".."} or ":" in part for part in parts) or any(
            part.rstrip(" .").casefold() in reserved_names for part in parts
        ):
            raise ValueError("Object key contains an unsafe path component")
        candidate = self.root.joinpath(*parts)
        resolved_parent = candidate.parent.resolve()
        if resolved_parent != self.root and self.root not in resolved_parent.parents:
            raise ValueError("Object key escapes the object-store root")
        if os.path.lexists(candidate):
            resolved_candidate = candidate.resolve()
            if (
                resolved_candidate != self.root
                and self.root not in resolved_candidate.parents
            ):
                raise ValueError("Object key escapes the object-store root")
        return candidate

    def put_stream(
        self, key: str, stream: BinaryReader, *, max_bytes: int | None = None
    ) -> StoredObject:
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        destination = self._path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                digest_builder = hashlib.sha256()
                size = 0
                while chunk := stream.read(1024 * 1024):
                    next_size = size + len(chunk)
                    if max_bytes is not None and next_size > max_bytes:
                        raise ObjectTooLarge(
                            f"Object exceeds the {max_bytes}-byte limit"
                        )
                    temporary.write(chunk)
                    digest_builder.update(chunk)
                    size = next_size
                digest = digest_builder.hexdigest()

            try:
                if os.name == "nt":
                    # Windows rename is atomic and fails when the destination
                    # already exists; POSIX rename would replace an existing
                    # object, so keep the hard-link publication there.
                    os.rename(temporary_path, destination)
                else:
                    os.link(temporary_path, destination)
            except FileExistsError:
                existing_digest, existing_size = self._digest(destination)
                if existing_digest == digest and existing_size == size:
                    return StoredObject(key, size, digest, already_exists=True)
                raise DuplicateObject(
                    f"Object key already contains different bytes: {key}"
                ) from None
            return StoredObject(key, size, digest)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def put_bytes(self, key: str, content: bytes) -> StoredObject:
        from io import BytesIO

        return self.put_stream(key, BytesIO(content))

    def open(self, key: str) -> BinaryIO:
        return self._path_for(key).open("rb")

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def delete(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str | None = None) -> tuple[str, ...]:
        """List regular files for maintenance discovery, never as deletion authority."""

        if prefix is not None and (
            "\\" in prefix
            or prefix.startswith("/")
            or PureWindowsPath(prefix).drive
            or PureWindowsPath(prefix).root
        ):
            raise ValueError("Object prefix must be a portable relative path")
        keys = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            key = path.relative_to(self.root).as_posix()
            if prefix is None or key.startswith(prefix):
                keys.append(key)
        return tuple(sorted(keys))

    @staticmethod
    def _digest(path: Path) -> tuple[str, int]:
        with path.open("rb") as stream:
            return sha256_stream(stream)
