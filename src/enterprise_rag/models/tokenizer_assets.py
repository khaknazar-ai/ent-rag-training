"""Small tokenizer-specific cache: pinned read handles, identity and digests.

Windows handles deny write/delete on files and rename on directory ancestors.
The expensive model hash is performed once at startup, never per prompt.
No persistent cache is trusted across restarts; changed assets require rebind.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import stat
import struct
from ctypes import wintypes as w
from pathlib import Path
from typing import BinaryIO, cast


class AssetIdentityError(ValueError):
    pass


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _handle_identity(value: os.stat_result) -> tuple[int, ...]:
    # CPython 3.12 Windows fstat uses creation time for st_ctime, whereas lstat
    # exposes change time. Compare stable file ID/size/write-time/link count;
    # compare path change-time separately against the cached lstat snapshot.
    identity = _identity(value)
    return identity[:4] + identity[5:]


class PinnedAssets:
    def __init__(self, files: dict[Path, str | None]):
        if os.name != "nt":
            raise AssetIdentityError("tokenizer_windows_required")
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        self._api.CreateFileW.argtypes = [
            w.LPCWSTR,
            w.DWORD,
            w.DWORD,
            ctypes.c_void_p,
            w.DWORD,
            w.DWORD,
            w.HANDLE,
        ]
        self._api.CreateFileW.restype = w.HANDLE
        self._api.CloseHandle.argtypes = [w.HANDLE]
        self._directory_fds: dict[Path, int] = {}
        self._streams: dict[Path, BinaryIO] = {}
        self._identities: dict[Path, tuple[int, ...]] = {}
        self.digests: dict[str, str] = {}
        self._closed = False
        try:
            for path, expected in files.items():
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or str(path).startswith("\\\\")
                ):
                    raise AssetIdentityError("tokenizer_asset_path")
                for directory in reversed(path.parents):
                    if directory not in self._identities:
                        self._pin(directory, directory=True)
                self._pin(path, directory=False)
                stream = self._streams[path]
                digest = hashlib.file_digest(
                    cast(io.BufferedReader, stream), "sha256"
                ).hexdigest()
                stream.seek(0)
                self.digests[str(path)] = digest
                if expected is not None and digest != expected:
                    raise AssetIdentityError("tokenizer_asset_digest_mismatch")
            self.validate()
        except BaseException:
            self.close()
            raise

    def _pin(self, path: Path, *, directory: bool) -> None:
        import msvcrt

        before = path.lstat()
        if getattr(before, "st_file_attributes", 0) & 0x400:
            raise AssetIdentityError("tokenizer_reparse_point")
        if directory != stat.S_ISDIR(before.st_mode):
            raise AssetIdentityError("tokenizer_asset_type")
        if not directory and (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1):
            raise AssetIdentityError("tokenizer_asset_type")
        # OPEN_REPARSE_POINT + BACKUP_SEMANTICS; dirs allow normal contents I/O,
        # but not directory replacement. Files allow other readers only.
        handle = self._api.CreateFileW(
            str(path),
            0 if directory else 0x80000000,
            3 if directory else 1,
            None,
            3,
            0x02200000,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise AssetIdentityError("tokenizer_asset_pin_failed")
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            self._api.CloseHandle(handle)
            raise
        os.set_inheritable(fd, False)
        if directory:
            self._directory_fds[path] = fd
            opened = os.fstat(fd)
            if (
                _identity(opened)[:2] != _identity(before)[:2]
                or not stat.S_ISDIR(opened.st_mode)
                or getattr(opened, "st_file_attributes", 0) & 0x400
            ):
                raise AssetIdentityError("tokenizer_asset_changed")
        else:
            self._streams[path] = os.fdopen(fd, "rb")
            if _handle_identity(os.fstat(fd)) != _handle_identity(before):
                raise AssetIdentityError("tokenizer_asset_changed")
        after = path.lstat()
        if (
            _identity(after) != _identity(before)
            or getattr(after, "st_file_attributes", 0) & 0x400
        ):
            raise AssetIdentityError("tokenizer_asset_changed")
        self._identities[path] = _identity(after)

    def validate(self) -> None:
        if self._closed:
            raise AssetIdentityError("tokenizer_binding_closed")
        try:
            for path, expected in self._identities.items():
                current = path.lstat()
                # Directory size/mtime can change when an unrelated file appears.
                actual = _identity(current)
                if path in self._streams:
                    handle = _handle_identity(os.fstat(self._streams[path].fileno()))
                    matches = (
                        actual == expected and handle == expected[:4] + expected[5:]
                    )
                else:
                    handle_id = _identity(os.fstat(self._directory_fds[path]))[:2]
                    matches = actual[:2] == expected[:2] == handle_id
                if not matches or getattr(current, "st_file_attributes", 0) & 0x400:
                    raise AssetIdentityError("tokenizer_asset_changed")
        except (OSError, ValueError):
            self.close()
            raise AssetIdentityError("tokenizer_asset_changed") from None

    def read(self, path: Path, limit: int) -> bytes:
        self.validate()
        stream = self._streams[path]
        stream.seek(0)
        raw = stream.read(limit + 1)
        stream.seek(0)
        if len(raw) > limit:
            raise AssetIdentityError("tokenizer_asset_size")
        return raw

    def vocabulary_metadata(self, path: Path) -> dict:
        """Read GGUF v3 KVs only, never the tensor directory or tensor data."""
        self.validate()
        stream = self._streams[path]
        stream.seek(0)
        digest = hashlib.sha256()

        def read(size: int) -> bytes:
            if size < 0 or stream.tell() + size > 16 * 1024 * 1024:
                raise AssetIdentityError("tokenizer_metadata_limit")
            raw = stream.read(size)
            if len(raw) != size:
                raise AssetIdentityError("tokenizer_metadata_truncated")
            digest.update(raw)
            return raw

        def number(fmt: str):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

        def string() -> str:
            return read(number("Q")).decode("utf-8", errors="strict")

        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }

        def scalar(kind: int):
            if kind == 8:
                return string()
            if kind not in formats:
                raise AssetIdentityError("tokenizer_metadata_type")
            return number(formats[kind])

        if read(4) != b"GGUF" or number("I") != 3:
            raise AssetIdentityError("tokenizer_gguf_version")
        number("Q")  # Tensor count; no tensor directory/data is read.
        count = number("Q")
        if not 1 <= count <= 4096:
            raise AssetIdentityError("tokenizer_metadata_limit")
        result = {}
        special = {}
        seen = set()
        for _ in range(count):
            key = string()
            if key in seen:
                raise AssetIdentityError("tokenizer_metadata_duplicate")
            seen.add(key)
            kind = number("I")
            if kind == 9:
                subtype, size = number("I"), number("Q")
                if size > 1_000_000:
                    raise AssetIdentityError("tokenizer_metadata_limit")
                for index in range(size):
                    value = scalar(subtype)
                    if key == "tokenizer.ggml.tokens" and value in (
                        "<|im_start|>",
                        "<|im_end|>",
                        "<|endoftext|>",
                    ):
                        special[value] = index
            else:
                value = scalar(kind)
                if key.startswith("tokenizer.") or key == "general.architecture":
                    result[key] = value
        result["metadata_sha256"] = digest.hexdigest()
        result["metadata_bytes"] = stream.tell()
        result["special_ids"] = special
        stream.seek(0)
        self.validate()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stream in self._streams.values():
            stream.close()
        for fd in self._directory_fds.values():
            os.close(fd)
        self._directory_fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
