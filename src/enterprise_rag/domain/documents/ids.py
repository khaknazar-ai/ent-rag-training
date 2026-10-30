"""Deterministic identity and content hashing helpers."""

from __future__ import annotations

import hashlib
import json
import uuid

from enterprise_rag.streams import BinaryReader

ID_SCHEME_REVISION = 2


def deterministic_id(namespace: str, *parts: object) -> str:
    """Return a stable UUID using an unambiguous versioned encoding."""

    components = [
        {
            "type": f"{type(part).__module__}.{type(part).__qualname__}",
            "value": part.hex() if isinstance(part, bytes) else str(part),
        }
        for part in parts
    ]
    payload = json.dumps(
        {
            "revision": ID_SCHEME_REVISION,
            "namespace": namespace,
            "parts": components,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_stream(
    stream: BinaryReader, chunk_size: int = 1024 * 1024
) -> tuple[str, int]:
    """Hash bytes from the current stream position and return digest plus byte count."""

    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size
