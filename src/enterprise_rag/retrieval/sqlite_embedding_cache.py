"""Opt-in durable vector cache; separate from canonical document metadata.

Only trusted internal callers may use this cache. A cache hit does not grant
document access, prove provenance, or replace live retrieval authorization.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from hashlib import sha256
from itertools import islice
from pathlib import Path

from .embeddings import EmbeddingCacheKey, EmbeddingSpace, RealScalar, validate_vector

APPLICATION_ID = 0x45524348
SCHEMA_VERSION = 1
MAX_DIMENSION = 8192
MAX_ENTRIES = 10_000
MAX_VECTOR_BYTES = 524_288
_KIND = "enterprise-rag-embedding-cache"
_COLUMNS = "tenant, provider, model, dimension, preprocessing, metric, content_hash"
_WHERE = " AND ".join(f"{name}=?" for name in _COLUMNS.split(", "))


class EmbeddingCacheFormatError(ValueError):
    """The path is not this adapter's known private cache format."""


class EmbeddingCacheCorruption(RuntimeError):
    """Stored data is invalid; it must not become an implicit cache miss."""


class EmbeddingCacheConflict(RuntimeError):
    """One content/space identity already has a different immutable vector."""


class EmbeddingCacheFull(RuntimeError):
    """The explicit local entry budget is exhausted; no eviction occurs."""


class SQLiteEmbeddingCache:
    """Bounded immutable vectors, keyed by full tenant/content/space identity.

    Opening never creates/migrates an existing database. Use create() once on a
    new private file. Each operation closes its connection; writes are atomic
    and serialized by SQLite. Operational/lock/disk failures remain explicit.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError("An existing embedding cache file is required")
        with closing(self._connect(write=False)) as connection:
            self.max_entries = self._check_format(connection)

    @classmethod
    def create(cls, path: Path, *, max_entries: int = 1024) -> SQLiteEmbeddingCache:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= MAX_ENTRIES
        ):
            raise ValueError("Embedding cache max_entries must be between 1 and 10000")
        path = Path(path).resolve()
        # Exclusive creation preserves every existing file, including wrong DBs.
        with path.open("xb"):
            pass
        with closing(
            sqlite3.connect(path, timeout=2, isolation_level=None)
        ) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE cache_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                    "kind TEXT NOT NULL, max_entries INTEGER NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE embedding_vectors ("
                    "tenant TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, "
                    "dimension INTEGER NOT NULL CHECK(dimension BETWEEN 1 AND 8192), "
                    "preprocessing TEXT NOT NULL, metric TEXT NOT NULL, "
                    "content_hash TEXT NOT NULL, vector_json TEXT NOT NULL "
                    "CHECK(length(vector_json) BETWEEN 1 AND 524288), "
                    "vector_sha256 TEXT NOT NULL, "
                    f"PRIMARY KEY ({_COLUMNS})) WITHOUT ROWID"
                )
                connection.execute(
                    "INSERT INTO cache_meta VALUES (1, ?, ?)", (_KIND, max_entries)
                )
                connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                # Preserve the newly created file for diagnosis; never delete it.
                raise
        return cls(path)

    def _connect(self, *, write: bool) -> sqlite3.Connection:
        mode = "rw" if write else "ro"
        return sqlite3.connect(
            self.path.as_uri() + "?mode=" + mode,
            uri=True,
            timeout=2,
            isolation_level=None,
        )

    @staticmethod
    def _check_format(connection: sqlite3.Connection) -> int:
        try:
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if application_id != APPLICATION_ID or version != SCHEMA_VERSION:
                raise EmbeddingCacheFormatError("Unrecognized embedding cache format")
            row = connection.execute(
                "SELECT kind, max_entries FROM cache_meta WHERE singleton=1"
            ).fetchone()
            if (
                row is None
                or row[0] != _KIND
                or not isinstance(row[1], int)
                or not 1 <= row[1] <= MAX_ENTRIES
            ):
                raise EmbeddingCacheFormatError("Invalid embedding cache metadata")
            return row[1]
        except sqlite3.DatabaseError as error:
            raise EmbeddingCacheFormatError(
                "Unrecognized embedding cache format"
            ) from error

    @staticmethod
    def _key_values(key: EmbeddingCacheKey) -> tuple[str | int, ...]:
        if not isinstance(key, EmbeddingCacheKey) or not isinstance(
            key.space, EmbeddingSpace
        ):
            raise ValueError("A versioned embedding cache key is required")
        space = key.space
        values = (
            key.tenant_id,
            space.provider_id,
            space.model_revision,
            space.preprocessing_revision,
            space.metric,
        )
        if any(
            not isinstance(value, str) or not 1 <= len(value) <= 512 for value in values
        ):
            raise ValueError("Embedding cache key fields must contain 1-512 characters")
        if space.dimension > MAX_DIMENSION:
            raise ValueError("Embedding cache dimension exceeds the local budget")
        return (
            key.tenant_id,
            space.provider_id,
            space.model_revision,
            space.dimension,
            space.preprocessing_revision,
            space.metric,
            key.normalized_content_sha256,
        )

    def _read(
        self,
        connection: sqlite3.Connection,
        values: tuple[str | int, ...],
        dimension: int,
    ) -> tuple[float, ...] | None:
        row = connection.execute(
            "SELECT CASE WHEN length(CAST(vector_json AS BLOB))<=? "
            "THEN vector_json END, CASE WHEN length(vector_sha256)=64 "
            "THEN vector_sha256 END FROM embedding_vectors WHERE " + _WHERE,
            (MAX_VECTOR_BYTES, *values),
        ).fetchone()
        if row is None:
            return None
        payload, checksum = row
        if (
            not isinstance(payload, str)
            or not isinstance(checksum, str)
            or sha256(payload.encode("utf-8")).hexdigest() != checksum
        ):
            raise EmbeddingCacheCorruption(
                "Embedding cache payload checksum is invalid"
            )
        try:
            vector = json.loads(payload)
            if not isinstance(vector, list):
                raise ValueError("Expected a vector array")
            return validate_vector(vector, expected_dimension=dimension)
        except (ValueError, TypeError, RecursionError) as error:
            raise EmbeddingCacheCorruption(
                "Embedding cache vector is invalid"
            ) from error

    def get(self, key: EmbeddingCacheKey) -> tuple[float, ...] | None:
        values = self._key_values(key)
        with closing(self._connect(write=False)) as connection:
            connection.execute("BEGIN")
            if self._check_format(connection) != self.max_entries:
                raise EmbeddingCacheFormatError("Embedding cache settings changed")
            return self._read(connection, values, key.space.dimension)

    def put(
        self, key: EmbeddingCacheKey, vector: Sequence[RealScalar]
    ) -> tuple[float, ...]:
        values = self._key_values(key)
        if not isinstance(vector, Sequence) or isinstance(
            vector, (str, bytes, bytearray)
        ):
            raise ValueError("Embedding cache vector must be a numeric sequence")
        bounded = tuple(islice(vector, key.space.dimension + 1))
        normalized = validate_vector(bounded, expected_dimension=key.space.dimension)
        payload = json.dumps(normalized, separators=(",", ":"), allow_nan=False)
        checksum = sha256(payload.encode("utf-8")).hexdigest()
        with closing(self._connect(write=True)) as connection:
            try:
                if self._check_format(connection) != self.max_entries:
                    raise EmbeddingCacheFormatError("Embedding cache settings changed")
                connection.execute("BEGIN IMMEDIATE")
                if self._check_format(connection) != self.max_entries:
                    raise EmbeddingCacheFormatError("Embedding cache settings changed")
                existing = self._read(connection, values, key.space.dimension)
                if existing is not None:
                    if existing != normalized:
                        raise EmbeddingCacheConflict(
                            "Embedding cache vector is immutable"
                        )
                    connection.commit()
                    return existing
                count = connection.execute(
                    "SELECT count(*) FROM embedding_vectors"
                ).fetchone()[0]
                if count >= self.max_entries:
                    raise EmbeddingCacheFull(
                        "Embedding cache entry budget is exhausted"
                    )
                connection.execute(
                    f"INSERT INTO embedding_vectors ({_COLUMNS}, vector_json, vector_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, payload, checksum),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return normalized
