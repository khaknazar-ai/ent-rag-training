"""Validation helpers for versioned embedding batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Protocol, TypeAlias

from enterprise_rag.domain.documents.ids import sha256_bytes

RealScalar: TypeAlias = int | float | Real


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Versioned identity of an embedding representation and its metric."""

    provider_id: str
    model_revision: str
    dimension: int
    preprocessing_revision: str
    metric: str = "cosine"

    def __post_init__(self) -> None:
        for field_name in (
            "provider_id",
            "model_revision",
            "preprocessing_revision",
            "metric",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Embedding {field_name} must be non-empty")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise ValueError("Embedding dimension must be a positive integer")
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")

    @property
    def compatibility_key(self) -> tuple[str, str, int, str, str]:
        """Return the immutable key required for safe vector reuse."""

        return (
            self.provider_id,
            self.model_revision,
            self.dimension,
            self.preprocessing_revision,
            self.metric,
        )

    def require_compatible(self, other: EmbeddingSpace) -> None:
        """Reject vectors produced in a different embedding space."""

        if self.compatibility_key != other.compatibility_key:
            raise ValueError("Embedding spaces are not compatible")


class EmbeddingProvider(Protocol):
    """Minimal provider boundary; concrete model clients stay outside lite."""

    @property
    def space(self) -> EmbeddingSpace: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[RealScalar]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Tenant- and embedding-space-scoped key for normalized content."""

    tenant_id: str
    space: EmbeddingSpace
    normalized_content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("Embedding cache tenant_id must be non-empty")
        digest = self.normalized_content_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Embedding cache content hash must be lowercase SHA-256")

    @classmethod
    def from_normalized_content(
        cls,
        tenant_id: str,
        space: EmbeddingSpace,
        normalized_content: str,
    ) -> EmbeddingCacheKey:
        if not isinstance(normalized_content, str):
            raise ValueError("Normalized embedding content must be text")
        return cls(
            tenant_id=tenant_id,
            space=space,
            normalized_content_sha256=sha256_bytes(normalized_content.encode("utf-8")),
        )


@dataclass(slots=True)
class InMemoryEmbeddingCache:
    """Small deterministic cache for local tests and the lite reference path."""

    _values: dict[EmbeddingCacheKey, tuple[float, ...]] = field(default_factory=dict)

    def put(
        self, key: EmbeddingCacheKey, vector: Sequence[RealScalar]
    ) -> tuple[float, ...]:
        normalized = validate_vector(vector, expected_dimension=key.space.dimension)
        self._values[key] = normalized
        return normalized

    def get(self, key: EmbeddingCacheKey) -> tuple[float, ...] | None:
        return self._values.get(key)


def validate_vector(
    vector: Sequence[RealScalar], *, expected_dimension: int | None = None
) -> tuple[float, ...]:
    """Normalize one vector and reject malformed or non-finite values."""

    if not vector:
        raise ValueError("Embedding vector must be non-empty")
    normalized: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Embedding values must be real numbers")
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError("Embedding values must be finite") from None
        if not isfinite(converted):
            raise ValueError("Embedding values must be finite")
        normalized.append(converted)
    if expected_dimension is not None and len(normalized) != expected_dimension:
        raise ValueError(
            "Embedding vector dimension must match the configured dimension"
        )
    return tuple(normalized)


def validate_embedding_batch(
    items: Sequence[object],
    vectors: Sequence[Sequence[RealScalar]],
    *,
    expected_dimension: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Validate item/vector alignment and return immutable normalized vectors."""

    if not items or not vectors:
        raise ValueError("Embedding batch must be non-empty")
    if len(items) != len(vectors):
        raise ValueError("Embedding items and vectors must have the same number")

    normalized: list[tuple[float, ...]] = []
    dimension = expected_dimension
    for vector in vectors:
        current = validate_vector(vector, expected_dimension=dimension)
        if dimension is None:
            dimension = len(current)
        normalized.append(current)
    return tuple(normalized)


def validate_provider_output(
    space: EmbeddingSpace,
    items: Sequence[object],
    vectors: Sequence[Sequence[RealScalar]],
) -> tuple[tuple[float, ...], ...]:
    """Validate provider output against its declared embedding space."""

    return validate_embedding_batch(
        items,
        vectors,
        expected_dimension=space.dimension,
    )
