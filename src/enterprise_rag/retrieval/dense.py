"""Deterministic exact-vector reference retrieval for small lite fixtures."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from math import fsum, hypot, isfinite

from .embeddings import RealScalar, validate_vector


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Reference-only vector metadata; document text is never stored here."""

    chunk_id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    run_id: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "tenant_id",
            "document_id",
            "document_version_id",
            "run_id",
        ):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        normalized = validate_vector(self.vector)
        object.__setattr__(self, "vector", normalized)


@dataclass(frozen=True, slots=True)
class DenseHit:
    """Authorized candidate reference returned by the exact-vector index."""

    chunk_id: str
    document_id: str
    document_version_id: str
    run_id: str
    score: float
    rank: int


class ExactVectorIndex:
    """In-memory exact cosine index for deterministic small test fixtures."""

    def __init__(self, records: Iterable[VectorRecord] = ()) -> None:
        self._records: dict[str, VectorRecord] = {}
        self._dimension: int | None = None
        for record in records:
            self.add(record)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(self, record: VectorRecord) -> None:
        """Add one immutable reference, rejecting conflicting duplicate IDs."""

        if self._dimension is None:
            self._dimension = len(record.vector)
        elif len(record.vector) != self._dimension:
            raise ValueError("Vector dimension must match the index dimension")
        existing = self._records.get(record.chunk_id)
        if existing is not None and existing != record:
            raise ValueError("Vector chunk id already has a different record")
        self._records[record.chunk_id] = record

    def search(
        self,
        query_vector: Sequence[RealScalar],
        *,
        tenant_id: str,
        allowed_document_ids: Collection[str],
        active_run_ids: Collection[str],
        limit: int = 20,
    ) -> tuple[DenseHit, ...]:
        """Return filtered candidate references ordered by exact cosine score."""

        if not tenant_id.strip():
            raise ValueError("Tenant id must be non-empty")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("Dense retrieval limit must be between 1 and 1000")
        query = validate_vector(query_vector, expected_dimension=self._dimension)
        query_norm = hypot(*query)
        candidates = (
            record
            for record in self._records.values()
            if record.tenant_id == tenant_id
            and record.document_id in allowed_document_ids
            and record.run_id in active_run_ids
        )
        ranked = sorted(
            (
                (record, _cosine_similarity(query, query_norm, record.vector))
                for record in candidates
            ),
            key=lambda item: (-item[1], item[0].chunk_id),
        )[:limit]
        return tuple(
            DenseHit(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                document_version_id=record.document_version_id,
                run_id=record.run_id,
                score=score,
                rank=rank,
            )
            for rank, (record, score) in enumerate(ranked, start=1)
        )


def _cosine_similarity(
    query: tuple[float, ...], query_norm: float, candidate: tuple[float, ...]
) -> float:
    candidate_norm = hypot(*candidate)
    if query_norm == 0 or candidate_norm == 0:
        return 0.0
    score = fsum(left * right for left, right in zip(query, candidate, strict=True)) / (
        query_norm * candidate_norm
    )
    if not isfinite(score):
        raise ValueError("Cosine similarity must be finite")
    return score
