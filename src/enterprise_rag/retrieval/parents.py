"""Bounded parent expansion over already-authorized retrieval references."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real

from .embeddings import RealScalar


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _normalize_page_numbers(page_numbers: Sequence[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for page_number in page_numbers:
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number <= 0
        ):
            raise ValueError("Page numbers must be positive integers")
        normalized.add(page_number)
    return tuple(sorted(normalized))


def _validate_score(score: RealScalar) -> float:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("Child score must be a real number")
    normalized = float(score)
    if not isfinite(normalized):
        raise ValueError("Child score must be finite")
    return normalized


def _validate_rank(rank: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("Child rank must be a positive integer")


def _validate_limit(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ChildReference:
    """Candidate metadata whose authorization was established upstream."""

    child_id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    run_id: str
    parent_id: str | None
    rank: int
    score: float
    page_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "child_id",
            "tenant_id",
            "document_id",
            "document_version_id",
            "run_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.parent_id is not None:
            _require_non_empty(self.parent_id, "parent_id")
        _validate_rank(self.rank)
        object.__setattr__(self, "score", _validate_score(self.score))
        object.__setattr__(
            self, "page_numbers", _normalize_page_numbers(self.page_numbers)
        )


@dataclass(frozen=True, slots=True)
class ParentRecord:
    """Parent context and identity returned by an authorized parent store."""

    parent_id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    run_id: str
    text: str
    page_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "parent_id",
            "tenant_id",
            "document_id",
            "document_version_id",
            "run_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Parent text must be non-empty")
        object.__setattr__(
            self, "page_numbers", _normalize_page_numbers(self.page_numbers)
        )


@dataclass(frozen=True, slots=True)
class ChildCitation:
    """Precise child-level provenance retained beside expanded parent text."""

    child_id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    run_id: str
    rank: int
    score: float
    page_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExpandedParent:
    """One bounded parent context and the child citations supporting it."""

    parent_id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    run_id: str
    text: str
    page_numbers: tuple[int, ...]
    best_score: float
    child_citations: tuple[ChildCitation, ...]


@dataclass(frozen=True, slots=True)
class ParentExpansionResult:
    """Expanded contexts plus all citations and any cap omissions."""

    parents: tuple[ExpandedParent, ...]
    child_citations: tuple[ChildCitation, ...]
    omitted_child_ids: tuple[str, ...]


def _citation_key(citation: ChildCitation) -> tuple[int, float, str]:
    return citation.rank, -citation.score, citation.child_id


def _to_citation(child: ChildReference) -> ChildCitation:
    return ChildCitation(
        child_id=child.child_id,
        tenant_id=child.tenant_id,
        document_id=child.document_id,
        document_version_id=child.document_version_id,
        run_id=child.run_id,
        rank=child.rank,
        score=child.score,
        page_numbers=child.page_numbers,
    )


def expand_parents(
    children: Sequence[ChildReference],
    parents: Mapping[str, ParentRecord],
    *,
    tenant_id: str,
    allowed_document_ids: Collection[str],
    active_run_ids: Collection[str],
    unavailable_parent_ids: Collection[str] = (),
    parent_limit: int = 20,
    max_children_per_parent: int = 8,
    max_context_chars: int = 12_000,
) -> ParentExpansionResult:
    """Map authorized child references to bounded, deduplicated parent context.

    Parent text is returned only when both the child and parent carry the
    requested tenant/document/run identity. The full accepted citation set is
    retained even when caps omit some parent context.
    """

    _require_non_empty(tenant_id, "Tenant id")
    _validate_limit(parent_limit, "Parent limit")
    _validate_limit(max_children_per_parent, "Maximum children per parent")
    _validate_limit(max_context_chars, "Maximum context characters")

    grouped: dict[str, list[ChildReference]] = {}
    unavailable = frozenset(unavailable_parent_ids)
    for parent_id in unavailable:
        _require_non_empty(parent_id, "Unavailable parent id")
    seen_child_ids: set[str] = set()
    for child in children:
        if child.child_id in seen_child_ids:
            raise ValueError("Expansion contains a duplicate child id")
        seen_child_ids.add(child.child_id)
        if child.tenant_id != tenant_id:
            raise ValueError("Child reference tenant is outside requested scope")
        if child.document_id not in allowed_document_ids:
            raise ValueError("Child reference document is outside requested scope")
        if child.run_id not in active_run_ids:
            raise ValueError("Child reference run is outside requested scope")
        if child.parent_id is None:
            raise ValueError("Child reference parent id must be present")

        parent = parents.get(child.parent_id)
        if parent is None:
            if child.parent_id in unavailable:
                grouped.setdefault(child.parent_id, []).append(child)
                continue
            raise ValueError(f"Missing parent record for {child.parent_id}")
        if child.parent_id in unavailable:
            raise ValueError("Parent cannot be both available and unavailable")
        if (
            parent.parent_id != child.parent_id
            or parent.tenant_id != child.tenant_id
            or parent.document_id != child.document_id
            or parent.document_version_id != child.document_version_id
            or parent.run_id != child.run_id
        ):
            raise ValueError("Parent record identity does not match child reference")
        grouped.setdefault(child.parent_id, []).append(child)

    all_citations = tuple(
        sorted(
            (_to_citation(child) for child in children),
            key=_citation_key,
        )
    )
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            min(child.rank for child in item[1]),
            -max(child.score for child in item[1]),
            item[0],
        ),
    )

    expanded: list[ExpandedParent] = []
    omitted_ids: set[str] = set()
    used_context_chars = 0
    for parent_id, group in ordered_groups:
        if parent_id in unavailable:
            omitted_ids.update(child.child_id for child in group)
            continue
        if len(expanded) >= parent_limit:
            omitted_ids.update(child.child_id for child in group)
            continue
        parent = parents[parent_id]
        if used_context_chars + len(parent.text) > max_context_chars:
            omitted_ids.update(child.child_id for child in group)
            continue

        citations = tuple(
            sorted((_to_citation(child) for child in group), key=_citation_key)
        )
        selected = citations[:max_children_per_parent]
        omitted_ids.update(
            citation.child_id for citation in citations[max_children_per_parent:]
        )
        expanded.append(
            ExpandedParent(
                parent_id=parent.parent_id,
                tenant_id=parent.tenant_id,
                document_id=parent.document_id,
                document_version_id=parent.document_version_id,
                run_id=parent.run_id,
                text=parent.text,
                page_numbers=parent.page_numbers,
                best_score=max(child.score for child in group),
                child_citations=selected,
            )
        )
        used_context_chars += len(parent.text)

    return ParentExpansionResult(
        parents=tuple(expanded),
        child_citations=all_citations,
        omitted_child_ids=tuple(
            citation.child_id
            for citation in all_citations
            if citation.child_id in omitted_ids
        ),
    )


def parent_context_references(expansion):
    return tuple(
        {
            "parent_id": p.parent_id,
            "tenant_id": p.tenant_id,
            "document_id": p.document_id,
            "document_version_id": p.document_version_id,
            "run_id": p.run_id,
            "content_sha256": sha256(p.text.encode()).hexdigest(),
            "child_ids": [c.child_id for c in p.child_citations],
        }
        for p in expansion.parents
    )
