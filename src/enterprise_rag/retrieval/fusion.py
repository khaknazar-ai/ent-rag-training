"""Deterministic rank fusion over candidate references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Untrusted index hints that SQLite must verify before returning text."""

    document_id: str
    document_version_id: str
    run_id: str

    def __post_init__(self) -> None:
        for value in (self.document_id, self.document_version_id, self.run_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Fused candidate identity fields must be non-empty")


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One strategy's ranked reference and its original backend score."""

    candidate_id: str
    rank: int
    raw_score: float
    identity: CandidateIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("Candidate id must be non-empty")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("Candidate rank must be a positive integer")
        if isinstance(self.raw_score, bool) or not isinstance(
            self.raw_score, (int, float)
        ):
            raise ValueError("Candidate score must be finite")
        try:
            finite_score = isfinite(float(self.raw_score))
        except OverflowError:
            finite_score = False
        if not finite_score:
            raise ValueError("Candidate score must be finite")
        if self.identity is not None and not isinstance(
            self.identity, CandidateIdentity
        ):
            raise ValueError("Candidate identity must be a CandidateIdentity")


@dataclass(frozen=True, slots=True)
class FusionContribution:
    strategy: str
    rank: int
    raw_score: float
    rrf_score: float


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    candidate_id: str
    score: float
    contributions: tuple[FusionContribution, ...]
    identity: CandidateIdentity | None = None


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[RankedCandidate]],
    *,
    k: int = 60,
    limit: int = 20,
) -> tuple[FusedCandidate, ...]:
    """Fuse strategy ranks while retaining raw-score provenance."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("RRF k must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("RRF limit must be between 1 and 1000")

    contributions: dict[str, list[FusionContribution]] = {}
    identities: dict[str, CandidateIdentity | None] = {}
    for strategy, ranked in ranked_lists.items():
        if not strategy.strip():
            raise ValueError("RRF strategy name must be non-empty")
        seen_ids: set[str] = set()
        seen_ranks: set[int] = set()
        for candidate in ranked:
            if candidate.candidate_id in seen_ids:
                raise ValueError("RRF strategy contains duplicate candidate id")
            if candidate.rank in seen_ranks:
                raise ValueError("RRF strategy contains duplicate rank")
            seen_ids.add(candidate.candidate_id)
            seen_ranks.add(candidate.rank)
            if candidate.candidate_id in identities:
                if identities[candidate.candidate_id] != candidate.identity:
                    raise ValueError(
                        "RRF candidate identity conflicts across strategies"
                    )
            else:
                identities[candidate.candidate_id] = candidate.identity
            contribution = FusionContribution(
                strategy=strategy,
                rank=candidate.rank,
                raw_score=candidate.raw_score,
                rrf_score=1 / (k + candidate.rank),
            )
            contributions.setdefault(candidate.candidate_id, []).append(contribution)

    fused = [
        FusedCandidate(
            candidate_id=candidate_id,
            score=sum(item.rrf_score for item in items),
            contributions=tuple(
                sorted(items, key=lambda item: (item.strategy, item.rank))
            ),
            identity=identities[candidate_id],
        )
        for candidate_id, items in contributions.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.candidate_id))
    return tuple(fused[:limit])
