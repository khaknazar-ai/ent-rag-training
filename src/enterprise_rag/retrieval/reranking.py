"""Bounded async reranker boundary; callers must authorize text before entry.

No provider is selected or enabled here. Scores are provider-specific, not
probabilities or substitutes for canonical evidence/provenance.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from math import isfinite
from numbers import Real
from typing import Protocol

MAX_RERANK_CANDIDATES = 100
MAX_RERANK_QUERY_CHARS = 2_000
MAX_RERANK_TEXT_CHARS = 16_000
MAX_RERANK_TOTAL_CHARS = 100_000


class RerankerUnavailable(RuntimeError):
    """The configured provider failed; no ranking fallback was substituted."""


class RerankerProtocolError(RuntimeError):
    """Provider output violated the requested bounded ID/score contract."""


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """Canonical, already-authorized text; document content remains untrusted."""

    candidate_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """Scored reference only: a provider cannot return replacement evidence."""

    candidate_id: str
    score: float


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        top_n: int,
    ) -> Sequence[RerankedCandidate]: ...


def validate_rerank_request(
    query: str, *, top_n: int = 10, timeout_seconds: float = 5.0
) -> float:
    """Validate scalar bounds before any data access; return normalized timeout."""

    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > MAX_RERANK_QUERY_CHARS
    ):
        raise ValueError("Rerank query must contain 1-2000 characters")
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or not 1 <= top_n <= MAX_RERANK_CANDIDATES
    ):
        raise ValueError("Rerank top_n must be between 1 and 100")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
        raise ValueError("Rerank timeout must be a finite positive number")
    try:
        timeout = float(timeout_seconds)
    except OverflowError as error:
        raise ValueError("Rerank timeout must be a finite positive number") from error
    if not isfinite(timeout) or not 0 < timeout <= 30:
        raise ValueError("Rerank timeout must be greater than zero and at most 30")
    return timeout


async def rerank_candidates(
    query: str,
    candidates: Sequence[RerankCandidate],
    provider: Reranker,
    *,
    top_n: int = 10,
    timeout_seconds: float = 5.0,
) -> tuple[RerankedCandidate, ...]:
    """Validate a complete batch and require exactly the requested top-N IDs.

    Sorting uses descending provider score and original input order for ties.
    The timeout uses cooperative asyncio cancellation: in-process providers must
    yield and honor cancellation. This is not a process-isolation deadline.
    ACL/version revalidation after inference belongs to the calling use case.
    """

    timeout = validate_rerank_request(
        query, top_n=top_n, timeout_seconds=timeout_seconds
    )
    if not isinstance(candidates, Sequence) or isinstance(
        candidates, (str, bytes, bytearray)
    ):
        raise ValueError("Rerank input must contain at most 100 candidates")
    # A Sequence may misreport its length; cap actual iteration before copying.
    batch = tuple(islice(candidates, MAX_RERANK_CANDIDATES + 1))
    if len(batch) > MAX_RERANK_CANDIDATES:
        raise ValueError("Rerank input must contain at most 100 candidates")
    positions: dict[str, int] = {}
    total_chars = 0
    for position, candidate in enumerate(batch):
        if not isinstance(candidate, RerankCandidate):
            raise ValueError("Rerank input contains an invalid candidate")
        if (
            not isinstance(candidate.candidate_id, str)
            or not candidate.candidate_id.strip()
            or len(candidate.candidate_id) > 512
        ):
            raise ValueError("Rerank candidate ID must contain 1-512 characters")
        if (
            not isinstance(candidate.text, str)
            or not candidate.text.strip()
            or len(candidate.text) > MAX_RERANK_TEXT_CHARS
        ):
            raise ValueError("Rerank candidate text must contain 1-16000 characters")
        if candidate.candidate_id in positions:
            raise ValueError("Rerank input contains duplicate candidate IDs")
        positions[candidate.candidate_id] = position
        total_chars += len(candidate.text)
        if total_chars > MAX_RERANK_TOTAL_CHARS:
            raise ValueError("Rerank input exceeds the total text budget")
    if not batch:
        return ()

    requested = min(top_n, len(batch))
    try:
        output = await asyncio.wait_for(
            provider.rerank(query, batch, requested), timeout=timeout
        )
    except TimeoutError as error:
        raise RerankerUnavailable("Reranker timed out") from error
    except Exception as error:
        raise RerankerUnavailable("Reranker provider failed") from error
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes, bytearray)):
        raise RerankerProtocolError("Reranker must return exactly the requested IDs")
    scored = tuple(islice(output, requested + 1))
    if len(scored) != requested:
        raise RerankerProtocolError("Reranker must return exactly the requested IDs")
    seen: set[str] = set()
    normalized: list[RerankedCandidate] = []
    for item in scored:
        if not isinstance(item, RerankedCandidate):
            raise RerankerProtocolError("Reranker returned an invalid scored reference")
        if not isinstance(item.candidate_id, str) or item.candidate_id not in positions:
            raise RerankerProtocolError("Reranker returned an unknown candidate ID")
        if item.candidate_id in seen:
            raise RerankerProtocolError("Reranker returned duplicate candidate IDs")
        if isinstance(item.score, bool) or not isinstance(item.score, Real):
            raise RerankerProtocolError("Reranker score must be a finite real number")
        try:
            score = float(item.score)
        except OverflowError as error:
            raise RerankerProtocolError("Reranker score must be finite") from error
        if not isfinite(score):
            raise RerankerProtocolError("Reranker score must be finite")
        seen.add(item.candidate_id)
        normalized.append(RerankedCandidate(item.candidate_id, score))
    return tuple(
        sorted(normalized, key=lambda item: (-item.score, positions[item.candidate_id]))
    )
