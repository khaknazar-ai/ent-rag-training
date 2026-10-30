"""Opt-in local semantic use cases over a separately sealed vector snapshot.

Snapshots are derived references, never an authorization or source authority.
The default lexical service and canonical schema are unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice

from enterprise_rag.application.search import SearchEvidence, SearchService
from enterprise_rag.persistence import LexicalRow
from enterprise_rag.retrieval.cached_embeddings import CachedEmbeddings
from enterprise_rag.retrieval.dense import ExactVectorIndex, VectorRecord
from enterprise_rag.retrieval.embeddings import (
    EmbeddingProvider,
    validate_provider_output,
)
from enterprise_rag.retrieval.fusion import CandidateIdentity, RankedCandidate
from enterprise_rag.retrieval.lifecycle import DenseSnapshotLifecycle
from enterprise_rag.retrieval.query_processing import plan_question
from enterprise_rag.retrieval.reranking import Reranker
from enterprise_rag.retrieval.sqlite_dense import (
    MAX_RECORDS,
    CanonicalVector,
    SQLiteDenseSnapshot,
)
from enterprise_rag.security import ActorContext


@dataclass(frozen=True, slots=True)
class SemanticHit:
    evidence: SearchEvidence
    dense_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class SemanticResult:
    strategy: str
    items: tuple[SemanticHit, ...]


class SemanticProviderUnavailable(RuntimeError):
    """Configured semantic provider failed; no silent alternate strategy follows."""


class LocalSemanticService:
    """Bounded exact-cosine/RRF/rerank integration, explicitly injected by callers.

    Snapshot generation is separate from ingestion activation. Updates require a
    fresh snapshot; live canonical filters immediately omit stale entries.
    This local scan is not ANN or a million-document backend.
    """

    def __init__(
        self,
        lexical: SearchService,
        provider: EmbeddingProvider,
        snapshot: SQLiteDenseSnapshot | None = None,
        reranker: Reranker | None = None,
        *,
        lifecycle: DenseSnapshotLifecycle | None = None,
        embedding_cache: CachedEmbeddings | None = None,
        external_dense=None,
    ) -> None:
        if (
            sum(value is not None for value in (snapshot, lifecycle, external_dense))
            != 1
        ):
            raise ValueError(
                "Choose exactly one static, managed or external dense backend"
            )
        if snapshot is not None:
            provider.space.require_compatible(snapshot.space)
        self.lexical = lexical
        self.provider = provider
        self.snapshot = snapshot
        self.lifecycle = lifecycle
        self.reranker = reranker
        self.embedding_cache = embedding_cache
        self.external_dense = external_dense

    def index_authorized_chunks(
        self, actor: ActorContext, chunk_ids: Sequence[str]
    ) -> int:
        if self.snapshot is None:
            raise ValueError("Use managed reconciliation for lifecycle-backed services")
        if isinstance(chunk_ids, (str, bytes)):
            raise ValueError("A bounded explicit chunk ID collection is required")
        ids = tuple(islice(chunk_ids, MAX_RECORDS + 1))
        if not 1 <= len(ids) <= MAX_RECORDS or len(set(ids)) != len(ids):
            raise ValueError("Snapshot build requires1..2048 unique chunk IDs")
        scope = self.lexical.metadata.resolve_security_scope(actor)
        if scope is None:
            raise PermissionError("Snapshot builder is not authorized")
        authorized: list[LexicalRow] = []
        for offset in range(0, len(ids), 100):
            authorized.extend(
                self.lexical.metadata.get_authorized_evidence_for_candidates(
                    scope, ids[offset : offset + 100]
                )
            )
        by_id = {row.chunk_id: row for row in authorized}
        if set(by_id) != set(ids):
            raise PermissionError("Every requested chunk must be live and authorized")
        count = 0
        for offset in range(0, len(ids), 8):
            batch_ids = ids[offset : offset + 8]
            current_scope = self.lexical.metadata.resolve_security_scope(actor)
            if current_scope is None:
                raise PermissionError("Snapshot builder authorization was revoked")
            current = self.lexical.metadata.get_authorized_evidence_for_candidates(
                current_scope, batch_ids
            )
            batch_map = {row.chunk_id: row for row in current}
            if set(batch_map) != set(batch_ids):
                raise PermissionError("Snapshot evidence authorization changed")
            batch = tuple(batch_map[key] for key in batch_ids)
            if any(
                not self.lexical._same_canonical_evidence(
                    self.lexical._evidence_from_row(by_id[row.chunk_id]),
                    self.lexical._evidence_from_row(row),
                )
                for row in batch
            ):
                raise PermissionError(
                    "Snapshot evidence changed before provider access"
                )
            texts = tuple(row.text for row in batch)

            def still_authorized(batch=batch):
                for row in batch:
                    live = self.lexical.get_evidence(actor, row.chunk_id)
                    if live is None or not self.lexical._same_canonical_evidence(
                        self.lexical._evidence_from_row(row), live
                    ):
                        raise PermissionError(
                            "Canonical evidence changed during embedding"
                        )

            vectors = self._embed(texts, actor=actor, before_store=still_authorized)
            for row, vector in zip(batch, vectors, strict=True):
                live = self.lexical.get_evidence(actor, row.chunk_id)
                before = self.lexical._evidence_from_row(row)
                if live is None or not self.lexical._same_canonical_evidence(
                    before, live
                ):
                    raise PermissionError("Canonical evidence changed during embedding")
                ref = VectorRecord(
                    row.chunk_id,
                    scope.tenant_id,
                    row.document_id,
                    row.document_version_id,
                    row.run_id,
                    vector,
                )
                self.snapshot.put(
                    CanonicalVector(ref, sha256(row.text.encode()).hexdigest())
                )
                count += 1
        self.snapshot.seal(expected_records=count)
        return count

    def _dense(self, actor, query, limit):
        if self.external_dense is not None:
            try:
                return self.external_dense.search(self, actor, query, limit)
            except Exception as error:
                raise SemanticProviderUnavailable(
                    "Configured external dense backend failed"
                ) from error
        snapshot = self.snapshot
        if self.lifecycle is not None and actor is not None:
            snapshot = self.lifecycle.reconcile(
                self.lexical, self.provider, actor, embedding_cache=self.embedding_cache
            )
        scope = self.lexical.metadata.resolve_security_scope(actor)
        if scope is None or snapshot is None:
            return ()
        records = snapshot.records(scope.tenant_id)
        by_id = {item.reference.chunk_id: item for item in records}
        ids = tuple(by_id)
        authorized = []
        canonical = {}
        for offset in range(0, len(ids), 100):
            rows = self.lexical.metadata.get_authorized_evidence_for_candidates(
                scope, ids[offset : offset + 100]
            )
            for row in rows:
                item = by_id[row.chunk_id]
                ref = item.reference
                if (row.document_id, row.document_version_id, row.run_id) == (
                    ref.document_id,
                    ref.document_version_id,
                    ref.run_id,
                ) and sha256(row.text.encode()).hexdigest() == item.content_sha256:
                    authorized.append(item.reference)
                    canonical[row.chunk_id] = self.lexical._evidence_from_row(row)
        if not authorized:
            return ()
        vectors = self._embed((query,), actor=actor)
        # Provider/cache latency can change ACL/version membership. Refilter the
        # entire candidate set before top-K, not only the selected output.
        current_scope = self.lexical.metadata.resolve_security_scope(actor)
        if current_scope is None:
            return ()
        current = {}
        for offset in range(0, len(ids), 100):
            for row in self.lexical.metadata.get_authorized_evidence_for_candidates(
                current_scope, ids[offset : offset + 100]
            ):
                if row.chunk_id in canonical and self.lexical._same_canonical_evidence(
                    canonical[row.chunk_id], self.lexical._evidence_from_row(row)
                ):
                    current[row.chunk_id] = row
        authorized = [r for r in authorized if r.chunk_id in current]
        if not authorized:
            return ()
        index = ExactVectorIndex(authorized)
        hits = index.search(
            vectors[0],
            tenant_id=scope.tenant_id,
            allowed_document_ids={r.document_id for r in authorized},
            active_run_ids={r.run_id for r in authorized},
            limit=limit,
        )
        # Preserve dense scores and revalidate live canonical identity before use.
        hydrated = self.lexical.hydrate_authorized_dense_candidates(
            actor, hits, limit=limit
        )
        return tuple(
            item
            for item in hydrated
            if self.lexical._same_canonical_evidence(
                canonical[item.evidence.chunk_id], item.evidence
            )
        )

    def _embed(self, texts, *, actor=None, before_store=None):
        try:
            if self.embedding_cache is not None:
                scope = self.lexical.metadata.resolve_security_scope(actor)
                if scope is None:
                    raise PermissionError("Embedding actor unavailable")

                def live_fence():
                    fresh = self.lexical.metadata.resolve_security_scope(actor)
                    if fresh is None or fresh.tenant_id != scope.tenant_id:
                        raise PermissionError("Embedding actor changed")
                    if before_store is not None:
                        before_store()

                return self.embedding_cache.embed(
                    scope.tenant_id, self.provider, texts, before_store=live_fence
                )
            return validate_provider_output(
                self.provider.space, texts, self.provider.embed(texts)
            )
        except Exception as error:
            raise SemanticProviderUnavailable(
                "Configured local embedding provider failed"
            ) from error

    async def search(
        self, actor: ActorContext | None, query: str, *, strategy: str, limit: int = 4
    ) -> SemanticResult:
        if strategy not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("Unknown explicit semantic strategy")
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Local semantic result limit must be1..20")
        plan_question(query)
        if strategy == "hybrid_rerank" and self.reranker is None:
            raise ValueError("Requested reranker is not configured")
        dense = await asyncio.to_thread(self._dense, actor, query, 20)
        if strategy == "dense":
            return SemanticResult(
                strategy,
                tuple(
                    SemanticHit(item.evidence, dense_score=item.candidate.score)
                    for item in dense[:limit]
                ),
            )
        lexical = await asyncio.to_thread(self.lexical.search, actor, query, limit=20)
        dense_scores = {item.evidence.chunk_id: item.candidate.score for item in dense}
        lexical_scores = {item.chunk_id: item.score for item in lexical}
        lists = {
            "dense": tuple(
                RankedCandidate(
                    item.evidence.chunk_id,
                    i,
                    item.candidate.score,
                    CandidateIdentity(
                        item.evidence.document_id,
                        item.evidence.document_version_id,
                        item.evidence.run_id,
                    ),
                )
                for i, item in enumerate(dense, 1)
            ),
            "lexical": tuple(
                RankedCandidate(
                    item.chunk_id,
                    i,
                    item.score,
                    CandidateIdentity(
                        item.document_id, item.document_version_id, item.run_id
                    ),
                )
                for i, item in enumerate(lexical, 1)
            ),
        }
        fused = await asyncio.to_thread(
            self.lexical.fuse_and_hydrate_authorized_candidates, actor, lists, limit=40
        )
        if strategy == "hybrid_rerank":
            assert self.reranker is not None
            ranked = await self.lexical.rerank_authorized_fused_candidates(
                actor,
                query,
                tuple(item.candidate for item in fused),
                self.reranker,
                limit=limit,
                timeout_seconds=30,
            )
            return SemanticResult(
                strategy,
                tuple(
                    SemanticHit(
                        item.evidence,
                        dense_scores.get(item.evidence.chunk_id),
                        lexical_scores.get(item.evidence.chunk_id),
                        item.candidate.score,
                        item.rerank_score,
                    )
                    for item in ranked
                ),
            )
        return SemanticResult(
            strategy,
            tuple(
                SemanticHit(
                    item.evidence,
                    dense_scores.get(item.evidence.chunk_id),
                    lexical_scores.get(item.evidence.chunk_id),
                    item.candidate.score,
                )
                for item in fused[:limit]
            ),
        )
