"""Authorized search and evidence use cases for the lite profile."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from math import isfinite
from numbers import Real

from enterprise_rag.ingestion.ir import PDFChartBar, PDFImageRegion
from enterprise_rag.ingestion.ocr import OCRTextRegion
from enterprise_rag.persistence import (
    AuthorizedParentContext,
    LexicalRow,
    SQLiteMetadataStore,
)
from enterprise_rag.retrieval import (
    CandidateIdentity,
    ChildReference,
    DenseHit,
    FusedCandidate,
    ParentExpansionResult,
    ParentRecord,
    RankedCandidate,
    build_literal_fts_query,
    expand_parents,
    reciprocal_rank_fusion,
)
from enterprise_rag.retrieval.query_processing import (
    FALLBACK_POOL,
    MAX_CANDIDATE_CHARS,
    QUERY_REVISION,
    SQL_VM_STEPS,
    LexicalQueryBudgetExceeded,
    plan_question,
)
from enterprise_rag.retrieval.reranking import (
    MAX_RERANK_CANDIDATES,
    RerankCandidate,
    Reranker,
    rerank_candidates,
    validate_rerank_request,
)
from enterprise_rag.security import ActorContext

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    """A search result safe to return after authorization and active checks."""

    chunk_id: str
    document_id: str
    document_version_id: str
    run_id: str
    title: str
    filename: str
    text: str
    score: float
    page_numbers: tuple[int, ...] = ()
    logical_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class PDFTableCellEvidence:
    """One authorized PDF cell with normalized page coordinates."""

    page_number: int
    table_ordinal: int
    row_index: int
    column_index: int
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PDFImageRegionsEvidence:
    chunk_id: str
    document_id: str
    document_version_id: str
    run_id: str
    page_number: int | None
    page_numbers: tuple[int, ...]
    source_pdf_sha256: str
    provider: str | None
    regions: tuple[PDFImageRegion, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    """Inspectable strategy metadata without query text or private candidates."""

    evidence: tuple[SearchEvidence, ...]
    mode: str
    content_units: int
    candidate_count: int
    revision: str = QUERY_REVISION


@dataclass(frozen=True, slots=True)
class ParentContextSearchResult:
    """Authorized child evidence and its bounded source-element contexts."""

    evidence: tuple[SearchEvidence, ...]
    expansion: ParentExpansionResult


@dataclass(frozen=True, slots=True)
class DenseSearchEvidence:
    """One dense candidate after canonical identity and ACL revalidation."""

    candidate: DenseHit
    evidence: SearchEvidence


@dataclass(frozen=True, slots=True)
class FusedSearchEvidence:
    """Fused score/contributions paired with freshly authorized evidence.

    The fused score is on candidate; evidence.score remains SQLite's lexical
    score and must not be presented as the fused score.
    """

    candidate: FusedCandidate
    evidence: SearchEvidence


@dataclass(frozen=True, slots=True)
class RerankedSearchEvidence:
    """Separate fusion/lexical/rerank scores with current canonical evidence."""

    candidate: FusedCandidate
    evidence: SearchEvidence
    rerank_score: float
    original_rank: int


class SearchService:
    """Resolve the actor from DB, then search and hydrate through one ACL path."""

    def __init__(self, metadata: SQLiteMetadataStore) -> None:
        self.metadata = metadata

    def search(
        self,
        actor: ActorContext | None,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[SearchEvidence, ...]:
        return self.search_with_trace(actor, query, limit=limit, offset=offset).evidence

    def search_with_trace(
        self,
        actor: ActorContext | None,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> LexicalSearchResult:
        """Literal-first, then explicit bounded natural-question lexical recovery.

        Same-document fallback retains every content atom; it does not broaden
        unknown terms away or combine unrelated documents. Quotes remain exact
        mandatory phrases in every returned chunk. No provider is involved.
        """
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("Search limit must be1..100 and offset non-negative")
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            return LexicalSearchResult((), "denied", 0, 0)
        plan = plan_question(query)
        if not plan.units:
            return LexicalSearchResult((), "empty", 0, 0)

        def finish(rows, mode, candidates):
            _LOGGER.info(
                "lexical_query revision=%s mode=%s units=%d candidates=%d",
                QUERY_REVISION,
                mode,
                len(plan.units),
                candidates,
            )
            return LexicalSearchResult(
                tuple(self._evidence_from_row(row) for row in rows),
                mode,
                len(plan.units),
                candidates,
            )

        def fetch(expression, take, skip=0):
            try:
                return self.metadata.search_lexical_authorized(
                    scope,
                    expression,
                    limit=take,
                    offset=skip,
                    vm_step_budget=SQL_VM_STEPS,
                    max_text_chars=MAX_CANDIDATE_CHARS,
                )
            except sqlite3.OperationalError as exc:
                if "interrupted" not in str(exc).lower():
                    raise
                _LOGGER.warning(
                    "lexical_query_budget_exceeded revision=%s", QUERY_REVISION
                )
                raise LexicalQueryBudgetExceeded(
                    "Lexical query exceeded its cooperative resource budget"
                ) from exc

        rows = fetch(plan.original_fts, limit, offset)
        if rows or not plan.natural:
            return finish(rows, "literal", len(rows))
        if plan.reduced_fts != plan.original_fts:
            rows = fetch(plan.reduced_fts, limit, offset)
            if rows:
                return finish(rows, "question_terms", len(rows))
        if not plan.fallback_fts:
            return finish((), "no_match", 0)
        if offset + limit > FALLBACK_POOL:
            raise ValueError(
                "Natural-question fallback supports only the first100 candidates"
            )
        pool = fetch(plan.fallback_fts, FALLBACK_POOL)
        masks: dict[str, frozenset[int]] = {}
        coverage: dict[tuple[str, str, str], set[int]] = {}
        for row in pool:
            key = (row.document_id, row.document_version_id, row.run_id)
            matched = plan.matches(row.text)
            masks[row.chunk_id] = matched
            coverage.setdefault(key, set()).update(matched)
        required = set(range(len(plan.units)))
        eligible = [
            row
            for row in pool
            if required
            <= coverage[(row.document_id, row.document_version_id, row.run_id)]
            and plan.preserves_phrases(row.text)
        ]
        eligible.sort(
            key=lambda row: (-len(masks[row.chunk_id]), row.score, row.chunk_id)
        )
        selected = []
        for row in eligible[offset : offset + limit]:
            live = self.get_evidence(actor, row.chunk_id)
            fields = (
                "chunk_id",
                "document_id",
                "document_version_id",
                "run_id",
                "text",
                "page_numbers",
                "title",
                "filename",
            )
            if live is not None and all(
                getattr(live, field) == getattr(row, field) for field in fields
            ):
                selected.append(row)
        return finish(selected, "document_coverage", len(pool))

    def get_evidence(
        self, actor: ActorContext | None, chunk_id: str
    ) -> SearchEvidence | None:
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            return None
        row = self.metadata.get_authorized_evidence(scope, chunk_id)
        return None if row is None else self._evidence_from_row(row)

    def get_pdf_image_regions(
        self,
        actor: ActorContext | None,
        chunk_id: str,
        *,
        page_number: int | None = None,
    ) -> PDFImageRegionsEvidence | None:
        """Expose placements only after authorization and a second live fence."""
        original = self.get_evidence(actor, chunk_id)
        if (
            original is None
            or not original.filename.lower().endswith(".pdf")
            or not original.page_numbers
        ):
            return None
        if page_number is not None and (
            type(page_number) is not int or page_number not in original.page_numbers
        ):
            return None
        stored = self.metadata.get_text_artifacts(original.run_id)
        if stored is None:
            return None
        parsed, chunks = stored
        chunk = next((item for item in chunks if item.id == chunk_id), None)
        if (
            chunk is None
            or chunk.text != original.text
            or chunk.page_numbers != original.page_numbers
        ):
            return None
        pages = original.page_numbers if page_number is None else (page_number,)
        result = PDFImageRegionsEvidence(
            chunk_id,
            original.document_id,
            original.document_version_id,
            original.run_id,
            pages[0] if len(pages) == 1 else None,
            pages,
            parsed.content_sha256,
            parsed.pdf_image_provider,
            tuple(
                item for item in parsed.pdf_image_regions if item.page_number in pages
            ),
            tuple(
                item
                for item in parsed.warnings
                if item.startswith("pdf_image_")
                and any(item.endswith(f"_page_{page}") for page in pages)
            ),
        )
        if self.get_evidence(actor, chunk_id) != original:
            return None
        return result

    def get_pdf_ocr_regions(
        self, actor: ActorContext | None, chunk_id: str
    ) -> tuple[OCRTextRegion, ...] | None:
        """Full-page OCR labels only; never rank a partially retrieved diagram."""
        original = self.get_evidence(actor, chunk_id)
        if (
            original is None
            or len(original.page_numbers) != 1
            or not original.filename.lower().endswith(".pdf")
        ):
            return None
        stored = self.metadata.get_text_artifacts(original.run_id)
        if stored is None:
            return None
        parsed, chunks = stored
        chunk = next((item for item in chunks if item.id == chunk_id), None)
        if (
            chunk is None
            or chunk.text != original.text
            or chunk.page_numbers != original.page_numbers
        ):
            return None
        pages = [
            item
            for item in parsed.ocr_pages
            if item.page_number == original.page_numbers[0]
        ]
        regions: tuple[OCRTextRegion, ...] = ()
        if len(pages) == 1:
            result = pages[0].result
            if (
                result.document_version_id != original.document_version_id
                or result.source_pdf_sha256 != parsed.content_sha256
                or result.run_id != original.run_id
            ):
                return None
            if result.raw_text.strip() == chunk.text.strip():
                regions = result.regions
        if self.get_evidence(actor, chunk_id) != original:
            return None
        return regions

    def get_pdf_chart_bars(
        self, actor: ActorContext | None, chunk_id: str
    ) -> tuple[PDFChartBar, ...] | None:
        """Exact native labels and geometry behind the same live ACL fence."""
        original = self.get_evidence(actor, chunk_id)
        if (
            original is None
            or len(original.page_numbers) != 1
            or not original.filename.lower().endswith(".pdf")
        ):
            return None
        stored = self.metadata.get_text_artifacts(original.run_id)
        if stored is None:
            return None
        parsed, chunks = stored
        chunk = next((item for item in chunks if item.id == chunk_id), None)
        if (
            chunk is None
            or chunk.text != original.text
            or chunk.page_numbers != original.page_numbers
        ):
            return None
        bars = tuple(
            bar
            for bar in parsed.pdf_chart_bars
            if bar.page_number == original.page_numbers[0]
        )
        # Never compare only the bars whose labels happened to survive chunking.
        if any(original.text.count(bar.label) != 1 for bar in bars):
            return ()
        if self.get_evidence(actor, chunk_id) != original:
            return None
        return bars

    def get_pdf_table_header_evidence(
        self, actor: ActorContext | None, chunk_id: str
    ) -> tuple[SearchEvidence, ...]:
        """Hydrate bounded source header chunks from the same authorized table.

        Search hits can contain a child header without its spanning parent.
        Load only row0/1 of the exact persisted page/table, never a text guess
        from another document or run. Every returned chunk is live-authorized.
        """
        original = self.get_evidence(actor, chunk_id)
        if original is None:
            return ()
        cells = self.get_pdf_table_cells(actor, chunk_id)
        if not cells:
            return ()
        tables = {(cell.page_number, cell.table_ordinal) for cell in cells}
        stored = self.metadata.get_text_artifacts(original.run_id)
        if stored is None:
            return ()
        parsed, chunks = stored
        header_ids = {
            element.id
            for element in parsed.elements
            if element.element_type == "table_row"
            and any(
                cell.row_index in (0, 1)
                and (element.page_number, cell.table_ordinal) in tables
                for cell in element.table_cells
            )
        }
        result: list[SearchEvidence] = []
        text_chars = 0
        for chunk in chunks:
            if not header_ids.intersection(chunk.element_ids):
                continue
            evidence = self.get_evidence(actor, chunk.id)
            if (
                evidence is None
                or evidence.run_id != original.run_id
                or evidence.document_version_id != original.document_version_id
            ):
                return ()
            text_chars += len(evidence.text)
            if len(result) >= 16 or text_chars > 16000:
                return ()
            result.append(evidence)
        if self.get_evidence(actor, chunk_id) != original:
            return ()
        return tuple(result)

    def get_pdf_table_cells(
        self, actor: ActorContext | None, chunk_id: str
    ) -> tuple[PDFTableCellEvidence, ...] | None:
        """Return table-cell provenance only while its chunk remains authorized."""

        original = self.get_evidence(actor, chunk_id)
        if original is None:
            return None
        stored = self.metadata.get_text_artifacts(original.run_id)
        if stored is None:
            return None
        parsed, chunks = stored
        chunk = next((item for item in chunks if item.id == chunk_id), None)
        if chunk is None:
            return None
        elements = {element.id: element for element in parsed.elements}
        cells_list: list[PDFTableCellEvidence] = []
        for element_id in chunk.element_ids:
            element = elements.get(element_id)
            if (
                element is None
                or element.element_type != "table_row"
                or element.page_number is None
            ):
                continue
            start = element.text.find(chunk.text)
            if start < 0 or element.text.find(chunk.text, start + 1) >= 0:
                continue
            end = start + len(chunk.text)
            cell_cursor = 0
            for cell in element.table_cells:
                if not cell.text:
                    continue
                cell_start = element.text.find(cell.text, cell_cursor)
                if cell_start < 0:
                    break
                cell_end = cell_start + len(cell.text)
                if start <= cell_start and cell_end <= end:
                    cells_list.append(
                        PDFTableCellEvidence(
                            page_number=element.page_number,
                            table_ordinal=cell.table_ordinal,
                            row_index=cell.row_index,
                            column_index=cell.column_index,
                            text=cell.text,
                            bbox=cell.bbox,
                        )
                    )
                cell_cursor = cell_end
        cells = tuple(cells_list)
        current = self.get_evidence(actor, chunk_id)
        if current is None or any(
            getattr(current, field) != getattr(original, field)
            for field in (
                "chunk_id",
                "document_id",
                "document_version_id",
                "run_id",
                "text",
                "page_numbers",
                "logical_chunk_id",
            )
        ):
            return None
        return cells

    def hydrate_authorized_dense_candidates(
        self,
        actor: ActorContext | None,
        candidates: Sequence[DenseHit],
        *,
        limit: int = 20,
    ) -> tuple[DenseSearchEvidence, ...]:
        """Reauthorize bounded, text-free dense references before hydration.

        Candidate index filters and identity hints are not authorization facts.
        SQLite applies the current tenant/ACL/active-version checks; identity
        hints for visible chunk IDs must then match canonical metadata.
        """

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("Dense evidence limit must be between 1 and 100")
        if not isinstance(candidates, Sequence) or len(candidates) > 100:
            raise ValueError("Dense candidate batch must contain at most 100 hits")
        seen_chunk_ids: set[str] = set()
        seen_ranks: set[int] = set()
        for candidate in candidates:
            if not isinstance(candidate, DenseHit):
                raise ValueError("Dense candidate batch contains an invalid hit")
            for field_name in (
                "chunk_id",
                "document_id",
                "document_version_id",
                "run_id",
            ):
                value = getattr(candidate, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Dense candidate {field_name} must be non-empty")
            if (
                isinstance(candidate.rank, bool)
                or not isinstance(candidate.rank, int)
                or candidate.rank < 1
            ):
                raise ValueError("Dense candidate rank must be a positive integer")
            if isinstance(candidate.score, bool) or not isinstance(
                candidate.score, Real
            ):
                raise ValueError("Dense candidate score must be finite")
            try:
                finite_score = isfinite(float(candidate.score))
            except OverflowError:
                finite_score = False
            if not finite_score:
                raise ValueError("Dense candidate score must be finite")
            if candidate.chunk_id in seen_chunk_ids:
                raise ValueError("Dense candidate batch contains duplicate chunk ids")
            if candidate.rank in seen_ranks:
                raise ValueError("Dense candidate batch contains duplicate ranks")
            seen_chunk_ids.add(candidate.chunk_id)
            seen_ranks.add(candidate.rank)

        ordered_candidates = tuple(sorted(candidates, key=lambda item: item.rank))
        if not ordered_candidates:
            return ()
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            return ()

        rows = self.metadata.get_authorized_evidence_for_candidates(
            scope, tuple(candidate.chunk_id for candidate in ordered_candidates)
        )
        rows_by_chunk_id = {row.chunk_id: row for row in rows}
        hydrated: list[DenseSearchEvidence] = []
        for candidate in ordered_candidates:
            row = rows_by_chunk_id.get(candidate.chunk_id)
            if row is None:
                continue
            if (
                row.document_id != candidate.document_id
                or row.document_version_id != candidate.document_version_id
                or row.run_id != candidate.run_id
            ):
                raise ValueError("Dense candidate identity does not match SQLite")
            hydrated.append(
                DenseSearchEvidence(
                    candidate=candidate, evidence=self._evidence_from_row(row)
                )
            )
        return tuple(hydrated[:limit])

    def hydrate_authorized_fused_candidates(
        self,
        actor: ActorContext | None,
        candidates: Sequence[FusedCandidate],
        *,
        limit: int = 20,
    ) -> tuple[FusedSearchEvidence, ...]:
        """Hydrate a complete fused batch through the live ACL/identity gate.

        Callers must not truncate fused candidates before this boundary;
        use fuse_and_hydrate_authorized_candidates for bounded full fusion.
        """

        if not isinstance(candidates, Sequence) or len(candidates) > 100:
            raise ValueError("Fused candidate batch must contain at most 100 hits")
        dense_references: list[DenseHit] = []
        for rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, FusedCandidate):
                raise ValueError("Fused candidate batch contains an invalid hit")
            if candidate.identity is None:
                raise ValueError("Fused candidate identity is required for hydration")
            if not isinstance(candidate.identity, CandidateIdentity):
                raise ValueError("Fused candidate identity is invalid")
            dense_references.append(
                DenseHit(
                    chunk_id=candidate.candidate_id,
                    document_id=candidate.identity.document_id,
                    document_version_id=candidate.identity.document_version_id,
                    run_id=candidate.identity.run_id,
                    score=candidate.score,
                    rank=rank,
                )
            )
        authorized = self.hydrate_authorized_dense_candidates(
            actor, tuple(dense_references), limit=limit
        )
        return tuple(
            FusedSearchEvidence(
                candidate=candidates[item.candidate.rank - 1], evidence=item.evidence
            )
            for item in authorized
        )

    def fuse_and_hydrate_authorized_candidates(
        self,
        actor: ActorContext | None,
        ranked_lists: Mapping[str, Sequence[RankedCandidate]],
        *,
        k: int = 60,
        limit: int = 20,
    ) -> tuple[FusedSearchEvidence, ...]:
        """Fuse a complete bounded batch, then apply the result limit after ACL."""

        if not isinstance(ranked_lists, Mapping):
            raise ValueError("Ranked lists must be a mapping")
        references: list[RankedCandidate] = []
        for ranked in ranked_lists.values():
            if not isinstance(ranked, Sequence):
                raise ValueError("Ranked strategy input must be a sequence")
            for item in ranked:
                if not isinstance(item, RankedCandidate):
                    raise ValueError("Ranked strategy input contains an invalid hit")
                references.append(item)
                if len(references) > 1000:
                    raise ValueError(
                        "Fusion input exceeds authorized hydration capacity"
                    )
        unique_ids = {item.candidate_id for item in references}
        if len(unique_ids) > 100:
            raise ValueError("Fusion input exceeds authorized hydration capacity")
        fused = reciprocal_rank_fusion(ranked_lists, k=k, limit=max(1, len(unique_ids)))
        return self.hydrate_authorized_fused_candidates(actor, fused, limit=limit)

    async def rerank_authorized_fused_candidates(
        self,
        actor: ActorContext | None,
        query: str,
        candidates: Sequence[FusedCandidate],
        provider: Reranker,
        *,
        limit: int = 10,
        timeout_seconds: float = 5.0,
    ) -> tuple[RerankedSearchEvidence, ...]:
        """Authorize before provider text access, then recheck selected evidence.

        The caller supplies the complete bounded fusion batch, not an already
        truncated top-N. Each SQLite read uses its live authorization snapshot;
        concurrent changes after the final snapshot are not made atomic with
        response delivery. Revoked/changed evidence is omitted without another
        provider request or unscored backfill. No provider is enabled by default.
        """

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_RERANK_CANDIDATES
        ):
            raise ValueError("Rerank evidence limit must be between 1 and 100")
        timeout = validate_rerank_request(
            query, top_n=limit, timeout_seconds=timeout_seconds
        )
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes, bytearray)
        ):
            raise ValueError("Rerank candidate batch must contain at most 100 hits")
        snapshot = tuple(islice(candidates, MAX_RERANK_CANDIDATES + 1))
        if len(snapshot) > MAX_RERANK_CANDIDATES:
            raise ValueError("Rerank candidate batch must contain at most 100 hits")
        authorized = await asyncio.to_thread(
            self.hydrate_authorized_fused_candidates,
            actor,
            snapshot,
            limit=MAX_RERANK_CANDIDATES,
        )
        if not authorized:
            return ()
        ranking = await rerank_candidates(
            query,
            tuple(
                RerankCandidate(item.evidence.chunk_id, item.evidence.text)
                for item in authorized
            ),
            provider,
            top_n=limit,
            timeout_seconds=timeout,
        )
        original = {item.evidence.chunk_id: item for item in authorized}
        selected = tuple(original[item.candidate_id].candidate for item in ranking)
        rechecked = await asyncio.to_thread(
            self.hydrate_authorized_fused_candidates,
            actor,
            selected,
            limit=MAX_RERANK_CANDIDATES,
        )
        current = {item.evidence.chunk_id: item.evidence for item in rechecked}
        positions = {item.candidate_id: rank for rank, item in enumerate(snapshot, 1)}
        result = []
        for item in ranking:
            before = original[item.candidate_id]
            evidence = current.get(item.candidate_id)
            if evidence is None or not self._same_canonical_evidence(
                before.evidence, evidence
            ):
                continue
            result.append(
                RerankedSearchEvidence(
                    candidate=before.candidate,
                    evidence=evidence,
                    rerank_score=item.score,
                    original_rank=positions[item.candidate_id],
                )
            )
        return tuple(result)

    @staticmethod
    def _same_canonical_evidence(left: SearchEvidence, right: SearchEvidence) -> bool:
        # SQLite's score may be query-dependent; it is not evidence identity.
        return (
            left.chunk_id,
            left.document_id,
            left.document_version_id,
            left.run_id,
            left.title,
            left.filename,
            left.text,
            left.page_numbers,
        ) == (
            right.chunk_id,
            right.document_id,
            right.document_version_id,
            right.run_id,
            right.title,
            right.filename,
            right.text,
            right.page_numbers,
        )

    def expand_persisted_parents(self, actor, evidence, *, max_context_chars=12000):
        """Hydrate whole structural ranges after retrieval/reranking, under live ACL."""
        from enterprise_rag.persistence.sqlite_hierarchy import hydrate

        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            raise ValueError("Current parent context actor required")
        with self.metadata._connection() as connection:
            connection.execute("BEGIN")
            return hydrate(
                self.metadata,
                scope,
                evidence,
                connection=connection,
                max_context_chars=max_context_chars,
            )

    def search_with_parent_context(
        self,
        actor: ActorContext | None,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        parent_limit: int = 20,
        max_children_per_parent: int = 8,
        max_context_chars: int = 12_000,
    ) -> ParentContextSearchResult:
        """Search authorized child chunks and expand persisted source elements.

        This lite-profile context is reconstructed from the exact source
        elements referenced by each child chunk; it is not a persisted
        hierarchical-parent implementation.
        """

        self._validate_parent_context_limits(
            parent_limit, max_children_per_parent, max_context_chars
        )
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            return self._empty_parent_context_result()

        rows = self.metadata.search_lexical_authorized_with_parent_context(
            scope,
            build_literal_fts_query(query),
            limit=limit,
            offset=offset,
            max_parent_chars=max_context_chars,
        )
        if not rows:
            return self._empty_parent_context_result()

        evidence = tuple(self._evidence_from_row(row) for row, _parent in rows)
        children: list[ChildReference] = []
        parent_records: dict[str, ParentRecord] = {}
        parent_contexts: dict[str, AuthorizedParentContext] = {}
        unavailable_parent_ids: set[str] = set()

        for rank, (row, context) in enumerate(rows, start=1):
            previous_context = parent_contexts.setdefault(context.parent_id, context)
            if previous_context != context:
                raise ValueError(
                    "Source-element parent context identity/provenance conflict"
                )
            children.append(
                ChildReference(
                    child_id=row.chunk_id,
                    tenant_id=scope.tenant_id,
                    document_id=row.document_id,
                    document_version_id=row.document_version_id,
                    run_id=row.run_id,
                    parent_id=context.parent_id,
                    rank=rank,
                    score=row.score,
                    page_numbers=row.page_numbers,
                )
            )
            if context.oversized:
                if context.text is not None:
                    raise ValueError("Oversized parent context unexpectedly has text")
                unavailable_parent_ids.add(context.parent_id)
                continue
            if context.text is None:
                raise ValueError("Authorized parent context text is missing")
            parent_records[context.parent_id] = ParentRecord(
                parent_id=context.parent_id,
                tenant_id=context.tenant_id,
                document_id=context.document_id,
                document_version_id=context.document_version_id,
                run_id=context.run_id,
                text=context.text,
                page_numbers=context.page_numbers,
            )

        expansion = expand_parents(
            children,
            parent_records,
            tenant_id=scope.tenant_id,
            allowed_document_ids={row.document_id for row, _context in rows},
            active_run_ids={row.run_id for row, _context in rows},
            unavailable_parent_ids=unavailable_parent_ids,
            parent_limit=parent_limit,
            max_children_per_parent=max_children_per_parent,
            max_context_chars=max_context_chars,
        )
        return ParentContextSearchResult(evidence=evidence, expansion=expansion)

    @staticmethod
    def _validate_parent_context_limits(
        parent_limit: int,
        max_children_per_parent: int,
        max_context_chars: int,
    ) -> None:
        for value, label, maximum in (
            (parent_limit, "Parent limit", 100),
            (max_children_per_parent, "Maximum children per parent", 100),
            (max_context_chars, "Maximum context characters", 100_000),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{label} must be between 1 and {maximum}")

    @staticmethod
    def _empty_parent_context_result() -> ParentContextSearchResult:
        return ParentContextSearchResult(
            evidence=(), expansion=ParentExpansionResult((), (), ())
        )

    @staticmethod
    def _evidence_from_row(row: LexicalRow) -> SearchEvidence:
        return SearchEvidence(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            document_version_id=row.document_version_id,
            run_id=row.run_id,
            title=row.title,
            filename=row.filename,
            text=row.text,
            score=row.score,
            page_numbers=row.page_numbers,
            logical_chunk_id=row.logical_chunk_id,
        )
