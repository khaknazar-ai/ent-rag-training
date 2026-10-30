"""Bounded, extractive answers from live authorized lite search evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Literal, Protocol

from enterprise_rag.application.context import (
    MAX_CONTEXT_CHARS_PER_ITEM as MAX_CONTEXT_CHARS_PER_ITEM,
)
from enterprise_rag.application.context import (
    MAX_QUOTE_CHARS,
    ContextWindowError,
    select_context_window,
)
from enterprise_rag.application.query_history import (
    history_request,
    measure_query_stage,
    record_answer,
)
from enterprise_rag.application.quote_matching import locate_quote_span
from enterprise_rag.application.search import SearchEvidence, SearchService
from enterprise_rag.domain.summaries import SummaryDetails
from enterprise_rag.ports.query_history import QueryHistory
from enterprise_rag.security import ActorContext

MAX_CONTEXT_ITEMS = 4


class ModelUnavailable(Exception):
    """The explicitly configured local model could not be reached."""


class ModelProtocolError(Exception):
    """The provider returned an invalid structured response."""


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    evidence_id: str
    text: str
    chunk_start_char: int = 0
    chunk_end_char: int = field(init=False)

    def __post_init__(self) -> None:
        if type(self.chunk_start_char) is not int or self.chunk_start_char < 0:
            raise ValueError("Snippet offset must be a non-negative integer")
        object.__setattr__(
            self, "chunk_end_char", self.chunk_start_char + len(self.text)
        )


@dataclass(frozen=True, slots=True)
class ModelSelection:
    abstain: bool
    evidence_id: str | None
    quote: str | None


class AnswerModel(Protocol):
    def select(
        self, query: str, snippets: tuple[EvidenceSnippet, ...]
    ) -> ModelSelection:
        """Choose one exact supporting span, or abstain."""


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    evidence_id: str
    document_id: str
    document_version_id: str
    run_id: str
    title: str
    filename: str
    page_numbers: tuple[int, ...]
    quote: str
    chunk_start_char: int
    chunk_end_char: int
    offset_unit: Literal["unicode_code_points"] = "unicode_code_points"


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str | None
    abstained: bool
    abstention_reason: str | None
    citations: tuple[AnswerCitation, ...]
    query_run_id: str | None = None
    summary: SummaryDetails | None = None


class AnswerService:
    """Use only ACL-filtered lexical evidence and recheck it after model inference."""

    def __init__(
        self,
        search: SearchService,
        model: AnswerModel,
        *,
        history: QueryHistory | None = None,
        model_configuration: dict | None = None,
    ) -> None:
        self.search = search
        self.model = model
        self.history, self.model_configuration = history, model_configuration

    def answer(
        self, actor: ActorContext, query: str, *, search_query: str | None = None
    ) -> AnswerResult:
        if self.history is not None:
            return record_answer(
                self.history,
                actor,
                history_request(
                    query,
                    query if search_query is None else search_query,
                    "lexical",
                    self.model_configuration,
                ),
                lambda: self._answer(actor, query, search_query=search_query),
            )
        return self._answer(actor, query, search_query=search_query)

    def _answer(self, actor, query, *, search_query=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Question must contain 1-500 characters")
        retrieval_query = query if search_query is None else search_query
        if (
            not isinstance(retrieval_query, str)
            or not retrieval_query.strip()
            or len(retrieval_query) > (500 if search_query is None else 200)
        ):
            raise ValueError("Search query must contain 1-200 characters")
        effective_query = retrieval_query
        with measure_query_stage("retrieval"):
            evidence = self.search.search(
                actor, retrieval_query, limit=MAX_CONTEXT_ITEMS
            )
        try:
            with measure_query_stage("context_packing"):
                snippets = pack_evidence_context(evidence, effective_query)
        except ContextWindowError as exc:
            return _abstain(str(exc))
        if not snippets:
            return _abstain("no_authorized_evidence")

        with measure_query_stage("answer_model"):
            selection = self.model.select(query, snippets)
        with measure_query_stage("citation_validation"):
            return validate_extractive_selection(
                actor, evidence, snippets, selection, self.search.get_evidence
            )


def validate_extractive_selection(actor, evidence, snippets, selection, get_evidence):
    """Validate supplied-context membership and live canonical citations, not entailment."""
    if selection.abstain:
        return _abstain("model_abstained")
    if not isinstance(selection.evidence_id, str) or not isinstance(
        selection.quote, str
    ):
        return _abstain("unsupported_model_output")
    quote = selection.quote.strip()
    snippet = next(
        (item for item in snippets if item.evidence_id == selection.evidence_id),
        None,
    )
    if snippet is None or not quote or len(quote) > MAX_QUOTE_CHARS:
        return _abstain("unsupported_model_output")
    span = locate_quote_span(snippet.text, quote)
    if span is None:
        return _abstain("unsupported_model_output")
    window_start, window_end = span
    quote = snippet.text[window_start:window_end]

    current = get_evidence(actor, snippet.evidence_id)
    original = next(
        (item for item in evidence if item.chunk_id == snippet.evidence_id), None
    )
    if current is None or original is None or not _same_evidence(current, original):
        return _abstain("evidence_no_longer_authorized")
    if len(current.page_numbers) > 1:
        return _abstain("citation_page_ambiguous")
    start_char = snippet.chunk_start_char + window_start
    end_char = snippet.chunk_start_char + window_end
    if current.text[start_char:end_char] != quote:
        return _abstain("evidence_no_longer_authorized")
    return AnswerResult(
        answer=quote,
        abstained=False,
        abstention_reason=None,
        citations=(
            AnswerCitation(
                evidence_id=current.chunk_id,
                document_id=current.document_id,
                document_version_id=current.document_version_id,
                run_id=current.run_id,
                title=current.title,
                filename=current.filename,
                page_numbers=current.page_numbers,
                quote=quote,
                chunk_start_char=start_char,
                chunk_end_char=end_char,
            ),
        ),
    )


def pack_evidence_context(
    evidence: tuple[SearchEvidence, ...], query: str
) -> tuple[EvidenceSnippet, ...]:
    """Pack already-authorized evidence; this helper does not grant access."""
    snippets = []
    seen: set[str] = set()
    for item in islice(evidence, MAX_CONTEXT_ITEMS):
        if not item.text.strip():
            continue
        if item.chunk_id in seen:
            raise ContextWindowError("duplicate_context_evidence")
        seen.add(item.chunk_id)
        window = select_context_window(item.text, query)
        snippets.append(EvidenceSnippet(item.chunk_id, window.text, window.start_char))
    return tuple(snippets)


def _same_evidence(current: SearchEvidence, original: SearchEvidence) -> bool:
    return (
        current.chunk_id == original.chunk_id
        and current.document_id == original.document_id
        and current.document_version_id == original.document_version_id
        and current.run_id == original.run_id
        and current.text == original.text
        and current.page_numbers == original.page_numbers
    )


def _abstain(reason: str) -> AnswerResult:
    return AnswerResult(None, True, reason, ())
