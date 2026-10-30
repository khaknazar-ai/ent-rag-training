"""Explicit active document/version highlights with full bounded canonical input."""

from dataclasses import asdict, replace

from enterprise_rag.application.answer import (
    AnswerResult,
    EvidenceSnippet,
    ModelProtocolError,
    ModelSelection,
    validate_extractive_selection,
)
from enterprise_rag.application.query_history import history_request, record_answer
from enterprise_rag.domain.summaries import (
    SUMMARY_BUDGET,
    SUMMARY_REVISION,
    SummaryDetails,
)
from enterprise_rag.ports.document_summary import SummaryProvider, SummarySourceReader
from enterprise_rag.ports.query_history import QueryHistory


class SummaryNotFound(LookupError):
    """Scope is absent, stale or unauthorized; no historical contents are returned."""


class DocumentSummaryService:
    def __init__(
        self,
        source: SummarySourceReader,
        search,
        provider: SummaryProvider,
        history: QueryHistory,
        *,
        provider_configuration: dict,
    ):
        self.source, self.search, self.provider, self.history = (
            source,
            search,
            provider,
            history,
        )
        self.provider_configuration = {
            key: provider_configuration.get(key)
            for key in ("provider", "kind", "revision")
        }
        if any(
            value is not None
            and (not isinstance(value, str) or not 1 <= len(value) <= 128)
            for value in self.provider_configuration.values()
        ):
            raise ValueError("Invalid declared summary provider identity")

    def summarize(self, actor, document_id: str, version_id: str) -> AnswerResult:
        if any(
            not isinstance(value, str) or not value.strip() or len(value) > 128
            for value in (document_id, version_id)
        ):
            raise ValueError("Explicit document and version required")
        source = self.source.read(actor, document_id, version_id)
        if source is None:
            raise SummaryNotFound()
        request = history_request(
            "Summarize " + document_id + " " + version_id,
            version_id,
            "document-summary:extractive",
            {
                **self.provider_configuration,
                "model": None,
                "summary_contract": SUMMARY_REVISION,
                "budget": asdict(SUMMARY_BUDGET),
                "scope_kind": "all_canonical_text",
                "output_coverage": "selected_extracts_not_exhaustive",
            },
        )
        return record_answer(
            self.history,
            actor,
            replace(request, scope=source.scope),
            lambda: self._summarize(actor, source),
        )

    def _summarize(self, actor, source):
        details = SummaryDetails(source.scope, SUMMARY_BUDGET, False)

        def abstain(reason):
            return AnswerResult(None, True, reason, (), summary=details)

        if source.refusal_reason:
            return abstain(source.refusal_reason)
        # Model adapters cannot inherit a character budget as a token guard.
        if self.provider_configuration.get("kind") != "deterministic":
            return abstain("summary_token_budget_unavailable")
        if any(len(item.page_numbers) > 1 for item in source.evidence):
            return abstain("summary_page_ambiguous")
        self._current(actor, source)
        snippets = tuple(
            EvidenceSnippet(item.chunk_id, item.text) for item in source.evidence
        )
        selections = self.provider.summarize(snippets)
        details = replace(details, input_complete=True)
        if (
            not isinstance(selections, tuple)
            or len(selections) > SUMMARY_BUDGET.max_points
        ):
            raise ModelProtocolError("Invalid summary selection count")
        if not selections:
            self._current(actor, source)
            return abstain("summary_no_extractable_points")
        citations = []
        for selection in selections:
            if (
                not isinstance(selection, ModelSelection)
                or type(selection.abstain) is not bool
            ):
                raise ModelProtocolError("Invalid summary selection")
            point = validate_extractive_selection(
                actor, source.evidence, snippets, selection, self.search.get_evidence
            )
            if point.abstained:
                self._current(actor, source)
                return abstain(point.abstention_reason)
            citations.extend(point.citations)
        if len(
            {(c.evidence_id, c.chunk_start_char, c.chunk_end_char) for c in citations}
        ) != len(citations):
            raise ModelProtocolError("Duplicate summary point")
        self._current(actor, source)
        return AnswerResult(
            "\n\n".join(c.quote for c in citations),
            False,
            None,
            tuple(citations),
            summary=details,
        )

    def _current(self, actor, source):
        live = self.source.read(
            actor, source.scope.document_id, source.scope.document_version_id
        )
        if live is None or live != source:
            raise SummaryNotFound()
