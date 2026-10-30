"""Explicit model paraphrase, whole-scope reauthorization and exact support spans."""

import re
from dataclasses import asdict, replace
from hashlib import sha256

from enterprise_rag.application.answer import (
    AnswerCitation,
    EvidenceSnippet,
    ModelProtocolError,
)
from enterprise_rag.application.document_summary import SummaryNotFound
from enterprise_rag.application.query_history import history_request, record_answer
from enterprise_rag.domain.model_summary import (
    MODEL_SUMMARY_REVISION,
    ModelSummaryDetails,
    ModelSummaryDraft,
    ModelSummaryResult,
    SummaryParagraph,
    SummaryParagraphDraft,
    SummarySupportDraft,
    validate_model_summary,
    validate_token_budget,
)
from enterprise_rag.domain.summaries import SUMMARY_BUDGET
from enterprise_rag.domain.summary_spans import (
    build_summary_spans,
    checked_span_catalog,
    span_config_sha256,
)
from enterprise_rag.ports.document_summary import SummarySourceReader
from enterprise_rag.ports.model_summary import (
    ModelSummaryProvider,
    PreparedModelSummary,
    SummaryBudgetRefused,
)
from enterprise_rag.ports.query_history import QueryHistory


class ModelDocumentSummaryService:
    def __init__(
        self,
        source: SummarySourceReader,
        provider: ModelSummaryProvider,
        history: QueryHistory,
    ):
        self.source, self.provider, self.history = source, provider, history

    def summarize(self, actor, document_id: str, version_id: str) -> ModelSummaryResult:
        if any(
            not isinstance(v, str) or not v.strip() or len(v) > 128
            for v in (document_id, version_id)
        ):
            raise ValueError("Explicit document/version required")
        source = self.source.read(actor, document_id, version_id)
        if source is None:
            raise SummaryNotFound()
        declared_identity = self.provider.identity
        identity = {
            key: declared_identity.get(key)
            for key in (
                "provider",
                "model",
                "digest",
                "summary_contract",
                "transport_revision",
            )
        }
        if any(
            v is not None and (not isinstance(v, str) or not 1 <= len(v) <= 128)
            for v in identity.values()
        ):
            raise ValueError("Invalid model summary identity")
        reason = source.refusal_reason
        prepared = None
        if reason is None and any(len(e.page_numbers) > 1 for e in source.evidence):
            reason = "summary_page_ambiguous"
        if reason is None:
            try:
                snippets = tuple(
                    EvidenceSnippet(e.chunk_id, e.text) for e in source.evidence
                )
                prepared = self.provider.prepare(snippets, scope=source.scope)
                if not isinstance(prepared, PreparedModelSummary):
                    raise ModelProtocolError("Invalid prepared summary")
                validate_token_budget(prepared.token_budget)
                binding = prepared.token_budget.binding
                if prepared.scope is None:
                    raise ValueError("Missing prepared summary scope")
                checked_span_catalog(prepared.spans)
                config = span_config_sha256(
                    binding,
                    prepared.token_budget.context_tokens,
                    prepared.token_budget.output_reserve_tokens,
                    prepared.token_budget.runtime_grant_sha256,
                )
                # Validate the prepared scope's types too: bool must not alias
                # integer metadata during dataclass equality.
                build_summary_spans(
                    prepared.scope,
                    tuple((s.evidence_id, s.text) for s in snippets),
                    config,
                )
                expected_spans = build_summary_spans(
                    source.scope,
                    tuple((s.evidence_id, s.text) for s in snippets),
                    config,
                )
                if (
                    not isinstance(prepared.prompt, str)
                    or not isinstance(prepared.schema_json, str)
                    or prepared.token_budget.prompt_sha256
                    != sha256(prepared.prompt.encode()).hexdigest()
                    or binding.schema_sha256
                    != sha256(prepared.schema_json.encode()).hexdigest()
                    or prepared.scope != source.scope
                    or prepared.evidence != snippets
                    or prepared.spans != expected_spans
                    or identity["digest"] != binding.model_digest
                    or identity["model"] != binding.api_model
                    or identity["transport_revision"] != binding.transport_revision
                    or not isinstance(prepared.request_sha256, str)
                    or re.fullmatch("[0-9a-f]{64}", prepared.request_sha256) is None
                ):
                    raise ValueError(
                        "Prepared binding does not match configured model/prompt"
                    )
            except SummaryBudgetRefused as error:
                reason = error.reason
                prepared = None
            except (ValueError, TypeError, ModelProtocolError):
                reason = "summary_prepared_binding_invalid"
                prepared = None
        details = ModelSummaryDetails(
            source.scope,
            False,
            reason or "within_budget",
            prepared.token_budget if prepared is not None else None,
        )
        request = history_request(
            "Model summary " + document_id + " " + version_id,
            version_id,
            "document-summary:model",
            {
                **identity,
                "model_summary_contract": MODEL_SUMMARY_REVISION,
                "secondary_budget": asdict(SUMMARY_BUDGET),
                "token_budget": asdict(details.token_budget)
                if details.token_budget
                else None,
                "budget_status": details.budget_status,
                "request_sha256": prepared.request_sha256 if prepared else None,
                "semantic_support": "unverified",
            },
        )
        return record_answer(
            self.history,
            actor,
            replace(request, scope=source.scope),
            lambda: self._generate(actor, source, prepared, details, reason),
        )

    def _current(self, actor, source):
        current = self.source.read(
            actor, source.scope.document_id, source.scope.document_version_id
        )
        if current is None or current != source:
            raise SummaryNotFound()

    def _generate(self, actor, source, prepared, details, reason):
        def refuse(code):
            value = ModelSummaryResult(None, True, code, (), details)
            validate_model_summary(value)
            return value

        if reason is not None:
            return refuse(reason)
        self._current(actor, source)
        try:
            draft = self.provider.generate(prepared)
        except SummaryBudgetRefused:
            raise ModelProtocolError(
                "Model summary token gate changed before transport"
            ) from None
        self._current(actor, source)
        details = replace(details, input_complete=True)
        if (
            not isinstance(draft, ModelSummaryDraft)
            or not isinstance(draft.status, str)
            or draft.status not in {"summary", "insufficient", "conflict"}
            or not isinstance(draft.paragraphs, tuple)
        ):
            raise ModelProtocolError("Invalid summary disposition")
        if draft.status != "summary":
            if draft.paragraphs:
                raise ModelProtocolError("Summary refusal contains content")
            return refuse(
                "conflicting_evidence"
                if draft.status == "conflict"
                else "insufficient_support"
            )
        if not 1 <= len(draft.paragraphs) <= 4:
            raise ModelProtocolError("Invalid summary paragraph count")
        evidence = {e.chunk_id: e for e in source.evidence}
        allowed_spans = {
            (s.evidence_id, s.quote, s.start_char, s.end_char) for s in prepared.spans
        }
        citations = []
        paragraphs = []
        for paragraph in draft.paragraphs:
            if (
                not isinstance(paragraph, SummaryParagraphDraft)
                or not isinstance(paragraph.text, str)
                or not 1 <= len(paragraph.text.strip()) <= 400
                or not isinstance(paragraph.support, tuple)
                or not 1 <= len(paragraph.support) <= 4
            ):
                raise ModelProtocolError("Invalid summary paragraph")
            indexes = []
            for span in paragraph.support:
                if (
                    not isinstance(span, SummarySupportDraft)
                    or not isinstance(span.evidence_id, str)
                    or not isinstance(span.quote, str)
                    or not 1 <= len(span.quote) <= 600
                    or not span.quote.strip()
                    or type(span.start_char) is not int
                    or type(span.end_char) is not int
                ):
                    raise ModelProtocolError("Invalid canonical summary span")
                item = evidence.get(span.evidence_id)
                if (
                    item is None
                    or (span.evidence_id, span.quote, span.start_char, span.end_char)
                    not in allowed_spans
                    or not 0 <= span.start_char < span.end_char <= len(item.text)
                    or item.text[span.start_char : span.end_char] != span.quote
                    or len(item.page_numbers) > 1
                ):
                    raise ModelProtocolError(
                        "Summary support is not canonical membership"
                    )
                citation = AnswerCitation(
                    item.chunk_id,
                    item.document_id,
                    item.document_version_id,
                    item.run_id,
                    item.title,
                    item.filename,
                    item.page_numbers,
                    span.quote,
                    span.start_char,
                    span.end_char,
                )
                if citation not in citations:
                    citations.append(citation)
                indexes.append(citations.index(citation) + 1)
            paragraphs.append(SummaryParagraph(paragraph.text, tuple(indexes)))
        value = ModelSummaryResult(
            "\n\n".join(p.text for p in paragraphs),
            False,
            None,
            tuple(citations),
            details,
            tuple(paragraphs),
        )
        try:
            validate_model_summary(value)
        except ValueError:
            raise ModelProtocolError(
                "Invalid model summary result/support mapping"
            ) from None
        self._current(actor, source)
        return value
