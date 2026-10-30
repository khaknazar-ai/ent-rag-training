"""Paraphrase and exact quoted support have different typed fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from enterprise_rag.domain.summaries import SUMMARY_BUDGET, SummaryBudget, SummaryScope

if TYPE_CHECKING:
    from enterprise_rag.application.answer import AnswerCitation

MODEL_SUMMARY_REVISION = "model-summary-spanids-v2"
MAX_SUMMARY_TEXT_CHARS = 1600
MAX_SUMMARY_POINT_CHARS = 400


@dataclass(frozen=True, slots=True)
class SummarySupportDraft:
    evidence_id: str
    quote: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class SummaryParagraphDraft:
    text: str
    support: tuple[SummarySupportDraft, ...]


@dataclass(frozen=True, slots=True)
class ModelSummaryDraft:
    status: str
    paragraphs: tuple[SummaryParagraphDraft, ...]


@dataclass(frozen=True, slots=True)
class SummaryTokenizerBinding:
    model_digest: str
    tokenizer_sha256: str
    template_sha256: str
    schema_sha256: str
    counter_revision: str
    special_tokens_revision: str
    qualification_sha256: str
    transport_revision: str
    # API selection and immutable local asset identities are distinct. Defaults
    # preserve historical receipts that used a synthetic digest-shaped API ref.
    model_tag: str | None = None
    manifest_sha256: str | None = None

    @property
    def api_model(self) -> str:
        return self.model_tag or "sha256:" + self.model_digest


@dataclass(frozen=True, slots=True)
class SummaryTokenBudget:
    binding: SummaryTokenizerBinding
    prompt_sha256: str
    input_tokens: int
    output_reserve_tokens: int
    context_tokens: int
    runtime_grant_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSummaryDetails:
    scope: SummaryScope
    input_complete: bool
    budget_status: str
    token_budget: SummaryTokenBudget | None
    secondary_budget: SummaryBudget = SUMMARY_BUDGET
    output_coverage: str = "model_summary_semantic_support_unverified"


@dataclass(frozen=True, slots=True)
class SummaryParagraph:
    text: str
    citation_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModelSummaryResult:
    answer: str | None
    abstained: bool
    abstention_reason: str | None
    citations: tuple[AnswerCitation, ...]
    summary: ModelSummaryDetails
    paragraphs: tuple[SummaryParagraph, ...] = ()
    query_run_id: str | None = None
    kind: str = "model_summary"


def validate_token_budget(value: SummaryTokenBudget) -> None:
    if not isinstance(value, SummaryTokenBudget) or not isinstance(
        value.binding, SummaryTokenizerBinding
    ):
        raise ValueError("Invalid token receipt")
    for number in (
        value.input_tokens,
        value.output_reserve_tokens,
        value.context_tokens,
    ):
        if type(number) is not int or number <= 0:
            raise ValueError("Invalid token count")
    granted = value.runtime_grant_sha256 is not None
    if granted and (
        not isinstance(value.runtime_grant_sha256, str)
        or re.fullmatch("[0-9a-f]{64}", value.runtime_grant_sha256) is None
        or value.context_tokens != 4096
    ):
        raise ValueError("Invalid scoped runtime token receipt")
    if (
        not 1024 <= value.context_tokens <= 4096
        or not 64 <= value.output_reserve_tokens <= (1024 if granted else 256)
        or value.input_tokens + value.output_reserve_tokens > value.context_tokens
    ):
        raise ValueError("Summary exceeds token context")
    binding = value.binding
    if binding.model_tag is not None and (
        not isinstance(binding.model_tag, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", binding.model_tag) is None
        or not isinstance(binding.manifest_sha256, str)
        or re.fullmatch("[0-9a-f]{64}", binding.manifest_sha256) is None
    ):
        raise ValueError("Invalid API model/manifest binding")
    if binding.model_tag is None and binding.manifest_sha256 is not None:
        raise ValueError("Invalid API model/manifest binding")
    for digest in (
        binding.model_digest,
        binding.tokenizer_sha256,
        binding.template_sha256,
        binding.schema_sha256,
        binding.qualification_sha256,
        value.prompt_sha256,
    ):
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid budget identity")
    for revision in (
        binding.counter_revision,
        binding.special_tokens_revision,
        binding.transport_revision,
    ):
        if not isinstance(revision, str) or not 1 <= len(revision) <= 128:
            raise ValueError("Invalid counter revision")


def validate_model_summary(value: ModelSummaryResult) -> None:
    if (
        not isinstance(value, ModelSummaryResult)
        or value.kind != "model_summary"
        or type(value.abstained) is not bool
        or not isinstance(value.summary, ModelSummaryDetails)
    ):
        raise ValueError("Invalid typed model summary")
    details = value.summary
    if (
        details.secondary_budget != SUMMARY_BUDGET
        or type(details.input_complete) is not bool
        or details.output_coverage != "model_summary_semantic_support_unverified"
    ):
        raise ValueError("Invalid model summary metadata")
    if (
        not isinstance(details.budget_status, str)
        or re.fullmatch("[a-z0-9_]{1,128}", details.budget_status) is None
    ):
        raise ValueError("Invalid token budget status")
    if details.token_budget is not None:
        validate_token_budget(details.token_budget)
    if value.abstained:
        if (
            value.answer is not None
            or value.citations
            or value.paragraphs
            or not isinstance(value.abstention_reason, str)
            or re.fullmatch("[a-z0-9_]{1,128}", value.abstention_reason) is None
        ):
            raise ValueError("Invalid model summary abstention")
        return
    if (
        value.abstention_reason is not None
        or not details.input_complete
        or details.scope.oversized
        or details.budget_status != "within_budget"
        or details.token_budget is None
        or not 1 <= len(value.citations) <= 4
        or not 1 <= len(value.paragraphs) <= 4
        or not isinstance(value.answer, str)
        or len(value.answer) > MAX_SUMMARY_TEXT_CHARS
    ):
        raise ValueError("Incomplete model summary")
    used: set[int] = set()
    for paragraph in value.paragraphs:
        if (
            not isinstance(paragraph, SummaryParagraph)
            or not isinstance(paragraph.text, str)
            or not 1 <= len(paragraph.text.strip()) <= MAX_SUMMARY_POINT_CHARS
            or not paragraph.citation_indexes
            or len(set(paragraph.citation_indexes)) != len(paragraph.citation_indexes)
        ):
            raise ValueError("Invalid summary paragraph")
        if any(
            type(index) is not int or not 1 <= index <= len(value.citations)
            for index in paragraph.citation_indexes
        ):
            raise ValueError("Invalid paragraph support")
        used.update(paragraph.citation_indexes)
    if (
        used != set(range(1, len(value.citations) + 1))
        or value.answer != "\n\n".join(p.text for p in value.paragraphs)
        or value.answer == "\n\n".join(c.quote for c in value.citations)
    ):
        raise ValueError("Invalid summary text/support mapping")


def model_details_from_dict(value: dict) -> ModelSummaryDetails:
    budget = value["token_budget"]
    return ModelSummaryDetails(
        SummaryScope(**value["scope"]),
        value["input_complete"],
        value["budget_status"],
        SummaryTokenBudget(
            SummaryTokenizerBinding(**budget["binding"]),
            budget["prompt_sha256"],
            budget["input_tokens"],
            budget["output_reserve_tokens"],
            budget["context_tokens"],
            budget.get("runtime_grant_sha256"),
        )
        if budget is not None
        else None,
        SummaryBudget(**value["secondary_budget"]),
        value["output_coverage"],
    )
