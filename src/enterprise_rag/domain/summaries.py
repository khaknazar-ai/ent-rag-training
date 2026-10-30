"""Bounded canonical-text summary scope; no claim about omitted visual content."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SummaryScope:
    document_id: str
    document_version_id: str
    run_id: str
    source_sha256: str
    source_chars: int
    source_utf8_bytes: int
    chunk_count: int
    element_count: int
    oversized: bool


@dataclass(frozen=True, slots=True)
class SummaryBudget:
    max_input_chars: int = 8000
    max_input_utf8_bytes: int = 32000
    max_chunks: int = 32
    max_elements: int = 128
    max_points: int = 4
    max_quote_chars: int = 600
    # The shipped deterministic extractor uses no model or token context.
    tokens_applicable: bool = False
    input_tokens: int | None = None
    token_limit: int | None = None


SUMMARY_BUDGET = SummaryBudget()
SUMMARY_REVISION = "bounded-extractive-summary-v1"


@dataclass(frozen=True, slots=True)
class SummaryDetails:
    scope: SummaryScope
    budget: SummaryBudget
    input_complete: bool
    output_coverage: str = "selected_extracts_not_exhaustive"


def summary_details_from_dict(value: dict) -> SummaryDetails:
    return SummaryDetails(
        SummaryScope(**value["scope"]),
        SummaryBudget(**value["budget"]),
        value["input_complete"],
        value["output_coverage"],
    )
