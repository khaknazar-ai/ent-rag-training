"""Server-assigned exact references; no model arithmetic or quote searching.

The complete canonical input is partitioned, never normalized or shortened.
IDs are namespaced by full source provenance and prompt/counter configuration.
Membership is a mechanical property, not proof of semantic entailment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from enterprise_rag.domain.model_summary import (
    SummaryTokenBudget,
    SummaryTokenizerBinding,
    validate_token_budget,
)
from enterprise_rag.domain.summaries import SummaryScope

SPAN_MAPPING_REVISION = "canonical-codepoint-blocks-v1"


@dataclass(frozen=True, slots=True)
class SummarySpan:
    span_id: str
    evidence_id: str
    quote: str
    start_char: int
    end_char: int


def checked_span_catalog(spans: tuple[SummarySpan, ...]) -> dict[str, SummarySpan]:
    """Do not let Python bool/int equality disguise corrupted prepared bounds."""
    if not isinstance(spans, tuple) or not spans:
        raise ValueError("Invalid summary span catalog")
    for span in spans:
        if (
            type(span) is not SummarySpan
            or not isinstance(span.span_id, str)
            or re.fullmatch(r"sp_[a-f0-9]{64}", span.span_id) is None
            or not isinstance(span.evidence_id, str)
            or not 1 <= len(span.evidence_id) <= 128
            or not isinstance(span.quote, str)
            or not 1 <= len(span.quote) <= 600
            or type(span.start_char) is not int
            or type(span.end_char) is not int
            or not 0 <= span.start_char < span.end_char
            or span.end_char - span.start_char != len(span.quote)
        ):
            raise ValueError("Invalid summary span catalog")
    catalog = {span.span_id: span for span in spans}
    if len(catalog) != len(spans):
        raise ValueError("Ambiguous summary span catalog")
    return catalog


def span_config_sha256(
    binding: SummaryTokenizerBinding,
    context: int,
    reserve: int,
    runtime_grant_sha256: str | None,
) -> str:
    validate_token_budget(
        SummaryTokenBudget(binding, "0" * 64, 1, reserve, context, runtime_grant_sha256)
    )
    return sha256(
        json.dumps(
            {
                "binding": asdict(binding),
                "context": context,
                "reserve": reserve,
                "runtime_grant": runtime_grant_sha256,
                "span_mapping_revision": SPAN_MAPPING_REVISION,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def build_summary_spans(
    scope: SummaryScope, evidence: tuple[tuple[str, str], ...], config_sha256: str
) -> tuple[SummarySpan, ...]:
    if (
        not isinstance(scope, SummaryScope)
        or type(scope.oversized) is not bool
        or scope.oversized
        or not isinstance(config_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", config_sha256) is None
        or not isinstance(scope.source_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", scope.source_sha256) is None
        or any(
            type(n) is not int
            for n in (
                scope.source_chars,
                scope.source_utf8_bytes,
                scope.chunk_count,
                scope.element_count,
            )
        )
        or any(
            not isinstance(x, str) or not 1 <= len(x) <= 128
            for x in (scope.document_id, scope.document_version_id, scope.run_id)
        )
        or not 1 <= len(evidence) <= 32
        or len(evidence) != scope.chunk_count
        or len({item[0] for item in evidence}) != len(evidence)
        or any(
            not isinstance(key, str)
            or not 1 <= len(key) <= 128
            or not isinstance(text, str)
            or not text
            for key, text in evidence
        )
        or not sum(len(text) for _, text in evidence) <= scope.source_chars <= 8000
        or not sum(len(text.encode()) for _, text in evidence)
        <= scope.source_utf8_bytes
        <= 32000
        or not 1 <= scope.element_count <= 128
    ):
        raise ValueError("Invalid complete summary span scope")
    namespace = sha256(
        json.dumps(
            {"scope": asdict(scope), "config": config_sha256}, sort_keys=True
        ).encode()
    ).hexdigest()
    spans = []
    for evidence_id, text in evidence:
        # Fixed canonical codepoint blocks avoid a separate long ID for every
        # line/blank line. This preserves input exactly and the existing600-char
        # quote bound without making fixture codecs or context guards looser.
        for start in range(0, len(text), 600):
            end = min(len(text), start + 600)
            quote = text[start:end]
            digest = sha256(
                json.dumps(
                    [
                        namespace,
                        evidence_id,
                        sha256(text.encode()).hexdigest(),
                        start,
                        end,
                        quote,
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            spans.append(SummarySpan("sp_" + digest, evidence_id, quote, start, end))
    if len({span.span_id for span in spans}) != len(spans):
        raise ValueError("Ambiguous summary span catalog")
    return tuple(spans)
