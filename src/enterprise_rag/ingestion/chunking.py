"""Deterministic, Unicode-safe text chunking over canonical elements."""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.domain.documents.ids import deterministic_id, sha256_bytes
from enterprise_rag.ingestion.ir import ParsedText

LEGACY_CHUNKING_REVISION = "canonical-element-slices-v1"
CHUNKING_REVISION = "canonical-section-slices-v2"


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk retaining the source element IDs used to construct it."""

    id: str
    run_id: str
    ordinal: int
    text: str
    element_ids: tuple[str, ...]
    content_sha256: str
    page_numbers: tuple[int, ...] = ()


def chunk_text(
    parsed: ParsedText, *, max_chars: int = 2000, revision: str = CHUNKING_REVISION
) -> tuple[TextChunk, ...]:
    """Split canonical elements without cutting a Unicode code point."""

    if max_chars <= 0:
        raise ValueError("Maximum chunk size must be positive")
    if revision not in {CHUNKING_REVISION, LEGACY_CHUNKING_REVISION}:
        raise ValueError("Unsupported canonical chunking revision")
    chunks: list[TextChunk] = []
    current_text = ""
    current_element_ids: list[str] = []
    current_page_numbers: list[int] = []

    def flush() -> None:
        nonlocal current_text, current_element_ids, current_page_numbers
        if not current_text:
            return
        digest = sha256_bytes(current_text.encode("utf-8"))
        ordinal = len(chunks)
        chunks.append(
            TextChunk(
                id=deterministic_id("text-chunk", parsed.run_id, ordinal, digest),
                run_id=parsed.run_id,
                ordinal=ordinal,
                text=current_text,
                element_ids=tuple(current_element_ids),
                content_sha256=digest,
                page_numbers=tuple(current_page_numbers),
            )
        )
        current_text = ""
        current_element_ids = []
        current_page_numbers = []

    for element in parsed.elements:
        isolated_element = element.element_type == "table_row"
        if isolated_element or (
            revision == CHUNKING_REVISION and element.element_type == "heading"
        ):
            flush()
        remaining = element.text
        while remaining:
            separator = "\n\n" if current_text else ""
            room = max_chars - len(current_text) - len(separator)
            if room <= 0:
                flush()
                continue
            piece, remaining = remaining[:room], remaining[room:]
            current_text += separator + piece
            current_element_ids.append(element.id)
            if (
                element.page_number is not None
                and element.page_number not in current_page_numbers
            ):
                current_page_numbers.append(element.page_number)
            if len(current_text) == max_chars:
                flush()
        if isolated_element:
            flush()
    flush()
    return tuple(chunks)
