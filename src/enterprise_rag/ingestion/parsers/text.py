"""Strict, bounded TXT/Markdown parser for the lite profile."""

from __future__ import annotations

import re
import unicodedata

from enterprise_rag.domain.documents.ids import sha256_bytes
from enterprise_rag.ingestion.ir import ParsedText, TextElement

DEFAULT_MAX_TEXT_BYTES = 25 * 1024 * 1024
PARSER_REVISION = "text-v1"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+[.)]\s+)(.+)$")
_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


class TextParseError(ValueError):
    """An inspectable terminal input/parser policy error."""


def parse_text(
    run_id: str,
    filename: str,
    content: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_TEXT_BYTES,
) -> ParsedText:
    """Parse bounded UTF-8 TXT/MD while retaining original decoded text."""

    if not run_id.strip():
        raise ValueError("Run id must be non-empty")
    if max_bytes <= 0:
        raise ValueError("Maximum text size must be positive")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{suffix}" not in _SUPPORTED_SUFFIXES:
        raise TextParseError("unsupported_text_extension")
    if len(content) > max_bytes:
        raise TextParseError("text_input_too_large")
    try:
        original_text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TextParseError("invalid_utf8") from exc

    normalized_text = unicodedata.normalize(
        "NFC", original_text.replace("\r\n", "\n").replace("\r", "\n")
    )
    elements = _extract_elements(run_id, normalized_text, suffix)
    warnings: list[str] = []
    if not normalized_text.strip():
        warnings.append("empty_document")
    return ParsedText(
        run_id=run_id,
        parser_revision=PARSER_REVISION,
        content_sha256=sha256_bytes(content),
        original_text=original_text,
        normalized_text=normalized_text,
        elements=tuple(elements),
        warnings=tuple(warnings),
    )


def _extract_elements(
    run_id: str, normalized_text: str, suffix: str
) -> list[TextElement]:
    lines = normalized_text.split("\n")
    elements: list[TextElement] = []
    paragraph_lines: list[str] = []
    paragraph_start = 0

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph_lines, paragraph_start
        if not paragraph_lines:
            return
        original = "\n".join(paragraph_lines)
        text = "\n".join(line.strip() for line in paragraph_lines).strip()
        elements.append(
            TextElement.create(
                run_id,
                "paragraph",
                len(elements),
                text,
                original,
                paragraph_start,
                end_line,
            )
        )
        paragraph_lines = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            flush_paragraph(line_number - 1)
            continue
        heading = _HEADING.match(line) if suffix in {"md", "markdown"} else None
        list_item = _LIST_ITEM.match(line) if suffix in {"md", "markdown"} else None
        if heading:
            flush_paragraph(line_number - 1)
            elements.append(
                TextElement.create(
                    run_id,
                    "heading",
                    len(elements),
                    heading.group(2).strip(),
                    line,
                    line_number,
                    line_number,
                )
            )
        elif list_item:
            flush_paragraph(line_number - 1)
            elements.append(
                TextElement.create(
                    run_id,
                    "list_item",
                    len(elements),
                    list_item.group(1).strip(),
                    line,
                    line_number,
                    line_number,
                )
            )
        else:
            if not paragraph_lines:
                paragraph_start = line_number
            paragraph_lines.append(line)
    flush_paragraph(len(lines))
    return elements
