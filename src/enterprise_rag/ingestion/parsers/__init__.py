"""Parser adapters selected by validated input metadata."""

from enterprise_rag.ingestion.ir import ParsedText

from .pdf import (
    DEFAULT_PDF_PARSER_TIMEOUT_SECONDS,
    PDFParseError,
    parse_pdf,
)
from .text import TextParseError, parse_text


def parse_document(
    run_id: str,
    filename: str,
    content: bytes,
    *,
    max_bytes: int,
    max_pdf_pages: int,
    pdf_timeout_seconds: float = DEFAULT_PDF_PARSER_TIMEOUT_SECONDS,
    allow_ocr_pages: bool = False,
    pdf_table_engine: str = "pypdf",
) -> ParsedText:
    """Route a bounded document to the currently supported native parser."""

    if filename.lower().endswith(".pdf"):
        return parse_pdf(
            run_id,
            filename,
            content,
            max_bytes=max_bytes,
            max_pages=max_pdf_pages,
            timeout_seconds=pdf_timeout_seconds,
            allow_ocr_pages=allow_ocr_pages,
            table_engine=pdf_table_engine,
        )
    return parse_text(run_id, filename, content, max_bytes=max_bytes)


__all__ = [
    "DEFAULT_PDF_PARSER_TIMEOUT_SECONDS",
    "PDFParseError",
    "TextParseError",
    "parse_document",
    "parse_pdf",
    "parse_text",
]
