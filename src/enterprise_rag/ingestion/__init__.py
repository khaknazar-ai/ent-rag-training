"""Bounded canonical ingestion primitives for the lite profile."""

from .chunking import TextChunk, chunk_text
from .ir import ParsedText, PDFTableCell, TextElement
from .ocr_pipeline import OCRPipelineError, PDFOCRPipeline, PopplerOCRPageRenderer
from .parsers import (
    DEFAULT_PDF_PARSER_TIMEOUT_SECONDS,
    PDFParseError,
    TextParseError,
    parse_document,
    parse_pdf,
    parse_text,
)

__all__ = [
    "DEFAULT_PDF_PARSER_TIMEOUT_SECONDS",
    "OCRPipelineError",
    "PDFOCRPipeline",
    "PDFParseError",
    "PDFTableCell",
    "ParsedText",
    "PopplerOCRPageRenderer",
    "TextChunk",
    "TextElement",
    "TextParseError",
    "chunk_text",
    "parse_document",
    "parse_pdf",
    "parse_text",
]
