"""Server-owned effective identity of the supported lite text pipeline."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from .chunking import CHUNKING_REVISION
from .hierarchy import HIERARCHY_REVISION
from .parsers.pdf import PARSER_REVISION as PDF_REVISION
from .parsers.pdf import parser_revision_for
from .parsers.text import PARSER_REVISION as TEXT_REVISION


def execution_configuration(
    filename: str,
    *,
    max_text_bytes: int,
    max_pdf_pages: int,
    pdf_parser_timeout_seconds: float,
    max_chunk_chars: int,
    ocr_pipeline: dict[str, object] | None = None,
    pdf_table_engine: str = "pypdf",
) -> dict:
    pdf_version = None
    if filename.lower().endswith(".pdf"):
        try:
            pdf_version = package_version("pypdf")
        except PackageNotFoundError:
            pass  # Actual parsing reports the missing dependency; no fallback.
    configuration: dict[str, object] = {
        "contract": "lite-text-execution-v1",
        "text_parser_revision": TEXT_REVISION,
        "pdf_parser_revision": PDF_REVISION,
        "pdf_provider_version": pdf_version,
        "chunking_revision": CHUNKING_REVISION,
        "hierarchy_revision": HIERARCHY_REVISION,
        "max_text_bytes": max_text_bytes,
        "max_pdf_pages": max_pdf_pages,
        "pdf_parser_timeout_seconds": float(pdf_parser_timeout_seconds),
        "max_chunk_chars": max_chunk_chars,
    }
    if ocr_pipeline is not None:
        configuration["ocr_pipeline"] = ocr_pipeline
    if pdf_table_engine != "pypdf" and filename.lower().endswith(".pdf"):
        configuration["pdf_parser_revision"] = parser_revision_for(pdf_table_engine)
        configuration["pdf_table_engine"] = pdf_table_engine
        try:
            configuration["pdf_table_provider_version"] = package_version("pdfplumber")
        except PackageNotFoundError:
            configuration["pdf_table_provider_version"] = None
    return configuration
