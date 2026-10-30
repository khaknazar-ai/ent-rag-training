"""Explicit selective OCR routing for pages without a native text layer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard

from enterprise_rag.domain.documents.ids import sha256_bytes
from enterprise_rag.ingestion.ir import ParsedText, TextElement
from enterprise_rag.ingestion.ocr import (
    MAX_OCR_EVIDENCE_PAGES,
    OCRContractError,
    OCRPageEvidence,
    OCRProvider,
    RasterPage,
    validate_ocr_result,
)
from enterprise_rag.ingestion.pdf_render import (
    DEFAULT_PDF_RENDER_MEMORY_BYTES,
    DEFAULT_PDF_RENDER_TIMEOUT_SECONDS,
    _renderer_revision,
    render_pdf_page,
)

_OCR_SELECTED_PAGE = re.compile(r"^ocr_(required|candidate_partial)_page_(\d+)$")


class OCRPipelineError(ValueError):
    """An explicitly configured OCR route cannot preserve its contract."""


class OCRPageRenderer(Protocol):
    """Renderer identity and one-page render boundary required by OCR routing."""

    @property
    def renderer_id(self) -> str:
        """Stable identity for the configured renderer."""

    @property
    def revision(self) -> str:
        """Return a stable identity for the renderer binary and options."""

    def render_page(
        self,
        run_id: str,
        document_version_id: str,
        content: bytes,
        page_number: int,
    ) -> RasterPage:
        """Render one source page with the declared renderer identity."""


@dataclass(frozen=True, slots=True)
class PopplerOCRPageRenderer:
    """Explicit Poppler page renderer for selective ingestion OCR."""

    command: tuple[str, ...]
    temp_root: Path
    renderer_id: str = "poppler"
    timeout_seconds: float = DEFAULT_PDF_RENDER_TIMEOUT_SECONDS
    memory_bytes: int = DEFAULT_PDF_RENDER_MEMORY_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.command, tuple) or not self.command:
            raise ValueError("PDF renderer command must be an explicit argv tuple")
        if not isinstance(self.temp_root, Path) or not self.temp_root.is_absolute():
            raise ValueError("PDF renderer temporary root must be an absolute path")
        if not isinstance(self.renderer_id, str) or not self.renderer_id.strip():
            raise ValueError("PDF renderer id must be non-empty")

    @property
    def revision(self) -> str:
        return _renderer_revision(self.command)

    def render_page(
        self,
        run_id: str,
        document_version_id: str,
        content: bytes,
        page_number: int,
    ) -> RasterPage:
        return render_pdf_page(
            run_id,
            document_version_id,
            content,
            page_number,
            renderer_command=self.command,
            renderer_id=self.renderer_id,
            temp_root=self.temp_root,
            timeout_seconds=self.timeout_seconds,
            memory_bytes=self.memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class PDFOCRPipeline:
    """Render/OCR textless pages and image-backed sparse-text candidates.

    OCR output is kept as exact page text in a separately typed text element
    and as page/region evidence. The original and normalized parser strings
    continue to describe the native PDF text layer. When OCR succeeds for a
    sparse-text candidate, its OCR element becomes the indexed page text; the
    native text layer remains preserved in the original parser fields.
    """

    renderer: OCRPageRenderer
    provider: OCRProvider
    low_confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.low_confidence_threshold, bool) or not isinstance(
            self.low_confidence_threshold, (int, float)
        ):
            raise ValueError("OCR confidence threshold must be numeric")
        if not 0 <= float(self.low_confidence_threshold) <= 1:
            raise ValueError("OCR confidence threshold must be between zero and one")
        self._validate_identity()

    @property
    def execution_configuration(self) -> dict[str, object]:
        """Return the source-free identity pinned to the ingestion run."""

        renderer_id, renderer_revision, provider_id, provider_revision = (
            self._validate_identity()
        )
        return {
            "contract": "pdf-selective-ocr-v2",
            "renderer_id": renderer_id,
            "renderer_revision": renderer_revision,
            "renderer_timeout_seconds": getattr(self.renderer, "timeout_seconds", None),
            "renderer_memory_bytes": getattr(self.renderer, "memory_bytes", None),
            "provider_id": provider_id,
            "provider_revision": provider_revision,
            "provider_timeout_seconds": getattr(self.provider, "timeout_seconds", None),
            "provider_memory_bytes": getattr(self.provider, "memory_bytes", None),
            "max_pages": MAX_OCR_EVIDENCE_PAGES,
            "low_confidence_threshold": float(self.low_confidence_threshold),
        }

    def enrich(
        self,
        parsed: ParsedText,
        document_version_id: str,
        content: bytes,
        *,
        expected_configuration: dict[str, object],
    ) -> ParsedText:
        """Add searchable OCR elements and durable evidence for textless pages."""

        if (
            parsed.ocr_pages
            or not isinstance(content, bytes)
            or sha256_bytes(content) != parsed.content_sha256
            or not isinstance(document_version_id, str)
            or not document_version_id.strip()
        ):
            raise OCRPipelineError("ocr_source_binding_invalid")
        if self.execution_configuration != expected_configuration:
            raise OCRPipelineError("ocr_execution_configuration_changed")

        selected_pages: list[int] = []
        partial_candidates: set[int] = set()
        for warning in parsed.warnings:
            match = _OCR_SELECTED_PAGE.fullmatch(warning)
            if match is None:
                continue
            page_number = int(match.group(2))
            selected_pages.append(page_number)
            if match.group(1) == "candidate_partial":
                partial_candidates.add(page_number)
        page_numbers = tuple(selected_pages)
        if not page_numbers:
            return parsed
        if (
            len(page_numbers) > MAX_OCR_EVIDENCE_PAGES
            or page_numbers != tuple(sorted(set(page_numbers)))
            or any(
                page_number < 1 or page_number > MAX_OCR_EVIDENCE_PAGES
                for page_number in page_numbers
            )
        ):
            raise OCRPipelineError("ocr_page_selection_invalid")

        native_by_page: dict[int, list[TextElement]] = {}
        for element in parsed.elements:
            if element.page_number is None:
                raise OCRPipelineError("ocr_native_page_binding_invalid")
            native_by_page.setdefault(element.page_number, []).append(element)
        native_text_pages = {
            page_number
            for page_number, elements in native_by_page.items()
            if any(element.element_type != "table_row" for element in elements)
        }
        native_overlap = native_text_pages.intersection(page_numbers)
        if native_overlap - partial_candidates or not partial_candidates.issubset(
            native_text_pages
        ):
            raise OCRPipelineError("ocr_page_selection_conflicts_with_native_text")

        ocr_by_page: dict[int, TextElement] = {}
        evidence: list[OCRPageEvidence] = []
        updated_warnings = list(parsed.warnings)
        for page_number in page_numbers:
            try:
                page = self.renderer.render_page(
                    parsed.run_id, document_version_id, content, page_number
                )
                if (
                    not isinstance(page, RasterPage)
                    or page.run_id != parsed.run_id
                    or page.document_version_id != document_version_id
                    or page.source_pdf_sha256 != parsed.content_sha256
                    or page.page_number != page_number
                    or page.renderer_id != expected_configuration["renderer_id"]
                    or page.renderer_revision
                    != expected_configuration["renderer_revision"]
                ):
                    raise OCRPipelineError("ocr_rendered_page_binding_invalid")
                result = self.provider.recognize_page(page)
                item = OCRPageEvidence.from_page(page, result)
                quality_warnings = validate_ocr_result(
                    page,
                    result,
                    low_confidence_threshold=self.low_confidence_threshold,
                )
            except OCRContractError as exc:
                raise OCRPipelineError("ocr_page_processing_failed") from exc
            if (
                result.provider_id != expected_configuration["provider_id"]
                or result.provider_revision
                != expected_configuration["provider_revision"]
            ):
                raise OCRPipelineError("ocr_provider_identity_mismatch")
            evidence.append(item)
            warning_key = (
                f"ocr_candidate_partial_page_{page_number}"
                if page_number in partial_candidates
                else f"ocr_required_page_{page_number}"
            )
            if result.raw_text.strip() and "ocr_low_confidence" not in quality_warnings:
                updated_warnings = [
                    warning for warning in updated_warnings if warning != warning_key
                ]
                element_warnings = tuple(
                    f"{warning}_page_{page_number}" for warning in quality_warnings
                )
                if element_warnings:
                    updated_warnings.extend(element_warnings)
                ocr_by_page[page_number] = TextElement.create(
                    parsed.run_id,
                    "ocr_page_text",
                    0,
                    result.raw_text,
                    result.raw_text,
                    1,
                    max(1, result.raw_text.count("\n") + 1),
                    element_warnings,
                    page_number,
                )
            else:
                if not result.raw_text.strip():
                    updated_warnings.append(f"ocr_empty_page_{page_number}")
                if "ocr_low_confidence" in quality_warnings:
                    updated_warnings.append(f"ocr_low_confidence_page_{page_number}")

        ordered_elements: list[TextElement] = []
        for page_number in sorted(set(native_by_page) | set(ocr_by_page)):
            if page_number in ocr_by_page:
                page_elements = [ocr_by_page[page_number]]
                page_elements.extend(
                    element
                    for element in native_by_page.get(page_number, [])
                    if element.element_type == "table_row"
                )
            else:
                page_elements = native_by_page[page_number]
            for existing in page_elements:
                ordered_elements.append(
                    TextElement.create(
                        parsed.run_id,
                        existing.element_type,
                        len(ordered_elements),
                        existing.text,
                        existing.original_text,
                        existing.start_line,
                        existing.end_line,
                        existing.warnings,
                        page_number,
                        existing.table_cells,
                    )
                )

        if self.execution_configuration != expected_configuration:
            raise OCRPipelineError("ocr_execution_configuration_changed")
        return ParsedText(
            run_id=parsed.run_id,
            parser_revision=parsed.parser_revision,
            content_sha256=parsed.content_sha256,
            original_text=parsed.original_text,
            normalized_text=parsed.normalized_text,
            elements=tuple(ordered_elements),
            warnings=tuple(dict.fromkeys(updated_warnings)),
            ocr_pages=tuple(evidence),
            pdf_image_regions=parsed.pdf_image_regions,
            pdf_image_provider=parsed.pdf_image_provider,
            pdf_chart_bars=parsed.pdf_chart_bars,
        )

    def _validate_identity(self) -> tuple[str, str, str, str]:
        renderer_id, renderer_revision = _renderer_identity(self.renderer)
        provider_id, provider_revision = _provider_identity(self.provider)
        return renderer_id, renderer_revision, provider_id, provider_revision


def _identity_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 128


def _provider_identity(provider: OCRProvider) -> tuple[str, str]:
    provider_id = getattr(provider, "provider_id", None)
    provider_revision = getattr(provider, "revision", None)
    if callable(provider_revision):
        provider_revision = provider_revision()
    if not _identity_text(provider_id) or not _identity_text(provider_revision):
        raise OCRPipelineError("ocr_provider_identity_unavailable")
    return provider_id, provider_revision


def _renderer_identity(renderer: OCRPageRenderer) -> tuple[str, str]:
    renderer_id = getattr(renderer, "renderer_id", None)
    renderer_revision = getattr(renderer, "revision", None)
    if callable(renderer_revision):
        renderer_revision = renderer_revision()
    if not _identity_text(renderer_id) or not _identity_text(renderer_revision):
        raise OCRPipelineError("ocr_renderer_identity_unavailable")
    return renderer_id, renderer_revision
