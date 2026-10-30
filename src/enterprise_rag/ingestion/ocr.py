"""Bounded, provenance-aware contract for page-scoped OCR adapters.

This module defines data and validation boundaries only. It does not render
PDFs, run an OCR engine, or make OCR-derived text searchable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

from enterprise_rag.domain.documents.ids import sha256_bytes

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_RASTER_PAGE_BYTES = 16 * 1024 * 1024
MAX_RASTER_PAGE_PIXELS = 20_000_000
MAX_OCR_REGIONS = 10_000
MAX_OCR_REGION_CHARS = 16_384
MAX_OCR_PAGE_CHARS = 1_000_000
MAX_OCR_EVIDENCE_PAGES = 100
MAX_OCR_RUN_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_PROVIDER_ID_CHARS = 128


class OCRContractError(ValueError):
    """An OCR page or provider result violates the bounded contract."""


@dataclass(frozen=True, slots=True)
class RasterPage:
    """One rotated-to-display PNG page with source and renderer identity."""

    run_id: str
    document_version_id: str
    source_pdf_sha256: str
    page_number: int
    image_bytes: bytes
    width: int
    height: int
    applied_rotation_degrees: int
    renderer_id: str
    renderer_revision: str
    image_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run id")
        _validate_identifier(self.document_version_id, "document version id")
        _validate_sha256(self.source_pdf_sha256, "source PDF SHA-256")
        _validate_identifier(self.renderer_id, "renderer id")
        _validate_identifier(self.renderer_revision, "renderer revision")
        if type(self.page_number) is not int or self.page_number < 1:
            raise OCRContractError("Page number must be positive")
        if type(self.width) is not int or type(self.height) is not int:
            raise OCRContractError("Raster dimensions must be integers")
        if self.width < 1 or self.height < 1:
            raise OCRContractError("Raster dimensions must be positive")
        if self.width * self.height > MAX_RASTER_PAGE_PIXELS:
            raise OCRContractError("Raster page exceeds the pixel limit")
        if type(self.applied_rotation_degrees) is not int or (
            self.applied_rotation_degrees not in {0, 90, 180, 270}
        ):
            raise OCRContractError("Applied page rotation must be a right angle")
        if not isinstance(self.image_bytes, bytes):
            raise OCRContractError("Raster page must contain PNG bytes")
        if not 33 <= len(self.image_bytes) <= MAX_RASTER_PAGE_BYTES:
            raise OCRContractError("Raster page exceeds the byte limit")
        if (
            not self.image_bytes.startswith(PNG_SIGNATURE)
            or self.image_bytes[8:12] != b"\x00\x00\x00\r"
            or self.image_bytes[12:16] != b"IHDR"
        ):
            raise OCRContractError("Raster page must be a PNG with an IHDR header")
        png_width = int.from_bytes(self.image_bytes[16:20], "big")
        png_height = int.from_bytes(self.image_bytes[20:24], "big")
        if (png_width, png_height) != (self.width, self.height):
            raise OCRContractError("Raster dimensions do not match the PNG header")
        object.__setattr__(self, "image_sha256", sha256_bytes(self.image_bytes))


@dataclass(frozen=True, slots=True)
class OCRTextRegion:
    """Exact provider span and normalized bounds on the rendered display page.

    Character offsets are Unicode code-point indexes into the untouched page text.
    """

    ordinal: int
    start_char: int
    end_char: int
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise OCRContractError("Region ordinal must be non-negative")
        if (
            type(self.start_char) is not int
            or type(self.end_char) is not int
            or self.start_char < 0
            or self.end_char <= self.start_char
        ):
            raise OCRContractError("Region character span must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise OCRContractError("OCR region text must be non-empty")
        if len(self.text) > MAX_OCR_REGION_CHARS:
            raise OCRContractError("OCR region text exceeds the character limit")
        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            raise OCRContractError("Region bbox must be an (x0, y0, x1, y1) tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.bbox
        ):
            raise OCRContractError("Region bbox coordinates must be numeric")
        x0, y0, x1, y1 = (float(value) for value in self.bbox)
        if not all(isfinite(value) for value in (x0, y0, x1, y1)):
            raise OCRContractError("Region bbox coordinates must be finite")
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise OCRContractError("Region bbox must fit the normalized page")
        object.__setattr__(self, "bbox", (x0, y0, x1, y1))
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise OCRContractError("OCR confidence must be numeric")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise OCRContractError("OCR confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class OCRPageResult:
    """OCR output tied to one page, preserving unnormalized text and spans."""

    run_id: str
    document_version_id: str
    source_pdf_sha256: str
    page_number: int
    image_sha256: str
    provider_id: str
    provider_revision: str
    raw_text: str
    regions: tuple[OCRTextRegion, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run id")
        _validate_identifier(self.document_version_id, "document version id")
        _validate_sha256(self.source_pdf_sha256, "source PDF SHA-256")
        _validate_sha256(self.image_sha256, "page image SHA-256")
        _validate_identifier(self.provider_id, "provider id")
        _validate_identifier(self.provider_revision, "provider revision")
        if (
            len(self.provider_id) > MAX_PROVIDER_ID_CHARS
            or len(self.provider_revision) > MAX_PROVIDER_ID_CHARS
        ):
            raise OCRContractError("OCR provider identity exceeds the length limit")
        if type(self.page_number) is not int or self.page_number < 1:
            raise OCRContractError("Page number must be positive")
        if not isinstance(self.regions, tuple):
            raise OCRContractError("OCR regions must be an immutable tuple")
        if not isinstance(self.raw_text, str):
            raise OCRContractError("OCR page text must be a string")
        if len(self.raw_text) > MAX_OCR_PAGE_CHARS:
            raise OCRContractError("OCR page text exceeds the character limit")
        if len(self.regions) > MAX_OCR_REGIONS:
            raise OCRContractError("OCR page exceeds the region limit")
        if not all(isinstance(region, OCRTextRegion) for region in self.regions):
            raise OCRContractError("OCR result contains an invalid region")
        if self.raw_text.strip() and not self.regions:
            raise OCRContractError("OCR text requires at least one provenance region")
        if tuple(region.ordinal for region in self.regions) != tuple(
            range(len(self.regions))
        ):
            raise OCRContractError("OCR region ordinals must be contiguous")
        spans = sorted(
            (region.start_char, region.end_char, region.text) for region in self.regions
        )
        previous_end = 0
        for start_char, end_char, text in spans:
            if end_char > len(self.raw_text) or start_char < previous_end:
                raise OCRContractError("OCR region spans overlap or exceed page text")
            if self.raw_text[start_char:end_char] != text:
                raise OCRContractError("OCR region text does not match its exact span")
            previous_end = end_char


@dataclass(frozen=True, slots=True)
class OCRPageEvidence:
    """OCR result plus the rendered-page metadata needed to interpret its boxes."""

    page_number: int
    page_image_sha256: str
    page_width: int
    page_height: int
    applied_rotation_degrees: int
    renderer_id: str
    renderer_revision: str
    result: OCRPageResult

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number < 1:
            raise OCRContractError("Evidence page number must be positive")
        _validate_sha256(self.page_image_sha256, "page image SHA-256")
        if (
            type(self.page_width) is not int
            or type(self.page_height) is not int
            or self.page_width < 1
            or self.page_height < 1
            or self.page_width * self.page_height > MAX_RASTER_PAGE_PIXELS
        ):
            raise OCRContractError("Evidence page dimensions are invalid")
        if type(self.applied_rotation_degrees) is not int or (
            self.applied_rotation_degrees not in {0, 90, 180, 270}
        ):
            raise OCRContractError("Evidence page rotation is invalid")
        _validate_identifier(self.renderer_id, "renderer id")
        _validate_identifier(self.renderer_revision, "renderer revision")
        if not isinstance(self.result, OCRPageResult):
            raise OCRContractError("Evidence must contain an OCR page result")
        if (
            self.result.page_number != self.page_number
            or self.result.image_sha256 != self.page_image_sha256
        ):
            raise OCRContractError("OCR result does not match evidence page")

    @classmethod
    def from_page(cls, page: RasterPage, result: OCRPageResult) -> OCRPageEvidence:
        """Bind a result to its exact raster and renderer provenance."""

        validate_ocr_result(page, result)
        return cls(
            page_number=page.page_number,
            page_image_sha256=page.image_sha256,
            page_width=page.width,
            page_height=page.height,
            applied_rotation_degrees=page.applied_rotation_degrees,
            renderer_id=page.renderer_id,
            renderer_revision=page.renderer_revision,
            result=result,
        )


class OCRProvider(Protocol):
    """Adapter contract; implementations must return page-scoped OCR evidence."""

    provider_id: str

    @property
    def revision(self) -> str:
        """Return the provider/model identity used for a recognition result."""

    def recognize_page(self, page: RasterPage) -> OCRPageResult:
        """Recognize one page image without changing its source identity."""


def validate_ocr_result(
    page: RasterPage,
    result: OCRPageResult,
    *,
    low_confidence_threshold: float = 0.5,
) -> tuple[str, ...]:
    """Require exact page binding and return non-destructive quality warnings."""

    if not isinstance(page, RasterPage) or not isinstance(result, OCRPageResult):
        raise OCRContractError("Expected a raster page and an OCR page result")
    if (
        result.run_id != page.run_id
        or result.document_version_id != page.document_version_id
        or result.source_pdf_sha256 != page.source_pdf_sha256
        or result.page_number != page.page_number
        or result.image_sha256 != page.image_sha256
    ):
        raise OCRContractError("OCR result does not match its source page")
    if isinstance(low_confidence_threshold, bool) or not isinstance(
        low_confidence_threshold, (int, float)
    ):
        raise OCRContractError("Confidence threshold must be numeric")
    threshold = float(low_confidence_threshold)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise OCRContractError("Confidence threshold must be between zero and one")
    warnings: list[str] = []
    if not result.raw_text.strip():
        warnings.append("ocr_empty_page")
    elif any(region.confidence < threshold for region in result.regions):
        warnings.append("ocr_low_confidence")
    return tuple(warnings)


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or not value.strip():
        raise OCRContractError(f"Invalid {label}")


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OCRContractError(f"Invalid {label}")
