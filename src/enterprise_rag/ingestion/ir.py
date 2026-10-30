"""Immutable canonical text IR used before persistence and indexing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from enterprise_rag.domain.documents.ids import deterministic_id, sha256_bytes
from enterprise_rag.ingestion.ocr import (
    MAX_OCR_EVIDENCE_PAGES,
    OCRPageEvidence,
)

MAX_PDF_TABLE_CELL_CHARS = 16_384
MAX_PDF_TABLE_CELLS = 10_000
MAX_PDF_IMAGE_REGIONS = 2048
PDF_IMAGE_GEOMETRY_PROVIDER = "pdfplumber-image-geometry-v1"
MAX_PDF_CHART_BARS = 3200


@dataclass(frozen=True, slots=True)
class PDFChartBar:
    """Native vertical bar and its exact, single-line source label."""

    page_number: int
    label: str
    bbox: tuple[float, float, float, float]
    label_bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or not 1 <= self.page_number <= 100:
            raise ValueError("Invalid chart page")
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or len(self.label) > 200
        ):
            raise ValueError("Invalid chart label")
        for name in ("bbox", "label_bbox"):
            box = PDFTableCell(0, 0, 0, "", getattr(self, name)).bbox
            object.__setattr__(self, name, box)


def pdf_chart_bars_from_payload(value: object) -> tuple[PDFChartBar, ...]:
    if not isinstance(value, list) or len(value) > MAX_PDF_CHART_BARS:
        raise ValueError("Invalid chart bar payload")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "page_number",
            "label",
            "bbox",
            "label_bbox",
        }:
            raise ValueError("Malformed chart bar")
        for name in ("bbox", "label_bbox"):
            if not isinstance(item[name], (list, tuple)) or len(item[name]) != 4:
                raise ValueError("Malformed chart box")
        result.append(
            PDFChartBar(
                item["page_number"],
                item["label"],
                tuple(item["bbox"]),
                tuple(item["label_bbox"]),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PDFImageRegion:
    """Embedded-image placement, not a semantic figure classification."""

    page_number: int
    image_ordinal: int
    pixel_width: int
    pixel_height: int
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("page_number", self.page_number, 1),
            ("image_ordinal", self.image_ordinal, 0),
            ("pixel_width", self.pixel_width, 1),
            ("pixel_height", self.pixel_height, 1),
        ):
            if type(value) is not int or not minimum <= value <= 1_000_000:
                raise ValueError(f"Invalid PDF image {name}")
        # Reuse the canonical normalized-box validation without changing text.
        validated = PDFTableCell(0, 0, 0, "", self.bbox)
        object.__setattr__(self, "bbox", validated.bbox)


def pdf_image_regions_from_payload(value: object) -> tuple[PDFImageRegion, ...]:
    if not isinstance(value, list) or len(value) > MAX_PDF_IMAGE_REGIONS:
        raise ValueError("PDF image region payload exceeds its contract")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "page_number",
            "image_ordinal",
            "pixel_width",
            "pixel_height",
            "bbox",
        }:
            raise ValueError("Malformed PDF image region")
        bbox = item["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Malformed PDF image region bbox")
        result.append(
            PDFImageRegion(
                item["page_number"],
                item["image_ordinal"],
                item["pixel_width"],
                item["pixel_height"],
                tuple(bbox),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PDFTableCell:
    """One exact extracted table cell tied to a page-normalized bounding box."""

    table_ordinal: int
    row_index: int
    column_index: int
    text: str
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name, value in (
            ("table ordinal", self.table_ordinal),
            ("row index", self.row_index),
            ("column index", self.column_index),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"PDF table {name} must be non-negative")
        if not isinstance(self.text, str) or len(self.text) > MAX_PDF_TABLE_CELL_CHARS:
            raise ValueError("PDF table cell text is invalid or too large")
        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            raise ValueError("PDF table cell bbox must contain four coordinates")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.bbox
        ):
            raise ValueError("PDF table cell bbox coordinates must be numeric")
        x0, y0, x1, y1 = (float(value) for value in self.bbox)
        if not all(isfinite(value) for value in (x0, y0, x1, y1)) or not (
            0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        ):
            raise ValueError("PDF table cell bbox must fit the normalized page")
        object.__setattr__(self, "bbox", (x0, y0, x1, y1))


def pdf_table_cells_from_payload(value: object) -> tuple[PDFTableCell, ...]:
    """Validate untrusted JSON table-cell data before constructing the IR."""

    if not isinstance(value, list) or len(value) > MAX_PDF_TABLE_CELLS:
        raise ValueError("PDF table cell payload is invalid or too large")
    cells: list[PDFTableCell] = []
    required = {"table_ordinal", "row_index", "column_index", "text", "bbox"}
    for item in value:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("PDF table cell payload is malformed")
        bbox = item["bbox"]
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError("PDF table cell bbox is malformed")
        cells.append(
            PDFTableCell(
                item["table_ordinal"],
                item["row_index"],
                item["column_index"],
                item["text"],
                tuple(bbox),
            )
        )
    return tuple(cells)


@dataclass(frozen=True, slots=True)
class TextElement:
    """A structural text unit with source line and optional page provenance."""

    id: str
    run_id: str
    element_type: str
    ordinal: int
    text: str
    original_text: str
    start_line: int
    end_line: int
    warnings: tuple[str, ...] = ()
    page_number: int | None = None
    table_cells: tuple[PDFTableCell, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.table_cells, tuple) or not all(
            isinstance(cell, PDFTableCell) for cell in self.table_cells
        ):
            raise ValueError("Text element table cell provenance is invalid")
        if self.table_cells:
            if self.element_type != "table_row" or self.page_number is None:
                raise ValueError("PDF table cells require a page-scoped table row")
            first = self.table_cells[0]
            if any(
                cell.table_ordinal != first.table_ordinal
                or cell.row_index != first.row_index
                for cell in self.table_cells
            ) or tuple(cell.column_index for cell in self.table_cells) != tuple(
                sorted(set(cell.column_index for cell in self.table_cells))
            ):
                raise ValueError("PDF table row cell coordinates are inconsistent")

    @classmethod
    def create(
        cls,
        run_id: str,
        element_type: str,
        ordinal: int,
        text: str,
        original_text: str,
        start_line: int,
        end_line: int,
        warnings: tuple[str, ...] = (),
        page_number: int | None = None,
        table_cells: tuple[PDFTableCell, ...] = (),
    ) -> TextElement:
        if page_number is not None and page_number <= 0:
            raise ValueError("Page number must be positive")
        id_parts: list[object] = [
            run_id,
            ordinal,
            element_type,
            sha256_bytes(original_text.encode("utf-8")),
        ]
        if page_number is not None:
            id_parts.append(page_number)
        for cell in table_cells:
            id_parts.extend(
                (
                    cell.table_ordinal,
                    cell.row_index,
                    cell.column_index,
                    cell.text,
                    *cell.bbox,
                )
            )
        return cls(
            id=deterministic_id("text-element", *id_parts),
            run_id=run_id,
            element_type=element_type,
            ordinal=ordinal,
            text=text,
            original_text=original_text,
            start_line=start_line,
            end_line=end_line,
            warnings=warnings,
            page_number=page_number,
            table_cells=table_cells,
        )


@dataclass(frozen=True, slots=True)
class ParsedText:
    """Canonical parser output retaining extraction truth and warnings."""

    run_id: str
    parser_revision: str
    content_sha256: str
    original_text: str
    normalized_text: str
    elements: tuple[TextElement, ...]
    warnings: tuple[str, ...] = ()
    ocr_pages: tuple[OCRPageEvidence, ...] = ()
    pdf_image_regions: tuple[PDFImageRegion, ...] = ()
    pdf_image_provider: str | None = None
    pdf_chart_bars: tuple[PDFChartBar, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pdf_chart_bars, tuple)
            or len(self.pdf_chart_bars) > MAX_PDF_CHART_BARS
            or not all(isinstance(bar, PDFChartBar) for bar in self.pdf_chart_bars)
        ):
            raise ValueError("Invalid PDF chart bars")
        chart_keys = [(bar.page_number, bar.bbox[0]) for bar in self.pdf_chart_bars]
        if chart_keys != sorted(set(chart_keys)):
            raise ValueError("Chart bars must be ordered and unique")
        for bar in self.pdf_chart_bars:
            if not any(
                element.page_number == bar.page_number
                and element.element_type == "page_text"
                and element.text.count(bar.label) == 1
                for element in self.elements
            ):
                raise ValueError("Chart label must match exact native page text")
        if self.pdf_image_provider not in {None, PDF_IMAGE_GEOMETRY_PROVIDER}:
            raise ValueError("Unknown PDF image geometry provider")
        if (
            not isinstance(self.pdf_image_regions, tuple)
            or len(self.pdf_image_regions) > MAX_PDF_IMAGE_REGIONS
            or not all(
                isinstance(item, PDFImageRegion) for item in self.pdf_image_regions
            )
            or (self.pdf_image_regions and self.pdf_image_provider is None)
        ):
            raise ValueError("Invalid PDF image regions")
        keys = tuple(
            (item.page_number, item.image_ordinal) for item in self.pdf_image_regions
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("PDF image region identity must be ordered and unique")
        next_ordinal: dict[int, int] = {}
        for region in self.pdf_image_regions:
            expected = next_ordinal.get(region.page_number, 0)
            if region.image_ordinal != expected:
                raise ValueError("PDF image ordinals must be contiguous per page")
            next_ordinal[region.page_number] = expected + 1
        if not isinstance(self.ocr_pages, tuple) or len(self.ocr_pages) > (
            MAX_OCR_EVIDENCE_PAGES
        ):
            raise ValueError("OCR page evidence exceeds the page limit")
        if not all(isinstance(item, OCRPageEvidence) for item in self.ocr_pages):
            raise ValueError("Parsed OCR evidence contains an invalid page")
        page_numbers = tuple(item.page_number for item in self.ocr_pages)
        if page_numbers != tuple(sorted(set(page_numbers))):
            raise ValueError("OCR page evidence must be ordered and unique")
        for item in self.ocr_pages:
            result = item.result
            if (
                result.run_id != self.run_id
                or result.source_pdf_sha256 != self.content_sha256
                or result.page_number != item.page_number
                or result.image_sha256 != item.page_image_sha256
            ):
                raise ValueError("OCR page evidence does not match parsed text")
