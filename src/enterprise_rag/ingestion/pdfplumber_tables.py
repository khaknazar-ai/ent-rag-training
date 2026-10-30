"""Map pdfplumber's public table API to the canonical IR inside parser containment."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from typing import Any

from enterprise_rag.ingestion.ir import (
    PDF_IMAGE_GEOMETRY_PROVIDER,
    ParsedText,
    PDFChartBar,
    PDFImageRegion,
    PDFTableCell,
    TextElement,
)
from enterprise_rag.ingestion.parsers.pdf import (
    MAX_PDF_TABLE_CELL_CHARS,
    MAX_PDF_TABLE_CELLS,
    MAX_PDF_TABLE_CELLS_PER_ROW,
    MAX_PDF_TABLE_ROWS,
    MAX_PDF_TABLE_TEXT_CHARS,
    MAX_PDF_TABLES,
    PDFParseError,
    _normalize_text,
    parser_revision_for,
)
from enterprise_rag.ingestion.pdf_charts import page_chart_bars
from enterprise_rag.ingestion.pdf_images import page_image_regions


def enrich_pdf_tables(parsed: ParsedText, content: bytes, engine: str) -> ParsedText:
    """Use an explicitly selected dependency, never a silent parser fallback."""
    revision = parser_revision_for(engine)
    if engine == "pypdf":
        raise ValueError("pdfplumber enrichment requires a pdfplumber engine")
    try:
        import pdfplumber
    except ImportError as error:
        raise PDFParseError("pdfplumber_dependency_unavailable") from error
    preserve_spans = engine == "pdfplumber-spans"
    strategy = "text" if engine == "pdfplumber-text" else "lines"
    settings = {"vertical_strategy": strategy, "horizontal_strategy": strategy}
    elements: list[TextElement] = []
    warnings = list(parsed.warnings)
    image_regions: list[PDFImageRegion] = []
    chart_bars: list[PDFChartBar] = []
    table_count = cell_count = row_count = text_chars = 0
    line_cursor = 1
    with pdfplumber.open(BytesIO(content)) as document:
        for page_number, page in enumerate(document.pages, 1):
            source_text = "\n".join(
                item.text
                for item in parsed.elements
                if item.page_number == page_number and item.element_type == "page_text"
            )
            chart_bars.extend(page_chart_bars(page, page_number, source_text))
            try:
                page_regions, image_warnings = page_image_regions(
                    page, page_number, len(image_regions)
                )
            except ValueError as error:
                raise PDFParseError(str(error)) from error
            image_regions.extend(page_regions)
            warnings.extend(image_warnings)
            for item in parsed.elements:
                if item.page_number == page_number and item.element_type == "page_text":
                    end_line = line_cursor + item.text.count("\n")
                    elements.append(
                        TextElement.create(
                            parsed.run_id,
                            "page_text",
                            len(elements),
                            item.text,
                            item.original_text,
                            line_cursor,
                            end_line,
                            item.warnings,
                            page_number,
                        )
                    )
                    line_cursor = end_line + 2
            if page.rotation or tuple(page.cropbox or page.mediabox) != tuple(
                page.mediabox
            ):
                warnings.append(
                    f"pdfplumber_table_geometry_unsupported_page_{page_number}"
                )
                continue
            tables = page.find_tables(table_settings=settings)
            for table in tables:
                if preserve_spans:
                    refined = _complete_outer_rules(page, table)
                    if refined is not table:
                        warnings.append(
                            f"pdfplumber_outer_rules_completed_page_{page_number}"
                        )
                    table = refined
                if strategy == "text":
                    table = _complete_text_table_words(page, table)
                    if table is None:
                        warnings.append(
                            f"pdfplumber_text_table_boundary_ambiguous_page_{page_number}"
                        )
                        continue
                table_count += 1
                if (
                    table_count > MAX_PDF_TABLES
                    or len(table.cells) + cell_count > MAX_PDF_TABLE_CELLS
                ):
                    raise PDFParseError("pdf_table_budget_exceeded")
                extracted = table.extract()
                geometry = table.rows
                if len(extracted) != len(geometry):
                    raise PDFParseError("pdf_table_geometry_invalid")
                merged = any(any(box is None for box in row.cells) for row in geometry)
                if merged and not preserve_spans:
                    warnings.append(
                        f"pdfplumber_merged_table_skipped_page_{page_number}"
                    )
                    continue
                if merged:
                    warnings.append(
                        f"pdfplumber_spanning_cells_preserved_page_{page_number}"
                    )
                logical_row = 0
                for physical_row, (values, row) in enumerate(
                    zip(extracted, geometry, strict=True)
                ):
                    if preserve_spans:
                        logical_row = physical_row
                    if not any(value and value.strip() for value in values):
                        continue
                    if (
                        len(values) != len(row.cells)
                        or len(values) > MAX_PDF_TABLE_CELLS_PER_ROW
                    ):
                        raise PDFParseError("pdf_table_budget_exceeded")
                    cells: list[PDFTableCell] = []
                    raw_cells = []
                    for column, (value, box) in enumerate(
                        zip(values, row.cells, strict=True)
                    ):
                        if box is None:
                            if preserve_spans and value is None:
                                continue  # A covered slot is not an empty or repeated source cell.
                            raise PDFParseError("pdf_table_geometry_invalid")
                        raw = value or ""
                        if len(raw) > MAX_PDF_TABLE_CELL_CHARS:
                            raise PDFParseError("pdf_table_budget_exceeded")
                        text = _normalize_text(raw)
                        left, top, right, bottom = box
                        x0, y0, _, _ = page.bbox
                        cells.append(
                            PDFTableCell(
                                table_count,
                                logical_row,
                                column,
                                text,
                                (
                                    (left - x0) / page.width,
                                    (top - y0) / page.height,
                                    (right - x0) / page.width,
                                    (bottom - y0) / page.height,
                                ),
                            )
                        )
                        raw_cells.append(raw)
                    row_text = "\n".join(cell.text for cell in cells)
                    row_count += 1
                    cell_count += len(cells)
                    text_chars += len(row_text)
                    if (
                        row_count > MAX_PDF_TABLE_ROWS
                        or cell_count > MAX_PDF_TABLE_CELLS
                        or text_chars > MAX_PDF_TABLE_TEXT_CHARS
                    ):
                        raise PDFParseError("pdf_table_budget_exceeded")
                    end_line = line_cursor + row_text.count("\n")
                    elements.append(
                        TextElement.create(
                            parsed.run_id,
                            "table_row",
                            len(elements),
                            row_text,
                            "\n".join(raw_cells),
                            line_cursor,
                            end_line,
                            page_number=page_number,
                            table_cells=tuple(cells),
                        )
                    )
                    line_cursor = end_line + 2
                    logical_row += 1
    return replace(
        parsed,
        parser_revision=revision,
        elements=tuple(elements),
        warnings=tuple(warnings),
        pdf_image_regions=tuple(image_regions),
        pdf_image_provider=PDF_IMAGE_GEOMETRY_PROVIDER,
        pdf_chart_bars=tuple(chart_bars),
    )


def _complete_text_table_words(page: Any, table: Any) -> Any:
    """Refine an inferred right edge with whole words using public library APIs."""
    left, top, right, bottom = table.bbox
    if not table.rows or any(box is None for box in table.rows[0].cells):
        return None
    words = [
        word
        for word in page.extract_words()
        if top <= word["top"] and word["bottom"] <= bottom
    ]
    extended = max(
        [right]
        + [word["x1"] for word in words if left <= word["x0"] < right < word["x1"]]
    )
    if any(word["x0"] >= extended and word["x1"] > extended for word in words):
        return None  # Do not silently drop separate words outside the inferred table.
    if extended > page.bbox[2]:
        return None
    if extended > right:
        columns = sorted({box[0] for box in table.rows[0].cells} | {extended})
        refined = page.crop((left, top, extended, bottom)).find_tables(
            {
                "vertical_strategy": "explicit",
                "explicit_vertical_lines": columns,
                "horizontal_strategy": "text",
            }
        )
        return refined[0] if len(refined) == 1 else None
    return table


def _complete_outer_rules(page: Any, table: Any) -> Any:
    """Use repeated source horizontal-rule endpoints, never text guesses.

    pdfplumber's explicit boundaries close missing outer sides while leaving
    internal partial rules and actual spanning-cell boxes intact. Both outer
    horizontal rules and at least one internal rule must agree on the width.
    Ambiguous or disconnected candidates retain their original geometry.
    """
    tolerance = 3.0  # pdfplumber's default snap/intersection tolerance, in PDF points.
    left, top, right, bottom = table.bbox
    edges = [
        edge
        for edge in page.edges
        if edge["orientation"] == "h"
        and top - tolerance <= edge["top"] <= bottom + tolerance
        and edge["x0"] <= left + tolerance
        and edge["x1"] >= right - tolerance
    ]
    if len(edges) > 256:
        raise PDFParseError("pdf_table_budget_exceeded")
    candidates = set()
    for upper in edges:
        if abs(upper["top"] - top) > tolerance:
            continue
        for lower in edges:
            if abs(lower["top"] - bottom) > tolerance:
                continue
            if (
                abs(upper["x0"] - lower["x0"]) > tolerance
                or abs(upper["x1"] - lower["x1"]) > tolerance
            ):
                continue
            matching = [
                edge
                for edge in edges
                if abs(edge["x0"] - upper["x0"]) <= tolerance
                and abs(edge["x1"] - upper["x1"]) <= tolerance
            ]
            if not any(
                top + tolerance < edge["top"] < bottom - tolerance for edge in matching
            ):
                continue
            candidates.add((upper["x0"], upper["x1"]))
    if len(candidates) != 1:
        return table
    outer_left, outer_right = next(iter(candidates))
    if abs(outer_left - left) <= tolerance and abs(outer_right - right) <= tolerance:
        return table
    if not (
        page.bbox[0] <= outer_left <= left and right <= outer_right <= page.bbox[2]
    ):
        return table
    refined = page.crop((outer_left, top, outer_right, bottom)).find_tables(
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [outer_left, outer_right],
        }
    )
    if len(refined) != 1:
        return table
    candidate = refined[0]
    expected = (outer_left, top, outer_right, bottom)
    if any(
        abs(actual - wanted) > tolerance
        for actual, wanted in zip(candidate.bbox, expected, strict=True)
    ):
        return table
    if len(candidate.cells) < len(table.cells):
        return table
    return candidate
