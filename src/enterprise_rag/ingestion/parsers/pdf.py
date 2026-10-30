"""Bounded native PDF text extraction for the lite profile."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import unicodedata
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from itertools import pairwise
from math import isfinite
from multiprocessing.connection import Connection
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Literal

from enterprise_rag.domain.documents.ids import sha256_bytes
from enterprise_rag.ingestion.errors import ParserContainmentError
from enterprise_rag.ingestion.ir import (
    MAX_PDF_TABLE_CELL_CHARS,
    MAX_PDF_TABLE_CELLS,
    PDF_IMAGE_GEOMETRY_PROVIDER,
    ParsedText,
    PDFTableCell,
    TextElement,
    pdf_chart_bars_from_payload,
    pdf_image_regions_from_payload,
    pdf_table_cells_from_payload,
)

DEFAULT_MAX_PDF_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_PDF_PAGES = 100
DEFAULT_MAX_PAGE_TEXT_CHARS = 5_000_000
DEFAULT_PDF_PARSER_TIMEOUT_SECONDS = 10.0
DEFAULT_PDF_MEMORY_BYTES = 1024**3
MAX_RESULT_BYTES = 8 * 1024 * 1024
CLEANUP_SECONDS = 2.0
MAX_NATIVE_TEXT_CHARS_FOR_PARTIAL_SCAN_CANDIDATE = 120
MAX_PDF_TABLES = 500
MAX_PDF_TABLE_ROWS = 5_000
MAX_PDF_TABLE_TEXT_CHARS = 1_000_000
MAX_PDF_TABLE_CELLS_PER_ROW = 256
MAX_PDF_LAYOUT_SEGMENTS_PER_PAGE = 20_000
MAX_PDF_LAYOUT_SPANS_PER_PAGE = 100_000
MAX_PDF_LAYOUT_COORDINATES_PER_AXIS = 256
PDF_LAYOUT_COORDINATE_TOLERANCE = 1.5
PARSER_REVISION = "pdf-pypdf-ruled-table-cells-v4"
PDF_TABLE_ENGINES = ("pypdf", "pdfplumber-lines", "pdfplumber-text", "pdfplumber-spans")


def parser_revision_for(table_engine: str) -> str:
    if table_engine not in PDF_TABLE_ENGINES:
        raise ValueError("Unsupported PDF table engine")
    if table_engine == "pdfplumber-spans":
        return "pdf-pypdf-pdfplumber-spans-v1-source-cells"
    return (
        PARSER_REVISION
        if table_engine == "pypdf"
        else f"pdf-pypdf-{table_engine}-v3-charts"
    )


class PDFParseError(ValueError):
    """An inspectable terminal PDF input/parser policy error."""


def parse_pdf(
    run_id: str,
    filename: str,
    content: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_page_text_chars: int = DEFAULT_MAX_PAGE_TEXT_CHARS,
    timeout_seconds: float = DEFAULT_PDF_PARSER_TIMEOUT_SECONDS,
    memory_bytes: int = DEFAULT_PDF_MEMORY_BYTES,
    allow_ocr_pages: bool = False,
    table_engine: str = "pypdf",
) -> ParsedText:
    """Extract native text in a bounded child process."""

    _validate_pdf_input(
        run_id,
        filename,
        content,
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_page_text_chars=max_page_text_chars,
    )
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("PDF parser timeout must be finite and positive")
    if (
        type(memory_bytes) is not int
        or not 64 * 1024 * 1024 <= memory_bytes <= 4 * 1024**3
    ):
        raise ValueError("PDF memory limit must be 64 MiB..4 GiB")
    if type(allow_ocr_pages) is not bool:
        raise ValueError("OCR page routing flag must be boolean")
    expected_revision = parser_revision_for(table_engine)

    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    gate_reader, gate_writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_pdf_parser_child,
        args=(
            sender,
            run_id,
            filename,
            content,
            max_bytes,
            max_pages,
            max_page_text_chars,
            gate_reader,
            memory_bytes,
            allow_ocr_pages,
            table_engine,
        ),
    )
    process.daemon = True
    job = None
    started = False
    reader: Thread | None = None
    received: Queue[tuple[bool, object]] = Queue(maxsize=1)
    try:
        # Only this child is assigned; no ACL, host policy, or global limits change.
        if os.name == "nt":
            try:
                from enterprise_rag.workers.processes import _WindowsJob

                job = _WindowsJob(memory_bytes=memory_bytes)
            except OSError as exc:
                raise PDFParseError("pdf_resource_limit_unavailable") from exc
        process.start()
        started = True
        sender.close()
        gate_reader.close()
        try:
            if job is not None:
                job.assign_handle(process.sentinel)
            gate_writer.send_bytes(b"1")
        except OSError as exc:
            raise PDFParseError("pdf_resource_limit_unavailable") from exc
        finally:
            gate_writer.close()
        deadline = monotonic() + timeout_seconds
        # Drain while the child runs: a large send cannot finish until it is read.
        # A separate receiver keeps partial pipe frames inside the same deadline.
        reader = Thread(
            target=_receive_pdf_outcome,
            args=(receiver, received, deadline),
            daemon=True,
        )
        reader.start()
        try:
            valid, outcome = received.get(timeout=max(0.0, deadline - monotonic()))
        except Empty as exc:
            raise PDFParseError("pdf_parser_timeout") from exc
        process.join(max(0.0, deadline - monotonic()))
        if process.is_alive():
            raise PDFParseError("pdf_parser_timeout")
        if not valid:
            raise PDFParseError("pdf_parser_failed")
        if not isinstance(outcome, tuple) or len(outcome) != 2:
            raise PDFParseError("pdf_parser_failed")
        status, value = outcome
        if status == "ok":
            return _decode_parsed(
                value,
                run_id,
                sha256_bytes(content),
                allow_empty_elements=allow_ocr_pages,
                expected_revision=expected_revision,
            )
        if status == "error" and isinstance(value, str) and value:
            raise PDFParseError(value)
        raise PDFParseError("pdf_parser_failed")
    except (EOFError, OSError) as exc:
        raise PDFParseError("pdf_parser_failed") from exc
    finally:
        sender.close()
        gate_reader.close()
        gate_writer.close()
        cleanup_deadline = monotonic() + CLEANUP_SECONDS
        confirmed = True
        if job is not None:
            try:
                job.close(timeout_seconds=max(0.0, cleanup_deadline - monotonic()))
            except OSError:
                confirmed = False
        if started:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(max(0.0, cleanup_deadline - monotonic()))
                if process.is_alive():
                    process.kill()
                    process.join(max(0.0, cleanup_deadline - monotonic()))
                if process.is_alive():
                    confirmed = False
                else:
                    process.close()
            except (OSError, ValueError):
                confirmed = False
        else:
            process.close()
        # Reap the sole writer before waiting for a partial frame to finish.
        if reader is not None:
            reader.join(timeout=max(0.0, cleanup_deadline - monotonic()))
        receiver.close()
        if not confirmed or (reader is not None and reader.is_alive()):
            raise ParserContainmentError("pdf_parser_cleanup_unconfirmed")


def _receive_pdf_outcome(
    receiver: Connection, received: Queue[tuple[bool, object]], deadline: float
) -> None:
    try:
        if not receiver.poll(max(0.0, deadline - monotonic())):
            received.put((False, None))
            return
        try:
            raw = receiver.recv_bytes(maxlength=MAX_RESULT_BYTES)
        except OSError:
            received.put((True, ("error", "pdf_result_unavailable_or_too_large")))
            return
        outcome = json.loads(raw)
        if isinstance(outcome, list):
            outcome = tuple(outcome)
    except Exception:
        received.put((False, None))
    else:
        received.put((True, outcome))


def _pdf_parser_child(
    sender: Connection,
    run_id: str,
    filename: str,
    content: bytes,
    max_bytes: int,
    max_pages: int,
    max_page_text_chars: int,
    gate: Connection,
    memory_bytes: int,
    allow_ocr_pages: bool,
    table_engine: str = "pypdf",
) -> None:
    """Run pypdf outside the worker process and return sanitized outcomes."""

    outcome: tuple[Literal["ok"], ParsedText] | tuple[Literal["error"], str]
    try:
        # Spawn bootstrap/input transfer happen before assignment; parsing does not.
        if not gate.poll(30) or gate.recv_bytes(maxlength=1) != b"1":
            return
        gate.close()
        if os.name != "nt":
            _set_posix_memory_limit(memory_bytes)
        result = _parse_pdf_native(
            run_id,
            filename,
            content,
            max_bytes=max_bytes,
            max_pages=max_pages,
            max_page_text_chars=max_page_text_chars,
            allow_ocr_pages=allow_ocr_pages,
        )
        if table_engine != "pypdf":
            from enterprise_rag.ingestion.pdfplumber_tables import enrich_pdf_tables

            result = enrich_pdf_tables(result, content, table_engine)
        outcome = ("ok", result)
    except MemoryError:
        outcome = ("error", "pdf_memory_limit")
    except PDFParseError as exc:
        outcome = ("error", str(exc))
    except Exception:
        outcome = ("error", "pdf_corrupt")
    try:
        serializable = ("ok", asdict(outcome[1])) if outcome[0] == "ok" else outcome
        raw = json.dumps(serializable, ensure_ascii=False).encode("utf-8")
        if len(raw) > MAX_RESULT_BYTES:
            raw = b'["error","pdf_result_too_large"]'
        sender.send_bytes(raw)
    except (BrokenPipeError, OSError):
        pass
    finally:
        sender.close()


def _parse_pdf_native(
    run_id: str,
    filename: str,
    content: bytes,
    *,
    max_bytes: int,
    max_pages: int,
    max_page_text_chars: int,
    allow_ocr_pages: bool = False,
) -> ParsedText:
    """Parse inside the isolated child process."""

    _validate_pdf_input(
        run_id,
        filename,
        content,
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_page_text_chars=max_page_text_chars,
    )
    if not run_id.strip():
        raise ValueError("Run id must be non-empty")

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise PDFParseError("pdf_dependency_unavailable") from exc
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise PDFParseError("pdf_encrypted")
        page_count = len(reader.pages)
        if page_count == 0:
            raise PDFParseError("pdf_empty")
        if page_count > max_pages:
            raise PDFParseError("pdf_too_many_pages")

        raw_pages: list[str] = []
        normalized_pages: list[str] = []
        elements: list[TextElement] = []
        warnings: list[str] = []
        line_cursor = 1
        table_ordinal = 0
        table_row_count = 0
        table_cell_count = 0
        table_text_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text, text_spans, line_segments, layout_bounded = (
                _extract_pdf_page_layout(page)
            )
            if not isinstance(raw_text, str):
                raise PDFParseError("pdf_corrupt")
            if len(raw_text) > max_page_text_chars:
                raise PDFParseError("pdf_page_text_too_large")
            normalized = _normalize_text(raw_text)
            raw_pages.append(raw_text)
            normalized_pages.append(normalized)
            if not normalized.strip():
                warnings.append(f"ocr_required_page_{page_number}")
            elif (
                len(normalized.strip())
                <= MAX_NATIVE_TEXT_CHARS_FOR_PARTIAL_SCAN_CANDIDATE
                and _page_has_embedded_image(page)
            ):
                warnings.append(f"ocr_candidate_partial_page_{page_number}")
            else:
                elements.append(
                    TextElement.create(
                        run_id,
                        "page_text",
                        len(elements),
                        normalized.strip(),
                        raw_text,
                        line_cursor,
                        line_cursor + normalized.count("\n"),
                        page_number=page_number,
                    )
                )
            line_cursor += max(1, normalized.count("\n") + 1) + 2
            bounds = _page_box_bounds(page)
            (
                table_rows,
                table_ordinal,
                table_row_count,
                table_cell_count,
                table_text_chars,
            ) = _extract_table_rows(
                text_spans,
                line_segments,
                bounds,
                int(getattr(page, "rotation", 0) or 0),
                layout_bounded,
                run_id,
                page_number,
                len(elements),
                line_cursor,
                table_ordinal,
                table_row_count,
                table_cell_count,
                table_text_chars,
            )
            elements.extend(table_rows)
            if table_rows:
                line_cursor = table_rows[-1].end_line + 2

        if not elements and not allow_ocr_pages:
            raise PDFParseError("ocr_required")
        return ParsedText(
            run_id=run_id,
            parser_revision=PARSER_REVISION,
            content_sha256=sha256_bytes(content),
            original_text="\n\n".join(raw_pages),
            normalized_text="\n\n".join(normalized_pages),
            elements=tuple(elements),
            warnings=tuple(warnings),
        )
    except (PDFParseError, MemoryError):
        raise
    except Exception as exc:
        raise PDFParseError("pdf_corrupt") from exc


def _page_has_embedded_image(page: object) -> bool:
    """Detect page XObject images without decoding their pixel streams."""

    get_inherited = getattr(page, "get_inherited", None)
    if not callable(get_inherited):
        return False
    resources = _resolve_pdf_object(get_inherited("/Resources"))
    if not isinstance(resources, dict):
        return False

    pending = [resources]
    visited_resources: set[int] = set()
    visited_xobjects = 0
    while pending:
        current = _resolve_pdf_object(pending.pop())
        if not isinstance(current, dict) or id(current) in visited_resources:
            continue
        visited_resources.add(id(current))
        xobjects = _resolve_pdf_object(current.get("/XObject"))
        if not isinstance(xobjects, dict):
            continue
        for reference in xobjects.values():
            visited_xobjects += 1
            if visited_xobjects > 128:
                return False
            xobject = _resolve_pdf_object(reference)
            if not isinstance(xobject, dict):
                continue
            subtype = xobject.get("/Subtype")
            if subtype == "/Image":
                return True
            if subtype == "/Form":
                nested_resources = xobject.get("/Resources")
                if nested_resources is not None:
                    pending.append(nested_resources)
    return False


@dataclass(frozen=True, slots=True)
class _PDFTextSpan:
    text: str
    x: float
    y: float
    font_size: float
    sequence: int


@dataclass(frozen=True, slots=True)
class _PDFLineTrack:
    coordinate: float
    intervals: tuple[tuple[float, float], ...]


def _extract_pdf_page_layout(
    page: object,
) -> tuple[str, list[_PDFTextSpan], list[tuple[float, float, float, float]], bool]:
    """Collect pypdf text anchors and axis-aligned painted path segments."""

    from pypdf import mult

    spans: list[_PDFTextSpan] = []
    segments: list[tuple[float, float, float, float]] = []
    path_segments: list[tuple[float, float, float, float]] = []
    current: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None
    path_supported = True
    spans_bounded = True
    geometry_bounded = True
    sequence = 0

    def point(x: Any, y: Any, matrix: Any) -> tuple[float, float]:
        a, b, c, d, e, f = (float(value) for value in matrix)  # type: ignore[union-attr]
        px, py = float(x), float(y)
        result = (px * a + py * c + e, px * b + py * d + f)
        if not all(isfinite(value) for value in result):
            raise ValueError("PDF path coordinate is not finite")
        return result

    def add_path_segment(start: tuple[float, float], end: tuple[float, float]) -> None:
        nonlocal geometry_bounded
        if len(segments) + len(path_segments) >= MAX_PDF_LAYOUT_SEGMENTS_PER_PAGE:
            geometry_bounded = False
            return
        path_segments.append((start[0], start[1], end[0], end[1]))

    def visitor_text(
        text: str,
        user_matrix: Any,
        text_matrix: Any,
        font_dictionary: object,
        font_size: float,
    ) -> None:
        nonlocal spans_bounded, sequence
        if not text:
            return
        if len(spans) >= MAX_PDF_LAYOUT_SPANS_PER_PAGE:
            spans_bounded = False
            return
        try:
            matrix = mult(text_matrix, user_matrix)
            x, y = float(matrix[4]), float(matrix[5])
            a, b, c, d = (float(matrix[index]) for index in range(4))
            size = float(font_size)
            if not all(isfinite(value) for value in (a, b, c, d, x, y, size)):
                return
            if abs(b) > 1e-6 or abs(c) > 1e-6 or a <= 0 or d <= 0:
                return
        except (IndexError, TypeError, ValueError, OverflowError):
            return
        sequence += 1
        spans.append(_PDFTextSpan(text, x, y, max(0.0, size), sequence))

    def visitor_operand_before(
        operator: bytes,
        operands: Any,
        user_matrix: Any,
        text_matrix: Any,
    ) -> None:
        nonlocal current, subpath_start, path_supported, geometry_bounded
        if not geometry_bounded or operator not in {
            b"m",
            b"l",
            b"h",
            b"re",
            b"c",
            b"v",
            b"y",
            b"S",
            b"s",
            b"f",
            b"F",
            b"f*",
            b"B",
            b"B*",
            b"b",
            b"b*",
            b"n",
        }:
            return
        try:
            values = list(operands)  # type: ignore[arg-type]
            if operator == b"m" and len(values) >= 2:
                current = point(values[0], values[1], user_matrix)
                subpath_start = current
            elif operator == b"l" and len(values) >= 2:
                end = point(values[0], values[1], user_matrix)
                if current is not None:
                    add_path_segment(current, end)
                current = end
            elif operator == b"h":
                if current is not None and subpath_start is not None:
                    add_path_segment(current, subpath_start)
                current = subpath_start
            elif operator == b"re" and len(values) >= 4:
                x, y, width, height = (float(value) for value in values[:4])
                corners = (
                    point(x, y, user_matrix),
                    point(x + width, y, user_matrix),
                    point(x + width, y + height, user_matrix),
                    point(x, y + height, user_matrix),
                )
                for start, end in zip(corners, (*corners[1:], corners[0]), strict=True):
                    add_path_segment(start, end)
                current = subpath_start = corners[0]
            elif operator in {b"c", b"v", b"y"}:
                path_supported = False
            elif operator in {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}:
                if path_supported:
                    segments.extend(path_segments)
                path_segments.clear()
                current = subpath_start = None
                path_supported = True
            elif operator == b"n":
                path_segments.clear()
                current = subpath_start = None
                path_supported = True
        except (IndexError, TypeError, ValueError, OverflowError):
            path_segments.clear()
            current = subpath_start = None
            path_supported = False

    extract_text = getattr(page, "extract_text", None)
    if not callable(extract_text):
        raise PDFParseError("pdf_text_extraction_failed")
    raw_text = (
        extract_text(
            visitor_text=visitor_text,
            visitor_operand_before=visitor_operand_before,
        )
        or ""
    )
    return raw_text, spans, segments, spans_bounded and geometry_bounded


def _page_box_bounds(
    page: object,
) -> tuple[float, float, float, float] | None:
    box = getattr(page, "cropbox", None)
    try:
        bounds = tuple(
            float(getattr(box, name)) for name in ("left", "bottom", "right", "top")
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    left, bottom, right, top = bounds
    if not all(isfinite(value) for value in bounds) or right <= left or top <= bottom:
        return None
    return left, bottom, right, top


def _line_tracks(
    segments: list[tuple[float, float, float, float]], axis: str
) -> list[_PDFLineTrack]:
    entries: list[tuple[float, float, float]] = []
    tolerance = PDF_LAYOUT_COORDINATE_TOLERANCE
    for x0, y0, x1, y1 in segments:
        if axis == "x" and abs(x1 - x0) <= tolerance and abs(y1 - y0) >= 8.0:
            entries.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))
        elif axis == "y" and abs(y1 - y0) <= tolerance and abs(x1 - x0) >= 12.0:
            entries.append(((y0 + y1) / 2, min(x0, x1), max(x0, x1)))
    entries.sort()
    grouped: list[tuple[list[float], list[tuple[float, float]]]] = []
    for coordinate, start, end in entries:
        if not grouped or coordinate - grouped[-1][0][0] > tolerance:
            grouped.append(([coordinate], [(start, end)]))
        else:
            grouped[-1][0].append(coordinate)
            grouped[-1][1].append((start, end))

    tracks: list[_PDFLineTrack] = []
    for coordinates, intervals in grouped:
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1] + tolerance:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        tracks.append(
            _PDFLineTrack(
                sum(coordinates) / len(coordinates),
                tuple((start, end) for start, end in merged),
            )
        )
    return tracks


def _track_covers(track: _PDFLineTrack, start: float, end: float) -> bool:
    tolerance = PDF_LAYOUT_COORDINATE_TOLERANCE
    cursor = start
    for low, high in track.intervals:
        if high < cursor - tolerance:
            continue
        if low > cursor + tolerance:
            return False
        cursor = max(cursor, high)
        if cursor >= end - tolerance:
            return True
    return False


def _cell_text(spans: list[_PDFTextSpan]) -> str:
    if not spans:
        return ""
    ordered = sorted(spans, key=lambda item: (-item.y, item.x, item.sequence))
    lines: list[list[_PDFTextSpan]] = []
    line_y = 0.0
    for span in ordered:
        tolerance = max(1.0, min(2.5, span.font_size * 0.15))
        if not lines or abs(span.y - line_y) > tolerance:
            lines.append([span])
            line_y = span.y
        else:
            lines[-1].append(span)
            line_y = sum(item.y for item in lines[-1]) / len(lines[-1])
    return "\n".join(
        "".join(
            item.text for item in sorted(line, key=lambda item: (item.x, item.sequence))
        )
        for line in lines
    )


def _extract_table_rows(
    spans: list[_PDFTextSpan],
    segments: list[tuple[float, float, float, float]],
    page_bounds: tuple[float, float, float, float] | None,
    rotation: int,
    layout_bounded: bool,
    run_id: str,
    page_number: int,
    first_element_ordinal: int,
    first_line: int,
    first_table_ordinal: int,
    prior_row_count: int,
    prior_cell_count: int,
    prior_text_chars: int,
) -> tuple[list[TextElement], int, int, int, int]:
    """Find complete axis-aligned ruled grids; skip uncertain page layouts."""

    unchanged: tuple[list[TextElement], int, int, int, int] = (
        [],
        first_table_ordinal,
        prior_row_count,
        prior_cell_count,
        prior_text_chars,
    )
    if not layout_bounded or page_bounds is None or rotation % 360 != 0 or not segments:
        return unchanged
    left, bottom, right, top = page_bounds
    page_width, page_height = right - left, top - bottom
    x_tracks = _line_tracks(segments, "x")
    y_tracks = _line_tracks(segments, "y")
    if (
        len(x_tracks) < 3
        or len(y_tracks) < 3
        or len(x_tracks) > MAX_PDF_LAYOUT_COORDINATES_PER_AXIS
        or len(y_tracks) > MAX_PDF_LAYOUT_COORDINATES_PER_AXIS
    ):
        return unchanged
    x_coords = [track.coordinate for track in x_tracks]
    y_coords = [track.coordinate for track in y_tracks]
    if (len(x_coords) - 1) * (len(y_coords) - 1) > MAX_PDF_TABLE_CELLS * 4:
        return unchanged

    cells: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for x_index, (x0, x1) in enumerate(pairwise(x_coords)):
        if x1 - x0 < 12.0:
            continue
        for y_index, (y0, y1) in enumerate(pairwise(y_coords)):
            if y1 - y0 < 8.0:
                continue
            if not (
                _track_covers(y_tracks[y_index], x0, x1)
                and _track_covers(y_tracks[y_index + 1], x0, x1)
                and _track_covers(x_tracks[x_index], y0, y1)
                and _track_covers(x_tracks[x_index + 1], y0, y1)
            ):
                continue
            cells[(x_index, y_index)] = (x0, y0, x1, y1)

    if not cells:
        return unchanged
    spans_by_cell: dict[tuple[int, int], list[_PDFTextSpan]] = defaultdict(list)
    for span in spans:
        x_index = bisect_right(x_coords, span.x) - 1
        y_index = bisect_right(y_coords, span.y) - 1
        key = (x_index, y_index)
        if key in cells:
            spans_by_cell[key].append(span)
    text_by_cell = {key: _cell_text(spans_by_cell.get(key, [])) for key in cells}

    components: list[set[tuple[int, int]]] = []
    remaining = set(cells)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        pending = [seed]
        while pending:
            x_index, y_index = pending.pop()
            for neighbor in (
                (x_index - 1, y_index),
                (x_index + 1, y_index),
                (x_index, y_index - 1),
                (x_index, y_index + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(component)

    accepted: list[tuple[int, int, set[tuple[int, int]]]] = []
    for component in components:
        min_x = min(key[0] for key in component)
        max_x = max(key[0] for key in component)
        min_y = min(key[1] for key in component)
        max_y = max(key[1] for key in component)
        columns, rows = max_x - min_x + 1, max_y - min_y + 1
        if (
            columns < 2
            or rows < 2
            or columns > MAX_PDF_TABLE_CELLS_PER_ROW
            or len(component) != columns * rows
        ):
            continue
        populated = [key for key in component if text_by_cell[key].strip()]
        if (
            len({key[0] for key in populated}) < 2
            or len({key[1] for key in populated}) < 2
        ):
            continue
        accepted.append((min_y, min_x, component))
    accepted.sort(key=lambda item: (-item[0], item[1]))

    elements: list[TextElement] = []
    table_ordinal = first_table_ordinal
    row_count = prior_row_count
    cell_count = prior_cell_count
    text_chars = prior_text_chars
    line_cursor = first_line
    for _min_y, _min_x, component in accepted:
        table_ordinal += 1
        if table_ordinal > MAX_PDF_TABLES:
            raise PDFParseError("pdf_table_limit_exceeded")
        min_x = min(key[0] for key in component)
        max_x = max(key[0] for key in component)
        min_y = min(key[1] for key in component)
        max_y = max(key[1] for key in component)
        for row_index, y_index in enumerate(range(max_y, min_y - 1, -1)):
            row_cells: list[PDFTableCell] = []
            row_text: list[str] = []
            for column_index, x_index in enumerate(range(min_x, max_x + 1)):
                key = (x_index, y_index)
                x0, y0, x1, y1 = cells[key]
                x0, x1 = max(left, x0), min(right, x1)
                y0, y1 = max(bottom, y0), min(top, y1)
                cell_text = text_by_cell[key]
                if len(cell_text) > MAX_PDF_TABLE_CELL_CHARS:
                    raise PDFParseError("pdf_table_limit_exceeded")
                try:
                    cell = PDFTableCell(
                        table_ordinal,
                        row_index,
                        column_index,
                        cell_text,
                        (
                            (x0 - left) / page_width,
                            (top - y1) / page_height,
                            (x1 - left) / page_width,
                            (top - y0) / page_height,
                        ),
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise PDFParseError("pdf_table_cell_provenance_invalid") from exc
                row_cells.append(cell)
                cell_count += 1
                text_chars += len(cell_text)
                if cell_text.strip():
                    row_text.append(cell_text)
            if (
                len(row_cells) > MAX_PDF_TABLE_CELLS_PER_ROW
                or cell_count > MAX_PDF_TABLE_CELLS
                or text_chars > MAX_PDF_TABLE_TEXT_CHARS
            ):
                raise PDFParseError("pdf_table_limit_exceeded")
            if not row_text:
                continue
            row_count += 1
            if row_count > MAX_PDF_TABLE_ROWS:
                raise PDFParseError("pdf_table_limit_exceeded")
            source_text = "\n".join(row_text)
            end_line = line_cursor + source_text.count("\n")
            elements.append(
                TextElement.create(
                    run_id,
                    "table_row",
                    first_element_ordinal + len(elements),
                    source_text,
                    source_text,
                    line_cursor,
                    end_line,
                    page_number=page_number,
                    table_cells=tuple(row_cells),
                )
            )
            line_cursor = end_line + 2
    return elements, table_ordinal, row_count, cell_count, text_chars


def _resolve_pdf_object(value: object) -> object:
    get_object = getattr(value, "get_object", None)
    return get_object() if callable(get_object) else value


def _validate_pdf_input(
    run_id: str,
    filename: str,
    content: bytes,
    *,
    max_bytes: int,
    max_pages: int,
    max_page_text_chars: int,
) -> None:
    if not run_id.strip():
        raise ValueError("Run id must be non-empty")
    if not filename.lower().endswith(".pdf"):
        raise PDFParseError("unsupported_pdf_extension")
    if max_bytes <= 0 or max_pages <= 0 or max_page_text_chars <= 0:
        raise ValueError("PDF limits must be positive")
    if len(content) > max_bytes:
        raise PDFParseError("pdf_input_too_large")
    if not content.startswith(b"%PDF-"):
        raise PDFParseError("invalid_pdf_signature")


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _set_posix_memory_limit(memory_bytes: int) -> None:
    """Linux address-space quota; unsupported platforms fail explicitly."""
    if not sys.platform.startswith("linux"):
        raise PDFParseError("pdf_resource_limit_unavailable")
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limits = [memory_bytes]
        limits.extend(x for x in (soft, hard) if x != resource.RLIM_INFINITY)
        limit = min(limits)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError) as exc:
        raise PDFParseError("pdf_resource_limit_unavailable") from exc


def _decode_parsed(
    value: object,
    run_id: str,
    digest: str,
    *,
    allow_empty_elements: bool = False,
    expected_revision: str = PARSER_REVISION,
) -> ParsedText:
    """Validate the bounded, non-executable result before constructing the IR."""
    try:
        if not isinstance(value, dict) or set(value) != {
            "run_id",
            "parser_revision",
            "content_sha256",
            "original_text",
            "normalized_text",
            "elements",
            "warnings",
            "ocr_pages",
            "pdf_image_regions",
            "pdf_image_provider",
            "pdf_chart_bars",
        }:
            raise ValueError
        if (value["run_id"], value["parser_revision"], value["content_sha256"]) != (
            run_id,
            expected_revision,
            digest,
        ):
            raise ValueError
        expected_image_provider = (
            None
            if expected_revision == PARSER_REVISION
            else PDF_IMAGE_GEOMETRY_PROVIDER
        )
        if value["pdf_image_provider"] != expected_image_provider:
            raise ValueError
        if expected_revision == PARSER_REVISION and value["pdf_chart_bars"]:
            raise ValueError
        if not all(
            isinstance(value[k], str) for k in ("original_text", "normalized_text")
        ):
            raise ValueError
        rows, warnings, ocr_pages = (
            value["elements"],
            value["warnings"],
            value["ocr_pages"],
        )
        if (
            not isinstance(rows, list)
            or (not rows and not allow_empty_elements)
            or not isinstance(warnings, list)
            or not isinstance(ocr_pages, list)
            or ocr_pages
        ):
            raise ValueError
        if not all(isinstance(item, str) for item in warnings):
            raise ValueError
        elements: list[TextElement] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "id",
                "run_id",
                "element_type",
                "ordinal",
                "text",
                "original_text",
                "start_line",
                "end_line",
                "warnings",
                "page_number",
                "table_cells",
            }:
                raise ValueError
            if (
                not all(
                    isinstance(row[k], str)
                    for k in ("id", "run_id", "element_type", "text", "original_text")
                )
                or row["run_id"] != run_id
                or row["element_type"] not in {"page_text", "table_row"}
            ):
                raise ValueError
            if (
                not all(
                    type(row[k]) is int
                    for k in ("ordinal", "start_line", "end_line", "page_number")
                )
                or row["ordinal"] != len(elements)
                or row["start_line"] < 1
            ):
                raise ValueError
            if row["end_line"] < row["start_line"] or row["page_number"] < 1:
                raise ValueError
            if not isinstance(row["warnings"], list) or not all(
                isinstance(item, str) for item in row["warnings"]
            ):
                raise ValueError
            table_cells = pdf_table_cells_from_payload(row["table_cells"])
            if (row["element_type"] == "table_row" and not table_cells) or (
                row["element_type"] == "page_text" and table_cells
            ):
                raise ValueError
            element = TextElement.create(
                run_id,
                row["element_type"],
                row["ordinal"],
                row["text"],
                row["original_text"],
                row["start_line"],
                row["end_line"],
                tuple(row["warnings"]),
                row["page_number"],
                table_cells,
            )
            if element.id != row["id"]:
                raise ValueError
            elements.append(element)
        return ParsedText(
            run_id,
            expected_revision,
            digest,
            value["original_text"],
            value["normalized_text"],
            tuple(elements),
            tuple(warnings),
            (),
            pdf_image_regions_from_payload(value["pdf_image_regions"]),
            value["pdf_image_provider"],
            pdf_chart_bars_from_payload(value["pdf_chart_bars"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise PDFParseError("pdf_parser_failed") from exc
