"""Explicit OpenCV geometry + persisted OCR; no generative model or OCR retry."""

from __future__ import annotations

import base64
import importlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from enterprise_rag.application.answer import EvidenceSnippet, ModelProtocolError
from enterprise_rag.application.converted_visual_answer import (
    converted_question_kind,
    exact_source_label,
    select_converted_label,
)
from enterprise_rag.application.structured_table_answer import parse_table_question
from enterprise_rag.application.visual_answer import VisualGroundingDecision
from enterprise_rag.ingestion.ocr import OCRTextRegion
from enterprise_rag.models.docling_decode import (
    ConvertedChart,
    ConvertedLabel,
    ConvertedVisualPage,
)
from enterprise_rag.models.docling_runtime import DoclingDecoderRuntime
from enterprise_rag.workers.processes import START_GATE_ENV

RASTER_CONTRACT = "opencv-ruled-table-solid-bar-bounded-paths-v3"


class RasterGeometryRuntime(DoclingDecoderRuntime):
    _package_directory = "cv2"
    _worker_module = "enterprise_rag.models.raster_geometry"

    def extract(self, png: bytes, source: str, regions: tuple[OCRTextRegion, ...]):
        return self._run(
            {
                "png": base64.b64encode(png).decode("ascii"),
                "source": source,
                "regions": [asdict(region) for region in regions],
            },
            10,
        )

    def graph(
        self,
        png: bytes,
        source: str,
        regions: tuple[OCRTextRegion, ...],
        *,
        include_conditions: bool = True,
    ):
        from enterprise_rag.models.raster_graph import graph_from_payload

        return graph_from_payload(
            self._invoke(
                {
                    "mode": "diagram-conditions-v3"
                    if include_conditions
                    else "diagram-graph",
                    "png": base64.b64encode(png).decode("ascii"),
                    "source": source,
                    "regions": [asdict(r) for r in regions],
                },
                10,
            )
        )


class _RasterOCRRuntime(RasterGeometryRuntime):
    _max_result_bytes = 1024**2


class RasterOCRPreprocessor:
    revision = "opencv-4.13.0-remove-long-rules-endcaps-max1600-v2"

    def __init__(self, python: Path, packages: Path, temp_root: Path):
        self.runtime = _RasterOCRRuntime(python, packages, temp_root)

    def __call__(self, png: bytes) -> bytes:
        payload = self.runtime._invoke(
            {"mode": "ocr-grid", "png": base64.b64encode(png).decode("ascii")}, 10
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"prepared_png"}
            or not isinstance(payload["prepared_png"], str)
        ):
            raise ModelProtocolError("Invalid raster OCR preprocessing output")
        result = base64.b64decode(payload["prepared_png"], validate=True)
        if not 33 <= len(result) <= 4 * 1024**2:
            raise ModelProtocolError(
                "Raster OCR preprocessing output exceeds its limit"
            )
        return result


def prepare_ocr_grid(png: bytes) -> bytes:
    cv = importlib.import_module("cv2")
    np = importlib.import_module("numpy")
    if (
        cv.__version__ != "4.13.0"
        or not 33 <= len(png) <= 4 * 1024**2
        or not png.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise ValueError("Invalid pinned raster input")
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    if not 0 < width * height <= 20_000_000:
        raise ValueError("Pixel limit")
    cv.setNumThreads(2)
    cv.ocl.setUseOpenCL(False)
    image = cv.imdecode(np.frombuffer(png, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    if image is None or image.shape != (height, width):
        raise ValueError("Invalid raster image")
    if max(width, height) > 1600:
        scale = 1600 / max(width, height)
        image = cv.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv.INTER_AREA,
        )
    height, width = image.shape
    _, ink = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    horizontal = cv.morphologyEx(
        ink,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (max(40, width // 8), 1)),
    )
    vertical = cv.morphologyEx(
        ink,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (1, max(40, height // 10))),
    )
    rules = cv.dilate(
        cv.bitwise_or(horizontal, vertical),
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
    )
    # Thin-rule endpoints can retain a filled arrowhead after the shaft is
    # erased. Extend the removal mask back along each endpoint, never beyond
    # the rule in its axial direction, keeping adjacent node text intact.
    reach, radius = (
        max(4, round(min(width, height) * 0.05)),
        max(3, round(min(width, height) * 0.03)),
    )
    for line_mask, is_horizontal in ((horizontal, True), (vertical, False)):
        contours, _ = cv.findContours(
            line_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 4096:
            raise ValueError("OCR rule count limit")
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if is_horizontal and h <= max(5, height * 0.012):
                center = y + h // 2
                cv.rectangle(
                    rules,
                    (x, center - radius),
                    (min(x + reach, x + w - 1), center + radius),
                    255,
                    -1,
                )
                cv.rectangle(
                    rules,
                    (max(x, x + w - 1 - reach), center - radius),
                    (x + w - 1, center + radius),
                    255,
                    -1,
                )
            elif not is_horizontal and w <= max(5, width * 0.012):
                center = x + w // 2
                cv.rectangle(
                    rules,
                    (center - radius, y),
                    (center + radius, min(y + reach, y + h - 1)),
                    255,
                    -1,
                )
                cv.rectangle(
                    rules,
                    (center - radius, max(y, y + h - 1 - reach)),
                    (center + radius, y + h - 1),
                    255,
                    -1,
                )
    image[rules > 0] = 255
    valid, encoded = cv.imencode(".png", image, [cv.IMWRITE_PNG_COMPRESSION, 9])
    if not valid:
        raise ValueError("PNG encoding failed")
    return encoded.tobytes()


class RasterGeometryModel:
    def __init__(self, runtime: RasterGeometryRuntime):
        self.runtime = runtime

    def decide(self, question, evidence, page_png):
        return VisualGroundingDecision("insufficient")

    def decide_with_ocr(
        self,
        question: str,
        evidence: EvidenceSnippet,
        page_png: bytes,
        regions: tuple[OCRTextRegion, ...],
    ):
        from enterprise_rag.models.raster_graph import parse_step_question, select_step

        table = parse_table_question(question)
        kind = converted_question_kind(question)
        step = parse_step_question(question)
        if not regions or (table is None and kind is None and step is None):
            return VisualGroundingDecision("insufficient")
        if (
            len(page_png) > 4 * 1024**2
            or len(evidence.text) > 4000
            or len(regions) > 512
        ):
            raise ModelProtocolError("Raster geometry input limit exceeded")
        if step is not None:
            graph = (
                self.runtime.graph(
                    page_png, evidence.text, regions, include_conditions=True
                )
                if step.condition is not None
                else self.runtime.graph(page_png, evidence.text, regions)
            )
            for edge in graph.edges:
                if edge.condition is not None:
                    condition_matches = [
                        r
                        for r in regions
                        if r.text == edge.condition
                        and r.bbox == edge.condition_bbox
                        and r.confidence >= 0.5
                        and evidence.text[r.start_char : r.end_char] == r.text
                    ]
                    if len(condition_matches) != 1:
                        return VisualGroundingDecision("insufficient")
            selected = select_step(graph, step, evidence.text)
            return (
                VisualGroundingDecision("insufficient")
                if selected is None
                else VisualGroundingDecision("answer", evidence.evidence_id, *selected)
            )
        page = self.runtime.extract(page_png, evidence.text, regions)
        selected = None
        if table is not None:
            matches = []
            for grid in page.charts:
                if grid.kind != "table" or not grid.rows:
                    continue
                columns = [
                    i
                    for i, value in enumerate(grid.rows[0])
                    if value.casefold() == table.column_label.casefold()
                ]
                rows = [
                    row
                    for row in grid.rows[1:]
                    if row and row[0].casefold() == table.row_label.casefold()
                ]
                if len(columns) == len(rows) == 1 and columns[0] < len(rows[0]):
                    quote = exact_source_label(evidence.text, rows[0][columns[0]])
                    if quote is not None:
                        matches.append((quote, grid.bbox))
            selected = matches[0] if len(matches) == 1 else None
        else:
            selected = select_converted_label(page, question, evidence.text)
        if selected is None:
            return VisualGroundingDecision("insufficient")
        return VisualGroundingDecision("answer", evidence.evidence_id, *selected)


@dataclass(frozen=True)
class _RasterScene:
    cv: Any
    np: Any
    image: Any
    gray: Any
    ink: Any
    width: int
    height: int
    norm: Callable
    words: Callable
    label: Callable


def _read_scene(
    png: bytes, source: str, regions: tuple[OCRTextRegion, ...]
) -> _RasterScene:
    cv = importlib.import_module("cv2")
    np = importlib.import_module("numpy")
    if cv.__version__ != "4.13.0":
        raise ValueError("Install the pinned raster geometry extra")

    cv.setNumThreads(2)
    cv.ocl.setUseOpenCL(False)
    if len(png) > 4 * 1024**2 or len(source) > 4000 or len(regions) > 512:
        raise ValueError("Input limit exceeded")
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 33:
        raise ValueError("PNG required")
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    if not 0 < width * height <= 20_000_000:
        raise ValueError("Pixel limit exceeded")
    image = cv.imdecode(np.frombuffer(png, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError("Invalid PNG")
    if max(width, height) > 1600:
        scale = 1600 / max(width, height)
        width, height = round(width * scale), round(height * scale)
        image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    for region in regions:
        if source[region.start_char : region.end_char] != region.text:
            raise ValueError("OCR provenance mismatch")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, ink = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    def norm(box):
        x, y, right, bottom = box
        return (x / width, y / height, right / width, bottom / height)

    def words(box):
        left, top, right, bottom = norm(box)
        return [
            r
            for r in regions
            if r.confidence >= 0.5
            and left <= r.bbox[0] < r.bbox[2] <= right
            and top <= r.bbox[1] < r.bbox[3] <= bottom
        ]

    def label(items):
        if not items:
            return None
        first, last = min(r.start_char for r in items), max(r.end_char for r in items)
        if any(
            r.start_char >= first and r.end_char <= last and r not in items
            for r in regions
        ):
            return None
        value = source[first:last]
        return (
            value if len(value) <= 600 and exact_source_label(source, value) else None
        )

    return _RasterScene(cv, np, image, gray, ink, width, height, norm, words, label)


def _rectangle_nodes(scene: _RasterScene, table_boxes=()):
    cv, ink, width, height = scene.cv, scene.ink, scene.width, scene.height
    words, label = scene.words, scene.label
    rectangles: list[tuple[tuple[int, int, int, int], str]] = []
    contours, _ = cv.findContours(ink, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 8192:
        raise ValueError("Contour limit exceeded")
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.015 * cv.arcLength(contour, True), True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue
        x, y, w, h = cv.boundingRect(approx)
        if w < width * 0.1 or h < height * 0.04 or w * h > width * height * 0.6:
            continue
        if cv.contourArea(contour) / (w * h) < 0.88:
            continue
        if any(
            x >= left and y >= top and x + w <= right and y + h <= bottom
            for left, top, right, bottom in table_boxes
        ):
            continue
        value = label(words((x, y, x + w, y + h)))
        if not value:
            continue
        candidate = (x, y, x + w, y + h)

        def same_outline(other, x=x, y=y, w=w, h=h):
            box = other[0]
            intersection = max(0, min(x + w, box[2]) - max(x, box[0])) * max(
                0, min(y + h, box[3]) - max(y, box[1])
            )
            union = w * h + (box[2] - box[0]) * (box[3] - box[1]) - intersection
            return intersection / union > 0.85

        if any(same_outline(other) for other in rectangles):
            continue
        rectangles.append((candidate, value))
    rectangles.sort(key=lambda item: item[0][1])
    return rectangles


def extract_geometry(
    png: bytes, source: str, regions: tuple[OCRTextRegion, ...]
) -> ConvertedVisualPage:
    # OpenCV's upstream morphology sample supplies line extraction; contour
    # approximation supplies rectangle candidates. Application checks below
    # bind them to untouched, already authorized OCR spans.
    scene = _read_scene(png, source, regions)
    cv, np, image, ink, width, height = (
        scene.cv,
        scene.np,
        scene.image,
        scene.ink,
        scene.width,
        scene.height,
    )
    norm, words, label = scene.norm, scene.words, scene.label
    gray = scene.gray
    horizontal = cv.morphologyEx(
        ink,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (max(12, width // 30), 1)),
    )
    vertical = cv.morphologyEx(
        ink,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (1, max(12, height // 30))),
    )

    def positions(values):
        indexes = np.flatnonzero(values)
        groups = np.split(indexes, np.where(np.diff(indexes) > 1)[0] + 1)
        return [round(float(group.mean())) for group in groups if len(group)]

    contours, _ = cv.findContours(
        cv.bitwise_or(horizontal, vertical), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 4096:
        raise ValueError("Contour limit exceeded")
    charts = []
    table_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w < width * 0.1 or h < height * 0.05:
            continue
        xs = positions(
            np.count_nonzero(vertical[y : y + h, x : x + w], axis=0) >= 0.9 * h
        )
        ys = positions(
            np.count_nonzero(horizontal[y : y + h, x : x + w], axis=1) >= 0.9 * w
        )
        if not 3 <= len(xs) <= 9 or not 3 <= len(ys) <= 65:
            continue
        rows = []
        for top, bottom in pairwise(ys):
            row = []
            for left, right in pairwise(xs):
                value = label(words((x + left, y + top, x + right, y + bottom)))
                row.append(value or "")
            rows.append(tuple(row))
        if all(all(row) for row in rows):
            box = (x, y, x + w, y + h)
            charts.append(ConvertedChart(tuple(rows), norm(box), "table"))
            table_boxes.append(box)

    # Solid colored or midtone rectangular bars only. No axis calibration or numerical
    # value is inferred; heights support a category-ranking answer only.
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    colored = cv.bitwise_or(
        cv.inRange(hsv, np.array([0, 80, 40]), np.array([179, 255, 245])),
        cv.inRange(gray, 32, 220),
    )
    # Antialiased gray axis edges can join otherwise separate solid bars.
    # Remove only long horizontal runs; individual bars are narrower than this.
    long_runs = cv.morphologyEx(
        colored,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (max(32, width // 3), 1)),
    )
    colored = cv.subtract(colored, long_runs)
    bar_contours, _ = cv.findContours(colored, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bars = []
    for contour in bar_contours:
        x, y, w, h = cv.boundingRect(contour)
        if (
            w < width * 0.025
            or h < height * 0.05
            or w > width * 0.3
            or h > height * 0.85
        ):
            continue
        if cv.contourArea(contour) / (w * h) < 0.92:
            continue
        bars.append((x, y, w, h))
    bars.sort()
    if not charts and 2 <= len(bars) <= 32:
        baseline = max(y + h for x, y, w, h in bars)
        widths = [w for x, y, w, h in bars]
        valid = max(widths) <= min(widths) * 1.2 and all(
            abs(y + h - baseline) <= max(3, height * 0.01) for x, y, w, h in bars
        )
        # Require a visible shared horizontal baseline across the bar group.
        span_left, span_right = bars[0][0], bars[-1][0] + bars[-1][2]
        tolerance = max(3, round(height * 0.005))
        baseline_rows = horizontal[
            max(0, baseline - tolerance) : min(height, baseline + tolerance),
            span_left:span_right,
        ]
        valid = valid and any(
            np.count_nonzero(row) >= 0.9 * (span_right - span_left)
            for row in baseline_rows
        )
        entries = []
        bounds = []
        for i, (x, y, w, h) in enumerate(bars):
            left = (bars[i - 1][0] + bars[i - 1][2] + x) / 2 if i else max(0, x - w)
            right = (
                (x + w + bars[i + 1][0]) / 2
                if i + 1 < len(bars)
                else min(width, x + 2 * w)
            )
            items = words(
                (left, baseline, right, min(height, baseline + height * 0.12))
            )
            value = label(items)
            if value is None or any(r.bbox[1] < baseline / height for r in items):
                valid = False
                break
            entries.append((value, str(h)))
            bounds.extend([norm((x, y, x + w, y + h)), *[r.bbox for r in items]])
        if valid and len({v for v, _ in entries}) == len(entries):
            bbox = (
                min(b[0] for b in bounds),
                min(b[1] for b in bounds),
                max(b[2] for b in bounds),
                max(b[3] for b in bounds),
            )
            charts.append(
                ConvertedChart(
                    (("Category", "Bar height (pixels)"), *entries), bbox, "bar_chart"
                )
            )

    rectangles = _rectangle_nodes(scene, table_boxes)
    diagrams = []
    labels = []
    if not charts and 2 <= len(rectangles) <= 32:
        centers = [(b[0] + b[2]) / 2 for b, _ in rectangles]
        valid = max(centers) - min(centers) <= width * 0.03
        for (a, _), (b, _) in pairwise(rectangles):
            center = round((a[0] + a[2] + b[0] + b[2]) / 4)
            if b[1] <= a[3] or np.count_nonzero(
                ink[a[3] : b[1], max(0, center - 2) : center + 3].any(axis=1)
            ) < 0.6 * (b[1] - a[3]):
                valid = False
        if valid:
            boxes = [b for b, _ in rectangles]
            diagrams.append(
                norm(
                    (
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    )
                )
            )
            labels = [ConvertedLabel(value, norm(box)) for box, value in rectangles]
    return ConvertedVisualPage(tuple(charts), tuple(diagrams), tuple(labels))


def runtime_main(request: str, result: str):
    if os.environ.get(START_GATE_ENV) != "1" or sys.stdin.buffer.read(1) != b"1":
        raise SystemExit(2)
    if os.name != "nt":
        from enterprise_rag.ingestion.parsers.pdf import _set_posix_memory_limit

        _set_posix_memory_limit(1024**3)
    try:
        payload = json.loads(Path(request).read_text())
        if payload.get("mode") == "ocr-grid":
            png = prepare_ocr_grid(base64.b64decode(payload["png"], validate=True))
            Path(result).write_text(
                json.dumps({"prepared_png": base64.b64encode(png).decode("ascii")}),
                encoding="utf-8",
            )
            return
        if len(payload.get("regions", [])) > 512:
            raise ValueError("OCR region limit")
        regions = tuple(
            OCRTextRegion(**{**r, "bbox": tuple(r["bbox"])}) for r in payload["regions"]
        )
        if payload.get("mode") in {"diagram-graph", "diagram-conditions-v3"}:
            from enterprise_rag.models.raster_graph import extract_graph

            graph = extract_graph(
                base64.b64decode(payload["png"], validate=True),
                payload["source"],
                regions,
            )
            output = asdict(graph)
            if payload.get("mode") == "diagram-graph":
                for edge in output["edges"]:
                    edge.pop("condition")
                    edge.pop("condition_bbox")
            Path(result).write_text(json.dumps(output), encoding="utf-8")
            return
        page = extract_geometry(
            base64.b64decode(payload["png"], validate=True), payload["source"], regions
        )
        output = asdict(page)
    except Exception:
        output = {"error": "raster_geometry_failed"}
    Path(result).write_text(json.dumps(output), encoding="utf-8")
