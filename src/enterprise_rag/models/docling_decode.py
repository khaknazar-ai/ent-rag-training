"""Use upstream Docling Core to decode model DocTags; never parse OTSL locally."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from io import BytesIO

from enterprise_rag.application.answer import ModelProtocolError, ModelUnavailable

DOCLING_CORE_VERSION = "2.104.0"
MAX_DOCTAGS_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class ConvertedChart:
    rows: tuple[tuple[str, ...], ...]
    bbox: tuple[float, float, float, float]
    kind: str


@dataclass(frozen=True, slots=True)
class ConvertedLabel:
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ConvertedVisualPage:
    charts: tuple[ConvertedChart, ...]
    diagram_boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[ConvertedLabel, ...]


def require_docling_core() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        if version("docling-core") != DOCLING_CORE_VERSION:
            raise ModelUnavailable("Install the pinned docling extra")
    except PackageNotFoundError:
        raise ModelUnavailable("Install the pinned docling extra") from None


def decode_doctags(raw: str, png: bytes) -> ConvertedVisualPage:
    require_docling_core()
    if (
        not isinstance(raw, str)
        or not raw.strip().startswith("<")
        or (
            raw.strip().startswith("<doctag>") and not raw.strip().endswith("</doctag>")
        )
        or len(raw.encode("utf-8")) > MAX_DOCTAGS_BYTES
        or "<page_break>" in raw
        or raw.count("<") > 2048
        or raw.count("<nl>") > 64
        or sum(
            raw.count(tag) for tag in ("<fcel>", "<ched>", "<rhed>", "<srow>", "<ecel>")
        )
        > 512
        or any(
            len(number) > 3 or int(number) > 500
            for number in re.findall(r"<loc_(\d+)>", raw)
        )
    ):
        raise ModelProtocolError("Incomplete or oversized single-page DocTags")
    try:
        from PIL import Image

        from enterprise_rag.application.visual_regions import VisualRegionService

        # Optional SDK boundary: the project does not expose upstream model types.
        core = importlib.import_module("docling_core.types.doc")
        doctags = importlib.import_module("docling_core.types.doc.document")

        with Image.open(BytesIO(png)) as image:
            width, height = image.size
            tags = doctags.DocTagsDocument.from_doctags_and_image_pairs([raw], [image])
            document = core.DoclingDocument.load_from_doctags(tags)

        def bbox(item):
            if len(item.prov) != 1 or item.prov[0].page_no != 1:
                raise ValueError("Missing single-page conversion provenance")
            box = item.prov[0].bbox.to_top_left_origin(height)
            return VisualRegionService._bbox(
                (box.l / width, box.t / height, box.r / width, box.b / height)
            )

        charts = []
        diagrams = []
        labels = []
        if len(document.pictures) + len(document.texts) + len(document.tables) > 256:
            raise ValueError("Conversion item limit exceeded")
        if not document.pictures and not document.texts and not document.tables:
            raise ModelProtocolError("Docling output contains no structural elements")
        for item in document.pictures:
            box = bbox(item)
            if box is None or item.meta is None:
                continue
            classification = item.meta.classification
            predictions = (
                classification.predictions if classification is not None else []
            )
            if len(predictions) != 1:
                continue
            kind = str(predictions[0].class_name)
            if kind == "flow_chart":
                diagrams.append(box)
            chart = item.meta.tabular_chart
            if kind not in {"bar_chart", "line_chart", "pie_chart"} or chart is None:
                continue
            data = chart.chart_data
            if data.num_rows > 64 or data.num_cols > 8 or len(data.table_cells) > 512:
                raise ValueError("Converted chart exceeds its cell limit")
            if any(
                cell.row_span != 1 or cell.col_span != 1 for cell in data.table_cells
            ):
                continue
            rows = tuple(tuple(cell.text for cell in row) for row in data.grid)
            if any(len(text) > 600 for row in rows for text in row):
                raise ValueError("Converted chart text limit exceeded")
            charts.append(ConvertedChart(rows, box, kind))
        for item in document.tables:
            data = item.data
            if data.num_rows > 64 or data.num_cols > 8 or len(data.table_cells) > 512:
                raise ValueError("Converted table exceeds its cell limit")
            if any(
                cell.row_span != 1 or cell.col_span != 1 for cell in data.table_cells
            ):
                continue
            rows = tuple(tuple(cell.text for cell in row) for row in data.grid)
            if any(len(text) > 600 for row in rows for text in row):
                raise ValueError("Converted table text limit exceeded")
            # Whole-image conversion without predicted locations cites its full
            # authorized input, never an invented cell/figure box.
            box = bbox(item) if item.prov else (0.0, 0.0, 1.0, 1.0)
            if box is not None:
                charts.append(ConvertedChart(rows, box, "table"))
        for item in document.texts:
            if str(item.label) != "text" or not item.text.strip():
                continue
            if len(item.text) > 600:
                raise ValueError("Converted label limit exceeded")
            box = bbox(item)
            if box is not None:
                labels.append(ConvertedLabel(item.text, box))
        return ConvertedVisualPage(tuple(charts), tuple(diagrams), tuple(labels))
    except ImportError as error:
        raise ModelUnavailable("Docling SDK native dependency unavailable") from error
    except ModelProtocolError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OSError,
    ) as error:
        raise ModelProtocolError(
            "Docling conversion violates the visual contract"
        ) from error
