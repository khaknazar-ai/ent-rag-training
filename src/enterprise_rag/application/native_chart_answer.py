"""Explicit geometry-only questions; no inferred values or general chart QA."""

from __future__ import annotations

import re

from enterprise_rag.ingestion.ir import PDFChartBar


def is_native_chart_question(question: str) -> bool:
    return (
        re.fullmatch(
            r"\s*(?:which|what) (?:region|category|label|bar) "
            r"(?:has the tallest bar|is the tallest(?: bar)?)\s*\??\s*",
            question,
            re.IGNORECASE,
        )
        is not None
    )


def tallest_native_bar(bars: tuple[PDFChartBar, ...]) -> PDFChartBar | None:
    if not 2 <= len(bars) <= 32 or len({bar.page_number for bar in bars}) != 1:
        return None
    ordered = sorted(bars, key=lambda bar: bar.bbox[3] - bar.bbox[1], reverse=True)
    # Near ties cannot be resolved reliably by the 1.5 point extraction tolerance.
    if (ordered[0].bbox[3] - ordered[0].bbox[1]) - (
        ordered[1].bbox[3] - ordered[1].bbox[1]
    ) <= 0.01:
        return None
    return ordered[0]


def bar_citation_bbox(
    bars: tuple[PDFChartBar, ...],
) -> tuple[float, float, float, float]:
    """Show the whole comparison, including every bar and category label."""
    boxes = [box for bar in bars for box in (bar.bbox, bar.label_bbox)]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
