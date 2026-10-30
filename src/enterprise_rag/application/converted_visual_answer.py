"""Narrow selections from upstream model conversion, with exact source labels."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from enterprise_rag.models.docling_decode import ConvertedVisualPage


def exact_source_label(source: str, label: str) -> str | None:
    """Align ASCII case differences but return the untouched source span."""
    if not label.strip() or len(label) > 600:
        return None
    matches = list(
        re.finditer(
            r"(?<!\w)" + re.escape(label) + r"(?!\w)",
            source,
            re.IGNORECASE if label.isascii() else 0,
        )
    )
    if len(matches) != 1:
        return None
    quote = matches[0].group()
    return quote if not label.isascii() or quote.isascii() else None


def converted_question_kind(question: str) -> str | None:
    if re.fullmatch(
        r"\s*(?:which|what) (?:region|category|label) has the (?:largest|highest) value\s*\??\s*",
        question,
        re.I,
    ):
        return "chart_max"
    if re.fullmatch(
        r"\s*which (?:region|category|label) has the tallest bar\s*\??\s*",
        question,
        re.I,
    ):
        return "bar_max"
    match = re.fullmatch(
        r"\s*which label is at the (top|bottom) of the (?:flowchart|diagram)\s*\??\s*",
        question,
        re.I,
    )
    return "diagram_" + match[1].lower() if match else None


def select_converted_label(page: ConvertedVisualPage, question: str, source_text: str):
    kind = converted_question_kind(question)
    if kind in {"chart_max", "bar_max"}:
        if len(page.charts) != 1:
            return None
        chart = page.charts[0]
        if kind == "bar_max" and chart.kind != "bar_chart":
            return None
        # One header row and one numeric series. Never guess separators or units.
        if not 3 <= len(chart.rows) <= 64 or any(len(row) != 2 for row in chart.rows):
            return None
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", chart.rows[0][1].strip()):
            return None  # Missing header: never silently omit the first data row.
        entries = []
        for label, value in chart.rows[1:]:
            if (
                not label.strip()
                or exact_source_label(source_text, label) is None
                or re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip()) is None
            ):
                return None
            try:
                number = Decimal(value.strip())
            except InvalidOperation:
                return None
            if not number.is_finite() or number < 0:
                return None
            entries.append((number, exact_source_label(source_text, label)))
        if len({label for _, label in entries}) != len(entries):
            return None
        entries.sort(reverse=True)
        if entries[0][0] == entries[1][0]:
            return None
        return entries[0][1], chart.bbox
    if kind in {"diagram_top", "diagram_bottom"}:
        if len(page.diagram_boxes) != 1:
            return None
        box = page.diagram_boxes[0]
        labels = [
            label
            for label in page.labels
            if box[0] <= label.bbox[0] < label.bbox[2] <= box[2]
            and box[1] <= label.bbox[1] < label.bbox[3] <= box[3]
        ]
        if not 2 <= len(labels) <= 32 or any(
            exact_source_label(source_text, label.text) is None for label in labels
        ):
            return None
        labels.sort(
            key=lambda label: (label.bbox[1] + label.bbox[3]) / 2,
            reverse=kind == "diagram_bottom",
        )
        if max(labels[0].bbox[1], labels[1].bbox[1]) <= min(
            labels[0].bbox[3], labels[1].bbox[3]
        ):
            return None
        return exact_source_label(source_text, labels[0].text), box
    return None
