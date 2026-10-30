"""Conservative native bar geometry using pdfplumber's public page APIs.

This recognizes only one unstacked vertical series per page. It does not infer
values, units, axes, or chart semantics from pixels. Call inside PDF containment.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from enterprise_rag.ingestion.ir import PDFChartBar


def page_chart_bars(
    page: Any, page_number: int, source_text: str
) -> tuple[PDFChartBar, ...]:
    if page.rotation or tuple(page.cropbox or page.mediabox) != tuple(page.mediabox):
        return ()
    # Images/curves can contain additional series, stacks or labels we cannot see.
    if page.images or page.curves:
        return ()
    rects = list(page.rects)
    if not 2 <= len(rects) <= 32:
        return ()
    rects.sort(key=lambda item: item["x0"])
    tolerance = 1.5  # PDF points, independent of raster resolution.
    widths = [item["x1"] - item["x0"] for item in rects]
    baseline = rects[0]["bottom"]
    if (
        min(widths) < 4
        or max(widths) - min(widths) > tolerance
        or any(not item.get("fill") for item in rects)
        or any(item["bottom"] - item["top"] < 8 for item in rects)
        or any(abs(item["bottom"] - baseline) > tolerance for item in rects)
        or any(a["x1"] + tolerance >= b["x0"] for a, b in pairwise(rects))
    ):
        return ()
    words = page.extract_words()
    origin_x, origin_y, right, bottom = page.bbox
    width, height = right - origin_x, bottom - origin_y

    def box(x0, top, x1, end):
        return (
            (x0 - origin_x) / width,
            (top - origin_y) / height,
            (x1 - origin_x) / width,
            (end - origin_y) / height,
        )

    result = []
    label_tops = []
    used_words: set[int] = set()
    for index, rect in enumerate(rects):
        left = origin_x if index == 0 else (rects[index - 1]["x1"] + rect["x0"]) / 2
        edge = (
            right
            if index + 1 == len(rects)
            else (rect["x1"] + rects[index + 1]["x0"]) / 2
        )
        # One label line immediately below each bar; extra text is ambiguous.
        candidates = [
            (i, word)
            for i, word in enumerate(words)
            if baseline <= word["top"] <= baseline + 36
            and left <= (word["x0"] + word["x1"]) / 2 < edge
        ]
        if not candidates:
            return ()
        candidates.sort(key=lambda pair: pair[1]["x0"])
        selected = [word for _, word in candidates]
        if (
            max(word["top"] for word in selected)
            - min(word["top"] for word in selected)
            > tolerance
        ):
            return ()
        label = " ".join(word["text"] for word in selected)
        if not label or len(label) > 200 or source_text.count(label) != 1:
            return ()
        x0, x1 = (
            min(word["x0"] for word in selected),
            max(word["x1"] for word in selected),
        )
        # A label must straddle its own bar center, not merely lie in its column.
        center = (rect["x0"] + rect["x1"]) / 2
        if not x0 <= center <= x1 or any(i in used_words for i, _ in candidates):
            return ()
        used_words.update(i for i, _ in candidates)
        top, end = (
            min(word["top"] for word in selected),
            max(word["bottom"] for word in selected),
        )
        label_tops.append(top)
        try:
            result.append(
                PDFChartBar(
                    page_number,
                    label,
                    box(rect["x0"], rect["top"], rect["x1"], rect["bottom"]),
                    box(x0, top, x1, end),
                )
            )
        except ValueError:
            return ()
    if max(label_tops) - min(label_tops) > tolerance or len(
        {bar.label for bar in result}
    ) != len(result):
        return ()
    return tuple(result)
