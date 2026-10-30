"""Read pdfplumber image placement metadata without decoding image pixels."""

from __future__ import annotations

from typing import Any

from enterprise_rag.ingestion.ir import MAX_PDF_IMAGE_REGIONS, PDFImageRegion

MAX_IMAGES_PER_PAGE = 128


def page_image_regions(
    page: Any, page_number: int, already: int
) -> tuple[tuple[PDFImageRegion, ...], tuple[str, ...]]:
    """Use public page.images coordinates; refuse unsupported page transforms."""
    if page.rotation or tuple(page.cropbox or page.mediabox) != tuple(page.mediabox):
        return (), (f"pdf_image_geometry_unsupported_page_{page_number}",)
    images = page.images
    if (
        len(images) > MAX_IMAGES_PER_PAGE
        or already + len(images) > MAX_PDF_IMAGE_REGIONS
    ):
        raise ValueError("pdf_image_region_budget_exceeded")
    candidates = []
    warnings = []
    x0, top0, _, _ = page.bbox
    for image in images:
        if image.get("imagemask"):
            warnings.append(f"pdf_image_mask_skipped_page_{page_number}")
            continue
        try:
            width, height = image["srcsize"]
            bbox = (
                (image["x0"] - x0) / page.width,
                (image["top"] - top0) / page.height,
                (image["x1"] - x0) / page.width,
                (image["bottom"] - top0) / page.height,
            )
            candidate = PDFImageRegion(page_number, 0, width, height, bbox)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            warnings.append(f"pdf_image_geometry_invalid_page_{page_number}")
            continue
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item.bbox[1],
            item.bbox[0],
            item.bbox[3],
            item.bbox[2],
            item.pixel_width,
            item.pixel_height,
        )
    )
    regions = tuple(
        PDFImageRegion(
            page_number, ordinal, item.pixel_width, item.pixel_height, item.bbox
        )
        for ordinal, item in enumerate(candidates)
    )
    return regions, tuple(dict.fromkeys(warnings))
