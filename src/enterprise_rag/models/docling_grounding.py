"""Shared source-grounding rules, independent of the DocTags serving backend."""

from __future__ import annotations

import time
from dataclasses import replace

from enterprise_rag.application.answer import (
    EvidenceSnippet,
    ModelProtocolError,
    ModelUnavailable,
)
from enterprise_rag.application.converted_visual_answer import (
    converted_question_kind,
    select_converted_label,
)
from enterprise_rag.application.visual_answer import VisualGroundingDecision
from enterprise_rag.ingestion.ocr import OCRTextRegion
from enterprise_rag.models.docling_decode import ConvertedLabel, decode_doctags
from enterprise_rag.models.docling_runtime import DoclingDecoderRuntime
from enterprise_rag.models.ollama_visual import (
    MAX_VISUAL_CONTEXT_CHARS,
    MAX_VISUAL_INPUT_BYTES,
    _fit_model_image,
    _map_bbox_to_page,
    _trim_page_whitespace,
)


class DoclingGroundingModel:
    timeout_seconds = 180.0
    decoder: DoclingDecoderRuntime | None

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelUnavailable("Local visual request deadline exceeded")
        return remaining

    def _convert(self, image: bytes, instruction: str, deadline: float) -> str:
        raise NotImplementedError

    def decide(
        self, question: str, evidence: EvidenceSnippet, page_png: bytes
    ) -> VisualGroundingDecision:
        return self._decide(question, evidence, page_png, None)

    def decide_with_ocr(
        self,
        question: str,
        evidence: EvidenceSnippet,
        page_png: bytes,
        regions: tuple[OCRTextRegion, ...],
    ) -> VisualGroundingDecision:
        return self._decide(question, evidence, page_png, regions)

    def _decide(
        self,
        question: str,
        evidence: EvidenceSnippet,
        page_png: bytes,
        source_regions: tuple[OCRTextRegion, ...] | None,
    ) -> VisualGroundingDecision:
        if not isinstance(question, str) or not 1 <= len(question) <= 500:
            raise ModelProtocolError("Invalid Docling question")
        kind = converted_question_kind(question)
        if kind is None or (kind.startswith("diagram_") and source_regions == ()):
            return VisualGroundingDecision("insufficient")
        if (
            not isinstance(evidence, EvidenceSnippet)
            or not evidence.evidence_id
            or not 1 <= len(evidence.text) <= MAX_VISUAL_CONTEXT_CHARS
        ):
            raise ModelProtocolError("Invalid Docling evidence")
        if (
            not isinstance(page_png, bytes)
            or not 33 <= len(page_png) <= MAX_VISUAL_INPUT_BYTES
        ):
            raise ModelProtocolError("Invalid Docling page bytes")
        deadline = time.monotonic() + self.timeout_seconds
        if self.decoder is not None:
            self.decoder.preflight(timeout_seconds=min(10, self._remaining(deadline)))
        cropped, crop, page_size = _trim_page_whitespace(page_png)
        submitted = _fit_model_image(cropped, 512)
        instruction = (
            "Convert chart to table."
            if kind in {"chart_max", "bar_max"}
            else "Convert this page to docling."
        )
        content = self._convert(submitted, instruction, deadline)
        converted = (
            decode_doctags(content, submitted)
            if self.decoder is None
            else self.decoder(
                content, submitted, timeout_seconds=min(10, self._remaining(deadline))
            )
        )
        self._remaining(deadline)
        page_coordinates = kind.startswith("diagram_") and source_regions is not None
        if page_coordinates and source_regions is not None:
            converted = replace(
                converted,
                diagram_boxes=tuple(
                    _map_bbox_to_page(box, crop, page_size)
                    for box in converted.diagram_boxes
                ),
                labels=tuple(
                    ConvertedLabel(region.text, region.bbox)
                    for region in source_regions
                ),
            )
        selected = select_converted_label(converted, question, evidence.text)
        if selected is None:
            return VisualGroundingDecision("insufficient")
        quote, bbox = selected
        original_bbox = (
            bbox if page_coordinates else _map_bbox_to_page(bbox, crop, page_size)
        )
        return VisualGroundingDecision(
            "answer", evidence.evidence_id, quote, original_bbox
        )
