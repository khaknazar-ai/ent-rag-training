"""One-page, ACL-first visual questions with exact text citation validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from enterprise_rag.application.answer import (
    AnswerResult,
    ContextWindowError,
    EvidenceSnippet,
    ModelProtocolError,
    ModelSelection,
    pack_evidence_context,
    validate_extractive_selection,
)
from enterprise_rag.application.native_chart_answer import (
    bar_citation_bbox,
    is_native_chart_question,
    tallest_native_bar,
)
from enterprise_rag.application.query_history import (
    history_request,
    measure_query_stage,
    record_answer,
)
from enterprise_rag.application.search import SearchService
from enterprise_rag.application.structured_table_answer import (
    find_table_cell_answer,
    parse_table_question,
)
from enterprise_rag.application.visual_regions import VisualRegionService
from enterprise_rag.ingestion.ocr import OCRTextRegion
from enterprise_rag.ports.query_history import QueryHistory
from enterprise_rag.security import ActorContext


@dataclass(frozen=True, slots=True)
class VisualGroundingDecision:
    status: Literal["answer", "insufficient"]
    evidence_id: str | None = None
    quote: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class VisualGroundingModel(Protocol):
    def decide(
        self, question: str, evidence: EvidenceSnippet, page_png: bytes
    ) -> VisualGroundingDecision: ...


@runtime_checkable
class OCRVisualGroundingModel(Protocol):
    def decide_with_ocr(
        self,
        question: str,
        evidence: EvidenceSnippet,
        page_png: bytes,
        regions: tuple[OCRTextRegion, ...],
    ) -> VisualGroundingDecision: ...


@dataclass(frozen=True, slots=True)
class VisualAnswerOutcome:
    result: AnswerResult
    bbox: tuple[float, float, float, float] | None


class VisualAnswerService:
    """Retrieve one authorized PDF page, inspect it locally, and save a cited result."""

    def __init__(
        self,
        search: SearchService,
        model: VisualGroundingModel | None,
        regions: VisualRegionService,
        history: QueryHistory,
        *,
        model_configuration: dict | None = None,
    ):
        self.search = search
        self.model = model
        self.regions = regions
        self.history = history
        self.model_configuration = model_configuration

    def answer(
        self, actor: ActorContext, question: str, *, search_query: str | None = None
    ) -> VisualAnswerOutcome:
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError("Question must contain 1-500 characters")
        retrieval_query = question if search_query is None else search_query
        if (
            not isinstance(retrieval_query, str)
            or not retrieval_query.strip()
            or len(retrieval_query) > (500 if search_query is None else 200)
        ):
            raise ValueError("Search query must contain 1-200 characters")
        selected_bbox: list[tuple[float, float, float, float] | None] = [None]

        def run():
            result, bbox = self._answer(actor, question, retrieval_query)
            selected_bbox[0] = bbox
            return result

        result = record_answer(
            self.history,
            actor,
            history_request(
                question,
                retrieval_query,
                "page:table-or-native-bar-v1"
                if self.model is None
                else "visual:table-native-bar-or-model-v1",
                self.model_configuration,
            ),
            run,
        )
        return VisualAnswerOutcome(result, selected_bbox[0])

    def _answer(
        self, actor: ActorContext, question: str, retrieval_query: str
    ) -> tuple[AnswerResult, tuple[float, float, float, float] | None]:
        table_question = parse_table_question(question)
        chart_question = is_native_chart_question(question)
        with measure_query_stage("retrieval"):
            candidates = list(self.search.search(actor, retrieval_query, limit=4))
            if table_question is not None and retrieval_query == question:
                for lookup in (
                    table_question.row_label,
                    table_question.column_label,
                ):
                    candidates.extend(self.search.search(actor, lookup, limit=4))
        if table_question is not None:
            for item in tuple(candidates):
                candidates.extend(
                    self.search.get_pdf_table_header_evidence(actor, item.chunk_id)
                )
        candidates_by_id = {item.chunk_id: item for item in candidates}
        candidates = list(candidates_by_id.values())
        table_answer = None
        chart_answers = []
        if chart_question:
            with measure_query_stage("native_chart_lookup"):
                seen_pages = set()
                for item in candidates:
                    if (
                        not item.filename.lower().endswith(".pdf")
                        or len(item.page_numbers) != 1
                    ):
                        continue
                    bars = self.search.get_pdf_chart_bars(actor, item.chunk_id)
                    key = (item.run_id, item.page_numbers)
                    if bars and key not in seen_pages:
                        seen_pages.add(key)
                        chart_answers.append((item, tallest_native_bar(bars), bars))
        chart_answer = (
            chart_answers[0]
            if len(chart_answers) == 1 and chart_answers[0][1] is not None
            else None
        )
        if table_question is not None:
            with measure_query_stage("table_lookup"):
                evidence_cells = []
                for item in candidates:
                    if (
                        item.filename.lower().endswith(".pdf")
                        and len(item.page_numbers) == 1
                    ):
                        cells = self.search.get_pdf_table_cells(actor, item.chunk_id)
                        if cells:
                            evidence_cells.append((item, cells))
                table_answer = find_table_cell_answer(table_question, evidence_cells)
        evidence = (
            chart_answer[0]
            if chart_answer is not None
            else table_answer.evidence
            if table_answer is not None
            else next(
                (
                    item
                    for item in candidates
                    if item.filename.lower().endswith(".pdf")
                    and len(item.page_numbers) == 1
                ),
                None,
            )
        )
        if evidence is None:
            return AnswerResult(None, True, "no_visual_evidence", ()), None
        try:
            with measure_query_stage("context_packing"):
                snippets = pack_evidence_context((evidence,), question)
        except ContextWindowError as error:
            return AnswerResult(None, True, str(error), ()), None
        if not snippets:
            return AnswerResult(None, True, "no_visual_evidence", ()), None
        if chart_answer is not None:
            bar = chart_answer[1]
            assert bar is not None
            decision = VisualGroundingDecision(
                "answer",
                evidence.chunk_id,
                bar.label,
                bar_citation_bbox(chart_answer[2]),
            )
        elif table_answer is not None:
            decision = VisualGroundingDecision(
                "answer",
                evidence.chunk_id,
                table_answer.quote,
                table_answer.bbox,
            )
        else:
            if self.model is None:
                return (
                    AnswerResult(
                        None,
                        True,
                        "unsupported_or_ambiguous_native_chart"
                        if chart_question
                        else "visual_model_not_configured",
                        (),
                    ),
                    None,
                )
            with measure_query_stage("visual_render"):
                image = self.regions.render_authorized_evidence_page(actor, evidence)
            with measure_query_stage("answer_model"):
                if isinstance(self.model, OCRVisualGroundingModel):
                    regions = self.search.get_pdf_ocr_regions(actor, evidence.chunk_id)
                    if regions is None:
                        return AnswerResult(
                            None, True, "visual_evidence_unavailable", ()
                        ), None
                    decision = self.model.decide_with_ocr(
                        question, snippets[0], image, regions
                    )
                else:
                    decision = self.model.decide(question, snippets[0], image)
        if not isinstance(decision, VisualGroundingDecision):
            raise ModelProtocolError("Visual provider returned an invalid contract")
        if decision.status not in {"answer", "insufficient"}:
            raise ModelProtocolError("Visual provider returned an unknown disposition")
        if decision.status == "insufficient":
            return AnswerResult(None, True, "insufficient_visual_support", ()), None
        if (
            decision.evidence_id != evidence.chunk_id
            or not isinstance(decision.quote, str)
            or not decision.quote.strip()
            or decision.bbox is None
        ):
            return AnswerResult(None, True, "unsupported_model_output", ()), None
        try:
            bbox = VisualRegionService._bbox(decision.bbox)
        except ValueError:
            return AnswerResult(None, True, "unsupported_model_output", ()), None
        if bbox is None:
            return AnswerResult(None, True, "unsupported_model_output", ()), None
        with measure_query_stage("citation_validation"):
            result = validate_extractive_selection(
                actor,
                (evidence,),
                snippets,
                ModelSelection(False, decision.evidence_id, decision.quote),
                self.search.get_evidence,
            )
        if result.abstained:
            return result, None
        return result, bbox
