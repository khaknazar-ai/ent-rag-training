"""Explicit evidence-only or extractive semantic answer contracts.

A provider's support disposition is not a deterministic entailment proof.
Nearest-neighbor scores never decide whether a question is answerable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from enterprise_rag.application.answer import (
    AnswerResult,
    EvidenceSnippet,
    ModelProtocolError,
    ModelSelection,
    pack_evidence_context,
    validate_extractive_selection,
)
from enterprise_rag.application.context import ContextWindowError
from enterprise_rag.application.query_history import (
    history_request,
    measure_query_stage,
    record_semantic_answer,
)
from enterprise_rag.application.search import SearchService
from enterprise_rag.application.semantic import SemanticResult
from enterprise_rag.ports.query_history import QueryHistory
from enterprise_rag.retrieval.parents import ParentExpansionResult


@dataclass(frozen=True, slots=True)
class GroundingDecision:
    status: Literal["answer", "insufficient", "conflict"]
    evidence_id: str | None = None
    quote: str | None = None

    def __post_init__(self):
        if self.status not in {"answer", "insufficient", "conflict"}:
            raise ValueError("Unknown grounding disposition")
        if self.status == "answer":
            if (
                not isinstance(self.evidence_id, str)
                or not self.evidence_id.strip()
                or not isinstance(self.quote, str)
                or not self.quote.strip()
            ):
                raise ValueError("Answer disposition requires evidence and quote")
        elif self.evidence_id is not None or self.quote is not None:
            raise ValueError("Abstention must not contain a proposed answer")


@dataclass(frozen=True, slots=True)
class GroundingContext(Sequence):
    children: tuple[EvidenceSnippet, ...]
    expansion: ParentExpansionResult
    budget_qualification: str = "character_caps_only_token_budget_unqualified"
    logical_chunk_ids: tuple[tuple[str, str], ...] = ()

    def __len__(self):
        return len(self.children)

    def __getitem__(self, index):
        return self.children[index]


class GroundedAnswerModel(Protocol):
    def decide(self, query: str, snippets: GroundingContext) -> GroundingDecision: ...


class GroundingRetrieval(Protocol):
    @property
    def lexical(self) -> SearchService: ...

    async def search(self, actor, question, *, strategy, limit) -> SemanticResult: ...


@dataclass(frozen=True, slots=True)
class SemanticAnswerResult:
    mode: str
    retrieval: SemanticResult
    context: tuple[EvidenceSnippet, ...]
    result: AnswerResult | None
    query_run_id: str | None = None
    parent_expansion: ParentExpansionResult | None = None


class SemanticAnswerService:
    def __init__(
        self,
        retrieval: GroundingRetrieval,
        model: GroundedAnswerModel | None = None,
        *,
        history: QueryHistory | None = None,
        model_configuration: dict | None = None,
    ):
        self.retrieval, self.model = retrieval, model
        self.history, self.model_configuration = history, model_configuration

    async def answer(
        self,
        actor,
        question: str,
        *,
        strategy: str = "hybrid",
        mode: str = "evidence_only",
    ) -> SemanticAnswerResult:
        if self.history is not None:
            return await record_semantic_answer(
                self.history,
                actor,
                history_request(
                    question,
                    question,
                    "semantic:" + strategy + ":" + mode,
                    self.model_configuration,
                ),
                lambda: self._answer(actor, question, strategy=strategy, mode=mode),
            )
        return await self._answer(actor, question, strategy=strategy, mode=mode)

    async def _answer(self, actor, question, *, strategy, mode):
        if mode not in {"evidence_only", "extractive"}:
            raise ValueError("Unknown explicit semantic answer mode")
        if mode == "extractive" and self.model is None:
            raise ValueError("Extractive semantic mode requires a grounding provider")
        with measure_query_stage("retrieval"):
            retrieved = await self.retrieval.search(
                actor, question, strategy=strategy, limit=4
            )
        evidence = tuple(hit.evidence for hit in retrieved.items)
        if not evidence:
            return SemanticAnswerResult(
                mode,
                retrieved,
                (),
                None
                if mode == "evidence_only"
                else AnswerResult(None, True, "no_authorized_evidence", ()),
            )
        try:
            with measure_query_stage("context_packing"):
                context = pack_evidence_context(evidence, question)
        except ContextWindowError as error:
            return SemanticAnswerResult(
                mode,
                SemanticResult(strategy, ()),
                (),
                AnswerResult(None, True, str(error), ()),
            )
        try:
            with measure_query_stage("parent_expansion"):
                expansion = await asyncio.to_thread(
                    self.retrieval.lexical.expand_persisted_parents, actor, evidence
                )
        except ValueError:
            return SemanticAnswerResult(
                mode,
                SemanticResult(strategy, ()),
                (),
                AnswerResult(None, True, "parent_context_unavailable", ()),
            )
        if mode == "evidence_only":
            with measure_query_stage("authorization_recheck"):
                return await self._live_result(
                    actor, mode, retrieved, context, None, expansion
                )
        if not context:
            return SemanticAnswerResult(
                mode,
                retrieved,
                (),
                AnswerResult(None, True, "no_authorized_evidence", ()),
            )
        assert self.model is not None
        decision = await asyncio.to_thread(
            self._decide_live,
            actor,
            question,
            evidence,
            GroundingContext(
                context,
                expansion,
                logical_chunk_ids=tuple(
                    (item.chunk_id, item.logical_chunk_id)
                    for item in evidence
                    if item.logical_chunk_id is not None
                    and any(child.evidence_id == item.chunk_id for child in context)
                ),
            ),
        )
        if decision is None:
            with measure_query_stage("authorization_recheck"):
                return await self._live_result(
                    actor,
                    mode,
                    retrieved,
                    context,
                    AnswerResult(None, True, "evidence_no_longer_authorized", ()),
                    expansion,
                )
        if not isinstance(decision, GroundingDecision):
            raise ModelProtocolError("Grounding provider returned an invalid contract")
        if decision.status == "answer":
            with measure_query_stage("citation_validation"):
                result = await asyncio.to_thread(
                    validate_extractive_selection,
                    actor,
                    evidence,
                    context,
                    ModelSelection(False, decision.evidence_id, decision.quote),
                    self.retrieval.lexical.get_evidence,
                )
        else:
            result = AnswerResult(
                None,
                True,
                "insufficient_support"
                if decision.status == "insufficient"
                else "conflicting_evidence",
                (),
            )
        with measure_query_stage("authorization_recheck"):
            return await self._live_result(
                actor, mode, retrieved, context, result, expansion
            )

    def _decide_live(self, actor, question, evidence, context):
        with measure_query_stage("authorization_recheck"):
            for item in evidence:
                current = self.retrieval.lexical.get_evidence(actor, item.chunk_id)
                if (
                    current is None
                    or not self.retrieval.lexical._same_canonical_evidence(
                        item, current
                    )
                ):
                    return None
            try:
                fresh = self.retrieval.lexical.expand_persisted_parents(actor, evidence)
            except ValueError:
                return None
        if fresh != context.expansion:
            return None
        assert self.model is not None
        with measure_query_stage("answer_model"):
            return self.model.decide(question, context)

    async def _live_result(self, actor, mode, retrieved, context, result, expansion):
        # Returning a retrieval trace/context is itself disclosure. Recheck all
        # items after provider latency, including abstentions/evidence-only mode.
        live = []
        for hit in retrieved.items:
            current = await asyncio.to_thread(
                self.retrieval.lexical.get_evidence, actor, hit.evidence.chunk_id
            )
            if current is not None and self.retrieval.lexical._same_canonical_evidence(
                hit.evidence, current
            ):
                live.append(hit)
        try:
            fresh = await asyncio.to_thread(
                self.retrieval.lexical.expand_persisted_parents,
                actor,
                tuple(hit.evidence for hit in live),
            )
            valid_parents = fresh == expansion
        except ValueError:
            valid_parents = False
        if not valid_parents:
            return SemanticAnswerResult(
                mode,
                SemanticResult(retrieved.strategy, ()),
                (),
                AnswerResult(
                    None,
                    True,
                    "evidence_no_longer_authorized"
                    if len(live) != len(retrieved.items)
                    else "parent_context_no_longer_authorized",
                    (),
                ),
            )
        ids = {hit.evidence.chunk_id for hit in live}
        if result is not None and (
            len(live) != len(retrieved.items)
            or any(c.evidence_id not in ids for c in result.citations)
        ):
            result = AnswerResult(None, True, "evidence_no_longer_authorized", ())
        return SemanticAnswerResult(
            mode,
            SemanticResult(retrieved.strategy, tuple(live)),
            tuple(snippet for snippet in context if snippet.evidence_id in ids),
            result,
            parent_expansion=expansion,
        )
