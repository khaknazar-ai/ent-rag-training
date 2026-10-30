"""Actor-owned query history; adapters must reauthorize evidence on every read."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from enterprise_rag.domain.summaries import SummaryScope
from enterprise_rag.security import ActorContext

QUERY_TRACE_STAGES = frozenset(
    {
        "retrieval",
        "context_packing",
        "table_lookup",
        "native_chart_lookup",
        "parent_expansion",
        "visual_render",
        "answer_model",
        "citation_validation",
        "authorization_recheck",
    }
)

if TYPE_CHECKING:
    from enterprise_rag.application.answer import AnswerResult
    from enterprise_rag.domain.model_summary import ModelSummaryResult


class QueryHistoryUnavailable(RuntimeError):
    """Persistence could not be confirmed; never silently return an unrecorded answer."""

    def __init__(self, query_run_id: str | None = None):
        super().__init__("Query history unavailable")
        self.query_run_id = query_run_id


class QueryHistoryForbidden(PermissionError):
    """An existing tenant-qualified user is required."""


@dataclass(frozen=True, slots=True)
class QueryRequest:
    query_text: str
    retrieval_query: str
    normalized_query: str
    retrieval_strategy: str
    model_configuration_json: str
    scope: SummaryScope | None = None


@dataclass(frozen=True, slots=True)
class QueryCompletion:
    status: str
    elapsed_seconds: float
    result: AnswerResult | ModelSummaryResult | None
    error_code: str | None = None
    parent_context_refs: tuple[dict, ...] = ()
    stage_latencies: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class StoredQuery:
    id: str
    tenant_id: str
    user_id: str
    query_text: str
    retrieval_query: str
    normalized_query: str
    retrieval_strategy: str
    model_configuration: dict
    configuration_sha256: str
    status: str
    started_at: str
    finished_at: str | None
    stage_latencies: dict[str, float] | None
    result: AnswerResult | ModelSummaryResult | None
    error_code: str | None
    # Current providers do not expose verified accounting/support information.
    usage: dict | None = None
    cost_usd: float | None = None
    scope: SummaryScope | None = None
    parent_context_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryHistoryPage:
    items: tuple[StoredQuery, ...]
    next_offset: int | None


class QueryHistory(Protocol):
    def start(self, actor: ActorContext, request: QueryRequest) -> str:
        """Durably create running state before provider execution."""

    def complete(
        self, actor: ActorContext, query_run_id: str, outcome: QueryCompletion
    ) -> None:
        """Atomically finalize state, answer and all citations; no provider I/O."""

    def get(self, actor: ActorContext, query_run_id: str) -> StoredQuery | None:
        """Owner only; return None if any cited canonical source is no longer visible."""

    def list(
        self, actor: ActorContext, *, limit: int = 20, offset: int = 0
    ) -> QueryHistoryPage:
        """List owner history; each returned result must pass the same live checks as get."""
