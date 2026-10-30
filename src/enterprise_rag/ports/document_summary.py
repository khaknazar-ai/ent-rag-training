"""Provider and canonical-source boundaries for explicit document summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from enterprise_rag.domain.summaries import SummaryScope
from enterprise_rag.security import ActorContext

if TYPE_CHECKING:
    from enterprise_rag.application.answer import EvidenceSnippet, ModelSelection
    from enterprise_rag.application.search import SearchEvidence


@dataclass(frozen=True, slots=True)
class SummarySource:
    scope: SummaryScope
    evidence: tuple[SearchEvidence, ...]
    refusal_reason: str | None = None


class SummarySourceReader(Protocol):
    def read(
        self, actor: ActorContext, document_id: str, version_id: str
    ) -> SummarySource | None:
        """Read the entire bounded active canonical text or an explicit refusal."""


class SummaryProvider(Protocol):
    def summarize(
        self, snippets: tuple[EvidenceSnippet, ...]
    ) -> tuple[ModelSelection, ...]:
        """Select up to four representative exact extracts from the full input.

        This legacy extractive capability remains strict. Paraphrase providers
        use the separate typed ModelSummaryProvider with its prompt token gate.
        """
