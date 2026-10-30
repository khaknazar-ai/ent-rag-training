"""Actual authorized lexical retrieval for the explicitly selected parent demo."""

import asyncio
from dataclasses import dataclass

from enterprise_rag.application.search import SearchService
from enterprise_rag.application.semantic import SemanticHit, SemanticResult


@dataclass(frozen=True)
class LexicalParentRetrieval:
    lexical: SearchService

    async def search(self, actor, question, *, strategy="lexical", limit=4):
        if strategy != "lexical":
            raise ValueError("Lexical parent retrieval requires strategy=lexical")
        evidence = await asyncio.to_thread(
            self.lexical.search, actor, question, limit=limit
        )
        return SemanticResult(
            "lexical",
            tuple(SemanticHit(item, lexical_score=item.score) for item in evidence),
        )
