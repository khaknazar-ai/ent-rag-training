"""Safe literal lexical retrieval over the SQLite FTS5 projection."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass

from enterprise_rag.persistence import SQLiteMetadataStore


@dataclass(frozen=True, slots=True)
class LexicalHit:
    chunk_id: str
    document_id: str
    document_version_id: str
    run_id: str
    text: str
    score: float
    page_numbers: tuple[int, ...] = ()


def build_literal_fts_query(query: str) -> str:
    """Convert user text to a literal AND query, never raw FTS syntax."""

    tokens = re.findall(
        r"[^\W_]+", unicodedata.normalize("NFKC", query), flags=re.UNICODE
    )
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def search_lite(
    metadata: SQLiteMetadataStore,
    tenant_id: str,
    query: str,
    *,
    allowed_document_ids: Collection[str],
    limit: int = 20,
    offset: int = 0,
) -> tuple[LexicalHit, ...]:
    """Search with explicit authorization scope and active-version filtering."""

    fts_query = build_literal_fts_query(query)
    return tuple(
        LexicalHit(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            document_version_id=row.document_version_id,
            run_id=row.run_id,
            text=row.text,
            score=row.score,
            page_numbers=row.page_numbers,
        )
        for row in metadata.search_lexical(
            tenant_id,
            fts_query,
            allowed_document_ids,
            limit=limit,
            offset=offset,
        )
    )
