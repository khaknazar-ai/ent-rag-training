"""Deterministic lexical windows over unchanged canonical chunk text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag.retrieval.query_processing import looks_like_question, variants

CONTEXT_PACKING_REVISION = "lexical-window-v2-question-variants"
MAX_CONTEXT_CHARS_PER_ITEM = 1800
MAX_QUOTE_CHARS = 600
MAX_CONTEXT_SOURCE_CHARS = 65536
MAX_CONTEXT_QUERY_TERMS = 16
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
QUESTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "should",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "often",
    }
)


class ContextWindowError(ValueError):
    """A safe inspectable context policy refusal, before provider access."""


@dataclass(frozen=True, slots=True)
class ContextWindow:
    text: str
    start_char: int
    end_char: int


def select_context_window(text: str, query: str) -> ContextWindow:
    """Select one original substring with a quote guard around lexical matches.

    Offsets are zero-based half-open Unicode code-point indices into the
    canonical chunk, not bytes or document-global positions. Case folding is
    used only for token comparison; source text and offsets never change.
    """
    if not isinstance(text, str) or len(text) > MAX_CONTEXT_SOURCE_CHARS:
        raise ContextWindowError("context_source_too_large_or_invalid")
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ContextWindowError("invalid_context_query")
    budget = MAX_CONTEXT_CHARS_PER_ITEM
    if len(text) <= budget:
        return ContextWindow(text, 0, len(text))

    terms: set[str] = set()
    for match in _WORD.finditer(query):
        key = match.group().casefold()
        if key not in QUESTION_STOPWORDS:
            terms.add(key)
            if len(terms) >= MAX_CONTEXT_QUERY_TERMS:
                break
    exact: dict[str, str] = {}
    prefixes: list[tuple[str, str]] = []
    natural = looks_like_question(query)
    for term in sorted(terms):
        for value, prefix in variants(term) if natural else ((term, False),):
            if prefix:
                prefixes.append((value, term))
            else:
                exact.setdefault(value, term)
    hits = []
    for match in _WORD.finditer(text):
        key = match.group().casefold()
        owner = exact.get(key)
        if owner is None:
            owner = next(
                (term for prefix, term in prefixes if key.startswith(prefix)), None
            )
        if owner is not None:
            hits.append((match.start(), match.end(), owner))
    if any(end - start > budget - 2 * MAX_QUOTE_CHARS for start, end, _ in hits):
        raise ContextWindowError("context_match_exceeds_core_budget")
    if not hits:
        return ContextWindow(text[:budget], 0, budget)

    core_budget = budget - 2 * MAX_QUOTE_CHARS
    counts: dict[str, int] = {}
    left = 0
    best_score = (-1, 0, 0)
    best_span = (0, 0)
    for _right, (_, end, key) in enumerate(hits):
        counts[key] = counts.get(key, 0) + 1
        while end - hits[left][0] > core_budget:
            previous = hits[left][2]
            counts[previous] -= 1
            if not counts[previous]:
                del counts[previous]
            left += 1
        # A redundant leading occurrence only widens this or any later cluster.
        while counts[hits[left][2]] > 1:
            counts[hits[left][2]] -= 1
            left += 1
        start = hits[left][0]
        score = (len(counts), -(end - start), -start)
        if score > best_score:
            best_score = score
            best_span = (start, end)

    start = max(0, min(len(text) - budget, (sum(best_span) - budget) // 2))
    end = start + budget
    return ContextWindow(text[start:end], start, end)
