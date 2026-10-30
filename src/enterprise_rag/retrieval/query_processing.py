"""Bounded lexical question plans; no model, synonym service or raw FTS syntax."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

QUERY_REVISION = "bounded-question-lexical-v1"
MAX_QUERY_CHARS = 500
MAX_QUERY_TOKENS = 32
MAX_CONTENT_UNITS = 16
FALLBACK_POOL = 100
MAX_CANDIDATE_CHARS = 65536
SQL_VM_STEPS = 1_000_000
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_QUESTION_START = frozenset(
    {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "does",
        "do",
        "can",
        "could",
        "would",
        "should",
        "please",
    }
)
# These words express an English question's scaffolding. Negation is retained.
# Apart from this explicit scaffold list, unknown words and identifiers remain.
_QUESTION_WORDS = frozenset(
    {
        "often",
        "we",
        "our",
        "you",
        "your",
        "a",
        "an",
        "the",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "have",
        "has",
        "had",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "to",
        "for",
        "from",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "there",
        "about",
        "around",
        "please",
        "tell",
        "me",
        "according",
        "regarding",
        "explain",
        "describe",
        "differ",
        "differs",
        "apply",
        "applies",
        "requirement",
        "requirements",
        "recommend",
        "recommends",
        "recommended",
    }
)


class LexicalQueryBudgetExceeded(ValueError):
    """The cooperative SQLite budget ended a query; no silent fallback follows."""


def words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(unicodedata.normalize("NFKC", text).lower()))


def looks_like_question(text: str) -> bool:
    tokens = words(text)
    return bool(tokens and tokens[0] in _QUESTION_START)


def variants(token: str) -> tuple[tuple[str, bool], ...]:
    """Small ASCII inflection alternatives, never arbitrary user wildcard syntax.

    Prefix expansion is restricted to the -ility/-ble correspondence with a
    six-letter minimum stem. It is a lexical approximation, not semantics.
    """
    result = [(token, False)]
    if token.isascii() and token.isalpha() and len(token) >= 4:
        if token.endswith("ies"):
            result.append((token[:-3] + "y", False))
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
            result.append((token[:-1], False))
            if token.endswith(("ses", "xes", "zes", "ches", "shes")):
                result.append((token[:-2], False))
        else:
            result.append((token + "s", False))
        if token.endswith("ility") and len(token[:-5]) >= 6:
            result.append((token[:-5], True))
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True, slots=True)
class QueryUnit:
    tokens: tuple[str, ...]
    quoted: bool = False
    expanded: bool = False

    @property
    def fts(self) -> str:
        if self.quoted:
            return '"' + " ".join(self.tokens) + '"'
        choices = (
            variants(self.tokens[0]) if self.expanded else ((self.tokens[0], False),)
        )
        expressions = [
            '"' + token + '"' + ("*" if prefix else "") for token, prefix in choices
        ]
        return (
            expressions[0]
            if len(expressions) == 1
            else "(" + " OR ".join(expressions) + ")"
        )

    def matches(self, tokens: tuple[str, ...]) -> bool:
        if self.quoted:
            n = len(self.tokens)
            return any(
                tokens[i : i + n] == self.tokens for i in range(len(tokens) - n + 1)
            )
        choices = (
            variants(self.tokens[0]) if self.expanded else ((self.tokens[0], False),)
        )
        return any(
            any(word.startswith(term) if prefix else word == term for word in tokens)
            for term, prefix in choices
        )


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_fts: str
    reduced_fts: str
    fallback_fts: str
    units: tuple[QueryUnit, ...]
    natural: bool

    def matches(self, text: str) -> frozenset[int]:
        tokens = words(text)
        return frozenset(i for i, unit in enumerate(self.units) if unit.matches(tokens))

    def preserves_phrases(self, text: str) -> bool:
        tokens = words(text)
        return all(unit.matches(tokens) for unit in self.units if unit.quoted)


def plan_question(query: str) -> QueryPlan:
    if (
        not isinstance(query, str)
        or not 1 <= len(query) <= MAX_QUERY_CHARS
        or not query.strip()
    ):
        raise ValueError("Search query must contain 1..500 characters")
    normalized = (
        unicodedata.normalize("NFKC", query)
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    if normalized.count('"') % 2:
        raise ValueError("Unclosed quoted phrase")
    natural = looks_like_question(normalized)
    original = []
    reduced = []
    total_tokens = 0
    for index, part in enumerate(normalized.split('"')):
        tokens = words(part)
        total_tokens += len(tokens)
        if total_tokens > MAX_QUERY_TOKENS:
            raise ValueError("Search query exceeds the 32-token limit")
        if index % 2:
            if not tokens:
                raise ValueError("Quoted phrase must contain words")
            unit = QueryUnit(tokens, quoted=True)
            original.append(unit)
            reduced.append(unit)
        else:
            for word_index, token in enumerate(tokens):
                original.append(QueryUnit((token,)))
                scaffold_respect = (
                    natural
                    and word_index > 0
                    and tokens[word_index - 1 : word_index + 2]
                    == ("with", "respect", "to")
                )
                if (
                    not natural or token not in _QUESTION_WORDS
                ) and not scaffold_respect:
                    reduced.append(QueryUnit((token,), expanded=natural))
    reduced = list(dict.fromkeys(reduced))
    if len(reduced) > MAX_CONTENT_UNITS:
        raise ValueError("Search query exceeds the 16-content-unit limit")
    if not reduced:
        return QueryPlan("", "", "", (), natural)
    strict = " AND ".join(unit.fts for unit in original)
    narrowed = " AND ".join(unit.fts for unit in reduced)
    phrases = [unit.fts for unit in reduced if unit.quoted]
    optional = [unit.fts for unit in reduced if not unit.quoted]
    broad = (
        " AND ".join([*phrases, "(" + " OR ".join(optional) + ")"])
        if optional
        else narrowed
    )
    return QueryPlan(
        strict,
        narrowed if natural else strict,
        broad if natural and len(reduced) > 1 else "",
        tuple(reduced),
        natural,
    )
