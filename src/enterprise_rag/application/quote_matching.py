"""Recover bounded source spans without rewriting canonical evidence."""

from __future__ import annotations

from enterprise_rag.application.context import (
    MAX_CONTEXT_CHARS_PER_ITEM,
    MAX_QUOTE_CHARS,
)

QUOTE_MATCHING_REVISION = "literal-or-unique-whitespace-v1"


def locate_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    """Return code-point offsets in the supplied window, or refuse a proposal.

    Literal membership keeps its existing first-occurrence behavior. Otherwise
    only runs of Unicode whitespace may differ, and the normalized match must
    be unique. Returned bounds always select the original contiguous text.
    This is formatting recovery, never semantic verification or fuzzy matching.
    """
    if (
        not quote
        or len(quote) > MAX_QUOTE_CHARS
        or len(text) > MAX_CONTEXT_CHARS_PER_ITEM
    ):
        return None
    literal = text.find(quote)
    if literal >= 0:
        return literal, literal + len(quote)

    proposed = " ".join(quote.split())
    if not proposed:
        return None
    characters: list[str] = []
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        start = index
        character = text[index]
        index += 1
        if character.isspace():
            character = " "
            while index < len(text) and text[index].isspace():
                index += 1
        characters.append(character)
        ranges.append((start, index))

    normalized = "".join(characters)
    found = normalized.find(proposed)
    if found < 0 or normalized.find(proposed, found + 1) >= 0:
        return None
    start_char = ranges[found][0]
    end_char = ranges[found + len(proposed) - 1][1]
    if end_char - start_char > MAX_QUOTE_CHARS:
        return None
    return start_char, end_char
