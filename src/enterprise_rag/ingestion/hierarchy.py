"""Deterministic document/ATX-section ranges over contiguous canonical elements."""

import re
from dataclasses import dataclass

from enterprise_rag.domain.documents.ids import deterministic_id, sha256_bytes

from .ir import ParsedText

HIERARCHY_REVISION = "native-heading-ranges-v1"
MAX_PARENT_NODES = 4096


@dataclass(frozen=True, slots=True)
class ParentNode:
    id: str
    run_id: str
    kind: str
    parent_id: str | None
    start_ordinal: int
    end_ordinal: int
    content_sha256: str
    page_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ChildParentLink:
    chunk_id: str
    run_id: str
    parent_id: str
    start_char: int
    end_char: int


def parent_id(run_id, kind, start, end, digest):
    return deterministic_id(
        "hierarchy-parent", HIERARCHY_REVISION, run_id, kind, start, end, digest
    )


def build_hierarchy(parsed: ParsedText, chunks):
    elements = parsed.elements
    if not elements:
        return (), ()
    if tuple(e.ordinal for e in elements) != tuple(range(len(elements))):
        raise ValueError("Hierarchy requires contiguous canonical element ordinals")
    headings = []
    for e in elements:
        if e.element_type == "heading":
            match = re.match(r"^(#{1,6})\s+", e.original_text)
            if match is None:
                raise ValueError("Unsupported structural heading provenance")
            headings.append((e.ordinal, len(match[1])))
    if len(headings) + 1 > MAX_PARENT_NODES:
        raise ValueError("Hierarchy node budget exceeded")
    canonical = "\n\n".join(e.text for e in elements)
    starts = []
    position = 0
    for e in elements:
        starts.append(position)
        position += len(e.text) + 2
    starts.append(len(canonical) + 2)
    nodes = []

    def node(kind, start, end, owner):
        body = canonical[starts[start] : starts[end] - 2]
        digest = sha256_bytes(body.encode())
        result = ParentNode(
            parent_id(parsed.run_id, kind, start, end, digest),
            parsed.run_id,
            kind,
            owner,
            start,
            end,
            digest,
            tuple(
                sorted(
                    {
                        e.page_number
                        for e in elements[start:end]
                        if e.page_number is not None
                    }
                )
            ),
        )
        nodes.append(result)
        return result

    root = node("document", 0, len(elements), None)
    owners = [root] * len(elements)
    stack: list[tuple[int, ParentNode]] = []
    for index, (start, level) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        end = next(
            (ordinal for ordinal, depth in headings[index + 1 :] if depth <= level),
            len(elements),
        )
        section = node("section", start, end, stack[-1][1].id if stack else root.id)
        owners[start:end] = [section] * (end - start)
        stack.append((level, section))
    by_element = {e.id: e.ordinal for e in elements}
    cursors: dict[str, int] = {}
    links = []
    for chunk in chunks:
        ordinals = [by_element[e] for e in chunk.element_ids]
        if (
            not ordinals
            or chunk.run_id != parsed.run_id
            or any(owners[i] != owners[ordinals[0]] for i in ordinals)
        ):
            raise ValueError("Child crosses structural section boundary")
        owner = owners[ordinals[0]]
        first = max(starts[ordinals[0]], cursors.get(owner.id, 0))
        last = starts[ordinals[-1] + 1] - 2
        offset = canonical.find(chunk.text, first, last)
        if offset < 0:
            raise ValueError("Child is not a contiguous canonical parent span")
        cursors[owner.id] = offset + len(chunk.text)
        links.append(
            ChildParentLink(
                chunk.id,
                parsed.run_id,
                owner.id,
                offset - starts[owner.start_ordinal],
                offset - starts[owner.start_ordinal] + len(chunk.text),
            )
        )
    return tuple(nodes), tuple(links)
