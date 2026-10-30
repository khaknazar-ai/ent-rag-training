"""Strict exact-cell lookup for natural-language ruled-table questions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from enterprise_rag.application.search import PDFTableCellEvidence, SearchEvidence

_QUESTION = re.compile(
    r"^\s*(?:which|what)\s+(?P<column>(?:[^?!.]|(?<=\d)\.(?=\d)){1,80}?)\s+is\s+"
    r"(?:listed|shown|recorded)\s+for\s+(?P<row>(?:[^?!.]|(?<=\d)\.(?=\d)){1,200}?)\s*[?!.]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TableQuestion:
    column_label: str
    row_label: str


@dataclass(frozen=True, slots=True)
class TableCellAnswer:
    evidence: SearchEvidence
    quote: str
    bbox: tuple[float, float, float, float]


def parse_table_question(question: str) -> TableQuestion | None:
    """Recognize one narrow, unambiguous row/column question form."""
    if not isinstance(question, str):
        return None
    match = _QUESTION.fullmatch(question)
    if match is None:
        return None
    column_label = match.group("column").strip()
    row_label = match.group("row").strip()
    if not _normalize(column_label) or not _normalize(row_label):
        return None
    return TableQuestion(column_label, row_label)


def find_table_cell_answer(
    question: TableQuestion,
    evidence_cells: Sequence[tuple[SearchEvidence, Sequence[PDFTableCellEvidence]]],
) -> TableCellAnswer | None:
    """Return a unique exact cell only when its header and row are both present."""
    grouped: dict[
        tuple[str, str, str, int, int],
        list[tuple[SearchEvidence, PDFTableCellEvidence]],
    ] = {}
    for evidence, cells in evidence_cells:
        if (
            not evidence.filename.lower().endswith(".pdf")
            or len(evidence.page_numbers) != 1
        ):
            continue
        for cell in cells:
            if (
                cell.page_number != evidence.page_numbers[0]
                or not cell.text.strip()
                or cell.text.strip() not in evidence.text
            ):
                continue
            table_key = (
                evidence.document_id,
                evidence.document_version_id,
                evidence.run_id,
                cell.page_number,
                cell.table_ordinal,
            )
            grouped.setdefault(table_key, []).append((evidence, cell))

    matches: list[TableCellAnswer] = []
    wanted_column = _normalize(question.column_label)
    wanted_row = _normalize(question.row_label)
    for entries in grouped.values():
        headers = [
            cell
            for _, cell in entries
            if _normalize(cell.text) == wanted_column
            and (
                cell.row_index == 0
                or (
                    cell.row_index == 1
                    and any(
                        parent.row_index == 0
                        and parent.bbox[0] <= cell.bbox[0] + 0.005
                        and parent.bbox[2] >= cell.bbox[2] - 0.005
                        and parent.bbox[2] - parent.bbox[0]
                        > cell.bbox[2] - cell.bbox[0] + 0.005
                        and abs(parent.bbox[3] - cell.bbox[1]) <= 0.005
                        for _, parent in entries
                    )
                )
            )
        ]
        header_columns = {cell.column_index for cell in headers}
        row_indexes = {
            cell.row_index
            for _, cell in entries
            if cell.row_index > 0 and _normalize(cell.text) == wanted_row
        }
        if len(header_columns) != 1 or len(row_indexes) != 1:
            continue
        column_index = next(iter(header_columns))
        row_index = next(iter(row_indexes))
        values: dict[str, TableCellAnswer] = {}
        for evidence, cell in entries:
            if cell.row_index != row_index or cell.column_index != column_index:
                continue
            if not any(
                header.row_index < cell.row_index
                and abs(header.bbox[0] - cell.bbox[0]) <= 0.005
                and abs(header.bbox[2] - cell.bbox[2]) <= 0.005
                for header in headers
            ):
                continue  # A spanning group header does not name its first child column.
            quote = cell.text.strip()
            if quote and quote in evidence.text:
                values[evidence.chunk_id] = TableCellAnswer(evidence, quote, cell.bbox)
        if len(values) == 1:
            matches.extend(values.values())
    return matches[0] if len(matches) == 1 else None


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())
