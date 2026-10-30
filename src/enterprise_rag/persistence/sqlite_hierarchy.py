"""Hierarchy storage/hydration inside the caller's fenced SQLite transaction."""

import json
import re
import sqlite3
from dataclasses import asdict
from hashlib import sha256

from enterprise_rag.domain.documents.ids import deterministic_id
from enterprise_rag.ingestion.chunking import TextChunk
from enterprise_rag.ingestion.hierarchy import (
    MAX_PARENT_NODES,
    build_hierarchy,
    parent_id,
)
from enterprise_rag.ingestion.ir import (
    ParsedText,
    TextElement,
    pdf_table_cells_from_payload,
)
from enterprise_rag.retrieval.parents import (
    ChildReference,
    ParentRecord,
    expand_parents,
)
from enterprise_rag.retrieval.parents import (
    parent_context_references as references,
)

MAX_PARENT_CONTEXT_CHARS = 12000


def assign_canonical(connection, parsed, chunks, nodes, links, *, write=False):
    run = connection.execute(
        "SELECT document_version_id,execution_configuration_json FROM ingestion_runs WHERE id=?",
        (parsed.run_id,),
    ).fetchone()
    config = (
        json.loads(run["execution_configuration_json"])
        if run["execution_configuration_json"]
        else {}
    )
    revision = config.get("chunking_revision", "canonical-section-slices-v2")
    offsets = []
    offset = 0
    for e in parsed.elements:
        offsets.append(offset)
        offset += len(e.text) + 2
    by_node = {n.id: n for n in nodes}
    by_chunk = {c.id: c for c in chunks}
    for link in links:
        chunk = by_chunk[link.chunk_id]
        owner = by_node[link.parent_id]
        start = offsets[owner.start_ordinal] + link.start_char
        end = start + len(chunk.text)
        digest = sha256(chunk.text.encode()).hexdigest()
        key = deterministic_id(
            "canonical-text-chunk",
            run["document_version_id"],
            revision,
            start,
            end,
            digest,
        )
        values = (key, run["document_version_id"], revision, start, end, digest)
        if write:
            connection.execute(
                "INSERT OR IGNORE INTO canonical_chunks VALUES(?,?,?,?,?,?)", values
            )
            connection.execute(
                "UPDATE text_chunks SET logical_chunk_id=? WHERE id=? AND logical_chunk_id IS NULL",
                (key, chunk.id),
            )
        row = connection.execute(
            "SELECT k.* FROM text_chunks c JOIN canonical_chunks k ON k.id=c.logical_chunk_id WHERE c.id=? AND c.run_id=?",
            (chunk.id, parsed.run_id),
        ).fetchone()
        if row is None or tuple(row) != values or digest != chunk.content_sha256:
            raise ValueError("Canonical logical chunk identity drift")


def persist(connection, artifact_id, parsed, chunks):
    nodes, links = build_hierarchy(parsed, chunks)
    assign_canonical(connection, parsed, chunks, nodes, links, write=True)
    expected_nodes = [
        (
            n.id,
            n.run_id,
            artifact_id,
            n.kind,
            n.parent_id,
            n.start_ordinal,
            n.end_ordinal,
            n.content_sha256,
            json.dumps(n.page_numbers, separators=(",", ":")),
        )
        for n in nodes
    ]
    expected_links = [tuple(asdict(link).values()) for link in links]
    existing = connection.execute(
        "SELECT * FROM hierarchy_parents WHERE run_id=? ORDER BY kind,start_ordinal",
        (parsed.run_id,),
    ).fetchall()
    if existing:
        if [tuple(r) for r in existing] != sorted(
            expected_nodes, key=lambda r: (r[3], r[5])
        ) or [
            tuple(r)
            for r in connection.execute(
                "SELECT * FROM hierarchy_children WHERE run_id=? ORDER BY chunk_id",
                (parsed.run_id,),
            )
        ] != sorted(expected_links):
            raise ValueError("Immutable hierarchy differs")
        return
    connection.executemany(
        "INSERT INTO hierarchy_parents VALUES(?,?,?,?,?,?,?,?,?)", expected_nodes
    )
    connection.executemany(
        "INSERT INTO hierarchy_children VALUES(?,?,?,?,?)", expected_links
    )


def verify(connection, run_id):
    """Compare the whole persisted structure with canonical IR, not mere existence."""
    artifact = connection.execute(
        "SELECT * FROM text_artifacts WHERE run_id=?", (run_id,)
    ).fetchone()
    if artifact is None:
        return False
    size = connection.execute(
        "SELECT coalesce(sum(length(text)),0) FROM text_elements WHERE run_id=?",
        (run_id,),
    ).fetchone()[0]
    if size > 50_000_000:
        return False
    elements = tuple(
        TextElement(
            r["id"],
            run_id,
            r["element_type"],
            r["ordinal"],
            r["text"],
            r["original_text"],
            r["start_line"],
            r["end_line"],
            tuple(json.loads(r["warnings"])),
            r["page_number"],
            pdf_table_cells_from_payload(json.loads(r["table_cells"])),
        )
        for r in connection.execute(
            "SELECT * FROM text_elements WHERE run_id=? ORDER BY ordinal", (run_id,)
        )
    )
    chunks = tuple(
        TextChunk(
            r["id"],
            run_id,
            r["ordinal"],
            r["text"],
            tuple(json.loads(r["element_ids"])),
            r["content_sha256"],
            tuple(json.loads(r["page_numbers"])),
        )
        for r in connection.execute(
            "SELECT * FROM text_chunks WHERE run_id=? ORDER BY ordinal", (run_id,)
        )
    )
    parsed = ParsedText(
        run_id,
        artifact["parser_revision"],
        artifact["content_sha256"],
        "",
        "",
        elements,
    )
    try:
        nodes, links = build_hierarchy(parsed, chunks)
    except (ValueError, KeyError):
        return False
    try:
        assign_canonical(connection, parsed, chunks, nodes, links)
    except ValueError:
        return False
    actual_nodes = [
        tuple(r)
        for r in connection.execute(
            "SELECT id,run_id,kind,parent_id,start_ordinal,end_ordinal,content_sha256,page_numbers FROM hierarchy_parents WHERE run_id=? ORDER BY kind,start_ordinal",
            (run_id,),
        )
    ]
    expected_nodes = sorted(
        [
            (
                n.id,
                n.run_id,
                n.kind,
                n.parent_id,
                n.start_ordinal,
                n.end_ordinal,
                n.content_sha256,
                json.dumps(n.page_numbers, separators=(",", ":")),
            )
            for n in nodes
        ],
        key=lambda r: (r[2], r[4]),
    )
    actual_links = [
        tuple(r)
        for r in connection.execute(
            "SELECT * FROM hierarchy_children WHERE run_id=? ORDER BY chunk_id",
            (run_id,),
        )
    ]
    return actual_nodes == expected_nodes and actual_links == sorted(
        tuple(asdict(link).values()) for link in links
    )


def _validate_ancestry(connection, selected):
    """Check canonical ancestry using bounded structural metadata, not parent text.

    Even a valid selected range must refuse a self/sibling/foreign ancestor or
    cycle. No oversized ancestor body is materialized for this check.
    """
    run_id = selected["run_id"]
    nodes = connection.execute(
        "SELECT * FROM hierarchy_parents WHERE run_id=? ORDER BY start_ordinal LIMIT ?",
        (run_id, MAX_PARENT_NODES + 1),
    ).fetchall()
    if not nodes or len(nodes) > MAX_PARENT_NODES:
        raise ValueError("Invalid hierarchy ancestry count")
    total = connection.execute(
        "SELECT count(*) FROM text_elements WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    headings = connection.execute(
        "SELECT ordinal,substr(original_text,1,8) AS prefix FROM text_elements WHERE run_id=? AND element_type='heading' ORDER BY ordinal LIMIT ?",
        (run_id, MAX_PARENT_NODES),
    ).fetchall()
    roots = [n for n in nodes if n["kind"] == "document"]
    if len(roots) != 1 or len(headings) + 1 != len(nodes):
        raise ValueError("Invalid canonical hierarchy root/section coverage")
    root = roots[0]
    expected = {root["id"]: (None, 0, total, "document")}
    sections = {n["start_ordinal"]: n for n in nodes if n["kind"] == "section"}
    if len(sections) != len(headings):
        raise ValueError("Invalid canonical section ordinals")
    stack: list[tuple[int, sqlite3.Row]] = []
    for heading in headings:
        match = re.match(r"^(#{1,6})\s+", heading["prefix"])
        if match is None:
            raise ValueError("Canonical heading metadata drift")
        depth = len(match[1])
        ordinal = heading["ordinal"]
        while stack and stack[-1][0] >= depth:
            _depth, previous = stack.pop()
            owner, start, _end, kind = expected[previous["id"]]
            expected[previous["id"]] = (owner, start, ordinal, kind)
        current = sections.get(ordinal)
        if current is None:
            raise ValueError("Canonical section missing")
        owner = stack[-1][1]["id"] if stack else root["id"]
        expected[current["id"]] = (owner, ordinal, total, "section")
        stack.append((depth, current))
    for node in nodes:
        actual = (
            node["parent_id"],
            node["start_ordinal"],
            node["end_ordinal"],
            node["kind"],
        )
        if (
            node["artifact_id"] != selected["artifact_id"]
            or expected.get(node["id"]) != actual
            or parent_id(
                run_id,
                node["kind"],
                node["start_ordinal"],
                node["end_ordinal"],
                node["content_sha256"],
            )
            != node["id"]
        ):
            raise ValueError("Canonical hierarchy ancestor drift")


def hydrate(
    metadata,
    scope,
    evidence,
    *,
    connection,
    max_context_chars=MAX_PARENT_CONTEXT_CHARS,
    historical=False,
):
    if (
        type(max_context_chars) is not int
        or not 1 <= max_context_chars <= MAX_PARENT_CONTEXT_CHARS
        or len(evidence) > 4
    ):
        raise ValueError("Invalid hierarchy context budget")
    owner = connection.execute(
        """SELECT u.id,coalesce(r.revision,0) AS revision FROM users u
           LEFT JOIN tenant_authorization_revisions r ON r.tenant_id=u.tenant_id
           WHERE u.id=? AND u.tenant_id=?""",
        (scope.user_id, scope.tenant_id),
    ).fetchone()
    if owner is None or owner["revision"] != scope.authorization_revision:
        raise ValueError("Current hierarchy actor or authorization revision changed")
    children = []
    parents = {}
    unavailable = set()
    seen = set()
    validated_runs = set()
    for rank, item in enumerate(evidence, 1):
        if item.chunk_id in seen:
            raise ValueError("Duplicate hierarchy child")
        seen.add(item.chunk_id)
        live = metadata.get_authorized_evidence(
            scope, item.chunk_id, connection=connection, historical=historical
        )
        if live is None or (
            live.document_id,
            live.document_version_id,
            live.run_id,
            live.text,
            live.logical_chunk_id,
        ) != (
            item.document_id,
            item.document_version_id,
            item.run_id,
            item.text,
            item.logical_chunk_id,
        ):
            raise ValueError("Hierarchy child is no longer current authorized evidence")
        row = connection.execute(
            """SELECT p.*,l.start_char,l.end_char FROM hierarchy_children l
            JOIN hierarchy_parents p ON p.id=l.parent_id AND p.run_id=l.run_id
            JOIN text_artifacts a ON a.id=p.artifact_id AND a.run_id=p.run_id
            WHERE l.chunk_id=? AND l.run_id=? AND a.document_version_id=?""",
            (item.chunk_id, item.run_id, item.document_version_id),
        ).fetchone()
        if row is None:
            # An old run with no hierarchy is explicitly unavailable, never reconstructed.
            raise ValueError("Persisted hierarchy is missing or invalid")
        if item.run_id not in validated_runs:
            _validate_ancestry(connection, row)
            validated_runs.add(item.run_id)
        expected_id = parent_id(
            item.run_id,
            row["kind"],
            row["start_ordinal"],
            row["end_ordinal"],
            row["content_sha256"],
        )
        if expected_id != row["id"]:
            raise ValueError("Hierarchy parent identity drift")
        bounds = connection.execute(
            """SELECT min(e.ordinal),max(e.ordinal) FROM text_chunks c,json_each(c.element_ids) ids
                JOIN text_elements e ON e.id=ids.value AND e.run_id=c.run_id
                WHERE c.id=? AND c.run_id=?""",
            (item.chunk_id, item.run_id),
        ).fetchone()
        nearest = connection.execute(
            """SELECT id FROM hierarchy_parents WHERE run_id=? AND start_ordinal<=?
               AND end_ordinal>? ORDER BY end_ordinal-start_ordinal,kind DESC LIMIT 1""",
            (item.run_id, bounds[0], bounds[1]),
        ).fetchone()
        if nearest is None or nearest[0] != row["id"]:
            raise ValueError("Hierarchy child mapped to a different structural parent")
        canonical = connection.execute(
            """SELECT k.*,r.execution_configuration_json FROM text_chunks c
               JOIN canonical_chunks k ON k.id=c.logical_chunk_id
               JOIN ingestion_runs r ON r.id=c.run_id WHERE c.id=? AND c.run_id=?""",
            (item.chunk_id, item.run_id),
        ).fetchone()
        prefix = connection.execute(
            "SELECT coalesce(sum(length(text)+2),0) FROM text_elements WHERE run_id=? AND ordinal<?",
            (item.run_id, row["start_ordinal"]),
        ).fetchone()[0]
        digest = sha256(item.text.encode()).hexdigest()
        revision = (
            json.loads(canonical["execution_configuration_json"]).get(
                "chunking_revision"
            )
            if canonical and canonical["execution_configuration_json"]
            else None
        )
        if (
            canonical is None
            or canonical["document_version_id"] != item.document_version_id
            or canonical["strategy_revision"] != revision
            or canonical["start_char"] != prefix + row["start_char"]
            or canonical["end_char"] != prefix + row["end_char"]
            or canonical["content_sha256"] != digest
            or canonical["id"]
            != deterministic_id(
                "canonical-text-chunk",
                item.document_version_id,
                revision,
                canonical["start_char"],
                canonical["end_char"],
                digest,
            )
        ):
            raise ValueError("Hierarchy canonical logical chunk identity drift")
        children.append(
            ChildReference(
                item.chunk_id,
                scope.tenant_id,
                item.document_id,
                item.document_version_id,
                item.run_id,
                row["id"],
                rank,
                item.score,
                item.page_numbers,
            )
        )
        length = connection.execute(
            """SELECT coalesce(sum(length(text)),0)+max(count(*)-1,0)*2,
            coalesce(sum(length(CAST(text AS BLOB))),0)+max(count(*)-1,0)*2,count(*)
            FROM text_elements WHERE run_id=? AND ordinal>=? AND ordinal<?""",
            (item.run_id, row["start_ordinal"], row["end_ordinal"]),
        ).fetchone()
        if length[0] > max_context_chars or length[1] > 4 * max_context_chars:
            unavailable.add(row["id"])
            continue
        elements = connection.execute(
            "SELECT id,ordinal,text,page_number FROM text_elements WHERE run_id=? AND ordinal>=? AND ordinal<? ORDER BY ordinal",
            (item.run_id, row["start_ordinal"], row["end_ordinal"]),
        ).fetchall()
        body = "\n\n".join(e["text"] for e in elements)
        pages = tuple(
            sorted({e["page_number"] for e in elements if e["page_number"] is not None})
        )
        ordinals = tuple(e["ordinal"] for e in elements)
        chunk_elements = connection.execute(
            "SELECT element_ids FROM text_chunks WHERE id=? AND run_id=?",
            (item.chunk_id, item.run_id),
        ).fetchone()
        ids = set(json.loads(chunk_elements[0]))
        if (
            ordinals != tuple(range(row["start_ordinal"], row["end_ordinal"]))
            or sha256(body.encode()).hexdigest() != row["content_sha256"]
            or pages != tuple(json.loads(row["page_numbers"]))
            or not ids.issubset({e["id"] for e in elements})
            or body[row["start_char"] : row["end_char"]] != item.text
        ):
            raise ValueError("Hierarchy canonical provenance drift")
        parents[row["id"]] = ParentRecord(
            row["id"],
            scope.tenant_id,
            item.document_id,
            item.document_version_id,
            item.run_id,
            body,
            pages,
        )
    return expand_parents(
        children,
        parents,
        tenant_id=scope.tenant_id,
        allowed_document_ids={e.document_id for e in evidence},
        active_run_ids={e.run_id for e in evidence},
        unavailable_parent_ids=unavailable,
        parent_limit=4,
        max_children_per_parent=4,
        max_context_chars=max_context_chars,
    )


def check_references(metadata, scope, refs, *, connection, historical=False):
    if len(refs) > 4:
        raise ValueError("Too many history parents")
    for ref in refs:
        if (
            set(ref)
            != {
                "parent_id",
                "tenant_id",
                "document_id",
                "document_version_id",
                "run_id",
                "content_sha256",
                "child_ids",
            }
            or ref["tenant_id"] != scope.tenant_id
            or not isinstance(ref["child_ids"], list)
            or not 1 <= len(ref["child_ids"]) <= 4
        ):
            raise ValueError("Invalid history parent reference")
        items = tuple(
            metadata.get_authorized_evidence(
                scope, c, connection=connection, historical=historical
            )
            for c in ref["child_ids"]
        )
        if any(i is None for i in items):
            return False
        expansion = hydrate(
            metadata, scope, items, connection=connection, historical=historical
        )
        if references(expansion) != (ref,):
            return False
    return True
