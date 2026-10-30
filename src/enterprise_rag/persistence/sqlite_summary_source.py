"""Complete bounded canonical scope, authorized in one SQLite read snapshot."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256

from enterprise_rag.application.search import SearchEvidence
from enterprise_rag.domain.summaries import SUMMARY_BUDGET, SummaryScope
from enterprise_rag.ingestion.chunking import LEGACY_CHUNKING_REVISION, chunk_text
from enterprise_rag.ingestion.ir import ParsedText, TextElement
from enterprise_rag.ports.document_summary import SummarySource


class SQLiteSummarySource:
    def __init__(self, metadata):
        self.metadata = metadata

    def read(
        self, actor, document_id, version_id, *, connection=None, historical_run_id=None
    ):
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None or actor.user_id is None:
            return None
        acl, parameters = self.metadata._acl_exists_clause(scope)
        with (
            self.metadata._connection()
            if connection is None
            else nullcontext(connection) as conn
        ):
            if connection is None:
                conn.execute("BEGIN")
            owner = conn.execute(
                """SELECT u.id,coalesce(r.revision,0) AS revision FROM users u
                   LEFT JOIN tenant_authorization_revisions r ON r.tenant_id=u.tenant_id
                   WHERE u.id=? AND u.tenant_id=?""",
                (actor.user_id, actor.tenant_id),
            ).fetchone()
            if owner is None or owner["revision"] != scope.authorization_revision:
                return None
            rows = conn.execute(
                f"""SELECT ir.execution_configuration_json,a.id,a.run_id,a.parser_revision,a.content_sha256,a.warnings,
                    length(a.normalized_text) AS chars,
                    length(CAST(a.normalized_text AS BLOB)) AS bytes,
                    (SELECT count(*) FROM text_chunks c WHERE c.run_id=a.run_id) AS chunks,
                    (SELECT count(*) FROM text_elements e WHERE e.run_id=a.run_id) AS elements,
                    coalesce((SELECT sum(length(c.text)) FROM text_chunks c WHERE c.run_id=a.run_id),0) AS chunk_chars,
                    coalesce((SELECT sum(length(CAST(c.text AS BLOB))) FROM text_chunks c WHERE c.run_id=a.run_id),0) AS chunk_bytes,
                    coalesce((SELECT sum(length(e.text)) FROM text_elements e WHERE e.run_id=a.run_id),0) AS element_chars,
                    coalesce((SELECT sum(length(CAST(e.text AS BLOB))) FROM text_elements e WHERE e.run_id=a.run_id),0) AS element_bytes
                FROM documents d JOIN document_versions v ON v.id=d.active_version_id
                JOIN ingestion_runs ir ON ir.document_version_id=v.id AND ir.tenant_id=d.tenant_id
                JOIN text_artifacts a ON a.run_id=ir.id AND a.document_version_id=v.id
                    AND a.content_sha256=v.content_sha256
                WHERE d.id=? AND v.id=? AND d.tenant_id=? AND v.document_id=d.id
                    AND v.status='active' AND ir.status='succeeded' AND (? IS NULL OR ir.id=?)
                    AND (ir.id=d.active_run_id OR (ir.id=? AND EXISTS (
                        SELECT 1 FROM stage_attempts sa WHERE sa.run_id=ir.id
                        AND sa.stage='activate' AND sa.status='succeeded')))
                    AND d.status NOT IN ('deleting','deleted') AND {acl} LIMIT 2""",
                (
                    document_id,
                    version_id,
                    actor.tenant_id,
                    historical_run_id,
                    historical_run_id,
                    historical_run_id,
                    *parameters,
                ),
            ).fetchall()
            if len(rows) != 1:
                return None
            row = rows[0]
            budget = SUMMARY_BUDGET
            oversized = (
                max(row["chars"], row["chunk_chars"], row["element_chars"])
                > budget.max_input_chars
                or max(row["bytes"], row["chunk_bytes"], row["element_bytes"])
                > budget.max_input_utf8_bytes
                or row["chunks"] > budget.max_chunks
                or row["elements"] > budget.max_elements
            )
            identity = {
                k: v
                for k, v in dict(row).items()
                if k != "execution_configuration_json"
            }
            source = SummaryScope(
                document_id,
                version_id,
                row["run_id"],
                sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                max(row["chars"], row["chunk_chars"], row["element_chars"]),
                max(row["bytes"], row["chunk_bytes"], row["element_bytes"]),
                row["chunks"],
                row["elements"],
                oversized,
            )
            if oversized:
                return SummarySource(source, (), "summary_scope_too_large")
            if json.loads(row["warnings"]):
                return SummarySource(source, (), "summary_source_warnings")
            chunks = conn.execute(
                "SELECT * FROM text_chunks WHERE run_id=? ORDER BY ordinal",
                (source.run_id,),
            ).fetchall()
            elements = conn.execute(
                "SELECT id,ordinal,element_type,text,page_number FROM text_elements WHERE run_id=? ORDER BY ordinal",
                (source.run_id,),
            ).fetchall()
            if not chunks or not elements:
                return SummarySource(source, (), "summary_empty_source")
            if (
                tuple(e["ordinal"] for e in elements) != tuple(range(len(elements)))
                or any("\x00" in e["text"] for e in elements)
                or any(not c["text"] or "\x00" in c["text"] for c in chunks)
            ):
                return SummarySource(source, (), "summary_source_incomplete")
            parsed = ParsedText(
                source.run_id,
                row["parser_revision"],
                row["content_sha256"],
                "",
                "",
                tuple(
                    TextElement(
                        e["id"],
                        source.run_id,
                        e["element_type"],
                        e["ordinal"],
                        e["text"],
                        "",
                        0,
                        0,
                        page_number=e["page_number"],
                    )
                    for e in elements
                ),
            )
            configuration = (
                json.loads(row["execution_configuration_json"])
                if row["execution_configuration_json"]
                else {}
            )
            expected = chunk_text(
                parsed,
                max_chars=max(len(c["text"]) for c in chunks),
                revision=configuration.get(
                    "chunking_revision", LEGACY_CHUNKING_REVISION
                ),
            )
            if tuple(asdict(item) for item in expected) != tuple(
                asdict(self.metadata._text_chunk_from_row(c)) for c in chunks
            ):
                return SummarySource(source, (), "summary_source_incomplete")
            evidence = []
            for chunk in chunks:
                live = self.metadata.get_authorized_evidence(
                    scope,
                    chunk["id"],
                    connection=conn,
                    historical=historical_run_id is not None,
                )
                if live is None or (
                    live.document_id,
                    live.document_version_id,
                    live.run_id,
                ) != (document_id, version_id, source.run_id):
                    return SummarySource(source, (), "summary_source_incomplete")
                evidence.append(
                    SearchEvidence(
                        live.chunk_id,
                        live.document_id,
                        live.document_version_id,
                        live.run_id,
                        live.title,
                        live.filename,
                        live.text,
                        live.score,
                        live.page_numbers,
                    )
                )
            # Hash all canonical chunks, including non-selected summary support.
            identity["canonical"] = [
                (e.id, e.text, e.page_number) for e in parsed.elements
            ]
            identity["chunks_complete"] = [asdict(item) for item in expected]
            source = SummaryScope(
                document_id,
                version_id,
                source.run_id,
                sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                source.source_chars,
                source.source_utf8_bytes,
                source.chunk_count,
                source.element_count,
                False,
            )
            return SummarySource(source, tuple(evidence))
