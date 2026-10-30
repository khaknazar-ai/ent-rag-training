"""SQLite DEV adapter with owner and live canonical-evidence read boundaries.

No provider runs in these transactions. No historical snippet/trace is stored;
text citations are retained privately and never substitute for current access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from enterprise_rag.application.answer import AnswerCitation, AnswerResult
from enterprise_rag.domain.model_summary import (
    ModelSummaryResult,
    SummaryParagraph,
    model_details_from_dict,
    validate_model_summary,
)
from enterprise_rag.domain.summaries import (
    SUMMARY_BUDGET,
    SummaryDetails,
    SummaryScope,
    summary_details_from_dict,
)
from enterprise_rag.persistence.sqlite_hierarchy import check_references
from enterprise_rag.persistence.sqlite_metadata import SQLiteMetadataStore
from enterprise_rag.persistence.sqlite_summary_source import SQLiteSummarySource
from enterprise_rag.ports.query_history import (
    QUERY_TRACE_STAGES,
    QueryCompletion,
    QueryHistoryForbidden,
    QueryHistoryPage,
    QueryHistoryUnavailable,
    QueryRequest,
    StoredQuery,
)


class SQLiteQueryHistory:
    def __init__(self, metadata: SQLiteMetadataStore):
        self.metadata = metadata

    @staticmethod
    def _owner(connection, scope, actor):
        # Resolve principals before entering the snapshot, then fence that
        # resolution against the revision and user row inside this transaction.
        if scope is None or actor.user_id is None:
            return False
        row = connection.execute(
            """SELECT u.id, coalesce(r.revision,0) AS revision FROM users u
               LEFT JOIN tenant_authorization_revisions r ON r.tenant_id=u.tenant_id
               WHERE u.id=? AND u.tenant_id=?""",
            (actor.user_id, actor.tenant_id),
        ).fetchone()
        return row is not None and row["revision"] == scope.authorization_revision

    def start(self, actor, request: QueryRequest) -> str:
        for text, maximum in (
            (request.query_text, 500),
            (request.retrieval_query, 500),
            (request.normalized_query, 10000),
            (request.retrieval_strategy, 80),
        ):
            if not isinstance(text, str) or not 1 <= len(text.strip()) <= maximum:
                raise ValueError("Invalid history request")
        configuration = request.model_configuration_json
        if len(configuration.encode("utf-8")) > 8192 or not isinstance(
            json.loads(configuration), dict
        ):
            raise ValueError("Invalid history configuration")
        query_id = uuid4().hex
        try:
            scope = self.metadata.resolve_security_scope(actor)
            with self.metadata._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if not self._owner(connection, scope, actor):
                    raise QueryHistoryForbidden("Registered query owner required")
                if request.scope is not None:
                    live = SQLiteSummarySource(self.metadata).read(
                        actor,
                        request.scope.document_id,
                        request.scope.document_version_id,
                        connection=connection,
                    )
                    if live is None or live.scope != request.scope:
                        raise QueryHistoryForbidden("Current summary scope required")
                connection.execute(
                    """INSERT INTO query_runs
                       (id,tenant_id,user_id,query_text,retrieval_query,normalized_query,retrieval_strategy,
                        model_configuration,configuration_sha256,status,started_at,scope_json)
                       VALUES (?,?,?,?,?,?,?,?,?, 'running',?,?)""",
                    (
                        query_id,
                        actor.tenant_id,
                        actor.user_id,
                        request.query_text,
                        request.retrieval_query,
                        request.normalized_query,
                        request.retrieval_strategy,
                        configuration,
                        hashlib.sha256(configuration.encode()).hexdigest(),
                        datetime.now(UTC).isoformat(),
                        json.dumps(asdict(request.scope), sort_keys=True)
                        if request.scope is not None
                        else None,
                    ),
                )
        except sqlite3.Error:
            raise QueryHistoryUnavailable() from None
        return query_id

    def complete(self, actor, query_run_id: str, outcome: QueryCompletion) -> None:
        elapsed = outcome.elapsed_seconds
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("Invalid history timing")
        stage_latencies = outcome.stage_latencies or {}
        if not isinstance(stage_latencies, dict) or set(stage_latencies) - set(
            QUERY_TRACE_STAGES
        ):
            raise ValueError("Invalid query trace stages")
        for stage, duration in stage_latencies.items():
            if (
                not isinstance(stage, str)
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
                or duration > elapsed + 0.001
            ):
                raise ValueError("Invalid query trace timing")
        stored_latencies = {"total_seconds": float(elapsed), **stage_latencies}
        encoded_latencies = json.dumps(
            stored_latencies, sort_keys=True, separators=(",", ":")
        )
        result = outcome.result
        if outcome.status not in {"answered", "abstained", "evidence_only", "failed"}:
            raise ValueError("Invalid history outcome")
        if (outcome.status == "failed") != (outcome.error_code is not None):
            raise ValueError("Invalid history failure")
        if outcome.status in {"failed", "evidence_only"}:
            if result is not None:
                raise ValueError("Non-answer outcome cannot contain an answer")
        elif isinstance(result, ModelSummaryResult):
            validate_model_summary(result)
            if (outcome.status == "abstained") != result.abstained:
                raise ValueError("Model summary status mismatch")
        elif not isinstance(result, AnswerResult) or (
            (outcome.status == "abstained") != result.abstained
            or type(result.abstained) is not bool
            or (result.abstained and (result.answer is not None or result.citations))
            or (
                not result.abstained
                and (not result.answer or not 1 <= len(result.citations) <= 4)
            )
            or (
                result.abstained
                and (
                    not isinstance(result.abstention_reason, str)
                    or re.fullmatch(r"[a-z0-9_]{1,128}", result.abstention_reason)
                    is None
                )
            )
            or (
                not result.abstained
                and (
                    result.abstention_reason is not None
                    or (
                        result.summary is None
                        and any(
                            cite.quote != result.answer for cite in result.citations
                        )
                    )
                    or (
                        result.summary is not None
                        and result.answer
                        != "\n\n".join(cite.quote for cite in result.citations)
                    )
                )
            )
        ):
            raise ValueError("Invalid stored answer")
        scope = self.metadata.resolve_security_scope(actor)
        with self.metadata._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._owner(connection, scope, actor):
                raise QueryHistoryForbidden("Registered query owner required")
            assert scope is not None
            existing = connection.execute(
                "SELECT status,scope_json,model_configuration,retrieval_strategy FROM query_runs WHERE id=? AND tenant_id=? AND user_id=?",
                (query_run_id, actor.tenant_id, actor.user_id),
            ).fetchone()
            if existing is None or existing["status"] != "running":
                raise ValueError("Query is absent or already finalized")
            refs = outcome.parent_context_refs
            encoded_refs = json.dumps(refs, sort_keys=True, separators=(",", ":"))
            if len(encoded_refs) > 8192 or not check_references(
                self.metadata, scope, refs, connection=connection
            ):
                raise ValueError(
                    "Parent context is not current authorized canonical scope"
                )
            connection.execute(
                "UPDATE query_runs SET parent_context_json=? WHERE id=?",
                (encoded_refs, query_run_id),
            )
            if result is not None:
                summary = result.summary
                is_model = isinstance(result, ModelSummaryResult)
                if is_model != (
                    existing["retrieval_strategy"] == "document-summary:model"
                ):
                    raise ValueError("Typed result does not match query strategy")
                if (summary is not None) != (existing["scope_json"] is not None):
                    raise ValueError("Summary scope/result mismatch")
                if summary is not None:
                    expected_scope = SummaryScope(**json.loads(existing["scope_json"]))
                    if isinstance(result, ModelSummaryResult):
                        validate_model_summary(result)
                        summary = result.summary
                        configuration = json.loads(existing["model_configuration"])[
                            "model"
                        ]
                        if (
                            summary.scope != expected_scope
                            or configuration.get("token_budget")
                            != (
                                asdict(summary.token_budget)
                                if summary.token_budget
                                else None
                            )
                            or configuration.get("budget_status")
                            != summary.budget_status
                        ):
                            raise ValueError("Model summary metadata/request mismatch")
                    elif (
                        not isinstance(summary, SummaryDetails)
                        or summary.scope != expected_scope
                        or summary.budget != SUMMARY_BUDGET
                        or type(summary.input_complete) is not bool
                        or summary.output_coverage != "selected_extracts_not_exhaustive"
                    ):
                        raise ValueError("Invalid summary metadata")
                    live = SQLiteSummarySource(self.metadata).read(
                        actor,
                        expected_scope.document_id,
                        expected_scope.document_version_id,
                        connection=connection,
                    )
                    if live is None or live.scope != expected_scope:
                        raise ValueError("Summary scope is no longer current")
                    if not result.abstained and (
                        not summary.input_complete
                        or expected_scope.oversized
                        or live.refusal_reason
                    ):
                        raise ValueError("Incomplete summary input")
            now = datetime.now(UTC).isoformat()
            if result is not None:
                model_payload = (
                    json.dumps(
                        {
                            "summary": asdict(result.summary),
                            "paragraphs": [asdict(p) for p in result.paragraphs],
                        },
                        sort_keys=True,
                    )
                    if isinstance(result, ModelSummaryResult)
                    else None
                )
                answer_id = "answer-" + query_run_id
                connection.execute(
                    """INSERT INTO query_answers
                       (id,query_run_id,tenant_id,user_id,answer_text,abstained,abstention_reason,created_at,summary_json,result_kind,model_summary_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        answer_id,
                        query_run_id,
                        actor.tenant_id,
                        actor.user_id,
                        result.answer,
                        int(result.abstained),
                        result.abstention_reason,
                        now,
                        json.dumps(asdict(result.summary), sort_keys=True)
                        if result.summary is not None and not is_model
                        else None,
                        "model_summary" if is_model else "extractive",
                        model_payload,
                    ),
                )
                for index, cite in enumerate(result.citations, 1):
                    live = self.metadata.get_authorized_evidence(
                        scope, cite.evidence_id, connection=connection
                    )
                    if live is None or not self._citation_matches(cite, live):
                        raise ValueError(
                            "Citation is not current authorized canonical evidence"
                        )
                    connection.execute(
                        """INSERT INTO query_citations
                           (answer_id,tenant_id,user_id,citation_index,evidence_type,chunk_id,
                            document_id,document_version_id,ingestion_run_id,title,filename,
                            page_numbers,quoted_span,start_char,end_char,offset_unit,source_sha256)
                           VALUES (?,?,?,?,'text_chunk',?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            answer_id,
                            actor.tenant_id,
                            actor.user_id,
                            index,
                            cite.evidence_id,
                            cite.document_id,
                            cite.document_version_id,
                            cite.run_id,
                            cite.title,
                            cite.filename,
                            json.dumps(cite.page_numbers),
                            cite.quote,
                            cite.chunk_start_char,
                            cite.chunk_end_char,
                            cite.offset_unit,
                            hashlib.sha256(live.text.encode()).hexdigest(),
                        ),
                    )
            connection.execute(
                """UPDATE query_runs SET status=?,finished_at=?,elapsed_seconds=?,error_code=?,
                   stage_latencies_json=?
                   WHERE id=? AND tenant_id=? AND user_id=? AND status='running'""",
                (
                    outcome.status,
                    now,
                    elapsed,
                    outcome.error_code,
                    encoded_latencies,
                    query_run_id,
                    actor.tenant_id,
                    actor.user_id,
                ),
            )

    @staticmethod
    def _citation_matches(cite, live):
        return (
            live is not None
            and cite.evidence_id == live.chunk_id
            and (
                cite.document_id,
                cite.document_version_id,
                cite.run_id,
                cite.title,
                cite.filename,
                cite.page_numbers,
            )
            == (
                live.document_id,
                live.document_version_id,
                live.run_id,
                live.title,
                live.filename,
                live.page_numbers,
            )
            and type(cite.chunk_start_char) is int
            and type(cite.chunk_end_char) is int
            and 0 <= cite.chunk_start_char < cite.chunk_end_char <= len(live.text)
            and live.text[cite.chunk_start_char : cite.chunk_end_char] == cite.quote
            and 1 <= len(cite.quote) <= 600
            and len(cite.page_numbers) <= 1
            and cite.offset_unit == "unicode_code_points"
        )

    def get(self, actor, query_run_id: str, *, connection=None) -> StoredQuery | None:
        if not isinstance(query_run_id, str) or len(query_run_id) != 32:
            return None
        try:
            scope = self.metadata.resolve_security_scope(actor)
            supplied = connection is not None
            with (
                nullcontext(connection) if supplied else self.metadata._connection()
            ) as connection:
                if not supplied:
                    connection.execute("BEGIN")
                if not self._owner(connection, scope, actor):
                    return None
                assert scope is not None
                row = connection.execute(
                    "SELECT * FROM query_runs WHERE id=? AND tenant_id=? AND user_id=?",
                    (query_run_id, actor.tenant_id, actor.user_id),
                ).fetchone()
                if row is None:
                    return None
                stored_scope = (
                    SummaryScope(**json.loads(row["scope_json"]))
                    if row["scope_json"] is not None
                    else None
                )
                if stored_scope is not None:
                    live_scope = SQLiteSummarySource(self.metadata).read(
                        actor,
                        stored_scope.document_id,
                        stored_scope.document_version_id,
                        connection=connection,
                        historical_run_id=stored_scope.run_id,
                    )
                    if live_scope is None or live_scope.scope != stored_scope:
                        return None
                answer = connection.execute(
                    "SELECT * FROM query_answers WHERE query_run_id=? AND tenant_id=? AND user_id=?",
                    (query_run_id, actor.tenant_id, actor.user_id),
                ).fetchone()
                if (answer is not None) != (row["status"] in {"answered", "abstained"}):
                    raise ValueError("Inconsistent history outcome")
                if answer is not None and bool(answer["abstained"]) != (
                    row["status"] == "abstained"
                ):
                    raise ValueError("Inconsistent history answer")
                result: AnswerResult | ModelSummaryResult | None = None
                if answer is not None:
                    citations = []
                    rows = connection.execute(
                        "SELECT * FROM query_citations WHERE answer_id=? AND tenant_id=? AND user_id=? ORDER BY citation_index LIMIT 5",
                        (answer["id"], actor.tenant_id, actor.user_id),
                    ).fetchall()
                    if len(rows) > 4:
                        raise ValueError("Invalid persisted citation count")
                    for citation in rows:
                        cite = AnswerCitation(
                            citation["chunk_id"],
                            citation["document_id"],
                            citation["document_version_id"],
                            citation["ingestion_run_id"],
                            citation["title"],
                            citation["filename"],
                            tuple(json.loads(citation["page_numbers"])),
                            citation["quoted_span"],
                            citation["start_char"],
                            citation["end_char"],
                            citation["offset_unit"],
                        )
                        live = self.metadata.get_authorized_evidence(
                            scope,
                            cite.evidence_id,
                            connection=connection,
                            historical=True,
                        )
                        if (
                            live is None
                            or not self._citation_matches(cite, live)
                            or hashlib.sha256(live.text.encode()).hexdigest()
                            != citation["source_sha256"]
                        ):
                            return None
                        citations.append(cite)
                    if not answer["abstained"] and not citations:
                        return None
                    if answer["result_kind"] == "model_summary":
                        if stored_scope is None or answer["summary_json"] is not None:
                            raise ValueError("Model summary has no canonical scope")
                        model_json = json.loads(answer["model_summary_json"])
                        result = ModelSummaryResult(
                            answer["answer_text"],
                            bool(answer["abstained"]),
                            answer["abstention_reason"],
                            tuple(citations),
                            model_details_from_dict(model_json["summary"]),
                            tuple(
                                SummaryParagraph(
                                    p["text"], tuple(p["citation_indexes"])
                                )
                                for p in model_json["paragraphs"]
                            ),
                            query_run_id,
                        )
                        validate_model_summary(result)
                        model_config = json.loads(row["model_configuration"])["model"]
                        if (
                            result.summary.scope != stored_scope
                            or model_config.get("token_budget")
                            != (
                                asdict(result.summary.token_budget)
                                if result.summary.token_budget
                                else None
                            )
                            or model_config.get("budget_status")
                            != result.summary.budget_status
                        ):
                            raise ValueError(
                                "Persisted model summary metadata mismatch"
                            )
                    else:
                        if answer["model_summary_json"] is not None:
                            raise ValueError(
                                "Legacy result contains model summary metadata"
                            )
                        result = AnswerResult(
                            answer["answer_text"],
                            bool(answer["abstained"]),
                            answer["abstention_reason"],
                            tuple(citations),
                            query_run_id,
                            summary_details_from_dict(
                                json.loads(answer["summary_json"])
                            )
                            if answer["summary_json"] is not None
                            else None,
                        )
                refs = (
                    tuple(json.loads(row["parent_context_json"]))
                    if row["parent_context_json"]
                    else ()
                )
                try:
                    parents_valid = check_references(
                        self.metadata,
                        scope,
                        refs,
                        connection=connection,
                        historical=True,
                    )
                except ValueError:
                    return None
                if not parents_valid:
                    return None
                configuration = row["model_configuration"]
                if (
                    hashlib.sha256(configuration.encode()).hexdigest()
                    != row["configuration_sha256"]
                ):
                    raise ValueError("History configuration checksum mismatch")
                if row["stage_latencies_json"] is not None:
                    stage_latencies = json.loads(row["stage_latencies_json"])
                    if (
                        not isinstance(stage_latencies, dict)
                        or set(stage_latencies)
                        - (set(QUERY_TRACE_STAGES) | {"total_seconds"})
                        or stage_latencies.get("total_seconds")
                        != row["elapsed_seconds"]
                        or any(
                            isinstance(duration, bool)
                            or not isinstance(duration, (int, float))
                            or not math.isfinite(duration)
                            or duration < 0
                            or duration > row["elapsed_seconds"] + 0.001
                            for duration in stage_latencies.values()
                        )
                    ):
                        raise ValueError("Invalid persisted query trace")
                else:
                    stage_latencies = (
                        {"total_seconds": row["elapsed_seconds"]}
                        if row["elapsed_seconds"] is not None
                        else None
                    )
                return StoredQuery(
                    query_run_id,
                    row["tenant_id"],
                    row["user_id"],
                    row["query_text"],
                    row["retrieval_query"],
                    row["normalized_query"],
                    row["retrieval_strategy"],
                    json.loads(configuration),
                    row["configuration_sha256"],
                    row["status"],
                    row["started_at"],
                    row["finished_at"],
                    stage_latencies,
                    result,
                    row["error_code"],
                    scope=stored_scope,
                    parent_context_refs=refs,
                )
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            raise QueryHistoryUnavailable(query_run_id) from None

    def list(self, actor, *, limit: int = 20, offset: int = 0) -> QueryHistoryPage:
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("History limit must be between 1 and 20")
        if type(offset) is not int or not 0 <= offset <= 100_000:
            raise ValueError("History offset is outside the allowed range")
        try:
            scope = self.metadata.resolve_security_scope(actor)
            with self.metadata._connection() as connection:
                connection.execute("BEGIN")
                if not self._owner(connection, scope, actor):
                    raise QueryHistoryForbidden("Registered query owner required")
                rows = connection.execute(
                    """SELECT id FROM query_runs WHERE tenant_id=? AND user_id=?
                       ORDER BY started_at DESC,id DESC LIMIT ? OFFSET ?""",
                    (actor.tenant_id, actor.user_id, limit + 1, offset),
                ).fetchall()
            page_ids = tuple(row["id"] for row in rows[:limit])
            # Each entry passes the same citation, version, ACL, and parent-reference
            # checks as an individual GET. Access changes between pages fail closed.
            items = tuple(
                item
                for query_run_id in page_ids
                if (item := self.get(actor, query_run_id)) is not None
            )
            next_offset = offset + len(page_ids) if len(rows) > limit else None
            return QueryHistoryPage(items, next_offset)
        except QueryHistoryForbidden:
            raise
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            raise QueryHistoryUnavailable() from None
