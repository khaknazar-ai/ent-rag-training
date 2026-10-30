"""SQLite metadata persistence for the Docker-free development profile."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from uuid import uuid4

from enterprise_rag.domain.documents.ids import deterministic_id, sha256_bytes
from enterprise_rag.domain.documents.lifecycle import (
    transition_document,
    transition_version,
)
from enterprise_rag.domain.documents.models import (
    Document,
    DocumentStatus,
    DocumentVersion,
    Group,
    Source,
    Tenant,
    User,
    VersionStatus,
)
from enterprise_rag.domain.ingestion.models import (
    IngestionJob,
    IngestionRun,
    JobLease,
    JobStatus,
    RunStatus,
    StageAttempt,
    StageStatus,
)
from enterprise_rag.ingestion.chunking import TextChunk
from enterprise_rag.ingestion.ir import (
    ParsedText,
    TextElement,
    pdf_chart_bars_from_payload,
    pdf_image_regions_from_payload,
    pdf_table_cells_from_payload,
)
from enterprise_rag.ingestion.ocr import (
    MAX_OCR_RUN_EVIDENCE_BYTES,
    OCRPageEvidence,
    OCRPageResult,
    OCRTextRegion,
)
from enterprise_rag.persistence.migrations import SQLITE_MIGRATIONS
from enterprise_rag.security import (
    ActorContext,
    DocumentPermission,
    Permission,
    PrincipalType,
    SecurityScope,
)


@contextmanager
def _lexical_read_budget(connection: sqlite3.Connection, vm_steps: int | None):
    """Connection-local cooperative query budget, never a global SQLite setting."""
    if vm_steps is None:
        yield
        return
    if type(vm_steps) is not int or not 1000 <= vm_steps <= 1_000_000:
        raise ValueError("Lexical VM budget must be between 1000 and1000000")
    remaining = vm_steps
    deadline = time.monotonic() + 0.5

    def interrupted():
        nonlocal remaining
        remaining -= 1000
        return int(remaining <= 0 or time.monotonic() >= deadline)

    connection.set_progress_handler(interrupted, 1000)
    try:
        yield
    finally:
        connection.set_progress_handler(None, 0)


def _ocr_page_evidence_json(evidence: OCRPageEvidence) -> str:
    result = evidence.result
    return json.dumps(
        {
            "page_number": evidence.page_number,
            "page_image_sha256": evidence.page_image_sha256,
            "page_width": evidence.page_width,
            "page_height": evidence.page_height,
            "applied_rotation_degrees": evidence.applied_rotation_degrees,
            "renderer_id": evidence.renderer_id,
            "renderer_revision": evidence.renderer_revision,
            "result": {
                "run_id": result.run_id,
                "document_version_id": result.document_version_id,
                "source_pdf_sha256": result.source_pdf_sha256,
                "page_number": result.page_number,
                "image_sha256": result.image_sha256,
                "provider_id": result.provider_id,
                "provider_revision": result.provider_revision,
                "raw_text": result.raw_text,
                "regions": [
                    {
                        "ordinal": region.ordinal,
                        "start_char": region.start_char,
                        "end_char": region.end_char,
                        "text": region.text,
                        "bbox": list(region.bbox),
                        "confidence": region.confidence,
                    }
                    for region in result.regions
                ],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _ocr_page_evidence_from_json(payload: str) -> OCRPageEvidence:
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or set(data) != {
            "page_number",
            "page_image_sha256",
            "page_width",
            "page_height",
            "applied_rotation_degrees",
            "renderer_id",
            "renderer_revision",
            "result",
        }:
            raise ValueError
        result_data = data["result"]
        if not isinstance(result_data, dict) or set(result_data) != {
            "run_id",
            "document_version_id",
            "source_pdf_sha256",
            "page_number",
            "image_sha256",
            "provider_id",
            "provider_revision",
            "raw_text",
            "regions",
        }:
            raise ValueError
        region_rows = result_data["regions"]
        if not isinstance(region_rows, list):
            raise ValueError
        regions = []
        for row in region_rows:
            if not isinstance(row, dict) or set(row) != {
                "ordinal",
                "start_char",
                "end_char",
                "text",
                "bbox",
                "confidence",
            }:
                raise ValueError
            bbox = row["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError
            regions.append(
                OCRTextRegion(
                    ordinal=row["ordinal"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    text=row["text"],
                    bbox=tuple(bbox),
                    confidence=row["confidence"],
                )
            )
        result = OCRPageResult(
            run_id=result_data["run_id"],
            document_version_id=result_data["document_version_id"],
            source_pdf_sha256=result_data["source_pdf_sha256"],
            page_number=result_data["page_number"],
            image_sha256=result_data["image_sha256"],
            provider_id=result_data["provider_id"],
            provider_revision=result_data["provider_revision"],
            raw_text=result_data["raw_text"],
            regions=tuple(regions),
        )
        return OCRPageEvidence(
            page_number=data["page_number"],
            page_image_sha256=data["page_image_sha256"],
            page_width=data["page_width"],
            page_height=data["page_height"],
            applied_rotation_degrees=data["applied_rotation_degrees"],
            renderer_id=data["renderer_id"],
            renderer_revision=data["renderer_revision"],
            result=result,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Stored OCR page evidence is invalid") from error


def _ocr_page_evidence_from_row(row: sqlite3.Row) -> OCRPageEvidence:
    payload = row["payload_json"]
    if sha256_bytes(payload.encode("utf-8")) != row["payload_sha256"]:
        raise ValueError("Stored OCR page evidence checksum is invalid")
    evidence = _ocr_page_evidence_from_json(payload)
    if (evidence.result.run_id, evidence.page_number) != (
        row["run_id"],
        row["page_number"],
    ):
        raise ValueError("Stored OCR page evidence identity is invalid")
    return evidence


@dataclass(frozen=True, slots=True)
class UploadDecision:
    """Result of the atomic metadata half of upload acceptance."""

    document: Document
    version: DocumentVersion
    job: IngestionJob | None
    duplicate: bool


@dataclass(frozen=True, slots=True)
class LexicalRow:
    """Authorized metadata returned by a tenant-scoped lexical query."""

    chunk_id: str
    document_id: str
    document_version_id: str
    run_id: str
    title: str
    filename: str
    text: str
    score: float
    page_numbers: tuple[int, ...] = ()
    logical_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorizedParentContext:
    """ACL-filtered source-element context hydrated for one child chunk."""

    parent_id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    run_id: str
    text: str | None
    page_numbers: tuple[int, ...]
    oversized: bool


@dataclass(frozen=True, slots=True)
class DeletionManifest:
    """Exact metadata-owned resources eligible for one tombstoned document."""

    document_id: str
    version_ids: tuple[str, ...]
    object_keys: tuple[str, ...]
    run_ids: tuple[str, ...]
    active_job_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentDeletionClaim:
    """Durable owner of one exact tombstoned-document cleanup pass."""

    tenant_id: str
    document_id: str
    claim_token: str
    expires_at: float
    manifest: DeletionManifest


class SharedDeletionObjectKeyError(ValueError):
    """A tombstone's raw key is also referenced by another document."""


@dataclass(frozen=True, slots=True)
class UploadAttempt:
    """Persisted ownership metadata for a private raw upload attempt."""

    object_key: str
    tenant_id: str
    document_id: str
    status: str
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class CleanupClaim:
    """A short-lived DB claim with an opaque owner token."""

    object_key: str
    tenant_id: str
    document_id: str
    claim_token: str


@dataclass(frozen=True, slots=True)
class IngestionAuditFinding:
    """One tenant-scoped consistency finding without document contents."""

    code: str
    severity: str
    job_id: str | None = None
    run_id: str | None = None
    document_version_id: str | None = None
    stage: str | None = None


@dataclass(frozen=True, slots=True)
class StageAttemptReconciliation:
    """Bounded result of marking only provably stale stage attempts."""

    candidate_attempt_ids: tuple[str, ...]
    repaired_attempt_ids: tuple[str, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class TerminalRunReconciliation:
    """Bounded result of repairing only terminal-job run mismatches."""

    candidate_run_ids: tuple[str, ...]
    repaired_run_ids: tuple[str, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class MaintenanceLease:
    """Fenced, tenant-scoped lease held by one maintenance invocation."""

    tenant_id: str
    owner_id: str
    lease_token: int
    expires_at: float


class SQLiteMetadataStore:
    """Small transactional metadata store with an explicit schema."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied = {
                int(row["version"]): row["checksum"]
                for row in connection.execute(
                    "SELECT version, checksum FROM schema_migrations"
                ).fetchall()
            }
            known_versions = {migration.version for migration in SQLITE_MIGRATIONS}
            unknown_versions = set(applied) - known_versions
            if unknown_versions:
                raise RuntimeError(
                    f"Database has unknown schema versions: {sorted(unknown_versions)}"
                )

            for migration in SQLITE_MIGRATIONS:
                stored_checksum = applied.get(migration.version)
                if stored_checksum is not None:
                    if stored_checksum != migration.checksum:
                        raise RuntimeError(
                            f"Migration {migration.version} checksum mismatch"
                        )
                    continue
                if migration.version > 1 and migration.version - 1 not in applied:
                    raise RuntimeError(
                        f"Cannot apply migration {migration.version} before "
                        f"{migration.version - 1}"
                    )
                # SQLite's documented table rebuild needs FK enforcement off only
                # on this migration connection, outside the transaction. Validate
                # every FK before commit and restore enforcement on every exit.
                rebuild = migration.rebuild_referenced_tables
                if rebuild:
                    connection.execute("PRAGMA foreign_keys = OFF")
                try:
                    connection.executescript(
                        f"""
                        BEGIN;
                        {migration.sql}
                        INSERT INTO schema_migrations (version, checksum)
                        VALUES ({migration.version!r}, {migration.checksum!r});
                        """
                    )
                    if (
                        rebuild
                        and connection.execute("PRAGMA foreign_key_check").fetchone()
                    ):
                        raise RuntimeError("Migration foreign key validation failed")
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    if rebuild:
                        connection.execute("PRAGMA foreign_keys = ON")
                applied[migration.version] = migration.checksum

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _assert_object_key_available_for_upload(
        connection: sqlite3.Connection,
        object_key: str,
        *,
        document_id: str,
        allow_existing_same_document: bool = False,
    ) -> None:
        row = connection.execute(
            "SELECT 1 FROM object_cleanup_claims WHERE object_key = ?",
            (object_key,),
        ).fetchone()
        if row is not None:
            raise ValueError("Object key is reserved by cleanup reconciliation")
        row = connection.execute(
            """SELECT 1 FROM document_versions
               WHERE object_key = ? AND (? = 0 OR document_id != ?)
               LIMIT 1""",
            (object_key, int(allow_existing_same_document), document_id),
        ).fetchone()
        if row is not None:
            raise ValueError("Object key is already referenced by a document version")
        if SQLiteMetadataStore._table_exists(connection, "citation_visual_regions"):
            row = connection.execute(
                "SELECT 1 FROM citation_visual_regions WHERE object_key=? LIMIT 1",
                (object_key,),
            ).fetchone()
            if row is not None:
                raise ValueError("Object key is already referenced by a visual region")
        row = connection.execute(
            """SELECT 1 FROM upload_attempts
               WHERE object_key = ? AND document_id != ? LIMIT 1""",
            (object_key, document_id),
        ).fetchone()
        if row is not None:
            raise ValueError("Object key is already owned by another upload attempt")

    @staticmethod
    def _object_key_is_referenced(
        connection: sqlite3.Connection, object_key: str
    ) -> bool:
        row = connection.execute(
            "SELECT 1 FROM document_versions WHERE object_key = ? LIMIT 1",
            (object_key,),
        ).fetchone()
        if row is not None:
            return True
        if not SQLiteMetadataStore._table_exists(connection, "citation_visual_regions"):
            return False
        return (
            connection.execute(
                "SELECT 1 FROM citation_visual_regions WHERE object_key = ? LIMIT 1",
                (object_key,),
            ).fetchone()
            is not None
        )

    def ensure_tenant(self, tenant: Tenant) -> Tenant:
        """Register a tenant idempotently before creating its principals."""

        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tenants (id, name) VALUES (?, ?)",
                (tenant.id, tenant.name),
            )
        return tenant

    def get_document(self, document_id: str) -> Document | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def get_user_for_tenant(self, tenant_id: str, user_id: str) -> User | None:
        """Read a user only through the caller's tenant boundary."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ? AND tenant_id = ?",
                (user_id, tenant_id),
            ).fetchone()
        return self._user_from_row(row) if row else None

    def get_group_for_tenant(self, tenant_id: str, group_id: str) -> Group | None:
        """Read a group only through the caller's tenant boundary."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM groups WHERE id = ? AND tenant_id = ?",
                (group_id, tenant_id),
            ).fetchone()
        return self._group_from_row(row) if row else None

    def get_document_for_tenant(
        self,
        tenant_id: str,
        document_id: str,
        *,
        actor: ActorContext | None = None,
    ) -> Document | None:
        """Read a document only when it belongs to the caller's tenant."""

        with self._connection() as connection:
            if actor is not None:
                connection.execute("BEGIN")
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            ).fetchone()
        return self._document_from_row(row) if row else None

    def get_version(self, version_id: str) -> DocumentVersion | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM document_versions WHERE id = ?",
                (version_id,),
            ).fetchone()
        return self._version_from_row(row) if row else None

    def get_version_for_tenant(
        self, tenant_id: str, version_id: str
    ) -> DocumentVersion | None:
        """Read a version only through its tenant-owned document."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT v.*
                FROM document_versions AS v
                JOIN documents AS d ON d.id = v.document_id
                WHERE v.id = ? AND d.tenant_id = ?
                """,
                (version_id, tenant_id),
            ).fetchone()
        return self._version_from_row(row) if row else None

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(row) if row else None

    def get_job_for_tenant(self, tenant_id: str, job_id: str) -> IngestionJob | None:
        """Read a job only when its tenant matches the caller's tenant."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def get_job_for_version(self, version_id: str) -> IngestionJob | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE document_version_id = ? ORDER BY rowid DESC LIMIT 1",
                (version_id,),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def get_job_for_version_for_tenant(
        self, tenant_id: str, version_id: str
    ) -> IngestionJob | None:
        """Read a version's job only after checking the owning tenant."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT j.*
                FROM ingestion_jobs AS j
                JOIN document_versions AS v
                    ON v.id = j.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                WHERE j.document_version_id = ? AND d.tenant_id = ?
                ORDER BY j.rowid DESC LIMIT 1
                """,
                (version_id, tenant_id),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list_documents_authorized(
        self, scope: SecurityScope, *, limit: int = 50, offset: int = 0
    ) -> tuple[Document, ...]:
        """List only ACL-visible documents, applying ACL before pagination."""

        self._validate_management_bounds(limit, offset)
        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT d.*,
                    COALESCE((
                        SELECT a.warnings
                        FROM text_artifacts AS a
                        JOIN ingestion_runs AS ir ON ir.id = a.run_id
                        WHERE a.run_id = d.active_run_id
                            AND a.document_version_id = d.active_version_id
                            AND ir.document_version_id = d.active_version_id
                            AND ir.tenant_id = d.tenant_id
                    ), '[]') AS ingestion_warnings
                FROM documents AS d
                WHERE d.tenant_id = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                ORDER BY d.canonical_key ASC, d.id ASC
                LIMIT ? OFFSET ?
                """,
                (
                    scope.tenant_id,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                    limit,
                    offset,
                ),
            ).fetchall()
        return tuple(self._document_from_row(row) for row in rows)

    def get_document_authorized(
        self, scope: SecurityScope, document_id: str
    ) -> Document | None:
        """Return a document only when the actor has a document read grant."""

        if not document_id.strip():
            return None
        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT d.*,
                    COALESCE((
                        SELECT a.warnings
                        FROM text_artifacts AS a
                        JOIN ingestion_runs AS ir ON ir.id = a.run_id
                        WHERE a.run_id = d.active_run_id
                            AND a.document_version_id = d.active_version_id
                            AND ir.document_version_id = d.active_version_id
                            AND ir.tenant_id = d.tenant_id
                    ), '[]') AS ingestion_warnings
                FROM documents AS d
                WHERE d.id = ?
                    AND d.tenant_id = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                """,
                (
                    document_id,
                    scope.tenant_id,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                ),
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_versions_authorized(
        self, scope: SecurityScope, document_id: str
    ) -> tuple[DocumentVersion, ...]:
        """Return versions only through an ACL-visible parent document."""

        if not document_id.strip():
            return ()
        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT v.*
                FROM document_versions AS v
                JOIN documents AS d ON d.id = v.document_id
                WHERE v.document_id = ?
                    AND d.tenant_id = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                ORDER BY v.version_number DESC, v.id DESC
                """,
                (
                    document_id,
                    scope.tenant_id,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                ),
            ).fetchall()
        return tuple(self._version_from_row(row) for row in rows)

    def get_job_authorized(
        self, scope: SecurityScope, job_id: str
    ) -> IngestionJob | None:
        """Return job state without exposing lease data across an ACL boundary."""

        if not job_id.strip():
            return None
        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT j.*
                FROM ingestion_jobs AS j
                JOIN document_versions AS v
                    ON v.id = j.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                WHERE j.id = ?
                    AND j.tenant_id = ?
                    AND d.tenant_id = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                """,
                (
                    job_id,
                    scope.tenant_id,
                    scope.tenant_id,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                ),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def get_run(self, run_id: str) -> IngestionRun | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row) if row else None

    def acquire_maintenance_lease(
        self,
        tenant_id: str,
        owner_id: str,
        *,
        now: float,
        lease_seconds: float,
        actor: ActorContext | None = None,
    ) -> MaintenanceLease | None:
        """Acquire one tenant maintenance lease, reclaiming only expired state."""

        if not tenant_id.strip() or not owner_id.strip():
            raise ValueError("Tenant and maintenance owner ids must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("Maintenance lease duration must be positive")
        expires_at = now + lease_seconds
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            current = connection.execute(
                """
                SELECT *
                FROM maintenance_leases
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
            if current is not None and current["expires_at"] > now:
                return None
            counter = connection.execute(
                "SELECT lease_token FROM maintenance_lease_counters WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            next_token = (
                max(
                    0 if counter is None else counter["lease_token"],
                    0 if current is None else current["lease_token"],
                )
                + 1
            )
            connection.execute(
                """
                INSERT INTO maintenance_lease_counters (tenant_id, lease_token)
                VALUES (?, ?)
                ON CONFLICT (tenant_id) DO UPDATE SET lease_token = excluded.lease_token
                """,
                (tenant_id, next_token),
            )
            if current is None:
                connection.execute(
                    """
                    INSERT INTO maintenance_leases
                        (tenant_id, owner_id, lease_token, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (tenant_id, owner_id, next_token, expires_at),
                )
            else:
                connection.execute(
                    """
                    UPDATE maintenance_leases
                    SET owner_id = ?, lease_token = ?, expires_at = ?
                    WHERE tenant_id = ? AND expires_at <= ?
                    """,
                    (owner_id, next_token, expires_at, tenant_id, now),
                )
            row = connection.execute(
                "SELECT * FROM maintenance_leases WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return MaintenanceLease(
            tenant_id=row["tenant_id"],
            owner_id=row["owner_id"],
            lease_token=row["lease_token"],
            expires_at=row["expires_at"],
        )

    def release_maintenance_lease(self, lease: MaintenanceLease) -> bool:
        """Release only the exact current owner/token pair."""

        with self._connection() as connection:
            deleted = connection.execute(
                """
                DELETE FROM maintenance_leases
                WHERE tenant_id = ? AND owner_id = ? AND lease_token = ?
                """,
                (lease.tenant_id, lease.owner_id, lease.lease_token),
            )
        return deleted.rowcount == 1

    def renew_maintenance_lease(
        self,
        lease: MaintenanceLease,
        *,
        now: float,
        lease_seconds: float,
    ) -> MaintenanceLease | None:
        """Extend a live lease without changing its fencing token."""

        if lease_seconds <= 0:
            raise ValueError("Maintenance lease duration must be positive")
        expires_at = now + lease_seconds
        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE maintenance_leases
                SET expires_at = ?
                WHERE tenant_id = ?
                    AND owner_id = ?
                    AND lease_token = ?
                    AND expires_at > ?
                """,
                (
                    expires_at,
                    lease.tenant_id,
                    lease.owner_id,
                    lease.lease_token,
                    now,
                ),
            )
            if updated.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM maintenance_leases WHERE tenant_id = ?",
                (lease.tenant_id,),
            ).fetchone()
        return MaintenanceLease(
            tenant_id=row["tenant_id"],
            owner_id=row["owner_id"],
            lease_token=row["lease_token"],
            expires_at=row["expires_at"],
        )

    def audit_ingestion(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        now: float | None = None,
        actor: ActorContext | None = None,
    ) -> tuple[tuple[IngestionAuditFinding, ...], bool]:
        """Inspect bounded ingestion consistency without mutating state.

        An expired worker lease is a restart/recovery signal only; it does not
        prove that the previous executor has stopped.
        """

        if not tenant_id.strip():
            raise ValueError("Tenant id must be non-empty")
        if limit <= 0 or limit > 1000:
            raise ValueError("Audit limit must be between 1 and 1000")
        current_time = time.time() if now is None else now

        findings: list[IngestionAuditFinding] = []
        with self._connection() as connection:
            if actor is not None:
                connection.execute("BEGIN")
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            rows = connection.execute(
                """
                WITH candidates AS (
                    SELECT
                        'job_missing_run' AS code,
                        'error' AS severity,
                        j.id AS job_id,
                        NULL AS run_id,
                        j.document_version_id AS document_version_id,
                        NULL AS stage
                    FROM ingestion_jobs AS j
                    JOIN document_versions AS v ON v.id = j.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    WHERE d.tenant_id = ? AND j.run_id IS NULL

                    UNION ALL

                    SELECT
                        'job_run_missing', 'error', j.id, j.run_id,
                        j.document_version_id, NULL
                    FROM ingestion_jobs AS j
                    JOIN document_versions AS v ON v.id = j.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    LEFT JOIN ingestion_runs AS r ON r.id = j.run_id
                    WHERE d.tenant_id = ?
                        AND j.run_id IS NOT NULL
                        AND r.id IS NULL

                    UNION ALL

                    SELECT
                        'job_run_identity_mismatch', 'error', j.id, j.run_id,
                        j.document_version_id, NULL
                    FROM ingestion_jobs AS j
                    JOIN document_versions AS v ON v.id = j.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    JOIN ingestion_runs AS r ON r.id = j.run_id
                    WHERE d.tenant_id = ?
                        AND (
                            r.tenant_id != j.tenant_id
                            OR r.document_version_id != j.document_version_id
                        )

                    UNION ALL

                    SELECT
                        'job_run_status_mismatch', 'error', j.id, r.id,
                        j.document_version_id, j.stage
                    FROM ingestion_jobs AS j
                    JOIN document_versions AS v ON v.id = j.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    JOIN ingestion_runs AS r ON r.id = j.run_id
                    WHERE d.tenant_id = ?
                        AND r.tenant_id = j.tenant_id
                        AND r.document_version_id = j.document_version_id
                        AND (
                            (j.status = 'queued' AND r.status != 'queued')
                            OR (j.status = 'running' AND r.status != 'running')
                            OR (j.status = 'retry_wait' AND r.status != 'running')
                            OR (j.status = 'succeeded' AND r.status != 'succeeded')
                            OR (j.status = 'failed' AND r.status != 'failed')
                            OR (j.status = 'cancelled' AND r.status != 'cancelled')
                        )

                    UNION ALL

                    SELECT
                        'expired_worker_lease', 'warning', j.id, j.run_id,
                        j.document_version_id, j.stage
                    FROM ingestion_jobs AS j
                    JOIN document_versions AS v ON v.id = j.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    WHERE d.tenant_id = ?
                        AND j.tenant_id = d.tenant_id
                        AND j.status = 'running'
                        AND j.lease_owner IS NOT NULL
                        AND j.lease_expires_at IS NOT NULL
                        AND j.lease_expires_at <= ?

                    UNION ALL

                    SELECT
                        'run_without_job', 'error', NULL, r.id,
                        r.document_version_id, NULL
                    FROM ingestion_runs AS r
                    JOIN document_versions AS v ON v.id = r.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    LEFT JOIN ingestion_jobs AS j ON j.run_id = r.id
                    WHERE d.tenant_id = ? AND j.id IS NULL

                    UNION ALL

                    SELECT
                        'active_job_on_tombstoned_document', 'error', j.id,
                        j.run_id, j.document_version_id, j.stage
                    FROM ingestion_jobs AS j
                    JOIN document_versions AS v ON v.id = j.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    WHERE d.tenant_id = ?
                        AND d.status IN ('deleting', 'deleted')
                        AND j.status IN ('queued', 'running', 'retry_wait')

                    UNION ALL

                    SELECT
                        'stage_job_run_mismatch', 'error', a.job_id, a.run_id,
                        r.document_version_id, a.stage
                    FROM stage_attempts AS a
                    JOIN ingestion_runs AS r ON r.id = a.run_id
                    JOIN document_versions AS v ON v.id = r.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    LEFT JOIN ingestion_jobs AS j ON j.id = a.job_id
                    WHERE d.tenant_id = ?
                        AND (
                            j.id IS NULL
                            OR j.tenant_id != r.tenant_id
                            OR j.run_id IS NULL
                            OR j.run_id != a.run_id
                            OR j.document_version_id != r.document_version_id
                        )

                    UNION ALL

                    SELECT
                        'running_stage_without_running_job', 'warning', a.job_id,
                        a.run_id, r.document_version_id, a.stage
                    FROM stage_attempts AS a
                    JOIN ingestion_runs AS r ON r.id = a.run_id
                    JOIN document_versions AS v ON v.id = r.document_version_id
                    JOIN documents AS d ON d.id = v.document_id
                    LEFT JOIN ingestion_jobs AS j ON j.id = a.job_id
                    WHERE d.tenant_id = ?
                        AND a.status = 'running'
                        AND (j.id IS NULL OR j.status != 'running')
                )
                SELECT code, severity, job_id, run_id, document_version_id, stage
                FROM candidates
                ORDER BY code, job_id, run_id, document_version_id, stage
                LIMIT ?
                """,
                (
                    tenant_id,
                    tenant_id,
                    tenant_id,
                    tenant_id,
                    tenant_id,
                    current_time,
                    tenant_id,
                    tenant_id,
                    tenant_id,
                    tenant_id,
                    limit + 1,
                ),
            ).fetchall()
            findings.extend(self._audit_finding_from_row(row) for row in rows)
            truncated = len(rows) > limit

            succeeded_runs = connection.execute(
                """
                SELECT r.id, r.document_version_id, r.required_stages, j.id AS job_id
                FROM ingestion_runs AS r
                JOIN document_versions AS v ON v.id = r.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                JOIN ingestion_jobs AS j ON j.run_id = r.id
                WHERE d.tenant_id = ?
                    AND r.status = 'succeeded'
                    AND j.status = 'succeeded'
                ORDER BY r.id
                LIMIT ?
                """,
                (tenant_id, limit + 1),
            ).fetchall()
            if len(succeeded_runs) > limit:
                truncated = True

            for run_row in succeeded_runs:
                try:
                    required_stages = json.loads(run_row["required_stages"])
                except (TypeError, json.JSONDecodeError):
                    findings.append(
                        IngestionAuditFinding(
                            code="run_required_stages_invalid",
                            severity="error",
                            run_id=run_row["id"],
                            document_version_id=run_row["document_version_id"],
                        )
                    )
                    continue
                if not isinstance(required_stages, list) or not all(
                    isinstance(stage, str) and stage.strip()
                    for stage in required_stages
                ):
                    findings.append(
                        IngestionAuditFinding(
                            code="run_required_stages_invalid",
                            severity="error",
                            run_id=run_row["id"],
                            document_version_id=run_row["document_version_id"],
                        )
                    )
                    continue
                succeeded_stages = self._latest_succeeded_stages(
                    connection, run_row["id"], run_row["job_id"]
                )
                for stage in required_stages:
                    if stage not in succeeded_stages:
                        findings.append(
                            IngestionAuditFinding(
                                code="succeeded_run_missing_stage",
                                severity="error",
                                run_id=run_row["id"],
                                document_version_id=run_row["document_version_id"],
                                stage=stage,
                            )
                        )

        findings.sort(
            key=lambda finding: (
                finding.code,
                finding.job_id or "",
                finding.run_id or "",
                finding.document_version_id or "",
                finding.stage or "",
            )
        )
        if len(findings) > limit:
            truncated = True
            findings = findings[:limit]
        return tuple(findings), truncated

    def tombstone_document(
        self,
        tenant_id: str,
        document_id: str,
        *,
        actor: ActorContext | None = None,
    ) -> tuple[Document | None, bool]:
        """Hide one document and fence its in-flight ingestion atomically."""

        if not tenant_id.strip() or not document_id.strip():
            raise ValueError("Tenant and document ids must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ? AND tenant_id = ?
                """,
                (document_id, tenant_id),
            ).fetchone()
            if row is None:
                return None, False
            document = self._document_from_row(row)
            if document.status in {
                DocumentStatus.DELETING,
                DocumentStatus.DELETED,
            }:
                return document, False

            transition_document(document, DocumentStatus.DELETING)
            connection.execute(
                """
                UPDATE documents
                SET status = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (document.status.value, document.id, tenant_id),
            )

            version_rows = connection.execute(
                """
                SELECT *
                FROM document_versions
                WHERE document_id = ?
                ORDER BY version_number
                """,
                (document.id,),
            ).fetchall()
            for version_row in version_rows:
                version = self._version_from_row(version_row)
                if version.status in {
                    VersionStatus.DELETING,
                    VersionStatus.DELETED,
                }:
                    continue
                transition_version(version, VersionStatus.DELETING)
                connection.execute(
                    """
                    UPDATE document_versions
                    SET status = ?
                    WHERE id = ? AND document_id = ?
                    """,
                    (version.status.value, version.id, document.id),
                )

            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    stage = 'cancelled',
                    error = 'document_deleted',
                    next_attempt_at = 0,
                    lease_owner = NULL,
                    lease_token = lease_token + 1,
                    lease_expires_at = NULL
                WHERE tenant_id = ?
                    AND document_version_id IN (
                        SELECT id FROM document_versions WHERE document_id = ?
                    )
                    AND status IN (?, ?, ?)
                """,
                (
                    JobStatus.CANCELLED.value,
                    tenant_id,
                    document.id,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                    JobStatus.RETRY_WAIT.value,
                ),
            )
            connection.execute(
                """
                UPDATE ingestion_runs
                SET status = ?
                WHERE tenant_id = ?
                    AND document_version_id IN (
                        SELECT id FROM document_versions WHERE document_id = ?
                    )
                    AND status IN (?, ?)
                """,
                (
                    RunStatus.CANCELLED.value,
                    tenant_id,
                    document.id,
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                ),
            )
            self._bump_authorization_revision(connection, tenant_id)
            updated_row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND tenant_id = ?",
                (document.id, tenant_id),
            ).fetchone()
        return self._document_from_row(updated_row), True

    def get_deletion_manifest_for_tenant(
        self,
        tenant_id: str,
        document_id: str,
        *,
        actor: ActorContext | None = None,
    ) -> DeletionManifest | None:
        """Read exact owned raw/run keys only for a pending tombstone."""

        if not tenant_id.strip() or not document_id.strip():
            raise ValueError("Tenant and document ids must be non-empty")
        with self._connection() as connection:
            if actor is not None:
                connection.execute("BEGIN")
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            return self._deletion_manifest(connection, tenant_id, document_id)

    @staticmethod
    def _deletion_manifest(
        connection: sqlite3.Connection, tenant_id: str, document_id: str
    ) -> DeletionManifest | None:
        document_row = connection.execute(
            """
                SELECT status
                FROM documents
                WHERE id = ? AND tenant_id = ?
                """,
            (document_id, tenant_id),
        ).fetchone()
        if (
            document_row is None
            or document_row["status"] != DocumentStatus.DELETING.value
        ):
            return None
        rows = connection.execute(
            """
                SELECT v.id AS version_id, v.object_key,
                       r.id AS run_id, j.id AS job_id, j.status AS job_status
                FROM document_versions AS v
                LEFT JOIN ingestion_runs AS r
                    ON r.document_version_id = v.id
                LEFT JOIN ingestion_jobs AS j
                    ON j.document_version_id = v.id
                WHERE v.document_id = ?
                ORDER BY v.version_number, v.id
                """,
            (document_id,),
        ).fetchall()
        object_keys = [row["object_key"] for row in rows if row["object_key"]]
        if SQLiteMetadataStore._table_exists(connection, "citation_visual_regions"):
            object_keys.extend(
                row[0]
                for row in connection.execute(
                    "SELECT object_key FROM citation_visual_regions WHERE document_id=?",
                    (document_id,),
                )
            )
        return DeletionManifest(
            document_id=document_id,
            version_ids=tuple(dict.fromkeys(row["version_id"] for row in rows)),
            object_keys=tuple(dict.fromkeys(object_keys)),
            run_ids=tuple(
                dict.fromkeys(row["run_id"] for row in rows if row["run_id"])
            ),
            active_job_ids=tuple(
                dict.fromkeys(
                    row["job_id"]
                    for row in rows
                    if row["job_id"]
                    and row["job_status"]
                    in {
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                        JobStatus.RETRY_WAIT.value,
                    }
                )
            ),
        )

    def claim_document_deletion(
        self,
        tenant_id: str,
        document_id: str,
        *,
        actor: ActorContext,
        now: float,
        ttl_seconds: float,
    ) -> DocumentDeletionClaim | None:
        """Accept one exact tombstone cleanup under a durable, fenced token."""

        if not tenant_id.strip() or not document_id.strip():
            raise ValueError("Tenant and document ids must be non-empty")
        if not isfinite(now) or not isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("Deletion claim time and TTL must be finite and positive")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_tenant_admin_actor(connection, tenant_id, actor)
            manifest = self._deletion_manifest(connection, tenant_id, document_id)
            if manifest is None or manifest.active_job_ids:
                return None
            key_sources = ["SELECT document_id,object_key FROM document_versions"]
            for table_name in ("citation_visual_regions", "upload_attempts"):
                if self._table_exists(connection, table_name):
                    key_sources.append(
                        f"SELECT document_id,object_key FROM {table_name}"
                    )
            shared_visual = connection.execute(
                f"""WITH all_keys AS ({" UNION ALL ".join(key_sources)})
                    SELECT 1 FROM all_keys AS owned JOIN all_keys AS other
                      ON owned.object_key=other.object_key
                     AND owned.document_id!=other.document_id
                    WHERE owned.document_id=? LIMIT 1""",
                (document_id,),
            ).fetchone()
            if shared_visual is not None:
                raise SharedDeletionObjectKeyError(
                    "Document deletion object key is shared with another document or attempt"
                )
            shared_key = connection.execute(
                """
                SELECT 1
                FROM document_versions AS owned
                JOIN document_versions AS other
                    ON other.object_key = owned.object_key
                    AND other.document_id != owned.document_id
                WHERE owned.document_id = ?
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            if shared_key is not None:
                raise SharedDeletionObjectKeyError(
                    "Document deletion object key is shared with another document"
                )
            shared_attempt = connection.execute(
                """
                SELECT 1
                FROM document_versions AS owned
                JOIN upload_attempts AS other
                    ON other.object_key = owned.object_key
                    AND other.document_id != owned.document_id
                WHERE owned.document_id = ?
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            if shared_attempt is not None:
                raise SharedDeletionObjectKeyError(
                    "Document deletion object key is shared with another upload attempt"
                )
            current = connection.execute(
                """
                SELECT tenant_id, expires_at
                FROM document_deletion_claims
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            if current is not None:
                if current["tenant_id"] != tenant_id:
                    raise ValueError("Deletion claim tenant mismatch")
                if current["expires_at"] > now:
                    return None
            token = uuid4().hex
            expires_at = now + ttl_seconds
            if not isfinite(expires_at):
                raise ValueError("Deletion claim expiry must be finite")
            if current is None:
                connection.execute(
                    """
                    INSERT INTO document_deletion_claims
                        (document_id, tenant_id, claim_token, claimed_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, tenant_id, token, now, expires_at),
                )
            else:
                connection.execute(
                    """
                    UPDATE document_deletion_claims
                    SET claim_token = ?, claimed_at = ?, expires_at = ?
                    WHERE document_id = ? AND tenant_id = ? AND expires_at <= ?
                    """,
                    (token, now, expires_at, document_id, tenant_id, now),
                )
        return DocumentDeletionClaim(
            tenant_id=tenant_id,
            document_id=document_id,
            claim_token=token,
            expires_at=expires_at,
            manifest=manifest,
        )

    def is_document_deletion_claim_current(
        self,
        tenant_id: str,
        document_id: str,
        *,
        claim_token: str,
        now: float,
    ) -> bool:
        """Fence external delete dispatch by the current owner and tombstone."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM document_deletion_claims AS c
                JOIN documents AS d
                    ON d.id = c.document_id AND d.tenant_id = c.tenant_id
                WHERE c.tenant_id = ? AND c.document_id = ?
                    AND c.claim_token = ?
                    AND c.claimed_at <= ? AND c.expires_at > ?
                    AND d.status = ?
                """,
                (
                    tenant_id,
                    document_id,
                    claim_token,
                    now,
                    now,
                    DocumentStatus.DELETING.value,
                ),
            ).fetchone()
        return row is not None

    def release_document_deletion_claim(
        self, tenant_id: str, document_id: str, *, claim_token: str
    ) -> bool:
        """Release only this owner after a definitely non-mutating delete error."""

        with self._connection() as connection:
            deleted = connection.execute(
                """
                DELETE FROM document_deletion_claims
                WHERE tenant_id = ? AND document_id = ? AND claim_token = ?
                """,
                (tenant_id, document_id, claim_token),
            )
        return deleted.rowcount == 1

    def finalize_document_deletion(
        self,
        tenant_id: str,
        document_id: str,
        *,
        claim_token: str,
        now: float,
    ) -> bool:
        """Finalize only the current deletion owner and clear its claim atomically."""

        if not tenant_id.strip() or not document_id.strip():
            raise ValueError("Tenant and document ids must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claim = connection.execute(
                """
                SELECT 1 FROM document_deletion_claims
                WHERE tenant_id = ? AND document_id = ?
                    AND claim_token = ?
                    AND claimed_at <= ? AND expires_at > ?
                """,
                (tenant_id, document_id, claim_token, now, now),
            ).fetchone()
            if claim is None:
                return False
            document_row = connection.execute(
                """
                SELECT status
                FROM documents
                WHERE id = ? AND tenant_id = ?
                """,
                (document_id, tenant_id),
            ).fetchone()
            if (
                document_row is None
                or document_row["status"] != DocumentStatus.DELETING.value
            ):
                return False
            active_job = connection.execute(
                """
                SELECT 1
                FROM ingestion_jobs AS j
                JOIN document_versions AS v
                    ON v.id = j.document_version_id
                WHERE v.document_id = ?
                    AND j.status IN (?, ?, ?)
                LIMIT 1
                """,
                (
                    document_id,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                    JobStatus.RETRY_WAIT.value,
                ),
            ).fetchone()
            if active_job is not None:
                return False

            run_ids = tuple(
                row["id"]
                for row in connection.execute(
                    """
                    SELECT r.id
                    FROM ingestion_runs AS r
                    JOIN document_versions AS v
                        ON v.id = r.document_version_id
                    WHERE v.document_id = ?
                    ORDER BY r.id
                    """,
                    (document_id,),
                ).fetchall()
            )
            if run_ids:
                placeholders = ", ".join("?" for _ in run_ids)
                connection.execute(
                    f"DELETE FROM text_chunks_fts WHERE run_id IN ({placeholders})",
                    run_ids,
                )
                connection.execute(
                    f"DELETE FROM lexical_index_records WHERE run_id IN ({placeholders})",
                    run_ids,
                )
                connection.execute(
                    f"DELETE FROM text_chunks WHERE run_id IN ({placeholders})",
                    run_ids,
                )
                connection.execute(
                    f"DELETE FROM text_elements WHERE run_id IN ({placeholders})",
                    run_ids,
                )
                connection.execute(
                    f"DELETE FROM text_artifacts WHERE run_id IN ({placeholders})",
                    run_ids,
                )
            connection.execute(
                "DELETE FROM document_permissions WHERE document_id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET status = ?
                WHERE document_id = ? AND status != ?
                """,
                (
                    VersionStatus.DELETED.value,
                    document_id,
                    VersionStatus.DELETED.value,
                ),
            )
            updated = connection.execute(
                """
                UPDATE documents
                SET status = ?, active_version_id = NULL, active_run_id = NULL
                WHERE id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    DocumentStatus.DELETED.value,
                    document_id,
                    tenant_id,
                    DocumentStatus.DELETING.value,
                ),
            ).rowcount
            if updated:
                self._bump_authorization_revision(connection, tenant_id)
                connection.execute(
                    """
                    DELETE FROM document_deletion_claims
                    WHERE tenant_id = ? AND document_id = ? AND claim_token = ?
                    """,
                    (tenant_id, document_id, claim_token),
                )
            return updated == 1

    def register_upload_attempt(
        self,
        tenant_id: str,
        document_id: str,
        object_key: str,
        *,
        created_at: float,
        expires_at: float,
    ) -> None:
        """Record ownership before an object-store write begins."""

        if not tenant_id.strip() or not document_id.strip() or not object_key.strip():
            raise ValueError("Upload attempt ownership fields must be non-empty")
        if expires_at < created_at:
            raise ValueError("Upload attempt expiry must not precede creation")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_object_key_available_for_upload(
                connection,
                object_key,
                document_id=document_id,
                allow_existing_same_document=True,
            )
            connection.execute(
                """
                INSERT INTO upload_attempts
                    (object_key, tenant_id, document_id, status, created_at, expires_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (object_key, tenant_id, document_id, created_at, expires_at),
            )

    def finish_upload_attempt(
        self, tenant_id: str, object_key: str, *, status: str, now: float
    ) -> bool:
        """Mark a tracked attempt committed or abandoned without weakening refs."""

        if status not in {"committed", "abandoned"}:
            raise ValueError("Upload attempt status must be committed or abandoned")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if status == "abandoned":
                updated = connection.execute(
                    """
                    UPDATE upload_attempts
                    SET status = ?, expires_at = MIN(expires_at, ?)
                    WHERE tenant_id = ? AND object_key = ? AND status = 'pending'
                    """,
                    (status, now, tenant_id, object_key),
                ).rowcount
            else:
                updated = connection.execute(
                    """
                    UPDATE upload_attempts
                    SET status = ?
                    WHERE tenant_id = ? AND object_key = ? AND status = 'pending'
                    """,
                    (status, tenant_id, object_key),
                ).rowcount
        return updated == 1

    def list_expired_upload_attempts_for_tenant(
        self,
        tenant_id: str,
        *,
        now: float,
        grace_seconds: float,
        limit: int = 100,
        actor: ActorContext | None = None,
    ) -> tuple[UploadAttempt, ...]:
        """List tracked, unreferenced attempts after their expiry plus grace."""

        if grace_seconds < 0:
            raise ValueError("Grace period must be non-negative")
        if limit <= 0 or limit > 1000:
            raise ValueError("GC batch size must be between 1 and 1000")
        cutoff = now - grace_seconds
        with self._connection() as connection:
            if actor is not None:
                connection.execute("BEGIN")
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            rows = connection.execute(
                """
                SELECT a.*
                FROM upload_attempts AS a
                WHERE a.tenant_id = ?
                    AND a.status IN ('pending', 'abandoned')
                    AND a.expires_at <= ?
                    AND NOT EXISTS (
                        SELECT 1
                        FROM document_versions AS v
                        JOIN documents AS d ON d.id = v.document_id
                        WHERE v.object_key = a.object_key
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM object_cleanup_claims AS c
                        WHERE c.object_key = a.object_key
                            AND c.status = 'deleted'
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM object_cleanup_claims AS c
                        WHERE c.object_key = a.object_key
                            AND c.tenant_id = a.tenant_id
                            AND c.status = 'claimed'
                            AND c.claim_expires_at > ?
                    )
                ORDER BY a.expires_at, a.object_key
                LIMIT ?
                """,
                (tenant_id, cutoff, now, limit + 1),
            ).fetchall()
        return tuple(
            UploadAttempt(
                object_key=row["object_key"],
                tenant_id=row["tenant_id"],
                document_id=row["document_id"],
                status=row["status"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
            )
            for row in rows
        )

    def claim_expired_upload_attempts(
        self,
        tenant_id: str,
        object_keys: Collection[str],
        *,
        now: float,
        grace_seconds: float,
        claim_ttl_seconds: float,
        actor: ActorContext | None = None,
        reclaim_deleted_keys: Collection[str] = (),
    ) -> tuple[CleanupClaim, ...]:
        """Claim only still-unreferenced expired attempts in one DB transaction."""

        if grace_seconds < 0 or claim_ttl_seconds <= 0:
            raise ValueError("Grace must be non-negative and claim TTL positive")
        unique_keys = tuple(dict.fromkeys(object_keys))
        deleted_recovery_keys = frozenset(reclaim_deleted_keys)
        if not deleted_recovery_keys.issubset(unique_keys):
            raise ValueError("Deleted recovery keys must be in the requested batch")
        if not unique_keys:
            return ()
        cutoff = now - grace_seconds
        claims: list[CleanupClaim] = []
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            for object_key in unique_keys:
                attempt = connection.execute(
                    """
                    SELECT document_id, status, expires_at
                    FROM upload_attempts
                    WHERE tenant_id = ? AND object_key = ?
                    """,
                    (tenant_id, object_key),
                ).fetchone()
                if (
                    attempt is None
                    or attempt["status"] not in {"pending", "abandoned"}
                    or attempt["expires_at"] > cutoff
                    or self._object_key_is_referenced(connection, object_key)
                ):
                    continue

                claim_token = uuid4().hex
                claim = connection.execute(
                    """
                    SELECT status, tenant_id, document_id, claim_expires_at
                    FROM object_cleanup_claims
                    WHERE object_key = ?
                    """,
                    (object_key,),
                ).fetchone()
                if claim is not None:
                    if (
                        claim["tenant_id"] != tenant_id
                        or claim["document_id"] != attempt["document_id"]
                    ):
                        continue
                    if (
                        claim["status"] == "deleted"
                        and object_key not in deleted_recovery_keys
                    ):
                        continue
                    if claim["status"] == "claimed" and claim["claim_expires_at"] > now:
                        continue
                    connection.execute(
                        """
                        UPDATE object_cleanup_claims
                        SET tenant_id = ?, document_id = ?, status = 'claimed',
                            claimed_at = ?, claim_expires_at = ?, claim_token = ?
                        WHERE object_key = ?
                        """,
                        (
                            tenant_id,
                            attempt["document_id"],
                            now,
                            now + claim_ttl_seconds,
                            claim_token,
                            object_key,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO object_cleanup_claims
                            (object_key, tenant_id, document_id, status,
                             claimed_at, claim_expires_at, claim_token)
                        VALUES (?, ?, ?, 'claimed', ?, ?, ?)
                        """,
                        (
                            object_key,
                            tenant_id,
                            attempt["document_id"],
                            now,
                            now + claim_ttl_seconds,
                            claim_token,
                        ),
                    )
                connection.execute(
                    """
                    UPDATE upload_attempts
                    SET status = 'abandoned'
                    WHERE tenant_id = ? AND object_key = ?
                    """,
                    (tenant_id, object_key),
                )
                claims.append(
                    CleanupClaim(
                        object_key=object_key,
                        tenant_id=tenant_id,
                        document_id=attempt["document_id"],
                        claim_token=claim_token,
                    )
                )
        return tuple(claims)

    def next_deleted_cleanup_audit_page(
        self,
        tenant_id: str,
        *,
        now: float,
        grace_seconds: float,
        limit: int,
        actor: ActorContext,
        advance: bool = True,
    ) -> tuple[CleanupClaim, ...]:
        """Advance a durable, bounded cursor over finalized attempt claims."""

        if grace_seconds < 0:
            raise ValueError("Grace period must be non-negative")
        if limit <= 0 or limit > 1000:
            raise ValueError("Cleanup audit limit must be between 1 and 1000")
        cutoff = now - grace_seconds
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if advance else "BEGIN")
            self._require_tenant_admin_actor(connection, tenant_id, actor)
            cursor = connection.execute(
                "SELECT after_object_key FROM cleanup_audit_cursors WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            after_key = cursor["after_object_key"] if cursor is not None else ""

            def page(after: str) -> list[sqlite3.Row]:
                return connection.execute(
                    """
                    SELECT c.object_key, c.tenant_id, c.document_id, c.claim_token
                    FROM object_cleanup_claims AS c
                    JOIN upload_attempts AS a
                        ON a.object_key = c.object_key
                        AND a.tenant_id = c.tenant_id
                        AND a.document_id = c.document_id
                    WHERE c.tenant_id = ? AND c.status = 'deleted'
                        AND c.object_key > ?
                        AND a.status IN ('pending', 'abandoned')
                        AND a.expires_at <= ?
                        AND NOT EXISTS (
                            SELECT 1 FROM document_versions AS v
                            WHERE v.object_key = c.object_key
                        )
                    ORDER BY c.object_key
                    LIMIT ?
                    """,
                    (tenant_id, after, cutoff, limit),
                ).fetchall()

            rows = page(after_key)
            if not rows and after_key:
                rows = page("")
            next_key = rows[-1]["object_key"] if rows else ""
            if advance:
                connection.execute(
                    """
                    INSERT INTO cleanup_audit_cursors (tenant_id, after_object_key)
                    VALUES (?, ?)
                    ON CONFLICT(tenant_id) DO UPDATE SET
                        after_object_key = excluded.after_object_key
                    """,
                    (tenant_id, next_key),
                )
        return tuple(
            CleanupClaim(
                object_key=row["object_key"],
                tenant_id=row["tenant_id"],
                document_id=row["document_id"],
                claim_token=row["claim_token"],
            )
            for row in rows
        )

    def list_expired_cleanup_claims(
        self,
        tenant_id: str,
        *,
        now: float,
        grace_seconds: float = 0,
        limit: int = 100,
        actor: ActorContext | None = None,
    ) -> tuple[CleanupClaim, ...]:
        """List bounded, eligible tenant-owned cleanup claims ready for recovery."""

        if grace_seconds < 0:
            raise ValueError("Grace period must be non-negative")
        if limit <= 0 or limit > 1000:
            raise ValueError("Cleanup claim recovery limit must be between 1 and 1000")
        cutoff = now - grace_seconds
        with self._connection() as connection:
            if actor is not None:
                connection.execute("BEGIN")
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            rows = connection.execute(
                """
                SELECT c.object_key, c.tenant_id, c.document_id, c.claim_token
                FROM object_cleanup_claims AS c
                JOIN upload_attempts AS a
                    ON a.object_key = c.object_key
                    AND a.tenant_id = c.tenant_id
                    AND a.document_id = c.document_id
                WHERE c.tenant_id = ?
                    AND c.status = 'claimed'
                    AND c.claim_expires_at <= ?
                    AND a.status IN ('pending', 'abandoned')
                    AND a.expires_at <= ?
                    AND NOT EXISTS (
                        SELECT 1
                        FROM document_versions AS v
                        JOIN documents AS d ON d.id = v.document_id
                        WHERE v.object_key = c.object_key
                    )
                ORDER BY c.claim_expires_at, c.object_key
                LIMIT ?
                """,
                (tenant_id, now, cutoff, limit + 1),
            ).fetchall()
        return tuple(
            CleanupClaim(
                object_key=row["object_key"],
                tenant_id=row["tenant_id"],
                document_id=row["document_id"],
                claim_token=row["claim_token"],
            )
            for row in rows
        )

    def is_cleanup_claim_current(
        self,
        tenant_id: str,
        object_key: str,
        *,
        claim_token: str,
        now: float,
    ) -> bool:
        """Recheck token ownership and the absence of any committed reference."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM object_cleanup_claims AS c
                WHERE c.object_key = ?
                    AND c.tenant_id = ?
                    AND c.claim_token = ?
                    AND c.status = 'claimed'
                    AND c.claim_expires_at > ?
                    AND NOT EXISTS (
                        SELECT 1
                        FROM document_versions AS v
                        JOIN documents AS d ON d.id = v.document_id
                        WHERE v.object_key = c.object_key
                    )
                """,
                (object_key, tenant_id, claim_token, now),
            ).fetchone()
        return row is not None

    def complete_cleanup_claim(
        self,
        tenant_id: str,
        object_key: str,
        *,
        claim_token: str,
        now: float,
    ) -> bool:
        """Commit a successful external delete only for the current token."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claim = connection.execute(
                """
                SELECT 1
                FROM object_cleanup_claims
                WHERE object_key = ? AND tenant_id = ?
                    AND claim_token = ?
                    AND status = 'claimed' AND claim_expires_at > ?
                """,
                (object_key, tenant_id, claim_token, now),
            ).fetchone()
            if claim is None or self._object_key_is_referenced(connection, object_key):
                return False
            updated = connection.execute(
                """
                UPDATE object_cleanup_claims
                SET status = 'deleted'
                WHERE object_key = ? AND tenant_id = ? AND claim_token = ?
                    AND status = 'claimed'
                """,
                (object_key, tenant_id, claim_token),
            ).rowcount
        return updated == 1

    def release_cleanup_claim(
        self, tenant_id: str, object_key: str, *, claim_token: str
    ) -> bool:
        """Release only the current failed cleanup claim for retry."""

        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE object_cleanup_claims
                SET status = 'released'
                WHERE object_key = ? AND tenant_id = ? AND claim_token = ?
                    AND status = 'claimed'
                """,
                (object_key, tenant_id, claim_token),
            ).rowcount
        return updated == 1

    def upsert_user(self, user: User, *, actor: ActorContext | None = None) -> User:
        """Register one tenant-owned user and bump auth revision on change."""

        if not user.email.strip():
            raise ValueError("User email must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, user.tenant_id, actor)
            existing = connection.execute(
                "SELECT tenant_id, email FROM users WHERE id = ?", (user.id,)
            ).fetchone()
            changed = False
            if existing is None:
                connection.execute(
                    "INSERT INTO users (id, tenant_id, email) VALUES (?, ?, ?)",
                    (user.id, user.tenant_id, user.email),
                )
                changed = True
            else:
                if existing["tenant_id"] != user.tenant_id:
                    raise ValueError("User belongs to a different tenant")
                if existing["email"] != user.email:
                    connection.execute(
                        "UPDATE users SET email = ? WHERE id = ?",
                        (user.email, user.id),
                    )
                    changed = True
            if changed:
                self._bump_authorization_revision(connection, user.tenant_id)
        return user

    def upsert_group(self, group: Group, *, actor: ActorContext | None = None) -> Group:
        """Register one tenant-owned group and bump auth revision on change."""

        if not group.name.strip():
            raise ValueError("Group name must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, group.tenant_id, actor)
            existing = connection.execute(
                "SELECT tenant_id, name FROM groups WHERE id = ?", (group.id,)
            ).fetchone()
            changed = False
            if existing is None:
                connection.execute(
                    "INSERT INTO groups (id, tenant_id, name) VALUES (?, ?, ?)",
                    (group.id, group.tenant_id, group.name),
                )
                changed = True
            else:
                if existing["tenant_id"] != group.tenant_id:
                    raise ValueError("Group belongs to a different tenant")
                if existing["name"] != group.name:
                    connection.execute(
                        "UPDATE groups SET name = ? WHERE id = ?",
                        (group.name, group.id),
                    )
                    changed = True
            if changed:
                self._bump_authorization_revision(connection, group.tenant_id)
        return group

    def add_group_member(
        self, user: User, group: Group, *, actor: ActorContext | None = None
    ) -> bool:
        """Add a tenant-validated membership; repeated adds are idempotent."""

        if user.tenant_id != group.tenant_id:
            raise ValueError("User and group must belong to the same tenant")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, user.tenant_id, actor)
            self._require_user_and_group(connection, user, group)
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO group_memberships
                    (tenant_id, user_id, group_id)
                VALUES (?, ?, ?)
                """,
                (user.tenant_id, user.id, group.id),
            ).rowcount
            if inserted:
                self._bump_authorization_revision(connection, user.tenant_id)
        return bool(inserted)

    def remove_group_member(
        self, user: User, group: Group, *, actor: ActorContext | None = None
    ) -> bool:
        """Remove a tenant-validated membership and bump auth revision."""

        if user.tenant_id != group.tenant_id:
            raise ValueError("User and group must belong to the same tenant")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, user.tenant_id, actor)
            deleted = connection.execute(
                """
                DELETE FROM group_memberships
                WHERE tenant_id = ? AND user_id = ? AND group_id = ?
                """,
                (user.tenant_id, user.id, group.id),
            ).rowcount
            if deleted:
                self._bump_authorization_revision(connection, user.tenant_id)
        return bool(deleted)

    def grant_document_permission(
        self,
        tenant_id: str,
        document_id: str,
        principal_type: PrincipalType,
        principal_id: str,
        permission: Permission,
        *,
        actor: ActorContext | None = None,
    ) -> DocumentPermission:
        """Grant read/admin; actorless calls are trusted internal setup only."""

        principal_type = PrincipalType(principal_type)
        permission = Permission(permission)
        if not principal_id.strip():
            raise ValueError("Permission principal id must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            document = connection.execute(
                "SELECT tenant_id FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None or document["tenant_id"] != tenant_id:
                raise ValueError("Document is not owned by tenant")
            if principal_type is PrincipalType.TENANT:
                if principal_id != tenant_id:
                    raise ValueError("Tenant principal must match permission tenant")
            elif principal_type is PrincipalType.USER:
                self._require_principal_row(
                    connection, "users", tenant_id, principal_id, "User"
                )
            else:
                self._require_principal_row(
                    connection, "groups", tenant_id, principal_id, "Group"
                )
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO document_permissions
                    (document_id, tenant_id, principal_type, principal_id, permission)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    tenant_id,
                    principal_type.value,
                    principal_id,
                    permission.value,
                ),
            ).rowcount
            if inserted:
                self._bump_authorization_revision(connection, tenant_id)
        return DocumentPermission(
            document_id=document_id,
            tenant_id=tenant_id,
            principal_type=principal_type,
            principal_id=principal_id,
            permission=permission,
        )

    def revoke_document_permission(
        self,
        tenant_id: str,
        document_id: str,
        principal_type: PrincipalType,
        principal_id: str,
        permission: Permission,
        *,
        actor: ActorContext | None = None,
    ) -> bool:
        """Revoke one permission; actorless calls are trusted internal setup only."""

        principal_type = PrincipalType(principal_type)
        permission = Permission(permission)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            document = connection.execute(
                "SELECT tenant_id FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None or document["tenant_id"] != tenant_id:
                raise ValueError("Document is not owned by tenant")
            deleted = connection.execute(
                """
                DELETE FROM document_permissions
                WHERE document_id = ? AND tenant_id = ?
                    AND principal_type = ? AND principal_id = ?
                    AND permission = ?
                """,
                (
                    document_id,
                    tenant_id,
                    principal_type.value,
                    principal_id,
                    permission.value,
                ),
            ).rowcount
            if deleted:
                self._bump_authorization_revision(connection, tenant_id)
        return bool(deleted)

    def get_authorization_revision(self, tenant_id: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT revision
                FROM tenant_authorization_revisions
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
        return 0 if row is None else int(row["revision"])

    def grant_tenant_admin(self, tenant_id: str, user_id: str) -> bool:
        """Bootstrap a tenant admin after validating the user tenant relationship."""

        if not tenant_id.strip() or not user_id.strip():
            raise ValueError("Tenant and user ids must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_principal_row(connection, "users", tenant_id, user_id, "User")
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO tenant_admins (tenant_id, user_id)
                VALUES (?, ?)
                """,
                (tenant_id, user_id),
            ).rowcount
            if inserted:
                self._bump_authorization_revision(connection, tenant_id)
        return bool(inserted)

    def revoke_tenant_admin(self, tenant_id: str, user_id: str) -> bool:
        """Remove a tenant-admin bootstrap grant idempotently."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute(
                """
                DELETE FROM tenant_admins
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).rowcount
            if deleted:
                self._bump_authorization_revision(connection, tenant_id)
        return bool(deleted)

    def is_tenant_admin(self, tenant_id: str, user_id: str) -> bool:
        """Check an explicit tenant-admin grant without trusting request claims."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM tenant_admins
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
        return row is not None

    @staticmethod
    def _require_tenant_admin_actor(
        connection: sqlite3.Connection, tenant_id: str, actor: ActorContext
    ) -> None:
        """Authorize an administrative write inside its SQLite transaction."""

        if actor.tenant_id != tenant_id or actor.user_id is None:
            raise PermissionError("Tenant admin permission required")
        admin = connection.execute(
            """
            SELECT 1
            FROM tenant_admins AS a
            JOIN users AS u ON u.id = a.user_id AND u.tenant_id = a.tenant_id
            WHERE a.tenant_id = ? AND a.user_id = ?
            """,
            (tenant_id, actor.user_id),
        ).fetchone()
        if admin is None:
            raise PermissionError("Tenant admin permission required")

    def resolve_security_scope(
        self, actor: ActorContext | None
    ) -> SecurityScope | None:
        """Resolve memberships from authoritative SQLite state; never trust input groups."""

        if actor is None or actor.user_id is None:
            return None
        with self._connection() as connection:
            user = connection.execute(
                """
                SELECT id
                FROM users
                WHERE id = ? AND tenant_id = ?
                """,
                (actor.user_id, actor.tenant_id),
            ).fetchone()
            if user is None:
                return None
            group_ids = tuple(
                row["group_id"]
                for row in connection.execute(
                    """
                    SELECT group_id
                    FROM group_memberships
                    WHERE tenant_id = ? AND user_id = ?
                    ORDER BY group_id
                    """,
                    (actor.tenant_id, actor.user_id),
                ).fetchall()
            )
            revision = connection.execute(
                """
                SELECT revision
                FROM tenant_authorization_revisions
                WHERE tenant_id = ?
                """,
                (actor.tenant_id,),
            ).fetchone()
        return SecurityScope(
            tenant_id=actor.tenant_id,
            user_id=actor.user_id,
            group_ids=group_ids,
            authorization_revision=0 if revision is None else int(revision["revision"]),
        )

    @staticmethod
    def _bump_authorization_revision(
        connection: sqlite3.Connection, tenant_id: str
    ) -> None:
        connection.execute(
            """
            INSERT INTO tenant_authorization_revisions (tenant_id, revision)
            VALUES (?, 1)
            ON CONFLICT(tenant_id) DO UPDATE SET revision = revision + 1
            """,
            (tenant_id,),
        )

    @staticmethod
    def _require_user_and_group(
        connection: sqlite3.Connection, user: User, group: Group
    ) -> None:
        user_row = connection.execute(
            "SELECT tenant_id FROM users WHERE id = ?", (user.id,)
        ).fetchone()
        group_row = connection.execute(
            "SELECT tenant_id FROM groups WHERE id = ?", (group.id,)
        ).fetchone()
        if (
            user_row is None
            or group_row is None
            or user_row["tenant_id"] != user.tenant_id
            or group_row["tenant_id"] != group.tenant_id
        ):
            raise ValueError("User and group must be registered in their tenant")

    @staticmethod
    def _require_principal_row(
        connection: sqlite3.Connection,
        table: str,
        tenant_id: str,
        principal_id: str,
        label: str,
    ) -> None:
        if table not in {"users", "groups"}:
            raise ValueError("Unsupported principal table")
        row = connection.execute(
            f"SELECT tenant_id FROM {table} WHERE id = ?", (principal_id,)
        ).fetchone()
        if row is None or row["tenant_id"] != tenant_id:
            raise ValueError(f"{label} is not registered in permission tenant")

    def get_stage_attempt(self, attempt_id: str) -> StageAttempt | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM stage_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        return self._stage_attempt_from_row(row) if row else None

    def get_stage_attempts(self, run_id: str) -> tuple[StageAttempt, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM stage_attempts
                WHERE run_id = ?
                ORDER BY started_at, id
                """,
                (run_id,),
            ).fetchall()
        return tuple(self._stage_attempt_from_row(row) for row in rows)

    def reconcile_stale_stage_attempts(
        self,
        tenant_id: str,
        *,
        now: float,
        limit: int = 100,
        dry_run: bool = False,
        actor: ActorContext | None = None,
    ) -> StageAttemptReconciliation:
        """Mark only running attempts owned by a terminal/non-running job as stale.

        A running job, a non-null lease expiry, and identity mismatches are intentionally
        excluded because they need worker/administrator investigation instead of
        an automatic state change. A historical lease owner with a null expiry is
        not an active lease and does not block cleanup.
        """

        if not tenant_id.strip():
            raise ValueError("Tenant id must be non-empty")
        if limit <= 0 or limit > 1000:
            raise ValueError("Reconciliation limit must be between 1 and 1000")
        with self._connection() as connection:
            if not dry_run:
                connection.execute("BEGIN IMMEDIATE")
            elif actor is not None:
                connection.execute("BEGIN")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            rows = connection.execute(
                """
                SELECT a.id
                FROM stage_attempts AS a
                JOIN ingestion_runs AS r ON r.id = a.run_id
                JOIN document_versions AS v ON v.id = r.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                JOIN ingestion_jobs AS j ON j.id = a.job_id
                WHERE d.tenant_id = ?
                    AND a.status = ?
                    AND j.tenant_id = r.tenant_id
                    AND j.run_id = a.run_id
                    AND j.document_version_id = r.document_version_id
                    AND j.status != ?
                    AND j.lease_expires_at IS NULL
                ORDER BY a.started_at, a.id
                LIMIT ?
                """,
                (
                    tenant_id,
                    StageStatus.RUNNING.value,
                    JobStatus.RUNNING.value,
                    limit + 1,
                ),
            ).fetchall()
            truncated = len(rows) > limit
            candidate_ids = tuple(row["id"] for row in rows[:limit])
            if dry_run:
                return StageAttemptReconciliation(
                    candidate_attempt_ids=candidate_ids,
                    repaired_attempt_ids=(),
                    truncated=truncated,
                )

            repaired_ids: list[str] = []
            for attempt_id in candidate_ids:
                updated = connection.execute(
                    """
                    UPDATE stage_attempts
                    SET status = ?, error = ?, finished_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        StageStatus.FAILED.value,
                        "stale_job_state",
                        now,
                        attempt_id,
                        StageStatus.RUNNING.value,
                    ),
                )
                if updated.rowcount == 1:
                    repaired_ids.append(attempt_id)
        return StageAttemptReconciliation(
            candidate_attempt_ids=candidate_ids,
            repaired_attempt_ids=tuple(repaired_ids),
            truncated=truncated,
        )

    def reconcile_terminal_runs(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        dry_run: bool = False,
        actor: ActorContext | None = None,
    ) -> TerminalRunReconciliation:
        """Repair only running runs with a terminal failed/cancelled job.

        A succeeded job is intentionally excluded because run completion also
        requires proof that every required stage succeeded. Non-null lease
        expiries, historical owner fields, active leases,
        tombstoned documents, identity mismatches, and ambiguous multiple-job
        relationships remain for explicit investigation.
        """

        if not tenant_id.strip():
            raise ValueError("Tenant id must be non-empty")
        if limit <= 0 or limit > 1000:
            raise ValueError("Reconciliation limit must be between 1 and 1000")
        with self._connection() as connection:
            if not dry_run:
                connection.execute("BEGIN IMMEDIATE")
            elif actor is not None:
                connection.execute("BEGIN")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            rows = connection.execute(
                """
                SELECT r.id, j.status AS job_status
                FROM ingestion_runs AS r
                JOIN document_versions AS v ON v.id = r.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                JOIN ingestion_jobs AS j ON j.run_id = r.id
                WHERE d.tenant_id = ?
                    AND r.tenant_id = d.tenant_id
                    AND r.status = ?
                    AND j.tenant_id = r.tenant_id
                    AND j.document_version_id = r.document_version_id
                    AND j.status IN (?, ?)
                    AND j.lease_expires_at IS NULL
                    AND d.status NOT IN (?, ?)
                    AND (
                        SELECT COUNT(*)
                        FROM ingestion_jobs AS related
                        WHERE related.run_id = r.id
                    ) = 1
                ORDER BY r.created_at, r.id
                LIMIT ?
                """,
                (
                    tenant_id,
                    RunStatus.RUNNING.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    limit + 1,
                ),
            ).fetchall()
            truncated = len(rows) > limit
            candidate_ids = tuple(row["id"] for row in rows[:limit])
            if dry_run:
                return TerminalRunReconciliation(
                    candidate_run_ids=candidate_ids,
                    repaired_run_ids=(),
                    truncated=truncated,
                )

            repaired_ids: list[str] = []
            for row in rows[:limit]:
                target_status = (
                    RunStatus.FAILED.value
                    if row["job_status"] == JobStatus.FAILED.value
                    else RunStatus.CANCELLED.value
                )
                updated = connection.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = ?
                    WHERE id = ? AND status = ?
                    """,
                    (target_status, row["id"], RunStatus.RUNNING.value),
                )
                if updated.rowcount == 1:
                    repaired_ids.append(row["id"])
        return TerminalRunReconciliation(
            candidate_run_ids=candidate_ids,
            repaired_run_ids=tuple(repaired_ids),
            truncated=truncated,
        )

    def active_run_id(self, document_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT active_run_id FROM documents WHERE id=?", (document_id,)
            ).fetchone()
        return row[0] if row else None

    def request_reprocess(
        self,
        actor: ActorContext,
        version_id: str,
        request_key: str,
        *,
        max_chunk_chars: int,
        now: float,
    ) -> tuple[IngestionJob, bool]:
        """Accept one explicit new run; never mutate prior run or raw version.

        Only the bounded chunk size is user-configurable in this lite contract.
        Parser/provider revisions and the remaining limits are server-owned.
        """
        from enterprise_rag.ingestion import DEFAULT_PDF_PARSER_TIMEOUT_SECONDS
        from enterprise_rag.ingestion.configuration import execution_configuration

        if (
            not isinstance(request_key, str)
            or len(request_key) != 32
            or any(c not in "0123456789abcdef" for c in request_key)
            or type(max_chunk_chars) is not int
            or not 1 <= max_chunk_chars <= 2000
        ):
            raise ValueError("Invalid bounded reprocess request")
        scope = self.resolve_security_scope(actor)
        if scope is None:
            raise PermissionError("Current actor required")
        assert actor.user_id is not None
        acl, parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_tenant_admin_actor(connection, actor.tenant_id, actor)
            version = connection.execute(
                f"""SELECT v.*,d.active_run_id,d.active_version_id FROM document_versions v
                    JOIN documents d ON d.id=v.document_id
                    WHERE v.id=? AND d.tenant_id=? AND d.status NOT IN ('deleting','deleted')
                    AND {acl}""",
                (version_id, actor.tenant_id, *parameters),
            ).fetchone()
            if version is None:
                raise LookupError("Accessible document version not found")
            config = execution_configuration(
                version["filename"],
                max_text_bytes=25 * 1024 * 1024,
                max_pdf_pages=100,
                pdf_parser_timeout_seconds=DEFAULT_PDF_PARSER_TIMEOUT_SECONDS,
                max_chunk_chars=max_chunk_chars,
            )
            encoded = json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            digest = sha256_bytes(encoded.encode())
            prior = connection.execute(
                """SELECT r.execution_configuration_json,j.* FROM ingestion_runs r
                   JOIN ingestion_jobs j ON j.run_id=r.id
                   WHERE r.tenant_id=? AND r.document_version_id=?
                   AND r.requested_by=? AND r.reprocess_request_key=?""",
                (actor.tenant_id, version_id, actor.user_id, request_key),
            ).fetchone()
            if prior is not None:
                # Retransmission preserves the original server-owned snapshot,
                # including after a deployment changes parser/provider revisions.
                if (
                    json.loads(prior["execution_configuration_json"])["max_chunk_chars"]
                    != max_chunk_chars
                ):
                    raise RuntimeError("Reprocess request key already bound")
                return self._job_from_row(prior), False
            if (
                version["active_version_id"] != version_id
                or version["status"] != "active"
            ):
                raise RuntimeError("Only the current active version can be reprocessed")
            latest_version = connection.execute(
                "SELECT max(version_number) FROM document_versions WHERE document_id=?",
                (version["document_id"],),
            ).fetchone()[0]
            if version["version_number"] != latest_version:
                raise RuntimeError("A newer document version already exists")
            if version["filename"].lower().rsplit(".", 1)[-1] not in {
                "txt",
                "md",
                "pdf",
            }:
                raise ValueError("Unsupported reprocess parser")
            if connection.execute(
                """SELECT 1 FROM ingestion_jobs j JOIN document_versions v ON v.id=j.document_version_id
                   WHERE v.document_id=? AND j.status IN ('queued','running','retry_wait')""",
                (version["document_id"],),
            ).fetchone():
                raise RuntimeError("Document already has pending processing")
            generation = connection.execute(
                "SELECT coalesce(max(request_generation),0)+1 FROM ingestion_runs WHERE document_version_id=?",
                (version_id,),
            ).fetchone()[0]
            fingerprint = "lite-reprocess-v1:" + sha256_bytes(
                (actor.user_id + request_key + digest).encode()
            )
            run = IngestionRun.create(actor.tenant_id, version_id, fingerprint)
            job_id = deterministic_id("ingestion-run-job", actor.tenant_id, run.id)
            connection.execute(
                """INSERT INTO ingestion_runs(id,tenant_id,document_version_id,pipeline_fingerprint,
                   required_stages,status,request_generation,created_at,execution_configuration_json,
                   execution_configuration_sha256,parent_run_id,reprocess_request_key,requested_by)
                   VALUES(?,?,?,?,?,'queued',?,?,?,?,?,?,?)""",
                (
                    run.id,
                    actor.tenant_id,
                    version_id,
                    fingerprint,
                    json.dumps(run.required_stages),
                    generation,
                    now,
                    encoded,
                    digest,
                    version["active_run_id"],
                    request_key,
                    actor.user_id,
                ),
            )
            connection.execute(
                """INSERT INTO ingestion_jobs(id,tenant_id,document_version_id,status,attempts,run_id)
                   VALUES(?,?,?,'queued',0,?)""",
                (job_id, actor.tenant_id, version_id, run.id),
            )
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)
            ).fetchone()
            return self._job_from_row(row), True

    def match_run_execution_configuration(
        self, lease: JobLease, configuration: dict, *, now: float, bind: bool = False
    ) -> bool:
        """Fence immutable effective worker configuration before using checkpoints.

        This supplements the historical upload fingerprint; it does not invent
        the configuration of an old run which already has canonical artifacts.
        """
        serialized = json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if (
            not isinstance(configuration, dict)
            or not configuration
            or len(serialized) > 8192
        ):
            raise ValueError("Invalid execution configuration")
        digest = sha256_bytes(serialized.encode())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT r.execution_configuration_json,r.execution_configuration_sha256
                   FROM ingestion_jobs j
                   JOIN ingestion_runs r ON r.id=j.run_id AND r.tenant_id=j.tenant_id
                       AND r.document_version_id=j.document_version_id
                   JOIN document_versions v ON v.id=r.document_version_id
                   JOIN documents d ON d.id=v.document_id AND d.tenant_id=j.tenant_id
                   WHERE j.id=? AND j.tenant_id=? AND j.run_id=?
                       AND j.document_version_id=? AND j.status=?
                       AND j.lease_owner=? AND j.lease_token=? AND j.lease_expires_at>?
                       AND r.status=? AND d.status NOT IN (?,?)""",
                (
                    lease.job.id,
                    lease.job.tenant_id,
                    lease.job.run_id,
                    lease.job.document_version_id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                    RunStatus.RUNNING.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                ),
            ).fetchone()
            if row is None:
                return False
            if row["execution_configuration_json"] is not None:
                return (
                    row["execution_configuration_json"] == serialized
                    and row["execution_configuration_sha256"] == digest
                )
            if not bind:
                return False
            if connection.execute(
                "SELECT 1 FROM text_artifacts WHERE run_id=?", (lease.job.run_id,)
            ).fetchone():
                raise ValueError("Existing artifacts have no execution configuration")
            connection.execute(
                """UPDATE ingestion_runs SET execution_configuration_json=?,
                   execution_configuration_sha256=? WHERE id=? AND tenant_id=?""",
                (serialized, digest, lease.job.run_id, lease.job.tenant_id),
            )
        return True

    def persist_text_artifacts(
        self,
        run_id: str,
        parsed: ParsedText,
        chunks: tuple[TextChunk, ...],
        *,
        lease: JobLease,
        now: float,
    ) -> str:
        """Persist one immutable snapshot under the current job fence."""
        from enterprise_rag.persistence import sqlite_hierarchy

        if parsed.run_id != run_id:
            raise ValueError("Parsed text and run must be consistent")
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Unknown ingestion run: {run_id}")
        version = self.get_version(run.document_version_id)
        if version is None or version.content_sha256 != parsed.content_sha256:
            raise ValueError("Parsed text hash does not match raw version")
        if parsed.ocr_pages and not version.filename.lower().endswith(".pdf"):
            raise ValueError("OCR page evidence requires a PDF source")
        ocr_payloads: list[tuple[OCRPageEvidence, str, str]] = []
        total_ocr_bytes = 0
        for evidence in parsed.ocr_pages:
            result = evidence.result
            if (
                result.run_id != run_id
                or result.document_version_id != version.id
                or result.source_pdf_sha256 != version.content_sha256
                or result.page_number != evidence.page_number
                or result.image_sha256 != evidence.page_image_sha256
            ):
                raise ValueError("OCR evidence does not match the source version")
            payload = _ocr_page_evidence_json(evidence)
            payload_size = len(payload.encode("utf-8"))
            total_ocr_bytes += payload_size
            if total_ocr_bytes > MAX_OCR_RUN_EVIDENCE_BYTES:
                raise ValueError("OCR evidence exceeds the per-run byte limit")
            ocr_payloads.append(
                (evidence, payload, sha256_bytes(payload.encode("utf-8")))
            )
        chunks = tuple(chunks)
        element_ids = {element.id for element in parsed.elements}
        if len(element_ids) != len(parsed.elements):
            raise ValueError("Parsed text contains duplicate element IDs")
        for chunk in chunks:
            if chunk.run_id != run_id:
                raise ValueError("Chunk and run must be consistent")
            if not set(chunk.element_ids).issubset(element_ids):
                raise ValueError("Chunk references an unknown text element")

        artifact_id = deterministic_id(
            "text-artifact",
            run_id,
            parsed.parser_revision,
            parsed.content_sha256,
        )
        warnings_json = json.dumps(
            parsed.warnings, ensure_ascii=False, separators=(",", ":")
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            live = connection.execute(
                """
                SELECT 1
                FROM ingestion_jobs AS j
                JOIN ingestion_runs AS r
                    ON r.id = j.run_id AND r.tenant_id = j.tenant_id
                    AND r.document_version_id = j.document_version_id
                JOIN document_versions AS v ON v.id = r.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                WHERE j.id = ? AND j.status = ? AND j.tenant_id = ?
                    AND j.run_id = ? AND j.document_version_id = ?
                    AND j.lease_owner = ? AND j.lease_token = ?
                    AND j.lease_expires_at > ? AND r.status = ?
                    AND d.tenant_id = j.tenant_id AND d.status NOT IN (?, ?)
                """,
                (
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.job.tenant_id,
                    run_id,
                    lease.job.document_version_id,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                    RunStatus.RUNNING.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                ),
            ).fetchone()
            if live is None or lease.job.run_id != run_id:
                raise ValueError("Job lease is not live for text artifacts")
            existing = connection.execute(
                "SELECT * FROM text_artifacts WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing:
                if (
                    existing["id"] != artifact_id
                    or existing["parser_revision"] != parsed.parser_revision
                    or existing["content_sha256"] != parsed.content_sha256
                    or existing["original_text"] != parsed.original_text
                    or existing["normalized_text"] != parsed.normalized_text
                    or json.loads(existing["warnings"]) != list(parsed.warnings)
                    or existing["pdf_image_provider"] != parsed.pdf_image_provider
                    or pdf_chart_bars_from_payload(
                        json.loads(existing["pdf_chart_bars"])
                    )
                    != parsed.pdf_chart_bars
                    or pdf_image_regions_from_payload(
                        json.loads(existing["pdf_image_regions"])
                    )
                    != parsed.pdf_image_regions
                ):
                    raise ValueError("Text artifacts are immutable per run")
                stored_elements = tuple(
                    self._text_element_from_row(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM text_elements
                        WHERE run_id = ?
                        ORDER BY ordinal
                        """,
                        (run_id,),
                    ).fetchall()
                )
                stored_chunks = tuple(
                    self._text_chunk_from_row(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM text_chunks
                        WHERE run_id = ?
                        ORDER BY ordinal
                        """,
                        (run_id,),
                    ).fetchall()
                )
                stored_ocr_pages = tuple(
                    _ocr_page_evidence_from_row(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM ocr_page_evidence
                        WHERE run_id = ?
                        ORDER BY page_number
                        """,
                        (run_id,),
                    ).fetchall()
                )
                if stored_elements != parsed.elements or stored_chunks != chunks:
                    raise ValueError("Text artifacts are immutable per run")
                if stored_ocr_pages != parsed.ocr_pages:
                    raise ValueError("Text artifacts are immutable per run")
                if run.execution_configuration_json and json.loads(
                    run.execution_configuration_json
                ).get("hierarchy_revision"):
                    sqlite_hierarchy.persist(connection, artifact_id, parsed, chunks)
                return artifact_id

            connection.execute(
                """
                INSERT INTO text_artifacts
                    (id, run_id, document_version_id, parser_revision,
                     content_sha256, original_text, normalized_text, warnings,
                     pdf_image_regions, pdf_image_provider, pdf_chart_bars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    run.document_version_id,
                    parsed.parser_revision,
                    parsed.content_sha256,
                    parsed.original_text,
                    parsed.normalized_text,
                    warnings_json,
                    json.dumps(
                        [asdict(item) for item in parsed.pdf_image_regions],
                        separators=(",", ":"),
                    ),
                    parsed.pdf_image_provider,
                    json.dumps(
                        [asdict(item) for item in parsed.pdf_chart_bars],
                        separators=(",", ":"),
                    ),
                ),
            )
            for element in parsed.elements:
                connection.execute(
                    """
                    INSERT INTO text_elements
                        (id, artifact_id, run_id, element_type, ordinal, text,
                         original_text, start_line, end_line, warnings, page_number,
                         table_cells)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        element.id,
                        artifact_id,
                        run_id,
                        element.element_type,
                        element.ordinal,
                        element.text,
                        element.original_text,
                        element.start_line,
                        element.end_line,
                        json.dumps(
                            element.warnings,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        element.page_number,
                        json.dumps(
                            [asdict(cell) for cell in element.table_cells],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                )
            for evidence, payload, payload_sha256 in ocr_payloads:
                connection.execute(
                    """
                    INSERT INTO ocr_page_evidence
                        (artifact_id, run_id, page_number, payload_json, payload_sha256)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        run_id,
                        evidence.page_number,
                        payload,
                        payload_sha256,
                    ),
                )
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO text_chunks
                        (id, artifact_id, run_id, ordinal, text, element_ids,
                         content_sha256, page_numbers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        artifact_id,
                        run_id,
                        chunk.ordinal,
                        chunk.text,
                        json.dumps(
                            chunk.element_ids,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        chunk.content_sha256,
                        json.dumps(
                            chunk.page_numbers,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                )
            if run.execution_configuration_json and json.loads(
                run.execution_configuration_json
            ).get("hierarchy_revision"):
                sqlite_hierarchy.persist(connection, artifact_id, parsed, chunks)
        return artifact_id

    def verify_run_hierarchy(self, run_id):
        from enterprise_rag.persistence import sqlite_hierarchy

        with self._connection() as connection:
            return sqlite_hierarchy.verify(connection, run_id)

    def get_text_artifacts(
        self, run_id: str
    ) -> tuple[ParsedText, tuple[TextChunk, ...]] | None:
        with self._connection() as connection:
            artifact = connection.execute(
                "SELECT * FROM text_artifacts WHERE run_id = ?", (run_id,)
            ).fetchone()
            if artifact is None:
                return None
            elements = tuple(
                self._text_element_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM text_elements
                    WHERE run_id = ?
                    ORDER BY ordinal
                    """,
                    (run_id,),
                ).fetchall()
            )
            chunks = tuple(
                self._text_chunk_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM text_chunks
                    WHERE run_id = ?
                    ORDER BY ordinal
                    """,
                    (run_id,),
                ).fetchall()
            )
            ocr_pages = tuple(
                _ocr_page_evidence_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM ocr_page_evidence
                    WHERE run_id = ?
                    ORDER BY page_number
                    """,
                    (run_id,),
                ).fetchall()
            )
        if any(
            evidence.result.document_version_id != artifact["document_version_id"]
            or evidence.result.source_pdf_sha256 != artifact["content_sha256"]
            for evidence in ocr_pages
        ):
            raise ValueError("Stored OCR page evidence does not match its artifact")
        parsed = ParsedText(
            run_id=artifact["run_id"],
            parser_revision=artifact["parser_revision"],
            content_sha256=artifact["content_sha256"],
            original_text=artifact["original_text"],
            normalized_text=artifact["normalized_text"],
            elements=elements,
            warnings=tuple(json.loads(artifact["warnings"])),
            ocr_pages=ocr_pages,
            pdf_image_regions=pdf_image_regions_from_payload(
                json.loads(artifact["pdf_image_regions"])
            ),
            pdf_image_provider=artifact["pdf_image_provider"],
            pdf_chart_bars=pdf_chart_bars_from_payload(
                json.loads(artifact["pdf_chart_bars"])
            ),
        )
        return parsed, chunks

    def index_run(self, lease: JobLease, now: float) -> int:
        """Build the idempotent SQLite FTS projection for one run."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease_row = connection.execute(
                """
                SELECT tenant_id, run_id
                FROM ingestion_jobs
                WHERE id = ?
                    AND status = ?
                    AND tenant_id = ?
                    AND run_id = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.job.tenant_id,
                    lease.job.run_id,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            ).fetchone()
            if lease_row is None:
                raise ValueError("Job lease is not live for indexing")
            tenant_id = lease_row["tenant_id"]
            run_id = lease_row["run_id"]
            run_row = connection.execute(
                """
                SELECT *
                FROM ingestion_runs
                WHERE id = ? AND tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
            if run_row is None:
                raise ValueError("Ingestion run is not available to this tenant")
            context = connection.execute(
                """
                SELECT d.id AS document_id, v.id AS document_version_id
                FROM ingestion_runs AS r
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                WHERE r.id = ? AND d.tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
            if context is None:
                raise ValueError("Ingestion run has invalid document ownership")
            chunks = connection.execute(
                """
                SELECT *
                FROM text_chunks
                WHERE run_id = ?
                ORDER BY ordinal
                """,
                (run_id,),
            ).fetchall()
            for chunk in chunks:
                existing = connection.execute(
                    """
                    SELECT *
                    FROM lexical_index_records
                    WHERE chunk_id = ? AND run_id = ?
                    """,
                    (chunk["id"], run_id),
                ).fetchone()
                if existing and (
                    existing["tenant_id"] != tenant_id
                    or existing["document_id"] != context["document_id"]
                    or existing["document_version_id"] != context["document_version_id"]
                ):
                    raise ValueError("Lexical index record ownership conflict")
                fts_rows = connection.execute(
                    "SELECT * FROM text_chunks_fts WHERE chunk_id = ?",
                    (chunk["id"],),
                ).fetchall()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO lexical_index_records
                            (chunk_id, run_id, tenant_id, document_id,
                             document_version_id, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk["id"],
                            run_id,
                            tenant_id,
                            context["document_id"],
                            context["document_version_id"],
                            now,
                        ),
                    )
                expected_fts = (
                    chunk["id"],
                    tenant_id,
                    context["document_id"],
                    context["document_version_id"],
                    run_id,
                    chunk["text"],
                )
                if len(fts_rows) != 1 or tuple(fts_rows[0]) != expected_fts:
                    connection.execute(
                        "DELETE FROM text_chunks_fts WHERE chunk_id = ?",
                        (chunk["id"],),
                    )
                    connection.execute(
                        """
                        INSERT INTO text_chunks_fts
                            (chunk_id, tenant_id, document_id,
                             document_version_id, run_id, text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk["id"],
                            tenant_id,
                            context["document_id"],
                            context["document_version_id"],
                            run_id,
                            chunk["text"],
                        ),
                    )
            return len(chunks)

    def verify_run_index(self, tenant_id: str, run_id: str) -> bool:
        """Verify every persisted chunk has matching durable index state."""

        with self._connection() as connection:
            connection.execute("BEGIN")
            run_row = connection.execute(
                """
                SELECT id
                FROM ingestion_runs
                WHERE id = ? AND tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
            if run_row is None:
                return False
            return self._verify_run_index_locked(connection, tenant_id, run_id)

    def activate_run(self, lease: JobLease, now: float) -> bool:
        """Publish one verified latest run without exposing stale content."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease_row = connection.execute(
                """
                SELECT tenant_id, run_id
                FROM ingestion_jobs
                WHERE id = ?
                    AND status = ?
                    AND tenant_id = ?
                    AND run_id = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.job.tenant_id,
                    lease.job.run_id,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            ).fetchone()
            if lease_row is None:
                return False
            tenant_id = lease_row["tenant_id"]
            run_id = lease_row["run_id"]
            run_row = connection.execute(
                """
                SELECT *
                FROM ingestion_runs
                WHERE id = ? AND tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
            if run_row is None:
                return False
            run = self._run_from_row(run_row)
            context = connection.execute(
                """
                SELECT d.*, v.version_number, v.status AS version_status
                FROM ingestion_runs AS r
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN documents AS d ON d.id = v.document_id
                WHERE r.id = ? AND d.tenant_id = ?
                """,
                (run_id, tenant_id),
            ).fetchone()
            if context is None:
                return False
            from enterprise_rag.persistence.sqlite_dense_publication import (
                activation_verified,
            )

            if not activation_verified(connection, lease):
                return False
            if (
                context["active_version_id"] == run.document_version_id
                and context["active_run_id"] == run.id
                and context["version_status"] == VersionStatus.ACTIVE.value
            ):
                return True
            latest_generation = connection.execute(
                "SELECT max(request_generation) FROM ingestion_runs WHERE document_version_id=?",
                (run.document_version_id,),
            ).fetchone()[0]
            if run.request_generation != latest_generation:
                return False
            required_before_activate = [
                stage for stage in run.required_stages if stage != "activate"
            ]
            completed_stages = self._latest_succeeded_stages(
                connection, run.id, lease.job.id
            )
            if set(required_before_activate) - completed_stages:
                return False
            if not self._verify_run_index_locked(connection, tenant_id, run.id):
                return False
            latest_version = connection.execute(
                """
                SELECT MAX(version_number) AS version_number
                FROM document_versions
                WHERE document_id = ?
                """,
                (context["id"],),
            ).fetchone()["version_number"]
            if context["version_number"] != latest_version:
                return False
            if context["status"] in {
                DocumentStatus.DELETING.value,
                DocumentStatus.DELETED.value,
            }:
                return False
            connection.execute(
                """
                UPDATE document_versions
                SET status = ?
                WHERE document_id = ? AND status = ? AND id != ?
                """,
                (
                    VersionStatus.SUPERSEDED.value,
                    context["id"],
                    VersionStatus.ACTIVE.value,
                    run.document_version_id,
                ),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET status = ?
                WHERE id = ? AND document_id = ?
                """,
                (
                    VersionStatus.ACTIVE.value,
                    run.document_version_id,
                    context["id"],
                ),
            )
            connection.execute(
                """
                UPDATE documents
                SET status = ?, active_version_id = ?, active_run_id = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (
                    DocumentStatus.READY.value,
                    run.document_version_id,
                    run.id,
                    context["id"],
                    tenant_id,
                ),
            )
            return True

    def search_lexical(
        self,
        tenant_id: str,
        fts_query: str,
        allowed_document_ids: Collection[str],
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[LexicalRow, ...]:
        """Search only inside an explicit authorization scope."""

        if not fts_query.strip():
            return ()
        if limit <= 0 or limit > 100:
            raise ValueError("Search limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("Search offset must be non-negative")
        document_ids = tuple(dict.fromkeys(allowed_document_ids))
        if not document_ids:
            return ()
        placeholders = ", ".join("?" for _ in document_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT f.chunk_id, r.document_id, r.document_version_id,
                       r.run_id, d.title, v.filename, f.text,
                       bm25(text_chunks_fts) AS score, c.page_numbers,
                       c.logical_chunk_id, c.content_sha256
                FROM text_chunks_fts AS f
                JOIN text_chunks AS c ON c.id = f.chunk_id
                JOIN lexical_index_records AS r ON r.chunk_id = f.chunk_id
                JOIN documents AS d ON d.id = r.document_id
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN ingestion_runs AS ir ON ir.id = r.run_id
                JOIN text_artifacts AS a ON a.id = c.artifact_id
                WHERE text_chunks_fts MATCH ?
                    AND f.tenant_id = ?
                    AND d.tenant_id = ?
                    AND f.tenant_id = r.tenant_id
                    AND f.document_id = r.document_id
                    AND f.document_version_id = r.document_version_id
                    AND f.run_id = r.run_id
                    AND f.text = c.text
                    AND c.run_id = r.run_id
                    AND a.run_id = r.run_id
                    AND a.document_version_id = v.id
                    AND ir.tenant_id = r.tenant_id
                    AND ir.document_version_id = v.id
                    AND v.document_id = d.id
                    AND d.active_version_id = v.id AND d.active_run_id = r.run_id
                    AND v.status = ?
                    AND d.id IN ({placeholders})
                ORDER BY score ASC, f.chunk_id ASC
                LIMIT ? OFFSET ?
                """,
                (
                    fts_query,
                    tenant_id,
                    tenant_id,
                    VersionStatus.ACTIVE.value,
                    *document_ids,
                    limit,
                    offset,
                ),
            ).fetchall()
        return tuple(
            LexicalRow(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_version_id=row["document_version_id"],
                run_id=row["run_id"],
                title=row["title"],
                filename=row["filename"],
                text=row["text"],
                score=float(row["score"]),
                page_numbers=tuple(json.loads(row["page_numbers"])),
            )
            for row in rows
            if self._chunk_text_matches_checksum(row)
        )

    def search_lexical_authorized(
        self,
        scope: SecurityScope,
        fts_query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        vm_step_budget: int | None = None,
        max_text_chars: int | None = None,
    ) -> tuple[LexicalRow, ...]:
        """Search with ACL, tenant and active-version predicates before LIMIT."""

        self._validate_search_bounds(limit, offset)
        if max_text_chars is not None and (
            type(max_text_chars) is not int or not 1 <= max_text_chars <= 65536
        ):
            raise ValueError("Lexical text budget must be between 1 and65536")
        if not fts_query.strip():
            return ()
        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with (
            self._connection() as connection,
            _lexical_read_budget(connection, vm_step_budget),
        ):
            rows = connection.execute(
                f"""
                SELECT f.chunk_id, r.document_id, r.document_version_id,
                       r.run_id, d.title, v.filename, f.text,
                       bm25(text_chunks_fts) AS score, c.page_numbers,
                       c.logical_chunk_id, c.content_sha256
                FROM text_chunks_fts AS f
                JOIN text_chunks AS c ON c.id = f.chunk_id
                JOIN lexical_index_records AS r ON r.chunk_id = f.chunk_id
                JOIN documents AS d ON d.id = r.document_id
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN ingestion_runs AS ir ON ir.id = r.run_id
                JOIN text_artifacts AS a ON a.id = c.artifact_id
                WHERE text_chunks_fts MATCH ?
                    AND f.tenant_id = ?
                    AND d.tenant_id = ?
                    AND f.tenant_id = r.tenant_id
                    AND f.document_id = r.document_id
                    AND f.document_version_id = r.document_version_id
                    AND f.run_id = r.run_id
                    AND f.text = c.text
                    AND c.run_id = r.run_id
                    AND a.run_id = r.run_id
                    AND a.document_version_id = v.id
                    AND ir.tenant_id = r.tenant_id
                    AND ir.document_version_id = v.id
                    AND v.document_id = d.id
                    AND d.active_version_id = v.id AND d.active_run_id = r.run_id
                    AND v.status = ?
                    AND d.status NOT IN (?, ?)
                    AND (? IS NULL OR (
                        instr(f.text, char(0)) = 0 AND length(f.text) <= ?
                    ))
                    AND {acl_sql}
                ORDER BY score ASC, f.chunk_id ASC
                LIMIT ? OFFSET ?
                """,
                (
                    fts_query,
                    scope.tenant_id,
                    scope.tenant_id,
                    VersionStatus.ACTIVE.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    max_text_chars,
                    max_text_chars,
                    *acl_parameters,
                    limit,
                    offset,
                ),
            ).fetchall()
        return tuple(
            self._lexical_row_from_row(row)
            for row in rows
            if self._chunk_text_matches_checksum(row)
        )

    def search_lexical_authorized_with_parent_context(
        self,
        scope: SecurityScope,
        fts_query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        max_parent_chars: int = 12_000,
    ) -> tuple[tuple[LexicalRow, AuthorizedParentContext], ...]:
        """Search and hydrate source-element context in one ACL read snapshot."""

        self._validate_search_bounds(limit, offset)
        if (
            isinstance(max_parent_chars, bool)
            or not isinstance(max_parent_chars, int)
            or not 1 <= max_parent_chars <= 100_000
        ):
            raise ValueError("Parent context limit must be between 1 and 100000")
        if not fts_query.strip():
            return ()

        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                f"""
                SELECT f.chunk_id, r.document_id, r.document_version_id,
                       r.run_id, d.title, v.filename, f.text,
                       bm25(text_chunks_fts) AS score, c.page_numbers,
                       c.logical_chunk_id, c.content_sha256
                FROM text_chunks_fts AS f
                JOIN text_chunks AS c ON c.id = f.chunk_id
                JOIN lexical_index_records AS r ON r.chunk_id = f.chunk_id
                JOIN documents AS d ON d.id = r.document_id
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN ingestion_runs AS ir ON ir.id = r.run_id
                JOIN text_artifacts AS a ON a.id = c.artifact_id
                WHERE text_chunks_fts MATCH ?
                    AND f.tenant_id = ?
                    AND d.tenant_id = ?
                    AND f.document_id = r.document_id
                    AND f.document_version_id = r.document_version_id
                    AND f.run_id = r.run_id
                    AND f.tenant_id = r.tenant_id
                    AND f.text = c.text
                    AND c.run_id = r.run_id
                    AND a.run_id = r.run_id
                    AND a.document_version_id = v.id
                    AND r.document_id = d.id
                    AND r.document_version_id = v.id
                    AND v.document_id = d.id
                    AND ir.tenant_id = r.tenant_id
                    AND ir.document_version_id = v.id
                    AND d.active_version_id = v.id AND d.active_run_id = r.run_id
                    AND v.status = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                ORDER BY score ASC, f.chunk_id ASC
                LIMIT ? OFFSET ?
                """,
                (
                    fts_query,
                    scope.tenant_id,
                    scope.tenant_id,
                    VersionStatus.ACTIVE.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                    limit,
                    offset,
                ),
            ).fetchall()

            hydrated: list[tuple[LexicalRow, AuthorizedParentContext]] = []
            for row in rows:
                if not self._chunk_text_matches_checksum(row):
                    continue
                child = self._lexical_row_from_row(row)
                parent = self._authorized_parent_context_locked(
                    connection,
                    scope,
                    child.chunk_id,
                    max_parent_chars=max_parent_chars,
                )
                hydrated.append((child, parent))
        return tuple(hydrated)

    def _authorized_parent_context_locked(
        self,
        connection: sqlite3.Connection,
        scope: SecurityScope,
        chunk_id: str,
        *,
        max_parent_chars: int,
    ) -> AuthorizedParentContext:
        """Hydrate one source-element group after repeated live ACL checks."""

        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        child = connection.execute(
            f"""
            SELECT c.artifact_id, c.run_id, c.element_ids, c.text,
                   c.page_numbers,
                   c.logical_chunk_id, c.content_sha256, a.id AS verified_artifact_id,
                   a.document_version_id AS artifact_document_version_id,
                   r.tenant_id, r.document_id, r.document_version_id
            FROM text_chunks AS c
            JOIN lexical_index_records AS r
                ON r.chunk_id = c.id AND r.run_id = c.run_id
            JOIN documents AS d ON d.id = r.document_id
            JOIN document_versions AS v ON v.id = r.document_version_id
            JOIN ingestion_runs AS ir
                ON ir.id = c.run_id AND ir.id = r.run_id
            LEFT JOIN text_artifacts AS a
                ON a.id = c.artifact_id AND a.run_id = c.run_id
            WHERE c.id = ?
                AND r.tenant_id = ?
                AND d.tenant_id = ?
                AND r.document_id = d.id
                AND r.document_version_id = v.id
                AND v.document_id = d.id
                AND ir.tenant_id = r.tenant_id
                AND ir.document_version_id = v.id
                AND d.active_version_id = v.id AND d.active_run_id = r.run_id
                AND v.status = ?
                AND d.status NOT IN (?, ?)
                AND {acl_sql}
            """,
            (
                chunk_id,
                scope.tenant_id,
                scope.tenant_id,
                VersionStatus.ACTIVE.value,
                DocumentStatus.DELETING.value,
                DocumentStatus.DELETED.value,
                *acl_parameters,
            ),
        ).fetchone()
        if child is None:
            raise ValueError(
                "Text element provenance does not match authorized child identity"
            )
        if (
            child["verified_artifact_id"] != child["artifact_id"]
            or child["artifact_document_version_id"] != child["document_version_id"]
        ):
            raise ValueError("Text element provenance artifact identity is invalid")

        try:
            element_ids_value = json.loads(child["element_ids"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Text element provenance is invalid") from exc
        if (
            not isinstance(element_ids_value, list)
            or not element_ids_value
            or any(
                not isinstance(element_id, str) or not element_id.strip()
                for element_id in element_ids_value
            )
            or len(set(element_ids_value)) != len(element_ids_value)
        ):
            raise ValueError("Text element provenance is invalid")
        element_ids = tuple(element_ids_value)
        placeholders = ", ".join("?" for _ in element_ids)
        identity = (
            child["artifact_id"],
            child["run_id"],
            *element_ids,
        )
        element_metadata = connection.execute(
            f"""
            SELECT id, ordinal, page_number, length(text) AS text_length,
                   instr(text, char(0)) AS contains_nul
            FROM text_elements
            WHERE artifact_id = ? AND run_id = ?
                AND id IN ({placeholders})
            ORDER BY ordinal
            """,
            identity,
        ).fetchall()
        if tuple(row["id"] for row in element_metadata) != element_ids:
            raise ValueError("Text element provenance is incomplete or out of order")
        if any(row["contains_nul"] for row in element_metadata):
            raise ValueError("Text element provenance contains unsupported NUL")
        element_ordinals = tuple(int(row["ordinal"]) for row in element_metadata)
        if any(
            int(row["text_length"]) <= 0 for row in element_metadata
        ) or element_ordinals != tuple(
            range(element_ordinals[0], element_ordinals[0] + len(element_ids))
        ):
            raise ValueError("Text element provenance is incomplete or out of order")
        if sha256_bytes(child["text"].encode("utf-8")) != child["content_sha256"]:
            raise ValueError("Text chunk provenance checksum is invalid")
        try:
            child_page_numbers = json.loads(child["page_numbers"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Text chunk page provenance is invalid") from exc
        if (
            not isinstance(child_page_numbers, list)
            or any(
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number <= 0
                for page_number in child_page_numbers
            )
            or len(set(child_page_numbers)) != len(child_page_numbers)
        ):
            raise ValueError("Text chunk page provenance is invalid")
        expected_child_pages = tuple(
            dict.fromkeys(
                int(row["page_number"])
                for row in element_metadata
                if row["page_number"] is not None
            )
        )
        if tuple(child_page_numbers) != expected_child_pages:
            raise ValueError(
                "Text chunk page provenance does not match source elements"
            )
        if not self._child_text_matches_source_elements_locked(
            connection,
            artifact_id=child["artifact_id"],
            run_id=child["run_id"],
            child_text=child["text"],
            element_ids=element_ids,
            element_metadata=element_metadata,
        ):
            raise ValueError("Text element provenance does not contain child chunk")

        page_numbers = tuple(
            sorted(
                {
                    int(row["page_number"])
                    for row in element_metadata
                    if row["page_number"] is not None
                }
            )
        )
        estimated_chars = sum(int(row["text_length"]) for row in element_metadata)
        estimated_chars += 2 * (len(element_metadata) - 1)
        parent_id = deterministic_id(
            "source-element-parent-context", child["run_id"], *element_ids
        )
        if estimated_chars > max_parent_chars:
            return AuthorizedParentContext(
                parent_id=parent_id,
                tenant_id=child["tenant_id"],
                document_id=child["document_id"],
                document_version_id=child["document_version_id"],
                run_id=child["run_id"],
                text=None,
                page_numbers=page_numbers,
                oversized=True,
            )

        text_rows = connection.execute(
            f"""
            SELECT id, text
            FROM text_elements
            WHERE artifact_id = ? AND run_id = ?
                AND id IN ({placeholders})
            ORDER BY ordinal
            """,
            identity,
        ).fetchall()
        if tuple(row["id"] for row in text_rows) != element_ids:
            raise ValueError("Text element provenance changed during hydration")
        text = "\n\n".join(row["text"] for row in text_rows)
        if len(text) > max_parent_chars:
            raise ValueError("Text element provenance size changed during hydration")
        if child["text"] not in text:
            raise ValueError("Text element provenance does not contain child chunk")
        return AuthorizedParentContext(
            parent_id=parent_id,
            tenant_id=child["tenant_id"],
            document_id=child["document_id"],
            document_version_id=child["document_version_id"],
            run_id=child["run_id"],
            text=text,
            page_numbers=page_numbers,
            oversized=False,
        )

    @staticmethod
    def _child_text_matches_source_elements_locked(
        connection: sqlite3.Connection,
        *,
        artifact_id: str,
        run_id: str,
        child_text: str,
        element_ids: tuple[str, ...],
        element_metadata: list[sqlite3.Row],
    ) -> bool:
        """Validate exact source-span coverage without returning parent text."""

        child_length = len(child_text)
        if not child_text or child_length > 100_000:
            raise ValueError("Text chunk exceeds the provenance validation bound")

        spans: list[tuple[int, int]] = []
        cursor = 1
        for index, row in enumerate(element_metadata):
            start = cursor
            end = start + int(row["text_length"]) - 1
            spans.append((start, end))
            if index + 1 < len(element_metadata):
                cursor = end + 3

        if len(element_ids) == 1:
            match = connection.execute(
                """
                SELECT instr(text, ?) AS child_position
                FROM text_elements
                WHERE id = ? AND artifact_id = ? AND run_id = ?
                """,
                (child_text, element_ids[0], artifact_id, run_id),
            ).fetchone()
            return match is not None and int(match["child_position"]) > 0

        first_start, first_end = spans[0]
        last_start, last_end = spans[-1]
        earliest_start = max(first_start, last_start - child_length + 1)
        latest_start = min(first_end, last_end - child_length + 1)
        if earliest_start > latest_start:
            return False

        prefix_ids = element_ids[:-1]
        suffix_ids = element_ids[1:]
        prefix_placeholders = ", ".join("?" for _ in prefix_ids)
        suffix_placeholders = ", ".join("?" for _ in suffix_ids)
        ambiguous_match = connection.execute(
            f"""
            WITH referenced_prefix(value) AS (
                SELECT group_concat(prefix_element.text, char(10) || char(10))
                FROM (
                    SELECT text
                    FROM text_elements
                    WHERE artifact_id = ? AND run_id = ?
                        AND id IN ({prefix_placeholders})
                    ORDER BY ordinal
                ) AS prefix_element
            ),
            referenced_suffix(value) AS (
                SELECT group_concat(suffix_element.text, char(10) || char(10))
                FROM (
                    SELECT text
                    FROM text_elements
                    WHERE artifact_id = ? AND run_id = ?
                        AND id IN ({suffix_placeholders})
                    ORDER BY ordinal
                ) AS suffix_element
            )
            SELECT
                EXISTS (
                    SELECT 1 FROM referenced_prefix
                    WHERE instr(value, ?) > 0
                )
                OR EXISTS (
                    SELECT 1 FROM referenced_suffix
                    WHERE instr(value, ?) > 0
                ) AS ambiguous
            """,
            (
                artifact_id,
                run_id,
                *prefix_ids,
                artifact_id,
                run_id,
                *suffix_ids,
                child_text,
                child_text,
            ),
        ).fetchone()
        if ambiguous_match is None or bool(ambiguous_match["ambiguous"]):
            return False

        placeholders = ", ".join("?" for _ in element_ids)
        match = connection.execute(
            f"""
            WITH RECURSIVE parent_text(value) AS (
                SELECT group_concat(source_element.text, char(10) || char(10))
                FROM (
                    SELECT text
                    FROM text_elements
                    WHERE artifact_id = ? AND run_id = ?
                        AND id IN ({placeholders})
                    ORDER BY ordinal
                ) AS source_element
            ),
            candidate_positions(position) AS (
                SELECT ? WHERE ? <= ?
                UNION ALL
                SELECT position + 1
                FROM candidate_positions
                WHERE position < ?
            )
            SELECT EXISTS (
                SELECT 1
                FROM parent_text, candidate_positions
                WHERE substr(value, position, ?) = ?
            ) AS child_matches
            """,
            (
                artifact_id,
                run_id,
                *element_ids,
                earliest_start,
                earliest_start,
                latest_start,
                latest_start,
                child_length,
                child_text,
            ),
        ).fetchone()
        return match is not None and bool(match["child_matches"])

    def get_authorized_evidence(
        self,
        scope: SecurityScope,
        chunk_id: str,
        *,
        connection: sqlite3.Connection | None = None,
        historical: bool = False,
    ) -> LexicalRow | None:
        """Hydrate one chunk only after the same active+ACL checks as search."""

        if not chunk_id.strip():
            raise ValueError("Evidence chunk id must be non-empty")
        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with (
            self._connection()
            if connection is None
            else nullcontext(connection) as connection
        ):
            row = connection.execute(
                f"""
                SELECT f.chunk_id, r.document_id, r.document_version_id,
                       r.run_id, d.title, v.filename, f.text,
                       bm25(text_chunks_fts) AS score, c.page_numbers,
                       c.logical_chunk_id, c.content_sha256
                FROM text_chunks_fts AS f
                JOIN text_chunks AS c ON c.id = f.chunk_id
                JOIN lexical_index_records AS r ON r.chunk_id = f.chunk_id
                JOIN documents AS d ON d.id = r.document_id
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN ingestion_runs AS ir ON ir.id = r.run_id
                JOIN text_artifacts AS a ON a.id = c.artifact_id
                WHERE f.chunk_id = ?
                    AND f.tenant_id = ?
                    AND d.tenant_id = ?
                    AND f.tenant_id = r.tenant_id
                    AND f.document_id = r.document_id
                    AND f.document_version_id = r.document_version_id
                    AND f.run_id = r.run_id
                    AND f.text = c.text
                    AND c.run_id = r.run_id
                    AND a.run_id = r.run_id
                    AND a.document_version_id = v.id
                    AND ir.tenant_id = r.tenant_id
                    AND ir.document_version_id = v.id
                    AND v.document_id = d.id
                    AND d.active_version_id = v.id AND (d.active_run_id = r.run_id OR
                        (? AND ir.status='succeeded' AND EXISTS (
                            SELECT 1 FROM stage_attempts sa WHERE sa.run_id=ir.id
                            AND sa.stage='activate' AND sa.status='succeeded')))
                    AND v.status = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                """,
                (
                    chunk_id,
                    scope.tenant_id,
                    scope.tenant_id,
                    int(historical),
                    VersionStatus.ACTIVE.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                ),
            ).fetchone()
        return (
            None
            if row is None or not self._chunk_text_matches_checksum(row)
            else self._lexical_row_from_row(row)
        )

    def list_authorized_dense_chunk_ids(
        self, scope: SecurityScope, *, limit: int = 1024
    ) -> tuple[str, ...]:
        """Enumerate a complete bounded live view; never truncate membership."""
        if type(limit) is not int or not 1 <= limit <= 2048:
            raise ValueError("Dense membership limit must be1..2048")
        acl_sql, parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                f"""
                SELECT r.chunk_id,
                    CASE WHEN length(CAST(c.text AS BLOB)) > 65536
                        OR instr(c.text,char(0))!=0 THEN 1 ELSE 0 END AS over_budget
                FROM lexical_index_records AS r
                JOIN documents AS d ON d.id=r.document_id AND d.tenant_id=r.tenant_id
                JOIN document_versions AS v ON v.id=r.document_version_id
                    AND v.document_id=d.id
                JOIN text_chunks AS c ON c.id=r.chunk_id AND c.run_id=r.run_id
                WHERE r.tenant_id=? AND d.active_version_id=v.id AND d.active_run_id=r.run_id
                    AND v.status=? AND d.status NOT IN (?,?) AND {acl_sql}
                ORDER BY r.chunk_id LIMIT ?
                """,
                (
                    scope.tenant_id,
                    VersionStatus.ACTIVE.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *parameters,
                    limit + 1,
                ),
            ).fetchall()
        if len(rows) > limit or any(row["over_budget"] for row in rows):
            raise ValueError("Authorized dense view exceeds its count/text budget")
        return tuple(row["chunk_id"] for row in rows)

    def get_authorized_evidence_for_candidates(
        self, scope: SecurityScope, chunk_ids: tuple[str, ...]
    ) -> tuple[LexicalRow, ...]:
        """Hydrate a bounded ordered candidate batch in one live ACL snapshot."""

        if len(chunk_ids) > 100:
            raise ValueError("Candidate batch must contain at most 100 chunks")
        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in chunk_ids
        ):
            raise ValueError("Candidate chunk ids must be non-empty strings")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Candidate batch contains duplicate chunk ids")
        if not chunk_ids:
            return ()

        acl_sql, acl_parameters = self._acl_exists_clause(scope)
        with self._connection() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                f"""
                WITH candidates AS (
                    SELECT CAST(key AS INTEGER) AS candidate_rank, value AS chunk_id
                    FROM json_each(?)
                )
                SELECT f.chunk_id, r.document_id, r.document_version_id,
                       r.run_id, d.title, v.filename, f.text,
                       bm25(text_chunks_fts) AS score, c.page_numbers,
                       c.logical_chunk_id, c.content_sha256, candidates.candidate_rank
                FROM candidates
                JOIN text_chunks_fts AS f
                    ON f.chunk_id = candidates.chunk_id
                JOIN text_chunks AS c ON c.id = f.chunk_id
                JOIN lexical_index_records AS r ON r.chunk_id = f.chunk_id
                JOIN documents AS d ON d.id = r.document_id
                JOIN document_versions AS v
                    ON v.id = r.document_version_id
                JOIN ingestion_runs AS ir ON ir.id = r.run_id
                JOIN text_artifacts AS a ON a.id = c.artifact_id
                WHERE f.tenant_id = ?
                    AND d.tenant_id = ?
                    AND f.tenant_id = r.tenant_id
                    AND f.document_id = r.document_id
                    AND f.document_version_id = r.document_version_id
                    AND f.run_id = r.run_id
                    AND f.text = c.text
                    AND c.run_id = r.run_id
                    AND a.run_id = r.run_id
                    AND a.document_version_id = v.id
                    AND ir.tenant_id = r.tenant_id
                    AND ir.document_version_id = v.id
                    AND v.document_id = d.id
                    AND d.active_version_id = v.id AND d.active_run_id = r.run_id
                    AND v.status = ?
                    AND d.status NOT IN (?, ?)
                    AND {acl_sql}
                ORDER BY candidates.candidate_rank ASC
                """,
                (
                    json.dumps(chunk_ids),
                    scope.tenant_id,
                    scope.tenant_id,
                    VersionStatus.ACTIVE.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                    *acl_parameters,
                ),
            ).fetchall()
        return tuple(
            self._lexical_row_from_row(row)
            for row in rows
            if self._chunk_text_matches_checksum(row)
        )

    def start_stage_attempt(
        self, lease: JobLease, stage: str, now: float
    ) -> StageAttempt | None:
        """Start one durable stage attempt under the current job fence."""

        if not stage.strip():
            raise ValueError("Stage name must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job_row = connection.execute(
                """
                SELECT *
                FROM ingestion_jobs
                WHERE id = ?
                    AND status = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            ).fetchone()
            if job_row is None or job_row["run_id"] is None:
                return None
            run_row = connection.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?", (job_row["run_id"],)
            ).fetchone()
            if run_row is None:
                return None
            run = self._run_from_row(run_row)
            if stage not in run.required_stages:
                raise ValueError(f"Stage {stage!r} is not required by run {run.id}")
            attempt_number = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_number
                FROM stage_attempts
                WHERE run_id = ? AND stage = ?
                """,
                (run.id, stage),
            ).fetchone()["next_number"]
            attempt = StageAttempt(
                id=deterministic_id(
                    "stage-attempt", run.id, stage, int(attempt_number)
                ),
                run_id=run.id,
                job_id=lease.job.id,
                stage=stage,
                attempt_number=int(attempt_number),
                started_at=now,
            )
            connection.execute(
                """
                INSERT INTO stage_attempts
                    (id, run_id, job_id, stage, attempt_number, status,
                     started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.id,
                    attempt.run_id,
                    attempt.job_id,
                    attempt.stage,
                    attempt.attempt_number,
                    attempt.status.value,
                    attempt.started_at,
                ),
            )
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET stage = ?
                WHERE id = ?
                    AND status = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    stage,
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            )
            return attempt

    def finish_stage_attempt(
        self,
        lease: JobLease,
        attempt_id: str,
        now: float,
        *,
        success: bool,
        error: str | None = None,
        output_checksum: str | None = None,
    ) -> bool:
        """Finish a stage only while its job lease remains live and fenced."""

        if not success and (error is None or not error.strip()):
            raise ValueError("Failed stage attempts require an error")
        target_status = StageStatus.SUCCEEDED if success else StageStatus.FAILED
        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE stage_attempts
                SET status = ?, error = ?, output_checksum = ?, finished_at = ?
                WHERE id = ?
                    AND job_id = ?
                    AND status = ?
                    AND EXISTS (
                        SELECT 1
                        FROM ingestion_jobs
                        WHERE id = ?
                            AND status = ?
                            AND lease_owner = ?
                            AND lease_token = ?
                            AND lease_expires_at > ?
                    )
                """,
                (
                    target_status.value,
                    error,
                    output_checksum,
                    now,
                    attempt_id,
                    lease.job.id,
                    StageStatus.RUNNING.value,
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            )
            if updated.rowcount == 1:
                return True
            row = connection.execute(
                """
                SELECT status, error, output_checksum
                FROM stage_attempts
                WHERE id = ? AND job_id = ?
                """,
                (attempt_id, lease.job.id),
            ).fetchone()
        return bool(
            row
            and row["status"] == target_status.value
            and row["error"] == error
            and row["output_checksum"] == output_checksum
        )

    def claim_job(
        self, worker_id: str, now: float, lease_seconds: float
    ) -> JobLease | None:
        """Claim one due job, or reclaim one whose previous lease expired."""

        if not worker_id.strip():
            raise ValueError("Worker id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive")

        expires_at = now + lease_seconds
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM ingestion_jobs AS j
                JOIN document_versions AS v
                    ON v.id = j.document_version_id
                WHERE (
                    j.status IN (?, ?) AND j.next_attempt_at <= ?
                ) OR (
                    j.status = ? AND j.lease_expires_at IS NOT NULL
                    AND j.lease_expires_at <= ?
                )
                ORDER BY j.next_attempt_at, v.version_number DESC, j.id
                LIMIT 1
                """,
                (
                    JobStatus.QUEUED.value,
                    JobStatus.RETRY_WAIT.value,
                    now,
                    JobStatus.RUNNING.value,
                    now,
                ),
            ).fetchone()
            if row is None:
                return None

            updated = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    attempts = attempts + 1,
                    lease_owner = ?,
                    lease_token = lease_token + 1,
                    lease_expires_at = ?,
                    error = NULL
                WHERE id = ?
                    AND (
                        (status IN (?, ?) AND next_attempt_at <= ?)
                        OR (
                            status = ? AND lease_expires_at IS NOT NULL
                            AND lease_expires_at <= ?
                        )
                    )
                """,
                (
                    JobStatus.RUNNING.value,
                    worker_id,
                    expires_at,
                    row["id"],
                    JobStatus.QUEUED.value,
                    JobStatus.RETRY_WAIT.value,
                    now,
                    JobStatus.RUNNING.value,
                    now,
                ),
            )
            if updated.rowcount != 1:
                return None
            reclaimed = row["status"] == JobStatus.RUNNING.value
            connection.execute(
                """
                UPDATE ingestion_runs
                SET status = ?
                WHERE id = (
                    SELECT run_id FROM ingestion_jobs WHERE id = ?
                ) AND status = ?
                """,
                (
                    RunStatus.RUNNING.value,
                    row["id"],
                    RunStatus.QUEUED.value,
                ),
            )
            claimed_row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (row["id"],)
            ).fetchone()

        job = self._job_from_row(claimed_row)
        return JobLease(
            job=job,
            worker_id=worker_id,
            lease_token=job.lease_token,
            lease_expires_at=expires_at,
            reclaimed=reclaimed,
        )

    def renew_job_lease(
        self, lease: JobLease, now: float, lease_seconds: float
    ) -> JobLease | None:
        """Extend a live lease; a stale or fenced worker gets no new lease."""

        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive")
        expires_at = now + lease_seconds
        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE ingestion_jobs
                SET lease_expires_at = ?
                WHERE id = ?
                    AND status = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    expires_at,
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            )
            if updated.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (lease.job.id,)
            ).fetchone()

        return JobLease(
            job=self._job_from_row(row),
            worker_id=lease.worker_id,
            lease_token=lease.lease_token,
            lease_expires_at=expires_at,
            reclaimed=lease.reclaimed,
        )

    @staticmethod
    def _latest_succeeded_stages(
        connection: sqlite3.Connection, run_id: str, job_id: str
    ) -> set[str]:
        """Only the newest attempt for each stage can prove its success."""

        return {
            row["stage"]
            for row in connection.execute(
                """
                SELECT a.stage FROM stage_attempts AS a
                WHERE a.run_id = ? AND a.job_id = ? AND a.status = ?
                    AND NOT EXISTS (
                        SELECT 1 FROM stage_attempts AS newer
                        WHERE newer.run_id = a.run_id AND newer.stage = a.stage
                            AND newer.attempt_number > a.attempt_number
                    )
                """,
                (run_id, job_id, StageStatus.SUCCEEDED.value),
            ).fetchall()
        }

    def complete_job(self, lease: JobLease, now: float) -> bool:
        """Complete a live job; repeated completion is a safe no-op."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (lease.job.id,)
            ).fetchone()
            if current is None or current["run_id"] is None:
                return False
            if current["status"] == JobStatus.SUCCEEDED.value:
                return bool(
                    current["lease_owner"] == lease.worker_id
                    and current["lease_token"] == lease.lease_token
                )
            run_row = connection.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?",
                (current["run_id"],),
            ).fetchone()
            if run_row is None:
                return False
            run = self._run_from_row(run_row)
            completed_stages = self._latest_succeeded_stages(
                connection, run.id, lease.job.id
            )
            if set(run.required_stages) - completed_stages:
                return False
            updated = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    stage = 'done',
                    error = NULL,
                    next_attempt_at = 0,
                    lease_expires_at = NULL
                WHERE id = ?
                    AND status = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            )
            if updated.rowcount == 1:
                connection.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = ?
                    WHERE id = (
                        SELECT run_id FROM ingestion_jobs WHERE id = ?
                    )
                    """,
                    (RunStatus.SUCCEEDED.value, lease.job.id),
                )
                return True
            row = connection.execute(
                "SELECT status, lease_owner, lease_token FROM ingestion_jobs WHERE id = ?",
                (lease.job.id,),
            ).fetchone()
            return bool(
                row
                and row["status"] == JobStatus.SUCCEEDED.value
                and row["lease_owner"] == lease.worker_id
                and row["lease_token"] == lease.lease_token
            )

    def fail_job(
        self,
        lease: JobLease,
        now: float,
        error: str,
        *,
        retryable: bool,
        max_attempts: int,
        retry_delay: float = 0,
    ) -> bool:
        """Record a classified failure while enforcing the active fence."""

        if max_attempts <= 0:
            raise ValueError("Maximum attempts must be positive")
        if retry_delay < 0:
            raise ValueError("Retry delay must be non-negative")
        if not error.strip():
            raise ValueError("Job error must be non-empty")

        should_retry = retryable and lease.job.attempts < max_attempts
        status = JobStatus.RETRY_WAIT if should_retry else JobStatus.FAILED
        next_attempt_at = now + retry_delay if should_retry else 0
        stage = lease.job.stage if should_retry else "failed"
        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    stage = ?,
                    error = ?,
                    next_attempt_at = ?,
                    lease_expires_at = NULL
                WHERE id = ?
                    AND status = ?
                    AND lease_owner = ?
                    AND lease_token = ?
                    AND lease_expires_at > ?
                """,
                (
                    status.value,
                    stage,
                    error,
                    next_attempt_at,
                    lease.job.id,
                    JobStatus.RUNNING.value,
                    lease.worker_id,
                    lease.lease_token,
                    now,
                ),
            )
            if updated.rowcount == 1:
                connection.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = ?
                    WHERE id = (
                        SELECT run_id FROM ingestion_jobs WHERE id = ?
                    )
                    """,
                    (
                        RunStatus.RUNNING.value
                        if should_retry
                        else RunStatus.FAILED.value,
                        lease.job.id,
                    ),
                )
        return updated.rowcount == 1

    def retry_failed_job(
        self,
        tenant_id: str,
        job_id: str,
        now: float,
        *,
        actor: ActorContext | None = None,
    ) -> IngestionJob | None:
        """Requeue one failed job and its run in one fenced metadata transaction."""

        if not tenant_id.strip() or not job_id.strip():
            raise ValueError("Tenant and job ids must be non-empty")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor is not None:
                self._require_tenant_admin_actor(connection, tenant_id, actor)
            row = connection.execute(
                """
                SELECT run_id
                FROM ingestion_jobs
                WHERE id = ?
                    AND tenant_id = ?
                    AND status = ?
                    AND NOT EXISTS (
                        SELECT 1
                        FROM document_versions AS v
                        JOIN documents AS d ON d.id = v.document_id
                        WHERE v.id = ingestion_jobs.document_version_id
                            AND d.status IN (?, ?)
                    )
                """,
                (
                    job_id,
                    tenant_id,
                    JobStatus.FAILED.value,
                    DocumentStatus.DELETING.value,
                    DocumentStatus.DELETED.value,
                ),
            ).fetchone()
            if row is None:
                return None
            if connection.execute(
                """SELECT 1 FROM ingestion_jobs other
                   JOIN document_versions ov ON ov.id=other.document_version_id
                   WHERE other.id!=? AND other.status IN ('queued','running','retry_wait')
                       AND ov.document_id=(
                           SELECT v.document_id FROM ingestion_jobs target
                           JOIN document_versions v ON v.id=target.document_version_id
                           WHERE target.id=? AND target.tenant_id=?) LIMIT 1""",
                (job_id, job_id, tenant_id),
            ).fetchone():
                return None
            updated = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    stage = 'validate',
                    error = NULL,
                    next_attempt_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    JobStatus.QUEUED.value,
                    now,
                    job_id,
                    tenant_id,
                    JobStatus.FAILED.value,
                ),
            )
            if updated.rowcount != 1:
                return None
            if row["run_id"] is not None:
                connection.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = ?
                    WHERE id = ? AND tenant_id = ? AND status = ?
                    """,
                    (
                        RunStatus.QUEUED.value,
                        row["run_id"],
                        tenant_id,
                        RunStatus.FAILED.value,
                    ),
                )
            retried_row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(retried_row)

    def find_version_by_checksum(
        self, document_id: str, content_sha256: str
    ) -> DocumentVersion | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id = ? AND content_sha256 = ?
                """,
                (document_id, content_sha256),
            ).fetchone()
        return self._version_from_row(row) if row else None

    def find_version_by_checksum_for_tenant(
        self, tenant_id: str, document_id: str, content_sha256: str
    ) -> DocumentVersion | None:
        """Find a version only under an explicitly authorized tenant."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT v.*
                FROM document_versions AS v
                JOIN documents AS d ON d.id = v.document_id
                WHERE v.document_id = ?
                    AND d.tenant_id = ?
                    AND v.content_sha256 = ?
                """,
                (document_id, tenant_id, content_sha256),
            ).fetchone()
        return self._version_from_row(row) if row else None

    def next_version_number(self, document_id: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 AS next_number FROM document_versions WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["next_number"])

    def next_version_number_for_tenant(self, tenant_id: str, document_id: str) -> int:
        """Allocate-read the next number only under an explicit tenant scope."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(v.version_number), 0) + 1 AS next_number
                FROM document_versions AS v
                JOIN documents AS d ON d.id = v.document_id
                WHERE v.document_id = ? AND d.tenant_id = ?
                """,
                (document_id, tenant_id),
            ).fetchone()
        return int(row["next_number"])

    def accept_upload(
        self,
        tenant: Tenant,
        source: Source,
        canonical_key: str,
        title: str,
        filename: str,
        content_sha256: str,
        byte_size: int,
        object_key: str,
        *,
        actor: ActorContext | None = None,
    ) -> UploadDecision:
        """Atomically deduplicate, number, and queue one staged upload.

        ``actor=None`` is reserved for trusted internal fixtures/bootstrap code;
        user-facing callers must pass an actor through DocumentUploadService.
        """

        if source.tenant_id != tenant.id:
            raise ValueError("Source and tenant must belong to the same tenant")
        if byte_size < 0:
            raise ValueError("Upload byte size must be non-negative")

        pipeline_fingerprint = "lite-upload-v1"
        document = Document.create(tenant.id, source.id, canonical_key, title)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO tenants (id, name) VALUES (?, ?)",
                (tenant.id, tenant.name),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO sources (id, tenant_id, name, kind)
                VALUES (?, ?, ?, ?)
                """,
                (source.id, source.tenant_id, source.name, source.kind),
            )
            document_row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND tenant_id = ?",
                (document.id, tenant.id),
            ).fetchone()
            if actor is not None:
                if actor.tenant_id != tenant.id or actor.user_id is None:
                    raise PermissionError("Upload actor does not belong to tenant")
                known_user = connection.execute(
                    "SELECT 1 FROM users WHERE id = ? AND tenant_id = ?",
                    (actor.user_id, tenant.id),
                ).fetchone()
                if known_user is None:
                    raise PermissionError("Upload actor is not a known tenant user")
                if document_row is not None:
                    admin = connection.execute(
                        "SELECT 1 FROM tenant_admins WHERE tenant_id = ? AND user_id = ?",
                        (tenant.id, actor.user_id),
                    ).fetchone()
                    if admin is None:
                        raise PermissionError("Existing document requires tenant admin")
            if document_row:
                document = self._document_from_row(document_row)

            if document.source_id != source.id:
                raise ValueError("Document and source must be consistent")
            if document.status in {DocumentStatus.DELETING, DocumentStatus.DELETED}:
                raise ValueError(
                    f"Document {document.id} is not accepting uploads from "
                    f"{document.status}"
                )

            version_row = connection.execute(
                """
                SELECT *
                FROM document_versions
                WHERE document_id = ? AND content_sha256 = ?
                """,
                (document.id, content_sha256),
            ).fetchone()
            if version_row:
                job_row = connection.execute(
                    """
                    SELECT *
                    FROM ingestion_jobs
                    WHERE document_version_id = ? AND tenant_id = ? ORDER BY rowid DESC LIMIT 1
                    """,
                    (version_row["id"], tenant.id),
                ).fetchone()
                return UploadDecision(
                    document=document,
                    version=self._version_from_row(version_row),
                    job=self._job_from_row(job_row) if job_row else None,
                    duplicate=True,
                )

            self._assert_object_key_available_for_upload(
                connection, object_key, document_id=document.id
            )
            if document.status is not DocumentStatus.INGESTING:
                transition_document(document, DocumentStatus.INGESTING)
            next_number = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1 AS next_number
                FROM document_versions
                WHERE document_id = ?
                """,
                (document.id,),
            ).fetchone()["next_number"]
            version = DocumentVersion.from_digest(
                document_id=document.id,
                version_number=int(next_number),
                filename=filename,
                content_sha256=content_sha256,
                byte_size=byte_size,
                object_key=object_key,
            )
            transition_version(version, VersionStatus.INGESTING)
            run = IngestionRun.create(tenant.id, version.id, pipeline_fingerprint)
            job = IngestionJob.create(tenant.id, version.id, run_id=run.id)

            connection.execute(
                """
                INSERT INTO documents
                    (id, tenant_id, source_id, canonical_key, title, status,
                     active_version_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_id = excluded.source_id,
                    title = excluded.title,
                    status = excluded.status
                """,
                (
                    document.id,
                    document.tenant_id,
                    document.source_id,
                    document.canonical_key,
                    document.title,
                    document.status.value,
                    document.active_version_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO document_versions
                    (id, document_id, version_number, filename, content_sha256,
                     byte_size, object_key, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.id,
                    version.document_id,
                    version.version_number,
                    version.filename,
                    version.content_sha256,
                    version.byte_size,
                    version.object_key,
                    version.status.value,
                ),
            )
            connection.execute(
                """
                INSERT INTO ingestion_runs
                    (id, tenant_id, document_version_id, pipeline_fingerprint,
                     required_stages, status, request_generation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.tenant_id,
                    run.document_version_id,
                    run.pipeline_fingerprint,
                    json.dumps(run.required_stages, separators=(",", ":")),
                    run.status.value,
                    run.request_generation,
                ),
            )
            connection.execute(
                """
                INSERT INTO ingestion_jobs
                    (id, tenant_id, document_version_id, status, attempts, error,
                     run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.tenant_id,
                    job.document_version_id,
                    job.status.value,
                    job.attempts,
                    job.error,
                    job.run_id,
                ),
            )
            return UploadDecision(
                document=document, version=version, job=job, duplicate=False
            )

    def persist_upload(
        self,
        tenant: Tenant,
        source: Source,
        document: Document,
        version: DocumentVersion,
        job: IngestionJob,
    ) -> None:
        """Persist the document, version, and queued job in one transaction."""

        if source.tenant_id != tenant.id or document.tenant_id != tenant.id:
            raise ValueError("Tenant, source, and document must be consistent")
        if document.source_id != source.id:
            raise ValueError("Document and source must be consistent")
        if version.document_id != document.id:
            raise ValueError("Version and document must be consistent")
        if job.tenant_id != tenant.id or job.document_version_id != version.id:
            raise ValueError("Job, tenant, and version must be consistent")

        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tenants (id, name) VALUES (?, ?)",
                (tenant.id, tenant.name),
            )
            connection.execute(
                "INSERT OR IGNORE INTO sources (id, tenant_id, name, kind) VALUES (?, ?, ?, ?)",
                (source.id, source.tenant_id, source.name, source.kind),
            )
            existing_document = connection.execute(
                "SELECT status FROM documents WHERE id = ? AND tenant_id = ?",
                (document.id, tenant.id),
            ).fetchone()
            if existing_document is not None and existing_document["status"] in {
                DocumentStatus.DELETING.value,
                DocumentStatus.DELETED.value,
            }:
                raise ValueError(
                    f"Document {document.id} is not accepting uploads from "
                    f"{existing_document['status']}"
                )
            self._assert_object_key_available_for_upload(
                connection, version.object_key, document_id=document.id
            )
            connection.execute(
                """
                INSERT INTO documents
                    (id, tenant_id, source_id, canonical_key, title, status, active_version_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_id = excluded.source_id,
                    title = excluded.title,
                    status = excluded.status
                """,
                (
                    document.id,
                    document.tenant_id,
                    document.source_id,
                    document.canonical_key,
                    document.title,
                    document.status.value,
                    document.active_version_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO document_versions
                    (id, document_id, version_number, filename, content_sha256, byte_size, object_key, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.id,
                    version.document_id,
                    version.version_number,
                    version.filename,
                    version.content_sha256,
                    version.byte_size,
                    version.object_key,
                    version.status.value,
                ),
            )
            connection.execute(
                """
                INSERT INTO ingestion_jobs
                    (id, tenant_id, document_version_id, status, attempts, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.tenant_id,
                    job.document_version_id,
                    job.status.value,
                    job.attempts,
                    job.error,
                ),
            )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> Document:
        keys = row.keys()
        ingestion_warnings = (
            json.loads(row["ingestion_warnings"])
            if "ingestion_warnings" in keys
            else []
        )
        if not isinstance(ingestion_warnings, list) or any(
            not isinstance(warning, str) or not 1 <= len(warning) <= 128
            for warning in ingestion_warnings
        ):
            raise ValueError("Invalid document ingestion warnings")
        return Document(
            id=row["id"],
            tenant_id=row["tenant_id"],
            source_id=row["source_id"],
            canonical_key=row["canonical_key"],
            title=row["title"],
            status=DocumentStatus(row["status"]),
            active_version_id=row["active_version_id"],
            ingestion_warnings=tuple(ingestion_warnings),
        )

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> User:
        return User(id=row["id"], tenant_id=row["tenant_id"], email=row["email"])

    @staticmethod
    def _group_from_row(row: sqlite3.Row) -> Group:
        return Group(id=row["id"], tenant_id=row["tenant_id"], name=row["name"])

    @staticmethod
    def _version_from_row(row: sqlite3.Row) -> DocumentVersion:
        return DocumentVersion(
            id=row["id"],
            document_id=row["document_id"],
            version_number=row["version_number"],
            filename=row["filename"],
            content_sha256=row["content_sha256"],
            byte_size=row["byte_size"],
            object_key=row["object_key"],
            status=VersionStatus(row["status"]),
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            id=row["id"],
            tenant_id=row["tenant_id"],
            document_version_id=row["document_version_id"],
            status=JobStatus(row["status"]),
            attempts=row["attempts"],
            error=row["error"],
            stage=row["stage"],
            next_attempt_at=row["next_attempt_at"],
            lease_owner=row["lease_owner"],
            lease_token=row["lease_token"],
            lease_expires_at=row["lease_expires_at"],
            run_id=row["run_id"],
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> IngestionRun:
        required_stages = tuple(json.loads(row["required_stages"]))
        return IngestionRun(
            id=row["id"],
            tenant_id=row["tenant_id"],
            document_version_id=row["document_version_id"],
            pipeline_fingerprint=row["pipeline_fingerprint"],
            required_stages=required_stages,
            status=RunStatus(row["status"]),
            request_generation=row["request_generation"],
            created_at=row["created_at"],
            parent_run_id=row["parent_run_id"],
            reprocess_request_key=row["reprocess_request_key"],
            execution_configuration_json=row["execution_configuration_json"],
        )

    @staticmethod
    def _stage_attempt_from_row(row: sqlite3.Row) -> StageAttempt:
        return StageAttempt(
            id=row["id"],
            run_id=row["run_id"],
            job_id=row["job_id"],
            stage=row["stage"],
            attempt_number=row["attempt_number"],
            status=StageStatus(row["status"]),
            error=row["error"],
            output_checksum=row["output_checksum"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    @staticmethod
    def _audit_finding_from_row(row: sqlite3.Row) -> IngestionAuditFinding:
        return IngestionAuditFinding(
            code=row["code"],
            severity=row["severity"],
            job_id=row["job_id"],
            run_id=row["run_id"],
            document_version_id=row["document_version_id"],
            stage=row["stage"],
        )

    @staticmethod
    def _text_element_from_row(row: sqlite3.Row) -> TextElement:
        return TextElement(
            id=row["id"],
            run_id=row["run_id"],
            element_type=row["element_type"],
            ordinal=row["ordinal"],
            text=row["text"],
            original_text=row["original_text"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            warnings=tuple(json.loads(row["warnings"])),
            page_number=row["page_number"],
            table_cells=pdf_table_cells_from_payload(json.loads(row["table_cells"])),
        )

    @staticmethod
    def _text_chunk_from_row(row: sqlite3.Row) -> TextChunk:
        return TextChunk(
            id=row["id"],
            run_id=row["run_id"],
            ordinal=row["ordinal"],
            text=row["text"],
            element_ids=tuple(json.loads(row["element_ids"])),
            content_sha256=row["content_sha256"],
            page_numbers=tuple(json.loads(row["page_numbers"])),
        )

    @staticmethod
    def _chunk_text_matches_checksum(row: sqlite3.Row) -> bool:
        return sha256_bytes(row["text"].encode("utf-8")) == row["content_sha256"]

    @staticmethod
    def _lexical_row_from_row(row: sqlite3.Row) -> LexicalRow:
        return LexicalRow(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_version_id=row["document_version_id"],
            run_id=row["run_id"],
            title=row["title"],
            filename=row["filename"],
            text=row["text"],
            score=float(row["score"]),
            page_numbers=tuple(json.loads(row["page_numbers"])),
            logical_chunk_id=row["logical_chunk_id"],
        )

    @staticmethod
    def _validate_search_bounds(limit: int, offset: int) -> None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > 100
        ):
            raise ValueError("Search limit must be between 1 and 100")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Search offset must be non-negative")

    @staticmethod
    def _validate_management_bounds(limit: int, offset: int) -> None:
        if limit <= 0 or limit > 100:
            raise ValueError("Management limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("Management offset must be non-negative")

    @staticmethod
    def _acl_exists_clause(
        scope: SecurityScope,
    ) -> tuple[str, tuple[object, ...]]:
        principal_clauses = [
            "(p.principal_type = ? AND p.principal_id = ?)",
            "(p.principal_type = ? AND p.principal_id = ?)",
        ]
        parameters: list[object] = [
            Permission.READ.value,
            Permission.ADMIN.value,
            PrincipalType.TENANT.value,
            scope.tenant_id,
            PrincipalType.USER.value,
            scope.user_id,
        ]
        if scope.group_ids:
            placeholders = ", ".join("?" for _ in scope.group_ids)
            principal_clauses.append(
                f"(p.principal_type = ? AND p.principal_id IN ({placeholders}) "
                "AND EXISTS ("
                "SELECT 1 FROM group_memberships AS gm "
                "WHERE gm.tenant_id = p.tenant_id "
                "AND gm.group_id = p.principal_id AND gm.user_id = ?))"
            )
            parameters.extend(
                [PrincipalType.GROUP.value, *scope.group_ids, scope.user_id]
            )
        return (
            "EXISTS ("
            "SELECT 1 FROM document_permissions AS p "
            "WHERE p.document_id = d.id "
            "AND p.tenant_id = d.tenant_id "
            "AND p.permission IN (?, ?) "
            "AND (" + " OR ".join(principal_clauses) + "))",
            tuple(parameters),
        )

    @staticmethod
    def _verify_run_index_locked(
        connection: sqlite3.Connection, tenant_id: str, run_id: str
    ) -> bool:
        from enterprise_rag.persistence import sqlite_hierarchy

        configuration = connection.execute(
            "SELECT execution_configuration_json FROM ingestion_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if (
            configuration
            and configuration[0]
            and json.loads(configuration[0]).get("hierarchy_revision")
        ):
            if not sqlite_hierarchy.verify(connection, run_id):
                return False
        chunk_count = connection.execute(
            "SELECT COUNT(*) AS count FROM text_chunks WHERE run_id = ?",
            (run_id,),
        ).fetchone()["count"]
        record_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM lexical_index_records
            WHERE run_id = ? AND tenant_id = ?
            """,
            (run_id, tenant_id),
        ).fetchone()["count"]
        if chunk_count == 0 or chunk_count != record_count:
            return False
        chunks = connection.execute(
            "SELECT text, content_sha256 FROM text_chunks WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        if any(
            sha256_bytes(chunk["text"].encode("utf-8")) != chunk["content_sha256"]
            for chunk in chunks
        ):
            return False
        projection_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM text_chunks_fts AS f
            JOIN text_chunks AS c ON c.id = f.chunk_id
            WHERE c.run_id = ?
            """,
            (run_id,),
        ).fetchone()["count"]
        if projection_count != chunk_count:
            return False
        missing = connection.execute(
            """
            SELECT 1
            FROM text_chunks AS c
            WHERE c.run_id = ?
                AND NOT EXISTS (
                    SELECT 1
                    FROM lexical_index_records AS r
                    JOIN ingestion_runs AS ir ON ir.id = r.run_id
                    JOIN document_versions AS v
                        ON v.id = r.document_version_id
                    JOIN documents AS d ON d.id = r.document_id
                    JOIN text_artifacts AS a ON a.id = c.artifact_id
                    JOIN text_chunks_fts AS f ON f.chunk_id = c.id
                    WHERE r.chunk_id = c.id
                        AND r.tenant_id = ?
                        AND r.run_id = c.run_id
                        AND ir.tenant_id = r.tenant_id
                        AND ir.document_version_id = v.id
                        AND d.tenant_id = r.tenant_id
                        AND v.document_id = d.id
                        AND a.run_id = r.run_id
                        AND a.document_version_id = v.id
                        AND f.tenant_id = r.tenant_id
                        AND f.document_id = r.document_id
                        AND f.document_version_id = v.id
                        AND f.run_id = r.run_id
                        AND f.text = c.text
                )
            LIMIT 1
            """,
            (run_id, tenant_id),
        ).fetchone()
        return missing is None
