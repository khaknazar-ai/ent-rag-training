"""Versioned SQLite schema migrations for the resource-aware lite profile."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    sql: str
    rebuild_referenced_tables: bool = False

    @property
    def checksum(self) -> str:
        return sha256(self.sql.encode("utf-8")).hexdigest()


SQLITE_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        """
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            UNIQUE (id, tenant_id)
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            source_id TEXT NOT NULL,
            canonical_key TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN
                ('registered', 'ingesting', 'ready', 'failed', 'deleting', 'deleted')
            ),
            active_version_id TEXT NULL,
            UNIQUE (tenant_id, source_id, canonical_key),
            FOREIGN KEY (source_id, tenant_id)
                REFERENCES sources(id, tenant_id)
        );

        CREATE TABLE IF NOT EXISTS document_versions (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id),
            version_number INTEGER NOT NULL CHECK (version_number > 0),
            filename TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
            object_key TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN
                ('uploaded', 'ingesting', 'ready', 'active', 'superseded',
                 'failed', 'deleting', 'deleted')
            ),
            UNIQUE (document_id, content_sha256),
            UNIQUE (document_id, version_number)
        );

        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            document_version_id TEXT NOT NULL REFERENCES document_versions(id),
            status TEXT NOT NULL CHECK (status IN
                ('queued', 'running', 'retry_wait', 'succeeded', 'failed',
                 'cancelled')
            ),
            attempts INTEGER NOT NULL CHECK (attempts >= 0),
            error TEXT NULL,
            UNIQUE (document_version_id)
        );
        """,
    ),
    Migration(
        2,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_document_versions_one_active
        ON document_versions(document_id)
        WHERE status = 'active';
        """,
    ),
    Migration(
        3,
        """
        ALTER TABLE ingestion_jobs
            ADD COLUMN stage TEXT NOT NULL DEFAULT 'validate';
        ALTER TABLE ingestion_jobs
            ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0;
        ALTER TABLE ingestion_jobs
            ADD COLUMN lease_owner TEXT NULL;
        ALTER TABLE ingestion_jobs
            ADD COLUMN lease_token INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE ingestion_jobs
            ADD COLUMN lease_expires_at REAL NULL;
        """,
    ),
    Migration(
        4,
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            document_version_id TEXT NOT NULL
                REFERENCES document_versions(id),
            pipeline_fingerprint TEXT NOT NULL,
            required_stages TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN
                ('queued', 'running', 'succeeded', 'failed', 'cancelled')
            ),
            request_generation INTEGER NOT NULL CHECK (request_generation > 0),
            created_at REAL NOT NULL DEFAULT (unixepoch()),
            UNIQUE (document_version_id, pipeline_fingerprint)
        );

        ALTER TABLE ingestion_jobs
            ADD COLUMN run_id TEXT NULL;

        CREATE TABLE IF NOT EXISTS stage_attempts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
            job_id TEXT NOT NULL REFERENCES ingestion_jobs(id),
            stage TEXT NOT NULL,
            attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
            status TEXT NOT NULL CHECK (status IN
                ('running', 'succeeded', 'failed')
            ),
            error TEXT NULL,
            output_checksum TEXT NULL,
            started_at REAL NOT NULL,
            finished_at REAL NULL,
            UNIQUE (run_id, stage, attempt_number)
        );

        CREATE INDEX IF NOT EXISTS ix_ingestion_runs_tenant
            ON ingestion_runs(tenant_id, document_version_id);
        CREATE INDEX IF NOT EXISTS ix_stage_attempts_job
            ON stage_attempts(job_id, stage, attempt_number);
        """,
    ),
    Migration(
        5,
        """
        CREATE TABLE IF NOT EXISTS text_artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
            document_version_id TEXT NOT NULL
                REFERENCES document_versions(id),
            parser_revision TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            original_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            warnings TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT (unixepoch()),
            UNIQUE (run_id)
        );

        CREATE TABLE IF NOT EXISTS text_elements (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES text_artifacts(id),
            run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
            element_type TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            text TEXT NOT NULL,
            original_text TEXT NOT NULL,
            start_line INTEGER NOT NULL CHECK (start_line > 0),
            end_line INTEGER NOT NULL CHECK (end_line >= start_line),
            warnings TEXT NOT NULL,
            UNIQUE (run_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS text_chunks (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES text_artifacts(id),
            run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            text TEXT NOT NULL,
            element_ids TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            UNIQUE (run_id, ordinal)
        );

        CREATE INDEX IF NOT EXISTS ix_text_elements_run
            ON text_elements(run_id, ordinal);
        CREATE INDEX IF NOT EXISTS ix_text_chunks_run
            ON text_chunks(run_id, ordinal);
        """,
    ),
    Migration(
        6,
        """
        CREATE TABLE IF NOT EXISTS lexical_index_records (
            chunk_id TEXT PRIMARY KEY REFERENCES text_chunks(id),
            run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            document_id TEXT NOT NULL REFERENCES documents(id),
            document_version_id TEXT NOT NULL
                REFERENCES document_versions(id),
            indexed_at REAL NOT NULL DEFAULT (unixepoch()),
            UNIQUE (run_id, chunk_id)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            tenant_id UNINDEXED,
            document_id UNINDEXED,
            document_version_id UNINDEXED,
            run_id UNINDEXED,
            text,
            tokenize = 'unicode61'
        );

        CREATE INDEX IF NOT EXISTS ix_lexical_index_records_scope
            ON lexical_index_records(tenant_id, document_id, document_version_id);
        CREATE INDEX IF NOT EXISTS ix_lexical_index_records_run
            ON lexical_index_records(run_id);
        """,
    ),
    Migration(
        7,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_id_tenant
            ON documents(id, tenant_id);

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            email TEXT NOT NULL,
            UNIQUE (id, tenant_id),
            UNIQUE (tenant_id, email)
        );

        CREATE TABLE IF NOT EXISTS groups (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            name TEXT NOT NULL,
            UNIQUE (id, tenant_id),
            UNIQUE (tenant_id, name)
        );

        CREATE TABLE IF NOT EXISTS group_memberships (
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            user_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            PRIMARY KEY (user_id, group_id),
            FOREIGN KEY (user_id, tenant_id)
                REFERENCES users(id, tenant_id),
            FOREIGN KEY (group_id, tenant_id)
                REFERENCES groups(id, tenant_id)
        );

        CREATE TABLE IF NOT EXISTS document_permissions (
            document_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            principal_type TEXT NOT NULL CHECK (principal_type IN
                ('user', 'group', 'tenant')),
            principal_id TEXT NOT NULL,
            permission TEXT NOT NULL CHECK (permission IN ('read', 'admin')),
            PRIMARY KEY (document_id, principal_type, principal_id, permission),
            FOREIGN KEY (document_id, tenant_id)
                REFERENCES documents(id, tenant_id)
        );

        CREATE TABLE IF NOT EXISTS tenant_authorization_revisions (
            tenant_id TEXT PRIMARY KEY REFERENCES tenants(id),
            revision INTEGER NOT NULL CHECK (revision >= 0)
        );

        CREATE INDEX IF NOT EXISTS ix_group_memberships_user
            ON group_memberships(tenant_id, user_id, group_id);
        CREATE INDEX IF NOT EXISTS ix_document_permissions_lookup
            ON document_permissions(tenant_id, document_id, principal_type,
                                    principal_id, permission);
        """,
    ),
    Migration(
        8,
        """
        ALTER TABLE text_elements ADD COLUMN page_number INTEGER
            CHECK (page_number IS NULL OR page_number > 0);

        ALTER TABLE text_chunks ADD COLUMN page_numbers TEXT NOT NULL DEFAULT '[]';
        """,
    ),
    Migration(
        9,
        """
        CREATE TABLE IF NOT EXISTS tenant_admins (
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            user_id TEXT NOT NULL,
            PRIMARY KEY (tenant_id, user_id),
            FOREIGN KEY (user_id, tenant_id)
                REFERENCES users(id, tenant_id)
        );

        CREATE INDEX IF NOT EXISTS ix_tenant_admins_user
            ON tenant_admins(tenant_id, user_id);
        """,
    ),
    Migration(
        10,
        """
        CREATE TABLE IF NOT EXISTS upload_attempts (
            object_key TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            document_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN
                ('pending', 'committed', 'abandoned')),
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL CHECK (expires_at >= created_at)
        );

        CREATE INDEX IF NOT EXISTS ix_upload_attempts_gc
            ON upload_attempts(tenant_id, status, expires_at);

        CREATE TABLE IF NOT EXISTS object_cleanup_claims (
            object_key TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            document_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN
                ('claimed', 'deleted', 'released')),
            claimed_at REAL NOT NULL,
            claim_expires_at REAL NOT NULL
                CHECK (claim_expires_at >= claimed_at)
        );

        CREATE INDEX IF NOT EXISTS ix_object_cleanup_claims_active
            ON object_cleanup_claims(tenant_id, status, claim_expires_at);
        """,
    ),
    Migration(
        11,
        """
        CREATE TABLE IF NOT EXISTS maintenance_leases (
            tenant_id TEXT PRIMARY KEY REFERENCES tenants(id),
            owner_id TEXT NOT NULL,
            lease_token INTEGER NOT NULL CHECK (lease_token > 0),
            expires_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ix_maintenance_leases_expiry
            ON maintenance_leases(expires_at);
        """,
    ),
    Migration(
        12,
        """
        ALTER TABLE object_cleanup_claims
            ADD COLUMN claim_token TEXT NOT NULL DEFAULT '';

        UPDATE object_cleanup_claims
        SET claim_token = lower(hex(randomblob(16)))
        WHERE claim_token = '';

        CREATE TRIGGER IF NOT EXISTS trg_object_cleanup_claims_token_insert
        BEFORE INSERT ON object_cleanup_claims
        WHEN NEW.claim_token IS NULL OR length(trim(NEW.claim_token)) = 0
        BEGIN
            SELECT RAISE(ABORT, 'cleanup claim token required');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_object_cleanup_claims_token_update
        BEFORE UPDATE OF claim_token ON object_cleanup_claims
        WHEN NEW.claim_token IS NULL OR length(trim(NEW.claim_token)) = 0
        BEGIN
            SELECT RAISE(ABORT, 'cleanup claim token required');
        END;
        """,
    ),
    Migration(
        13,
        """
        CREATE INDEX IF NOT EXISTS ix_document_versions_object_key
            ON document_versions(object_key);

        CREATE TABLE IF NOT EXISTS document_deletion_claims (
            document_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            claim_token TEXT NOT NULL CHECK (length(claim_token) = 32),
            claimed_at REAL NOT NULL,
            expires_at REAL NOT NULL CHECK (expires_at > claimed_at),
            FOREIGN KEY (document_id, tenant_id)
                REFERENCES documents(id, tenant_id)
        );

        CREATE INDEX IF NOT EXISTS ix_document_deletion_claims_expiry
            ON document_deletion_claims(tenant_id, expires_at);
        """,
    ),
    Migration(
        14,
        """
        CREATE TABLE IF NOT EXISTS cleanup_audit_cursors (
            tenant_id TEXT PRIMARY KEY REFERENCES tenants(id),
            after_object_key TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS ix_object_cleanup_claims_deleted_key
            ON object_cleanup_claims(tenant_id, status, object_key);
        """,
    ),
    Migration(
        15,
        """
        CREATE TABLE maintenance_lease_counters (
            tenant_id TEXT PRIMARY KEY REFERENCES tenants(id),
            lease_token INTEGER NOT NULL CHECK (lease_token > 0)
        );

        INSERT INTO maintenance_lease_counters (tenant_id, lease_token)
        SELECT tenant_id, lease_token FROM maintenance_leases;
        """,
    ),
    Migration(
        16,
        """
        CREATE TABLE query_runs (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            query_text TEXT NOT NULL CHECK(length(query_text) BETWEEN 1 AND 500),
            retrieval_query TEXT NOT NULL CHECK(length(retrieval_query) BETWEEN 1 AND 500),
            normalized_query TEXT NOT NULL CHECK(length(normalized_query) BETWEEN 1 AND 10000),
            retrieval_strategy TEXT NOT NULL,
            model_configuration TEXT NOT NULL CHECK(json_valid(model_configuration)),
            configuration_sha256 TEXT NOT NULL CHECK(length(configuration_sha256)=64),
            status TEXT NOT NULL CHECK(status IN
                ('running','answered','abstained','evidence_only','failed')),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            elapsed_seconds REAL CHECK(elapsed_seconds >= 0),
            error_code TEXT CHECK(error_code IN ('execution_failed','history_write_failed')),
            usage_json TEXT DEFAULT NULL CHECK(usage_json IS NULL OR json_valid(usage_json)),
            cost_usd REAL DEFAULT NULL CHECK(cost_usd >= 0),
            UNIQUE(id, tenant_id, user_id),
            FOREIGN KEY(user_id, tenant_id) REFERENCES users(id, tenant_id),
            CHECK((status='running' AND finished_at IS NULL AND elapsed_seconds IS NULL)
                OR (status!='running' AND finished_at IS NOT NULL AND elapsed_seconds IS NOT NULL)),
            CHECK((status='failed' AND error_code IS NOT NULL)
                OR (status!='failed' AND error_code IS NULL))
        );
        CREATE INDEX ix_query_runs_owner ON query_runs(tenant_id,user_id,started_at);
        CREATE TABLE query_answers (
            id TEXT PRIMARY KEY,
            query_run_id TEXT NOT NULL UNIQUE,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            answer_text TEXT,
            abstained INTEGER NOT NULL CHECK(abstained IN (0,1)),
            abstention_reason TEXT,
            claims_json TEXT DEFAULT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(id,tenant_id,user_id),
            FOREIGN KEY(query_run_id,tenant_id,user_id)
                REFERENCES query_runs(id,tenant_id,user_id),
            CHECK((abstained=1 AND answer_text IS NULL AND abstention_reason IS NOT NULL)
                OR (abstained=0 AND answer_text IS NOT NULL AND abstention_reason IS NULL))
        );
        -- Canonical source IDs are immutable provenance snapshots, not FKs that
        -- could prevent document cleanup or cascade away the query outcome.
        -- Current evidence/ACL is checked on every write/read by the adapter.
        CREATE TABLE query_citations (
            answer_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            citation_index INTEGER NOT NULL CHECK(citation_index BETWEEN 1 AND 4),
            evidence_type TEXT NOT NULL CHECK(evidence_type='text_chunk'),
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            document_version_id TEXT NOT NULL,
            ingestion_run_id TEXT NOT NULL,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            page_numbers TEXT NOT NULL CHECK(json_valid(page_numbers)),
            quoted_span TEXT NOT NULL CHECK(length(quoted_span) BETWEEN 1 AND 600),
            start_char INTEGER NOT NULL CHECK(start_char >= 0),
            end_char INTEGER NOT NULL CHECK(end_char > start_char),
            offset_unit TEXT NOT NULL CHECK(offset_unit='unicode_code_points'),
            source_sha256 TEXT NOT NULL CHECK(length(source_sha256)=64),
            visual_region_id TEXT DEFAULT NULL CHECK(visual_region_id IS NULL),
            bbox TEXT DEFAULT NULL CHECK(bbox IS NULL),
            support_score REAL DEFAULT NULL CHECK(support_score IS NULL),
            PRIMARY KEY(answer_id,citation_index),
            FOREIGN KEY(answer_id,tenant_id,user_id)
                REFERENCES query_answers(id,tenant_id,user_id)
        );
        """,
    ),
    Migration(
        17,
        """
        ALTER TABLE query_runs ADD COLUMN scope_json TEXT DEFAULT NULL
            CHECK(scope_json IS NULL OR json_valid(scope_json));
        ALTER TABLE query_answers ADD COLUMN summary_json TEXT DEFAULT NULL
            CHECK(summary_json IS NULL OR json_valid(summary_json));
        """,
    ),
    Migration(
        18,
        """
        ALTER TABLE query_answers ADD COLUMN result_kind TEXT NOT NULL DEFAULT 'extractive'
            CHECK(result_kind IN ('extractive','model_summary'));
        ALTER TABLE query_answers ADD COLUMN model_summary_json TEXT DEFAULT NULL
            CHECK(model_summary_json IS NULL OR json_valid(model_summary_json));
        """,
    ),
    Migration(
        19,
        """
        ALTER TABLE ingestion_runs ADD COLUMN execution_configuration_json TEXT
            CHECK(execution_configuration_json IS NULL
                  OR json_valid(execution_configuration_json));
        ALTER TABLE ingestion_runs ADD COLUMN execution_configuration_sha256 TEXT
            CHECK((execution_configuration_json IS NULL
                   AND execution_configuration_sha256 IS NULL)
                  OR (execution_configuration_json IS NOT NULL
                      AND execution_configuration_sha256 IS NOT NULL
                      AND length(execution_configuration_sha256)=64));
        """,
    ),
    Migration(
        20,
        """
        CREATE TABLE ingestion_jobs_new (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            document_version_id TEXT NOT NULL REFERENCES document_versions(id),
            status TEXT NOT NULL CHECK(status IN
                ('queued','running','retry_wait','succeeded','failed','cancelled')),
            attempts INTEGER NOT NULL CHECK(attempts>=0), error TEXT,
            stage TEXT NOT NULL DEFAULT 'validate',
            next_attempt_at REAL NOT NULL DEFAULT 0,
            lease_owner TEXT, lease_token INTEGER NOT NULL DEFAULT 0,
            lease_expires_at REAL, run_id TEXT UNIQUE REFERENCES ingestion_runs(id)
        );
        INSERT INTO ingestion_jobs_new SELECT * FROM ingestion_jobs;
        DROP TABLE ingestion_jobs;
        ALTER TABLE ingestion_jobs_new RENAME TO ingestion_jobs;
        CREATE INDEX ix_ingestion_jobs_version ON ingestion_jobs(document_version_id);
        ALTER TABLE documents ADD COLUMN active_run_id TEXT REFERENCES ingestion_runs(id);
        UPDATE documents SET active_run_id=(
            SELECT r.id FROM ingestion_runs r JOIN text_artifacts a ON a.run_id=r.id
            WHERE r.document_version_id=documents.active_version_id
                AND r.tenant_id=documents.tenant_id
            GROUP BY r.document_version_id HAVING count(*)=1
        );
        ALTER TABLE ingestion_runs ADD COLUMN parent_run_id TEXT REFERENCES ingestion_runs(id);
        ALTER TABLE ingestion_runs ADD COLUMN reprocess_request_key TEXT;
        ALTER TABLE ingestion_runs ADD COLUMN requested_by TEXT REFERENCES users(id);
        CREATE UNIQUE INDEX ux_reprocess_request ON ingestion_runs
            (tenant_id,document_version_id,requested_by,reprocess_request_key)
            WHERE reprocess_request_key IS NOT NULL;
        """,
        rebuild_referenced_tables=True,
    ),
    Migration(
        21,
        """
        CREATE TABLE canonical_chunks(
            id TEXT PRIMARY KEY, document_version_id TEXT NOT NULL REFERENCES document_versions(id),
            strategy_revision TEXT NOT NULL, start_char INTEGER NOT NULL CHECK(start_char>=0),
            end_char INTEGER NOT NULL CHECK(end_char>start_char),
            content_sha256 TEXT NOT NULL CHECK(length(content_sha256)=64),
            UNIQUE(document_version_id,strategy_revision,start_char,end_char,content_sha256)
        );
        ALTER TABLE text_chunks ADD COLUMN logical_chunk_id TEXT REFERENCES canonical_chunks(id);
        CREATE UNIQUE INDEX ux_run_logical_chunk ON text_chunks(run_id,logical_chunk_id);
        CREATE UNIQUE INDEX ux_chunk_run ON text_chunks(id,run_id);
        CREATE UNIQUE INDEX ux_artifact_run ON text_artifacts(id,run_id);
        CREATE TABLE hierarchy_parents(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES ingestion_runs(id),
            artifact_id TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('document','section')),
            parent_id TEXT, start_ordinal INTEGER NOT NULL CHECK(start_ordinal>=0),
            end_ordinal INTEGER NOT NULL CHECK(end_ordinal>start_ordinal),
            content_sha256 TEXT NOT NULL CHECK(length(content_sha256)=64),
            page_numbers TEXT NOT NULL CHECK(json_valid(page_numbers)),
            UNIQUE(id,run_id), UNIQUE(run_id,kind,start_ordinal),
            FOREIGN KEY(artifact_id,run_id) REFERENCES text_artifacts(id,run_id) ON DELETE CASCADE,
            FOREIGN KEY(parent_id,run_id) REFERENCES hierarchy_parents(id,run_id) ON DELETE CASCADE
        );
        CREATE TABLE hierarchy_children(
            chunk_id TEXT PRIMARY KEY,run_id TEXT NOT NULL,parent_id TEXT NOT NULL,
            start_char INTEGER NOT NULL CHECK(start_char>=0),
            end_char INTEGER NOT NULL CHECK(end_char>start_char),
            FOREIGN KEY(chunk_id,run_id) REFERENCES text_chunks(id,run_id) ON DELETE CASCADE,
            FOREIGN KEY(parent_id,run_id) REFERENCES hierarchy_parents(id,run_id) ON DELETE CASCADE
        );
        ALTER TABLE query_runs ADD COLUMN parent_context_json TEXT
            CHECK(parent_context_json IS NULL OR json_valid(parent_context_json));
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        22,
        """
        CREATE TABLE dense_backend_binding (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            configuration TEXT NOT NULL CHECK(json_valid(configuration))
        );
        CREATE TABLE dense_publications (
            run_id TEXT PRIMARY KEY REFERENCES ingestion_runs(id) ON DELETE CASCADE,
            backend_key TEXT NOT NULL, content_digest TEXT NOT NULL,
            point_count INTEGER NOT NULL CHECK(point_count BETWEEN 1 AND 1024),
            configuration_sha256 TEXT NOT NULL,
            lease_owner TEXT NOT NULL, lease_token INTEGER NOT NULL,
            remote_digest TEXT
        );
        CREATE TABLE dense_write_epochs (
            run_id TEXT NOT NULL REFERENCES ingestion_runs(id) ON DELETE CASCADE,
            lease_owner TEXT NOT NULL, lease_token INTEGER NOT NULL,
            point_count INTEGER NOT NULL CHECK(point_count BETWEEN 1 AND 1024),
            PRIMARY KEY(run_id,lease_owner,lease_token)
        );
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        23,
        """
        CREATE TABLE citation_visual_regions (
            id TEXT PRIMARY KEY,
            answer_id TEXT NOT NULL,
            citation_index INTEGER NOT NULL,
            document_id TEXT NOT NULL REFERENCES documents(id),
            cache_key TEXT NOT NULL,
            object_key TEXT NOT NULL UNIQUE,
            manifest TEXT NOT NULL CHECK(json_valid(manifest)),
            manifest_sha256 TEXT NOT NULL,
            FOREIGN KEY(answer_id,citation_index)
                REFERENCES query_citations(answer_id,citation_index) ON DELETE CASCADE,
            UNIQUE(answer_id,citation_index,cache_key)
        );
        CREATE TABLE visual_region_reservations (
            id TEXT PRIMARY KEY,
            answer_id TEXT NOT NULL,
            citation_index INTEGER NOT NULL,
            cache_key TEXT NOT NULL,
            expires_at REAL NOT NULL,
            FOREIGN KEY(answer_id,citation_index)
                REFERENCES query_citations(answer_id,citation_index) ON DELETE CASCADE,
            UNIQUE(answer_id,citation_index,cache_key)
        );
        CREATE INDEX ix_visual_region_document ON citation_visual_regions(document_id);
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        24,
        """
        CREATE TABLE ocr_page_evidence (
            artifact_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            page_number INTEGER NOT NULL CHECK(page_number > 0),
            payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
            payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256)=64),
            PRIMARY KEY(run_id,page_number),
            FOREIGN KEY(artifact_id,run_id)
                REFERENCES text_artifacts(id,run_id) ON DELETE CASCADE
        );
        CREATE INDEX ix_ocr_page_evidence_artifact
            ON ocr_page_evidence(artifact_id,page_number);
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        25,
        """
        ALTER TABLE text_elements ADD COLUMN table_cells TEXT NOT NULL DEFAULT '[]'
            CHECK(json_valid(table_cells));
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        26,
        """
        ALTER TABLE query_runs ADD COLUMN stage_latencies_json TEXT DEFAULT NULL
            CHECK(stage_latencies_json IS NULL OR json_valid(stage_latencies_json));
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        27,
        """
        ALTER TABLE text_artifacts ADD COLUMN pdf_image_regions TEXT NOT NULL DEFAULT '[]'
            CHECK(json_valid(pdf_image_regions));
        ALTER TABLE text_artifacts ADD COLUMN pdf_image_provider TEXT DEFAULT NULL;
        """,
    ),
)


SQLITE_MIGRATIONS += (
    Migration(
        28,
        """
        ALTER TABLE text_artifacts ADD COLUMN pdf_chart_bars TEXT NOT NULL DEFAULT '[]'
            CHECK(json_valid(pdf_chart_bars));
        """,
    ),
)


def migration_checksum(migration: Migration) -> str:
    return migration.checksum
