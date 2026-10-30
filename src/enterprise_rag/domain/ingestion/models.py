"""Durable ingestion job state used by the local upload boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from enterprise_rag.domain.documents.ids import deterministic_id


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestionStage(StrEnum):
    VALIDATE = "validate"
    PARSE = "parse"
    CHUNK = "chunk"
    INDEX = "index"
    VERIFY = "verify"
    ACTIVATE = "activate"


LITE_REQUIRED_STAGES = (
    IngestionStage.VALIDATE.value,
    IngestionStage.PARSE.value,
    IngestionStage.CHUNK.value,
    IngestionStage.INDEX.value,
    IngestionStage.VERIFY.value,
    IngestionStage.ACTIVATE.value,
)


@dataclass(frozen=True, slots=True)
class IngestionRun:
    id: str
    tenant_id: str
    document_version_id: str
    pipeline_fingerprint: str
    required_stages: tuple[str, ...] = LITE_REQUIRED_STAGES
    status: RunStatus = RunStatus.QUEUED
    request_generation: int = 1
    created_at: float = 0.0
    parent_run_id: str | None = None
    reprocess_request_key: str | None = None
    execution_configuration_json: str | None = None

    @classmethod
    def create(
        cls,
        tenant_id: str,
        document_version_id: str,
        pipeline_fingerprint: str,
        required_stages: tuple[str, ...] = LITE_REQUIRED_STAGES,
    ) -> IngestionRun:
        if not pipeline_fingerprint.strip():
            raise ValueError("Pipeline fingerprint must be non-empty")
        if not required_stages:
            raise ValueError("At least one ingestion stage is required")
        return cls(
            id=deterministic_id(
                "ingestion-run",
                tenant_id,
                document_version_id,
                pipeline_fingerprint,
            ),
            tenant_id=tenant_id,
            document_version_id=document_version_id,
            pipeline_fingerprint=pipeline_fingerprint,
            required_stages=required_stages,
        )


@dataclass(slots=True)
class IngestionJob:
    id: str
    tenant_id: str
    document_version_id: str
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    error: str | None = None
    stage: str = "validate"
    next_attempt_at: float = 0.0
    lease_owner: str | None = None
    lease_token: int = 0
    lease_expires_at: float | None = None
    run_id: str | None = None

    @classmethod
    def create(
        cls, tenant_id: str, document_version_id: str, run_id: str | None = None
    ) -> IngestionJob:
        return cls(
            id=deterministic_id("ingestion-job", tenant_id, document_version_id),
            tenant_id=tenant_id,
            document_version_id=document_version_id,
            run_id=run_id,
        )


@dataclass(frozen=True, slots=True)
class JobLease:
    """A worker's fenced, time-bounded ownership of one ingestion job."""

    job: IngestionJob
    worker_id: str
    lease_token: int
    lease_expires_at: float
    reclaimed: bool = False


@dataclass(frozen=True, slots=True)
class StageAttempt:
    id: str
    run_id: str
    job_id: str
    stage: str
    attempt_number: int
    status: StageStatus = StageStatus.RUNNING
    error: str | None = None
    output_checksum: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None
