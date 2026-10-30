"""Real lite TXT/MD stage executor wired to raw objects and SQLite artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from enterprise_rag.domain.documents import VersionStatus
from enterprise_rag.domain.documents.ids import sha256_bytes
from enterprise_rag.domain.documents.models import DocumentVersion
from enterprise_rag.domain.ingestion import (
    IngestionJob,
    JobLease,
    StageAttempt,
    StageStatus,
)
from enterprise_rag.ingestion import (
    DEFAULT_PDF_PARSER_TIMEOUT_SECONDS,
    ParsedText,
    PDFParseError,
    TextParseError,
    chunk_text,
    parse_document,
)
from enterprise_rag.ingestion.configuration import execution_configuration
from enterprise_rag.ingestion.errors import ParserContainmentError
from enterprise_rag.ingestion.ocr_pipeline import OCRPipelineError, PDFOCRPipeline
from enterprise_rag.ingestion.parsers.pdf import parser_revision_for
from enterprise_rag.ingestion.pdf_render import (
    PDFRenderContainmentError,
    PDFRenderError,
)
from enterprise_rag.ingestion.tesseract import OCRProviderFailed, OCRProviderUnavailable
from enterprise_rag.persistence import SQLiteMetadataStore
from enterprise_rag.storage import LocalObjectStore

from .lite import StageExecutionError


@dataclass(frozen=True, slots=True)
class LiteTextStageExecutor:
    """Execute the bounded native TXT/MD/PDF stages in the lite profile."""

    metadata: SQLiteMetadataStore
    object_store: LocalObjectStore
    max_text_bytes: int = 25 * 1024 * 1024
    max_pdf_pages: int = 100
    pdf_parser_timeout_seconds: float = DEFAULT_PDF_PARSER_TIMEOUT_SECONDS
    max_chunk_chars: int = 2000
    clock: Callable[[], float] = time.time
    _parsed_by_run: dict[str, ParsedText] = field(default_factory=dict)
    dense_publisher: object | None = None
    ocr_pipeline: PDFOCRPipeline | None = None
    pdf_table_engine: str = "pypdf"

    def __post_init__(self) -> None:
        parser_revision_for(self.pdf_table_engine)
        if type(self.max_text_bytes) is not int or self.max_text_bytes <= 0:
            raise ValueError("Maximum text size must be positive")
        if type(self.max_pdf_pages) is not int or self.max_pdf_pages <= 0:
            raise ValueError("Maximum PDF pages must be positive")
        if (
            isinstance(self.pdf_parser_timeout_seconds, bool)
            or not math.isfinite(self.pdf_parser_timeout_seconds)
            or self.pdf_parser_timeout_seconds <= 0
        ):
            raise ValueError("PDF parser timeout must be positive")
        if type(self.max_chunk_chars) is not int or self.max_chunk_chars <= 0:
            raise ValueError("Maximum chunk size must be positive")

    def execute(self, lease: JobLease, stage: str) -> str:
        return self._executor_for_job(lease.job)._execute(lease, stage)

    def _execute(self, lease: JobLease, stage: str) -> str:
        """Run a supported stage or fail explicitly for an absent backend."""

        job = lease.job
        run_id = self._run_id_for_job(job)
        try:
            matches = self.metadata.match_run_execution_configuration(
                lease, self._execution_configuration(job), now=self.clock(), bind=True
            )
        except ValueError:
            raise StageExecutionError(
                "execution_configuration_unbound", retryable=False
            ) from None
        if not matches:
            raise StageExecutionError(
                "execution_configuration_changed", retryable=False
            )
        with self.metadata._connection() as db:
            bound = (
                db.execute("SELECT 1 FROM dense_backend_binding").fetchone() is not None
            )
        if bound != (self.dense_publisher is not None):
            raise StageExecutionError(
                "dense_backend_configuration_missing", retryable=False
            )
        if stage == "validate":
            version = self._version_for_job(job)
            digest, size = self._raw_digest(version)
            if digest != version.content_sha256 or size != version.byte_size:
                raise StageExecutionError(
                    "raw_object_checksum_mismatch", retryable=False
                )
            return digest
        if stage == "parse":
            parsed = self._parse_job(job)
            self._parsed_by_run[run_id] = parsed
            return parsed.content_sha256
        if stage == "chunk":
            parsed = self._parsed_for_job(job)
            chunks = chunk_text(parsed, max_chars=self.max_chunk_chars)
            self.metadata.persist_text_artifacts(
                run_id, parsed, chunks, lease=lease, now=self.clock()
            )
            return self._chunk_output_checksum(chunks)
        if stage == "index":
            count = self.metadata.index_run(lease, now=self.clock())
            if count == 0:
                raise StageExecutionError("no_text_chunks", retryable=False)
            self._dense_action(lease, "publish")
            return sha256_bytes(f"{job.run_id}:{count}".encode())
        if stage == "verify":
            if not self.metadata.verify_run_index(job.tenant_id, run_id):
                raise StageExecutionError("index_verification_failed", retryable=False)
            self._dense_action(lease, "verify")
            return sha256_bytes(f"{run_id}:verified".encode())
        if stage == "activate":
            self._dense_action(lease, "verify")
            if not self.metadata.activate_run(lease, now=self.clock()):
                raise StageExecutionError(
                    "activation_precondition_failed", retryable=False
                )
            return sha256_bytes(f"{run_id}:active".encode())
        raise StageExecutionError("stage_backend_not_implemented", retryable=False)

    def verify_checkpoint(self, lease: JobLease, attempt: StageAttempt) -> bool:
        try:
            if attempt.stage in {"index", "verify", "activate"}:
                self._dense_action(lease, "verify")
            return self._executor_for_job(lease.job)._verify_checkpoint(lease, attempt)
        except StageExecutionError:
            # Reopen validation so the normal fenced failure path records the
            # incompatible runtime, rather than escaping the worker claim loop.
            return False

    def _dense_action(self, lease, action):
        if self.dense_publisher is not None:
            try:
                getattr(self.dense_publisher, action)(lease, self.clock)
            except Exception as error:
                raise StageExecutionError(
                    "external_dense_verification_failed", retryable=True
                ) from error

    def _verify_checkpoint(self, lease: JobLease, attempt: StageAttempt) -> bool:
        """Validate durable inputs and outputs before skipping a successful stage."""

        if attempt.status is not StageStatus.SUCCEEDED:
            return False
        if attempt.output_checksum is None:
            return False

        job = lease.job
        run_id = self._run_id_for_job(job)
        if not self.metadata.match_run_execution_configuration(
            lease, self._execution_configuration(job), now=self.clock()
        ):
            return False
        if attempt.run_id != run_id or attempt.job_id != job.id:
            return False

        try:
            version = self._version_for_job(job)
        except StageExecutionError:
            return False
        if attempt.stage == "validate":
            try:
                digest, size = self._raw_digest(version)
            except StageExecutionError:
                return False
            return (
                digest == version.content_sha256
                and size == version.byte_size
                and attempt.output_checksum == version.content_sha256
            )

        stored = self.metadata.get_text_artifacts(run_id)
        if stored is None or stored[0].content_sha256 != version.content_sha256:
            return False
        parsed, chunks = stored
        if not self.metadata.verify_run_hierarchy(run_id):
            return False
        element_ids = {element.id for element in parsed.elements}
        if (
            parsed.run_id != run_id
            or any(element.run_id != run_id for element in parsed.elements)
            or any(
                chunk.run_id != run_id
                or chunk.content_sha256 != sha256_bytes(chunk.text.encode("utf-8"))
                or not set(chunk.element_ids).issubset(element_ids)
                for chunk in chunks
            )
        ):
            return False

        if attempt.stage == "parse":
            return attempt.output_checksum == version.content_sha256
        if attempt.stage == "chunk":
            return attempt.output_checksum == self._chunk_output_checksum(chunks)
        if attempt.stage == "index":
            return self.metadata.verify_run_index(
                job.tenant_id, run_id
            ) and attempt.output_checksum == sha256_bytes(
                f"{run_id}:{len(chunks)}".encode()
            )
        if attempt.stage == "verify":
            return self.metadata.verify_run_index(
                job.tenant_id, run_id
            ) and attempt.output_checksum == sha256_bytes(f"{run_id}:verified".encode())
        if attempt.stage == "activate":
            document = self.metadata.get_document(version.document_id)
            return (
                version.status is VersionStatus.ACTIVE
                and document is not None
                and document.active_version_id == version.id
                and self.metadata.active_run_id(version.document_id) == run_id
                and attempt.output_checksum == sha256_bytes(f"{run_id}:active".encode())
            )
        return False

    @staticmethod
    def _chunk_output_checksum(chunks) -> str:
        return sha256_bytes(
            "".join(chunk.content_sha256 for chunk in chunks).encode("utf-8")
        )

    def _executor_for_job(self, job: IngestionJob) -> LiteTextStageExecutor:
        run = self.metadata.get_run(self._run_id_for_job(job))
        if run is None or run.reprocess_request_key is None:
            return self
        if run.execution_configuration_json is None:
            raise StageExecutionError(
                "execution_configuration_unbound", retryable=False
            )
        config = json.loads(run.execution_configuration_json)
        limits = (
            "max_text_bytes",
            "max_pdf_pages",
            "pdf_parser_timeout_seconds",
            "max_chunk_chars",
        )
        if any(config[key] > getattr(self, key) for key in limits):
            raise StageExecutionError(
                "execution_configuration_exceeds_worker_limits", retryable=False
            )
        # A reprocess request owns its immutable limits. Server revisions are
        # recomputed below, so a provider/parser upgrade refuses an old checkpoint.
        return replace(
            self,
            **{k: config[k] for k in limits},
        )

    def _execution_configuration(self, job: IngestionJob) -> dict:
        try:
            ocr_configuration = (
                None
                if self.ocr_pipeline is None
                or not self._version_for_job(job).filename.lower().endswith(".pdf")
                else self.ocr_pipeline.execution_configuration
            )
        except (OCRPipelineError, PDFRenderError, OCRProviderUnavailable) as exc:
            raise StageExecutionError(
                "ocr_execution_configuration_unavailable", retryable=False
            ) from exc
        return execution_configuration(
            self._version_for_job(job).filename,
            max_text_bytes=self.max_text_bytes,
            max_pdf_pages=self.max_pdf_pages,
            pdf_parser_timeout_seconds=self.pdf_parser_timeout_seconds,
            max_chunk_chars=self.max_chunk_chars,
            ocr_pipeline=ocr_configuration,
            pdf_table_engine=self.pdf_table_engine,
        )

    @staticmethod
    def _run_id_for_job(job: IngestionJob) -> str:
        if job.run_id is None:
            raise StageExecutionError("run_not_found", retryable=False)
        return job.run_id

    def _version_for_job(self, job: IngestionJob) -> DocumentVersion:
        version = self.metadata.get_version_for_tenant(
            job.tenant_id, job.document_version_id
        )
        if version is None:
            raise StageExecutionError("version_not_found", retryable=False)
        return version

    def _parse_job(self, job: IngestionJob) -> ParsedText:
        version = self._version_for_job(job)
        run_id = self._run_id_for_job(job)
        try:
            content = self._read_bounded(version.object_key)
            parsed = parse_document(
                run_id,
                version.filename,
                content,
                max_bytes=self.max_text_bytes,
                max_pdf_pages=self.max_pdf_pages,
                pdf_timeout_seconds=self.pdf_parser_timeout_seconds,
                allow_ocr_pages=(
                    self.ocr_pipeline is not None
                    and version.filename.lower().endswith(".pdf")
                ),
                pdf_table_engine=self.pdf_table_engine,
            )
            if self.ocr_pipeline is not None and version.filename.lower().endswith(
                ".pdf"
            ):
                expected_configuration = self._bound_ocr_configuration(run_id)
                assert expected_configuration is not None
                try:
                    parsed = self.ocr_pipeline.enrich(
                        parsed,
                        version.id,
                        content,
                        expected_configuration=expected_configuration,
                    )
                except PDFRenderContainmentError as exc:
                    raise ParserContainmentError(
                        "pdf_ocr_renderer_cleanup_unconfirmed"
                    ) from exc
                except OCRProviderUnavailable as exc:
                    raise StageExecutionError(
                        "ocr_provider_unavailable", retryable=False
                    ) from exc
                except OCRProviderFailed as exc:
                    raise StageExecutionError(
                        exc.diagnostic_code, retryable=True
                    ) from exc
                except PDFRenderError as exc:
                    raise StageExecutionError(
                        "ocr_render_failed", retryable=True
                    ) from exc
                except OCRPipelineError as exc:
                    raise StageExecutionError(str(exc), retryable=False) from exc
            return parsed
        except (PDFParseError, TextParseError) as exc:
            raise StageExecutionError(str(exc), retryable=False) from exc

    def _bound_ocr_configuration(self, run_id: str) -> dict[str, object] | None:
        if self.ocr_pipeline is None:
            return None
        run = self.metadata.get_run(run_id)
        if run is None or run.execution_configuration_json is None:
            raise StageExecutionError(
                "execution_configuration_unbound", retryable=False
            )
        try:
            configuration = json.loads(run.execution_configuration_json)
            expected = configuration["ocr_pipeline"]
            current = self.ocr_pipeline.execution_configuration
        except (
            KeyError,
            TypeError,
            ValueError,
            OCRPipelineError,
            PDFRenderError,
            OCRProviderUnavailable,
        ) as exc:
            raise StageExecutionError(
                "ocr_execution_configuration_unavailable", retryable=False
            ) from exc
        if not isinstance(expected, dict) or expected != current:
            raise StageExecutionError(
                "ocr_execution_configuration_changed", retryable=False
            )
        return expected

    def _parsed_for_job(self, job: IngestionJob) -> ParsedText:
        run_id = self._run_id_for_job(job)
        parsed = self._parsed_by_run.get(run_id)
        if parsed is not None:
            return parsed
        stored = self.metadata.get_text_artifacts(run_id)
        if stored is not None:
            parsed, _chunks = stored
            self._parsed_by_run[run_id] = parsed
            return parsed
        parsed = self._parse_job(job)
        self._parsed_by_run[run_id] = parsed
        return parsed

    def _read_bounded(self, object_key: str) -> bytes:
        pieces: list[bytes] = []
        size = 0
        try:
            with self.object_store.open(object_key) as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_text_bytes:
                        raise TextParseError("text_input_too_large")
                    pieces.append(chunk)
        except FileNotFoundError as exc:
            raise StageExecutionError("raw_object_missing", retryable=False) from exc
        return b"".join(pieces)

    def _raw_digest(self, version: DocumentVersion) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        try:
            with self.object_store.open(version.object_key) as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_text_bytes:
                        raise StageExecutionError(
                            "text_input_too_large", retryable=False
                        )
                    digest.update(chunk)
        except FileNotFoundError as exc:
            raise StageExecutionError("raw_object_missing", retryable=False) from exc
        except OSError as exc:
            raise StageExecutionError("raw_object_unavailable", retryable=True) from exc
        return digest.hexdigest(), size
