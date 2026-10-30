"""Upload acceptance boundary: store bytes and queue ingestion, but do not parse."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.domain.documents.models import (
    Document,
    DocumentStatus,
    DocumentVersion,
    Source,
    Tenant,
)
from enterprise_rag.domain.ingestion.models import IngestionJob
from enterprise_rag.persistence.sqlite_metadata import SQLiteMetadataStore
from enterprise_rag.security import ActorContext
from enterprise_rag.storage.local_object_store import LocalObjectStore
from enterprise_rag.streams import BinaryReader, BinaryWriter

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_UPLOAD_ATTEMPT_TTL_SECONDS = 60 * 60


class UploadTooLarge(ValueError):
    """Raised when an upload exceeds the configured request limit."""


class UploadForbidden(PermissionError):
    """The actor may not accept this upload or replace an existing document."""


@dataclass(frozen=True, slots=True)
class UploadAccepted:
    document: Document
    version: DocumentVersion
    job: IngestionJob | None
    duplicate: bool


class DocumentUploadService:
    """Accept a stream without running parsing or model work in the request path."""

    def __init__(
        self,
        metadata: SQLiteMetadataStore,
        object_store: LocalObjectStore,
        spool_dir: Path | None = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        upload_attempt_ttl_seconds: float = DEFAULT_UPLOAD_ATTEMPT_TTL_SECONDS,
    ) -> None:
        if max_upload_bytes < 0:
            raise ValueError("max_upload_bytes must be non-negative")
        if upload_attempt_ttl_seconds < 0:
            raise ValueError("upload_attempt_ttl_seconds must be non-negative")
        self.metadata = metadata
        self.object_store = object_store
        self.max_upload_bytes = max_upload_bytes
        self.upload_attempt_ttl_seconds = upload_attempt_ttl_seconds
        self.spool_dir = spool_dir.resolve() if spool_dir else None
        if self.spool_dir:
            self.spool_dir.mkdir(parents=True, exist_ok=True)

    async def accept_upload(
        self,
        *,
        actor: ActorContext,
        tenant: Tenant,
        source: Source,
        canonical_key: str,
        title: str,
        filename: str,
        stream: BinaryReader,
    ) -> UploadAccepted:
        """Run blocking local I/O off the event loop and return a queued job."""

        return await asyncio.to_thread(
            self._accept_upload,
            actor=actor,
            tenant=tenant,
            source=source,
            canonical_key=canonical_key,
            title=title,
            filename=filename,
            stream=stream,
        )

    def _accept_upload(
        self,
        *,
        actor: ActorContext,
        tenant: Tenant,
        source: Source,
        canonical_key: str,
        title: str,
        filename: str,
        stream: BinaryReader,
    ) -> UploadAccepted:
        if source.tenant_id != tenant.id:
            raise ValueError("Source and tenant must belong to the same tenant")

        document = Document.create(tenant.id, source.id, canonical_key, title)
        existing_document = self.metadata.get_document_for_tenant(
            tenant.id, document.id
        )
        if existing_document is not None:
            document = existing_document
        if (
            not isinstance(actor, ActorContext)
            or actor.tenant_id != tenant.id
            or actor.user_id is None
        ):
            raise UploadForbidden("Upload actor does not belong to tenant")
        if existing_document is not None and not self.metadata.is_tenant_admin(
            tenant.id, actor.user_id
        ):
            raise UploadForbidden("Existing document requires tenant admin")
        if document.status in {DocumentStatus.DELETING, DocumentStatus.DELETED}:
            raise ValueError(
                f"Document {document.id} is not accepting uploads from {document.status}"
            )

        with tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024,
            mode="w+b",
            dir=str(self.spool_dir) if self.spool_dir else None,
        ) as buffered:
            checksum, byte_size = self._copy_and_hash(
                stream, buffered, max_bytes=self.max_upload_bytes
            )
            buffered.seek(0)
            object_key = f"documents/{document.id}/attempts/{uuid.uuid4().hex}.bin"
            created_at = time.time()
            self.metadata.ensure_tenant(tenant)
            self.metadata.register_upload_attempt(
                tenant.id,
                document.id,
                object_key,
                created_at=created_at,
                expires_at=created_at + self.upload_attempt_ttl_seconds,
            )
            stored = self.object_store.put_stream(
                object_key, buffered, max_bytes=self.max_upload_bytes
            )

        if stored.sha256 != checksum or stored.size != byte_size:
            raise RuntimeError(
                "Object-store checksum or size did not match the upload stream"
            )

        try:
            decision = self.metadata.accept_upload(
                tenant=tenant,
                source=source,
                canonical_key=canonical_key,
                title=title,
                filename=filename,
                content_sha256=checksum,
                byte_size=byte_size,
                object_key=object_key,
                actor=actor,
            )
        except PermissionError as exc:
            raise UploadForbidden("Upload authorization changed before commit") from exc
        self.metadata.finish_upload_attempt(
            tenant.id,
            object_key,
            status="abandoned" if decision.duplicate else "committed",
            now=time.time(),
        )
        if decision.duplicate:
            self.object_store.delete(object_key)
            return UploadAccepted(
                document=decision.document,
                version=decision.version,
                job=decision.job,
                duplicate=True,
            )

        return UploadAccepted(
            document=decision.document,
            version=decision.version,
            job=decision.job,
            duplicate=False,
        )

    @staticmethod
    def _copy_and_hash(
        stream: BinaryReader, buffered: BinaryWriter, *, max_bytes: int | None = None
    ) -> tuple[str, int]:
        digest = hashlib.sha256()
        byte_size = 0
        while chunk := stream.read(1024 * 1024):
            next_size = byte_size + len(chunk)
            if max_bytes is not None and next_size > max_bytes:
                raise UploadTooLarge(f"Upload exceeds the {max_bytes}-byte limit")
            buffered.write(chunk)
            digest.update(chunk)
            byte_size = next_size
        return digest.hexdigest(), byte_size
