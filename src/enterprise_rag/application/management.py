"""ACL-aware document and ingestion management use cases."""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.application.document_catalog import DocumentCatalogService
from enterprise_rag.domain.documents.models import (
    Document,
    DocumentVersion,
    Group,
    User,
)
from enterprise_rag.domain.ingestion.models import IngestionJob, JobStatus
from enterprise_rag.persistence import SQLiteMetadataStore
from enterprise_rag.security import (
    ActorContext,
    DocumentPermission,
    Permission,
    PrincipalType,
)


class ManagementForbidden(PermissionError):
    """The authenticated actor is not allowed to mutate tenant state."""


class ManagementNotFound(LookupError):
    """A tenant-scoped management target does not exist."""


class ManagementConflict(RuntimeError):
    """A management mutation conflicts with the current durable state."""


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Result of an idempotent administrative job retry request."""

    job: IngestionJob
    requeued: bool


@dataclass(frozen=True, slots=True)
class ReprocessDecision:
    """A new processing run, or an idempotent reference to the original request."""

    job: IngestionJob
    created: bool


@dataclass(frozen=True, slots=True)
class DeletionDecision:
    """Result of an idempotent logical document deletion request."""

    document: Document
    tombstoned: bool


class DocumentManagementService:
    """Expose metadata and admin mutations after resolving actor state in SQLite."""

    def __init__(self, metadata: SQLiteMetadataStore) -> None:
        self.metadata = metadata
        self.catalog = DocumentCatalogService(metadata)

    def list_documents(
        self,
        actor: ActorContext | None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Document, ...]:
        return self.catalog.list_documents(actor, limit=limit, offset=offset)

    def get_document(
        self, actor: ActorContext | None, document_id: str
    ) -> Document | None:
        return self.catalog.get_document(actor, document_id)

    def list_versions(
        self, actor: ActorContext | None, document_id: str
    ) -> tuple[DocumentVersion, ...] | None:
        return self.catalog.list_versions(actor, document_id)

    def get_job(self, actor: ActorContext | None, job_id: str) -> IngestionJob | None:
        return self.catalog.get_job(actor, job_id)

    def delete_document(
        self, actor: ActorContext, document_id: str
    ) -> DeletionDecision:
        self._require_tenant_admin(actor)
        try:
            document, tombstoned = self.metadata.tombstone_document(
                actor.tenant_id, document_id, actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        if document is None:
            raise ManagementNotFound("Document not found")
        return DeletionDecision(document=document, tombstoned=tombstoned)

    def reprocess_version(
        self,
        actor: ActorContext,
        version_id: str,
        request_key: str,
        *,
        max_chunk_chars: int,
        now: float,
    ) -> ReprocessDecision:
        self._require_tenant_admin(actor)
        try:
            job, created = self.metadata.request_reprocess(
                actor,
                version_id,
                request_key,
                max_chunk_chars=max_chunk_chars,
                now=now,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Current tenant admin required") from exc
        except LookupError as exc:
            raise ManagementNotFound("Accessible version not found") from exc
        except RuntimeError as exc:
            raise ManagementConflict(
                "Reprocess request conflicts with current state"
            ) from exc
        return ReprocessDecision(job, created)

    def retry_job(
        self, actor: ActorContext, job_id: str, *, now: float
    ) -> RetryDecision:
        self._require_tenant_admin(actor)
        job = self.metadata.get_job_for_tenant(actor.tenant_id, job_id)
        if job is None:
            raise ManagementNotFound("Job not found")
        if job.status in {JobStatus.QUEUED, JobStatus.RETRY_WAIT}:
            return RetryDecision(job=job, requeued=False)
        if job.status is not JobStatus.FAILED:
            raise ManagementConflict(
                f"Job in status {job.status.value!r} cannot be retried"
            )
        try:
            retried = self.metadata.retry_failed_job(
                actor.tenant_id, job_id, now, actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        if retried is not None:
            return RetryDecision(job=retried, requeued=True)
        current = self.metadata.get_job_for_tenant(actor.tenant_id, job_id)
        if current is not None and current.status in {
            JobStatus.QUEUED,
            JobStatus.RETRY_WAIT,
        }:
            return RetryDecision(job=current, requeued=False)
        raise ManagementConflict("Job changed before retry could be committed")

    def provision_user(self, actor: ActorContext, email: str) -> User:
        self._require_tenant_admin(actor)
        if not email.strip():
            raise ValueError("User email must be non-empty")
        try:
            return self.metadata.upsert_user(
                User.create(actor.tenant_id, email.strip()), actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc

    def provision_group(self, actor: ActorContext, name: str) -> Group:
        self._require_tenant_admin(actor)
        if not name.strip():
            raise ValueError("Group name must be non-empty")
        try:
            return self.metadata.upsert_group(
                Group.create(actor.tenant_id, name.strip()), actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc

    def add_group_member(
        self, actor: ActorContext, group_id: str, user_id: str
    ) -> bool:
        self._require_tenant_admin(actor)
        group = self.metadata.get_group_for_tenant(actor.tenant_id, group_id)
        user = self.metadata.get_user_for_tenant(actor.tenant_id, user_id)
        if group is None or user is None:
            raise ManagementNotFound("Group or user not found")
        try:
            return self.metadata.add_group_member(user, group, actor=actor)
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc

    def remove_group_member(
        self, actor: ActorContext, group_id: str, user_id: str
    ) -> bool:
        self._require_tenant_admin(actor)
        group = self.metadata.get_group_for_tenant(actor.tenant_id, group_id)
        user = self.metadata.get_user_for_tenant(actor.tenant_id, user_id)
        if group is None or user is None:
            raise ManagementNotFound("Group or user not found")
        try:
            return self.metadata.remove_group_member(user, group, actor=actor)
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc

    def grant_permission(
        self,
        actor: ActorContext,
        document_id: str,
        principal_type: PrincipalType | str,
        principal_id: str,
        permission: Permission | str,
    ) -> DocumentPermission:
        self._require_tenant_admin(actor)
        self._require_document(actor, document_id)
        if not principal_id.strip():
            raise ValueError("Permission principal id must be non-empty")
        try:
            return self.metadata.grant_document_permission(
                actor.tenant_id,
                document_id,
                PrincipalType(principal_type),
                principal_id,
                Permission(permission),
                actor=actor,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc

    def revoke_permission(
        self,
        actor: ActorContext,
        document_id: str,
        principal_type: PrincipalType | str,
        principal_id: str,
        permission: Permission | str,
    ) -> bool:
        self._require_tenant_admin(actor)
        self._require_document(actor, document_id)
        if not principal_id.strip():
            raise ValueError("Permission principal id must be non-empty")
        try:
            return self.metadata.revoke_document_permission(
                actor.tenant_id,
                document_id,
                PrincipalType(principal_type),
                principal_id,
                Permission(permission),
                actor=actor,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc

    def _require_tenant_admin(self, actor: ActorContext | None) -> None:
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None or not self.metadata.is_tenant_admin(
            scope.tenant_id, scope.user_id
        ):
            raise ManagementForbidden("Tenant admin permission required")

    def _require_document(self, actor: ActorContext, document_id: str) -> Document:
        document = self.metadata.get_document_for_tenant(actor.tenant_id, document_id)
        if document is None:
            raise ManagementNotFound("Document not found")
        return document
