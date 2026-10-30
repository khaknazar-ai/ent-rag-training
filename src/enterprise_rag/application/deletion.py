"""Logical-deletion cleanup and reconciliation for the lite profile."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from enterprise_rag.application.management import (
    ManagementConflict,
    ManagementForbidden,
    ManagementNotFound,
)
from enterprise_rag.domain.documents import DocumentStatus
from enterprise_rag.persistence import (
    DeletionManifest,
    SharedDeletionObjectKeyError,
    SQLiteMetadataStore,
)
from enterprise_rag.security import ActorContext
from enterprise_rag.storage import LocalObjectStore

DEFAULT_ORPHAN_GRACE_SECONDS = 5 * 60
DEFAULT_CLEANUP_CLAIM_TTL_SECONDS = 5 * 60
DEFAULT_ORPHAN_GC_BATCH_SIZE = 100
MAX_ORPHAN_GC_BATCH_SIZE = 1000
DEFAULT_OBJECT_DELETE_DEADLINE_SECONDS = 30.0

UnknownObjectPolicy = Literal["preserve"]


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Explicit, conservative retention rules for tracked upload-attempt cleanup."""

    orphan_grace_seconds: float = DEFAULT_ORPHAN_GRACE_SECONDS
    cleanup_claim_ttl_seconds: float = DEFAULT_CLEANUP_CLAIM_TTL_SECONDS
    orphan_gc_batch_size: int = DEFAULT_ORPHAN_GC_BATCH_SIZE
    object_delete_deadline_seconds: float = DEFAULT_OBJECT_DELETE_DEADLINE_SECONDS
    unknown_object_policy: UnknownObjectPolicy = "preserve"

    def __post_init__(self) -> None:
        if self.orphan_grace_seconds < 0:
            raise ValueError("Retention grace period must be non-negative")
        if self.cleanup_claim_ttl_seconds <= 0:
            raise ValueError("Retention cleanup claim TTL must be positive")
        if (
            self.orphan_gc_batch_size <= 0
            or self.orphan_gc_batch_size > MAX_ORPHAN_GC_BATCH_SIZE
        ):
            raise ValueError("Retention GC batch size must be between 1 and 1000")
        if (
            not isfinite(self.object_delete_deadline_seconds)
            or self.object_delete_deadline_seconds <= 0
        ):
            raise ValueError(
                "Retention object delete deadline must be finite and positive"
            )
        if self.unknown_object_policy != "preserve":
            raise ValueError("Unknown objects must be preserved")


DEFAULT_RETENTION_POLICY = RetentionPolicy()


class DeletionStorageError(OSError):
    """Raised when an owned raw object cannot be removed safely."""

    def __init__(
        self, key: str, message: str = "Owned object-store deletion failed"
    ) -> None:
        self.key = key
        super().__init__(message)


class AmbiguousDeletionError(DeletionStorageError):
    """Raised when deletion may have happened but cannot be confirmed.

    The cleanup claim remains active until it expires. A later reconciliation
    pass may finalize a missing object or retry an object that is still there;
    this class deliberately does not attempt byte-level compensation.
    """

    def __init__(self, key: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            key,
            f"Owned object-store deletion outcome is ambiguous: {reason}",
        )


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Observable result of a dry-run or idempotent deletion reconciliation."""

    document_id: str
    dry_run: bool
    object_keys: tuple[str, ...]
    deleted_object_count: int
    finalized: bool


@dataclass(frozen=True, slots=True)
class OrphanReconciliationResult:
    """Observable result of a tracked upload-attempt garbage-collection pass.

    ``deleted_object_count`` counts objects present before the idempotent delete;
    finalizing a claim whose object is already absent is not counted as a
    physical deletion.
    """

    dry_run: bool
    object_keys: tuple[str, ...]
    claimed_object_count: int
    deleted_object_count: int
    truncated: bool = False
    limit: int = DEFAULT_ORPHAN_GC_BATCH_SIZE
    reconciled_missing_count: int = 0
    eligible_missing_count: int = 0


class DocumentDeletionService:
    """Reconcile one tenant-owned tombstone without deleting unknown objects."""

    def __init__(
        self,
        metadata: SQLiteMetadataStore,
        object_store: LocalObjectStore,
        retention_policy: RetentionPolicy = DEFAULT_RETENTION_POLICY,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.metadata = metadata
        self.object_store = object_store
        self.retention_policy = retention_policy
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock

    def reconcile(
        self,
        actor: ActorContext,
        document_id: str,
        *,
        dry_run: bool = False,
    ) -> ReconciliationResult:
        self._require_tenant_admin(actor)
        try:
            document = self.metadata.get_document_for_tenant(
                actor.tenant_id, document_id, actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        if document is None:
            raise ManagementNotFound("Document not found")
        if document.status is DocumentStatus.DELETED:
            return ReconciliationResult(
                document_id=document_id,
                dry_run=dry_run,
                object_keys=(),
                deleted_object_count=0,
                finalized=True,
            )
        if document.status is not DocumentStatus.DELETING:
            raise ManagementConflict(
                "Document must be tombstoned before reconciliation"
            )

        try:
            manifest = self.metadata.get_deletion_manifest_for_tenant(
                actor.tenant_id, document_id, actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        if manifest is None:
            raise ManagementConflict("Document deletion manifest is unavailable")
        if manifest.active_job_ids:
            raise ManagementConflict("Document still has active ingestion jobs")
        if dry_run:
            return self._dry_run_result(manifest)

        try:
            claim = self.metadata.claim_document_deletion(
                actor.tenant_id,
                document_id,
                actor=actor,
                now=self._wall_clock(),
                ttl_seconds=self.retention_policy.cleanup_claim_ttl_seconds,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        except SharedDeletionObjectKeyError as exc:
            raise ManagementConflict(str(exc)) from exc
        if claim is None:
            raise ManagementConflict("Document deletion is already claimed or changed")

        deleted_count = 0
        for key in claim.manifest.object_keys:
            if not self.metadata.is_document_deletion_claim_current(
                actor.tenant_id,
                document_id,
                claim_token=claim.claim_token,
                now=self._wall_clock(),
            ):
                raise ManagementConflict("Document deletion claim was lost")
            try:
                object_was_present = self.object_store.exists(key)
            except (OSError, ValueError) as probe_error:
                inspection_error = DeletionStorageError(
                    key, "Owned object-store pre-delete inspection failed"
                )
                try:
                    self.metadata.release_document_deletion_claim(
                        actor.tenant_id, document_id, claim_token=claim.claim_token
                    )
                except Exception as release_error:
                    inspection_error.add_note(
                        "Claim release also failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )
                raise inspection_error from probe_error
            started_at = self._monotonic_clock()
            try:
                self.object_store.delete(key)
            except (OSError, ValueError) as exc:
                try:
                    object_present = self.object_store.exists(key)
                except (OSError, ValueError) as probe_error:
                    ambiguous = AmbiguousDeletionError(
                        key, "delete failed and object inspection also failed"
                    )
                    ambiguous.add_note(
                        "Object presence probe failed: "
                        f"{type(probe_error).__name__}: {probe_error}"
                    )
                    raise ambiguous from exc
                if not object_present:
                    raise AmbiguousDeletionError(
                        key, "delete failed after the object may have been removed"
                    ) from exc
                deletion_error = DeletionStorageError(key)
                try:
                    self.metadata.release_document_deletion_claim(
                        actor.tenant_id, document_id, claim_token=claim.claim_token
                    )
                except Exception as release_error:
                    deletion_error.add_note(
                        "Claim release also failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )
                raise deletion_error from exc
            if object_was_present:
                deleted_count += 1
            if (
                self._monotonic_clock() - started_at
                > self.retention_policy.object_delete_deadline_seconds
            ):
                raise AmbiguousDeletionError(
                    key, "delete completed after the observation deadline"
                )
            try:
                object_still_present = self.object_store.exists(key)
            except (OSError, ValueError) as probe_error:
                raise AmbiguousDeletionError(
                    key, "post-delete object inspection failed"
                ) from probe_error
            if object_still_present:
                raise AmbiguousDeletionError(
                    key, "delete returned but the object remains"
                )
            if not self.metadata.is_document_deletion_claim_current(
                actor.tenant_id,
                document_id,
                claim_token=claim.claim_token,
                now=self._wall_clock(),
            ):
                raise AmbiguousDeletionError(
                    key, "deletion claim was lost after delete"
                )

        finalized = self.metadata.finalize_document_deletion(
            actor.tenant_id,
            document_id,
            claim_token=claim.claim_token,
            now=self._wall_clock(),
        )
        if not finalized:
            raise ManagementConflict(
                "Document deletion claim was lost before finalization"
            )
        return ReconciliationResult(
            document_id=document_id,
            dry_run=False,
            object_keys=claim.manifest.object_keys,
            deleted_object_count=deleted_count,
            finalized=finalized,
        )

    def _dry_run_result(self, manifest: DeletionManifest) -> ReconciliationResult:
        return ReconciliationResult(
            document_id=manifest.document_id,
            dry_run=True,
            object_keys=manifest.object_keys,
            deleted_object_count=0,
            finalized=False,
        )

    def reconcile_orphans(
        self,
        actor: ActorContext,
        *,
        dry_run: bool = False,
        grace_seconds: float | None = None,
        now: float | None = None,
        limit: int | None = None,
        mutation_fence: Callable[[], None] | None = None,
    ) -> OrphanReconciliationResult:
        """Reconcile tracked private attempts with optional mutation fencing.

        ``mutation_fence`` is an internal maintenance callback invoked immediately
        before and after each external delete. It cannot interrupt a blocking
        object-store call or roll back bytes; a post-delete failure therefore
        leaves the claim incomplete for a later recovery pass.
        """

        self._require_tenant_admin(actor)
        grace_seconds = (
            self.retention_policy.orphan_grace_seconds
            if grace_seconds is None
            else grace_seconds
        )
        limit = self.retention_policy.orphan_gc_batch_size if limit is None else limit
        if grace_seconds < 0:
            raise ValueError("Grace period must be non-negative")
        if limit <= 0 or limit > MAX_ORPHAN_GC_BATCH_SIZE:
            raise ValueError("GC batch size must be between 1 and 1000")
        fixed_time = now is not None
        current_time = time.time() if now is None else now
        try:
            attempts = self.metadata.list_expired_upload_attempts_for_tenant(
                actor.tenant_id,
                now=current_time,
                grace_seconds=grace_seconds,
                limit=MAX_ORPHAN_GC_BATCH_SIZE,
                actor=actor,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        try:
            audited_deleted_claims = self.metadata.next_deleted_cleanup_audit_page(
                actor.tenant_id,
                now=current_time,
                grace_seconds=grace_seconds,
                limit=limit,
                actor=actor,
                advance=not dry_run,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        reappeared_keys = [
            claim.object_key
            for claim in audited_deleted_claims
            if self._is_private_attempt_key(claim.object_key, claim.document_id)
            and self._gc_object_present(claim.object_key)
        ]
        truncated = len(attempts) > limit
        attempts = attempts[:MAX_ORPHAN_GC_BATCH_SIZE]
        try:
            expired_claims = self.metadata.list_expired_cleanup_claims(
                actor.tenant_id,
                now=current_time,
                grace_seconds=grace_seconds,
                limit=MAX_ORPHAN_GC_BATCH_SIZE,
                actor=actor,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        recovery_truncated = len(expired_claims) > limit
        expired_claims = expired_claims[:MAX_ORPHAN_GC_BATCH_SIZE]
        candidate_keys = list(reappeared_keys)
        for claim in expired_claims:
            if len(candidate_keys) == limit:
                break
            if self._is_private_attempt_key(
                claim.object_key, claim.document_id
            ) and not self._gc_object_present(claim.object_key):
                candidate_keys.append(claim.object_key)

        candidate_keys = list(dict.fromkeys(candidate_keys))
        missing_attempt_keys: list[str] = []
        for attempt in attempts:
            if len(candidate_keys) == limit:
                break
            if not self._is_private_attempt_key(
                attempt.object_key, attempt.document_id
            ):
                continue
            if attempt.object_key in candidate_keys:
                continue
            if self._gc_object_present(attempt.object_key):
                candidate_keys.append(attempt.object_key)
            else:
                missing_attempt_keys.append(attempt.object_key)
        truncated = truncated or recovery_truncated
        missing_work = missing_attempt_keys[:limit] if not candidate_keys else []
        if dry_run:
            return OrphanReconciliationResult(
                dry_run=True,
                object_keys=tuple(candidate_keys),
                claimed_object_count=0,
                deleted_object_count=0,
                truncated=truncated,
                limit=limit,
                eligible_missing_count=len(missing_work),
            )

        # Progress past already-missing attempts without exceeding the caller's
        # batch bound. Present objects keep priority over metadata-only cleanup.
        claim_keys = candidate_keys if candidate_keys else missing_work
        try:
            claims = self.metadata.claim_expired_upload_attempts(
                actor.tenant_id,
                claim_keys,
                now=current_time,
                grace_seconds=grace_seconds,
                claim_ttl_seconds=self.retention_policy.cleanup_claim_ttl_seconds,
                actor=actor,
                reclaim_deleted_keys=reappeared_keys,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        deleted_count = 0
        reconciled_missing_count = 0
        for claim in claims:
            claim_check_time = current_time if fixed_time else time.time()
            if not self.metadata.is_cleanup_claim_current(
                actor.tenant_id,
                claim.object_key,
                claim_token=claim.claim_token,
                now=claim_check_time,
            ):
                continue
            if mutation_fence is not None:
                mutation_fence()
            try:
                object_was_present = self.object_store.exists(claim.object_key)
            except (OSError, ValueError) as probe_error:
                inspection_error = DeletionStorageError(
                    claim.object_key, "Owned object-store pre-delete inspection failed"
                )
                try:
                    self.metadata.release_cleanup_claim(
                        actor.tenant_id,
                        claim.object_key,
                        claim_token=claim.claim_token,
                    )
                except Exception as probe_release_error:
                    inspection_error.add_note(
                        "Cleanup claim release also failed: "
                        f"{type(probe_release_error).__name__}: {probe_release_error}"
                    )
                raise inspection_error from probe_error
            deletion_error: DeletionStorageError | None = None
            delete_exception: OSError | ValueError | None = None
            delete_returned = False
            started_at = self._monotonic_clock()
            try:
                self.object_store.delete(claim.object_key)
            except (OSError, ValueError) as exc:
                delete_exception = exc
                deletion_error = self._classify_delete_exception(claim.object_key, exc)
            else:
                delete_returned = True
                elapsed = self._monotonic_clock() - started_at
                if elapsed > self.retention_policy.object_delete_deadline_seconds:
                    deletion_error = AmbiguousDeletionError(
                        claim.object_key,
                        "delete completed after the observation deadline",
                    )
            if delete_returned and mutation_fence is not None:
                mutation_fence()
            if delete_returned and deletion_error is None:
                try:
                    object_still_present = self.object_store.exists(claim.object_key)
                except (OSError, ValueError) as probe_error:
                    deletion_error = AmbiguousDeletionError(
                        claim.object_key, "post-delete object inspection failed"
                    )
                    deletion_error.add_note(
                        "Object presence probe failed: "
                        f"{type(probe_error).__name__}: {probe_error}"
                    )
                else:
                    if object_still_present:
                        deletion_error = AmbiguousDeletionError(
                            claim.object_key,
                            "delete returned but the object remains",
                        )
            if deletion_error is not None:
                if isinstance(deletion_error, AmbiguousDeletionError):
                    if delete_exception is None:
                        raise deletion_error
                    raise deletion_error from delete_exception
                release_error: Exception | None = None
                try:
                    self.metadata.release_cleanup_claim(
                        actor.tenant_id,
                        claim.object_key,
                        claim_token=claim.claim_token,
                    )
                except Exception as error:
                    release_error = error
                if release_error is not None:
                    deletion_error.add_note(
                        "Cleanup claim release also failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )
                if delete_exception is None:
                    raise deletion_error
                raise deletion_error from delete_exception
            complete_time = current_time if fixed_time else time.time()
            if self.metadata.complete_cleanup_claim(
                actor.tenant_id,
                claim.object_key,
                claim_token=claim.claim_token,
                now=complete_time,
            ):
                if object_was_present:
                    deleted_count += 1
                else:
                    reconciled_missing_count += 1
        return OrphanReconciliationResult(
            dry_run=False,
            object_keys=tuple(
                claim.object_key
                for claim in claims
                if claim.object_key in candidate_keys
            ),
            claimed_object_count=len(claims),
            deleted_object_count=deleted_count,
            truncated=truncated,
            limit=limit,
            reconciled_missing_count=reconciled_missing_count,
            eligible_missing_count=len(missing_work),
        )

    def _classify_delete_exception(
        self, key: str, delete_error: OSError | ValueError
    ) -> DeletionStorageError:
        try:
            object_is_present = self.object_store.exists(key)
        except (OSError, ValueError) as probe_error:
            ambiguous = AmbiguousDeletionError(
                key,
                "delete failed and post-error object inspection also failed",
            )
            ambiguous.add_note(
                "Object presence probe failed: "
                f"{type(probe_error).__name__}: {probe_error}"
            )
            return ambiguous
        if not object_is_present:
            return AmbiguousDeletionError(
                key,
                "delete failed after the object may have been removed; "
                "outcome is unknown",
            )
        return DeletionStorageError(key)

    def _gc_object_present(self, key: str) -> bool:
        """Classify candidate inspection failures before any cleanup claim."""

        try:
            return self.object_store.exists(key)
        except (OSError, ValueError) as exc:
            raise DeletionStorageError(
                key, "Owned object-store inspection failed before cleanup claim"
            ) from exc

    @staticmethod
    def _is_private_attempt_key(key: str, document_id: str) -> bool:
        parts = key.split("/")
        if len(parts) != 4 or parts[:3] != ["documents", document_id, "attempts"]:
            return False
        filename = parts[3]
        nonce = filename.removesuffix(".bin")
        return (
            len(nonce) == 32
            and filename == f"{nonce}.bin"
            and all(character in "0123456789abcdef" for character in nonce)
        )

    def _require_tenant_admin(self, actor: ActorContext | None) -> None:
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None or not self.metadata.is_tenant_admin(
            scope.tenant_id, scope.user_id
        ):
            raise ManagementForbidden("Tenant admin permission required")
