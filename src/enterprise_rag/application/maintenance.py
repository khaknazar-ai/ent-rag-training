"""Bounded maintenance operations suitable for an external scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from enterprise_rag.security import ActorContext

from .deletion import (
    DocumentDeletionService,
    OrphanReconciliationResult,
)
from .ingestion_reconciliation import (
    IngestionReconciliationService,
    StageAttemptReconciliationResult,
    TerminalRunReconciliationResult,
)
from .management import ManagementConflict, ManagementForbidden

DEFAULT_MAINTENANCE_LEASE_SECONDS = 5 * 60


class MaintenanceBusy(ManagementConflict):
    """Raised when another maintenance invocation owns the tenant lease."""

    def __init__(self) -> None:
        super().__init__("Maintenance is already running for this tenant")


class MaintenanceLeaseLost(ManagementConflict):
    """Raised when ownership is lost before the next phase or final result."""

    def __init__(self) -> None:
        super().__init__("Maintenance lease was lost before the next phase")


@dataclass(frozen=True, slots=True)
class MaintenanceRunResult:
    """Results of one bounded, explicitly invoked maintenance pass."""

    dry_run: bool
    batch_size: int
    stale_stages: StageAttemptReconciliationResult
    terminal_runs: TerminalRunReconciliationResult
    orphan_gc: OrphanReconciliationResult


class MaintenanceService:
    """Compose safe maintenance passes without starting a background daemon."""

    def __init__(
        self,
        deletion: DocumentDeletionService,
        ingestion_reconciliation: IngestionReconciliationService,
    ) -> None:
        self.deletion = deletion
        self.ingestion_reconciliation = ingestion_reconciliation

    def run_once(
        self,
        actor: ActorContext | None,
        *,
        dry_run: bool = False,
        grace_seconds: float | None = None,
        batch_size: int | None = None,
        lease_seconds: float = DEFAULT_MAINTENANCE_LEASE_SECONDS,
        now: float | None = None,
    ) -> MaintenanceRunResult:
        """Run bounded repair and orphan GC with a final lease-result fence."""

        policy = self.deletion.retention_policy
        effective_grace_seconds = (
            policy.orphan_grace_seconds if grace_seconds is None else grace_seconds
        )
        effective_batch_size = (
            policy.orphan_gc_batch_size if batch_size is None else batch_size
        )
        if effective_batch_size <= 0 or effective_batch_size > 1000:
            raise ValueError("Maintenance batch size must be between 1 and 1000")
        if effective_grace_seconds < 0:
            raise ValueError("Grace period must be non-negative")
        if lease_seconds <= 0:
            raise ValueError("Maintenance lease duration must be positive")
        self._require_tenant_admin(actor)
        assert actor is not None
        current_time = time.time() if now is None else now
        try:
            lease = self.deletion.metadata.acquire_maintenance_lease(
                actor.tenant_id,
                f"maintenance-{uuid4().hex}",
                now=current_time,
                lease_seconds=lease_seconds,
                actor=actor,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        if lease is None:
            raise MaintenanceBusy
        active_lease = lease

        def heartbeat() -> None:
            nonlocal active_lease
            heartbeat_time = time.time() if now is None else current_time
            renewed = self.deletion.metadata.renew_maintenance_lease(
                active_lease,
                now=heartbeat_time,
                lease_seconds=lease_seconds,
            )
            if renewed is None:
                raise MaintenanceLeaseLost
            active_lease = renewed

        operation_error: BaseException | None = None
        try:
            stale_stages = self.ingestion_reconciliation.reconcile_stale_stages(
                actor,
                dry_run=dry_run,
                limit=effective_batch_size,
                now=current_time,
            )
            heartbeat()
            terminal_runs = self.ingestion_reconciliation.reconcile_terminal_runs(
                actor,
                dry_run=dry_run,
                limit=effective_batch_size,
            )
            heartbeat()
            orphan_gc = self.deletion.reconcile_orphans(
                actor,
                dry_run=dry_run,
                grace_seconds=effective_grace_seconds,
                limit=effective_batch_size,
                now=now,
                mutation_fence=heartbeat,
            )
            # The final bounded phase can outlive the lease just like the
            # earlier phases. Fence the result before making it observable.
            heartbeat()
        except BaseException as error:
            operation_error = error
            raise
        finally:
            try:
                released = self.deletion.metadata.release_maintenance_lease(
                    active_lease
                )
            except BaseException:
                if operation_error is None:
                    raise
            else:
                if not released and operation_error is None:
                    raise MaintenanceLeaseLost
        return MaintenanceRunResult(
            dry_run=dry_run,
            batch_size=effective_batch_size,
            stale_stages=stale_stages,
            terminal_runs=terminal_runs,
            orphan_gc=orphan_gc,
        )

    def _require_tenant_admin(self, actor: ActorContext | None) -> None:
        scope = self.deletion.metadata.resolve_security_scope(actor)
        if scope is None or not self.deletion.metadata.is_tenant_admin(
            scope.tenant_id, scope.user_id
        ):
            raise ManagementForbidden("Tenant admin permission required")
