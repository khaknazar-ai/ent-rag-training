"""Tenant-admin audit of durable ingestion consistency."""

from __future__ import annotations

import time
from dataclasses import dataclass

from enterprise_rag.persistence import (
    IngestionAuditFinding,
    SQLiteMetadataStore,
    StageAttemptReconciliation,
)
from enterprise_rag.security import ActorContext

from .management import ManagementForbidden

DEFAULT_INGESTION_AUDIT_LIMIT = 100


@dataclass(frozen=True, slots=True)
class IngestionReconciliationResult:
    """Bounded, content-free findings from one ingestion audit."""

    findings: tuple[IngestionAuditFinding, ...]
    truncated: bool
    limit: int


@dataclass(frozen=True, slots=True)
class StageAttemptReconciliationResult:
    """Result of a dry-run or explicit stale-stage repair."""

    dry_run: bool
    candidate_attempt_ids: tuple[str, ...]
    repaired_attempt_ids: tuple[str, ...]
    truncated: bool
    limit: int


@dataclass(frozen=True, slots=True)
class TerminalRunReconciliationResult:
    """Result of a safe terminal-job run-state repair."""

    dry_run: bool
    candidate_run_ids: tuple[str, ...]
    repaired_run_ids: tuple[str, ...]
    truncated: bool
    limit: int


class IngestionReconciliationService:
    """Expose read-only state inspection behind the tenant-admin boundary."""

    def __init__(self, metadata: SQLiteMetadataStore) -> None:
        self.metadata = metadata

    def audit(
        self,
        actor: ActorContext | None,
        *,
        limit: int = DEFAULT_INGESTION_AUDIT_LIMIT,
        now: float | None = None,
    ) -> IngestionReconciliationResult:
        if limit <= 0 or limit > 1000:
            raise ValueError("Audit limit must be between 1 and 1000")
        self._require_tenant_admin(actor)
        assert actor is not None
        try:
            findings, truncated = self.metadata.audit_ingestion(
                actor.tenant_id, limit=limit, now=now, actor=actor
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        return IngestionReconciliationResult(
            findings=findings,
            truncated=truncated,
            limit=limit,
        )

    def _require_tenant_admin(self, actor: ActorContext | None) -> None:
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None or not self.metadata.is_tenant_admin(
            scope.tenant_id, scope.user_id
        ):
            raise ManagementForbidden("Tenant admin permission required")

    def reconcile_terminal_runs(
        self,
        actor: ActorContext | None,
        *,
        dry_run: bool = False,
        limit: int = DEFAULT_INGESTION_AUDIT_LIMIT,
    ) -> TerminalRunReconciliationResult:
        """Repair only identity-consistent failed/cancelled terminal-run mismatches."""

        if limit <= 0 or limit > 1000:
            raise ValueError("Reconciliation limit must be between 1 and 1000")
        self._require_tenant_admin(actor)
        assert actor is not None
        try:
            outcome = self.metadata.reconcile_terminal_runs(
                actor.tenant_id,
                limit=limit,
                dry_run=dry_run,
                actor=actor,
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        return TerminalRunReconciliationResult(
            dry_run=dry_run,
            candidate_run_ids=outcome.candidate_run_ids,
            repaired_run_ids=outcome.repaired_run_ids,
            truncated=outcome.truncated,
            limit=limit,
        )

    def reconcile_stale_stages(
        self,
        actor: ActorContext | None,
        *,
        dry_run: bool = False,
        limit: int = DEFAULT_INGESTION_AUDIT_LIMIT,
        now: float | None = None,
    ) -> StageAttemptReconciliationResult:
        """Repair only stage attempts proven stale by terminal job state."""

        if limit <= 0 or limit > 1000:
            raise ValueError("Reconciliation limit must be between 1 and 1000")
        self._require_tenant_admin(actor)
        assert actor is not None
        try:
            outcome: StageAttemptReconciliation = (
                self.metadata.reconcile_stale_stage_attempts(
                    actor.tenant_id,
                    now=time.time() if now is None else now,
                    limit=limit,
                    dry_run=dry_run,
                    actor=actor,
                )
            )
        except PermissionError as exc:
            raise ManagementForbidden("Tenant admin permission required") from exc
        return StageAttemptReconciliationResult(
            dry_run=dry_run,
            candidate_attempt_ids=outcome.candidate_attempt_ids,
            repaired_attempt_ids=outcome.repaired_attempt_ids,
            truncated=outcome.truncated,
            limit=limit,
        )
