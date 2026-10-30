"""One-job-at-a-time worker orchestration for the Docker-free profile."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, TypeVar

from enterprise_rag.domain.ingestion import (
    IngestionRun,
    JobLease,
    StageAttempt,
    StageStatus,
)
from enterprise_rag.ingestion.errors import ParserContainmentError
from enterprise_rag.persistence import SQLiteMetadataStore

from .heartbeat import HeartbeatLost, HeartbeatShutdownError, LeaseHeartbeat

T = TypeVar("T")


def validate_worker_settings(
    worker_id,
    lease_seconds,
    max_attempts,
    retry_delay,
    heartbeat_interval_seconds,
    heartbeat_max_seconds,
):
    """Pure preflight shared by worker construction and scheduled startup."""
    if not worker_id.strip():
        raise ValueError("Worker id must be non-empty")
    if not isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("Lease duration must be positive")
    if max_attempts <= 0:
        raise ValueError("Maximum attempts must be positive")
    if retry_delay < 0:
        raise ValueError("Retry delay must be non-negative")
    if not isfinite(heartbeat_max_seconds) or heartbeat_max_seconds <= 0:
        raise ValueError("Heartbeat maximum duration must be finite and positive")
    interval = heartbeat_interval_seconds
    if interval is not None and (
        not isfinite(interval)
        or not 0 < interval < lease_seconds / 2
        or interval >= heartbeat_max_seconds
    ):
        raise ValueError(
            "Heartbeat interval must be positive and below half the lease and its duration bound"
        )


class StageExecutor(Protocol):
    """Explicit backend contract; real parsers/indexers are separate slices."""

    def execute(self, lease: JobLease, stage: str) -> str | None:
        """Execute one stage and return an optional output checksum."""


class StageExecutionError(RuntimeError):
    """A classified stage failure safe to persist without document contents."""

    def __init__(
        self,
        error_code: str,
        *,
        retryable: bool,
        retry_delay: float | None = None,
    ) -> None:
        if not error_code.strip():
            raise ValueError("Stage error code must be non-empty")
        if retry_delay is not None and retry_delay < 0:
            raise ValueError("Retry delay must be non-negative")
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable
        self.retry_delay = retry_delay


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Observable outcome of one bounded worker poll."""

    claimed: bool
    job_id: str | None
    reclaimed: bool = False
    completed: bool = False
    failure_recorded: bool = False
    lease_lost: bool = False


@dataclass(slots=True)
class LiteWorker:
    """Process one run with durable stage attempts and fenced mutations."""

    metadata: SQLiteMetadataStore
    worker_id: str
    lease_seconds: float = 30
    max_attempts: int = 3
    retry_delay: float = 1
    clock: Callable[[], float] = time.time
    heartbeat_interval_seconds: float | None = None
    heartbeat_max_seconds: float = 300
    _retired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        validate_worker_settings(
            self.worker_id,
            self.lease_seconds,
            self.max_attempts,
            self.retry_delay,
            self.heartbeat_interval_seconds,
            self.heartbeat_max_seconds,
        )

    def run_once(self, executor: StageExecutor) -> WorkerResult:
        """Claim and process at most one queued run."""

        if self._retired:
            raise ParserContainmentError("worker_retired_after_parser_cleanup")
        claimed_lease = self.metadata.claim_job(
            self.worker_id, self.clock(), self.lease_seconds
        )
        if claimed_lease is None:
            return WorkerResult(claimed=False, job_id=None)
        lease: JobLease = claimed_lease

        job_id = lease.job.id
        if lease.job.run_id is None or self.metadata.get_run(lease.job.run_id) is None:
            recorded = self.metadata.fail_job(
                lease,
                now=self.clock(),
                error="missing_run_metadata",
                retryable=False,
                max_attempts=self.max_attempts,
            )
            return WorkerResult(
                claimed=True,
                job_id=job_id,
                reclaimed=lease.reclaimed,
                failure_recorded=recorded,
                lease_lost=not recorded,
            )

        run = self.metadata.get_run(lease.job.run_id)
        assert run is not None
        try:
            stages = self._with_heartbeat(
                lease, lambda: self._pending_stages(lease, run, executor)
            )
        except HeartbeatLost:
            return WorkerResult(
                claimed=True, job_id=job_id, reclaimed=lease.reclaimed, lease_lost=True
            )
        for stage in stages:
            renewed = self._renew_lease(lease)
            if renewed is None:
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    lease_lost=True,
                )
            lease = renewed

            attempt = self.metadata.start_stage_attempt(lease, stage, now=self.clock())
            if attempt is None:
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    lease_lost=True,
                )

            def execute_stage(
                lease: JobLease = lease, stage: str = stage
            ) -> str | None:
                return executor.execute(lease, stage)

            try:
                output_checksum = self._with_heartbeat(lease, execute_stage)
            except HeartbeatLost:
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    lease_lost=True,
                )
            except ParserContainmentError:
                self._retired = True
                raise
            except HeartbeatShutdownError:
                raise
            except StageExecutionError as exc:
                renewed = self._renew_lease(lease)
                if renewed is None:
                    return WorkerResult(
                        claimed=True,
                        job_id=job_id,
                        reclaimed=lease.reclaimed,
                        lease_lost=True,
                    )
                lease = renewed
                finished = self.metadata.finish_stage_attempt(
                    lease,
                    attempt.id,
                    now=self.clock(),
                    success=False,
                    error=exc.error_code,
                )
                recorded = self.metadata.fail_job(
                    lease,
                    now=self.clock(),
                    error=exc.error_code,
                    retryable=exc.retryable,
                    max_attempts=self.max_attempts,
                    retry_delay=(
                        self.retry_delay if exc.retry_delay is None else exc.retry_delay
                    ),
                )
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    failure_recorded=recorded,
                    lease_lost=not (finished and recorded),
                )
            except Exception as exc:
                error_code = f"unhandled_{type(exc).__name__}"
                renewed = self._renew_lease(lease)
                if renewed is None:
                    return WorkerResult(
                        claimed=True,
                        job_id=job_id,
                        reclaimed=lease.reclaimed,
                        lease_lost=True,
                    )
                lease = renewed
                finished = self.metadata.finish_stage_attempt(
                    lease,
                    attempt.id,
                    now=self.clock(),
                    success=False,
                    error=error_code,
                )
                recorded = self.metadata.fail_job(
                    lease,
                    now=self.clock(),
                    error=error_code,
                    retryable=False,
                    max_attempts=self.max_attempts,
                )
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    failure_recorded=recorded,
                    lease_lost=not (finished and recorded),
                )

            renewed = self._renew_lease(lease)
            if renewed is None:
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    lease_lost=True,
                )
            lease = renewed
            finished = self.metadata.finish_stage_attempt(
                lease,
                attempt.id,
                now=self.clock(),
                success=True,
                output_checksum=output_checksum,
            )
            if not finished:
                return WorkerResult(
                    claimed=True,
                    job_id=job_id,
                    reclaimed=lease.reclaimed,
                    lease_lost=True,
                )

        renewed = self._renew_lease(lease)
        if renewed is None:
            return WorkerResult(
                claimed=True,
                job_id=job_id,
                reclaimed=lease.reclaimed,
                lease_lost=True,
            )
        completed = self.metadata.complete_job(renewed, now=self.clock())
        return WorkerResult(
            claimed=True,
            job_id=job_id,
            reclaimed=lease.reclaimed,
            completed=completed,
            lease_lost=not completed,
        )

    def _pending_stages(
        self,
        lease: JobLease,
        run: IngestionRun,
        executor: StageExecutor,
    ) -> tuple[str, ...]:
        """Resume after status and, when available, artifact validation."""

        verifier = getattr(executor, "verify_checkpoint", None)
        if not callable(verifier):
            verifier = None
        latest_attempts: dict[str, StageAttempt] = {}
        for attempt in self.metadata.get_stage_attempts(run.id):
            previous = latest_attempts.get(attempt.stage)
            if previous is None or attempt.attempt_number >= previous.attempt_number:
                latest_attempts[attempt.stage] = attempt

        pending: list[str] = []
        for stage in run.required_stages:
            latest_attempt = latest_attempts.get(stage)
            if (
                latest_attempt is None
                or latest_attempt.status is not StageStatus.SUCCEEDED
            ):
                pending.append(stage)
                continue
            if verifier is not None and not verifier(lease, latest_attempt):
                pending.append(stage)
        return tuple(pending)

    def _renew_lease(self, lease: JobLease) -> JobLease | None:
        """Fence each stage boundary without pretending to cancel the executor."""

        return self.metadata.renew_job_lease(
            lease, now=self.clock(), lease_seconds=self.lease_seconds
        )

    def _with_heartbeat(self, lease: JobLease, operation: Callable[[], T]) -> T:
        def guarded() -> T:
            try:
                return operation()
            except ParserContainmentError:
                # Latch before any context manager can replace this exception.
                self._retired = True
                raise

        if self.heartbeat_interval_seconds is None:
            return guarded()
        with LeaseHeartbeat(
            self.metadata,
            lease,
            interval=self.heartbeat_interval_seconds,
            lease_seconds=self.lease_seconds,
            max_seconds=self.heartbeat_max_seconds,
            clock=self.clock,
        ):
            return guarded()
