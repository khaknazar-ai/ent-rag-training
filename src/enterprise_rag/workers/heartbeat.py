"""Opt-in lease renewal during one bounded executor/checkpoint operation."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event, Thread

from enterprise_rag.domain.ingestion import JobLease
from enterprise_rag.ingestion.errors import ParserContainmentError
from enterprise_rag.persistence import SQLiteMetadataStore


class HeartbeatLost(RuntimeError):
    """Renewal is lost or unconfirmed; no stage result may be acknowledged."""


class HeartbeatShutdownError(RuntimeError):
    """The caller must retire a process with an unjoined renewal thread."""


HEARTBEAT_JOIN_SECONDS = 6


class LeaseHeartbeat:
    """Renew only one original fence, with explicit stop and duration limits.

    SQLite's busy timeout is five seconds. Join allows one such in-flight
    renewal plus a margin. An unconfirmed stop fails closed; a supervisor owns
    the process deadline, because a thread cannot cancel blocking executor I/O.
    """

    def __init__(
        self,
        metadata: SQLiteMetadataStore,
        lease: JobLease,
        *,
        interval: float,
        lease_seconds: float,
        max_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._metadata = metadata
        self._lease = lease
        self._interval = interval
        self._lease_seconds = lease_seconds
        self._max_seconds = max_seconds
        self._clock = clock
        self._stop = Event()
        self._lost = Event()
        self._thread = Thread(
            target=self._run,
            name=f"rag-lease-heartbeat-{lease.job.id}",
            daemon=True,
        )

    def __enter__(self) -> LeaseHeartbeat:
        self._deadline = time.monotonic() + self._max_seconds
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=HEARTBEAT_JOIN_SECONDS)
        if isinstance(exc, ParserContainmentError):
            # Cleanup has already failed: preserve the fatal retirement signal.
            return
        if self._thread.is_alive():
            raise HeartbeatShutdownError("Heartbeat termination was not confirmed")
        if time.monotonic() >= self._deadline:
            self._lost.set()
        if self._lost.is_set() and (
            exc_type is None or issubclass(exc_type, Exception)
        ):
            raise HeartbeatLost("Job heartbeat was lost or unconfirmed")

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            if time.monotonic() >= self._deadline:
                self._lost.set()
                return
            try:
                renewed = self._metadata.renew_job_lease(
                    self._lease, now=self._clock(), lease_seconds=self._lease_seconds
                )
                if renewed is None:
                    self._lost.set()
                    return
            except Exception:
                self._lost.set()
                return
