"""Finite foreground lifecycle for explicitly scheduled maintenance passes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from enterprise_rag.security import ActorContext

from .maintenance import MaintenanceRunResult

StopReason = Literal["completed", "stopped"]
RunOnce = Callable[..., MaintenanceRunResult]
Wait = Callable[[float], bool]
StopRequested = Callable[[], bool]
MAX_MAINTENANCE_LOOP_INTERVAL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class MaintenanceLoopResult:
    """Bounded lifecycle summary without retaining an unbounded run history."""

    attempted_runs: int
    completed_runs: int
    stop_reason: StopReason
    last_result: MaintenanceRunResult | None


class MaintenanceScheduler:
    """Run maintenance sequentially for a finite, explicitly bounded schedule.

    This class is deliberately a foreground runner. It does not create threads,
    install signal handlers, retry failures, or start itself from application
    bootstrap. An external caller owns process lifecycle and supplies an
    interruptible wait and stop predicate.
    """

    def __init__(self, run_once: RunOnce, *, wait: Wait | None = None) -> None:
        self._run_once = run_once
        self._wait = _default_wait if wait is None else wait

    def run(
        self,
        actor: ActorContext | None,
        *,
        max_runs: int,
        interval_seconds: float,
        dry_run: bool = False,
        grace_seconds: float | None = None,
        batch_size: int | None = None,
        stop_requested: StopRequested | None = None,
    ) -> MaintenanceLoopResult:
        """Run up to max_runs, stopping only between bounded maintenance passes."""

        _validate_configuration(max_runs, interval_seconds)
        should_stop = _never_stop if stop_requested is None else stop_requested
        attempted_runs = 0
        completed_runs = 0
        last_result: MaintenanceRunResult | None = None

        while attempted_runs < max_runs:
            if should_stop():
                return MaintenanceLoopResult(
                    attempted_runs,
                    completed_runs,
                    "stopped",
                    last_result,
                )

            attempted_runs += 1
            last_result = self._run_once(
                actor,
                dry_run=dry_run,
                grace_seconds=grace_seconds,
                batch_size=batch_size,
            )
            completed_runs += 1

            if should_stop():
                return MaintenanceLoopResult(
                    attempted_runs,
                    completed_runs,
                    "stopped",
                    last_result,
                )
            if attempted_runs == max_runs:
                return MaintenanceLoopResult(
                    attempted_runs,
                    completed_runs,
                    "completed",
                    last_result,
                )
            if should_stop() or self._wait(interval_seconds):
                return MaintenanceLoopResult(
                    attempted_runs,
                    completed_runs,
                    "stopped",
                    last_result,
                )

        raise AssertionError("Maintenance scheduler exited without a result")


def _validate_configuration(max_runs: int, interval_seconds: float) -> None:
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
        raise ValueError("Maintenance loop max-runs must be at least 1")
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or not isfinite(interval_seconds)
        or interval_seconds <= 0
        or interval_seconds > MAX_MAINTENANCE_LOOP_INTERVAL_SECONDS
    ):
        raise ValueError(
            "Maintenance loop interval-seconds must be finite, positive, "
            "and at most 86400 seconds"
        )


def _default_wait(seconds: float) -> bool:
    time.sleep(seconds)
    return False


def _never_stop() -> bool:
    return False
