"""Explicit foreground recovery of one owned worker child at a time."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from math import isfinite

from .health import responsive_wait
from .processes import OwnedChild


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    poll_seconds: float = 1
    restart_backoff_seconds: float = 1
    job_timeout_seconds: float = 300
    max_restarts: int = 3
    max_runs: int | None = None

    def __post_init__(self) -> None:
        for value in (
            self.poll_seconds,
            self.restart_backoff_seconds,
            self.job_timeout_seconds,
        ):
            if isinstance(value, bool) or not isfinite(value) or not 0 < value <= 86400:
                raise ValueError(
                    "Supervisor durations must be finite and in (0, 86400]"
                )
        if (
            isinstance(self.max_restarts, bool)
            or not isinstance(self.max_restarts, int)
            or self.max_restarts < 0
        ):
            raise ValueError("Supervisor max-restarts must be a non-negative integer")
        if self.max_runs is not None and (
            isinstance(self.max_runs, bool)
            or not isinstance(self.max_runs, int)
            or self.max_runs < 1
        ):
            raise ValueError("Supervisor max-runs must be a positive integer")


class WorkerSupervisor:
    def __init__(
        self,
        command: Sequence[str],
        config: SupervisorConfig,
        *,
        child_factory: Callable = OwnedChild,
        emit: Callable[[dict], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        pulse: Callable[[], None] | None = None,
    ) -> None:
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("Supervisor child command must be non-empty")
        self.command = tuple(command)
        self.config = config
        self._child_factory = child_factory
        self._emit = emit or (lambda _: None)
        self._monotonic = monotonic
        self._pulse = pulse

    def run(
        self, *, stop_requested: Callable[[], bool], wait: Callable[[float], bool]
    ) -> dict:
        if self._pulse is not None:
            raw_wait = wait
            wait = partial(
                responsive_wait, raw_wait, self._pulse, monotonic=self._monotonic
            )
        polls = restarts = failures = 0
        last_exit = None
        last_failure = None
        pending_restart = False

        def result(reason):
            return {
                "polls": polls,
                "restarts": restarts,
                "failures": failures,
                "last_exit_code": last_exit,
                "last_failure": last_failure,
                "stop_reason": reason,
            }

        while not stop_requested():
            if self._pulse is not None:
                self._pulse()
            if pending_restart:
                restarts += 1
            polls += 1
            timed_out = stopped = False
            deadline = self._monotonic() + self.config.job_timeout_seconds
            with self._child_factory(self.command) as child:
                self._emit(
                    {
                        "event": "worker_supervisor_started",
                        "pid": child.pid,
                        "poll": polls,
                    }
                )
                while (last_exit := child.poll()) is None:
                    if self._pulse is not None:
                        self._pulse()
                    if stop_requested():
                        stopped = True
                        break
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    if wait(min(0.1, remaining)):
                        stopped = True
                        break
                stopped = stopped or stop_requested()
                if not stopped and self._monotonic() >= deadline:
                    timed_out = True
            # The owned tree is confirmed stopped before any restart/result.
            if stopped:
                return result("stopped")
            if not timed_out and last_exit == 0:
                pending_restart = False
                if self.config.max_runs is not None and polls >= self.config.max_runs:
                    return result("completed")
                if wait(self.config.poll_seconds):
                    return result("stopped")
                continue
            failures += 1
            last_failure = "timeout" if timed_out else "child_exit"
            self._emit(
                {
                    "event": "worker_supervisor_failure",
                    "reason": last_failure,
                    "exit_code": last_exit,
                }
            )
            if last_exit == 2 and not timed_out:
                return result("configuration_error")
            if failures > self.config.max_restarts:
                return result("restart_budget_exhausted")
            if self.config.max_runs is not None and polls >= self.config.max_runs:
                return result("failed_at_poll_limit")
            self._emit(
                {
                    "event": "worker_supervisor_restart",
                    "reason": "timeout" if timed_out else "child_exit",
                    "exit_code": last_exit,
                }
            )
            pending_restart = True
            backoff = min(
                self.config.restart_backoff_seconds * (2 ** min(failures - 1, 16)),
                86400,
            )
            if wait(backoff):
                return result("stopped")
        return result("stopped")
