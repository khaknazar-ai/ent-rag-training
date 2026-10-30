"""Explicit Windows foreground parent for crash recovery of a supervisor."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from math import isfinite

from .health import SupervisorHealth
from .processes import ChildStartCancelled, OwnedChild


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    max_restarts: int = 3
    restart_backoff_seconds: float = 1
    heartbeat_timeout_seconds: float = 15
    startup_timeout_seconds: float = 30

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_restarts, bool)
            or not isinstance(self.max_restarts, int)
            or self.max_restarts < 0
        ):
            raise ValueError("Watchdog max-restarts must be a non-negative integer")
        value = self.restart_backoff_seconds
        if isinstance(value, bool) or not isfinite(value) or not 0 < value <= 86400:
            raise ValueError("Watchdog backoff must be finite and in (0, 86400]")
        for duration in (self.heartbeat_timeout_seconds, self.startup_timeout_seconds):
            if (
                isinstance(duration, bool)
                or not isfinite(duration)
                or not 1 <= duration <= 86400
            ):
                raise ValueError(
                    "Watchdog health timeouts must be finite and in [1, 86400]"
                )


class SupervisorWatchdog:
    def __init__(
        self,
        command: Sequence[str],
        config: WatchdogConfig,
        *,
        child_factory: Callable = OwnedChild,
        emit: Callable[[dict], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if os.name != "nt":
            raise ValueError(
                "Supervisor watchdog requires Windows Job Object containment"
            )
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("Watchdog supervisor command must be non-empty")
        self.command = tuple(command)
        self.config = config
        self._child_factory = child_factory
        self._emit = emit or (lambda _: None)
        self._monotonic = monotonic

    def run(
        self, *, stop_requested: Callable[[], bool], wait: Callable[[float], bool]
    ) -> dict:
        starts = restarts = failures = 0
        last_exit = None

        def result(reason):
            return {
                "starts": starts,
                "restarts": restarts,
                "failures": failures,
                "last_exit_code": last_exit,
                "stop_reason": reason,
            }

        while not stop_requested():
            stopped = False
            health_failure = None
            acknowledged = False
            # Any startup/ownership/cleanup error escapes: never replace an
            # unconfirmed tree. OwnedChild contains nested descendants on Windows.
            try:
                with ExitStack() as lifetime:
                    health = lifetime.enter_context(
                        SupervisorHealth(
                            self.config.heartbeat_timeout_seconds,
                            self.config.startup_timeout_seconds,
                            started_at=self._monotonic(),
                        )
                    )
                    child = lifetime.enter_context(
                        self._child_factory(
                            self.command,
                            start_cancelled=stop_requested,
                            environment_overlay=health.environment,
                        )
                    )
                    starts += 1
                    restarts = starts - 1
                    self._emit(
                        {
                            "event": "supervisor_watchdog_started",
                            "pid": child.pid,
                            "start": starts,
                        }
                    )
                    while (last_exit := child.poll()) is None:
                        if stop_requested():
                            stopped = True
                            break
                        health_failure = health.check(
                            child.pid,
                            self._monotonic(),
                            pid_validator=getattr(child, "accepts_health_pid", None),
                        )
                        if health.responses and not acknowledged:
                            acknowledged = True
                            self._emit(
                                {
                                    "event": "supervisor_watchdog_responsive",
                                    "pid": child.pid,
                                    "reporter_pid": health.peer_pid,
                                }
                            )
                        if health_failure:
                            break
                        if wait(0.1):
                            stopped = True
                            break
                    stopped = stopped or stop_requested()
            except ChildStartCancelled:
                # OwnedChild raises this only after confirmed cleanup. A failed
                # cleanup instead escapes as its own error and cannot say stopped.
                return result("stopped")
            # Close has confirmed that the OLD supervisor AND workers are gone.
            if stopped or stop_requested():
                return result("stopped")
            terminal = {
                0: "completed",
                2: "configuration_error",
                3: "supervisor_failed",
                130: "stopped",
                143: "stopped",
            }
            if health_failure is None and last_exit in terminal:
                return result(terminal[last_exit])
            failures += 1
            self._emit(
                {
                    "event": "supervisor_watchdog_failure",
                    "exit_code": last_exit,
                    "reason": health_failure or "child_exit",
                    "pid": child.pid,
                    "responsive_before_failure": acknowledged,
                }
            )
            if failures > self.config.max_restarts:
                return result("restart_budget_exhausted")
            backoff = min(
                self.config.restart_backoff_seconds * 2 ** min(failures - 1, 16), 86400
            )
            self._emit(
                {
                    "event": "supervisor_watchdog_restart",
                    "exit_code": last_exit,
                    "backoff_seconds": backoff,
                }
            )
            if wait(backoff):
                return result("stopped")
        return result("stopped")
