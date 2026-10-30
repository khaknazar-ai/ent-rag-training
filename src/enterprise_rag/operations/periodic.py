"""Persisted admission, not Scheduler exit codes, bounds periodic recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

from enterprise_rag.operations import scheduled
from enterprise_rag.operations.scheduled import single_instance
from enterprise_rag.workers.processes import (
    ChildStartCancelled,
    ChildTerminationError,
    OwnedChild,
    _WindowsJob,
    windows_job_active,
)

MAX_ADMISSIONS = 4
OUTCOME_CONFIG = "RAG_PERIODIC_OUTCOME_CONFIG"
OUTCOME_RUN = "RAG_PERIODIC_OUTCOME_RUN"


class Control:
    def __init__(self, config):
        self.database = str(
            (Path(config["data_dir"]) / "enterprise_rag.sqlite3").resolve()
        ).casefold()
        self.sid = config["sid"]
        self.key = hashlib.sha256(self.database.encode()).hexdigest()[:16]
        base = Path(config["data_dir"]) / f".scheduled-{self.key}"
        self.path = base.with_suffix(".control.json")
        self.pending = base.with_suffix(".control.pending")
        self.marker = base.with_suffix(".initialized")
        self.control_lock = base.with_suffix(".control.lock")
        self.root_lock = base.with_suffix(".lock")

    @contextmanager
    def locked(self):
        deadline = time.monotonic() + 10
        while True:
            manager = single_instance(self.control_lock)
            try:
                manager.__enter__()
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        try:
            yield
        finally:
            manager.__exit__(None, None, None)

    def _commit(self, value):
        # A surviving pending file is evidence of an interrupted write, never
        # automatically removed or treated as a new lifetime.
        with self.pending.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.pending, self.path)

    def _read(self, *, initialize=False):
        if self.pending.exists():
            raise ValueError("Unresolved periodic control pending write")
        if not self.marker.exists() and not self.path.exists() and initialize:
            with self.marker.open("x", encoding="utf-8") as stream:
                stream.write("WTL2\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._commit(
                {
                    "version": 1,
                    "database": self.database,
                    "sid": self.sid,
                    "attempts": 0,
                    "status": "ready",
                    "run_id": None,
                    "job_name": None,
                    "reason": None,
                    "exit_code": None,
                }
            )
        if not self.marker.exists() or not self.path.exists():
            raise ValueError("Missing initialized periodic control; no implicit reset")
        if (
            self.marker.read_text(encoding="utf-8") != "WTL2\n"
            or self.path.stat().st_size > 16384
        ):
            raise ValueError("Invalid periodic control marker or size")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        fields = {
            "version",
            "database",
            "sid",
            "attempts",
            "status",
            "run_id",
            "job_name",
            "reason",
            "exit_code",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["database"] != self.database
            or value["sid"] != self.sid
            or type(value["attempts"]) is not int
            or not 0 <= value["attempts"] <= MAX_ADMISSIONS
            or not isinstance(value["status"], str)
            or value["status"]
            not in {"ready", "running", "terminal", "stopped", "exhausted"}
            or (
                value["reason"] is not None
                and (not isinstance(value["reason"], str) or len(value["reason"]) > 512)
            )
            or (
                value["exit_code"] is not None
                and (
                    type(value["exit_code"]) is not int
                    or value["exit_code"] not in {0, 2, 3, 130}
                )
            )
        ):
            raise ValueError("Invalid periodic control record")
        count = value["attempts"]
        run_id = value["run_id"]
        if count == 0:
            if (
                run_id is not None
                or value["job_name"] is not None
                or value["status"] not in {"ready", "stopped"}
            ):
                raise ValueError("Inconsistent initial periodic state")
        elif (
            not isinstance(run_id, str)
            or not re.fullmatch(r"[a-f0-9]{32}", run_id)
            or value["job_name"] != self.job_name(run_id)
        ):
            raise ValueError("Inconsistent periodic admission identity")
        if (value["status"] == "ready" and count != 0) or (
            value["status"] == "exhausted" and count != MAX_ADMISSIONS
        ):
            raise ValueError("Inconsistent periodic budget")
        if value["status"] in {"ready", "running"} and (
            value["reason"] is not None or value["exit_code"] is not None
        ):
            raise ValueError("Running admission cannot contain a terminal outcome")
        if value["status"] in {"terminal", "stopped", "exhausted"} and (
            not value["reason"] or value["exit_code"] is None
        ):
            raise ValueError("Terminal admission requires a durable outcome")
        return value

    def job_name(self, run_id):
        return f"Local\\EnterpriseRAG-{self.key}-{run_id}"

    def read(self):
        with self.locked():
            return self._read()

    def quiescent(self):
        try:
            with single_instance(self.root_lock):
                value = self.read()
                return (
                    value["job_name"] is None
                    or windows_job_active(value["job_name"]) == 0
                )
        except BlockingIOError:
            return False

    def admit(self, run_id, *, job_active=windows_job_active):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid admission id")
        with self.locked():
            value = self._read(initialize=True)
            if value["status"] not in {"ready", "running"}:
                return None
            if value["job_name"] is not None and job_active(value["job_name"]) != 0:
                raise ChildTerminationError("Previous containment job still active")
            if value["attempts"] == MAX_ADMISSIONS:
                value.update(
                    status="exhausted",
                    reason="root_crash_budget_exhausted",
                    exit_code=3,
                )
                self._commit(value)
                return None
            value.update(
                attempts=value["attempts"] + 1,
                status="running",
                run_id=run_id,
                job_name=self.job_name(run_id),
                reason=None,
                exit_code=None,
            )
            self._commit(value)
            return value

    def finish(self, run_id, reason, code):
        with self.locked():
            value = self._read()
            if value["run_id"] != run_id:
                raise ValueError("Periodic terminal outcome has wrong admission")
            if value["status"] in {"stopped", "terminal"}:
                return value  # Never overwrite an already committed Stop/outcome.
            value.update(status="terminal", reason=reason, exit_code=code)
            self._commit(value)
            return value

    def stop(self):
        with self.locked():
            value = self._read(initialize=True)
            value.update(status="stopped", reason="intentional_stop", exit_code=130)
            self._commit(value)
            return value

    @contextmanager
    def start_guard(self, run_id):
        # Serialize Stop commit with actual gate release, not a stale check.
        with self.locked():
            value = self._read()
            if value["status"] != "running" or value["run_id"] != run_id:
                raise ChildStartCancelled("Periodic admission stopped before gate")
            yield


def run_periodic(config, path=None):
    control = Control(config)
    lifetime = single_instance(control.root_lock)
    try:
        lifetime.__enter__()
    except BlockingIOError:
        return 0  # Active root: no admission consumed and no per-tick log growth.
    try:
        return _run_owned(config, path, control)
    finally:
        lifetime.__exit__(None, None, None)


def _run_owned(config, path, control):
    run_id = uuid4().hex
    admission = control.admit(run_id)
    if admission is None:
        return 0
    state_dir = Path(config["state_dir"])

    def emit(event):
        scheduled.append_event(
            state_dir / "events.jsonl",
            {
                **event,
                "config_id": config["config_id"],
                "run_id": run_id,
                "root_pid": os.getpid(),
                "timestamp": datetime.now(UTC).isoformat(),
                "attempt": admission["attempts"],
            },
        )

    last_check, stopped = -1.0, False

    def stop_requested():
        nonlocal last_check, stopped
        if time.monotonic() - last_check >= 0.2:
            stopped = control.read()["status"] != "running"
            last_check = time.monotonic()
        return stopped

    def wait(seconds):
        deadline = time.monotonic() + seconds
        while not stop_requested():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, 0.2))
        return True

    owner = None
    code, reason = 2, "configuration_error"
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        emit(
            {
                "event": "scheduled_watchdog_root_started",
                "sid": config["sid"],
                "elevated": False,
            }
        )
        owner = _WindowsJob(admission["job_name"])
        env = scheduled.child_environment(config)
        if path is not None:
            env.update({OUTCOME_CONFIG: str(path.resolve()), OUTCOME_RUN: run_id})
        if config.get("diagnostic_hold"):
            if path is None:
                raise ValueError("Diagnostic hold requires a verified config path")
            env["RAG_SYNTHETIC_HOLD_CONFIG"] = str(path.resolve())
        services = scheduled.build_services(scheduled.Settings.from_env(env))
        args = scheduled._build_parser().parse_args(
            ["worker-watchdog", *config["watchdog_args"]]
        )
        with open(os.devnull, "wb") as output:
            result = scheduled._worker_watchdog(
                services,
                args,
                stop_requested=stop_requested,
                wait=wait,
                emit=emit,
                child_factory=partial(
                    OwnedChild,
                    environment=env,
                    output=output,
                    owner_job=owner,
                    gate_guard=partial(control.start_guard, run_id),
                ),
            )
        reason = result["stop_reason"]
        code = (
            0
            if reason == "completed"
            else 130
            if reason == "stopped"
            else 2
            if reason == "configuration_error"
            else 3
        )
        terminal = control.finish(run_id, reason, code)
        code = terminal["exit_code"]
        reason = terminal["reason"]
        emit(
            {
                "event": "scheduled_watchdog_root_finished",
                "result": result,
                "actual_exit_code": code,
                "scheduler_exit_code": 0,
                "terminal_reason": reason,
            }
        )
        return 0
    except Exception as exc:
        # Observed terminal failures latch before diagnostics or lock release.
        terminal = control.finish(run_id, reason, code)
        code, reason = terminal["exit_code"], terminal["reason"]
        scheduled.write_diagnostic(
            state_dir / "latest-failure.json",
            {
                "event": "periodic_watchdog_failure",
                "type": type(exc).__name__,
                "message": str(exc),
                "actual_exit_code": code,
                "terminal_reason": reason,
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return 0
    finally:
        if owner is not None:
            owner.close()
