"""Real HTTP and process smoke on retained, disposable synthetic state only."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn

from enterprise_rag.api import StaticTokenAuthenticator, create_app
from enterprise_rag.bootstrap import build_services
from enterprise_rag.cli import main as cli_main
from enterprise_rag.diagnostics._evidence import require_evidence
from enterprise_rag.domain.documents import Tenant, User
from enterprise_rag.domain.ingestion import JobLease
from enterprise_rag.persistence import SQLiteMetadataStore
from enterprise_rag.security import ActorContext
from enterprise_rag.settings import Settings
from enterprise_rag.workers import LiteTextStageExecutor, LiteWorker
from enterprise_rag.workers.health import SupervisorPulse
from enterprise_rag.workers.processes import START_GATE_ENV, OwnedChild
from enterprise_rag.workers.supervisor import SupervisorConfig, WorkerSupervisor
from enterprise_rag.workers.watchdog import SupervisorWatchdog, WatchdogConfig

REPOSITORY = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPOSITORY / "scripts/smoke_lite_supervised.py"
AUTH_ENV = "RAG_ISOLATED_SMOKE_AUTH"


def validate_timeout(value: float) -> None:
    if not math.isfinite(value) or not 10 <= value <= 180:
        raise ValueError("Smoke timeout must be finite and between 10 and 180 seconds")


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _settings(root: Path) -> Settings:
    return Settings(
        profile="lite",
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        database_url=f"sqlite:///{(root / 'data/metadata.sqlite3').as_posix()}",
        object_store="local",
        llm_provider="none",
    )


def _environment(root: Path) -> dict[str, str]:
    # No provider credentials, user RAG configuration or proxy settings leak in.
    keep = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "SYSTEMDRIVE",
    }
    environment = {k: v for k, v in os.environ.items() if k.upper() in keep}
    settings = _settings(root)
    environment.update(
        {
            "PYTHONPATH": str(REPOSITORY / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RAG_PROFILE": "lite",
            "RAG_OBJECT_STORE": "local",
            "RAG_LLM_PROVIDER": "none",
            "RAG_DATA_DIR": str(settings.data_dir),
            "RAG_ARTIFACTS_DIR": str(settings.artifacts_dir),
            "RAG_DATABASE_URL": settings.database_url,
        }
    )
    return environment


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        p.relative_to(REPOSITORY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def _verify_sources(paths: list[Path], expected: dict[str, str]) -> None:
    _check(_source_hashes(paths) == expected, "Source files changed during smoke")


def _ready_client(base_url: str) -> httpx.Client | None:
    client = httpx.Client(base_url=base_url, timeout=2, trust_env=False)
    ready = False
    try:
        response = client.get("/health/ready")
        ready = response.status_code == 200 and response.json().get("status") == "ready"
        return client if ready else None
    except httpx.TransportError:
        return None
    finally:
        if not ready:
            client.close()


def _windows_process_alive(pid: int) -> bool:
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    api.OpenProcess.restype = ctypes.c_void_p
    api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
            return False
        raise ctypes.WinError(error)
    try:
        code = ctypes.c_ulong()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259  # STILL_ACTIVE
    finally:
        api.CloseHandle(handle)


def _exit_supervisor(marker: Path, crash: dict) -> None:
    _save(marker, crash)
    print(json.dumps({"event": "smoke_supervisor_crash", **crash}), flush=True)
    # No potentially blocking file/log operation may follow this final check.
    # Exit 29 is the evidence that this guard passed, not the earlier marker.
    _check(
        time.time() < crash["lease_expires_at"],
        "Job lease expired before injected exit",
    )
    os._exit(29)


def _hang_supervisor(marker: Path, fault: dict) -> None:
    # An accepted fault must prove the lease guard passed. Keep preparation
    # private: rejected guard + slow exception cleanup must not impersonate hang.
    pending = marker.with_suffix(".pending")
    _save(pending, fault)
    print(json.dumps({"event": "smoke_supervisor_hang", **fault}), flush=True)
    _check(
        time.time() < fault["lease_expires_at"],
        "Job lease expired before injected hang",
    )
    pending.replace(marker)
    while True:
        time.sleep(0.1)  # Real blocked control thread, never a responsive helper.


def _child(root: Path, role: str) -> int:
    _check(sys.stdin.buffer.read(1) == b"1", "Child start gate not released")
    os.environ.pop(START_GATE_ENV, None)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _check(
        manifest["kind"] == "synthetic" and manifest["purpose"] == "contract",
        "Not an isolated smoke directory",
    )
    hang_supervisor = manifest["configuration"]["fault_target"] == "supervisor_hang"
    fault_supervisor = manifest["configuration"]["fault_target"] in {
        "supervisor",
        "supervisor_hang",
    }
    marker = root / "injected-crash.json"
    held_path = root / "worker-held.json"
    if role == "api":
        bindings = json.loads(os.environ.pop(AUTH_ENV))
        auth = StaticTokenAuthenticator(
            {
                item["token"]: ActorContext(item["tenant_id"], item["user_id"])
                for item in bindings.values()
            }
        )
        app = create_app(build_services(_settings(root)), auth)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            _save(root / "api-port.json", {"pid": os.getpid(), "port": port})
            uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=port,
                    log_level="warning",
                    access_log=False,
                    timeout_graceful_shutdown=2,
                )
            ).run(sockets=[listener])
        return 0
    if role == "watchdog":
        old_worker_checked = False

        def factory(command, *, start_cancelled, environment_overlay):
            nonlocal old_worker_checked
            if marker.exists():
                crash = json.loads(marker.read_text(encoding="utf-8"))
                _check(
                    not _windows_process_alive(crash["worker_pid"]),
                    "Old worker still alive BEFORE supervisor replacement",
                )
                old_worker_checked = True
            return OwnedChild(
                command,
                environment=_environment(root),
                start_cancelled=start_cancelled,
                environment_overlay=environment_overlay,
            )

        def emit(event):
            if event["event"] == "supervisor_watchdog_responsive":
                path = root / "watchdog-responsive.json"
                pending = root / "watchdog-responsive.pending"
                _save(pending, {"pid": event["reporter_pid"]})
                pending.replace(path)
            if event["event"] == "supervisor_watchdog_started" and event["start"] > 1:
                event = {**event, "old_worker_terminated": old_worker_checked}
            print(json.dumps(event), flush=True)

        result = SupervisorWatchdog(
            [
                sys.executable,
                "-B",
                str(ENTRYPOINT),
                "--_role",
                "supervisor",
                "--_root",
                str(root),
            ],
            WatchdogConfig(
                max_restarts=1,
                restart_backoff_seconds=0.1,
                heartbeat_timeout_seconds=manifest["configuration"][
                    "watchdog_health_timeout_seconds"
                ],
                startup_timeout_seconds=manifest["configuration"][
                    "watchdog_startup_timeout_seconds"
                ],
            ),
            child_factory=factory,
            emit=emit,
        ).run(
            stop_requested=lambda: (root / "stop-watchdog").exists(),
            wait=threading.Event().wait,
        )
        _save(root / "watchdog-result.json", result)
        return 0 if result["stop_reason"] in {"completed", "stopped"} else 1
    if role == "supervisor":

        def stop_or_crash():
            if fault_supervisor and not marker.exists() and held_path.exists():
                # Inject only after the watchdog authenticated this control
                # loop. A Windows venv launcher has a distinct owned root PID.
                ready_path = root / "watchdog-responsive.json"
                if (
                    not ready_path.exists()
                    or json.loads(ready_path.read_text())["pid"] != os.getpid()
                ):
                    return False
                held = json.loads(held_path.read_text(encoding="utf-8"))
                _check(
                    _windows_process_alive(held["worker_pid"]),
                    "Worker exited before supervisor crash",
                )
                live_job = require_evidence(
                    build_services(_settings(root)).metadata.get_job(held["job_id"]),
                    "Missing live job before supervisor fault",
                )
                observed_at = time.time()
                _check(
                    live_job.lease_expires_at is not None
                    and live_job.lease_expires_at > observed_at,
                    "Job lease expired before supervisor crash",
                )
                inject = _hang_supervisor if hang_supervisor else _exit_supervisor
                inject(
                    marker,
                    {
                        **held,
                        "exit_code": None if hang_supervisor else 29,
                        "fault": "hang" if hang_supervisor else "crash",
                        "supervisor_pid": os.getpid(),
                        "observed_at": observed_at,
                        "lease_expires_at": live_job.lease_expires_at,
                    },
                )
            return (root / "stop-supervisor").exists()

        config = SupervisorConfig(
            poll_seconds=0.1,
            restart_backoff_seconds=0.1,
            job_timeout_seconds=15,
            max_restarts=2,
        )
        with SupervisorPulse.from_environment() as health:
            result = WorkerSupervisor(
                [
                    sys.executable,
                    "-B",
                    str(ENTRYPOINT),
                    "--_role",
                    "worker",
                    "--_root",
                    str(root),
                ],
                config,
                emit=lambda event: print(json.dumps(event), flush=True),
                child_factory=partial(OwnedChild, environment=_environment(root)),
                pulse=health.tick,
            ).run(stop_requested=stop_or_crash, wait=threading.Event().wait)
        _save(root / "supervisor-result.json", result)
        return 0 if result["stop_reason"] == "stopped" else 1
    _check(role == "worker", "Unknown child role")
    if marker.exists():
        return cli_main(
            [
                "worker",
                "--once",
                "--worker-id",
                "http-smoke-worker",
                "--lease-seconds",
                "1",
                "--heartbeat-interval-seconds",
                "0.1",
                "--heartbeat-max-seconds",
                "10",
            ]
        )

    class CrashAfterArtifactCommit(SQLiteMetadataStore):
        def finish_stage_attempt(
            self,
            lease: JobLease,
            attempt_id: str,
            now: float,
            *,
            success: bool,
            error: str | None = None,
            output_checksum: str | None = None,
        ) -> bool:
            attempt = require_evidence(
                self.get_stage_attempt(attempt_id), "Missing stage attempt before fault"
            )
            if attempt.stage == "chunk":
                if fault_supervisor:
                    pending = root / "worker-held.pending"
                    _save(
                        pending,
                        {
                            "job_id": lease.job.id,
                            "run_id": lease.job.run_id,
                            "stage": "chunk",
                            "worker_pid": os.getpid(),
                        },
                    )
                    pending.replace(held_path)
                    # Hold the real worker after the artifact transaction. The
                    # Last stage-executor heartbeat has joined; its freshly
                    # renewed lease is checked by the supervisor fault hook.
                    # Only owned-tree termination may retire this held worker.
                    while True:
                        time.sleep(0.05)
                _save(
                    marker,
                    {
                        "job_id": lease.job.id,
                        "run_id": lease.job.run_id,
                        "stage": "chunk",
                        "exit_code": 23,
                    },
                )
                print(
                    json.dumps(
                        {"event": "smoke_injected_crash", "job_id": lease.job.id}
                    ),
                    flush=True,
                )
                os._exit(23)
            return super().finish_stage_attempt(
                lease,
                attempt_id,
                now,
                success=success,
                error=error,
                output_checksum=output_checksum,
            )

    services = build_services(_settings(root))
    metadata = CrashAfterArtifactCommit(services.metadata.path)
    worker_result = LiteWorker(
        metadata,
        "http-smoke-worker",
        lease_seconds=1,
        heartbeat_interval_seconds=0.1,
        heartbeat_max_seconds=10,
    ).run_once(LiteTextStageExecutor(metadata, services.object_store))
    return 3 if worker_result.lease_lost else 0


def _bootstrap(root: Path, run_id: str, *, settings=None):
    services = build_services(settings or _settings(root))
    tenant = Tenant.create(f"http-smoke-{run_id}", "Synthetic smoke")
    foreign_tenant = Tenant.create(f"foreign-{run_id}", "Synthetic foreign tenant")
    services.metadata.ensure_tenant(tenant)
    services.metadata.ensure_tenant(foreign_tenant)
    identities = {
        "admin": User.create(tenant.id, "admin@example.invalid"),
        "reader": User.create(tenant.id, "reader@example.invalid"),
        "outsider": User.create(tenant.id, "outsider@example.invalid"),
        "foreign": User.create(foreign_tenant.id, "reader@example.invalid"),
    }
    for user in identities.values():
        services.metadata.upsert_user(user)
    services.metadata.grant_tenant_admin(tenant.id, identities["admin"].id)
    bindings = {
        name: {
            "token": secrets.token_urlsafe(32),
            "tenant_id": user.tenant_id,
            "user_id": user.id,
        }
        for name, user in identities.items()
    }
    return identities, bindings


def run_smoke(
    output_dir: Path,
    timeout_seconds: float = 60,
    *,
    restart_supervisor: bool = False,
    hang_supervisor: bool = False,
) -> tuple[Path, dict]:
    validate_timeout(timeout_seconds)
    if restart_supervisor and hang_supervisor:
        raise ValueError("Select one supervisor fault mode")
    restart_supervisor = restart_supervisor or hang_supervisor
    if restart_supervisor and os.name != "nt":
        raise ValueError("Supervisor recovery smoke requires Windows containment")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir)).resolve()
    run_id = uuid4().hex
    text = f"Supervised smoke{run_id}. Rotate synthetic keys every 30 days."
    content = text.encode("utf-8")
    manifest = {
        "run_id": run_id,
        "kind": "synthetic",
        "purpose": "contract",
        "created_at": datetime.now(UTC).isoformat(),
        "input_sha256": hashlib.sha256(content).hexdigest(),
        "configuration": {
            "profile": "lite",
            "model": "none",
            "lease_seconds": 1,
            "heartbeat_seconds": 0.1,
            "timeout_seconds": timeout_seconds,
            "watchdog_health_timeout_seconds": 2 if restart_supervisor else None,
            "watchdog_startup_timeout_seconds": 15 if restart_supervisor else None,
            "fault_target": "supervisor_hang"
            if hang_supervisor
            else "supervisor"
            if restart_supervisor
            else "worker",
        },
    }
    _save(root / "manifest.json", manifest)
    (root / "input.txt").write_bytes(content)
    source_paths = [
        ENTRYPOINT,
        *sorted((REPOSITORY / "src/enterprise_rag").rglob("*.py")),
    ]
    api_pids: list[int] = []
    checks: dict[str, bool] = {}
    http_transcript: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    report = {
        **manifest,
        "status": "running",
        "cleanup_confirmed": False,
        "api_pids": api_pids,
        "checks": checks,
        "http_transcript": http_transcript,
        "source_sha256": source_hashes,
    }
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    processes: list[OwnedChild] = []
    phase = "source snapshot"
    base_env = _environment(root)

    def expired():
        _check(time.monotonic() < deadline, f"Smoke deadline exceeded during {phase}")

    def request(client, method, path, role="admin", expected=200, **kwargs):
        expired()
        headers = kwargs.pop("headers", {})
        if role is not None:
            token = (
                bindings[role]["token"]
                if role in bindings
                else "invalid-disposable-token"
            )
            headers["Authorization"] = f"Bearer {token}"
        response = client.request(method, path, headers=headers, **kwargs)
        body = response.json()
        http_transcript.append(
            {
                "method": method,
                "path": path,
                "role": role,
                "status": response.status_code,
                "body": body,
            }
        )
        _check(
            response.status_code == expected,
            f"Unexpected HTTP status for {method} {path}: {response.status_code}",
        )
        return body

    def launch(stack, role, number=0):
        log = stack.enter_context((root / f"{role}-{number}.log").open("wb"))
        child = OwnedChild(
            (
                sys.executable,
                "-B",
                str(ENTRYPOINT),
                "--_role",
                role,
                "--_root",
                str(root),
            ),
            environment=api_env if role == "api" else base_env,
            output=log,
        )
        processes.append(child)
        return stack.enter_context(child)

    def connect(stack, api):
        while True:
            expired()
            _check(api.poll() is None, "API process exited before readiness")
            try:
                port_info = json.loads((root / "api-port.json").read_text())
                reporter_pid = port_info["pid"]
                if type(reporter_pid) is int and (
                    reporter_pid == api.pid or api.accepts_health_pid(reporter_pid)
                ):
                    client = _ready_client(f"http://127.0.0.1:{port_info['port']}")
                    if client is not None:
                        stack.callback(client.close)
                        return client
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.05)

    try:
        source_hashes = _source_hashes(source_paths)
        report["source_sha256"] = source_hashes
        phase = "bootstrap"
        identities, bindings = _bootstrap(root, run_id)
        api_env = {**base_env, AUTH_ENV: json.dumps(bindings)}
        with ExitStack() as stack:
            phase = "HTTP startup and upload"
            api_scope = stack.enter_context(ExitStack())
            api = launch(api_scope, "api")
            api_pids.append(api.pid)
            client = connect(api_scope, api)
            query = f"smoke{run_id}"
            search_path = f"/v1/search?q={query}"
            request(client, "GET", search_path, role=None, expected=401)
            request(client, "GET", search_path, role="invalid", expected=401)
            checks["unauthenticated_denied"] = True
            upload = request(
                client,
                "POST",
                "/v1/documents",
                expected=202,
                content=content,
                headers={
                    "X-Canonical-Key": f"smoke/{run_id}.txt",
                    "X-Title": "Synthetic smoke",
                    "X-Filename": "smoke.txt",
                    "Content-Type": "text/plain",
                },
            )
            document_id, job_id = upload["document_id"], upload["job_id"]
            _check(bool(job_id), "Upload did not return a durable job")
            permissions = f"/v1/admin/documents/{document_id}/permissions"

            def grant(who):
                return request(
                    client,
                    "POST",
                    permissions,
                    json={
                        "principal_type": "user",
                        "principal_id": identities[who].id,
                        "permission": "read",
                    },
                )

            grant("admin")
            if restart_supervisor:
                grant("reader")
                request(
                    client,
                    "DELETE",
                    f"{permissions}/user/{identities['reader'].id}/read",
                )
                checks["pre_crash_revoke"] = True
            phase = "supervisor crash and reclaim"
            process_role = "watchdog" if restart_supervisor else "supervisor"
            supervisor = launch(stack, process_role)
            while True:
                expired()
                _check(
                    supervisor.poll() is None,
                    "Supervisor exited before ingestion completed",
                )
                job = request(client, "GET", f"/v1/jobs/{job_id}")
                if job["status"] == "succeeded":
                    break
                _check(
                    job["status"] not in {"failed", "cancelled"},
                    "Ingestion became terminal without success",
                )
                time.sleep(0.1)
            _check(job["attempts"] == 2, "Job was not reclaimed exactly once")
            crash = json.loads((root / "injected-crash.json").read_text())
            _check(
                crash["job_id"] == job_id and crash["run_id"] == job["run_id"],
                "Crash did not belong to this run",
            )
            report["job"], report["crash"] = job, crash
            hit = request(client, "GET", search_path)["items"]
            _check(
                len(hit) == 1 and hit[0]["text"] == text,
                "Admin exact evidence mismatch",
            )
            hit = hit[0]
            _check(
                hit["document_id"] == document_id
                and hit["document_version_id"] == upload["version_id"]
                and hit["run_id"] == job["run_id"],
                "Evidence source identity mismatch",
            )
            evidence_path = f"/v1/evidence/{hit['chunk_id']}"

            def denied(who):
                _check(
                    request(client, "GET", search_path, role=who)["items"] == [],
                    "Denied actor received search evidence",
                )
                request(client, "GET", evidence_path, role=who, expected=404)
                request(client, "GET", f"/v1/jobs/{job_id}", role=who, expected=404)

            phase = "live ACL grant and revoke"
            for who in ("reader", "outsider", "foreign"):
                denied(who)
            checks["cross_tenant_denied"] = True
            request(
                client,
                "POST",
                permissions,
                role="reader",
                expected=403,
                json={
                    "principal_type": "user",
                    "principal_id": identities["reader"].id,
                    "permission": "read",
                },
            )
            grant("reader")

            def reader_evidence():
                found = request(client, "GET", search_path, role="reader")["items"]
                _check(
                    len(found) == 1
                    and found[0]["chunk_id"] == hit["chunk_id"]
                    and found[0]["text"] == text,
                    "Granted reader evidence mismatch",
                )
                evidence = request(client, "GET", evidence_path, role="reader")
                for key in (
                    "chunk_id",
                    "document_id",
                    "document_version_id",
                    "run_id",
                    "text",
                    "page_numbers",
                ):
                    _check(
                        evidence[key] == hit[key], "Evidence endpoint source mismatch"
                    )

            reader_evidence()
            request(
                client, "DELETE", f"{permissions}/user/{identities['reader'].id}/read"
            )
            denied("reader")
            checks["live_revoke_denied"] = True
            phase = "fresh API process with persisted revocation"
            api_scope.close()
            api_scope = stack.enter_context(ExitStack())
            api = launch(api_scope, "api", 1)
            api_pids.append(api.pid)
            client = connect(api_scope, api)
            denied("reader")
            denied("foreign")
            checks["revoked_after_api_restart"] = True
            grant("reader")
            reader_evidence()
            checks["regrant_same_evidence"] = True
            repair = request(
                client, "POST", "/v1/admin/ingestion/stage-attempts/reconcile"
            )
            _check(
                len(repair["repaired_attempt_ids"]) == 1,
                "Abandoned crash-stage repair mismatch",
            )
            phase = "supervisor stop and owned tree cleanup"
            (root / "stop-supervisor").touch(exist_ok=False)
            while supervisor.poll() is None:
                expired()
                time.sleep(0.05)
            _check(supervisor.poll() == 0, "Supervisor failed during requested stop")
            result = json.loads((root / "supervisor-result.json").read_text())
            _check(
                result["restarts"] == (0 if restart_supervisor else 1)
                and result["failures"] == (0 if restart_supervisor else 1)
                and result["stop_reason"] == "stopped",
                "Supervisor restart evidence mismatch",
            )
            report["supervisor"] = result
            if restart_supervisor:
                watchdog_result = json.loads(
                    (root / "watchdog-result.json").read_text()
                )
                _check(
                    watchdog_result["starts"] == 2
                    and watchdog_result["restarts"] == 1
                    and watchdog_result["failures"] == 1
                    and watchdog_result["stop_reason"] == "completed",
                    "Watchdog replacement result mismatch",
                )
                report["watchdog"] = watchdog_result
            events = []
            for line in (
                (root / f"{process_role}-0.log")
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
            failures = [
                e
                for e in events
                if e.get("event")
                == (
                    "supervisor_watchdog_failure"
                    if restart_supervisor
                    else "worker_supervisor_failure"
                )
            ]
            _check(
                len(failures) == 1
                and failures[0].get("exit_code")
                == (None if hang_supervisor else 29 if restart_supervisor else 23),
                "Observed child exit did not match injected crash",
            )
            if hang_supervisor:
                _check(
                    failures[0].get("reason") == "heartbeat_timeout"
                    and failures[0].get("responsive_before_failure") is True
                    and any(
                        e.get("event") == "supervisor_watchdog_responsive"
                        and e.get("pid") == failures[0].get("pid")
                        and e.get("reporter_pid") == crash["supervisor_pid"]
                        for e in events
                    ),
                    "Missing responsive-then-alive-but-hung timeout evidence",
                )
                checks["responsive_then_hung_timeout"] = True
            reclaimed = [
                e
                for e in events
                if e.get("job_id") == job_id
                and e.get("completed") is True
                and e.get("reclaimed") is True
            ]
            _check(len(reclaimed) == 1, "No actual reclaimed-worker success observed")
            report["supervisor_events"] = events
            if restart_supervisor:
                replacements = [
                    e for e in events if e.get("event") == "supervisor_watchdog_started"
                ]
                responsive = [
                    e
                    for e in events
                    if e.get("event") == "supervisor_watchdog_responsive"
                ]
                _check(
                    len(replacements) == 2
                    and any(
                        e["pid"] == replacements[0]["pid"]
                        and e["reporter_pid"] == crash["supervisor_pid"]
                        for e in responsive
                    )
                    and replacements[0]["pid"] != replacements[1]["pid"]
                    and replacements[1].get("old_worker_terminated") is True,
                    "Old worker/replacement ownership evidence missing",
                )
                checks["old_tree_gone_before_replacement"] = True
            if not hang_supervisor:
                checks["crash_exit_observed"] = True
            checks["reclaimed_worker_observed"] = True
            reader_evidence()
        _check(
            all(child.termination_confirmed for child in processes),
            "Owned process termination unconfirmed",
        )
        phase = "source stability verification"
        _verify_sources(source_paths, source_hashes)
        checks["source_unchanged"] = True
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        (root / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        report["cleanup_confirmed"] = all(
            child.termination_confirmed for child in processes
        )
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_at"] = datetime.now(UTC).isoformat()
        # Never serialize role bindings or request Authorization headers.
        _save(root / "report.json", report)
    return root / "report.json", report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY / ".local-artifacts/smoke-runs"
    )
    parser.add_argument("--timeout-seconds", type=float, default=60)
    faults = parser.add_mutually_exclusive_group()
    faults.add_argument(
        "--restart-supervisor",
        action="store_true",
        help="Crash a supervisor during ingestion and require watchdog recovery (Windows)",
    )
    faults.add_argument(
        "--hang-supervisor",
        action="store_true",
        help="Hang a live supervisor and require heartbeat timeout/recovery (Windows)",
    )
    parser.add_argument(
        "--_role",
        choices=("api", "supervisor", "worker", "watchdog"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._role:
        _check(args._root is not None, "Missing isolated run directory")
        return _child(args._root.resolve(), args._role)
    try:
        path, report = run_smoke(
            args.output_dir,
            args.timeout_seconds,
            restart_supervisor=args.restart_supervisor,
            hang_supervisor=args.hang_supervisor,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(path),
                "cleanup_confirmed": report["cleanup_confirmed"],
            }
        )
    )
    return 0 if report["status"] == "passed" else 1
