"""Actual scheduled root crash with a live job, fenced reclaim and HTTP ACL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import uvicorn

from enterprise_rag.api import StaticTokenAuthenticator, create_app
from enterprise_rag.bootstrap import build_services
from enterprise_rag.cli import _build_parser
from enterprise_rag.diagnostics._evidence import require_evidence
from enterprise_rag.diagnostics.local_smoke import (
    AUTH_ENV,
    REPOSITORY,
    _bootstrap,
    _check,
    _ready_client,
    _save,
    _source_hashes,
    _verify_sources,
)
from enterprise_rag.diagnostics.scheduled_hold import signal
from enterprise_rag.domain.ingestion import JobLease
from enterprise_rag.operations.periodic import Control
from enterprise_rag.operations.scheduled import (
    child_environment,
    current_identity,
    load_config,
    prepare,
)
from enterprise_rag.persistence import SQLiteMetadataStore
from enterprise_rag.security import ActorContext
from enterprise_rag.settings import Settings
from enterprise_rag.workers.processes import START_GATE_ENV, OwnedChild

ENTRYPOINT = REPOSITORY / "scripts/smoke_lite_scheduled.py"
TASK_SCRIPT = REPOSITORY / "scripts/windows_watchdog_task.ps1"


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _configuration(path):
    config = load_config(path)
    _check(
        config.get("diagnostic_hold") is True,
        "Not an explicitly isolated scheduled smoke",
    )
    return config


def smoke_sources():
    return [
        ENTRYPOINT,
        TASK_SCRIPT,
        REPOSITORY / "scripts/run_scheduled_watchdog.py",
        REPOSITORY / "scripts/prepare_windows_watchdog.py",
        *sorted((REPOSITORY / "src/enterprise_rag").rglob("*.py")),
    ]


def expected_commands(path):
    config = _configuration(path)
    args = _build_parser().parse_args(["worker-watchdog", *config["watchdog_args"]])
    supervisor = [config["pythonw"], "-m", "enterprise_rag.cli", "worker-supervisor"]
    for name in (
        "worker_id",
        "lease_seconds",
        "heartbeat_interval_seconds",
        "heartbeat_max_seconds",
        "interval_seconds",
        "restart_backoff_seconds",
        "max_restarts",
        "max_runs",
        "job_timeout_seconds",
    ):
        value = getattr(args, name)
        if value is not None:
            supervisor.extend([f"--{name.replace('_', '-')}", str(value)])
    parsed = _build_parser().parse_args(supervisor[3:])
    worker = [config["pythonw"], "-m", "enterprise_rag.cli", "worker", "--once"]
    for name in (
        "worker_id",
        "lease_seconds",
        "heartbeat_interval_seconds",
        "heartbeat_max_seconds",
    ):
        worker.extend([f"--{name.replace('_', '-')}", str(getattr(parsed, name))])
    return {"supervisor": supervisor, "worker": worker}


def lease_guard(path):
    config = _configuration(path)
    held = _read(path.parent / "worker-held-1.json")
    scenario = _read(path.parent / "scenario.json")
    _check(held is not None and scenario is not None, "Missing held/scenario evidence")
    metadata = SQLiteMetadataStore(Path(config["data_dir"]) / "enterprise_rag.sqlite3")
    job = require_evidence(
        metadata.get_job(scenario["job_id"]), "Missing held job evidence"
    )
    now = time.time()
    _check(
        held["config_id"] == config["config_id"]
        and job.run_id == scenario["run_id"] == held["run_id"],
        "Wrong held job/run",
    )
    _check(
        job.id == held["job_id"] and job.status == "running" and job.attempts == 1,
        "Job is not active first attempt",
    )
    _check(
        job.lease_token == held["lease_token"]
        and job.lease_owner == held["lease_owner"]
        and require_evidence(job.lease_expires_at, "Missing held lease expiry") > now,
        "Expired or fenced pre-crash lease",
    )
    _check(Control(config).read()["attempts"] == 1, "Wrong root admission")
    return {**asdict(job), "checked_at": now, "worker_pid": held["worker_pid"]}


def _api(path):
    _check(sys.stdin.buffer.read(1) == b"1", "API gate not released")
    os.environ.pop(START_GATE_ENV, None)
    config = _configuration(path)
    bindings = json.loads(os.environ.pop(AUTH_ENV))
    auth = StaticTokenAuthenticator(
        {
            v["token"]: ActorContext(v["tenant_id"], v["user_id"])
            for v in bindings.values()
        }
    )
    app = create_app(build_services(Settings.from_env(child_environment(config))), auth)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        _save(
            path.parent / "api-port.json",
            {"pid": os.getpid(), "port": listener.getsockname()[1]},
        )
        uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                log_level="warning",
                access_log=False,
                timeout_graceful_shutdown=2,
            )
        ).run(sockets=[listener])
    return 0


REQUIRED_CHECKS = {
    "pre_crash_grant_revoke",
    "live_lease_at_kill",
    "no_early_takeover",
    "scheduled_same_job_reclaim",
    "live_replacement_fencing",
    "exact_evidence",
    "cross_tenant_denied",
    "live_grant_revoke",
    "persisted_revoke_new_api",
    "stale_completion_after_success",
    "abandoned_stage_reconciled",
    "source_unchanged",
    "native_cleanup",
}


def validate_report(report):
    _check(
        report.get("kind") == "synthetic" and report.get("purpose") == "contract",
        "Not contract evidence",
    )
    _check(
        report.get("status") == "passed" and report.get("cleanup_confirmed") is True,
        "Smoke not completed/cleaned",
    )
    _check(
        all(report.get("checks", {}).get(key) is True for key in REQUIRED_CHECKS),
        "Missing required behavioral checks",
    )
    scheduler = report["scheduler"]
    _check(
        scheduler["config_id"]
        == report["config_id"]
        == scheduler["workload"]["config_id"],
        "Wrong native configuration",
    )
    _check(
        scheduler["status"] == "passed"
        and scheduler["run_calls"] == 0
        and scheduler["cleanup_confirmed"]
        and scheduler["task_removed"],
        "Missing native schedule/cleanup proof",
    )
    old, recovered, final = (
        report["before_crash"],
        report["recovered_lease"],
        report["job"],
    )
    _check(
        old["id"] == recovered["id"] == final["id"] == scheduler["workload"]["job_id"],
        "Different recovered job",
    )
    _check(
        old["run_id"]
        == recovered["run_id"]
        == final["run_id"]
        == scheduler["workload"]["run_id"],
        "Different recovered run",
    )
    _check(
        old["attempts"] == 1
        and recovered["attempts"] == final["attempts"] == 2
        and recovered["lease_token"] > old["lease_token"],
        "No fenced reclaim",
    )
    _check(final["status"] == "succeeded", "Recovered job did not succeed")
    guard = scheduler["live_lease_guard"]
    killed = datetime.fromisoformat(
        scheduler["killed_at"].replace("Z", "+00:00")
    ).timestamp()
    _check(
        guard["id"] == old["id"]
        and guard["run_id"] == old["run_id"]
        and guard["lease_token"] == old["lease_token"]
        and guard["checked_at"] <= killed < guard["lease_expires_at"],
        "Native kill not guarded by this job's live lease",
    )
    confirmed = datetime.fromisoformat(
        scheduler["root_exit_confirmed_at"].replace("Z", "+00:00")
    ).timestamp()
    _check(confirmed >= killed, "Termination confirmation predates injection")
    _check(
        any(
            confirmed < sample["observed_at"] < guard["lease_expires_at"]
            and sample["attempts"] == 1
            and sample["lease_token"] == old["lease_token"]
            for sample in report["live_lease_samples"]
        ),
        "No post-kill unexpired lease observation",
    )
    _check(
        recovered["status"] == "running"
        and recovered["lease_expires_at"] > report["recovered_checked_at"],
        "Replacement fence not exercised while lease live",
    )
    _check(
        all(report["fencing"].values())
        and set(report["fencing"]) == {"renew", "finish_stage", "complete", "fail"},
        "Missing stale-lease mutation checks",
    )
    _check(
        scheduler["initial_process"]["pid"] != scheduler["replacement_process"]["pid"],
        "No root replacement",
    )


def run_smoke(output_dir: Path, timeout_seconds=300):
    _check(
        os.name == "nt" and 180 <= timeout_seconds <= 420,
        "Native smoke requires Windows and timeout180..420s",
    )
    sid, elevated = current_identity()
    _check(not elevated, "Run only as current non-elevated user")
    output_dir.mkdir(parents=True, exist_ok=True)
    container = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir)).resolve()
    path = prepare(
        container / "scenario",
        repository=REPOSITORY,
        pythonw=Path(sys.executable).with_name("pythonw.exe"),
        sid=sid,
        verification=True,
        periodic=True,
        diagnostic_hold=True,
    )
    root, config = path.parent, _configuration(path)
    run_id = uuid4().hex
    text = f"Scheduled scheduledsmoke{run_id}. Retain synthetic evidence accurately."
    content = text.encode()
    report = {
        "kind": "synthetic",
        "purpose": "contract",
        "status": "running",
        "run_id": run_id,
        "config_id": config["config_id"],
        "task": config["task_name"],
        "input_sha256": hashlib.sha256(content).hexdigest(),
        "checks": {},
        "http_transcript": [],
        "api_pids": [],
        "cleanup_confirmed": False,
    }
    sources = smoke_sources()
    initial_config_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("config.json", "task.xml")
    }
    report["configuration_sha256"] = initial_config_hashes
    report["source_sha256"] = _source_hashes(sources)
    _save(root / "manifest.json", {**report, "configuration": config})
    (root / "input.txt").write_bytes(content)
    driver = None
    processes = []
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    phase = "bootstrap"

    def expired():
        _check(time.monotonic() < deadline, f"Deadline during {phase}")

    def wait_signal(name):
        while True:
            expired()
            value = _read(root / name)
            if value is not None:
                return value
            if driver is not None and driver.poll() is not None:
                raise RuntimeError(
                    f"Native driver exited{driver.returncode} awaiting {name}"
                )
            time.sleep(0.1)

    try:
        settings = Settings.from_env(child_environment(config))
        identities, bindings = _bootstrap(root, run_id, settings=settings)
        metadata = build_services(settings).metadata
        env = {**child_environment(config), AUTH_ENV: json.dumps(bindings)}
        with ExitStack() as stack:
            driver_log = stack.enter_context((root / "scheduler-driver.log").open("wb"))

            def launch_api(number):
                scope = stack.enter_context(ExitStack())
                log = scope.enter_context((root / f"api-{number}.log").open("wb"))
                child = OwnedChild(
                    (sys.executable, "-B", str(ENTRYPOINT), "--_api", str(path)),
                    environment=env,
                    output=log,
                )
                processes.append(child)
                scope.enter_context(child)
                report["api_pids"].append(child.pid)
                while True:
                    expired()
                    _check(child.poll() is None, "API exited before readiness")
                    port = _read(root / "api-port.json")
                    if port and port["pid"] == child.pid:
                        client = _ready_client(f"http://127.0.0.1:{port['port']}")
                        if client:
                            scope.callback(client.close)
                            return scope, client
                    time.sleep(0.05)

            scope, client = launch_api(0)

            def request(method, endpoint, role="admin", expected=200, **kwargs):
                expired()
                headers = kwargs.pop("headers", {})
                if role is not None:
                    headers["Authorization"] = (
                        f"Bearer {bindings[role]['token']}"
                        if role in bindings
                        else "Bearer invalid-disposable-token"
                    )
                response = client.request(method, endpoint, headers=headers, **kwargs)
                body = response.json()
                report["http_transcript"].append(
                    {
                        "method": method,
                        "path": endpoint,
                        "role": role,
                        "status": response.status_code,
                        "body": body,
                    }
                )
                _check(
                    response.status_code == expected,
                    f"Unexpected HTTP{response.status_code} {method} {endpoint}",
                )
                return body

            query = f"/v1/search?q=scheduledsmoke{run_id}"
            request("GET", query, role=None, expected=401)
            request("GET", query, role="invalid", expected=401)
            upload = request(
                "POST",
                "/v1/documents",
                expected=202,
                content=content,
                headers={
                    "X-Canonical-Key": f"scheduled/{run_id}.txt",
                    "X-Title": "Synthetic scheduled smoke",
                    "X-Filename": "scheduled.txt",
                    "Content-Type": "text/plain",
                },
            )
            job_id, document_id = upload["job_id"], upload["document_id"]
            job_path = f"/v1/jobs/{job_id}"
            permissions = f"/v1/admin/documents/{document_id}/permissions"

            def grant(who):
                request(
                    "POST",
                    permissions,
                    json={
                        "principal_type": "user",
                        "principal_id": identities[who].id,
                        "permission": "read",
                    },
                )

            def revoke():
                request("DELETE", f"{permissions}/user/{identities['reader'].id}/read")

            grant("admin")
            grant("reader")
            request("GET", job_path, role="reader")
            revoke()
            request("GET", job_path, role="reader", expected=404)
            request("GET", job_path, role="foreign", expected=404)
            report["checks"]["pre_crash_grant_revoke"] = True
            job = require_evidence(
                metadata.get_job(job_id), "Missing uploaded job evidence"
            )
            signal(
                root / "scenario.json",
                {
                    "kind": "synthetic",
                    "purpose": "contract",
                    "config_id": config["config_id"],
                    "job_id": job_id,
                    "run_id": job.run_id,
                },
            )
            phase = "native scheduled activation and live pre-crash lease"
            driver = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(TASK_SCRIPT),
                    "-Mode",
                    "VerifyJobAndRemove",
                    "-ConfigPath",
                    str(path),
                ],
                stdout=driver_log,
                stderr=subprocess.STDOUT,
                creationflags=0x08000000,
            )
            wait_signal("fault-prepared.json")
            guard = lease_guard(path)
            old_job = require_evidence(
                metadata.get_job(job_id), "Missing pre-crash job evidence"
            )
            old_lease = JobLease(
                old_job,
                require_evidence(old_job.lease_owner, "Missing pre-crash lease owner"),
                old_job.lease_token,
                require_evidence(
                    old_job.lease_expires_at, "Missing pre-crash lease expiry"
                ),
            )
            report["before_crash"] = asdict(old_job)
            request("GET", job_path, role="reader", expected=404)
            signal(
                root / "fault-allow.json",
                {
                    "config_id": config["config_id"],
                    "job_id": job_id,
                    "lease_token": old_job.lease_token,
                },
            )
            phase = "natural lease expiry and scheduled takeover"
            samples = []
            while True:
                expired()
                latest = require_evidence(
                    metadata.get_job(job_id), "Missing job during takeover observation"
                )
                checked_at = time.time()
                if checked_at < guard["lease_expires_at"]:
                    _check(
                        latest.attempts == 1
                        and latest.lease_token == old_job.lease_token,
                        "Unexpired lease was reclaimed",
                    )
                    samples.append(
                        {
                            "observed_at": checked_at,
                            "attempts": latest.attempts,
                            "lease_token": latest.lease_token,
                        }
                    )
                held2 = _read(root / "worker-held-2.json")
                if (
                    held2 is not None
                    and _read(root / "scheduler-recovered.json") is not None
                ):
                    break
                _check(driver.poll() is None, "Scheduler driver failed before takeover")
                time.sleep(0.3)
            _check(bool(samples), "No unexpired post-fault lease observation")
            report["live_lease_samples"] = samples
            report["checks"]["no_early_takeover"] = True
            new_job = require_evidence(
                metadata.get_job(job_id), "Missing reclaimed job evidence"
            )
            _check(
                new_job.id == old_job.id
                and new_job.run_id == old_job.run_id
                and new_job.attempts == 2
                and new_job.lease_token > old_job.lease_token
                and new_job.status == "running"
                and require_evidence(
                    new_job.lease_expires_at, "Missing reclaimed lease expiry"
                )
                > time.time(),
                "Not the same live reclaimed job",
            )
            report["recovered_lease"] = asdict(new_job)
            report["recovered_checked_at"] = time.time()
            before = asdict(new_job)
            held1 = _read(root / "worker-held-1.json")
            rejected = {
                "renew": metadata.renew_job_lease(old_lease, time.time(), 30) is None,
                "finish_stage": metadata.finish_stage_attempt(
                    old_lease, held1["stage_attempt_id"], now=time.time(), success=True
                )
                is False,
                "complete": metadata.complete_job(old_lease, time.time()) is False,
                "fail": metadata.fail_job(
                    old_lease,
                    time.time(),
                    "stale_smoke_mutation",
                    retryable=False,
                    max_attempts=3,
                )
                is False,
            }
            _check(
                all(rejected.values())
                and asdict(
                    require_evidence(
                        metadata.get_job(job_id), "Missing fenced job evidence"
                    )
                )
                == before,
                "Old lease mutated new live owner",
            )
            report["fencing"] = rejected
            report["checks"]["live_replacement_fencing"] = True
            report["checks"]["scheduled_same_job_reclaim"] = True
            signal(
                root / "release-recovered.json",
                {
                    "config_id": config["config_id"],
                    "job_id": job_id,
                    "lease_token": new_job.lease_token,
                },
            )
            phase = "job success and exact authorized evidence"
            while True:
                result = request("GET", job_path)
                if result["status"] == "succeeded":
                    break
                _check(
                    result["status"] not in {"failed", "cancelled"},
                    "Recovered job failed",
                )
                time.sleep(0.1)
            final = require_evidence(
                metadata.get_job(job_id), "Missing final job evidence"
            )
            report["job"] = asdict(final)
            _check(
                final.attempts == 2 and final.lease_token == new_job.lease_token,
                "Unexpected extra ingestion attempt",
            )
            _check(
                metadata.complete_job(old_lease, time.time()) is False
                and metadata.get_job(job_id) == final,
                "Old completion accepted after success",
            )
            report["checks"]["stale_completion_after_success"] = True
            hits = request("GET", query)["items"]
            _check(
                len(hits) == 1 and hits[0]["text"] == text,
                "Missing/duplicate/incorrect active evidence",
            )
            hit = hits[0]
            _check(
                hit["document_id"] == document_id
                and hit["document_version_id"] == upload["version_id"]
                and hit["run_id"] == final.run_id,
                "Wrong evidence provenance",
            )
            evidence_path = f"/v1/evidence/{hit['chunk_id']}"
            report["evidence"] = hit
            report["checks"]["exact_evidence"] = True

            def denied(who):
                _check(
                    request("GET", query, role=who)["items"] == [],
                    "ACL leaked search evidence",
                )
                request("GET", evidence_path, role=who, expected=404)
                request("GET", job_path, role=who, expected=404)

            def reader_evidence():
                found = request("GET", query, role="reader")["items"]
                _check(
                    len(found) == 1 and found[0] == hit,
                    "Reader search changed evidence",
                )
                evidence = request("GET", evidence_path, role="reader")
                _check(
                    all(
                        evidence[key] == hit[key]
                        for key in (
                            "chunk_id",
                            "document_id",
                            "document_version_id",
                            "run_id",
                            "text",
                            "page_numbers",
                        )
                    ),
                    "Reader evidence changed provenance",
                )

            for who in ("reader", "outsider", "foreign"):
                denied(who)
            report["checks"]["cross_tenant_denied"] = True
            request(
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
            reader_evidence()
            revoke()
            denied("reader")
            report["checks"]["live_grant_revoke"] = True
            scope.close()
            scope, client = launch_api(1)
            denied("reader")
            denied("foreign")
            grant("reader")
            reader_evidence()
            report["checks"]["persisted_revoke_new_api"] = True
            repaired = request("POST", "/v1/admin/ingestion/stage-attempts/reconcile")
            _check(
                repaired["repaired_attempt_ids"] == [held1["stage_attempt_id"]],
                "Wrong abandoned-stage reconciliation",
            )
            report["checks"]["abandoned_stage_reconciled"] = True
            signal(
                root / "http-checks-complete.json",
                {
                    "config_id": config["config_id"],
                    "job_id": job_id,
                    "run_id": final.run_id,
                    "status": "passed",
                },
            )
            phase = "native Stop/post-Stop tick and cleanup"
            while driver.poll() is None:
                expired()
                time.sleep(0.1)
            report["scheduler"] = wait_signal("scheduler-recovery-report.json")
            _check(driver.returncode == 0, "Native Scheduler qualification failed")
            native_guard = report["scheduler"]["live_lease_guard"]
            _check(
                native_guard["lease_expires_at"]
                > datetime.fromisoformat(
                    report["scheduler"]["killed_at"].replace("Z", "+00:00")
                ).timestamp(),
                "Lease not live at actual kill",
            )
            report["checks"]["live_lease_at_kill"] = True
            report["checks"]["native_cleanup"] = True
        _verify_sources(sources, report["source_sha256"])
        _check(
            {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in initial_config_hashes
            }
            == initial_config_hashes,
            "Configuration changed during smoke",
        )
        _check(
            hashlib.sha256((root / "registered-task.xml").read_bytes())
            .hexdigest()
            .upper()
            == report["scheduler"]["export_sha256"],
            "Registered XML evidence changed",
        )
        report["checks"]["source_unchanged"] = True
        report["status"] = "passed"
        report["cleanup_confirmed"] = (
            all(p.termination_confirmed for p in processes)
            and report["scheduler"]["cleanup_confirmed"]
        )
        validate_report(report)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        (root / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if driver is not None and driver.poll() is None:
            if not (root / "smoke-abort.json").exists():
                signal(
                    root / "smoke-abort.json",
                    {"reason": "coordinator failed/cancelled"},
                )
            end = time.monotonic() + 40
            while driver.poll() is None and time.monotonic() < end:
                time.sleep(0.1)  # Do not kill a driver before its task-cleanup finally.
        native = _read(root / "scheduler-recovery-report.json")
        if native is not None:
            report["scheduler"] = native
        report["cleanup_confirmed"] = (
            all(p.termination_confirmed for p in processes)
            and bool(native and native["cleanup_confirmed"] and native["task_removed"])
            and driver is not None
            and driver.poll() is not None
        )
        if not report["cleanup_confirmed"]:
            report["status"] = "failed"
        report["elapsed_seconds"] = time.monotonic() - started
        _save(root / "report.json", report)
    return root / "report.json", report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--_guard", type=Path)
    modes.add_argument("--_api", type=Path)
    modes.add_argument("--_commands", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / ".local-artifacts/scheduled-smoke-runs",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    if args._guard:
        print(json.dumps(lease_guard(args._guard)))
        return 0
    if args._api:
        return _api(args._api)
    if args._commands:
        print(json.dumps(expected_commands(args._commands)))
        return 0
    path, report = run_smoke(args.output_dir, args.timeout_seconds)
    print(
        json.dumps(
            {
                "report": str(path),
                "status": report["status"],
                "cleanup_confirmed": report["cleanup_confirmed"],
            }
        )
    )
    return 0 if report["status"] == "passed" else 1
