"""Prepared, non-elevated Task Scheduler action for the Windows lite watchdog."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

from enterprise_rag.bootstrap import build_services
from enterprise_rag.cli import _build_parser, _watchdog_preflight, _worker_watchdog
from enterprise_rag.settings import Settings
from enterprise_rag.workers.lite import validate_worker_settings
from enterprise_rag.workers.processes import OwnedChild

NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
FIELDS = {
    "version",
    "kind",
    "config_id",
    "task_name",
    "sid",
    "repository",
    "pythonw",
    "data_dir",
    "artifacts_dir",
    "state_dir",
    "watchdog_args",
}
EVENT_LIMIT = 1024 * 1024


class DiagnosticFailure(OSError):
    def __init__(self, cause, outcome):
        super().__init__(str(cause))
        self.outcome = outcome


def _save(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) not in (FIELDS, FIELDS | {"diagnostic_hold"})
        or type(config["version"]) is not int
        or config["version"] not in {1, 2}
    ):
        raise ValueError("Invalid scheduled-watchdog configuration schema")
    if not all(
        isinstance(config[key], str) for key in FIELDS - {"version", "watchdog_args"}
    ):
        raise ValueError("Scheduled configuration fields must be strings")
    if config["kind"] not in {"operational", "synthetic"}:
        raise ValueError("Invalid task kind")
    if "diagnostic_hold" in config and (
        config["diagnostic_hold"] is not True
        or config["version"] != 2
        or config["kind"] != "synthetic"
    ):
        raise ValueError("Diagnostic hold requires version2 isolated synthetic config")
    if not re.fullmatch(r"S-1-5-(?:\d+-)*\d+", config["sid"]):
        raise ValueError("Invalid principal SID")
    if not re.fullmatch(
        r"EnterpriseRAG-Lite-Watchdog-[A-Za-z0-9-]+", config["task_name"]
    ):
        raise ValueError("Invalid task name")
    if not re.fullmatch(r"[a-f0-9]{32}", config["config_id"]):
        raise ValueError("Invalid configuration id")
    for key in ("repository", "pythonw", "data_dir", "artifacts_dir", "state_dir"):
        value = config[key]
        if (
            not isinstance(value, str)
            or any(c in value for c in "\0\r\n")
            or not Path(value).is_absolute()
        ):
            raise ValueError(f"Invalid absolute {key}")
        if Path(value).resolve() != Path(value):
            raise ValueError(f"Require canonical {key}, not an alias")
    base = path.resolve().parent
    if Path(config["state_dir"]) != base / "state":
        raise ValueError("State must be isolated beside the configuration")
    if config["kind"] == "synthetic" and (
        Path(config["data_dir"]) != base / "runtime/data"
        or Path(config["artifacts_dir"]) != base / "runtime/artifacts"
    ):
        raise ValueError("Synthetic verification requires exact isolated runtime paths")
    if not isinstance(config["watchdog_args"], list) or not all(
        isinstance(s, str) for s in config["watchdog_args"]
    ):
        raise ValueError("Invalid watchdog arguments")
    try:
        args = _build_parser().parse_args(["worker-watchdog", *config["watchdog_args"]])
    except SystemExit as exc:
        raise ValueError("Invalid watchdog arguments") from exc
    _watchdog_preflight(args)
    validate_worker_settings(
        args.worker_id,
        args.lease_seconds,
        3,
        1,
        args.heartbeat_interval_seconds,
        args.heartbeat_max_seconds,
    )
    return config


def prepare(
    output: Path,
    *,
    repository: Path,
    pythonw: Path,
    sid: str,
    verification: bool = False,
    max_runs: int | None = None,
    periodic: bool = False,
    diagnostic_hold: bool = False,
) -> Path:
    if diagnostic_hold and not (verification and periodic):
        raise ValueError("Diagnostic hold requires periodic synthetic preparation")
    repository, output = repository.resolve(), output.resolve()
    config_id = uuid4().hex
    suffix = hashlib.sha256(str(repository).casefold().encode()).hexdigest()[:10]
    name = f"EnterpriseRAG-Lite-Watchdog-{suffix}"
    if verification:
        name += f"-Verify-{config_id[:10]}"
    data = output / "runtime/data" if verification else repository / ".local-data"
    artifacts = (
        output / "runtime/artifacts"
        if verification
        else repository / ".local-artifacts"
    )
    args = ["--worker-id", f"scheduled-{config_id[:10]}"]
    if max_runs is not None:
        args += ["--max-runs", str(max_runs)]
    config = {
        "version": 2 if periodic else 1,
        "kind": "synthetic" if verification else "operational",
        "config_id": config_id,
        "task_name": name,
        "sid": sid,
        "repository": str(repository),
        "pythonw": str(pythonw.resolve()),
        "data_dir": str(data),
        "artifacts_dir": str(artifacts),
        "state_dir": str(output / "state"),
        "watchdog_args": args,
    }
    if diagnostic_hold:
        config["diagnostic_hold"] = True
    output.mkdir(parents=True, exist_ok=False)
    path = output / "config.json"
    _save(path, config)
    load_config(path)
    (output / "task.xml").write_text(build_task_xml(path, config), encoding="utf-8")
    return path


def build_task_xml(path: Path, config: dict) -> str:
    ET.register_namespace("", NS)
    task = ET.Element(f"{{{NS}}}Task", version="1.2")

    def node(parent, name, value=None, **attributes):
        element = ET.SubElement(parent, f"{{{NS}}}{name}", attributes)
        if value is not None:
            element.text = value
        return element

    registration = node(task, "RegistrationInfo")
    node(
        registration,
        "Description",
        f"Enterprise RAG lite watchdog {config['config_id']}; current user only",
    )
    triggers = node(task, "Triggers")
    trigger = node(triggers, "LogonTrigger")
    if config["version"] == 2:
        repeat = node(trigger, "Repetition")
        node(repeat, "Interval", "PT1M")
        node(repeat, "StopAtDurationEnd", "false")
    node(trigger, "Enabled", "true")
    node(trigger, "UserId", config["sid"])
    principal = node(node(task, "Principals"), "Principal", id="CurrentUser")
    node(principal, "UserId", config["sid"])
    node(principal, "LogonType", "InteractiveToken")
    node(principal, "RunLevel", "LeastPrivilege")
    settings = node(task, "Settings")
    for key, value in {
        "MultipleInstancesPolicy": "IgnoreNew",
        "DisallowStartIfOnBatteries": "false",
        "StopIfGoingOnBatteries": "false",
        "AllowHardTerminate": "true",
        "StartWhenAvailable": "false",
        "RunOnlyIfNetworkAvailable": "false",
        "AllowStartOnDemand": "true",
        "Enabled": "true",
        "Hidden": "false",
        "RunOnlyIfIdle": "false",
        "WakeToRun": "false",
        "ExecutionTimeLimit": "PT0S",
        "Priority": "7",
    }.items():
        node(settings, key, value)
    if config["version"] == 1:
        retry = node(settings, "RestartOnFailure")
        node(retry, "Interval", "PT1M")
        node(retry, "Count", "3")
    action = node(node(task, "Actions", Context="CurrentUser"), "Exec")
    node(action, "Command", config["pythonw"])
    node(
        action,
        "Arguments",
        subprocess.list2cmdline(
            [
                "-B",
                str(Path(config["repository"]) / "scripts/run_scheduled_watchdog.py"),
                "--config",
                str(path.resolve()),
            ]
        ),
    )
    node(action, "WorkingDirectory", config["repository"])
    return ET.tostring(task, encoding="unicode", xml_declaration=False)


def scheduler_exit(code: int) -> int:
    # Otherwise Scheduler would reset deliberately exhausted inner budgets.
    return 0 if code in {0, 2, 3, 130} else code


def current_identity():
    result = subprocess.run(
        [
            str(Path(os.environ["SYSTEMROOT"]) / "System32/whoami.exe"),
            "/user",
            "/fo",
            "csv",
            "/nh",
        ],
        capture_output=True,
        text=True,
        check=True,
        creationflags=0x08000000,
    )
    sid = next(csv.reader(result.stdout.splitlines()))[1]
    from ctypes import wintypes as w

    api = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    api.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    api.GetTokenInformation.argtypes = [
        w.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        w.DWORD,
        ctypes.POINTER(w.DWORD),
    ]
    token, elevation, size = w.HANDLE(), w.DWORD(), w.DWORD()
    if not api.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not api.GetTokenInformation(
            token,
            20,
            ctypes.byref(elevation),
            ctypes.sizeof(elevation),
            ctypes.byref(size),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(token)
    return sid, bool(elevation.value)


@contextmanager
def single_instance(path: Path):
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(
                "Scheduled watchdog already running or lock unavailable"
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def child_environment(config):
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
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    env.update(
        {
            "PYTHONPATH": str(Path(config["repository"]) / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RAG_PROFILE": "lite",
            "RAG_OBJECT_STORE": "local",
            "RAG_LLM_PROVIDER": "none",
            "RAG_DATA_DIR": config["data_dir"],
            "RAG_ARTIFACTS_DIR": config["artifacts_dir"],
            "RAG_DATABASE_URL": f"sqlite:///{(Path(config['data_dir']) / 'enterprise_rag.sqlite3').as_posix()}",
        }
    )
    return env


def append_event(path: Path, payload: dict):
    line = json.dumps(payload) + "\n"
    if path.exists() and path.stat().st_size + len(line.encode()) > EVENT_LIMIT:
        raise OSError(
            "Lifecycle event limit reached; preserve/archive before restarting"
        )
    with path.open("a", encoding="utf-8") as output:
        output.write(line)


def write_diagnostic(path: Path, payload: dict):
    pending = path.with_suffix(".pending")
    _save(pending, payload)
    pending.replace(path)


def run(path: Path) -> int:
    config = load_config(path)
    sid, elevated = current_identity()
    if sid != config["sid"] or elevated:
        raise ValueError(
            "Scheduled watchdog requires its exact non-elevated current user"
        )
    if config["version"] == 2:
        from enterprise_rag.operations.periodic import run_periodic

        return run_periodic(config, path)
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    outcome = {"actual_exit_code": 2, "result": None}

    def emit(event):
        payload = {
            **event,
            "config_id": config["config_id"],
            "run_id": run_id,
            "root_pid": os.getpid(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        append_event(state / "events.jsonl", payload)

    env = child_environment(config)
    settings = Settings.from_env(env)
    canonical_db = str((Path(config["data_dir"]) / "enterprise_rag.sqlite3").resolve())
    lock_id = hashlib.sha256(canonical_db.casefold().encode()).hexdigest()[:16]
    try:
        with single_instance(Path(config["data_dir"]) / f".scheduled-{lock_id}.lock"):
            emit(
                {
                    "event": "scheduled_watchdog_root_started",
                    "sid": sid,
                    "elevated": elevated,
                }
            )
            try:
                services = build_services(settings)
            except Exception as exc:
                emit(
                    {
                        "event": "scheduled_watchdog_bootstrap_failed",
                        "type": type(exc).__name__,
                        "actual_exit_code": 2,
                        "scheduler_exit_code": 0,
                    }
                )
                return scheduler_exit(2)
            args = _build_parser().parse_args(
                ["worker-watchdog", *config["watchdog_args"]]
            )
            stop = threading.Event()
            with open(os.devnull, "wb") as output:
                result = _worker_watchdog(
                    services,
                    args,
                    stop_requested=stop.is_set,
                    wait=stop.wait,
                    emit=emit,
                    child_factory=partial(OwnedChild, environment=env, output=output),
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
            outcome.update(actual_exit_code=code, result=result)
            emit(
                {
                    "event": "scheduled_watchdog_root_finished",
                    "result": result,
                    "actual_exit_code": code,
                    "scheduler_exit_code": scheduler_exit(code),
                }
            )
            return scheduler_exit(code)
    except (ValueError, OSError) as exc:
        payload = {
            "event": "scheduled_watchdog_diagnostic_failure",
            "diagnostic_failure": True,
            "type": type(exc).__name__,
            "message": str(exc),
            **outcome,
            "config_id": config["config_id"],
            "root_pid": os.getpid(),
            "scheduler_exit_code": 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            write_diagnostic(state / "latest-failure.json", payload)
        except OSError as failure:
            raise DiagnosticFailure(failure, outcome) from exc
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--verification", action="store_true")
    prep.add_argument("--max-runs", type=int)
    prep.add_argument("--periodic", action="store_true")
    prep.add_argument("--diagnostic-hold", action="store_true")
    action = sub.add_parser("run")
    action.add_argument("--config", required=True, type=Path)
    check = sub.add_parser("check")
    check.add_argument("--config", required=True, type=Path)
    for mode in ("stop", "status", "quiescent"):
        command = sub.add_parser(mode)
        command.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.name != "nt":
        raise ValueError("Task Scheduler lifecycle requires Windows")
    if args.mode == "check":
        print(build_task_xml(args.config, load_config(args.config)))
        return 0
    if args.mode in {"stop", "status", "quiescent"}:
        from enterprise_rag.operations.periodic import Control

        config = load_config(args.config)
        if config["version"] != 2 or current_identity() != (config["sid"], False):
            raise ValueError("Periodic controls require exact non-elevated user")
        control = Control(config)
        print(
            json.dumps(
                control.stop()
                if args.mode == "stop"
                else control.quiescent()
                if args.mode == "quiescent"
                else control.read()
            )
        )
        return 0
    if args.mode == "prepare":
        sid, elevated = current_identity()
        if elevated:
            raise ValueError("Prepare from a non-elevated current-user process")
        repository = Path(__file__).resolve().parents[3]
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if not pythonw.is_file():
            raise ValueError("Installed pythonw.exe not found")
        result = prepare(
            args.output,
            repository=repository,
            pythonw=pythonw,
            sid=sid,
            verification=args.verification,
            max_runs=args.max_runs,
            periodic=args.periodic,
            diagnostic_hold=args.diagnostic_hold,
        )
        print(
            json.dumps(
                {"config": str(result), "task_xml": str(result.parent / "task.xml")}
            )
        )
        return 0
    try:
        return run(args.config)
    except (ValueError, OSError) as exc:
        try:
            write_diagnostic(
                args.config.resolve().parent / "launcher-failure.json",
                {
                    "event": "scheduled_watchdog_launch_failure",
                    "diagnostic_failure": True,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "actual_exit_code": 2,
                    **getattr(exc, "outcome", {}),
                    "scheduler_exit_code": 0,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return 0  # Invalid preparation is terminal, never an inner-budget reset.
        except OSError:
            return (
                74  # No diagnostic sink works: expose I/O failure, never fake success.
            )


if __name__ == "__main__":
    raise SystemExit(main())
