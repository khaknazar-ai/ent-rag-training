"""Local CLI entry points sharing the same lite application services as the API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import signal
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import FrameType

from enterprise_rag.application import (
    DEFAULT_INGESTION_AUDIT_LIMIT,
    DeletionStorageError,
    MaintenanceBusy,
    MaintenanceLeaseLost,
    MaintenanceScheduler,
    ManagementConflict,
    ManagementForbidden,
    ManagementNotFound,
)
from enterprise_rag.application.uploads import UploadForbidden
from enterprise_rag.application.visual_regions import RegionUnavailable
from enterprise_rag.bootstrap import ServiceBundle, build_services
from enterprise_rag.domain.documents.models import Source, Tenant
from enterprise_rag.ingestion import PDFOCRPipeline, PopplerOCRPageRenderer
from enterprise_rag.ingestion.errors import ParserContainmentError
from enterprise_rag.ingestion.pdf_render import PDFRenderError
from enterprise_rag.ingestion.tesseract import (
    OCRProviderUnavailable,
    TesseractCLIProvider,
)
from enterprise_rag.security import ActorContext
from enterprise_rag.workers import LiteTextStageExecutor, LiteWorker
from enterprise_rag.workers.health import SupervisorPulse
from enterprise_rag.workers.heartbeat import HeartbeatShutdownError
from enterprise_rag.workers.processes import START_GATE_ENV, OwnedChild
from enterprise_rag.workers.supervisor import SupervisorConfig, WorkerSupervisor
from enterprise_rag.workers.watchdog import SupervisorWatchdog, WatchdogConfig

MAINTENANCE_STOP_EXIT_CODE = 130
WORKER_LEASE_LOST_EXIT_CODE = 3


def main(argv: Sequence[str] | None = None) -> int:
    # Consume this parent-only channel before a supervisor can spawn workers.
    context = {
        "config": os.environ.pop("RAG_PERIODIC_OUTCOME_CONFIG", None),
        "run_id": os.environ.pop("RAG_PERIODIC_OUTCOME_RUN", None),
        "admitted": False,
        "command": None,
    }
    code = _main_impl(argv, periodic_context=context)
    _publish_periodic_outcome(context, code)
    return code


def _publish_periodic_outcome(context, code):
    if (
        context is not None
        and context["config"]
        and context["admitted"]
        and context["command"] == "worker-supervisor"
    ):
        from enterprise_rag.operations.periodic import Control
        from enterprise_rag.operations.scheduled import current_identity, load_config

        config = load_config(Path(context["config"]))
        if config["version"] != 2 or current_identity() != (config["sid"], False):
            raise ValueError("Invalid periodic supervisor outcome principal")
        # Publish the terminal decision BEFORE exiting, including when the root
        # dies before observing this process's exit. A rejected gate is excluded.
        Control(config).finish(context["run_id"], f"supervisor_exit_{code}", code)


def _main_impl(argv: Sequence[str] | None = None, *, periodic_context=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        gate = os.environ.pop(START_GATE_ENV, None)
        if (
            periodic_context is not None
            and periodic_context["config"]
            and (gate != "1" or args.command != "worker-supervisor")
        ):
            raise ValueError("Periodic supervisor requires an accepted start gate")
        if gate == "1":
            if (
                args.command not in {"worker", "worker-supervisor", "worker-watchdog"}
                or sys.stdin.buffer.read(1) != b"1"
            ):
                raise ValueError("Supervised worker start gate was not released")
            if periodic_context is not None:
                periodic_context.update(admitted=True, command=args.command)
        if args.command == "worker-watchdog":
            _watchdog_preflight(args)
        try:
            services = build_services()
        except Exception as exc:
            _publish_periodic_outcome(periodic_context, 2)
            print(
                json.dumps(
                    {
                        "error": "bootstrap_failed",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            )
            return 2
        if args.command in {
            "maintenance-loop",
            "worker-loop",
            "worker-supervisor",
            "worker-watchdog",
        }:
            stop_event = Event()
            with _maintenance_signal_handlers(stop_event):
                payload = args.handler(
                    services,
                    args,
                    stop_requested=stop_event.is_set,
                    wait=stop_event.wait,
                )
        else:
            payload = args.handler(services, args)
    except ParserContainmentError:
        _publish_periodic_outcome(periodic_context, WORKER_LEASE_LOST_EXIT_CODE)
        print(json.dumps({"error": "parser_cleanup_unconfirmed", "retryable": True}))
        return WORKER_LEASE_LOST_EXIT_CODE
    except HeartbeatShutdownError:
        _publish_periodic_outcome(periodic_context, WORKER_LEASE_LOST_EXIT_CODE)
        print(json.dumps({"error": "worker_heartbeat_unconfirmed", "retryable": True}))
        return WORKER_LEASE_LOST_EXIT_CODE
    except MaintenanceBusy as exc:
        _publish_periodic_outcome(periodic_context, 3)
        print(
            json.dumps(
                {
                    "error": "maintenance_busy",
                    "retryable": True,
                    "message": str(exc),
                }
            )
        )
        return 3
    except MaintenanceLeaseLost as exc:
        _publish_periodic_outcome(periodic_context, 3)
        print(
            json.dumps(
                {
                    "error": "maintenance_lease_lost",
                    "retryable": True,
                    "message": str(exc),
                }
            )
        )
        return 3
    except (
        DeletionStorageError,
        ManagementConflict,
        ManagementForbidden,
        ManagementNotFound,
        OSError,
        UploadForbidden,
        RegionUnavailable,
        ValueError,
    ) as exc:
        _publish_periodic_outcome(periodic_context, 2)
        print(json.dumps({"error": "command_failed", "message": str(exc)}))
        return 2
    code = _command_exit_code(args.command, payload)
    _publish_periodic_outcome(periodic_context, code)
    print(json.dumps(payload, ensure_ascii=False))
    return code


def _command_exit_code(command, payload):
    if command in {"worker-watchdog", "worker-supervisor"}:
        if payload["stop_reason"] == "stopped":
            return MAINTENANCE_STOP_EXIT_CODE
        if payload["stop_reason"] != "completed":
            return 2 if payload["stop_reason"] == "configuration_error" else 3
    if command in {"worker", "worker-loop"} and payload.get("lease_lost") is True:
        return WORKER_LEASE_LOST_EXIT_CODE
    if (
        command in {"maintenance-loop", "worker-loop"}
        and payload.get("stop_reason") == "stopped"
    ):
        return MAINTENANCE_STOP_EXIT_CODE
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enterprise-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("doctor", _doctor),
        ("migrate", _migrate),
        ("worker", _worker),
        ("worker-loop", _worker_loop),
        ("worker-supervisor", _worker_supervisor),
        ("worker-watchdog", _worker_watchdog),
        ("reconcile", _reconcile),
        ("gc", _gc),
        ("reconcile-ingestion", _reconcile_ingestion),
        ("reconcile-ingestion-stages", _reconcile_ingestion_stages),
        ("maintenance-once", _maintenance_once),
        ("maintenance-loop", _maintenance_loop),
    ):
        command = subparsers.add_parser(name)
        command.set_defaults(handler=handler)
    subparsers.choices["worker"].add_argument("--once", action="store_true")
    subparsers.choices["worker"].add_argument("--worker-id", default="cli-worker")
    worker_loop = subparsers.choices["worker-loop"]
    worker_loop.add_argument("--worker-id", default="cli-worker")
    worker_loop.add_argument("--interval-seconds", type=float, default=1.0)
    worker_loop.add_argument("--max-runs", type=int, default=None)
    for worker_name in (
        "worker",
        "worker-loop",
        "worker-supervisor",
        "worker-watchdog",
    ):
        worker_command = subparsers.choices[worker_name]
        worker_command.add_argument("--lease-seconds", type=float, default=30)
        worker_command.add_argument(
            "--heartbeat-interval-seconds",
            type=float,
            default=5
            if worker_name in {"worker-supervisor", "worker-watchdog"}
            else None,
        )
        worker_command.add_argument("--heartbeat-max-seconds", type=float, default=300)
        _add_worker_ocr_arguments(worker_command)
        worker_command.add_argument(
            "--pdf-table-engine",
            choices=(
                "pypdf",
                "pdfplumber-lines",
                "pdfplumber-text",
                "pdfplumber-spans",
            ),
            default="pypdf",
        )
    for name in ("worker-supervisor", "worker-watchdog"):
        supervisor = subparsers.choices[name]
        supervisor.add_argument("--worker-id", default="supervised-worker")
        supervisor.add_argument("--interval-seconds", type=float, default=1)
        supervisor.add_argument("--restart-backoff-seconds", type=float, default=1)
        supervisor.add_argument("--max-restarts", type=int, default=3)
        supervisor.add_argument("--max-runs", type=int, default=None)
        supervisor.add_argument("--job-timeout-seconds", type=float, default=300)
    watchdog = subparsers.choices["worker-watchdog"]
    watchdog.add_argument("--max-supervisor-restarts", type=int, default=3)
    watchdog.add_argument("--supervisor-restart-backoff-seconds", type=float, default=1)
    watchdog.add_argument(
        "--supervisor-heartbeat-timeout-seconds", type=float, default=15
    )
    watchdog.add_argument(
        "--supervisor-startup-timeout-seconds", type=float, default=30
    )

    reconcile = subparsers.choices["reconcile"]
    reconcile.add_argument("document_id")
    reconcile.add_argument("--tenant-id", required=True)
    reconcile.add_argument("--user-id", required=True)
    reconcile.add_argument("--dry-run", action="store_true")

    gc = subparsers.choices["gc"]
    gc.add_argument("--tenant-id", required=True)
    gc.add_argument("--user-id", required=True)
    gc.add_argument("--dry-run", action="store_true")
    gc.add_argument(
        "--grace-seconds",
        type=float,
        default=None,
    )
    gc.add_argument("--limit", type=int, default=None)

    ingestion_reconcile = subparsers.choices["reconcile-ingestion"]
    ingestion_reconcile.add_argument("--tenant-id", required=True)
    ingestion_reconcile.add_argument("--user-id", required=True)
    ingestion_reconcile.add_argument(
        "--limit", type=int, default=DEFAULT_INGESTION_AUDIT_LIMIT
    )

    stage_reconcile = subparsers.choices["reconcile-ingestion-stages"]
    stage_reconcile.add_argument("--tenant-id", required=True)
    stage_reconcile.add_argument("--user-id", required=True)
    stage_reconcile.add_argument("--dry-run", action="store_true")
    stage_reconcile.add_argument(
        "--limit", type=int, default=DEFAULT_INGESTION_AUDIT_LIMIT
    )

    maintenance = subparsers.choices["maintenance-once"]
    maintenance.add_argument("--tenant-id", required=True)
    maintenance.add_argument("--user-id", required=True)
    maintenance.add_argument("--dry-run", action="store_true")
    maintenance.add_argument("--grace-seconds", type=float, default=None)
    maintenance.add_argument("--batch-size", type=int, default=None)

    maintenance_loop = subparsers.choices["maintenance-loop"]
    maintenance_loop.add_argument("--tenant-id", required=True)
    maintenance_loop.add_argument("--user-id", required=True)
    maintenance_loop.add_argument("--max-runs", type=int, required=True)
    maintenance_loop.add_argument("--interval-seconds", type=float, required=True)
    maintenance_loop.add_argument("--dry-run", action="store_true")
    maintenance_loop.add_argument("--grace-seconds", type=float, default=None)
    maintenance_loop.add_argument("--batch-size", type=int, default=None)

    upload = subparsers.add_parser("upload")
    upload.set_defaults(handler=_upload)
    upload.add_argument("path", type=Path)
    upload.add_argument("--tenant-id", required=True)
    upload.add_argument("--user-id", required=True)
    upload.add_argument("--canonical-key", required=True)
    upload.add_argument("--title", required=True)
    upload.add_argument("--source-name", default="cli")

    search = subparsers.add_parser("search")
    search.set_defaults(handler=_search)
    search.add_argument("query")
    search.add_argument("--tenant-id", required=True)
    search.add_argument("--user-id", required=True)
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--offset", type=int, default=0)
    for name in ("query-get", "region-create", "region-read"):
        command = subparsers.add_parser(name)
        command.set_defaults(handler=_visual_region)
        command.add_argument("--tenant-id", required=True)
        command.add_argument("--user-id", required=True)
        if name == "region-read":
            command.add_argument("region_id")
        else:
            command.add_argument("query_run_id")
        if name == "region-create":
            command.add_argument("--citation-index", type=int, required=True)
            command.add_argument("--bbox", type=float, nargs=4)
            command.add_argument("--renderer", type=Path, required=True)
            command.add_argument("--render-temp", type=Path, required=True)
    return parser


def _visual_region(services: ServiceBundle, args: argparse.Namespace) -> dict:
    from enterprise_rag.bootstrap import with_pdf_regions

    actor = _known_actor(services, args.tenant_id, args.user_id)
    if args.command == "region-create":
        services = with_pdf_regions(
            services, renderer_command=(str(args.renderer),), temp_root=args.render_temp
        )
    service = services.visual_regions
    if service is None:
        raise ValueError("Visual regions unavailable")
    if args.command == "query-get":
        return service.history_payload(actor, args.query_run_id)
    if args.command == "region-create":
        return service.create(actor, args.query_run_id, args.citation_index, args.bbox)
    metadata, png = service.read(actor, args.region_id)
    return {"region": metadata, "png_base64": base64.b64encode(png).decode("ascii")}


def _doctor(services: ServiceBundle, _args: argparse.Namespace) -> dict[str, object]:
    services.metadata.get_authorization_revision("doctor")
    return {
        "status": "ok",
        "profile": services.settings.profile,
        "database": str(services.metadata.path),
        "object_store": str(services.object_store.root),
    }


def _migrate(services: ServiceBundle, _args: argparse.Namespace) -> dict[str, object]:
    return {"status": "ok", "profile": services.settings.profile, "migrated": True}


def _add_worker_ocr_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--ocr-renderer",
        type=Path,
        default=None,
        metavar="ABSOLUTE_PATH",
        help="explicit Poppler pdftoppm executable; enables selective PDF OCR",
    )
    command.add_argument(
        "--ocr-render-temp",
        type=Path,
        default=None,
        metavar="ABSOLUTE_PATH",
        help="existing temporary directory for bounded PDF page rendering",
    )
    command.add_argument(
        "--ocr-tesseract-executable",
        type=Path,
        default=None,
        metavar="ABSOLUTE_PATH",
        help="explicit preinstalled Tesseract executable",
    )
    command.add_argument(
        "--ocr-tessdata-dir",
        type=Path,
        default=None,
        metavar="ABSOLUTE_PATH",
        help="existing directory containing selected traineddata files",
    )
    command.add_argument(
        "--ocr-temp",
        type=Path,
        default=None,
        metavar="ABSOLUTE_PATH",
        help="existing temporary directory for OCR page processing",
    )
    command.add_argument(
        "--ocr-languages",
        default=None,
        metavar="LANG[,LANG]",
        help="one or two explicit Tesseract language codes, such as eng or eng,rus",
    )
    command.add_argument(
        "--ocr-raster-config",
        type=Path,
        default=None,
        help="Explicit bounded OpenCV long-rule removal before OCR",
    )

    command.add_argument(
        "--ocr-psm",
        type=int,
        choices=(3, 6, 11),
        default=None,
        help="Explicit Tesseract page segmentation: automatic3, block6, sparse11",
    )


def _validate_worker_ocr_arguments(args: argparse.Namespace) -> bool:
    names = (
        "ocr_renderer",
        "ocr_render_temp",
        "ocr_tesseract_executable",
        "ocr_tessdata_dir",
        "ocr_temp",
        "ocr_languages",
    )
    values = tuple(getattr(args, name, None) for name in names)
    raster_config = getattr(args, "ocr_raster_config", None)
    if getattr(args, "ocr_psm", None) is not None and not all(
        value is not None for value in values
    ):
        raise ValueError("OCR segmentation requires complete OCR options")
    if raster_config is not None and not all(value is not None for value in values):
        raise ValueError("Raster OCR preprocessing requires complete OCR options")
    if not any(value is not None for value in values):
        return False
    if not all(value is not None for value in values):
        raise ValueError("Selective OCR requires all renderer and Tesseract options")

    files = (args.ocr_renderer, args.ocr_tesseract_executable)
    directories = (args.ocr_render_temp, args.ocr_tessdata_dir, args.ocr_temp)
    if any(
        not isinstance(path, Path) or not path.is_absolute() or not path.is_file()
        for path in files
    ):
        raise ValueError(
            "OCR renderer and Tesseract executable must be existing absolute files"
        )
    if any(
        not isinstance(path, Path) or not path.is_absolute() or not path.is_dir()
        for path in directories
    ):
        raise ValueError(
            "OCR data and temporary roots must be existing absolute directories"
        )
    languages = args.ocr_languages.split(",")
    if (
        not 1 <= len(languages) <= 2
        or len(set(languages)) != len(languages)
        or any(
            not language
            or not language.isascii()
            or not language.replace("_", "").isalnum()
            for language in languages
        )
    ):
        raise ValueError("Select one or two explicit ASCII OCR language codes")
    return True


def _ocr_pipeline_from_args(args: argparse.Namespace) -> PDFOCRPipeline | None:
    if not _validate_worker_ocr_arguments(args):
        return None
    try:
        preprocessor = None
        raster_config = getattr(args, "ocr_raster_config", None)
        if raster_config is not None:
            from enterprise_rag.models.raster_geometry import RasterOCRPreprocessor

            if (
                not raster_config.is_absolute()
                or not raster_config.is_file()
                or raster_config.stat().st_size > 8192
            ):
                raise ValueError(
                    "Raster OCR config must be an existing absolute JSON path"
                )
            values = json.loads(raster_config.read_text(encoding="utf-8"))
            if (
                not isinstance(values, dict)
                or set(values) != {"python", "packages", "temp"}
                or any(not isinstance(v, str) for v in values.values())
            ):
                raise ValueError("Invalid raster OCR config")
            preprocessor = RasterOCRPreprocessor(
                *(Path(values[k]) for k in ("python", "packages", "temp"))
            )
        renderer = PopplerOCRPageRenderer(
            (str(args.ocr_renderer),), args.ocr_render_temp
        )
        provider = TesseractCLIProvider(
            args.ocr_tesseract_executable,
            tessdata_directory=args.ocr_tessdata_dir,
            languages=tuple(args.ocr_languages.split(",")),
            temp_root=args.ocr_temp,
            image_preprocessor=preprocessor,
            page_segmentation_mode=getattr(args, "ocr_psm", None) or 3,
        )
        return PDFOCRPipeline(renderer, provider)
    except (OCRProviderUnavailable, PDFRenderError, ValueError) as exc:
        raise ValueError("Configured OCR runtime or models are unavailable") from exc


def _text_executor(services: ServiceBundle, args: argparse.Namespace):
    return LiteTextStageExecutor(
        services.metadata,
        services.object_store,
        ocr_pipeline=_ocr_pipeline_from_args(args),
        pdf_table_engine=getattr(args, "pdf_table_engine", "pypdf"),
    )


def _append_worker_ocr_arguments(command: list[str], args: argparse.Namespace) -> None:
    options = (
        ("pdf_table_engine", "--pdf-table-engine"),
        ("ocr_renderer", "--ocr-renderer"),
        ("ocr_render_temp", "--ocr-render-temp"),
        ("ocr_tesseract_executable", "--ocr-tesseract-executable"),
        ("ocr_tessdata_dir", "--ocr-tessdata-dir"),
        ("ocr_temp", "--ocr-temp"),
        ("ocr_languages", "--ocr-languages"),
        ("ocr_raster_config", "--ocr-raster-config"),
        ("ocr_psm", "--ocr-psm"),
    )
    for name, option in options:
        value = getattr(args, name, None)
        if value is not None:
            command.extend((option, str(value)))


def _worker(services: ServiceBundle, args: argparse.Namespace) -> dict[str, object]:
    if not args.once:
        raise ValueError("Only bounded worker --once is available in the lite CLI")
    result = _configured_worker(services, args).run_once(_text_executor(services, args))
    return {
        "claimed": result.claimed,
        "reclaimed": result.reclaimed,
        "job_id": result.job_id,
        "completed": result.completed,
        "failure_recorded": result.failure_recorded,
        "lease_lost": result.lease_lost,
    }


def _worker_loop(
    services: ServiceBundle,
    args: argparse.Namespace,
    *,
    stop_requested,
    wait,
) -> dict[str, object]:
    """Poll in the foreground; do not conceal lease loss or worker exceptions."""

    interval = args.interval_seconds
    if not math.isfinite(interval) or not 0 < interval <= 86400:
        raise ValueError("Worker poll interval must be finite and in (0, 86400]")
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("Worker max-runs must be at least 1")
    worker = _configured_worker(services, args)
    executor = _text_executor(services, args)
    polls = 0
    handled_jobs = 0
    while not stop_requested():
        result = worker.run_once(executor)
        polls += 1
        if result.claimed:
            handled_jobs += 1
            print(
                json.dumps(
                    {
                        "event": "worker_result",
                        "job_id": result.job_id,
                        "completed": result.completed,
                        "failure_recorded": result.failure_recorded,
                        "lease_lost": result.lease_lost,
                    }
                ),
                flush=True,
            )
        if result.lease_lost:
            return {
                "polls": polls,
                "handled_jobs": handled_jobs,
                "stop_reason": "lease_lost",
                "lease_lost": True,
            }
        if args.max_runs is not None and polls >= args.max_runs:
            return {
                "polls": polls,
                "handled_jobs": handled_jobs,
                "stop_reason": "completed",
                "lease_lost": False,
            }
        if not result.claimed and wait(interval):
            break
    return {
        "polls": polls,
        "handled_jobs": handled_jobs,
        "stop_reason": "stopped",
        "lease_lost": False,
    }


def _configured_worker(services: ServiceBundle, args: argparse.Namespace) -> LiteWorker:
    metadata = services.metadata
    diagnostic = os.environ.get("RAG_SYNTHETIC_HOLD_CONFIG")
    if diagnostic:
        from enterprise_rag.diagnostics.scheduled_hold import isolated_hold_metadata

        metadata = isolated_hold_metadata(services, Path(diagnostic))
    return LiteWorker(
        metadata,
        args.worker_id,
        lease_seconds=args.lease_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        heartbeat_max_seconds=args.heartbeat_max_seconds,
    )


def _supervisor_config(args):
    return SupervisorConfig(
        poll_seconds=args.interval_seconds,
        restart_backoff_seconds=args.restart_backoff_seconds,
        job_timeout_seconds=args.job_timeout_seconds,
        max_restarts=args.max_restarts,
        max_runs=args.max_runs,
    )


def _worker_supervisor(services, args, *, stop_requested, wait):
    config = _supervisor_config(args)
    _configured_worker(services, args)
    _validate_worker_ocr_arguments(args)
    command = [
        sys.executable,
        "-m",
        "enterprise_rag.cli",
        "worker",
        "--once",
        "--worker-id",
        args.worker_id,
        "--lease-seconds",
        str(args.lease_seconds),
        "--heartbeat-interval-seconds",
        str(args.heartbeat_interval_seconds),
        "--heartbeat-max-seconds",
        str(args.heartbeat_max_seconds),
    ]
    _append_worker_ocr_arguments(command, args)
    with SupervisorPulse.from_environment() as health:
        return WorkerSupervisor(
            command,
            config,
            emit=lambda event: print(json.dumps(event), flush=True),
            pulse=health.tick,
        ).run(stop_requested=stop_requested, wait=wait)


def _watchdog_preflight(args):
    if os.name != "nt":
        raise ValueError("Supervisor watchdog requires Windows Job Object containment")
    _supervisor_config(args)
    return WatchdogConfig(
        max_restarts=args.max_supervisor_restarts,
        restart_backoff_seconds=args.supervisor_restart_backoff_seconds,
        heartbeat_timeout_seconds=args.supervisor_heartbeat_timeout_seconds,
        startup_timeout_seconds=args.supervisor_startup_timeout_seconds,
    )


def _worker_watchdog(
    services, args, *, stop_requested, wait, emit=None, child_factory=OwnedChild
):
    config = _watchdog_preflight(args)
    _configured_worker(services, args)
    _validate_worker_ocr_arguments(args)
    command = [sys.executable, "-m", "enterprise_rag.cli", "worker-supervisor"]
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
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    _append_worker_ocr_arguments(command, args)
    return SupervisorWatchdog(
        command,
        config,
        emit=emit or (lambda event: print(json.dumps(event), flush=True)),
        child_factory=child_factory,
    ).run(stop_requested=stop_requested, wait=wait)


def _reconcile(services: ServiceBundle, args: argparse.Namespace) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    result = services.deletion.reconcile(
        actor,
        args.document_id,
        dry_run=args.dry_run,
    )
    payload: dict[str, object] = {
        "document_id": result.document_id,
        "dry_run": result.dry_run,
        "owned_object_count": len(result.object_keys),
        "deleted_object_count": result.deleted_object_count,
        "finalized": result.finalized,
    }
    if result.dry_run:
        payload["owned_object_keys"] = list(result.object_keys)
    return payload


def _gc(services: ServiceBundle, args: argparse.Namespace) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    result = services.deletion.reconcile_orphans(
        actor,
        dry_run=args.dry_run,
        grace_seconds=args.grace_seconds,
        limit=args.limit,
    )
    payload: dict[str, object] = {
        "dry_run": result.dry_run,
        "candidate_object_count": len(result.object_keys),
        "claimed_object_count": result.claimed_object_count,
        "deleted_object_count": result.deleted_object_count,
        "reconciled_missing_count": result.reconciled_missing_count,
        "eligible_missing_count": result.eligible_missing_count,
        "truncated": result.truncated,
        "limit": result.limit,
    }
    if result.dry_run:
        payload["candidate_object_keys"] = list(result.object_keys)
    return payload


def _reconcile_ingestion(
    services: ServiceBundle, args: argparse.Namespace
) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    result = services.ingestion_reconciliation.audit(actor, limit=args.limit)
    return {
        "limit": result.limit,
        "truncated": result.truncated,
        "finding_count": len(result.findings),
        "findings": [
            {
                "code": finding.code,
                "severity": finding.severity,
                "job_id": finding.job_id,
                "run_id": finding.run_id,
                "document_version_id": finding.document_version_id,
                "stage": finding.stage,
            }
            for finding in result.findings
        ],
    }


def _reconcile_ingestion_stages(
    services: ServiceBundle, args: argparse.Namespace
) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    result = services.ingestion_reconciliation.reconcile_stale_stages(
        actor,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    return {
        "dry_run": result.dry_run,
        "limit": result.limit,
        "truncated": result.truncated,
        "candidate_attempt_count": len(result.candidate_attempt_ids),
        "repaired_attempt_count": len(result.repaired_attempt_ids),
        "candidate_attempt_ids": list(result.candidate_attempt_ids),
        "repaired_attempt_ids": list(result.repaired_attempt_ids),
    }


def _maintenance_once(
    services: ServiceBundle, args: argparse.Namespace
) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    result = services.maintenance.run_once(
        actor,
        dry_run=args.dry_run,
        grace_seconds=args.grace_seconds,
        batch_size=args.batch_size,
    )
    return _maintenance_result_payload(result)


def _maintenance_loop(
    services: ServiceBundle,
    args: argparse.Namespace,
    *,
    stop_requested,
    wait,
) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    result = MaintenanceScheduler(services.maintenance.run_once, wait=wait).run(
        actor,
        max_runs=args.max_runs,
        interval_seconds=args.interval_seconds,
        dry_run=args.dry_run,
        grace_seconds=args.grace_seconds,
        batch_size=args.batch_size,
        stop_requested=stop_requested,
    )
    return {
        "attempted_runs": result.attempted_runs,
        "completed_runs": result.completed_runs,
        "stop_reason": result.stop_reason,
        "last_result": (
            None
            if result.last_result is None
            else _maintenance_result_payload(result.last_result)
        ),
    }


def _maintenance_result_payload(result) -> dict[str, object]:
    return {
        "dry_run": result.dry_run,
        "batch_size": result.batch_size,
        "stale_stages": {
            "candidate_attempt_count": len(result.stale_stages.candidate_attempt_ids),
            "repaired_attempt_count": len(result.stale_stages.repaired_attempt_ids),
            "truncated": result.stale_stages.truncated,
        },
        "terminal_runs": {
            "candidate_run_count": len(result.terminal_runs.candidate_run_ids),
            "repaired_run_count": len(result.terminal_runs.repaired_run_ids),
            "truncated": result.terminal_runs.truncated,
        },
        "orphan_gc": {
            "candidate_object_count": len(result.orphan_gc.object_keys),
            "claimed_object_count": result.orphan_gc.claimed_object_count,
            "deleted_object_count": result.orphan_gc.deleted_object_count,
            "reconciled_missing_count": result.orphan_gc.reconciled_missing_count,
            "eligible_missing_count": result.orphan_gc.eligible_missing_count,
            "truncated": result.orphan_gc.truncated,
        },
    }


@contextmanager
def _maintenance_signal_handlers(stop_event: Event) -> Iterator[None]:
    previous: dict[
        signal.Signals, Callable[[int, FrameType | None], object] | int | None
    ] = {}

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal_numbers = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signal_numbers.append(signal.SIGTERM)
    try:
        for signum in signal_numbers:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _upload(services: ServiceBundle, args: argparse.Namespace) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    tenant = Tenant(actor.tenant_id, actor.tenant_id)
    source = Source.create(actor.tenant_id, args.source_name)
    with args.path.open("rb") as stream:
        accepted = asyncio.run(
            services.uploads.accept_upload(
                actor=actor,
                tenant=tenant,
                source=source,
                canonical_key=args.canonical_key,
                title=args.title,
                filename=args.path.name,
                stream=stream,
            )
        )
    return {
        "document_id": accepted.document.id,
        "version_id": accepted.version.id,
        "job_id": accepted.job.id if accepted.job else None,
        "job_status": accepted.job.status.value if accepted.job else None,
        "duplicate": accepted.duplicate,
    }


def _search(services: ServiceBundle, args: argparse.Namespace) -> dict[str, object]:
    actor = _known_actor(services, args.tenant_id, args.user_id)
    items = services.search.search(
        actor,
        args.query,
        limit=args.limit,
        offset=args.offset,
    )
    return {
        "items": [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "document_version_id": item.document_version_id,
                "run_id": item.run_id,
                "title": item.title,
                "filename": item.filename,
                "text": item.text,
                "score": item.score,
            }
            for item in items
        ],
        "limit": args.limit,
        "offset": args.offset,
    }


def _known_actor(services: ServiceBundle, tenant_id: str, user_id: str) -> ActorContext:
    actor = ActorContext(tenant_id, user_id)
    if services.metadata.resolve_security_scope(actor) is None:
        raise ValueError("Unknown user or tenant")
    return actor


if __name__ == "__main__":
    sys.exit(main())
