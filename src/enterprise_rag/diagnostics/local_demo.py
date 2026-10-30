"""Run one isolated local text ingestion/search/evidence scenario without sockets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from enterprise_rag.bootstrap import build_services
from enterprise_rag.domain.documents import Source, Tenant, User
from enterprise_rag.domain.ingestion import JobStatus
from enterprise_rag.security import ActorContext, Permission, PrincipalType
from enterprise_rag.settings import Settings
from enterprise_rag.workers import LiteTextStageExecutor, LiteWorker

MAX_INPUT_BYTES = 64 * 1024


@dataclass
class DemoReport:
    status: str
    started_at: str
    input_path: str
    input_sha256: str
    input_bytes: int
    query: str
    output_path: str
    profile: str = "lite"
    llm_provider: str = "none"
    downloaded_bytes: int = 0
    checks: dict[str, bool] = field(default_factory=dict)
    timings_seconds: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    report_error: str | None = None
    source_sha256: dict[str, str] = field(default_factory=dict)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_demo(source: Path, output: Path, query: str) -> DemoReport:
    """Require a new output directory; preserve every previous run and input."""
    if source.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Use a TXT or Markdown input")
    if not query.strip():
        raise ValueError("Query must be non-empty")
    with source.open("rb") as stream:
        content = stream.read(MAX_INPUT_BYTES + 1)
    if not content or len(content) > MAX_INPUT_BYTES:
        raise ValueError("Input must contain 1..65536 bytes")
    content.decode("utf-8")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = DemoReport(
        status="running",
        started_at=datetime.now(UTC).isoformat(),
        input_path=str(source.resolve()),
        input_sha256=hashlib.sha256(content).hexdigest(),
        input_bytes=len(content),
        query=query,
        output_path=str(output),
    )
    started = time.perf_counter()
    try:
        # Source checkouts work without installed distribution metadata or API extras.
        report.versions = {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        }
        package = Path(__file__).resolve().parents[1]
        report.source_sha256 = {
            path.relative_to(package).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(package.rglob("*.py"))
            if path.is_file()
        }
        settings = Settings(
            profile="lite",
            data_dir=output / "data",
            artifacts_dir=output / "artifacts",
            database_url="sqlite:///" + (output / "data/metadata.sqlite3").as_posix(),
            object_store="local",
            llm_provider="none",
        )
        services = build_services(settings)
        tenant = Tenant.create("isolated-local-demo", "Isolated synthetic local demo")
        user = User.create(tenant.id, "demo@example.invalid")
        foreign = Tenant.create("isolated-foreign", "Synthetic foreign tenant")
        foreign_user = User.create(foreign.id, "foreign@example.invalid")
        for current_tenant, current_user in ((tenant, user), (foreign, foreign_user)):
            services.metadata.ensure_tenant(current_tenant)
            services.metadata.upsert_user(current_user)
        # This role exists only in this newly created synthetic SQLite database.
        _require(
            services.metadata.grant_tenant_admin(tenant.id, user.id),
            "Could not establish the isolated demo actor",
        )
        actor = ActorContext(tenant.id, user.id)
        tick = time.perf_counter()
        accepted = asyncio.run(
            services.uploads.accept_upload(
                actor=actor,
                tenant=tenant,
                source=Source.create(tenant.id, "local-demo"),
                canonical_key="demo/" + source.name,
                title=source.name,
                filename=source.name,
                stream=BytesIO(content),
            )
        )
        report.timings_seconds["upload"] = time.perf_counter() - tick
        if accepted.job is None:
            raise RuntimeError("Upload did not queue a job")
        # Tenant-admin provisioning does not grant document READ implicitly.
        # This admin-boundary mutation affects only this new synthetic database.
        permission = services.management.grant_permission(
            actor, accepted.document.id, PrincipalType.USER, user.id, Permission.READ
        )
        _require(
            permission.document_id == accepted.document.id
            and permission.principal_type is PrincipalType.USER
            and permission.principal_id == user.id
            and permission.permission is Permission.READ,
            "Could not grant READ for the isolated demo document",
        )
        report.checks["queued"] = accepted.job.status is JobStatus.QUEUED
        report.checks["not_visible_before_worker"] = not services.search.search(
            actor, query
        )
        tick = time.perf_counter()
        result = LiteWorker(services.metadata, "local-demo-worker").run_once(
            LiteTextStageExecutor(services.metadata, services.object_store)
        )
        report.timings_seconds["worker"] = time.perf_counter() - tick
        _require(result.completed and not result.lease_lost, "Worker did not complete")
        job = services.metadata.get_job(accepted.job.id)
        report.checks["job_succeeded"] = (
            job is not None and job.status is JobStatus.SUCCEEDED
        )
        tick = time.perf_counter()
        hits = services.search.search(actor, query)
        report.timings_seconds["search"] = time.perf_counter() - tick
        _require(bool(hits), "No evidence for the requested query")
        hit = hits[0]
        evidence = services.search.get_evidence(actor, hit.chunk_id)
        if evidence is None:
            raise RuntimeError("Evidence could not be rehydrated")
        report.checks["identity_and_evidence_consistent"] = (
            hit.document_id == accepted.document.id
            and hit.document_version_id == accepted.version.id
            and hit.run_id == accepted.job.run_id
            and evidence.chunk_id == hit.chunk_id
            and evidence.document_id == hit.document_id
            and evidence.document_version_id == hit.document_version_id
            and evidence.run_id == hit.run_id
            and evidence.text == hit.text
            and evidence.page_numbers == hit.page_numbers
        )
        report.evidence = asdict(evidence)
        report.checks["anonymous_denied"] = not services.search.search(None, query)
        foreign_actor = ActorContext(foreign.id, foreign_user.id)
        report.checks["foreign_tenant_denied"] = (
            not services.search.search(foreign_actor, query)
            and services.search.get_evidence(foreign_actor, hit.chunk_id) is None
        )
        with services.object_store.open(accepted.version.object_key) as raw:
            report.checks["stored_input_matches"] = raw.read() == content
        reopened = build_services(settings)
        persisted = reopened.search.search(actor, query)
        report.checks["reopen_same_evidence"] = bool(persisted) and (
            persisted[0].chunk_id == hit.chunk_id
            and persisted[0].text == hit.text
            and persisted[0].run_id == hit.run_id
        )
        _require(all(report.checks.values()), "One or more scenario checks failed")
        report.status = "passed"
    except Exception as error:
        report.status = "failed"
        report.error = f"{type(error).__name__}: {error}"
    finally:
        report.timings_seconds["total"] = time.perf_counter() - started
        try:
            (output / "report.json").write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            # Retain the primary processing error; stdout can still carry both errors.
            report.status = "failed"
            report.report_error = f"{type(error).__name__}: {error}"
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query", default="safecanoe")
    args = parser.parse_args(argv)
    try:
        report = run_demo(args.input, args.output_dir, args.query)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}))
        return 2
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
