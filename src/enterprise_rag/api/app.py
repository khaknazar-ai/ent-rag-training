"""Small authenticated FastAPI transport over the lite application services."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Protocol
from urllib.parse import unquote_to_bytes

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from enterprise_rag.application import (
    DEFAULT_INGESTION_AUDIT_LIMIT,
    DeletionStorageError,
    MaintenanceBusy,
    MaintenanceLeaseLost,
    ManagementConflict,
    ManagementForbidden,
    ManagementNotFound,
    SearchEvidence,
)
from enterprise_rag.application.answer import (
    AnswerResult,
    ModelProtocolError,
    ModelUnavailable,
)
from enterprise_rag.application.document_summary import SummaryNotFound
from enterprise_rag.application.semantic import SemanticProviderUnavailable
from enterprise_rag.application.uploads import UploadForbidden, UploadTooLarge
from enterprise_rag.application.visual_regions import RegionNotFound, RegionUnavailable
from enterprise_rag.bootstrap import ServiceBundle, build_services
from enterprise_rag.domain.documents.models import (
    Document,
    DocumentVersion,
    Group,
    Source,
    Tenant,
    User,
)
from enterprise_rag.domain.ingestion.models import IngestionJob
from enterprise_rag.domain.model_summary import ModelSummaryResult
from enterprise_rag.ports.query_history import (
    QueryHistoryForbidden,
    QueryHistoryUnavailable,
)
from enterprise_rag.retrieval.query_processing import LexicalQueryBudgetExceeded
from enterprise_rag.retrieval.reranking import (
    RerankerProtocolError,
    RerankerUnavailable,
)
from enterprise_rag.retrieval.sqlite_dense import DenseSnapshotError
from enterprise_rag.security import ActorContext, DocumentPermission
from enterprise_rag.storage import ObjectTooLarge

MAX_JSON_BODY_BYTES = 64 * 1024


class TokenAuthenticator(Protocol):
    def authenticate(self, authorization: str | None) -> ActorContext | None:
        """Return the actor bound to a bearer token, or None on failure."""


class SummaryHandler(Protocol):
    def summarize(
        self, actor: ActorContext, document_id: str, version_id: str
    ) -> AnswerResult | ModelSummaryResult: ...


@dataclass(frozen=True, slots=True)
class StaticTokenAuthenticator:
    """Explicit local token map; no token or actor is invented by the API."""

    tokens: Mapping[str, ActorContext]

    def authenticate(self, authorization: str | None) -> ActorContext | None:
        if authorization is None:
            return None
        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.casefold() != "bearer" or not token:
            return None
        return self.tokens.get(token)


@dataclass(frozen=True, slots=True)
class _ApiState:
    services: ServiceBundle
    authenticator: TokenAuthenticator


def create_app(
    services: ServiceBundle | None = None,
    authenticator: TokenAuthenticator | None = None,
) -> FastAPI:
    """Create the API; production callers must inject an explicit authenticator."""

    bundle = services or build_services()
    state = _ApiState(bundle, authenticator or _authenticator_from_env())
    app = FastAPI(title="Enterprise RAG Lite", version="0.1.0")
    app.state.enterprise_rag = state

    @app.exception_handler(RegionNotFound)
    async def region_not_found(request: Request, error: RegionNotFound):
        return JSONResponse(
            status_code=404,
            content={"detail": {"code": "region_not_found"}},
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(RegionUnavailable)
    async def region_unavailable(request: Request, error: RegionUnavailable):
        return JSONResponse(
            status_code=503,
            content={"detail": {"code": "region_unavailable"}},
            headers={"Cache-Control": "no-store"},
        )

    def regions():
        if state.services.visual_regions is None:
            raise RegionUnavailable("Visual regions unavailable")
        return state.services.visual_regions

    @app.post("/v1/queries/{query_run_id}/citations/{citation_index}/regions")
    async def create_region(request: Request, query_run_id: str, citation_index: int):
        actor = _authenticated_actor(request, state)
        body = await _json_object(request)
        if set(body) - {"bbox"}:
            raise HTTPException(400, detail={"code": "invalid_region_request"})
        try:
            result = await asyncio.to_thread(
                regions().create, actor, query_run_id, citation_index, body.get("bbox")
            )
        except RegionNotFound:
            raise
        except ValueError:
            raise HTTPException(
                400, detail={"code": "invalid_region_request"}
            ) from None
        except (OSError, sqlite3.Error):
            raise RegionUnavailable("Image storage unavailable") from None
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/v1/queries/{query_run_id}/citations/{citation_index}/ocr-regions")
    async def get_citation_ocr_regions(
        request: Request, query_run_id: str, citation_index: int
    ):
        actor = _authenticated_actor(request, state)
        try:
            result = await asyncio.to_thread(
                regions().read_ocr_regions, actor, query_run_id, citation_index
            )
        except RegionNotFound:
            raise
        except (OSError, sqlite3.Error):
            raise RegionUnavailable("OCR evidence unavailable") from None
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/v1/regions/{region_id}")
    def get_region(request: Request, region_id: str):
        actor = _authenticated_actor(request, state)
        try:
            result, _ = regions().read(actor, region_id)
        except (OSError, sqlite3.Error):
            raise RegionUnavailable("Image storage unavailable") from None
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/v1/regions/{region_id}/image")
    def get_region_image(request: Request, region_id: str):
        actor = _authenticated_actor(request, state)
        try:
            _, png = regions().read(actor, region_id)
        except (OSError, sqlite3.Error):
            raise RegionUnavailable("Image storage unavailable") from None
        return Response(
            png,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.exception_handler(QueryHistoryUnavailable)
    async def history_unavailable(request: Request, error: QueryHistoryUnavailable):
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "query_history_unavailable",
                    "query_run_id": error.query_run_id,
                }
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(QueryHistoryForbidden)
    async def history_forbidden(request: Request, error: QueryHistoryForbidden):
        return JSONResponse(
            status_code=403,
            content={"detail": {"code": "query_history_forbidden"}},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/queries")
    def query_history_list(
        request: Request,
        limit: int = Query(default=10, ge=1, le=20),
        offset: int = Query(default=0, ge=0, le=100_000),
    ) -> JSONResponse:
        actor = _authenticated_actor(request, state)
        history = state.services.query_history
        if history is None:
            raise HTTPException(
                status_code=503, detail={"code": "query_history_unavailable"}
            )
        page = history.list(actor, limit=limit, offset=offset)
        return JSONResponse(
            content={
                "items": [
                    {
                        "query_run_id": item.id,
                        "query_text": item.query_text,
                        "status": item.status,
                        "started_at": item.started_at,
                        "finished_at": item.finished_at,
                        "elapsed_seconds": (item.stage_latencies or {}).get(
                            "total_seconds"
                        ),
                    }
                    for item in page.items
                ],
                "next_offset": page.next_offset,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/queries/{query_run_id}")
    def query_history(request: Request, query_run_id: str) -> JSONResponse:
        actor = _authenticated_actor(request, state)
        if state.services.visual_regions is not None:
            try:
                payload = state.services.visual_regions.history_payload(
                    actor, query_run_id
                )
            except RegionNotFound:
                raise HTTPException(
                    status_code=404, detail={"code": "query_history_not_found"}
                ) from None
            except (OSError, sqlite3.Error):
                raise RegionUnavailable("Image storage unavailable") from None
            return JSONResponse(payload, headers={"Cache-Control": "no-store"})
        if state.services.query_history is None:
            raise HTTPException(
                status_code=503, detail={"code": "query_history_unavailable"}
            )
        item = state.services.query_history.get(actor, query_run_id)
        if item is None:
            raise HTTPException(
                status_code=404, detail={"code": "query_history_not_found"}
            )
        return JSONResponse(content=asdict(item), headers={"Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def web_home() -> HTMLResponse:
        # Public shell contains no tenant, token, configuration or document data.
        return HTMLResponse(_web_asset("index.html"), headers=_web_headers())

    @app.get("/ui/{name}", include_in_schema=False)
    def web_asset(name: str) -> Response:
        media = {"app.js": "text/javascript", "app.css": "text/css"}
        if name not in media:
            raise HTTPException(status_code=404)
        return Response(
            _web_asset(name), media_type=media[name], headers=_web_headers()
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live", "profile": state.services.settings.profile}

    @app.get("/health/ready", response_model=None)
    def ready() -> JSONResponse | dict[str, str]:
        try:
            state.services.metadata.get_authorization_revision("healthcheck")
        except (OSError, sqlite3.Error):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "code": "metadata_unavailable"},
            )
        return {"status": "ready", "profile": state.services.settings.profile}

    @app.post("/v1/documents", status_code=202)
    async def upload(request: Request) -> JSONResponse:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        canonical_key = _required_header(request, "X-Canonical-Key")
        title = _upload_text_header(request, "X-Title")
        filename = _upload_text_header(request, "X-Filename")
        source_name = request.headers.get("X-Source-Name", "api")
        if not source_name.strip():
            raise _bad_request("source_name_required")
        max_bytes = state.services.uploads.max_upload_bytes
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_bytes:
                    raise _payload_too_large()
            except ValueError:
                raise _bad_request("invalid_content_length") from None

        total = 0
        with tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024,
            mode="w+b",
            dir=(
                str(state.services.uploads.spool_dir)
                if state.services.uploads.spool_dir is not None
                else None
            ),
        ) as body:
            try:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > max_bytes:
                        raise _payload_too_large()
                    await asyncio.to_thread(body.write, chunk)
                await asyncio.to_thread(body.seek, 0)
                accepted = await state.services.uploads.accept_upload(
                    actor=actor,
                    tenant=_tenant_for(actor),
                    source=Source.create(actor.tenant_id, source_name),
                    canonical_key=canonical_key,
                    title=title,
                    filename=filename,
                    stream=body,
                )
            except (UploadTooLarge, ObjectTooLarge):
                raise _payload_too_large() from None
            except UploadForbidden:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "upload_forbidden",
                        "message": "Upload is not allowed for this document",
                    },
                ) from None
            except ValueError:
                raise _bad_request("invalid_upload_metadata") from None

        return JSONResponse(
            status_code=202,
            content={
                "document_id": accepted.document.id,
                "version_id": accepted.version.id,
                "job_id": accepted.job.id if accepted.job else None,
                "job_status": accepted.job.status.value if accepted.job else None,
                "duplicate": accepted.duplicate,
            },
        )

    @app.get("/v1/documents")
    def list_documents(
        request: Request,
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            items = state.services.management.list_documents(
                actor, limit=limit, offset=offset
            )
        except ValueError:
            raise _bad_request("invalid_management_parameters") from None
        return {
            "items": [_document_payload(item) for item in items],
            "limit": limit,
            "offset": offset,
        }

    @app.get("/v1/documents/{document_id}/versions")
    def list_versions(request: Request, document_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        items = state.services.management.list_versions(actor, document_id)
        if items is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "versions_not_found",
                    "message": "Document versions not found",
                },
            )
        return {
            "document_id": document_id,
            "items": [_version_payload(item) for item in items],
        }

    @app.get("/v1/documents/{document_id}")
    def get_document(request: Request, document_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        item = state.services.management.get_document(actor, document_id)
        if item is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "document_not_found",
                    "message": "Document not found",
                },
            )
        return _document_payload(item)

    @app.delete("/v1/documents/{document_id}", status_code=202)
    def delete_document(request: Request, document_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            decision = state.services.management.delete_document(actor, document_id)
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return {
            "tombstoned": decision.tombstoned,
            "document": _document_payload(decision.document),
        }

    @app.post("/v1/admin/documents/{document_id}/reconcile")
    def reconcile_document(
        request: Request,
        document_id: str,
        dry_run: bool = Query(False),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            result = state.services.deletion.reconcile(
                actor, document_id, dry_run=dry_run
            )
        except DeletionStorageError:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "deletion_storage_unavailable",
                    "message": "Owned object cleanup is temporarily unavailable",
                },
            ) from None
        except (
            ManagementConflict,
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(
                exc, conflict_code="deletion_conflict"
            ) from None
        return {
            "document_id": result.document_id,
            "dry_run": result.dry_run,
            "owned_object_count": len(result.object_keys),
            "deleted_object_count": result.deleted_object_count,
            "finalized": result.finalized,
        }

    @app.post("/v1/admin/object-store/gc")
    def reconcile_orphan_objects(
        request: Request,
        dry_run: bool = Query(False),
        grace_seconds: float | None = Query(None, ge=0),
        limit: int | None = Query(None, ge=1, le=1000),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            result = state.services.deletion.reconcile_orphans(
                actor,
                dry_run=dry_run,
                grace_seconds=grace_seconds,
                limit=limit,
            )
        except DeletionStorageError:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "deletion_storage_unavailable",
                    "message": "Owned object cleanup is temporarily unavailable",
                },
            ) from None
        except (
            ManagementConflict,
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(
                exc, conflict_code="orphan_gc_conflict"
            ) from None
        return {
            "dry_run": result.dry_run,
            "candidate_object_count": len(result.object_keys),
            "claimed_object_count": result.claimed_object_count,
            "deleted_object_count": result.deleted_object_count,
            "reconciled_missing_count": result.reconciled_missing_count,
            "eligible_missing_count": result.eligible_missing_count,
            "truncated": result.truncated,
            "limit": result.limit,
        }

    @app.post("/v1/admin/maintenance/run-once")
    def run_maintenance_once(
        request: Request,
        dry_run: bool = Query(False),
        grace_seconds: float | None = Query(None, ge=0),
        batch_size: int | None = Query(None, ge=1, le=1000),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            result = state.services.maintenance.run_once(
                actor,
                dry_run=dry_run,
                grace_seconds=grace_seconds,
                batch_size=batch_size,
            )
        except DeletionStorageError:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "maintenance_storage_unavailable",
                    "message": "Maintenance object cleanup is temporarily unavailable",
                },
            ) from None
        except MaintenanceBusy as exc:
            raise _management_http_error(
                exc, conflict_code="maintenance_busy"
            ) from None
        except MaintenanceLeaseLost as exc:
            raise _management_http_error(
                exc, conflict_code="maintenance_lease_lost"
            ) from None
        except (
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(
                exc, conflict_code="maintenance_conflict"
            ) from None
        return {
            "dry_run": result.dry_run,
            "batch_size": result.batch_size,
            "stale_stages": {
                "candidate_attempt_count": len(
                    result.stale_stages.candidate_attempt_ids
                ),
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
                "reconciled_missing_count": (result.orphan_gc.reconciled_missing_count),
                "eligible_missing_count": result.orphan_gc.eligible_missing_count,
                "truncated": result.orphan_gc.truncated,
            },
        }

    @app.post("/v1/admin/ingestion/reconcile")
    def reconcile_ingestion(
        request: Request,
        limit: int = Query(DEFAULT_INGESTION_AUDIT_LIMIT, ge=1, le=1000),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            result = state.services.ingestion_reconciliation.audit(actor, limit=limit)
        except (
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(
                exc, conflict_code="ingestion_reconcile_conflict"
            ) from None
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

    @app.post("/v1/admin/ingestion/stage-attempts/reconcile")
    def reconcile_stale_stage_attempts(
        request: Request,
        dry_run: bool = Query(False),
        limit: int = Query(DEFAULT_INGESTION_AUDIT_LIMIT, ge=1, le=1000),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            result = state.services.ingestion_reconciliation.reconcile_stale_stages(
                actor,
                dry_run=dry_run,
                limit=limit,
            )
        except (
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(
                exc, conflict_code="stage_reconcile_conflict"
            ) from None
        return {
            "dry_run": result.dry_run,
            "limit": result.limit,
            "truncated": result.truncated,
            "candidate_attempt_count": len(result.candidate_attempt_ids),
            "repaired_attempt_count": len(result.repaired_attempt_ids),
            "candidate_attempt_ids": list(result.candidate_attempt_ids),
            "repaired_attempt_ids": list(result.repaired_attempt_ids),
        }

    @app.get("/v1/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        item = state.services.management.get_job(actor, job_id)
        if item is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job not found"},
            )
        return _job_payload(item)

    @app.post("/v1/document-versions/{version_id}/reprocess", status_code=202)
    async def reprocess_version(request: Request, version_id: str) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        if set(payload) != {"request_id", "max_chunk_chars"}:
            raise _bad_request("invalid_reprocess_fields")
        request_id = payload["request_id"]
        chunk_chars = payload["max_chunk_chars"]
        if not isinstance(request_id, str) or type(chunk_chars) is not int:
            raise _bad_request("invalid_reprocess_configuration")
        try:
            decision = await asyncio.to_thread(
                state.services.management.reprocess_version,
                actor,
                version_id,
                request_id,
                max_chunk_chars=chunk_chars,
                now=time.time(),
            )
        except (
            ManagementConflict,
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(
                exc, conflict_code="reprocess_conflict"
            ) from None
        return {"created": decision.created, "job": _job_payload(decision.job)}

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    def retry_job(request: Request, job_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            decision = state.services.management.retry_job(
                actor, job_id, now=time.time()
            )
        except (
            ManagementConflict,
            ManagementForbidden,
            ManagementNotFound,
            ValueError,
        ) as exc:
            raise _management_http_error(exc) from None
        return {
            "requeued": decision.requeued,
            "job": _job_payload(decision.job),
        }

    @app.post("/v1/admin/users", status_code=201)
    async def provision_user(request: Request) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        try:
            item = await asyncio.to_thread(
                state.services.management.provision_user,
                actor,
                _required_payload_string(payload, "email"),
            )
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return _user_payload(item)

    @app.post("/v1/admin/groups", status_code=201)
    async def provision_group(request: Request) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        try:
            item = await asyncio.to_thread(
                state.services.management.provision_group,
                actor,
                _required_payload_string(payload, "name"),
            )
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return _group_payload(item)

    @app.post("/v1/admin/groups/{group_id}/members")
    async def add_group_member(request: Request, group_id: str) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        try:
            added = await asyncio.to_thread(
                state.services.management.add_group_member,
                actor,
                group_id,
                _required_payload_string(payload, "user_id"),
            )
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return {"group_id": group_id, "user_id": payload["user_id"], "added": added}

    @app.delete("/v1/admin/groups/{group_id}/members/{user_id}")
    def remove_group_member(
        request: Request, group_id: str, user_id: str
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            removed = state.services.management.remove_group_member(
                actor, group_id, user_id
            )
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return {"group_id": group_id, "user_id": user_id, "removed": removed}

    @app.post("/v1/admin/documents/{document_id}/permissions")
    async def grant_permission(request: Request, document_id: str) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        try:
            permission = await asyncio.to_thread(
                state.services.management.grant_permission,
                actor,
                document_id,
                _required_payload_string(payload, "principal_type"),
                _required_payload_string(payload, "principal_id"),
                _required_payload_string(payload, "permission"),
            )
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return _permission_payload(permission)

    @app.delete(
        "/v1/admin/documents/{document_id}/permissions/"
        "{principal_type}/{principal_id}/{permission}"
    )
    def revoke_permission(
        request: Request,
        document_id: str,
        principal_type: str,
        principal_id: str,
        permission: str,
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            revoked = state.services.management.revoke_permission(
                actor,
                document_id,
                principal_type,
                principal_id,
                permission,
            )
        except (ManagementForbidden, ManagementNotFound, ValueError) as exc:
            raise _management_http_error(exc) from None
        return {
            "document_id": document_id,
            "principal_type": principal_type,
            "principal_id": principal_id,
            "permission": permission,
            "revoked": revoked,
        }

    @app.get("/v1/search")
    def search(
        request: Request,
        q: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        try:
            items = state.services.search.search(actor, q, limit=limit, offset=offset)
        except LexicalQueryBudgetExceeded:
            raise _bad_request("search_query_budget_exceeded") from None
        except ValueError:
            raise _bad_request("invalid_search_parameters") from None
        return {
            "items": [_evidence_payload(item) for item in items],
            "limit": limit,
            "offset": offset,
        }

    @app.get("/v1/retrieval")
    async def semantic_retrieval(
        request: Request,
        q: str = Query(..., min_length=1),
        strategy: str = Query("dense"),
        limit: int = Query(4, ge=1, le=20),
    ) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        service = state.services.semantic
        if service is None:
            raise HTTPException(
                status_code=503, detail={"code": "semantic_not_configured"}
            )
        try:
            result = await service.search(actor, q, strategy=strategy, limit=limit)
        except (
            DenseSnapshotError,
            SemanticProviderUnavailable,
            RerankerUnavailable,
            RerankerProtocolError,
            OSError,
            sqlite3.Error,
        ):
            raise HTTPException(
                status_code=503, detail={"code": "semantic_unavailable"}
            ) from None
        except LexicalQueryBudgetExceeded:
            raise _bad_request("search_query_budget_exceeded") from None
        except ValueError:
            raise _bad_request("invalid_semantic_parameters") from None
        return {
            "strategy": result.strategy,
            "items": [
                {
                    "evidence": _evidence_payload(item.evidence),
                    "scores": {
                        "dense": item.dense_score,
                        "lexical": item.lexical_score,
                        "fusion": item.fusion_score,
                        "rerank": item.rerank_score,
                    },
                }
                for item in result.items
            ],
            "limit": limit,
        }

    @app.post("/v1/semantic-answer")
    async def semantic_answer(request: Request) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        question = _required_payload_string(payload, "question")
        strategy, mode = (
            payload.get("strategy", "hybrid"),
            payload.get("mode", "evidence_only"),
        )
        if (
            len(question) > 500
            or not isinstance(strategy, str)
            or not isinstance(mode, str)
            or strategy not in {"lexical", "dense", "hybrid", "hybrid_rerank"}
            or mode not in {"evidence_only", "extractive"}
        ):
            raise _bad_request("invalid_semantic_answer_parameters")
        service = state.services.semantic_answer
        if service is None or (mode == "extractive" and service.model is None):
            raise HTTPException(
                status_code=503, detail={"code": "semantic_answer_not_configured"}
            )
        try:
            result = await service.answer(actor, question, strategy=strategy, mode=mode)
        except (
            DenseSnapshotError,
            SemanticProviderUnavailable,
            RerankerUnavailable,
            RerankerProtocolError,
            ModelUnavailable,
            OSError,
            sqlite3.Error,
        ) as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "semantic_answer_unavailable",
                    "query_run_id": getattr(exc, "query_run_id", None),
                },
            ) from None
        except ModelProtocolError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "semantic_answer_invalid_response",
                    "query_run_id": getattr(exc, "query_run_id", None),
                },
            ) from None
        except (ValueError, LexicalQueryBudgetExceeded):
            raise _bad_request("invalid_semantic_answer_parameters") from None
        return asdict(result)

    @app.get("/v1/evidence/{chunk_id}")
    def evidence(request: Request, chunk_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        item = state.services.search.get_evidence(actor, chunk_id)
        if item is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "evidence_not_found", "message": "Evidence not found"},
            )
        return _evidence_payload(item)

    @app.get("/v1/evidence/{chunk_id}/image-regions")
    def evidence_image_regions(
        request: Request, chunk_id: str, page_number: int | None = Query(None, ge=1)
    ) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        result = state.services.search.get_pdf_image_regions(
            actor, chunk_id, page_number=page_number
        )
        if result is None:
            raise HTTPException(
                404,
                detail={"code": "evidence_not_found", "message": "Evidence not found"},
            )
        payload = asdict(result)
        payload["status"] = (
            "not_extracted"
            if result.provider is None
            else "partial"
            if result.warnings
            else "available"
        )
        return payload

    @app.get("/v1/evidence/{chunk_id}/table-cells")
    def evidence_table_cells(request: Request, chunk_id: str) -> dict[str, object]:
        actor = _authenticated_actor(request, state)
        cells = state.services.search.get_pdf_table_cells(actor, chunk_id)
        if cells is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "evidence_not_found", "message": "Evidence not found"},
            )
        return {"chunk_id": chunk_id, "cells": [asdict(cell) for cell in cells]}

    @app.post("/v1/answer")
    async def answer(request: Request) -> dict[str, object]:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        question = _required_payload_string(payload, "question")
        if len(question) > 500:
            raise _bad_request("question_too_long")
        search_query = payload.get("search_query")
        if search_query is not None and (
            not isinstance(search_query, str)
            or not search_query.strip()
            or len(search_query) > 200
        ):
            raise _bad_request("invalid_search_query")
        service = state.services.answer
        if service is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "model_not_configured",
                    "message": "Local model not configured",
                },
            )
        try:
            result = await asyncio.to_thread(
                service.answer, actor, question, search_query=search_query
            )
        except LexicalQueryBudgetExceeded:
            raise _bad_request("search_query_budget_exceeded") from None
        except ValueError:
            raise _bad_request("invalid_question") from None
        except ModelUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "model_unavailable",
                    "query_run_id": getattr(exc, "query_run_id", None),
                    "message": "Local model unavailable",
                },
            ) from None
        except ModelProtocolError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "model_invalid_response",
                    "query_run_id": getattr(exc, "query_run_id", None),
                    "message": "Invalid local model response",
                },
            ) from None
        return asdict(result)

    @app.post("/v1/visual-answer")
    async def visual_answer(request: Request) -> JSONResponse:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        if set(payload) - {"question", "search_query"}:
            raise _bad_request("invalid_visual_answer_parameters")
        question = _required_payload_string(payload, "question")
        if len(question) > 500:
            raise _bad_request("question_too_long")
        search_query = payload.get("search_query")
        if search_query is not None and (
            not isinstance(search_query, str)
            or not search_query.strip()
            or len(search_query) > 200
        ):
            raise _bad_request("invalid_search_query")
        service = state.services.visual_answer
        regions_service = state.services.visual_regions
        if (
            service is None
            or regions_service is None
            or regions_service.renderer is None
        ):
            raise HTTPException(
                status_code=503,
                detail={"code": "visual_answer_not_configured"},
            )
        query_run_id = None
        try:
            outcome = await asyncio.to_thread(
                service.answer, actor, question, search_query=search_query
            )
            query_run_id = outcome.result.query_run_id
            if not outcome.result.abstained:
                if outcome.result.query_run_id is None or outcome.bbox is None:
                    raise ModelProtocolError(
                        "Visual result has no saved region binding"
                    )
                await asyncio.to_thread(
                    regions_service.create,
                    actor,
                    outcome.result.query_run_id,
                    1,
                    outcome.bbox,
                )
            saved = await asyncio.to_thread(
                regions_service.history_payload,
                actor,
                outcome.result.query_run_id,
            )
        except RegionNotFound:
            raise HTTPException(
                status_code=404, detail={"code": "visual_evidence_not_found"}
            ) from None
        except (RegionUnavailable, OSError, sqlite3.Error, ModelUnavailable):
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "visual_answer_unavailable",
                    "query_run_id": query_run_id,
                },
            ) from None
        except ModelProtocolError:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "visual_answer_invalid_response",
                    "query_run_id": query_run_id,
                },
            ) from None
        except (ValueError, LexicalQueryBudgetExceeded):
            raise _bad_request("invalid_visual_answer_parameters") from None
        return JSONResponse(content=saved, headers={"Cache-Control": "no-store"})

    @app.post("/v1/documents/{document_id}/summary")
    async def document_summary(request: Request, document_id: str) -> JSONResponse:
        actor = await asyncio.to_thread(_authenticated_actor, request, state)
        payload = await _json_object(request)
        if (
            set(payload) != {"version_id", "mode"}
            or not isinstance(payload.get("mode"), str)
            or payload.get("mode")
            not in {
                "extractive",
                "model",
            }
        ):
            raise _bad_request("explicit_summary_scope_required")
        version_id = _required_payload_string(payload, "version_id")
        if len(document_id) > 128 or len(version_id) > 128:
            raise _bad_request("invalid_summary_scope")
        service: SummaryHandler | None = (
            state.services.model_summary
            if payload["mode"] == "model"
            else state.services.summary
        )
        if service is None:
            raise HTTPException(
                status_code=503, detail={"code": "summary_not_configured"}
            )
        try:
            result = await asyncio.to_thread(
                service.summarize, actor, document_id, version_id
            )
        except SummaryNotFound:
            raise HTTPException(
                status_code=404, detail={"code": "summary_scope_not_found"}
            ) from None
        except (ModelUnavailable, ModelProtocolError, sqlite3.Error, OSError) as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "summary_unavailable",
                    "query_run_id": getattr(error, "query_run_id", None),
                },
            ) from None
        return JSONResponse(
            content=asdict(result), headers={"Cache-Control": "no-store"}
        )

    return app


def _authenticator_from_env(
    environ: Mapping[str, str] | None = None,
) -> StaticTokenAuthenticator:
    env = os.environ if environ is None else environ
    token = env.get("RAG_API_TOKEN")
    tenant_id = env.get("RAG_DEMO_TENANT_ID")
    user_id = env.get("RAG_DEMO_USER_ID")
    if not token or not tenant_id or not user_id:
        return StaticTokenAuthenticator({})
    return StaticTokenAuthenticator(
        {token: ActorContext(tenant_id=tenant_id, user_id=user_id)}
    )


def _authenticated_actor(request: Request, state: _ApiState) -> ActorContext:
    actor = state.authenticator.authenticate(request.headers.get("Authorization"))
    try:
        scope = (
            None
            if actor is None
            else state.services.metadata.resolve_security_scope(actor)
        )
    except (OSError, sqlite3.Error):
        raise HTTPException(
            status_code=503, detail={"code": "authorization_unavailable"}
        ) from None
    if actor is None or scope is None:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "unauthenticated",
                "message": "Valid bearer token required",
            },
        )
    claimed_tenant = request.headers.get("X-Tenant-ID")
    claimed_user = request.headers.get("X-User-ID")
    if claimed_tenant is not None and claimed_tenant != actor.tenant_id:
        raise HTTPException(
            status_code=403,
            detail={"code": "tenant_spoofing", "message": "Tenant claim mismatch"},
        )
    if claimed_user is not None and claimed_user != actor.user_id:
        raise HTTPException(
            status_code=403,
            detail={"code": "user_spoofing", "message": "User claim mismatch"},
        )
    return actor


def _required_header(request: Request, name: str) -> str:
    value = request.headers.get(name)
    if value is None or not value.strip():
        raise _bad_request(f"{name.lower().replace('-', '_')}_required")
    return value


def _upload_text_header(request: Request, name: str) -> str:
    """UTF-8 extended headers for browser uploads; retain the existing API form."""
    extended = request.headers.get(name + "*")
    if extended is None:
        return _required_header(request, name)
    if request.headers.get(name) is not None or not extended.startswith("UTF-8''"):
        raise _bad_request("invalid_upload_metadata")
    encoded = extended[7:]
    if re.search(r"%(?![0-9A-Fa-f]{2})", encoded):
        raise _bad_request("invalid_upload_metadata")
    try:
        value = unquote_to_bytes(encoded).decode("utf-8")
    except (ValueError, UnicodeError):
        raise _bad_request("invalid_upload_metadata") from None
    if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise _bad_request("invalid_upload_metadata")
    return value


def _web_asset(name: str) -> str:
    return files("enterprise_rag.api").joinpath("web", name).read_text(encoding="utf-8")


def _web_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": (
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        ),
    }


def _tenant_for(actor: ActorContext):
    return Tenant(actor.tenant_id, actor.tenant_id)


def _evidence_payload(item: SearchEvidence) -> dict[str, object]:
    return asdict(item)


async def _json_object(request: Request) -> dict[str, object]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            raise _bad_request("invalid_content_length") from None
        if declared_size < 0:
            raise _bad_request("invalid_content_length")
        if declared_size > MAX_JSON_BODY_BYTES:
            raise _json_payload_too_large()

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_JSON_BODY_BYTES:
            raise _json_payload_too_large()
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (TypeError, ValueError, RecursionError):
        raise _bad_request("invalid_json") from None
    if not isinstance(payload, dict):
        raise _bad_request("json_object_required")
    return payload


def _required_payload_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise _bad_request(f"{name}_required")
    return value


def _management_http_error(
    error: Exception, *, conflict_code: str = "job_retry_conflict"
) -> HTTPException:
    if isinstance(error, ManagementForbidden):
        return HTTPException(
            status_code=403,
            detail={
                "code": "admin_required",
                "message": "Tenant admin permission required",
            },
        )
    if isinstance(error, ManagementNotFound):
        return HTTPException(
            status_code=404,
            detail={"code": "resource_not_found", "message": "Resource not found"},
        )
    if isinstance(error, ManagementConflict):
        return HTTPException(
            status_code=409,
            detail={
                "code": conflict_code,
                "message": "Management operation conflicts with current state",
            },
        )
    return _bad_request("invalid_management_request")


def _document_payload(item: Document) -> dict[str, object]:
    return {
        "id": item.id,
        "tenant_id": item.tenant_id,
        "source_id": item.source_id,
        "canonical_key": item.canonical_key,
        "title": item.title,
        "status": item.status.value,
        "active_version_id": item.active_version_id,
        "ingestion_warnings": list(item.ingestion_warnings),
    }


def _user_payload(item: User) -> dict[str, object]:
    return {"id": item.id, "tenant_id": item.tenant_id, "email": item.email}


def _group_payload(item: Group) -> dict[str, object]:
    return {"id": item.id, "tenant_id": item.tenant_id, "name": item.name}


def _permission_payload(item: DocumentPermission) -> dict[str, object]:
    return {
        "document_id": item.document_id,
        "tenant_id": item.tenant_id,
        "principal_type": item.principal_type.value,
        "principal_id": item.principal_id,
        "permission": item.permission.value,
    }


def _version_payload(item: DocumentVersion) -> dict[str, object]:
    return {
        "id": item.id,
        "document_id": item.document_id,
        "version_number": item.version_number,
        "filename": item.filename,
        "content_sha256": item.content_sha256,
        "byte_size": item.byte_size,
        "status": item.status.value,
    }


def _job_payload(item: IngestionJob) -> dict[str, object]:
    return {
        "id": item.id,
        "tenant_id": item.tenant_id,
        "document_version_id": item.document_version_id,
        "run_id": item.run_id,
        "status": item.status.value,
        "attempts": item.attempts,
        "stage": item.stage,
        "next_attempt_at": item.next_attempt_at,
        "error": item.error,
    }


def _bad_request(code: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"code": code, "message": code})


def _payload_too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail={
            "code": "upload_too_large",
            "message": "Upload exceeds configured limit",
        },
    )


def _json_payload_too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail={
            "code": "json_body_too_large",
            "message": "JSON body exceeds the 65536-byte limit",
        },
    )
