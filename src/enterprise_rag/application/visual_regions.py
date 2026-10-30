"""Owned saved-citation page images. No OCR or character-to-pixel claims."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from enterprise_rag.application.search import SearchEvidence
from enterprise_rag.ingestion.pdf_render import (
    MAX_PDF_RENDER_INPUT_BYTES,
    MAX_RASTER_PAGE_BYTES,
    PDFRenderContainmentError,
    PDFRenderError,
    _renderer_revision,
    crop_raster_page,
    render_pdf_page,
)
from enterprise_rag.persistence.sqlite_query_history import SQLiteQueryHistory


class RegionNotFound(ValueError):
    """Missing, unauthorized, or no longer canonical evidence."""


class RegionUnavailable(RuntimeError):
    """Unavailable renderer, invalid stored bytes, or bounded capacity exhausted."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class PDFRegionRenderer:
    """Explicit local Poppler configuration; fixed existing renderer limits."""

    def __init__(self, command: tuple[str, ...], temp_root: Path):
        self.command = command
        self.temp_root = temp_root

    @property
    def revision(self):
        try:
            return _renderer_revision(self.command)
        except PDFRenderError:
            raise RegionUnavailable("PDF renderer unavailable") from None

    def render(self, binding, content, bbox):
        page = render_pdf_page(
            binding["run_id"],
            binding["document_version_id"],
            content,
            binding["page_number"],
            renderer_command=self.command,
            renderer_id="poppler",
            temp_root=self.temp_root,
        )
        image = (
            page
            if bbox is None
            else crop_raster_page(
                page,
                content,
                bbox,
                renderer_command=self.command,
                temp_root=self.temp_root,
            )
        )
        return page, image


class VisualRegionService:
    """Shared API/CLI path; object store remains the existing immutable store.

    SQLite reservations bound concurrent renders to one and each citation to
    eight images. Rendering and object IO happen outside metadata transactions.
    Expired reservations can be reclaimed; expired workers cannot publish.
    """

    def __init__(self, metadata, object_store, *, renderer=None):
        self.metadata = metadata
        self.store = object_store
        self.history = SQLiteQueryHistory(metadata)
        self.renderer = renderer

    def _binding(self, actor, query_id, index, connection=None):
        item = self.history.get(actor, query_id, connection=connection)
        if (
            item is None
            or item.result is None
            or type(index) is not int
            or not 1 <= index <= len(item.result.citations)
        ):
            raise RegionNotFound("Citation unavailable")
        cite = item.result.citations[index - 1]
        if len(cite.page_numbers) != 1 or not cite.filename.lower().endswith(".pdf"):
            raise RegionNotFound("Citation has no single PDF page")
        if connection is None:
            with self.metadata._connection() as current:
                current.execute("BEGIN")
                return self._binding(actor, query_id, index, current)
        version = connection.execute(
            "SELECT content_sha256,byte_size,object_key FROM document_versions "
            "WHERE id=? AND document_id=?",
            (cite.document_version_id, cite.document_id),
        ).fetchone()
        if (
            version is None
            or not 0 < version["byte_size"] <= MAX_PDF_RENDER_INPUT_BYTES
        ):
            raise RegionNotFound("Source unavailable")
        return {
            "query_run_id": query_id,
            "citation_index": index,
            "tenant_id": actor.tenant_id,
            "user_id": actor.user_id,
            "document_id": cite.document_id,
            "document_version_id": cite.document_version_id,
            "run_id": cite.run_id,
            "evidence_id": cite.evidence_id,
            "page_number": cite.page_numbers[0],
            "source_pdf_sha256": version["content_sha256"],
            "source_bytes": version["byte_size"],
            "source_object_key": version["object_key"],
            "citation_sha256": _digest(_json(asdict(cite)).encode()),
        }

    def _fence(self, actor, binding, connection=None):
        current = self._binding(
            actor, binding["query_run_id"], binding["citation_index"], connection
        )
        if current != binding:
            raise RegionNotFound("Source changed")

    def _source(self, actor, binding):
        self._fence(actor, binding)
        with self.store.open(binding["source_object_key"]) as stream:
            content = stream.read(MAX_PDF_RENDER_INPUT_BYTES + 1)
        self._fence(actor, binding)
        if (
            len(content) != binding["source_bytes"]
            or _digest(content) != binding["source_pdf_sha256"]
        ):
            raise RegionUnavailable("Source checksum mismatch")
        return content

    def _candidate_binding(self, actor, evidence: SearchEvidence):
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            raise RegionNotFound("Source unavailable")
        with self.metadata._connection() as connection:
            connection.execute("BEGIN")
            live = self.metadata.get_authorized_evidence(
                scope, evidence.chunk_id, connection=connection
            )
            if live is None or (
                live.chunk_id,
                live.document_id,
                live.document_version_id,
                live.run_id,
                live.title,
                live.filename,
                live.text,
                live.page_numbers,
            ) != (
                evidence.chunk_id,
                evidence.document_id,
                evidence.document_version_id,
                evidence.run_id,
                evidence.title,
                evidence.filename,
                evidence.text,
                evidence.page_numbers,
            ):
                raise RegionNotFound("Source changed or is no longer authorized")
            if len(live.page_numbers) != 1 or not live.filename.lower().endswith(
                ".pdf"
            ):
                raise RegionNotFound("Evidence has no single PDF page")
            version = connection.execute(
                """SELECT content_sha256,byte_size,object_key FROM document_versions
                   WHERE id=? AND document_id=?""",
                (live.document_version_id, live.document_id),
            ).fetchone()
            if (
                version is None
                or not 0 < version["byte_size"] <= MAX_PDF_RENDER_INPUT_BYTES
            ):
                raise RegionNotFound("Source unavailable")
            return {
                "tenant_id": actor.tenant_id,
                "user_id": actor.user_id,
                "evidence_id": live.chunk_id,
                "document_id": live.document_id,
                "document_version_id": live.document_version_id,
                "run_id": live.run_id,
                "page_number": live.page_numbers[0],
                "source_pdf_sha256": version["content_sha256"],
                "source_bytes": version["byte_size"],
                "source_object_key": version["object_key"],
            }

    def render_authorized_evidence_page(self, actor, evidence: SearchEvidence) -> bytes:
        """Render one currently authorized PDF page for a local visual provider."""
        if self.renderer is None:
            raise RegionUnavailable("PDF renderer is not configured")
        binding = self._candidate_binding(actor, evidence)
        with self.store.open(binding["source_object_key"]) as stream:
            content = stream.read(MAX_PDF_RENDER_INPUT_BYTES + 1)
        if (
            len(content) != binding["source_bytes"]
            or _digest(content) != binding["source_pdf_sha256"]
        ):
            raise RegionUnavailable("Source checksum mismatch")
        if self._candidate_binding(actor, evidence) != binding:
            raise RegionNotFound("Source changed during rendering")
        try:
            page, raster = self.renderer.render(binding, content, None)
        except (PDFRenderError, PDFRenderContainmentError):
            raise RegionUnavailable("PDF page rendering failed") from None
        if (
            page.run_id != binding["run_id"]
            or page.document_version_id != binding["document_version_id"]
            or page.page_number != binding["page_number"]
            or page.source_pdf_sha256 != binding["source_pdf_sha256"]
            or raster.run_id != page.run_id
            or raster.document_version_id != page.document_version_id
            or raster.page_number != page.page_number
            or raster.image_sha256 != page.image_sha256
            or raster.source_pdf_sha256 != page.source_pdf_sha256
            or raster.renderer_revision != page.renderer_revision
            or _digest(raster.image_bytes) != raster.image_sha256
        ):
            raise RegionUnavailable("Renderer provenance mismatch")
        if self._candidate_binding(actor, evidence) != binding:
            raise RegionNotFound("Source changed during rendering")
        return raster.image_bytes

    @staticmethod
    def _bbox(bbox):
        if bbox is None:
            return None
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(type(x) not in (float, int) for x in bbox)
        ):
            raise ValueError("bbox requires four finite normalized coordinates")
        try:
            x0, y0, x1, y1 = (float(x) for x in bbox)
        except OverflowError:
            raise ValueError("bbox requires finite normalized coordinates") from None
        if not all(math.isfinite(x) for x in (x0, y0, x1, y1)):
            raise ValueError("bbox requires finite normalized coordinates")
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("bbox must fit the normalized page")
        return (x0, y0, x1, y1)

    def create(self, actor, query_id, index, bbox=None):
        bbox = self._bbox(bbox)
        binding = self._binding(actor, query_id, index)
        if self.renderer is None:
            raise RegionUnavailable("PDF renderer is not configured")
        revision = self.renderer.revision
        cache_key = _digest(_json([binding, bbox, revision, "page-region-v1"]).encode())
        region_id = uuid.uuid4().hex
        answer_id = "answer-" + query_id
        with self.metadata._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._fence(actor, binding, connection)
            cached = connection.execute(
                "SELECT id FROM citation_visual_regions WHERE answer_id=? AND citation_index=? AND cache_key=?",
                (answer_id, index, cache_key),
            ).fetchone()
            if cached is None:
                now = time.time()
                connection.execute(
                    "DELETE FROM visual_region_reservations WHERE expires_at<?", (now,)
                )
                total = connection.execute(
                    "SELECT count(*) FROM visual_region_reservations"
                ).fetchone()[0]
                count = connection.execute(
                    "SELECT (SELECT count(*) FROM citation_visual_regions WHERE answer_id=? AND citation_index=?)"
                    "+(SELECT count(*) FROM visual_region_reservations WHERE answer_id=? AND citation_index=?)",
                    (answer_id, index, answer_id, index),
                ).fetchone()[0]
                pending = connection.execute(
                    "SELECT 1 FROM visual_region_reservations WHERE answer_id=? AND citation_index=? AND cache_key=?",
                    (answer_id, index, cache_key),
                ).fetchone()
                if total >= 1 or count >= 8 or pending:
                    raise RegionUnavailable(
                        "Visual region capacity is busy or exhausted"
                    )
                connection.execute(
                    "INSERT INTO visual_region_reservations VALUES (?,?,?,?,?)",
                    (region_id, answer_id, index, cache_key, now + 300),
                )
        if cached is not None:
            return self.read(actor, cached["id"])[0]
        object_key = f"documents/{binding['document_id']}/attempts/{region_id}.bin"
        registered = False
        try:
            content = self._source(actor, binding)
            self._fence(actor, binding)
            try:
                page, raster = self.renderer.render(binding, content, bbox)
            except (PDFRenderError, PDFRenderContainmentError):
                raise RegionUnavailable("PDF page rendering failed") from None
            if (
                page.run_id != binding["run_id"]
                or page.document_version_id != binding["document_version_id"]
                or page.source_pdf_sha256 != binding["source_pdf_sha256"]
                or page.page_number != binding["page_number"]
                or page.renderer_revision != revision
                or raster.renderer_revision != revision
                or raster.run_id != page.run_id
                or raster.document_version_id != page.document_version_id
                or raster.source_pdf_sha256 != page.source_pdf_sha256
                or raster.page_number != page.page_number
                or (
                    bbox is not None
                    and (
                        raster.normalized_bbox != bbox
                        or raster.page_image_sha256 != page.image_sha256
                    )
                )
            ):
                raise RegionUnavailable("Renderer provenance mismatch")
            self._fence(actor, binding)
            png = raster.image_bytes
            if len(png) > MAX_RASTER_PAGE_BYTES or _digest(png) != raster.image_sha256:
                raise RegionUnavailable("Renderer checksum mismatch")
            manifest = {
                "id": region_id,
                "binding": binding,
                "kind": "wholepage" if bbox is None else "explicitcrop",
                "normalized_bbox": bbox,
                "pixel_bounds": None if bbox is None else raster.pixel_bounds,
                "width": raster.width,
                "height": raster.height,
                "page_width": page.width,
                "page_height": page.height,
                "applied_rotation_degrees": page.applied_rotation_degrees,
                "renderer_id": page.renderer_id,
                "renderer_revision": revision,
                "page_image_sha256": page.image_sha256,
                "image_sha256": raster.image_sha256,
                "image_bytes": len(png),
                "object_key": object_key,
            }
            encoded = _json(manifest)
            now = time.time()
            self.metadata.register_upload_attempt(
                actor.tenant_id,
                binding["document_id"],
                object_key,
                created_at=now,
                expires_at=now + 300,
            )
            registered = True
            self.store.put_stream(
                object_key, BytesIO(png), max_bytes=MAX_RASTER_PAGE_BYTES
            )
            with self.metadata._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._fence(actor, binding, connection)
                self.metadata._assert_object_key_available_for_upload(
                    connection,
                    object_key,
                    document_id=binding["document_id"],
                    allow_existing_same_document=True,
                )
                reserved = connection.execute(
                    "SELECT 1 FROM visual_region_reservations WHERE id=? AND expires_at>?",
                    (region_id, time.time()),
                ).fetchone()
                updated = connection.execute(
                    "UPDATE upload_attempts SET status='committed' WHERE object_key=? AND tenant_id=? AND status='pending'",
                    (object_key, actor.tenant_id),
                ).rowcount
                if reserved is None or updated != 1:
                    raise RegionUnavailable("Visual publication reservation expired")
                connection.execute(
                    "INSERT INTO citation_visual_regions VALUES (?,?,?,?,?,?,?,?)",
                    (
                        region_id,
                        answer_id,
                        index,
                        binding["document_id"],
                        cache_key,
                        object_key,
                        encoded,
                        _digest(encoded.encode()),
                    ),
                )
                connection.execute(
                    "DELETE FROM visual_region_reservations WHERE id=?", (region_id,)
                )
            return self.read(actor, region_id)[0]
        finally:
            if registered:
                self.metadata.finish_upload_attempt(
                    actor.tenant_id, object_key, status="abandoned", now=time.time()
                )
            with self.metadata._connection() as connection:
                connection.execute(
                    "DELETE FROM visual_region_reservations WHERE id=?", (region_id,)
                )

    def read(self, actor, region_id):
        with self.metadata._connection() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT * FROM citation_visual_regions WHERE id=?", (region_id,)
            ).fetchone()
            if row is None:
                raise RegionNotFound("Region unavailable")
            binding = self._binding(
                actor,
                row["answer_id"].removeprefix("answer-"),
                row["citation_index"],
                connection,
            )
            if _digest(row["manifest"].encode()) != row["manifest_sha256"]:
                raise RegionUnavailable("Region manifest checksum mismatch")
            manifest = json.loads(row["manifest"])
            if (
                manifest["binding"] != binding
                or manifest["id"] != region_id
                or manifest["object_key"] != row["object_key"]
                or binding["document_id"] != row["document_id"]
            ):
                raise RegionUnavailable("Region manifest binding mismatch")
        self._source(actor, binding)
        with self.store.open(row["object_key"]) as stream:
            png = stream.read(MAX_RASTER_PAGE_BYTES + 1)
        self._fence(actor, binding)
        if (
            len(png) > MAX_RASTER_PAGE_BYTES
            or len(png) != manifest["image_bytes"]
            or _digest(png) != manifest["image_sha256"]
        ):
            raise RegionUnavailable("Region image checksum mismatch")
        public = {
            key: value
            for key, value in manifest.items()
            if key not in {"object_key", "binding"}
        }
        public.update(
            {
                key: value
                for key, value in binding.items()
                if key
                not in {
                    "source_object_key",
                    "source_bytes",
                    "tenant_id",
                    "user_id",
                    "citation_sha256",
                }
            }
        )
        public["image_url"] = f"/v1/regions/{region_id}/image"
        return public, png

    def read_ocr_regions(self, actor, query_id, index):
        """Return OCR boxes intersecting one exact, saved PDF citation quote.

        The citation offsets are Unicode code-point offsets into its canonical
        chunk. A unique exact chunk occurrence binds those offsets to the
        untouched OCR page text; ambiguous or stale bindings fail closed.
        """
        binding = self._binding(actor, query_id, index)
        item = self.history.get(actor, query_id)
        if (
            item is None
            or item.result is None
            or type(index) is not int
            or not 1 <= index <= len(item.result.citations)
        ):
            raise RegionNotFound("Citation unavailable")
        citation = item.result.citations[index - 1]
        if (
            citation.evidence_id != binding["evidence_id"]
            or citation.document_id != binding["document_id"]
            or citation.document_version_id != binding["document_version_id"]
            or citation.run_id != binding["run_id"]
            or citation.page_numbers != (binding["page_number"],)
            or citation.offset_unit != "unicode_code_points"
            or _digest(_json(asdict(citation)).encode()) != binding["citation_sha256"]
        ):
            raise RegionNotFound("Citation unavailable")

        self._fence(actor, binding)
        try:
            stored = self.metadata.get_text_artifacts(binding["run_id"])
        except (TypeError, ValueError, KeyError):
            raise RegionUnavailable("OCR evidence is invalid") from None
        self._fence(actor, binding)
        if stored is None:
            raise RegionNotFound("OCR evidence unavailable")
        parsed, chunks = stored
        if (
            parsed.run_id != binding["run_id"]
            or parsed.content_sha256 != binding["source_pdf_sha256"]
        ):
            raise RegionUnavailable("OCR evidence source binding mismatch")

        page_number = binding["page_number"]
        evidence_items = [
            evidence
            for evidence in parsed.ocr_pages
            if evidence.page_number == page_number
        ]
        if not evidence_items:
            raise RegionNotFound("OCR evidence unavailable")
        if len(evidence_items) != 1:
            raise RegionUnavailable("OCR page evidence is ambiguous")
        evidence = evidence_items[0]
        result = evidence.result
        if (
            result.run_id != binding["run_id"]
            or result.document_version_id != binding["document_version_id"]
            or result.source_pdf_sha256 != binding["source_pdf_sha256"]
            or result.page_number != page_number
            or result.image_sha256 != evidence.page_image_sha256
        ):
            raise RegionUnavailable("OCR page provenance mismatch")

        matching_chunks = [
            chunk for chunk in chunks if chunk.id == citation.evidence_id
        ]
        if len(matching_chunks) != 1:
            raise RegionNotFound("Citation evidence unavailable")
        chunk = matching_chunks[0]
        page_elements = [
            element
            for element in parsed.elements
            if element.page_number == page_number
            and element.element_type == "ocr_page_text"
        ]
        if len(page_elements) != 1:
            raise RegionUnavailable("OCR text element is ambiguous")
        element = page_elements[0]
        if (
            element.text != result.raw_text
            or element.original_text != result.raw_text
            or not chunk.element_ids
            or set(chunk.element_ids) != {element.id}
            or chunk.run_id != binding["run_id"]
            or chunk.page_numbers != (page_number,)
            or not chunk.text
            or _digest(chunk.text.encode("utf-8")) != chunk.content_sha256
        ):
            raise RegionUnavailable("OCR chunk provenance mismatch")

        chunk_start = result.raw_text.find(chunk.text)
        if chunk_start < 0 or result.raw_text.find(chunk.text, chunk_start + 1) >= 0:
            raise RegionUnavailable("OCR chunk location is ambiguous")
        if (
            type(citation.chunk_start_char) is not int
            or type(citation.chunk_end_char) is not int
            or not 0
            <= citation.chunk_start_char
            < citation.chunk_end_char
            <= len(chunk.text)
        ):
            raise RegionUnavailable("Citation offsets do not match OCR text")
        quote_start = chunk_start + citation.chunk_start_char
        quote_end = chunk_start + citation.chunk_end_char
        if result.raw_text[quote_start:quote_end] != citation.quote:
            raise RegionUnavailable("Citation offsets do not match OCR text")

        regions = [
            asdict(region)
            for region in result.regions
            if region.start_char < quote_end and region.end_char > quote_start
        ]
        if not regions:
            raise RegionUnavailable("Citation has no OCR regions")
        self._fence(actor, binding)
        return {
            "query_run_id": query_id,
            "citation_index": index,
            "evidence_id": citation.evidence_id,
            "document_id": binding["document_id"],
            "document_version_id": binding["document_version_id"],
            "run_id": binding["run_id"],
            "page_number": page_number,
            "source_pdf_sha256": binding["source_pdf_sha256"],
            "quote": citation.quote,
            "quote_span": {
                "start_char": quote_start,
                "end_char": quote_end,
                "offset_unit": "unicode_code_points",
            },
            "page_image_sha256": evidence.page_image_sha256,
            "page_width": evidence.page_width,
            "page_height": evidence.page_height,
            "applied_rotation_degrees": evidence.applied_rotation_degrees,
            "renderer_id": evidence.renderer_id,
            "renderer_revision": evidence.renderer_revision,
            "provider_id": result.provider_id,
            "provider_revision": result.provider_revision,
            "regions": regions,
        }

    def history_payload(self, actor, query_id):
        item = self.history.get(actor, query_id)
        if item is None:
            raise RegionNotFound("Query unavailable")
        payload = asdict(item)
        if item.result is not None:
            for index, citation in enumerate(payload["result"]["citations"], 1):
                with self.metadata._connection() as connection:
                    rows = connection.execute(
                        "SELECT id FROM citation_visual_regions WHERE answer_id=? AND citation_index=? ORDER BY id LIMIT 8",
                        ("answer-" + query_id, index),
                    ).fetchall()
                if rows:
                    citation["visual_regions"] = [
                        self.read(actor, row["id"])[0] for row in rows
                    ]
        if self.history.get(actor, query_id) != item:
            raise RegionNotFound("Query changed")
        return payload
