"""Bounded, page-scoped PDF rasterization for the OCR contract."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import zlib
from binascii import crc32
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil, floor, isfinite
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic, sleep
from typing import Final

from enterprise_rag.domain.documents.ids import sha256_bytes
from enterprise_rag.ingestion.ocr import (
    MAX_RASTER_PAGE_BYTES,
    MAX_RASTER_PAGE_PIXELS,
    PNG_SIGNATURE,
    OCRContractError,
    RasterPage,
)

DEFAULT_PDF_RENDER_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_PDF_RENDER_MEMORY_BYTES: Final = 1024**3
MAX_PDF_RENDER_INPUT_BYTES: Final = 25 * 1024 * 1024
MAX_PDF_RENDER_PAGES: Final = 100
MAX_PDF_RENDER_TIMEOUT_SECONDS: Final = 60.0
MAX_PDF_RENDER_MEMORY_BYTES: Final = 1024**3
MAX_PDF_RENDER_EDGE: Final = 3200
MIN_PDF_RENDER_MEMORY_BYTES: Final = 64 * 1024 * 1024
RENDER_CLEANUP_SECONDS: Final = 2.0
PNG_OUTPUT_NAME: Final = "page.png"
MAX_RENDERER_BINARY_BYTES: Final = 512 * 1024 * 1024


class PDFRenderError(ValueError):
    """An input, renderer, or output violates the bounded render contract."""


class PDFRenderContainmentError(RuntimeError):
    """The isolated renderer process tree could not be confirmed stopped."""


@dataclass(frozen=True, slots=True)
class RasterCrop:
    """A rendered pixel crop tied to its original PDF page and page image."""

    run_id: str
    document_version_id: str
    source_pdf_sha256: str
    page_number: int
    page_image_sha256: str
    normalized_bbox: tuple[float, float, float, float]
    pixel_bounds: tuple[int, int, int, int]
    image_bytes: bytes
    width: int
    height: int
    renderer_id: str
    renderer_revision: str
    image_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise PDFRenderError("Invalid run id")
        if not isinstance(self.document_version_id, str) or not (
            self.document_version_id.strip()
        ):
            raise PDFRenderError("Invalid document version id")
        for value in (self.source_pdf_sha256, self.page_image_sha256):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise PDFRenderError("Invalid crop provenance digest")
        if type(self.page_number) is not int or self.page_number < 1:
            raise PDFRenderError("Crop page number must be positive")
        if (
            not isinstance(self.normalized_bbox, tuple)
            or len(self.normalized_bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in self.normalized_bbox
            )
        ):
            raise PDFRenderError("Crop bbox must contain four numeric coordinates")
        try:
            bbox = tuple(float(value) for value in self.normalized_bbox)
        except OverflowError as exc:
            raise PDFRenderError("Crop bbox must fit the normalized page") from exc
        x0, y0, x1, y1 = bbox
        if not all(isfinite(value) for value in bbox) or not (
            0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        ):
            raise PDFRenderError("Crop bbox must fit the normalized page")
        if (
            not isinstance(self.pixel_bounds, tuple)
            or len(self.pixel_bounds) != 4
            or any(type(value) is not int for value in self.pixel_bounds)
        ):
            raise PDFRenderError("Crop pixel bounds must contain four integers")
        px0, py0, px1, py1 = self.pixel_bounds
        if px0 < 0 or py0 < 0 or px1 <= px0 or py1 <= py0:
            raise PDFRenderError("Crop pixel bounds must be non-empty")
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or (self.width, self.height) != (px1 - px0, py1 - py0)
        ):
            raise PDFRenderError("Crop dimensions do not match pixel bounds")
        if not isinstance(self.image_bytes, bytes) or not (
            33 <= len(self.image_bytes) <= MAX_RASTER_PAGE_BYTES
        ):
            raise PDFRenderError("Crop image must contain PNG bytes")
        if not isinstance(self.renderer_id, str) or not self.renderer_id.strip():
            raise PDFRenderError("Invalid renderer id")
        if (
            not isinstance(self.renderer_revision, str)
            or len(self.renderer_revision) != 64
            or any(char not in "0123456789abcdef" for char in self.renderer_revision)
        ):
            raise PDFRenderError("Invalid renderer revision")
        if _validate_poppler_png(self.image_bytes) != (self.width, self.height):
            raise PDFRenderError("Crop dimensions do not match its PNG")
        object.__setattr__(self, "normalized_bbox", bbox)
        object.__setattr__(self, "image_sha256", sha256_bytes(self.image_bytes))


def render_pdf_page(
    run_id: str,
    document_version_id: str,
    content: bytes,
    page_number: int,
    *,
    renderer_command: tuple[str, ...],
    renderer_id: str,
    temp_root: str | Path,
    timeout_seconds: float = DEFAULT_PDF_RENDER_TIMEOUT_SECONDS,
    memory_bytes: int = DEFAULT_PDF_RENDER_MEMORY_BYTES,
) -> RasterPage:
    """Render exactly one PDF page inside a bounded, owned process tree.

    ``renderer_command`` is an explicit argv prefix, normally the absolute path
    to ``pdftoppm``. No shell or PATH lookup is used. ``temp_root`` must be an
    existing directory selected by the caller, preferably outside the repo.
    """

    _validate_request(
        run_id,
        document_version_id,
        content,
        page_number,
        renderer_command,
        renderer_id,
        temp_root,
        timeout_seconds,
        memory_bytes,
    )
    renderer_revision = _renderer_revision(renderer_command)
    try:
        resolved_temp_root = Path(temp_root).resolve(strict=True)
    except OSError as exc:
        raise PDFRenderError("pdf_render_temp_root_unavailable") from exc
    source_digest = sha256_bytes(content)
    try:
        with tempfile.TemporaryDirectory(
            prefix="pdf-page-render-", dir=resolved_temp_root
        ) as temp_name:
            temp_dir = Path(temp_name)
            pdf_path = temp_dir / "source.pdf"
            output_path = temp_dir / PNG_OUTPUT_NAME
            output_prefix = temp_dir / "page"
            pdf_path.write_bytes(content)
            rotation = _run_isolated_renderer(
                pdf_path,
                output_prefix,
                page_number,
                renderer_command,
                timeout_seconds=timeout_seconds,
                memory_bytes=memory_bytes,
            )
            if output_path.is_symlink() or not output_path.is_file():
                raise PDFRenderError("pdf_render_output_missing")
            output_size = output_path.stat().st_size
            if not 33 <= output_size <= MAX_RASTER_PAGE_BYTES:
                raise PDFRenderError("pdf_render_output_size_limit")
            image_bytes = output_path.read_bytes()
            width, height = _validate_poppler_png(image_bytes)
            return RasterPage(
                run_id=run_id,
                document_version_id=document_version_id,
                source_pdf_sha256=source_digest,
                page_number=page_number,
                image_bytes=image_bytes,
                width=width,
                height=height,
                applied_rotation_degrees=rotation,
                renderer_id=renderer_id,
                renderer_revision=renderer_revision,
            )
    except OCRContractError as exc:
        raise PDFRenderError("pdf_render_invalid_output") from exc
    except OSError as exc:
        raise PDFRenderError("pdf_render_io_failed") from exc


def crop_raster_page(
    page: RasterPage,
    content: bytes,
    normalized_bbox: tuple[float, float, float, float],
    *,
    renderer_command: tuple[str, ...],
    temp_root: str | Path,
    timeout_seconds: float = DEFAULT_PDF_RENDER_TIMEOUT_SECONDS,
    memory_bytes: int = DEFAULT_PDF_RENDER_MEMORY_BYTES,
) -> RasterCrop:
    """Render a normalized crop from the same source page and renderer build."""

    if not isinstance(page, RasterPage):
        raise PDFRenderError("Expected a raster page")
    _validate_request(
        page.run_id,
        page.document_version_id,
        content,
        page.page_number,
        renderer_command,
        page.renderer_id,
        temp_root,
        timeout_seconds,
        memory_bytes,
    )
    if (
        not isinstance(content, bytes)
        or sha256_bytes(content) != page.source_pdf_sha256
    ):
        raise PDFRenderError("pdf_render_source_page_mismatch")
    if _renderer_revision(renderer_command) != page.renderer_revision:
        raise PDFRenderError("pdf_renderer_identity_mismatch")
    bbox, pixel_bounds = _crop_pixel_bounds(page, normalized_bbox)
    try:
        resolved_temp_root = Path(temp_root).resolve(strict=True)
    except OSError as exc:
        raise PDFRenderError("pdf_render_temp_root_unavailable") from exc
    try:
        with tempfile.TemporaryDirectory(
            prefix="pdf-region-render-", dir=resolved_temp_root
        ) as temp_name:
            temp_dir = Path(temp_name)
            pdf_path = temp_dir / "source.pdf"
            output_path = temp_dir / PNG_OUTPUT_NAME
            output_prefix = temp_dir / "page"
            pdf_path.write_bytes(content)
            rotation = _run_isolated_renderer(
                pdf_path,
                output_prefix,
                page.page_number,
                renderer_command,
                crop_box=pixel_bounds,
                timeout_seconds=timeout_seconds,
                memory_bytes=memory_bytes,
            )
            if rotation != page.applied_rotation_degrees:
                raise PDFRenderError("pdf_render_source_page_mismatch")
            if output_path.is_symlink() or not output_path.is_file():
                raise PDFRenderError("pdf_render_output_missing")
            output_size = output_path.stat().st_size
            if not 33 <= output_size <= MAX_RASTER_PAGE_BYTES:
                raise PDFRenderError("pdf_render_output_size_limit")
            image_bytes = output_path.read_bytes()
            width, height = _validate_poppler_png(image_bytes)
            x0, y0, x1, y1 = pixel_bounds
            if (width, height) != (x1 - x0, y1 - y0):
                raise PDFRenderError("pdf_render_crop_dimensions_mismatch")
            return RasterCrop(
                run_id=page.run_id,
                document_version_id=page.document_version_id,
                source_pdf_sha256=page.source_pdf_sha256,
                page_number=page.page_number,
                page_image_sha256=page.image_sha256,
                normalized_bbox=bbox,
                pixel_bounds=pixel_bounds,
                image_bytes=image_bytes,
                width=width,
                height=height,
                renderer_id=page.renderer_id,
                renderer_revision=page.renderer_revision,
            )
    except OCRContractError as exc:
        raise PDFRenderError("pdf_render_invalid_output") from exc
    except OSError as exc:
        raise PDFRenderError("pdf_render_io_failed") from exc


def _crop_pixel_bounds(
    page: RasterPage, bbox: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise PDFRenderError("Crop bbox must be an (x0, y0, x1, y1) tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox
    ):
        raise PDFRenderError("Crop bbox coordinates must be numeric")
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except OverflowError as exc:
        raise PDFRenderError("Crop bbox must fit the normalized page") from exc
    if not all(isfinite(value) for value in (x0, y0, x1, y1)) or not (
        0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    ):
        raise PDFRenderError("Crop bbox must fit the normalized page")
    bounds = (
        floor(x0 * page.width),
        floor(y0 * page.height),
        ceil(x1 * page.width),
        ceil(y1 * page.height),
    )
    if (
        page.width > MAX_PDF_RENDER_EDGE
        or page.height > MAX_PDF_RENDER_EDGE
        or bounds[0] < 0
        or bounds[1] < 0
        or bounds[2] > page.width
        or bounds[3] > page.height
        or bounds[2] <= bounds[0]
        or bounds[3] <= bounds[1]
    ):
        raise PDFRenderError("pdf_render_crop_bounds_invalid")
    return (x0, y0, x1, y1), bounds


def _validate_request(
    run_id: str,
    document_version_id: str,
    content: bytes,
    page_number: int,
    renderer_command: tuple[str, ...],
    renderer_id: str,
    temp_root: str | Path,
    timeout_seconds: float,
    memory_bytes: int,
) -> None:
    for value, label in (
        (run_id, "run id"),
        (document_version_id, "document version id"),
        (renderer_id, "renderer id"),
    ):
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise PDFRenderError(f"Invalid {label}")
    if not isinstance(content, bytes):
        raise PDFRenderError("PDF content must be bytes")
    if len(content) > MAX_PDF_RENDER_INPUT_BYTES:
        raise PDFRenderError("pdf_render_input_size_limit")
    if not content.startswith(b"%PDF-"):
        raise PDFRenderError("pdf_render_invalid_signature")
    if type(page_number) is not int or page_number < 1:
        raise PDFRenderError("PDF page number must be positive")
    if not isinstance(renderer_command, tuple) or not 1 <= len(renderer_command) <= 8:
        raise PDFRenderError("Renderer command must be a non-empty argv tuple")
    if any(
        not isinstance(part, str) or not part or "\x00" in part
        for part in renderer_command
    ):
        raise PDFRenderError("Renderer command must be a non-empty argv tuple")
    try:
        command_bytes = sum(len(part.encode("utf-8")) for part in renderer_command)
    except UnicodeEncodeError as exc:
        raise PDFRenderError("Renderer command contains invalid text") from exc
    if command_bytes > 16 * 1024:
        raise PDFRenderError("Renderer command exceeds the argument size limit")
    executable = Path(renderer_command[0])
    if not executable.is_absolute():
        raise PDFRenderError("Renderer executable path must be absolute")
    try:
        resolved_executable = executable.resolve(strict=True)
    except OSError as exc:
        raise PDFRenderError("pdf_renderer_unavailable") from exc
    if not resolved_executable.is_file():
        raise PDFRenderError("pdf_renderer_unavailable")
    try:
        executable_size = resolved_executable.stat().st_size
    except OSError as exc:
        raise PDFRenderError("pdf_renderer_unavailable") from exc
    if executable_size > MAX_RENDERER_BINARY_BYTES:
        raise PDFRenderError("pdf_renderer_binary_size_limit")
    if not is_finite_positive(timeout_seconds) or (
        timeout_seconds > MAX_PDF_RENDER_TIMEOUT_SECONDS
    ):
        raise PDFRenderError("PDF render timeout must be in (0, 60] seconds")
    if (
        type(memory_bytes) is not int
        or not MIN_PDF_RENDER_MEMORY_BYTES
        <= memory_bytes
        <= MAX_PDF_RENDER_MEMORY_BYTES
    ):
        raise PDFRenderError("PDF render memory limit must be 64 MiB..1 GiB")
    if not isinstance(temp_root, (str, Path)):
        raise PDFRenderError("PDF render temporary root must be a path")
    try:
        resolved_root = Path(temp_root).resolve(strict=True)
    except OSError as exc:
        raise PDFRenderError("pdf_render_temp_root_unavailable") from exc
    if not resolved_root.is_dir():
        raise PDFRenderError("pdf_render_temp_root_unavailable")


def _renderer_revision(renderer_command: tuple[str, ...]) -> str:
    try:
        executable = Path(renderer_command[0]).resolve(strict=True)
        if executable.stat().st_size > MAX_RENDERER_BINARY_BYTES:
            raise PDFRenderError("pdf_renderer_binary_size_limit")
        binary_digest = sha256()
        with executable.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                binary_digest.update(block)
        digest = sha256(b"pdf-renderer-command-v1\0")
        digest.update(binary_digest.digest())
        for argument in renderer_command[1:]:
            encoded = argument.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
    except (OSError, UnicodeEncodeError) as exc:
        raise PDFRenderError("pdf_renderer_identity_unavailable") from exc
    return digest.hexdigest()


def is_finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and value < float("inf")
    )


def _run_isolated_renderer(
    pdf_path: Path,
    output_prefix: Path,
    page_number: int,
    renderer_command: tuple[str, ...],
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    timeout_seconds: float,
    memory_bytes: int,
) -> int:
    context = mp.get_context("spawn")
    result_reader, result_sender = context.Pipe(duplex=False)
    ready_reader, ready_sender = context.Pipe(duplex=False)
    gate_reader, gate_writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_pdf_render_child,
        args=(
            result_sender,
            ready_sender,
            gate_reader,
            str(pdf_path),
            str(output_prefix),
            page_number,
            renderer_command,
            crop_box,
            timeout_seconds,
            memory_bytes,
        ),
    )
    process.daemon = True
    job = None
    started = False
    process_group_ready = False
    process_pid: int | None = None
    deadline = monotonic() + timeout_seconds
    try:
        if os.name == "nt":
            try:
                from enterprise_rag.workers.processes import _WindowsJob

                job = _WindowsJob(memory_bytes=memory_bytes)
            except OSError as exc:
                raise PDFRenderError("pdf_render_resource_limit_unavailable") from exc
        process.start()
        started = True
        process_pid = process.pid
        result_sender.close()
        ready_sender.close()
        gate_reader.close()
        remaining = max(0.0, deadline - monotonic())
        if not ready_reader.poll(remaining):
            raise PDFRenderError("pdf_render_timeout")
        if ready_reader.recv_bytes(maxlength=16) != b"ready":
            raise PDFRenderError("pdf_render_failed")
        process_group_ready = os.name != "nt"
        if job is not None:
            try:
                job.assign_handle(process.sentinel)
            except OSError as exc:
                raise PDFRenderError("pdf_render_resource_limit_unavailable") from exc
        if monotonic() >= deadline:
            raise PDFRenderError("pdf_render_timeout")
        gate_writer.send_bytes(b"1")
        gate_writer.close()
        remaining = max(0.0, deadline - monotonic())
        if not result_reader.poll(remaining):
            raise PDFRenderError("pdf_render_timeout")
        try:
            outcome = json.loads(result_reader.recv_bytes(maxlength=1024))
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            raise PDFRenderError("pdf_render_failed") from exc
        process.join(max(0.0, deadline - monotonic()))
        if process.is_alive():
            raise PDFRenderError("pdf_render_timeout")
        if (
            not isinstance(outcome, dict)
            or set(outcome) != {"status", "rotation"}
            or outcome.get("status") != "ok"
            or type(outcome.get("rotation")) is not int
            or outcome["rotation"] not in {0, 90, 180, 270}
        ):
            code = outcome.get("status") if isinstance(outcome, dict) else None
            if isinstance(code, str) and code.startswith("pdf_"):
                raise PDFRenderError(code)
            raise PDFRenderError("pdf_render_failed")
        return outcome["rotation"]
    except (EOFError, OSError) as exc:
        raise PDFRenderError("pdf_render_failed") from exc
    finally:
        result_sender.close()
        ready_sender.close()
        gate_reader.close()
        gate_writer.close()
        ready_reader.close()
        cleanup_deadline = monotonic() + RENDER_CLEANUP_SECONDS
        confirmed = True
        if job is not None:
            try:
                job.close(timeout_seconds=max(0.0, cleanup_deadline - monotonic()))
            except OSError:
                confirmed = False
        if started:
            try:
                if os.name != "nt" and process_group_ready:
                    assert process_pid is not None
                    with suppress(ProcessLookupError):
                        _kill_process_group(process_pid)
                if process.is_alive():
                    process.terminate()
                process.join(max(0.0, cleanup_deadline - monotonic()))
                if process.is_alive():
                    process.kill()
                    process.join(max(0.0, cleanup_deadline - monotonic()))
                if process.is_alive():
                    confirmed = False
                else:
                    process.close()
                if os.name != "nt" and process_group_ready:
                    assert process_pid is not None
                    _confirm_process_group_exit(process_pid, cleanup_deadline)
            except (OSError, ValueError, PDFRenderContainmentError):
                confirmed = False
        else:
            process.close()
        result_reader.close()
        if not confirmed:
            raise PDFRenderContainmentError("pdf_render_cleanup_unconfirmed")


def _pdf_render_child(
    result: Connection,
    ready: Connection,
    gate: Connection,
    pdf_path: str,
    output_prefix: str,
    page_number: int,
    renderer_command: tuple[str, ...],
    crop_box: tuple[int, int, int, int] | None,
    timeout_seconds: float,
    memory_bytes: int,
) -> None:
    outcome: dict[str, str | int] = {"status": "pdf_render_failed", "rotation": -1}
    try:
        if os.name != "nt":
            _start_posix_session()
        ready.send_bytes(b"ready")
        if not gate.poll(30.0) or gate.recv_bytes(maxlength=1) != b"1":
            return
        gate.close()
        if os.name != "nt":
            _set_posix_memory_limit(memory_bytes)
        rotation = _inspect_pdf_page(pdf_path, page_number)
        output_file = Path(output_prefix + ".png")
        crop_args: tuple[str, ...] = ()
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                raise PDFRenderError("pdf_render_crop_bounds_invalid")
            crop_args = (
                "-x",
                str(x0),
                "-y",
                str(y0),
                "-W",
                str(x1 - x0),
                "-H",
                str(y1 - y0),
            )
        command = (
            *renderer_command,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-scale-to",
            str(MAX_PDF_RENDER_EDGE),
            *crop_args,
            "-gray",
            "-cropbox",
            "-png",
            pdf_path,
            output_prefix,
        )
        child = subprocess.Popen(
            command,
            cwd=str(Path(output_prefix).parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        try:
            return_code = child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            raise PDFRenderError("pdf_render_timeout") from None
        if return_code != 0:
            raise PDFRenderError("pdf_render_failed")
        if output_file.is_symlink() or not output_file.is_file():
            raise PDFRenderError("pdf_render_output_missing")
        if output_file.stat().st_size > MAX_RASTER_PAGE_BYTES:
            raise PDFRenderError("pdf_render_output_size_limit")
        outcome = {"status": "ok", "rotation": rotation}
    except MemoryError:
        outcome = {"status": "pdf_render_memory_limit", "rotation": -1}
    except PDFRenderError as exc:
        outcome = {"status": str(exc), "rotation": -1}
    except Exception:
        outcome = {"status": "pdf_render_failed", "rotation": -1}
    try:
        result.send_bytes(json.dumps(outcome, separators=(",", ":")).encode("ascii"))
    except (BrokenPipeError, OSError):
        pass
    finally:
        result.close()
        ready.close()
        gate.close()


def _inspect_pdf_page(pdf_path: str, page_number: int) -> int:
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path, strict=False)
        if reader.is_encrypted:
            raise PDFRenderError("pdf_render_encrypted")
        page_count = len(reader.pages)
        if page_count == 0:
            raise PDFRenderError("pdf_render_empty")
        if page_count > MAX_PDF_RENDER_PAGES:
            raise PDFRenderError("pdf_render_page_limit")
        if page_number > page_count:
            raise PDFRenderError("pdf_render_page_not_found")
        rotation = reader.pages[page_number - 1].rotation % 360
        if rotation not in {0, 90, 180, 270}:
            raise PDFRenderError("pdf_render_invalid_rotation")
        return rotation
    except PDFRenderError:
        raise
    except Exception as exc:
        raise PDFRenderError("pdf_render_corrupt") from exc


def _set_posix_memory_limit(memory_bytes: int) -> None:
    if not sys.platform.startswith("linux"):
        raise PDFRenderError("pdf_render_resource_limit_unavailable")
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limits = [memory_bytes]
        limits.extend(
            value for value in (soft, hard) if value != resource.RLIM_INFINITY
        )
        limit = min(limits)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError) as exc:
        raise PDFRenderError("pdf_render_resource_limit_unavailable") from exc


def _validate_poppler_png(image: bytes) -> tuple[int, int]:
    if not image.startswith(PNG_SIGNATURE):
        raise PDFRenderError("pdf_render_invalid_png")
    offset = len(PNG_SIGNATURE)
    width = height = 0
    seen_header = False
    seen_data = False
    seen_end = False
    decompressor = zlib.decompressobj()
    decompressed_bytes = 0
    expected_bytes = 0
    while offset < len(image):
        if offset + 12 > len(image):
            raise PDFRenderError("pdf_render_invalid_png")
        chunk_length = int.from_bytes(image[offset : offset + 4], "big")
        end = offset + 12 + chunk_length
        if end > len(image):
            raise PDFRenderError("pdf_render_invalid_png")
        name = image[offset + 4 : offset + 8]
        chunk_data = image[offset + 8 : offset + 8 + chunk_length]
        supplied_crc = int.from_bytes(image[offset + 8 + chunk_length : end], "big")
        if (crc32(name + chunk_data) & 0xFFFFFFFF) != supplied_crc:
            raise PDFRenderError("pdf_render_invalid_png")
        if not seen_header:
            if name != b"IHDR" or chunk_length != 13:
                raise PDFRenderError("pdf_render_invalid_png")
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            if (
                width < 1
                or height < 1
                or width * height > MAX_RASTER_PAGE_PIXELS
                or width > MAX_PDF_RENDER_EDGE
                or height > MAX_PDF_RENDER_EDGE
                or chunk_data[8] != 8
                or chunk_data[9] not in {0, 2}
                or chunk_data[10:] != bytes((0, 0, 0))
            ):
                raise PDFRenderError("pdf_render_invalid_png")
            channels = 1 if chunk_data[9] == 0 else 3
            expected_bytes = (width * channels + 1) * height
            seen_header = True
        elif name == b"IHDR":
            raise PDFRenderError("pdf_render_invalid_png")
        elif name == b"IDAT":
            if seen_end:
                raise PDFRenderError("pdf_render_invalid_png")
            seen_data = True
            try:
                decoded = decompressor.decompress(
                    chunk_data, expected_bytes - decompressed_bytes + 1
                )
            except zlib.error as exc:
                raise PDFRenderError("pdf_render_invalid_png") from exc
            decompressed_bytes += len(decoded)
            if decompressed_bytes > expected_bytes:
                raise PDFRenderError("pdf_render_invalid_png")
        elif name == b"IEND":
            if chunk_length != 0 or not seen_data or seen_end:
                raise PDFRenderError("pdf_render_invalid_png")
            try:
                decoded = decompressor.flush()
            except zlib.error as exc:
                raise PDFRenderError("pdf_render_invalid_png") from exc
            decompressed_bytes += len(decoded)
            if (
                decompressed_bytes != expected_bytes
                or not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                raise PDFRenderError("pdf_render_invalid_png")
            seen_end = True
            if end != len(image):
                raise PDFRenderError("pdf_render_invalid_png")
        elif seen_data and not seen_end:
            raise PDFRenderError("pdf_render_invalid_png")
        offset = end
    if not (seen_header and seen_data and seen_end):
        raise PDFRenderError("pdf_render_invalid_png")
    return width, height


def _confirm_process_group_exit(pid: int, deadline: float) -> None:
    while True:
        try:
            _probe_process_group(pid)
        except ProcessLookupError:
            return
        if monotonic() >= deadline:
            raise PDFRenderContainmentError(
                "PDF renderer process group termination was not confirmed"
            )
        sleep(0.02)


def _start_posix_session() -> None:
    start_session = getattr(os, "setsid", None)
    if not callable(start_session):
        raise PDFRenderError("pdf_render_resource_limit_unavailable")
    start_session()


def _kill_process_group(pid: int) -> None:
    kill_group = getattr(os, "killpg", None)
    kill_signal = getattr(signal, "SIGKILL", None)
    if not callable(kill_group) or type(kill_signal) is not int:
        raise PDFRenderContainmentError("POSIX process groups unavailable")
    kill_group(pid, kill_signal)


def _probe_process_group(pid: int) -> None:
    kill_group = getattr(os, "killpg", None)
    if not callable(kill_group):
        raise PDFRenderContainmentError("POSIX process groups unavailable")
    kill_group(pid, 0)
