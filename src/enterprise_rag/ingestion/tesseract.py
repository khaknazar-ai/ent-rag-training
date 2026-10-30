"""Explicit, bounded Tesseract adapter for one rendered page at a time."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Protocol

from enterprise_rag.ingestion.ocr import (
    MAX_OCR_PAGE_CHARS,
    MAX_OCR_REGIONS,
    MAX_RASTER_PAGE_BYTES,
    OCRContractError,
    OCRPageResult,
    OCRProvider,
    OCRTextRegion,
    RasterPage,
)
from enterprise_rag.ingestion.pdf_render import (
    PDFRenderContainmentError,
)

MAX_TESSERACT_BINARY_BYTES = 512 * 1024 * 1024
MAX_TESSDATA_FILE_BYTES = 64 * 1024 * 1024
MAX_TESSERACT_RESULT_BYTES = 8 * 1024 * 1024
TESSERACT_TIMEOUT_SECONDS = 10.0
TESSERACT_MEMORY_BYTES = 1024**3
TESSERACT_CLEANUP_SECONDS = 2.0
_LANGUAGE = re.compile(r"^[A-Za-z0-9_]{1,32}$")


class OCRImagePreprocessor(Protocol):
    revision: str

    def __call__(self, png: bytes) -> bytes: ...


_TSV_COLUMNS = {
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
}


class OCRProviderUnavailable(RuntimeError):
    """An explicitly configured Tesseract binary or model is unavailable."""


class OCRProviderFailed(RuntimeError):
    """A bounded Tesseract invocation or its TSV output failed validation."""

    def __init__(
        self, message: str, *, reason: str = "failed", exit_code: int | None = None
    ):
        if reason not in {
            "failed",
            "startup",
            "timeout",
            "invalid_response",
            "process_failed",
            "exit_failed",
            "launch_failed",
            "output_missing",
            "output_limit",
            "invalid_output",
            "runtime_changed",
            "io_failed",
            "containment",
            "cleanup_unconfirmed",
        }:
            raise ValueError("Unknown OCR failure reason")
        if exit_code is not None and (
            type(exit_code) is not int or not -(2**31) <= exit_code < 2**32
        ):
            raise ValueError("Invalid OCR exit code")
        super().__init__(message)
        self.reason = reason
        self.exit_code = exit_code

    @property
    def error_code(self) -> str:
        return (
            "ocr_provider_failed"
            if self.reason == "failed"
            else f"ocr_provider_{self.reason}"
        )

    @property
    def diagnostic_code(self) -> str:
        return (
            self.error_code
            if self.exit_code is None
            else f"{self.error_code}:{self.exit_code}"
        )


class TesseractCLIProvider(OCRProvider):
    """Use pre-existing pinned Tesseract/model files; never download or install.

    The executable path and tessdata directory are explicit. A provider
    revision binds the executable bytes, selected traineddata hashes, language
    set and fixed OEM/PSM configuration. TSV word boxes align against
    Tesseract's untouched text output so offsets retain its exact whitespace.
    One spawned worker owns the process tree; the existing renderer limits
    apply: at most 10 seconds and 1 GiB.
    """

    provider_id = "tesseract-cli"

    def __init__(
        self,
        executable: Path,
        *,
        tessdata_directory: Path,
        languages: tuple[str, ...],
        temp_root: Path,
        timeout_seconds: float = TESSERACT_TIMEOUT_SECONDS,
        memory_bytes: int = TESSERACT_MEMORY_BYTES,
        image_preprocessor: OCRImagePreprocessor | None = None,
        page_segmentation_mode: int = 3,
    ) -> None:
        self.executable = executable
        self.tessdata_directory = tessdata_directory
        self.languages = languages
        self.temp_root = temp_root
        self.timeout_seconds = timeout_seconds
        self.memory_bytes = memory_bytes
        self.image_preprocessor = image_preprocessor
        if type(page_segmentation_mode) is not int or page_segmentation_mode not in {
            3,
            6,
            11,
        }:
            raise ValueError("OCR page segmentation mode must be 3, 6 or 11")
        self.page_segmentation_mode = page_segmentation_mode
        if not isinstance(executable, Path) or not executable.is_absolute():
            raise ValueError("Tesseract executable must be an absolute path")
        if (
            not isinstance(tessdata_directory, Path)
            or not tessdata_directory.is_absolute()
        ):
            raise ValueError("Tessdata directory must be an absolute path")
        if not isinstance(temp_root, Path) or not temp_root.is_absolute():
            raise ValueError("Tesseract temporary root must be an absolute path")
        if not isinstance(languages, tuple) or not 1 <= len(languages) <= 2:
            raise ValueError("Select one or two explicit Tesseract languages")
        if len(set(languages)) != len(languages) or any(
            not isinstance(language, str) or not _LANGUAGE.fullmatch(language)
            for language in languages
        ):
            raise ValueError("Invalid Tesseract language code")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not (math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 60)
        ):
            raise ValueError("Tesseract timeout must be in (0, 60] seconds")
        if type(memory_bytes) is not int or not 64 * 1024**2 <= memory_bytes <= 1024**3:
            raise ValueError("Tesseract memory limit must be 64 MiB..1 GiB")

    @property
    def revision(self) -> str:
        executable, directory, models = self._resolved_files()
        digest = sha256(b"tesseract-cli-ocr-v1\0")
        for path in (executable, *models):
            digest.update(path.name.encode("utf-8"))
            digest.update(self._file_digest(path).digest())
        digest.update(directory.as_posix().encode("utf-8"))
        for language in self.languages:
            digest.update(language.encode("ascii") + b"\0")
        digest.update(
            f"oem=1\0psm={self.page_segmentation_mode}\0tsv+txt-v1\0process=owned-child-v1".encode(
                "ascii"
            )
        )
        if self.image_preprocessor is not None:
            digest.update(self.image_preprocessor.revision.encode("utf-8"))
        return digest.hexdigest()

    def recognize_page(self, page: RasterPage) -> OCRPageResult:
        from enterprise_rag.workers.temporary import OwnedTemporaryDirectory

        if not isinstance(page, RasterPage):
            raise OCRContractError("Expected a raster page")
        if len(page.image_bytes) > MAX_RASTER_PAGE_BYTES:
            raise OCRContractError("Raster page exceeds the OCR input limit")
        executable, directory, _ = self._resolved_files()
        revision = self.revision
        image_bytes, image_width, image_height = (
            page.image_bytes,
            page.width,
            page.height,
        )
        if self.image_preprocessor is not None:
            from PIL import Image

            image_bytes = self.image_preprocessor(image_bytes)
            with Image.open(io.BytesIO(image_bytes)) as prepared:
                image_width, image_height = prepared.size
                if (
                    prepared.format != "PNG"
                    or image_width * image_height > page.width * page.height
                    or abs(image_width / page.width - image_height / page.height)
                    > 1 / min(page.width, page.height)
                ):
                    raise OCRContractError(
                        "OCR preprocessing must preserve normalized page geometry"
                    )
        try:
            temp_root = self.temp_root.resolve(strict=True)
            if not temp_root.is_dir():
                raise OCRProviderUnavailable(
                    "Tesseract temporary directory unavailable"
                )
            with OwnedTemporaryDirectory(prefix="ocr-page-", dir=temp_root) as name:
                folder = Path(name)
                image_path = folder / "page.png"
                output_base = folder / "result"
                image_path.write_bytes(image_bytes)
                self._run(executable, directory, image_path, output_base)
                tsv_path = output_base.with_suffix(".tsv")
                text_path = output_base.with_suffix(".txt")
                if any(
                    path.is_symlink() or not path.is_file()
                    for path in (tsv_path, text_path)
                ):
                    raise OCRProviderFailed(
                        "Tesseract TSV output missing", reason="output_missing"
                    )
                if any(
                    path.stat().st_size > MAX_TESSERACT_RESULT_BYTES
                    for path in (tsv_path, text_path)
                ):
                    raise OCRProviderFailed(
                        "Tesseract text output exceeds the limit", reason="output_limit"
                    )
                tsv = tsv_path.read_bytes().decode("utf-8-sig", errors="strict")
                raw_text = text_path.read_bytes().decode("utf-8-sig", errors="strict")
        except OCRProviderUnavailable:
            raise
        except OCRProviderFailed:
            raise
        except PDFRenderContainmentError:
            raise OCRProviderFailed(
                "Tesseract process cleanup could not be confirmed",
                reason="cleanup_unconfirmed",
            ) from None
        except (OSError, UnicodeDecodeError, ValueError):
            raise OCRProviderFailed(
                "Tesseract page processing failed", reason="io_failed"
            ) from None
        if self.revision != revision:
            raise OCRProviderFailed(
                "Tesseract runtime changed during recognition", reason="runtime_changed"
            )
        try:
            raw_text, regions = _parse_tsv(tsv, raw_text, image_width, image_height)
            return OCRPageResult(
                run_id=page.run_id,
                document_version_id=page.document_version_id,
                source_pdf_sha256=page.source_pdf_sha256,
                page_number=page.page_number,
                image_sha256=page.image_sha256,
                provider_id="tesseract-cli",
                provider_revision=revision,
                raw_text=raw_text,
                regions=regions,
            )
        except (OCRContractError, ValueError, csv.Error):
            raise OCRProviderFailed(
                "Tesseract TSV result is invalid", reason="invalid_output"
            ) from None

    def _resolved_files(self) -> tuple[Path, Path, tuple[Path, ...]]:
        try:
            executable = self.executable.resolve(strict=True)
            directory = self.tessdata_directory.resolve(strict=True)
            temp_root = self.temp_root.resolve(strict=True)
            if (
                not executable.is_file()
                or not directory.is_dir()
                or not temp_root.is_dir()
            ):
                raise OSError
            if executable.stat().st_size > MAX_TESSERACT_BINARY_BYTES:
                raise OCRProviderUnavailable(
                    "Tesseract executable exceeds the size limit"
                )
            models = []
            for language in self.languages:
                model = (directory / f"{language}.traineddata").resolve(strict=True)
                if directory != model.parent or not model.is_file():
                    raise OSError
                if model.stat().st_size > MAX_TESSDATA_FILE_BYTES:
                    raise OCRProviderUnavailable(
                        "Tesseract traineddata exceeds the size limit"
                    )
                models.append(model)
            return executable, directory, tuple(models)
        except OCRProviderUnavailable:
            raise
        except OSError:
            raise OCRProviderUnavailable(
                "Configured Tesseract files are unavailable"
            ) from None

    @staticmethod
    def _file_digest(path: Path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest

    def _run(
        self, executable: Path, directory: Path, image: Path, output: Path
    ) -> None:
        command = (
            str(executable),
            str(image),
            str(output),
            "--tessdata-dir",
            str(directory),
            "-l",
            "+".join(self.languages),
            "--oem",
            "1",
            "--psm",
            str(self.page_segmentation_mode),
            "txt",
            "tsv",
        )
        _run_isolated_tesseract(
            command,
            output,
            cwd=image.parent,
            timeout_seconds=float(self.timeout_seconds),
            memory_bytes=self.memory_bytes,
        )


def _parse_tsv(
    tsv: str, raw_text: str, page_width: int, page_height: int
) -> tuple[str, tuple[OCRTextRegion, ...]]:
    if (
        not isinstance(tsv, str)
        or not isinstance(raw_text, str)
        or len(tsv.encode("utf-8")) > MAX_TESSERACT_RESULT_BYTES
        or len(raw_text.encode("utf-8")) > MAX_TESSERACT_RESULT_BYTES
        or page_width < 1
        or page_height < 1
    ):
        raise ValueError("Tesseract text output exceeds the page limit")
    reader = csv.DictReader(io.StringIO(tsv, newline=""), delimiter="\t")
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(_TSV_COLUMNS)
        or set(reader.fieldnames) != _TSV_COLUMNS
    ):
        raise ValueError("Tesseract TSV header is invalid")
    words: list[
        tuple[tuple[int, int, int], str, tuple[float, float, float, float], float]
    ] = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Tesseract TSV row is malformed")
        if not row or row.get("level") != "5":
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if len(words) >= MAX_OCR_REGIONS:
            raise ValueError("Tesseract region count exceeds the limit")
        try:
            page_number = int(row["page_num"])
            block_number = int(row["block_num"])
            paragraph_number = int(row["par_num"])
            line_number = int(row["line_num"])
            left, top = int(row["left"]), int(row["top"])
            width, height = int(row["width"]), int(row["height"])
            confidence = float(row["conf"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("Tesseract word row is invalid") from None
        if (
            page_number != 1
            or width <= 0
            or height <= 0
            or left < 0
            or top < 0
            or left + width > page_width
            or top + height > page_height
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 100
        ):
            raise ValueError("Tesseract word geometry is invalid")
        line = (block_number, paragraph_number, line_number)
        bbox = (
            left / page_width,
            top / page_height,
            (left + width) / page_width,
            (top + height) / page_height,
        )
        words.append((line, text, bbox, confidence / 100))
    if not words:
        if raw_text.strip():
            raise ValueError("Tesseract text has no matching word regions")
        return raw_text, ()
    regions = []
    cursor = 0
    previous_line = None
    previous_end = 0
    for ordinal, (line, text, bbox, confidence) in enumerate(words):
        start = raw_text.find(text, cursor)
        if start < 0:
            raise ValueError("Tesseract word does not match its text output")
        gap = raw_text[previous_end:start]
        if gap.strip() or (
            previous_line is not None and ("\n" in gap) != (line != previous_line)
        ):
            raise ValueError("Tesseract text spacing does not match TSV lines")
        end = start + len(text)
        regions.append(OCRTextRegion(ordinal, start, end, text, bbox, confidence))
        cursor = previous_end = end
        previous_line = line
    if raw_text[previous_end:].strip():
        raise ValueError("Tesseract text contains unbound content")
    if len(raw_text) > MAX_OCR_PAGE_CHARS:
        raise ValueError("Tesseract text exceeds the page limit")
    return raw_text, tuple(regions)


def _check_tesseract_outcome(outcome: object) -> None:
    """Decode bounded worker metadata; never propagate native stderr as an error."""
    if outcome == {"status": "ok"}:
        return
    if outcome == {"status": "timeout"}:
        raise OCRProviderFailed("Tesseract timed out", reason="timeout")
    if isinstance(outcome, dict):
        status = outcome.get("status")
        if status == "failed" and set(outcome) == {"status", "exit_code"}:
            code = outcome["exit_code"]
            if type(code) is int and code != 0 and -(2**31) <= code < 2**32:
                raise OCRProviderFailed(
                    "Tesseract exited unsuccessfully",
                    reason="exit_failed",
                    exit_code=code,
                )
        if status == "launch_failed" and set(outcome) == {"status", "os_error"}:
            code = outcome["os_error"]
            if code is None or (type(code) is int and 0 <= code < 2**32):
                raise OCRProviderFailed(
                    "Tesseract could not be launched",
                    reason="launch_failed",
                    exit_code=code,
                )
    raise OCRProviderFailed(
        "Tesseract worker response invalid", reason="invalid_response"
    )


def _run_isolated_tesseract(
    command: tuple[str, ...],
    output_base: Path,
    *,
    cwd: Path,
    timeout_seconds: float,
    memory_bytes: int,
) -> None:
    from enterprise_rag.workers.processes import OwnedChild, _WindowsJob
    from enterprise_rag.workers.temporary import OwnedTemporaryDirectory

    # One owned Python gate starts native work only after job assignment. Avoid
    # a second multiprocessing startup/pipe lifecycle around the native engine.
    bootstrap = """import sys,os,json,subprocess
assert os.environ.get('RAG_WORKER_START_GATE')=='1'
assert sys.stdin.buffer.read(1)==b'1'
if os.name!='nt':
    from enterprise_rag.ingestion.pdf_render import _set_posix_memory_limit
    _set_posix_memory_limit(int(sys.argv[3]))
try:
    child=subprocess.run(sys.argv[4:],cwd=sys.argv[2],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=0x08000000 if os.name=='nt' else 0)
    outcome={'status':'ok'} if child.returncode==0 else {'status':'failed','exit_code':child.returncode}
except OSError as error:
    outcome={'status':'launch_failed','os_error':getattr(error,'winerror',None) or error.errno}
with open(sys.argv[1],'w',encoding='utf-8') as stream:json.dump(outcome,stream)
"""
    environment = dict(os.environ, OMP_THREAD_LIMIT="2", OMP_NUM_THREADS="2")
    package_root = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import sys;sys.path.insert(0," + repr(str(package_root)) + ");" + bootstrap
    )
    with OwnedTemporaryDirectory(prefix="ocr-process-", dir=cwd) as temporary:
        result = Path(temporary) / "status.json"
        job = _WindowsJob(memory_bytes=memory_bytes) if os.name == "nt" else None
        try:
            deadline = monotonic() + timeout_seconds
            with open(os.devnull, "wb") as quiet:
                with OwnedChild(
                    (
                        sys.executable,
                        "-I",
                        "-B",
                        "-c",
                        bootstrap,
                        str(result),
                        str(cwd),
                        str(memory_bytes),
                        *command,
                    ),
                    environment=environment,
                    output=quiet,
                    owner_job=job,
                ) as child:
                    assert child.process is not None
                    child.process.wait(timeout=max(0.001, deadline - monotonic()))
                    if (
                        child.poll() != 0
                        or not result.is_file()
                        or result.stat().st_size > 1024
                    ):
                        raise OCRProviderFailed(
                            "Tesseract worker response invalid",
                            reason="invalid_response",
                        )
                    outcome = json.loads(result.read_text(encoding="utf-8"))
            _check_tesseract_outcome(outcome)
            for path in (
                output_base.with_suffix(".tsv"),
                output_base.with_suffix(".txt"),
            ):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size > MAX_TESSERACT_RESULT_BYTES
                ):
                    raise OCRProviderFailed(
                        "Tesseract text output exceeds the limit", reason="output_limit"
                    )
        except subprocess.TimeoutExpired:
            raise OCRProviderFailed("Tesseract timed out", reason="timeout") from None
        except OCRProviderFailed:
            raise
        except (OSError, ValueError) as error:
            raise OCRProviderFailed(
                "Tesseract process containment failed", reason="containment"
            ) from error
        finally:
            if job is not None:
                try:
                    job.close(timeout_seconds=TESSERACT_CLEANUP_SECONDS)
                except OSError:
                    raise PDFRenderContainmentError(
                        "Tesseract process cleanup unconfirmed"
                    ) from None
