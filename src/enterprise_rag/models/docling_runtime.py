"""Explicit SDK interpreter bridge: owned decoder, 10 seconds and 1 GiB."""

from __future__ import annotations

import base64
import importlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from enterprise_rag.application.answer import ModelProtocolError, ModelUnavailable
from enterprise_rag.application.visual_regions import VisualRegionService
from enterprise_rag.models.docling_decode import (
    ConvertedChart,
    ConvertedLabel,
    ConvertedVisualPage,
    decode_doctags,
    require_docling_core,
)
from enterprise_rag.workers.processes import START_GATE_ENV, OwnedChild, _WindowsJob


def converted_page_from_payload(value: object) -> ConvertedVisualPage:
    if not isinstance(value, dict) or set(value) != {
        "charts",
        "diagram_boxes",
        "labels",
    }:
        raise ModelProtocolError("Invalid decoded visual page")
    try:
        if any(
            not isinstance(value[key], list) or len(value[key]) > 256 for key in value
        ):
            raise ValueError
        charts = []
        labels = []
        diagrams = []

        def box(item):
            result = VisualRegionService._bbox(item)
            if result is None:
                raise ValueError
            return result

        for item in value["charts"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"rows", "bbox", "kind"}
                or item["kind"] not in {"bar_chart", "pie_chart", "line_chart", "table"}
            ):
                raise ValueError
            rows = item["rows"]
            if (
                not isinstance(rows, list)
                or len(rows) > 64
                or any(
                    not isinstance(row, list)
                    or len(row) > 8
                    or any(not isinstance(text, str) or len(text) > 600 for text in row)
                    for row in rows
                )
            ):
                raise ValueError
            charts.append(
                ConvertedChart(
                    tuple(tuple(row) for row in rows), box(item["bbox"]), item["kind"]
                )
            )
        for item in value["labels"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"text", "bbox"}
                or not isinstance(item["text"], str)
                or not 1 <= len(item["text"]) <= 600
            ):
                raise ValueError
            labels.append(ConvertedLabel(item["text"], box(item["bbox"])))
        for item in value["diagram_boxes"]:
            diagrams.append(box(item))
        return ConvertedVisualPage(tuple(charts), tuple(diagrams), tuple(labels))
    except (TypeError, ValueError, KeyError) as error:
        raise ModelProtocolError("Invalid decoded visual page") from error


class DoclingDecoderRuntime:
    _package_directory = "docling_core"
    _worker_module = "enterprise_rag.models.docling_runtime"
    _max_result_bytes = 65_536

    def __init__(self, python: Path, packages: Path, temp_root: Path):
        if (
            not all(path.is_absolute() for path in (python, packages, temp_root))
            or not python.is_file()
            or not (packages / self._package_directory).is_dir()
            or not temp_root.is_dir()
        ):
            raise ValueError("Docling decoder requires existing absolute runtime paths")
        self.python, self.packages, self.temp_root = python, packages, temp_root

    def __call__(
        self, raw: str, png: bytes, *, timeout_seconds: float = 10
    ) -> ConvertedVisualPage:
        return self._run(
            {"doctags": raw, "png": base64.b64encode(png).decode("ascii")},
            timeout_seconds,
        )

    def preflight(self, *, timeout_seconds: float = 10) -> None:
        self._run({"probe": True}, timeout_seconds)

    def _run(self, payload: dict, timeout_seconds: float) -> ConvertedVisualPage:
        return converted_page_from_payload(self._invoke(payload, timeout_seconds))

    def _invoke(self, payload: dict, timeout_seconds: float) -> object:
        if not 0 < timeout_seconds <= 10:
            raise ModelUnavailable("Docling decoder deadline exhausted")
        package_root = Path(__file__).resolve().parents[2]
        bootstrap = (
            "import sys;sys.path[:0]=sys.argv[1:3];"
            f"from {self._worker_module} import runtime_main;"
            "runtime_main(sys.argv[3],sys.argv[4])"
        )
        with tempfile.TemporaryDirectory(
            prefix="docling-", dir=self.temp_root
        ) as temporary:
            root = Path(temporary)
            request, result = root / "input.json", root / "output.json"
            request.write_text(json.dumps(payload), encoding="utf-8")
            owner = _WindowsJob(memory_bytes=1024**3) if os.name == "nt" else None
            try:
                with open(os.devnull, "wb") as quiet:
                    with OwnedChild(
                        (
                            str(self.python),
                            "-I",
                            "-B",
                            "-c",
                            bootstrap,
                            str(package_root),
                            str(self.packages),
                            str(request),
                            str(result),
                        ),
                        output=quiet,
                        owner_job=owner,
                    ) as child:
                        assert child.process is not None
                        child.process.wait(timeout=timeout_seconds)
                        if (
                            child.process.returncode != 0
                            or not result.is_file()
                            or result.stat().st_size > self._max_result_bytes
                        ):
                            raise ModelUnavailable(
                                "Docling decoder runtime unavailable"
                            )
                        payload = json.loads(result.read_text(encoding="utf-8"))
                return payload
            except subprocess.TimeoutExpired as error:
                raise ModelUnavailable("Docling decoder timed out") from error
            except (OSError, ValueError) as error:
                raise ModelUnavailable("Docling decoder runtime unavailable") from error
            finally:
                if owner is not None:
                    owner.close(timeout_seconds=2)


def runtime_main(request: str, result: str) -> None:
    if os.environ.get(START_GATE_ENV) != "1" or sys.stdin.buffer.read(1) != b"1":
        raise SystemExit(2)
    if os.name != "nt":
        from enterprise_rag.ingestion.parsers.pdf import _set_posix_memory_limit

        _set_posix_memory_limit(1024**3)
    source = Path(request)
    if source.stat().st_size > 6 * 1024 * 1024:
        raise SystemExit(3)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload == {"probe": True}:
        require_docling_core()
        importlib.import_module("docling_core.types.doc")
        page = ConvertedVisualPage((), (), ())
    else:
        try:
            page = decode_doctags(
                payload["doctags"], base64.b64decode(payload["png"], validate=True)
            )
        except ModelProtocolError:
            Path(result).write_text('{"error":"invalid_conversion"}', encoding="utf-8")
            return
    encoded = json.dumps(asdict(page), allow_nan=False)
    if len(encoded.encode("utf-8")) > 65_536:
        raise SystemExit(4)
    Path(result).write_text(encoded, encoding="utf-8")
