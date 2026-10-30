"""Start a prepared local demo on loopback, without model assets or inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_IDENTITY_KEYS = (
    "RAG_API_TOKEN",
    "RAG_DEMO_TENANT_ID",
    "RAG_DEMO_USER_ID",
    "RAG_DEMO_VIEWER_TOKEN",
    "RAG_DEMO_VIEWER_USER_ID",
)


def configure_environment(root: Path, renderer: Path | None = None) -> None:
    """Load prepared identities without printing or rewriting their tokens."""
    root = root.resolve()
    env_file = root / ".env"
    if not env_file.is_file():
        raise ValueError("Prepare this demo root with scripts/prepare_lite.py first")
    if os.environ.get("RAG_LLM_PROVIDER", "none") != "none":
        raise ValueError("Lite startup requires RAG_LLM_PROVIDER=none")
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or name in values:
            raise ValueError(
                "The prepared demo configuration is malformed or duplicated"
            )
        values[name] = value
    if (
        values.get("RAG_PROFILE", "lite") != "lite"
        or values.get("RAG_LLM_PROVIDER", "none") != "none"
    ):
        raise ValueError("The prepared profile must select lite and no model provider")
    missing = [name for name in _IDENTITY_KEYS if not values.get(name)]
    if missing:
        raise ValueError(
            "Prepared demo identities are incomplete: " + ", ".join(missing)
        )
    database = root / ".local-data" / "enterprise_rag.sqlite3"
    if not database.is_file():
        raise ValueError("The prepared demo database is missing")
    render_settings = {"RAG_DEMO_PDF_RENDERER": "", "RAG_DEMO_PDF_RENDER_TEMP": ""}
    if renderer is not None:
        renderer = renderer.resolve()
        if not renderer.is_file():
            raise ValueError("The configured PDF renderer does not exist")
        render_temp = root / ".local-artifacts" / "render-temp"
        render_temp.mkdir(parents=True, exist_ok=True)
        render_settings = {
            "RAG_DEMO_PDF_RENDERER": str(renderer),
            "RAG_DEMO_PDF_RENDER_TEMP": str(render_temp),
        }
    os.environ.update({name: values[name] for name in _IDENTITY_KEYS})
    os.environ.update(
        RAG_PROFILE="lite",
        RAG_LLM_PROVIDER="none",
        RAG_DATA_DIR=str(root / ".local-data"),
        RAG_ARTIFACTS_DIR=str(root / ".local-artifacts"),
        RAG_DATABASE_URL="sqlite:///" + database.as_posix(),
    )
    os.environ.update(render_settings)


def worker_arguments(
    root: Path,
    *,
    table_engine: str = "pypdf",
    once: bool = False,
    renderer: Path | None = None,
    tesseract: Path | None = None,
    tessdata: Path | None = None,
    languages: str = "eng",
    ocr_raster_config: Path | None = None,
    ocr_psm: int | None = None,
) -> list[str]:
    command = (
        ["worker", "--once"] if once else ["worker-loop", "--interval-seconds", "1"]
    )
    command += ["--worker-id", "lite-demo-worker", "--pdf-table-engine", table_engine]
    if tesseract is None and tessdata is None:
        if languages != "eng" or ocr_raster_config is not None or ocr_psm is not None:
            raise ValueError("OCR languages require OCR executable and data paths")
        return command
    if renderer is None or tesseract is None or tessdata is None:
        raise ValueError("OCR requires --renderer, --ocr-tesseract and --ocr-data")
    if not tesseract.is_file() or not tessdata.is_dir():
        raise ValueError("OCR executable and data directory must already exist")
    artifacts = root.resolve() / ".local-artifacts"
    render_temp, ocr_temp = artifacts / "render-temp", artifacts / "ocr-temp"
    render_temp.mkdir(parents=True, exist_ok=True)
    ocr_temp.mkdir(parents=True, exist_ok=True)
    for flag, value in (
        ("--ocr-renderer", renderer.resolve()),
        ("--ocr-render-temp", render_temp),
        ("--ocr-tesseract-executable", tesseract.resolve()),
        ("--ocr-tessdata-dir", tessdata.resolve()),
        ("--ocr-temp", ocr_temp),
        ("--ocr-languages", languages),
    ):
        command.extend((flag, str(value)))
    if ocr_raster_config is not None:
        if not ocr_raster_config.is_file():
            raise ValueError("The raster OCR configuration must exist")
        command.extend(("--ocr-raster-config", str(ocr_raster_config.resolve())))
    if ocr_psm is not None:
        if ocr_psm not in {3, 6, 11}:
            raise ValueError("OCR segmentation must be 3, 6 or 11")
        command.extend(("--ocr-psm", str(ocr_psm)))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--renderer", type=Path, help="Existing pdftoppm executable")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--visual-config",
        type=Path,
        help="Explicit pinned local vision configuration JSON",
    )
    parser.add_argument(
        "--worker", action="store_true", help="Run the Lite worker loop"
    )
    parser.add_argument(
        "--docling-config",
        type=Path,
        help="Explicit pinned Granite-Docling conversion configuration JSON",
    )
    parser.add_argument(
        "--docling-cli-config",
        type=Path,
        help="Explicit owned llama.cpp CPU conversion JSON",
    )
    parser.add_argument(
        "--raster-config",
        type=Path,
        help="Explicit OpenCV + persisted OCR configuration JSON",
    )
    parser.add_argument(
        "--once", action="store_true", help="Process one job with --worker"
    )
    parser.add_argument(
        "--table-engine",
        choices=("pypdf", "pdfplumber-lines", "pdfplumber-text", "pdfplumber-spans"),
        default="pypdf",
    )
    parser.add_argument(
        "--ocr-tesseract", type=Path, help="Existing Tesseract executable"
    )
    parser.add_argument("--ocr-data", type=Path, help="Existing traineddata directory")
    parser.add_argument("--ocr-languages", default="eng")
    parser.add_argument("--ocr-raster-config", type=Path)
    parser.add_argument("--ocr-psm", type=int, choices=(3, 6, 11), default=None)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    if args.raster_config and (
        args.worker
        or args.renderer is None
        or args.visual_config
        or args.docling_config
        or args.docling_cli_config
    ):
        parser.error(
            "--raster-config requires --renderer, API mode and no other model configuration"
        )
    if args.visual_config and (args.worker or args.renderer is None):
        parser.error("--visual-config requires --renderer and API mode")
    if args.docling_config and (
        args.worker or args.renderer is None or args.visual_config
    ):
        parser.error(
            "--docling-config requires --renderer, API mode and no --visual-config"
        )
    if args.docling_cli_config and (
        args.worker
        or args.renderer is None
        or args.visual_config
        or args.docling_config
    ):
        parser.error(
            "--docling-cli-config requires --renderer, API mode and no other model configuration"
        )
    if not args.worker and (
        args.once
        or args.table_engine != "pypdf"
        or args.ocr_tesseract
        or args.ocr_data
        or args.ocr_languages != "eng"
        or args.ocr_raster_config is not None
        or args.ocr_psm is not None
    ):
        parser.error("Table/OCR processing options require --worker")
    try:
        configure_environment(args.root, args.renderer)
        if args.visual_config and not args.visual_config.is_file():
            raise ValueError("The visual configuration file does not exist")
        os.environ["RAG_DEMO_VISUAL_CONFIG"] = (
            str(args.visual_config.resolve()) if args.visual_config else ""
        )
        if args.docling_config and not args.docling_config.is_file():
            raise ValueError("The Docling configuration file does not exist")
        os.environ["RAG_DEMO_DOCLING_CONFIG"] = (
            str(args.docling_config.resolve()) if args.docling_config else ""
        )
        if args.docling_cli_config and not args.docling_cli_config.is_file():
            raise ValueError("The CLI configuration file does not exist")
        os.environ["RAG_DEMO_DOCLING_CLI_CONFIG"] = (
            str(args.docling_cli_config.resolve()) if args.docling_cli_config else ""
        )
        if args.raster_config and not args.raster_config.is_file():
            raise ValueError("The raster configuration file does not exist")
        os.environ["RAG_DEMO_RASTER_CONFIG"] = (
            str(args.raster_config.resolve()) if args.raster_config else ""
        )
        command = (
            worker_arguments(
                args.root,
                table_engine=args.table_engine,
                once=args.once,
                renderer=args.renderer,
                tesseract=args.ocr_tesseract,
                tessdata=args.ocr_data,
                languages=args.ocr_languages,
                ocr_raster_config=args.ocr_raster_config,
                ocr_psm=args.ocr_psm,
            )
            if args.worker
            else None
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.worker:
        from enterprise_rag.cli import main as worker_main

        return worker_main(command)
    import uvicorn

    uvicorn.run(
        "enterprise_rag.api.demo:create_raster_app"
        if args.raster_config
        else "enterprise_rag.api.demo:create_docling_cli_app"
        if args.docling_cli_config
        else "enterprise_rag.api.demo:create_docling_app"
        if args.docling_config
        else "enterprise_rag.api.demo:create_visual_app"
        if args.visual_config
        else "enterprise_rag.api.demo:create_lite_app",
        factory=True,
        host="127.0.0.1",
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
