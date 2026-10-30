"""Short commands for an existing local demo profile; no provisioning or downloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from start_lite_demo import main as start_demo

from enterprise_rag.ingestion.parsers.pdf import PDF_TABLE_ENGINES

_REQUIRED = {"root", "port", "table_engine"}
_PATHS = {"root", "renderer", "raster_config", "ocr_tesseract", "ocr_data"}
_ALLOWED = _REQUIRED | _PATHS


def load_config(path: Path) -> dict:
    path = path.resolve()
    if path.stat().st_size > 16384:
        raise ValueError("Demo configuration exceeds16KiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate configuration key")
            result[key] = value
        return result

    values = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if (
        not isinstance(values, dict)
        or not _REQUIRED <= values.keys()
        or values.keys() - _ALLOWED
    ):
        raise ValueError("Unsupported or missing demo configuration fields")
    if type(values["port"]) is not int or not 1 <= values["port"] <= 65535:
        raise ValueError("Port must be1..65535")
    if values["table_engine"] not in PDF_TABLE_ENGINES:
        raise ValueError("Unsupported table engine")
    for key in _PATHS & values.keys():
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError("Configured paths must be nonempty strings")
        candidate = Path(values[key]).expanduser()
        values[key] = (
            (path.parent / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        expected = (
            values[key].is_dir()
            if key in {"root", "ocr_data"}
            else values[key].is_file()
        )
        if not expected:
            raise ValueError("Missing configured path: " + key)
    root = values["root"]
    if (
        not (root / ".env").is_file()
        or not (root / ".local-data/enterprise_rag.sqlite3").is_file()
    ):
        raise ValueError(
            "An existing prepared profile with .env and SQLite is required"
        )
    if "raster_config" in values and "renderer" not in values:
        raise ValueError("Raster answers require the existing renderer")
    ocr = {"ocr_tesseract", "ocr_data"} & values.keys()
    if ocr and (ocr != {"ocr_tesseract", "ocr_data"} or "renderer" not in values):
        raise ValueError(
            "OCR requires renderer, executable and data directory together"
        )
    if "raster_config" in values:
        config = json.loads(values["raster_config"].read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) != {
            "python",
            "packages",
            "temp",
        }:
            raise ValueError("Unsupported raster runtime configuration")
        if (
            not Path(config["python"]).is_file()
            or not Path(config["packages"]).is_dir()
        ):
            raise ValueError("Raster runtime must already be installed")
    return values


def arguments(values: dict, mode: str) -> list[str]:
    result = ["--root", str(values["root"])]
    if "renderer" in values:
        result += ["--renderer", str(values["renderer"])]
    if mode == "api":
        result += ["--port", str(values["port"])]
        if "raster_config" in values:
            result += ["--raster-config", str(values["raster_config"])]
    elif mode == "worker":
        result += ["--worker", "--table-engine", values["table_engine"]]
        if "ocr_tesseract" in values:
            result += [
                "--ocr-tesseract",
                str(values["ocr_tesseract"]),
                "--ocr-data",
                str(values["ocr_data"]),
            ]
            if "raster_config" in values:
                result += ["--ocr-raster-config", str(values["raster_config"])]
            result += ["--ocr-psm", "11"]
    else:
        raise ValueError("Unsupported launch mode")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "api", "worker"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        values = load_config(args.config)
        required = ["fastapi", "uvicorn", "pypdf"]
        if values["table_engine"] != "pypdf":
            required.append("pdfplumber")
        if "renderer" in values:
            required.append("PIL")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError("Missing installed dependencies: " + ", ".join(missing))
        if args.mode == "check":
            print(
                json.dumps(
                    {
                        "status": "CONFIGURED",
                        "url": f"http://127.0.0.1:{values['port']}/",
                        "table_engine": values["table_engine"],
                        "raster_enabled": "raster_config" in values,
                        "ocr_enabled": "ocr_tesseract" in values,
                        "runtime_executed": False,
                        "services_started": False,
                        "downloads": False,
                    }
                )
            )
            return 0
        return start_demo(arguments(values, args.mode))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
