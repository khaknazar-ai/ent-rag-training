# Local Document Intelligence

[English](README.en.md) · [Русский](README.md)

A local document workspace: upload files, find information, ask supported questions, and check the source on the page. Documents and metadata are stored locally with SQLite.

## Preview

![Documents, question, and answer](screenshots/workspace.jpg)

[Table answer with its source](screenshots/table-source.jpg) · [Approval workflow](screenshots/approval-source.jpg)

The screenshots show synthetic service desk requests in the running application. Answers in these examples use deterministic document processing; they are not a demonstration of language model quality.

## Install and run

Use Python 3.11 or 3.12. PDF page previews require a separately installed Poppler renderer (`pdftoppm`). The commands below use Windows PowerShell.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install ".[api,pdf,pdf-tables,visual]"
```

Use a prepared local profile containing `.env` and a SQLite database. To create a profile for the first time, run `scripts/prepare_lite.py prepare --root .`. This helper requires the Docker Compose CLI to validate configuration, does not start containers, and refuses to replace existing local data.

Copy `examples/demo-config.example.json` to `demo.local.json` and point it to existing components. Native PDF and table mode needs the `root`, `port`, `table_engine`, and `renderer` fields. The `raster_config`, `ocr_tesseract`, and `ocr_data` fields are only needed for a separately configured OCR/OpenCV mode; omit them if those components are not configured.

```powershell
.venv/Scripts/python.exe scripts/demo.py check --config demo.local.json
.venv/Scripts/python.exe scripts/demo.py api --config demo.local.json
```

Start the worker in a separate terminal:

```powershell
.venv/Scripts/python.exe scripts/demo.py worker --config demo.local.json
```

Open http://127.0.0.1:8880/ or the port in your configuration, and connect using the viewer token from your local `.env`. Credentials and local databases are not included in the repository.

## A two-minute demonstration

Ready-to-use synthetic documents are in `examples/demo/`: a request register, an approval workflow, and a short support handbook. The [demo walkthrough](examples/demo/SHOWCASE.md) is in Russian; the main questions and expected results are below.

After uploading the documents, wait for the worker to mark them ready. Grant the viewer READ access through the admin API, then choose **Options → Page answer · local**.

- `Which Priority is listed for VPN access?` → `High`.
- `Which Status is listed for VPN access?` → `Approved`.
- **View on page** opens the source row and answer cell.
- With OCR/OpenCV configured, `Which step follows Review when Yes?` and **Find source: Request approval** returns `Approve`.

The viewer needs an explicit READ grant to access uploaded documents. Having the admin role does not itself grant document READ access.

## Scope and dependencies

Table and diagram processing supports bounded patterns. OCR and raster diagram processing require external Tesseract/OpenCV components. Fully automated setup on a clean machine, representative quality across varied documents, and production-scale operation are not claimed.

Main libraries: [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [pypdf](https://github.com/py-pdf/pypdf), [pdfplumber](https://github.com/jsvine/pdfplumber), and [Pillow](https://github.com/python-pillow/Pillow).
