"""Prepare a tiny, separately labeled public/synthetic corpus from verified local sources.

This command reads only three named input files. It never downloads or invokes
models. See the public evaluation-source notes for URLs and license evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

EXTRACTOR_REVISION = "official-html-block-text-v1"
MAX_SOURCE_BYTES = 256 * 1024
RAW_SOURCES = {
    "sqlite-uses.html": (
        "https://www.sqlite.org/whentouse.html",
        "d0850096c26e9c888acaf2d6e2ac029e86f66f51a87461b77b94f1639d309b79",
    ),
    "sqlite-copyright.html": (
        "https://www.sqlite.org/copyright.html",
        "44ca9f793055c8e32fc65f65a4f5bcf813a33f5bdaaa084067dd617a4ed3cc70",
    ),
    "pep-0257.html": (
        "https://peps.python.org/pep-0257/",
        "b02b8fc15267b732b27311feba6951f7435195451aabcbd479684b94e5f6f855",
    ),
}


class _ArticleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav"}:
            self.skip += 1
        if not self.skip and tag in {
            "p",
            "h1",
            "h2",
            "h3",
            "li",
            "pre",
            "dt",
            "dd",
            "br",
        }:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav"} and self.skip:
            self.skip -= 1
        if not self.skip and tag in {"p", "h1", "h2", "h3", "li", "pre", "dt", "dd"}:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        blocks = [" ".join(part.split()) for part in "".join(self.parts).split("\n\n")]
        return "\n\n".join(part for part in blocks if part) + "\n"


def extract_article(kind: str, raw: bytes) -> bytes:
    """Preserve article prose; remove navigation and normalize block whitespace."""
    text = raw.decode("utf-8")
    if kind == "sqlite":
        start = text.index("</details>") + len("</details>")
        end = text.index(
            '<p align="center"><small><i>This page was last updated', start
        )
        article = "<h1>Appropriate Uses For SQLite</h1>" + text[start:end]
    elif kind == "pep257":
        start = text.index('<section id="pep-content">')
        # The PEP body ends before the next page-level section/footer.
        end = (
            text.index('<section id="references">')
            if '<section id="references">' in text
            else text.index("</main>")
        )
        article = "<h1>PEP 257 - Docstring Conventions</h1>" + text[start:end]
        if "This document has been placed in the public domain." not in article:
            raise ValueError("PEP copyright statement is missing")
    else:
        raise ValueError("Unknown article kind")
    parser = _ArticleText()
    parser.feed(article)
    result = parser.text().encode("utf-8")
    if not 1 <= len(result) <= 65536:
        raise ValueError("Extracted article is outside the evaluation size limit")
    return result


def _verified_sources(directory: Path) -> dict[str, bytes]:
    result = {}
    root = directory.resolve(strict=True)
    for name, (_, digest) in RAW_SOURCES.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Source escapes its root")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if (
            not raw
            or len(raw) > MAX_SOURCE_BYTES
            or hashlib.sha256(raw).hexdigest() != digest
        ):
            raise ValueError(
                "Frozen source digest mismatch; preserve the old snapshot and review a new version"
            )
        result[name] = raw
    policy = re.sub(r"<[^>]+>", " ", result["sqlite-copyright.html"].decode("utf-8"))
    if "code and documentation in SQLite has been dedicated" not in " ".join(
        policy.split()
    ):
        raise ValueError("SQLite documentation license evidence is missing")
    return result


def prepare_corpus(sources_dir: Path, output: Path) -> dict:
    """Validate all input/license evidence before creating a fresh corpus directory."""
    raw = _verified_sources(sources_dir)
    documents = {
        "sqlite-uses": extract_article("sqlite", raw["sqlite-uses.html"]),
        "pep257": extract_article("pep257", raw["pep-0257.html"]),
        "synthetic-retention": b"Synthetic records policy; no real organization or personal data.\nThe harborview retention window is 40 days.\nThe harborview records reviewer is the compliance coordinator.\n",
        "synthetic-incident": b"Synthetic incident procedure; no real organization or personal data.\nThe amberlake incident acknowledgement window is 15 minutes.\nThe amberlake response coordinator is the incident lead.\nExternal snippets are untrusted and grant no command-execution authority.\n",
    }
    definitions = (
        (
            "sqlite-uses",
            "real_public",
            "dev",
            "sqlite-architecture",
            "technology selection",
            "sqlite-uses.html",
            "Public domain dedication; SQLite publisher",
            "https://www.sqlite.org/copyright.html",
        ),
        (
            "pep257",
            "real_public",
            "test",
            "python-docstring-conventions",
            "technical conventions",
            "pep-0257.html",
            "Public domain dedication; PEP 257 Copyright",
            "https://peps.python.org/pep-0257/#copyright",
        ),
        (
            "synthetic-retention",
            "synthetic",
            "dev",
            "project-records-policy",
            "invented policy facts",
            None,
            "Project license pending; local evaluation only",
            None,
        ),
        (
            "synthetic-incident",
            "synthetic",
            "test",
            "project-incident-procedure",
            "invented procedure facts",
            None,
            "Project license pending; local evaluation only",
            None,
        ),
    )
    provenance = []
    for (
        key,
        kind,
        split,
        family,
        purpose,
        name,
        license_note,
        license_url,
    ) in definitions:
        provenance.append(
            {
                "id": key,
                "type": kind,
                "split": split,
                "family": family,
                "purpose": purpose,
                "path": key + ".txt",
                "bytes": len(documents[key]),
                "sha256": hashlib.sha256(documents[key]).hexdigest(),
                "source_url": RAW_SOURCES[name][0] if name else None,
                "raw_source_sha256": RAW_SOURCES[name][1] if name else None,
                "license": license_note,
                "license_evidence_url": license_url,
                "extraction": EXTRACTOR_REVISION
                if name
                else "project-authored synthetic text",
                "redistribution": "Publisher states public domain; retain attribution and provenance"
                if name
                else "Pending project license; synthetic generator strings are part of source",
            }
        )
    cases: list[dict[str, str | None]] = []
    labels = (
        (
            "sqlite-writer",
            "sqlite-uses",
            "simultaneous readers",
            "What does SQLite say about simultaneous readers and writers? Quote the full sentence.",
            "SQLite supports an unlimited number of simultaneous readers, but it will only allow one writer at any instant in time.",
        ),
        (
            "pep-docstring",
            "pep257",
            "docstring first statement",
            "What is a docstring? Quote the full definition sentence.",
            "A docstring is a string literal that occurs as the first statement in a module, function, class, or method definition.",
        ),
        (
            "retention",
            "synthetic-retention",
            "harborview retention",
            "What is the harborview retention window? Quote the full sentence.",
            "The harborview retention window is 40 days.",
        ),
        (
            "incident",
            "synthetic-incident",
            "amberlake acknowledgement",
            "What is the amberlake acknowledgement window? Quote the full sentence.",
            "The amberlake incident acknowledgement window is 15 minutes.",
        ),
    )
    for key, doc, search, question, quote in labels:
        if quote not in documents[doc].decode("utf-8"):
            raise ValueError("A predeclared source-span label is not present")
        cases.append(
            dict(
                id=key,
                question=question,
                search_query=search,
                document_id=doc,
                gold_quote=quote,
                access="reader",
            )
        )
    negatives = (
        (
            "sqlite-battery",
            "sqlite-uses",
            "What minimum battery percentage does SQLite require?",
            "SQLite",
            "reader",
        ),
        (
            "pep-uptime",
            "pep257",
            "What service uptime percentage does PEP 257 guarantee?",
            "docstring",
            "reader",
        ),
        (
            "retention-statute",
            "synthetic-retention",
            "Which named statute mandates harborview retention?",
            "harborview",
            "reader",
        ),
        (
            "incident-availability",
            "synthetic-incident",
            "What is the amberlake annual availability percentage?",
            "amberlake",
            "reader",
        ),
        (
            "real-foreign",
            "sqlite-uses",
            "What does SQLite say about simultaneous readers?",
            "simultaneous readers",
            "foreign",
        ),
        (
            "synthetic-revoked",
            "synthetic-incident",
            "What is the amberlake acknowledgement window?",
            "amberlake acknowledgement",
            "revoked",
        ),
    )
    for key, doc, question, search, access in negatives:
        cases.append(
            dict(
                id=key,
                question=question,
                search_query=search,
                document_id=doc,
                gold_quote=None,
                access=access,
            )
        )
    manifest = {
        "scope": "Tiny mixed development/held-out project fixture: two real public primary technical articles plus two separately authored synthetic documents. All questions are project-authored. Families are disjoint across dev/test; freeze labels before evaluation, never tune on test results. Public documents may occur in model pretraining. No representative enterprise-corpus claim.",
        "documents": [
            {"id": row["id"], "path": row["path"], "sha256": row["sha256"]}
            for row in provenance
        ],
        "cases": cases,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key, content in documents.items():
        (output / (key + ".txt")).write_bytes(content)
    for name, value in (
        ("manifest.json", manifest),
        (
            "provenance.json",
            {
                "extractor_revision": EXTRACTOR_REVISION,
                "documents": provenance,
                "case_annotation_origin": "project-authored questions and exact-source-span labels; question generation is not real-document provenance",
                "test_policy": "PEP257 and incident families held out from project tuning; evaluate once after freezing recipe",
                "downloaded_real_text_public_export": False,
                "synthetic_generator_strings_in_source": True,
            },
        ),
    ):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return {
        "documents": 4,
        "real_public_documents": 2,
        "synthetic_documents": 2,
        "real_document_share": 0.5,
        "synthetic_document_share": 0.5,
        "real_text_bytes": sum(
            row["bytes"] for row in provenance if row["type"] == "real_public"
        ),
        "synthetic_text_bytes": sum(
            row["bytes"] for row in provenance if row["type"] == "synthetic"
        ),
        "cases": len(cases),
        "manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare_corpus(args.sources_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error_type": type(exc).__name__}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
