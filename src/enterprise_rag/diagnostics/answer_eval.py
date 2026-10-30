"""Small manifest-frozen lexical/answer evaluation on a fresh isolated database.

This is an extractive-answer experiment, not semantic or enterprise qualification.
No model is invoked unless the CLI receives --execute-model explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from enterprise_rag.application.answer import AnswerService, pack_evidence_context
from enterprise_rag.application.context import (
    CONTEXT_PACKING_REVISION,
    MAX_CONTEXT_CHARS_PER_ITEM,
    MAX_CONTEXT_SOURCE_CHARS,
)
from enterprise_rag.bootstrap import build_services
from enterprise_rag.diagnostics.answer_protocol import load_protocol, score_protocol
from enterprise_rag.domain.documents import Source, Tenant, User
from enterprise_rag.models import OllamaAnswerModel
from enterprise_rag.security import ActorContext, Permission, PrincipalType
from enterprise_rag.settings import Settings
from enterprise_rag.workers import LiteTextStageExecutor, LiteWorker

MAX_INPUT_BYTES = 65536


def _read_small(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Evaluation input must contain 1..65536 bytes")
    return raw


def load_inputs(manifest: Path, corpus_root: Path):
    """Validate every path, digest and label before creating output or loading a model."""
    raw = _read_small(manifest)
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"scope", "documents", "cases"}:
        raise ValueError("Invalid evaluation manifest")
    if not isinstance(value["scope"], str) or not 1 <= len(value["scope"]) <= 500:
        raise ValueError("A truthful corpus scope description is required")
    documents, cases = value["documents"], value["cases"]
    if not isinstance(documents, list) or not 1 <= len(documents) <= 4:
        raise ValueError("Use 1..4 lawful TXT/Markdown documents")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError("Use 1..12 frozen cases")
    root = corpus_root.resolve(strict=True)
    contents: dict[str, bytes] = {}
    for document in documents:
        if not isinstance(document, dict) or set(document) != {"id", "path", "sha256"}:
            raise ValueError("Invalid corpus document")
        key, relative, digest = document["id"], document["path"], document["sha256"]
        if (
            not isinstance(key, str)
            or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", key) is None
            or key in contents
        ):
            raise ValueError("Invalid or duplicate document id")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("Corpus paths must be relative and stay inside their root")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Unsupported corpus path")
        content = _read_small(path)
        content.decode("utf-8")
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Corpus digest mismatch")
        contents[key] = content
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "question",
            "search_query",
            "document_id",
            "gold_quote",
            "access",
        }:
            raise ValueError("Invalid case")
        for key, maximum in (("id", 64), ("question", 500), ("search_query", 200)):
            if (
                not isinstance(case[key], str)
                or not 1 <= len(case[key].strip()) <= maximum
            ):
                raise ValueError("Invalid case text")
        if case["id"] in ids or case["document_id"] not in contents:
            raise ValueError("Duplicate case or unknown document")
        ids.add(case["id"])
        if case["access"] not in {"reader", "foreign", "revoked"}:
            raise ValueError("Unknown access case")
        gold = case["gold_quote"]
        if gold is not None and (
            not isinstance(gold, str)
            or not gold.strip()
            or len(gold) > 600
            or gold not in contents[case["document_id"]].decode("utf-8")
            or case["access"] != "reader"
        ):
            raise ValueError(
                "Gold must be an exact supported reader span, or null for abstention"
            )
    return raw, value, contents


class _RecordedModel:
    def __init__(self, model):
        self.model = model
        self.calls: list[dict[str, object]] = []

    def select(self, query, snippets):
        record: dict[str, Any] = {
            "question": query,
            "snippets": [asdict(item) for item in snippets],
        }
        self.calls.append(record)
        started = time.perf_counter()
        try:
            result = self.model.select(query, snippets)
            record["selection"] = asdict(result)
            return result
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["seconds"] = time.perf_counter() - started


def evaluate(
    manifest: Path,
    corpus_root: Path,
    output: Path,
    *,
    model=None,
    model_metadata=None,
    protocol: Path | None = None,
):
    """Use real upload/worker/search/answer services; doubles are labeled by default."""
    raw, inputs, contents = load_inputs(manifest, corpus_root)
    protocol_raw, labels = (
        load_protocol(protocol, raw, inputs, contents)
        if protocol is not None
        else (None, None)
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(raw)
    if protocol_raw is not None:
        (output / "protocol.json").write_bytes(protocol_raw)
    (output / "inputs").mkdir()
    for document in inputs["documents"]:
        target = output / "inputs" / document["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[document["id"]])
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "scope": inputs["scope"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "retrieval-only"
        if model is None
        else "retrieval-and-extractive-answer",
        "model": model_metadata or {"kind": "none" if model is None else "test-double"},
        "strategy": "sqlite-fts5-bm25; explicit assisted search_query; no semantic embeddings",
        "model_download_requests": 0,
        "context_packing": {
            "revision": CONTEXT_PACKING_REVISION,
            "max_chars_per_item": MAX_CONTEXT_CHARS_PER_ITEM,
            "max_source_chars_per_item": MAX_CONTEXT_SOURCE_CHARS,
            "offset_unit": "unicode_code_points",
        },
        "cases": [],
        "source_sha256": {},
    }
    if protocol_raw is not None:
        report["protocol_sha256"] = hashlib.sha256(protocol_raw).hexdigest()
    package = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        str(p.relative_to(package).as_posix()): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    records: list[dict[str, Any]] = []
    report["cases"] = records
    started = time.perf_counter()
    try:
        settings = Settings(
            data_dir=output / "data",
            artifacts_dir=output / "artifacts",
            database_url="sqlite:///" + (output / "data/metadata.sqlite3").as_posix(),
        )
        services = build_services(settings)
        tenant = Tenant.create("eval-reader", "Isolated evaluation")
        user = User.create(tenant.id, "reader@example.invalid")
        foreign = Tenant.create("eval-foreign", "Isolated foreign evaluation")
        outsider = User.create(foreign.id, "outsider@example.invalid")
        for t, u in ((tenant, user), (foreign, outsider)):
            services.metadata.ensure_tenant(t)
            services.metadata.upsert_user(u)
        if not services.metadata.grant_tenant_admin(tenant.id, user.id):
            raise RuntimeError("Isolated admin provisioning failed")
        actor = ActorContext(tenant.id, user.id)
        foreign_actor = ActorContext(foreign.id, outsider.id)
        documents = {}
        for key, content in contents.items():
            accepted = asyncio.run(
                services.uploads.accept_upload(
                    actor=actor,
                    tenant=tenant,
                    source=Source.create(tenant.id, "eval"),
                    canonical_key="eval/" + key,
                    title=key,
                    filename=key + ".txt",
                    stream=BytesIO(content),
                )
            )
            documents[key] = accepted.document.id
            services.management.grant_permission(
                actor,
                accepted.document.id,
                PrincipalType.USER,
                user.id,
                Permission.READ,
            )
            worker_result = LiteWorker(services.metadata, "eval-worker").run_once(
                LiteTextStageExecutor(services.metadata, services.object_store)
            )
            if not worker_result.completed or worker_result.lease_lost:
                raise RuntimeError("Evaluation ingestion failed")
        recorded = _RecordedModel(model) if model is not None else None
        answer = (
            AnswerService(services.search, recorded) if recorded is not None else None
        )
        for case in inputs["cases"]:
            row: dict[str, Any] = {"id": case["id"], "input": case, "error_type": None}
            records.append(row)
            tick = time.perf_counter()
            before = len(recorded.calls) if recorded is not None else 0
            document = documents[case["document_id"]]
            row["expected_document_id"] = document
            if case["access"] == "revoked":
                services.management.revoke_permission(
                    actor, document, PrincipalType.USER, user.id, Permission.READ
                )
            current_actor = foreign_actor if case["access"] == "foreign" else actor
            try:
                hits = services.search.search(
                    current_actor, case["search_query"], limit=4
                )
                gold = case["gold_quote"]
                row["retrieved"] = [asdict(hit) for hit in hits]
                packed = pack_evidence_context(hits, case["search_query"])
                row["packed_context"] = [asdict(snippet) for snippet in packed]
                context_by_id = {
                    snippet.evidence_id: snippet.text for snippet in packed
                }
                rank = next(
                    (
                        i
                        for i, hit in enumerate(hits, 1)
                        if gold is not None
                        and hit.document_id == document
                        and gold in context_by_id.get(hit.chunk_id, "")
                    ),
                    None,
                )
                row["gold_context_rank"] = rank
                if (case["access"] == "foreign" and hits) or (
                    case["access"] == "revoked"
                    and any(hit.document_id == document for hit in hits)
                ):
                    raise RuntimeError("Access boundary failed")
                if answer is not None:
                    assert recorded is not None
                    result = answer.answer(
                        current_actor,
                        case["question"],
                        search_query=case["search_query"],
                    )
                    row["answer"] = asdict(result)
                    # These are separate metrics: authorized source membership is not factual support.
                    row["citation_valid"] = (
                        not result.abstained
                        and bool(result.citations)
                        and all(
                            (
                                live := services.search.get_evidence(
                                    current_actor, cite.evidence_id
                                )
                            )
                            is not None
                            and live.document_id == cite.document_id
                            and live.document_version_id == cite.document_version_id
                            and live.run_id == cite.run_id
                            and 0
                            <= cite.chunk_start_char
                            < cite.chunk_end_char
                            <= len(live.text)
                            and live.text[cite.chunk_start_char : cite.chunk_end_char]
                            == cite.quote
                            and live.page_numbers == cite.page_numbers
                            for cite in result.citations
                        )
                    )
                    row["gold_exact_match"] = (
                        result.answer == gold if gold is not None else result.abstained
                    )
                    row["expected_outcome_met"] = row["gold_exact_match"] and (
                        gold is None or row["citation_valid"]
                    )
                    row["model_calls"] = recorded.calls[before:]
                    if case["access"] == "foreign" and (
                        recorded.calls[before:] or not result.abstained
                    ):
                        raise RuntimeError("Access boundary failed")
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                row["expected_outcome_met"] = False
            finally:
                if recorded is not None:
                    row["model_calls"] = recorded.calls[before:]
                if case["access"] == "revoked":
                    services.management.grant_permission(
                        actor, document, PrincipalType.USER, user.id, Permission.READ
                    )
                row["seconds"] = time.perf_counter() - tick
                (output / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps({"case": case["id"], "error_type": row["error_type"]}),
                    flush=True,
                )
        positive = [row for row in records if row["input"]["gold_quote"] is not None]
        report["metrics"] = {
            "cases": len(records),
            "errors": sum(row["error_type"] is not None for row in records),
            "positive_cases": len(positive),
            "positive_context_hit_at_4": sum(
                row.get("gold_context_rank") is not None for row in positive
            )
            / len(positive)
            if positive
            else None,
            "positive_context_mrr_at_4": sum(
                1 / row["gold_context_rank"]
                for row in positive
                if row.get("gold_context_rank") is not None
            )
            / len(positive)
            if positive
            else None,
            "expected_outcome_rate": sum(
                bool(row.get("expected_outcome_met")) for row in records
            )
            / len(records)
            if model is not None
            else None,
        }
        report["status"] = (
            "completed-with-errors" if report["metrics"]["errors"] else "completed"
        )
        if labels is not None:
            report["dev_protocol"] = score_protocol(labels, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def installed_model_digest(model: OllamaAnswerModel) -> str:
    """Read installed model metadata only; never pull or load a model."""
    import urllib.request

    request = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
    with model._opener.open(request, timeout=5) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Model metadata response is too large")
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise ValueError("Invalid model metadata")
    matching = [
        item
        for item in value["models"]
        if isinstance(item, dict) and item.get("name") == model.model
    ]
    if len(matching) != 1 or not isinstance(matching[0].get("digest"), str):
        raise ValueError("The exact requested model must already be installed")
    digest = matching[0]["digest"]
    if re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", digest) is None:
        raise ValueError("Invalid installed model digest")
    return digest.removeprefix("sha256:")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute-model", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--model-digest")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)
    if args.execute_model and (not args.model or not args.model_digest):
        parser.error("Explicit local model and verified digest are required")
    model = None
    metadata = None
    try:
        if args.execute_model:
            model = OllamaAnswerModel(
                "http://127.0.0.1:11434", args.model, context_tokens=2048
            )
            digest = installed_model_digest(model)
            if digest != args.model_digest.removeprefix("sha256:"):
                raise ValueError(
                    "Installed model digest does not match the frozen configuration"
                )
            metadata = {
                "kind": "local-ollama",
                "name": args.model,
                "digest": digest,
                "digest_verified_before_run": True,
                "requested_options": model.options,
                "timeout_seconds": model.timeout_seconds,
            }
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}))
        return 2
    try:
        report = evaluate(
            args.manifest,
            args.corpus_root,
            args.output_dir,
            model=model,
            model_metadata=metadata,
            protocol=args.protocol,
        )
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}))
        return 2
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}))
    return (
        0
        if report["status"] == "completed"
        and (model is None or report["metrics"]["expected_outcome_rate"] == 1)
        and (
            model is None
            or report.get("dev_protocol", {}).get("expected_disposition_rate", 1) == 1
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
