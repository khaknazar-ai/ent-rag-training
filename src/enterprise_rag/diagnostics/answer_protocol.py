"""Predeclared dev labels, separate from strict gold and historical reports.

Accepted quotations are human-authored support labels, not an entailment judge.
They must be frozen with the manifest before running the existing evaluator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REVISION = "predeclared-extractive-dev-v1"
MAX_PROTOCOL_BYTES = 65536


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate protocol field")
        result[key] = value
    return result


def load_protocol(path: Path, manifest_raw: bytes, inputs: dict, contents: dict):
    """Reject stale, incomplete or invalid labels before any output/model call."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_PROTOCOL_BYTES + 1)
    if not 1 <= len(raw) <= MAX_PROTOCOL_BYTES:
        raise ValueError("Invalid protocol size")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if (
        not isinstance(value, dict)
        or set(value) != {"revision", "manifest_sha256", "cases"}
        or value["revision"] != REVISION
        or value["manifest_sha256"] != hashlib.sha256(manifest_raw).hexdigest()
        or not isinstance(value["cases"], list)
    ):
        raise ValueError("Invalid protocol or manifest binding")
    cases = {case["id"]: case for case in inputs["cases"]}
    seen = set()
    for label in value["cases"]:
        if not isinstance(label, dict) or set(label) != {
            "id",
            "kind",
            "accepted_quotes",
        }:
            raise ValueError("Invalid protocol label")
        key, kind, quotes = label["id"], label["kind"], label["accepted_quotes"]
        if not isinstance(key, str) or key not in cases or key in seen:
            raise ValueError("Duplicate or unknown protocol case")
        seen.add(key)
        case = cases[key]
        if not isinstance(kind, str) or kind not in {
            "answer",
            "unanswerable",
            "conflict",
            "acl",
        }:
            raise ValueError("Unknown protocol disposition")
        if not isinstance(quotes, list) or len(quotes) > 8:
            raise ValueError("Invalid accepted quotations")
        if kind == "answer":
            source = contents[case["document_id"]].decode("utf-8")
            if (
                case["gold_quote"] is None
                or not quotes
                or any(
                    not isinstance(quote, str)
                    or not quote.strip()
                    or len(quote) > 600
                    or quote not in source
                    for quote in quotes
                )
            ):
                raise ValueError("Answer labels require exact original source spans")
            if len(set(quotes)) != len(quotes) or case["gold_quote"] not in quotes:
                raise ValueError("Keep strict gold among unique accepted quotations")
        elif quotes or case["gold_quote"] is not None:
            raise ValueError("Abstention labels cannot contain a gold answer")
        if (kind == "acl") != (case["access"] != "reader"):
            raise ValueError("ACL labels must match the manifest access boundary")
    if seen != set(cases):
        raise ValueError("Protocol must label every manifest case exactly once")
    return raw, value


def score_protocol(protocol: dict, report: dict) -> dict:
    """Score each declared case; missing/error rows remain in the denominator.

    citation_valid comes from the runner's live canonical-source checks. A
    predeclared support match additionally requires the intended document and
    the exact returned answer in the citations. No substring/LLM judging.
    """
    rows = {row["id"]: row for row in report["cases"]}
    executed = report["mode"] != "retrieval-only"
    results = []
    for label in protocol["cases"]:
        row = rows.get(label["id"], {})
        answer = row.get("answer") or {}
        citations = answer.get("citations") or []
        failed = bool(row.get("error_type")) or not answer
        abstained = answer.get("abstained") is True
        clean_refusal = abstained and answer.get("answer") is None and not citations
        attribution = bool(
            not failed and row.get("citation_valid") is True and citations
        )
        supported = bool(
            not failed
            and not abstained
            and attribution
            and answer.get("answer") in label["accepted_quotes"]
            and row.get("expected_document_id")
            and all(
                cite.get("document_id") == row["expected_document_id"]
                and cite.get("quote") == answer["answer"]
                for cite in citations
            )
        )
        reason = answer.get("abstention_reason")
        kind = label["kind"]
        boundary = (
            clean_refusal
            and not row.get("model_calls")
            and not row.get("retrieved")
            and not row.get("packed_context")
            and reason == "no_authorized_evidence"
        )
        if kind == "answer":
            outcome = supported
        elif kind == "acl":
            outcome = boundary
        elif kind == "conflict":
            outcome = clean_refusal and reason == "conflicting_evidence"
        else:
            outcome = clean_refusal and reason in {
                "insufficient_support",
                "model_abstained",
            }
        results.append(
            {
                "id": label["id"],
                "kind": kind,
                "evaluated": executed,
                "strict_gold_exact": row.get("gold_exact_match") if executed else None,
                "citation_attribution": attribution if executed else None,
                "predeclared_support": supported
                if executed and kind == "answer"
                else None,
                "expected_disposition": bool(outcome and not failed)
                if executed
                else None,
                "false_abstention": abstained
                if executed and kind == "answer"
                else None,
            }
        )
    groups = {}
    for kind in ("answer", "unanswerable", "conflict", "acl"):
        selected = [item for item in results if item["kind"] == kind]
        groups[kind] = {
            "cases": len(selected),
            "passed": sum(item["expected_disposition"] is True for item in selected)
            if executed
            else None,
        }
    return {
        "revision": REVISION,
        "support_basis": "predeclared exact source spans; not an entailment proof",
        "cases": results,
        "groups": groups,
        "expected_disposition_rate": sum(
            item["expected_disposition"] is True for item in results
        )
        / len(results)
        if executed
        else None,
    }
