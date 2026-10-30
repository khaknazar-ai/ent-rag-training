"""Explicit, bounded fault window for native scheduled synthetic smoke only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from enterprise_rag.diagnostics._evidence import require_evidence
from enterprise_rag.domain.ingestion import JobLease
from enterprise_rag.operations.scheduled import load_config
from enterprise_rag.persistence import SQLiteMetadataStore


def signal(path: Path, payload):
    pending = path.with_suffix(".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def await_release(root, held):
    deadline = time.monotonic() + 90
    while True:
        if time.time() >= held["lease_expires_at"]:
            raise ValueError("Synthetic hold lease expired before release")
        if (root / "smoke-abort.json").exists() or time.monotonic() >= deadline:
            raise ValueError("Synthetic hold aborted/timed out")
        release = root / "release-recovered.json"
        if held["attempts"] == 2 and release.exists():
            value = json.loads(release.read_text(encoding="utf-8"))
            if value != {
                "config_id": held["config_id"],
                "job_id": held["job_id"],
                "lease_token": held["lease_token"],
            }:
                raise ValueError("Synthetic release identity mismatch")
            return
        time.sleep(0.05)


def isolated_hold_metadata(services, config_path: Path):
    config = load_config(config_path)
    if not config.get("diagnostic_hold"):
        raise ValueError("Synthetic hold was not explicitly enabled")
    database = Path(config["data_dir"]) / "enterprise_rag.sqlite3"
    if (
        services.metadata.path.resolve() != database.resolve()
        or services.settings.profile != "lite"
        or services.settings.llm_provider != "none"
        or services.settings.artifacts_dir.resolve() != Path(config["artifacts_dir"])
    ):
        raise ValueError("Synthetic hold does not match isolated runtime")
    root = config_path.resolve().parent
    scenario = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    if (scenario.get("kind"), scenario.get("purpose"), scenario.get("config_id")) != (
        "synthetic",
        "contract",
        config["config_id"],
    ):
        raise ValueError("Synthetic scenario identity mismatch")

    class HeldMetadata(SQLiteMetadataStore):
        def finish_stage_attempt(
            self,
            lease: JobLease,
            attempt_id: str,
            now: float,
            *,
            success: bool,
            error: str | None = None,
            output_checksum: str | None = None,
        ) -> bool:
            attempt = self.get_stage_attempt(attempt_id)
            if (
                success is True
                and require_evidence(
                    attempt, "Missing stage attempt at diagnostic hold"
                ).stage
                == "chunk"
                and lease.job.id == scenario["job_id"]
                and lease.job.run_id == scenario["run_id"]
                and lease.job.attempts in {1, 2}
            ):
                current = require_evidence(
                    self.get_job(lease.job.id), "Missing job at diagnostic hold"
                )
                if (
                    self.get_text_artifacts(
                        require_evidence(
                            lease.job.run_id, "Missing run at diagnostic hold"
                        )
                    )
                    is None
                ):
                    raise ValueError("No committed text artifact at diagnostic hold")
                if (
                    current.lease_token != lease.lease_token
                    or current.lease_expires_at is None
                    or current.lease_expires_at <= time.time()
                ):
                    raise ValueError("Cannot publish hold for an expired/fenced lease")
                held = {
                    "config_id": config["config_id"],
                    "job_id": current.id,
                    "run_id": current.run_id,
                    "attempts": current.attempts,
                    "lease_token": current.lease_token,
                    "lease_owner": current.lease_owner,
                    "lease_expires_at": current.lease_expires_at,
                    "worker_pid": os.getpid(),
                    "stage_attempt_id": attempt_id,
                    "stage": "chunk",
                    "observed_at": time.time(),
                }
                signal(root / f"worker-held-{lease.job.attempts}.json", held)
                await_release(root, held)
                # Never backdate a completion to the timestamp before the hold.
                now = time.time()
            return super().finish_stage_attempt(
                lease,
                attempt_id,
                now,
                success=success,
                error=error,
                output_checksum=output_checksum,
            )

    return HeldMetadata(database)
