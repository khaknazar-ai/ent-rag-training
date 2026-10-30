"""One bounded summary run; runtime verification and durable spend precede HTTP.

Not wired by bootstrap or exposed through API options. The caller supplies the
trusted metadata verifier and must retain raw runtime evidence. An existing
ledger can only resume the explicitly authorized1-spent reserve amendment;
ambiguous, exhausted, changed or concurrently open ledgers fail closed.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from threading import RLock
from time import monotonic

from enterprise_rag.domain.model_summary import SummaryTokenizerBinding
from enterprise_rag.ports.model_summary import SummaryBudgetRefused


class BoundedSummaryRuntimeGrant:
    gpu_layers = -1
    _fresh_span_run = False

    def __init__(
        self,
        *,
        url: str,
        binding: SummaryTokenizerBinding,
        runtime_sha256: str,
        verify_runtime: Callable[[], str],
        ledger: Path,
        run_plan_sha256: str,
        output_reserve_tokens: int = 256,
        resume_one_spent: bool = False,
    ):
        if (
            url != "http://127.0.0.1:11434/api/generate"
            or binding.model_tag is None
            or binding.manifest_sha256 is None
            or re.fullmatch(r"[a-f0-9]{64}", runtime_sha256) is None
            or re.fullmatch(r"[a-f0-9]{64}", run_plan_sha256) is None
            or type(output_reserve_tokens) is not int
            or output_reserve_tokens not in (256, 1024)
            or (
                not self._fresh_span_run
                and resume_one_spent != (output_reserve_tokens == 1024)
            )
            or (
                self._fresh_span_run
                and (resume_one_spent or output_reserve_tokens != 1024)
            )
        ):
            raise ValueError("Invalid bounded runtime identity")
        if self._fresh_span_run:
            from enterprise_rag.models.ollama_summary import (
                SCHEMA_SHA256,
                TEMPLATE_SHA256,
                TRANSPORT_REVISION,
            )

            if (
                binding.model_tag != "qwen3.5:9b-q4_K_M"
                or binding.template_sha256 != TEMPLATE_SHA256
                or binding.schema_sha256 != SCHEMA_SHA256
                or binding.transport_revision != TRANSPORT_REVISION
            ):
                raise ValueError("Fresh span run requires its exact approved binding")
        self.url, self.binding = url, binding
        self.runtime_sha256, self.verify_runtime = runtime_sha256, verify_runtime
        self._started = monotonic()
        self._spent = 0
        self._closed = False
        self._active = False
        self._lock = RLock()
        self._output_reserve_tokens = output_reserve_tokens
        self._identity_sha256 = sha256(
            json.dumps(
                {
                    "plan": run_plan_sha256,
                    "runtime": runtime_sha256,
                    "binding": asdict(binding),
                    "output_reserve_tokens": output_reserve_tokens,
                    "context_tokens": 4096,
                    "global_max": 3,
                    **(
                        {"policy": "fresh-span-summary-max3-v1"}
                        if self._fresh_span_run
                        else {}
                    ),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if resume_one_spent:
            self._resume(ledger, run_plan_sha256)
            return
        # O_EXCL refuses any reuse/reset of this global run budget.
        fd = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._ledger = os.fdopen(fd, "wb", buffering=0)
        try:
            self._append(
                {
                    "event": "run_budget",
                    "MAX_GENERATIONS_GLOBAL": 3,
                    "plan_sha256": run_plan_sha256,
                    "runtime_sha256": runtime_sha256,
                    "binding": asdict(binding),
                    **(
                        {
                            "policy": "fresh-span-summary-max3-v1",
                            "output_reserve_tokens": 1024,
                        }
                        if self._fresh_span_run
                        else {}
                    ),
                }
            )
        except BaseException:
            self.close()
            raise

    def _append(self, record: dict) -> None:
        raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        written = self._ledger.write(raw)
        if written != len(raw):
            raise OSError("Incomplete durable summary budget write")
        os.fsync(self._ledger.fileno())

    @property
    def output_reserve_tokens(self) -> int:
        return self._output_reserve_tokens

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def _resume(self, path: Path, plan: str) -> None:
        # Windows read/write handle denies other writers/deletion for this run.
        import ctypes
        import msvcrt
        from ctypes import wintypes as w

        before = path.lstat()
        if (
            os.name != "nt"
            or before.st_size > 65536
            or before.st_nlink != 1
            or getattr(before, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("Invalid remaining summary budget")
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateFileW.argtypes = [
            w.LPCWSTR,
            w.DWORD,
            w.DWORD,
            ctypes.c_void_p,
            w.DWORD,
            w.DWORD,
            w.HANDLE,
        ]
        api.CreateFileW.restype = w.HANDLE
        handle = api.CreateFileW(
            str(path.absolute()), 0xC0000000, 1, None, 3, 0x00200080, None
        )
        if handle == ctypes.c_void_p(-1).value:
            raise OSError("Remaining summary budget already open or unavailable")
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        self._ledger = os.fdopen(fd, "r+b", buffering=0)
        try:
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino, current.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise ValueError("Remaining summary budget changed")
            rows = [json.loads(line) for line in self._ledger.read().splitlines()]
            if len(rows) != 2 or rows[0] != {
                "event": "run_budget",
                "MAX_GENERATIONS_GLOBAL": 3,
                "plan_sha256": plan,
                "runtime_sha256": self.runtime_sha256,
                "binding": asdict(self.binding),
            }:
                raise ValueError("Remaining summary budget identity mismatch")
            if (
                rows[1].get("event") != "attempt_spent_before_transport"
                or rows[1].get("attempt") != 1
            ):
                raise ValueError("Remaining summary budget is not exactly1 spent")
            if rows[1].get("options", {}).get("num_predict") != 256:
                raise ValueError("Unexpected prior output reserve")
            self._spent = 1
            self._ledger.seek(0, os.SEEK_END)
            self._append(
                {
                    "event": "authorized_reserve_amendment",
                    "spent_unchanged": 1,
                    "from_output_tokens": 256,
                    "to_output_tokens": 1024,
                    "context_tokens": 4096,
                    "grant_sha256": self.identity_sha256,
                }
            )
        except BaseException:
            self.close()
            raise

    def authorize(
        self, url: str, binding: SummaryTokenizerBinding, request: bytes
    ) -> None:
        with self._lock:
            self._authorize(url, binding, request)

    def _authorize(
        self, url: str, binding: SummaryTokenizerBinding, request: bytes
    ) -> None:
        try:
            if (
                self._closed
                or self._active
                or self._spent >= 3
                or monotonic() - self._started > 400
            ):
                raise SummaryBudgetRefused("summary_runtime_budget_closed")
            self._active = True
            payload = json.loads(request)
            if (
                url != self.url
                or binding != self.binding
                or set(payload)
                != {
                    "model",
                    "raw",
                    "stream",
                    "think",
                    "keep_alive",
                    "prompt",
                    "format",
                    "options",
                }
                or payload["model"] != binding.api_model
                or payload["raw"] is not True
                or payload["stream"] is not False
                or payload["think"] is not False
                or payload["keep_alive"] != "0"
                or payload["options"]
                != {
                    "temperature": 0,
                    "num_gpu": -1,
                    "num_thread": 2,
                    "num_ctx": 4096,
                    "num_predict": self.output_reserve_tokens,
                }
                or not isinstance(payload["prompt"], str)
                or len(payload["prompt"].encode()) > 64000
            ):
                raise SummaryBudgetRefused("summary_runtime_request_mismatch")
            if self._fresh_span_run and (
                any(type(value) is not int for value in payload["options"].values())
                or not payload["prompt"]
                or sha256(
                    json.dumps(
                        payload["format"], sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                != binding.schema_sha256
            ):
                raise SummaryBudgetRefused("summary_runtime_request_mismatch")
            if self.verify_runtime() != self.runtime_sha256:
                raise SummaryBudgetRefused("summary_runtime_identity_changed")
            # Failed/timeout/unknown sends remain spent. There is no retry method.
            self._spent += 1
            self._append(
                {
                    "event": "attempt_spent_before_transport",
                    "attempt": self._spent,
                    "request_sha256": sha256(request).hexdigest(),
                    "prompt_sha256": sha256(payload["prompt"].encode()).hexdigest(),
                    "options": payload["options"],
                    "runtime_sha256": self.runtime_sha256,
                }
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._ledger.close()

    def finish(self, *, succeeded: bool) -> None:
        with self._lock:
            self._active = False
            if not succeeded:
                self.close()


class FreshSpanSummaryRuntimeGrant(BoundedSummaryRuntimeGrant):
    """Explicit fresh MAX3/1024 injection; never resumes an historical ledger.

    Only a trusted, separately authorized runner constructs this policy. The
    default application does not expose it; O_EXCL and the existing durable
    spend/identity/timeout gates apply from attempt zero.
    """

    _fresh_span_run = True

    def __init__(self, **kwargs):
        kwargs.setdefault("output_reserve_tokens", 1024)
        super().__init__(**kwargs)
