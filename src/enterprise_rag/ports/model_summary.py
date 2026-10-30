"""Exact prompt preparation/counting precedes the separate model transport call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from enterprise_rag.domain.model_summary import (
    ModelSummaryDraft,
    SummaryTokenBudget,
    SummaryTokenizerBinding,
)
from enterprise_rag.domain.summaries import SummaryScope
from enterprise_rag.domain.summary_spans import SummarySpan

if TYPE_CHECKING:
    from enterprise_rag.application.answer import EvidenceSnippet


class SummaryBudgetRefused(ValueError):
    def __init__(self, reason: str):
        super().__init__("Summary token budget unavailable")
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PreparedModelSummary:
    prompt: str
    schema_json: str
    token_budget: SummaryTokenBudget
    request_sha256: str
    scope: SummaryScope | None = None
    evidence: tuple[EvidenceSnippet, ...] = ()
    spans: tuple[SummarySpan, ...] = ()


class SummaryPromptCounter(Protocol):
    # This is a trusted adapter dependency, never a request/client supplied flag.
    # Local vocabulary qualification does not verify the running server's model.
    # Network qualification is a separate trusted dependency, never client input.
    binding: SummaryTokenizerBinding
    network_identity_verified: bool

    def count_prompt(self, prompt: str) -> int:
        """Count the complete rendered prompt with all actual special tokens."""


class SummaryRuntimeGrant(Protocol):
    """Trusted per-run dependency, never an API flag or global configuration.

    The implementation must reverify endpoint/runtime and pinned model/manifest,
    enforce its fixed request limits and durably spend an attempt BEFORE return.
    Any exception forbids transport. Unknown outcomes remain spent, without retry.
    """

    @property
    def gpu_layers(self) -> int: ...

    @property
    def output_reserve_tokens(self) -> int: ...

    @property
    def identity_sha256(self) -> str: ...

    def authorize(
        self, url: str, binding: SummaryTokenizerBinding, request: bytes
    ) -> None: ...

    def finish(self, *, succeeded: bool) -> None:
        """Release the single in-flight call; close grant on uncertain/failure."""


class ModelSummaryProvider(Protocol):
    @property
    def identity(self) -> dict[str, str | None]: ...

    def prepare(
        self, snippets: tuple[EvidenceSnippet, ...], *, scope: SummaryScope
    ) -> PreparedModelSummary:
        """No inference/transport; refuse unqualified bindings or token overflow."""

    def generate(self, prepared: PreparedModelSummary) -> ModelSummaryDraft:
        """Verify the exact prepared request again before transport; no fallback."""
