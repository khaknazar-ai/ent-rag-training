"""Small dependency-free settings object for local development profiles."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

SUPPORTED_PROFILES = frozenset({"lite", "core", "full", "scale"})
DEFAULT_DATA_DIR = Path(".local-data")
DEFAULT_ARTIFACTS_DIR = Path(".local-artifacts")


def _sqlite_url_for(data_dir: Path) -> str:
    """Build a portable SQLite URL from the selected data directory."""

    return f"sqlite:///{(data_dir / 'enterprise_rag.sqlite3').as_posix()}"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings that are safe to construct without external services."""

    profile: str = "lite"
    data_dir: Path = DEFAULT_DATA_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    object_store: str = "local"
    database_url: str = _sqlite_url_for(DEFAULT_DATA_DIR)
    llm_provider: str = "none"
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_timeout_seconds: float = 90.0
    visual_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.profile not in SUPPORTED_PROFILES:
            allowed = ", ".join(sorted(SUPPORTED_PROFILES))
            raise ValueError(
                f"Unsupported RAG_PROFILE={self.profile!r}; expected one of: {allowed}"
            )
        if self.object_store not in {"local", "minio"}:
            raise ValueError("RAG_OBJECT_STORE must be 'local' or 'minio'")
        if not isfinite(self.visual_timeout_seconds) or not (
            1 <= self.visual_timeout_seconds <= 300
        ):
            raise ValueError("RAG_VISUAL_TIMEOUT_SECONDS must be 1-300")
        if self.llm_provider not in {"none", "ollama"}:
            raise ValueError("RAG_LLM_PROVIDER must be 'none' or 'ollama'")
        if self.llm_provider == "ollama":
            url = urlsplit(self.llm_base_url)
            try:
                port = url.port
            except ValueError as exc:
                raise ValueError("RAG_LLM_BASE_URL has an invalid port") from exc
            if (
                url.scheme != "http"
                or url.hostname
                not in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
                or url.username is not None
                or url.password is not None
                or url.path not in {"", "/"}
                or url.query
                or url.fragment
                or port is None
            ):
                raise ValueError("RAG_LLM_BASE_URL must point to a local HTTP host")
            if not self.llm_model.strip() or len(self.llm_model) > 128:
                raise ValueError("RAG_LLM_MODEL must be a nonempty model name")
            if not isfinite(self.llm_timeout_seconds) or not (
                1 <= self.llm_timeout_seconds <= 300
            ):
                raise ValueError("RAG_LLM_TIMEOUT_SECONDS must be 1-300")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables without reading secrets from files."""

        env = os.environ if environ is None else environ
        data_dir = Path(env.get("RAG_DATA_DIR", str(DEFAULT_DATA_DIR)))
        return cls(
            profile=env.get("RAG_PROFILE", "lite"),
            data_dir=data_dir,
            artifacts_dir=Path(
                env.get("RAG_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))
            ),
            object_store=env.get("RAG_OBJECT_STORE", "local"),
            database_url=env.get("RAG_DATABASE_URL", _sqlite_url_for(data_dir)),
            llm_provider=env.get("RAG_LLM_PROVIDER", "none"),
            llm_model=env.get("RAG_LLM_MODEL", "qwen2.5:3b"),
            llm_base_url=env.get("RAG_LLM_BASE_URL", "http://127.0.0.1:11434"),
            llm_timeout_seconds=float(env.get("RAG_LLM_TIMEOUT_SECONDS", "90")),
            visual_timeout_seconds=float(env.get("RAG_VISUAL_TIMEOUT_SECONDS", "180")),
        )

    def ensure_directories(self) -> None:
        """Create only explicitly configured local directories."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
