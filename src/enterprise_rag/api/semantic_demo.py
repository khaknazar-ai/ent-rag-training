"""Public upload/retrieval with explicitly chosen installed CPU assets/cache."""

import os
from dataclasses import replace
from pathlib import Path

from enterprise_rag.api.app import create_app as create_api
from enterprise_rag.bootstrap import build_services, with_managed_local_semantic
from enterprise_rag.settings import Settings


def create_app():
    directory = os.environ.get("RAG_EMBEDDING_DIR", "").strip()
    if not directory:
        raise ValueError("Set RAG_EMBEDDING_DIR to the installed pinned model")
    settings = replace(Settings.from_env(), llm_provider="none")
    state = settings.artifacts_dir / "semantic"
    state.mkdir(parents=True, exist_ok=True)
    services = with_managed_local_semantic(
        build_services(settings),
        embedding_directory=Path(directory),
        index_directory=state / "index",
        embedding_cache_path=state / "embeddings.sqlite3",
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "english-minilm"),
    )
    return create_api(services)
