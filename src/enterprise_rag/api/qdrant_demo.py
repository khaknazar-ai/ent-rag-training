"""Explicit local Qdrant runtime shared by the public API and bounded worker."""

import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from enterprise_rag.api.app import create_app as create_api
from enterprise_rag.bootstrap import build_services, with_qdrant_semantic
from enterprise_rag.settings import Settings


def services_from_env(*, initialize_collection=False):
    required = ("RAG_EMBEDDING_DIR", "RAG_QDRANT_URL", "RAG_QDRANT_NAMESPACE")
    if any(not os.environ.get(key, "").strip() for key in required):
        raise ValueError(
            "Qdrant runtime requires model directory, loopback URL and namespace"
        )
    settings = replace(Settings.from_env(), llm_provider="none")
    state = settings.artifacts_dir / "qdrant"
    state.mkdir(parents=True, exist_ok=True)
    return with_qdrant_semantic(
        build_services(settings),
        embedding_directory=Path(os.environ[required[0]]),
        endpoint=os.environ[required[1]],
        namespace=os.environ[required[2]],
        embedding_cache_path=state / "embeddings.sqlite3",
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "multilingual-minilm"),
        initialize_collection=initialize_collection,
    )


def create_app():
    services = services_from_env()
    backend = services.semantic.external_dense.backend
    try:
        app = create_api(services)
    except BaseException:
        backend.close()
        raise
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with previous(application) as state:
                yield state
        finally:
            backend.close()

    app.router.lifespan_context = lifespan
    return app
