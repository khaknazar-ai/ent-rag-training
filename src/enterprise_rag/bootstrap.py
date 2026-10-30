"""Dependency-free service wiring for the resource-aware lite profile."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from enterprise_rag.application import (
    DocumentDeletionService,
    DocumentManagementService,
    DocumentUploadService,
    IngestionReconciliationService,
    MaintenanceService,
    SearchService,
)
from enterprise_rag.application.answer import AnswerService
from enterprise_rag.application.document_summary import DocumentSummaryService
from enterprise_rag.application.model_document_summary import (
    ModelDocumentSummaryService,
)
from enterprise_rag.application.semantic import LocalSemanticService
from enterprise_rag.application.semantic_answer import SemanticAnswerService
from enterprise_rag.application.visual_answer import VisualAnswerService
from enterprise_rag.application.visual_regions import (
    PDFRegionRenderer,
    VisualRegionService,
)
from enterprise_rag.models import OllamaAnswerModel
from enterprise_rag.models.extractive_summary import LocalExtractiveSummary
from enterprise_rag.models.ollama_summary import OllamaSummaryModel
from enterprise_rag.persistence import SQLiteMetadataStore
from enterprise_rag.persistence.sqlite_query_history import SQLiteQueryHistory
from enterprise_rag.persistence.sqlite_summary_source import SQLiteSummarySource
from enterprise_rag.ports.query_history import QueryHistory
from enterprise_rag.settings import Settings
from enterprise_rag.storage import LocalObjectStore


@dataclass(frozen=True, slots=True)
class ServiceBundle:
    """Concrete services shared by the API, CLI and worker entry points."""

    settings: Settings
    metadata: SQLiteMetadataStore
    object_store: LocalObjectStore
    uploads: DocumentUploadService
    deletion: DocumentDeletionService
    ingestion_reconciliation: IngestionReconciliationService
    maintenance: MaintenanceService
    management: DocumentManagementService
    search: SearchService
    answer: AnswerService | None = None
    semantic: LocalSemanticService | None = None
    semantic_answer: SemanticAnswerService | None = None
    query_history: QueryHistory | None = None
    summary: DocumentSummaryService | None = None
    model_summary: ModelDocumentSummaryService | None = None
    visual_regions: VisualRegionService | None = None
    visual_answer: VisualAnswerService | None = None


def with_pdf_regions(
    services: ServiceBundle, *, renderer_command: tuple[str, ...], temp_root: Path
) -> ServiceBundle:
    """Opt in to local page rendering; default bootstrap only reads saved images."""
    return replace(
        services,
        visual_regions=VisualRegionService(
            services.metadata,
            services.object_store,
            renderer=PDFRegionRenderer(renderer_command, temp_root),
        ),
    )


def with_local_semantic(
    services: ServiceBundle,
    *,
    embedding_directory: Path,
    snapshot_path: Path,
    reranker_directory: Path | None = None,
) -> ServiceBundle:
    """Explicit CPU-only model/snapshot wiring; never download or alter metadata.

    The operator supplies existing pinned model files and a separately built
    snapshot. Default bootstrap does not import optional inference libraries.
    """
    from enterprise_rag.models.local_onnx import MiniLMOnnxEmbedding, MiniLMOnnxReranker
    from enterprise_rag.retrieval.sqlite_dense import SQLiteDenseSnapshot

    embedding = MiniLMOnnxEmbedding(embedding_directory)
    snapshot = SQLiteDenseSnapshot(snapshot_path, embedding.space)
    reranker = (
        MiniLMOnnxReranker(reranker_directory)
        if reranker_directory is not None
        else None
    )
    return replace(
        services,
        semantic=LocalSemanticService(services.search, embedding, snapshot, reranker),
    )


def with_managed_local_semantic(
    services: ServiceBundle,
    *,
    embedding_directory: Path,
    index_directory: Path,
    reranker_directory: Path | None = None,
    enable_extractive: bool = False,
    embedding_cache_path: Path | None = None,
    embedding_model: str = "english-minilm",
) -> ServiceBundle:
    """Explicit managed CPU retrieval; model calls occur only on requests.

    Extractive answers require an operator opt-in and existing loopback Ollama
    settings. Their model support decisions remain subject to quality evaluation.
    """
    from enterprise_rag.models.local_onnx import (
        MiniLMOnnxEmbedding,
        MiniLMOnnxReranker,
        MultilingualMiniLMOnnxEmbedding,
    )
    from enterprise_rag.models.ollama_grounded import OllamaGroundedAnswerModel
    from enterprise_rag.retrieval.cached_embeddings import CachedEmbeddings
    from enterprise_rag.retrieval.lifecycle import DenseSnapshotLifecycle
    from enterprise_rag.retrieval.sqlite_embedding_cache import SQLiteEmbeddingCache

    model = None
    if enable_extractive:
        if services.settings.llm_provider != "ollama":
            raise ValueError(
                "Extractive semantic answers require configured local Ollama"
            )
        model = OllamaGroundedAnswerModel(
            services.settings.llm_base_url,
            services.settings.llm_model,
            timeout_seconds=services.settings.llm_timeout_seconds,
        )
    providers = {
        "english-minilm": MiniLMOnnxEmbedding,
        "multilingual-minilm": MultilingualMiniLMOnnxEmbedding,
    }
    if embedding_model not in providers:
        raise ValueError("Unknown explicitly configured embedding model")
    embedding = providers[embedding_model](embedding_directory)
    reranker = (
        MiniLMOnnxReranker(reranker_directory)
        if reranker_directory is not None
        else None
    )
    semantic = LocalSemanticService(
        services.search,
        embedding,
        reranker=reranker,
        lifecycle=DenseSnapshotLifecycle(index_directory),
        embedding_cache=None
        if embedding_cache_path is None
        else CachedEmbeddings(
            SQLiteEmbeddingCache(embedding_cache_path)
            if embedding_cache_path.exists()
            else SQLiteEmbeddingCache.create(embedding_cache_path)
        ),
    )
    return replace(
        services,
        semantic=semantic,
        semantic_answer=SemanticAnswerService(
            semantic,
            model,
            history=services.query_history,
            model_configuration={
                "provider": "local-onnx" if model is None else "local-onnx-and-ollama",
                "embedding_space": {
                    "provider_id": semantic.provider.space.provider_id,
                    "revision": semantic.provider.space.model_revision,
                    "dimension": semantic.provider.space.dimension,
                    "preprocessing": semantic.provider.space.preprocessing_revision,
                },
                "reranker": {
                    "provider_id": "onnxruntime-cpu",
                    "model_revision": reranker.core.model_revision,
                    "preprocessing": "wordpiece-query96-pair512-overlap32-windowmax-v1",
                }
                if reranker is not None
                else None,
                "generation": _history_model_configuration(model),
            },
        ),
    )


def with_qdrant_semantic(
    services: ServiceBundle,
    *,
    embedding_directory: Path,
    endpoint: str,
    namespace: str,
    embedding_cache_path: Path,
    embedding_model: str = "multilingual-minilm",
    initialize_collection: bool = False,
) -> ServiceBundle:
    """Explicit bounded external dense configuration; no generation or downloads.

    Use the same endpoint/namespace/model/cache for API and worker. Initialization
    is a one-time operator action; reopen fails if its collection is missing.
    Existing lexical databases require an explicit migration, not silent adoption.
    """
    from enterprise_rag.application.qdrant import QdrantPublication
    from enterprise_rag.models.local_onnx import (
        MiniLMOnnxEmbedding,
        MultilingualMiniLMOnnxEmbedding,
    )
    from enterprise_rag.retrieval.cached_embeddings import CachedEmbeddings
    from enterprise_rag.retrieval.qdrant import QdrantDenseRetriever
    from enterprise_rag.retrieval.sqlite_embedding_cache import SQLiteEmbeddingCache

    providers = {
        "english-minilm": MiniLMOnnxEmbedding,
        "multilingual-minilm": MultilingualMiniLMOnnxEmbedding,
    }
    if embedding_model not in providers:
        raise ValueError("Unknown explicitly configured embedding model")
    provider = providers[embedding_model](embedding_directory)
    backend = QdrantDenseRetriever(endpoint, namespace, provider.space)
    try:
        cache = CachedEmbeddings(
            SQLiteEmbeddingCache(embedding_cache_path)
            if embedding_cache_path.exists()
            else SQLiteEmbeddingCache.create(embedding_cache_path)
        )
        publication = QdrantPublication(services.metadata, backend, provider, cache)
        if initialize_collection:
            backend.initialize()
        else:
            backend.check_schema()
        semantic = LocalSemanticService(
            services.search, provider, embedding_cache=cache, external_dense=publication
        )
        return replace(
            services,
            semantic=semantic,
            semantic_answer=SemanticAnswerService(
                semantic, None, history=services.query_history
            ),
        )
    except BaseException:
        backend.close()
        raise


def build_services(settings: Settings | None = None) -> ServiceBundle:
    """Build lite services from explicit settings without starting background work."""

    runtime_settings = settings or Settings.from_env()
    if runtime_settings.profile != "lite" or runtime_settings.object_store != "local":
        raise ValueError(
            "Only RAG_PROFILE=lite with RAG_OBJECT_STORE=local is implemented"
        )
    runtime_settings.ensure_directories()
    database_path = _sqlite_path(runtime_settings.database_url)
    metadata = SQLiteMetadataStore(database_path)
    object_store = LocalObjectStore(runtime_settings.artifacts_dir / "objects")
    uploads = DocumentUploadService(
        metadata,
        object_store,
        spool_dir=runtime_settings.artifacts_dir / "upload-spool",
    )
    deletion = DocumentDeletionService(metadata, object_store)
    ingestion_reconciliation = IngestionReconciliationService(metadata)
    search = SearchService(metadata)
    query_history = SQLiteQueryHistory(metadata)
    answer = None
    if runtime_settings.llm_provider == "ollama":
        model = OllamaAnswerModel(
            runtime_settings.llm_base_url,
            runtime_settings.llm_model,
            timeout_seconds=runtime_settings.llm_timeout_seconds,
        )
        answer = AnswerService(
            search,
            model,
            history=query_history,
            model_configuration=_history_model_configuration(model),
        )
    return ServiceBundle(
        settings=runtime_settings,
        metadata=metadata,
        object_store=object_store,
        uploads=uploads,
        deletion=deletion,
        ingestion_reconciliation=ingestion_reconciliation,
        maintenance=MaintenanceService(deletion, ingestion_reconciliation),
        management=DocumentManagementService(metadata),
        search=search,
        answer=answer,
        query_history=query_history,
        visual_regions=VisualRegionService(metadata, object_store),
        summary=DocumentSummaryService(
            SQLiteSummarySource(metadata),
            search,
            LocalExtractiveSummary(),
            query_history,
            provider_configuration={
                "provider": "local-extractive",
                "kind": "deterministic",
                "revision": LocalExtractiveSummary.revision,
                "model": None,
            },
        ),
        model_summary=ModelDocumentSummaryService(
            SQLiteSummarySource(metadata),
            OllamaSummaryModel(
                runtime_settings.llm_base_url,
                runtime_settings.llm_model,
                timeout_seconds=runtime_settings.llm_timeout_seconds,
            ),
            query_history,
        )
        if runtime_settings.llm_provider == "ollama"
        else None,
    )


def _history_model_configuration(model) -> dict:
    # Explicit declared model/options only. Never dump provider attributes,
    # endpoint credentials, private paths or guessed usage/digests into history.
    if model is None:
        return {"provider": "none", "model": None, "digest": None}
    return {
        "provider": "local-ollama",
        "model": model.model,
        "digest": None,
        "requested_options": dict(model.options),
        "timeout_seconds": model.timeout_seconds,
        "contract": "grounding-v1" if hasattr(model, "decide") else "selection-v1",
    }


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Lite profile requires a sqlite:/// database URL")
    raw_path = database_url[len(prefix) :]
    if not raw_path:
        raise ValueError("SQLite database URL must include a file path")
    return Path(raw_path)
