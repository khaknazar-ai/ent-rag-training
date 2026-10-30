"""Document read use cases against a storage-independent authorization port."""

from __future__ import annotations

from enterprise_rag.domain.documents.models import Document, DocumentVersion
from enterprise_rag.domain.ingestion.models import IngestionJob
from enterprise_rag.ports.document_catalog import DocumentCatalog
from enterprise_rag.security import ActorContext


class DocumentCatalogService:
    def __init__(self, catalog: DocumentCatalog) -> None:
        self.catalog = catalog

    def list_documents(
        self, actor: ActorContext | None, *, limit: int = 50, offset: int = 0
    ) -> tuple[Document, ...]:
        scope = self.catalog.resolve_security_scope(actor)
        if scope is None:
            return ()
        return self.catalog.list_documents_authorized(scope, limit=limit, offset=offset)

    def get_document(
        self, actor: ActorContext | None, document_id: str
    ) -> Document | None:
        scope = self.catalog.resolve_security_scope(actor)
        if scope is None:
            return None
        return self.catalog.get_document_authorized(scope, document_id)

    def list_versions(
        self, actor: ActorContext | None, document_id: str
    ) -> tuple[DocumentVersion, ...] | None:
        scope = self.catalog.resolve_security_scope(actor)
        if (
            scope is None
            or self.catalog.get_document_authorized(scope, document_id) is None
        ):
            return None
        return self.catalog.list_versions_authorized(scope, document_id)

    def get_job(self, actor: ActorContext | None, job_id: str) -> IngestionJob | None:
        scope = self.catalog.resolve_security_scope(actor)
        if scope is None:
            return None
        return self.catalog.get_job_authorized(scope, job_id)
