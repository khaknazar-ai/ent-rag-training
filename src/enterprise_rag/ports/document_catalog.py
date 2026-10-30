"""Read-only metadata boundary for document catalog and lifecycle inspection.

Adapters must enforce current tenant/READ filters inside their queries, before
pagination, and reject stale authorization scopes. No raw SQL or connections
cross this port. Mutations and processing authority remain separate contracts.
"""

from __future__ import annotations

from typing import Protocol

from enterprise_rag.domain.documents.models import Document, DocumentVersion
from enterprise_rag.domain.ingestion.models import IngestionJob
from enterprise_rag.security import ActorContext, SecurityScope


class DocumentCatalog(Protocol):
    def resolve_security_scope(
        self, actor: ActorContext | None
    ) -> SecurityScope | None: ...
    def list_documents_authorized(
        self, scope: SecurityScope, *, limit: int = 50, offset: int = 0
    ) -> tuple[Document, ...]: ...
    def get_document_authorized(
        self, scope: SecurityScope, document_id: str
    ) -> Document | None: ...
    def list_versions_authorized(
        self, scope: SecurityScope, document_id: str
    ) -> tuple[DocumentVersion, ...]: ...
    def get_job_authorized(
        self, scope: SecurityScope, job_id: str
    ) -> IngestionJob | None: ...
