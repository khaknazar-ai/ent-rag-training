"""Opt-in Qdrant publication and public semantic search over native authority."""

from hashlib import sha256

from enterprise_rag.persistence.sqlite_dense_publication import (
    DensePublicationError,
    SQLiteDensePublication,
    digest,
)
from enterprise_rag.retrieval.embeddings import validate_provider_output


class QdrantPublication:
    def __init__(self, metadata, backend, provider, cache):
        provider.space.require_compatible(backend.space)
        self.metadata, self.backend, self.provider, self.cache = (
            metadata,
            backend,
            provider,
            cache,
        )
        self.state = SQLiteDensePublication(metadata, backend.configuration)

    def _points(self, lease, signature):
        stored = self.metadata.get_text_artifacts(lease.job.run_id)
        version = self.metadata.get_version_for_tenant(
            lease.job.tenant_id, lease.job.document_version_id
        )
        if stored is None or version is None:
            raise DensePublicationError("Dense source missing")
        chunks = sorted(stored[1], key=lambda c: c.id)
        points = [
            self.backend.point(
                lease.job.tenant_id,
                version.document_id,
                version.id,
                lease.job.run_id,
                c.id,
                c.content_sha256,
                digest([signature, lease.worker_id, lease.lease_token]),
            )
            for c in chunks
        ]
        return points, chunks

    def publish(self, lease, clock):
        self.provider.space.require_compatible(self.backend.space)
        signature = self.state.reserve(lease, clock())
        points, chunks = self._points(lease, signature)
        self.backend.check_schema()
        vectors: list[tuple[float, ...]] = []

        def fence():
            self.state.verify_lease(lease, clock())

        for start in range(0, len(chunks), 8):
            fence()
            texts = tuple(c.text for c in chunks[start : start + 8])
            batch = (
                self.cache.embed(
                    lease.job.tenant_id, self.provider, texts, before_store=fence
                )
                if self.cache
                else validate_provider_output(
                    self.provider.space, texts, self.provider.embed(texts)
                )
            )
            fence()
            self.backend.upsert(points[start : start + 8], batch)
            vectors.extend(batch)
        remote = self.backend.verify(points, expected_vectors=vectors)
        self.state.seal(lease, signature, remote, clock())

    def verify(self, lease, clock):
        self.provider.space.require_compatible(self.backend.space)
        self.state.verify_lease(lease, clock())
        manifest = self.state.manifest(lease.job.run_id)
        if (manifest["lease_owner"], manifest["lease_token"]) != (
            lease.worker_id,
            lease.lease_token,
        ):
            raise DensePublicationError(
                "Replacement lease requires its own remote write epoch"
            )
        points, _ = self._points(lease, manifest["content_digest"])
        self.backend.check_schema()
        remote = self.backend.verify(points)
        if remote != manifest["remote_digest"]:
            raise DensePublicationError("Dense stored generation changed")
        # A reclaimed lease must independently reverify before native activation.
        self.state.seal(lease, manifest["content_digest"], remote, clock())

    def search(self, service, actor, query, limit):
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            return ()
        ids = self.metadata.list_authorized_dense_chunk_ids(scope)
        if not ids:
            return ()
        self.provider.space.require_compatible(self.backend.space)
        vector = service._embed((query,), actor=actor)[0]
        # Obtain current ACL/active generation after inference, before backend top-K.
        scope = self.metadata.resolve_security_scope(actor)
        if scope is None:
            return ()
        rows = []
        for start in range(0, len(ids), 100):
            rows.extend(
                self.metadata.get_authorized_evidence_for_candidates(
                    scope, ids[start : start + 100]
                )
            )
        points, canonical = [], {}
        manifests = {}
        for row in rows:
            if row.run_id not in manifests:
                manifests[row.run_id] = self.state.manifest(row.run_id)
            manifest = manifests[row.run_id]
            points.append(
                self.backend.point(
                    scope.tenant_id,
                    row.document_id,
                    row.document_version_id,
                    row.run_id,
                    row.chunk_id,
                    sha256(row.text.encode()).hexdigest(),
                    digest(
                        [
                            manifest["content_digest"],
                            manifest["lease_owner"],
                            manifest["lease_token"],
                        ]
                    ),
                )
            )
            canonical[row.chunk_id] = service.lexical._evidence_from_row(row)
        hits = self.backend.search(
            vector, points, tenant_id=scope.tenant_id, limit=limit
        )
        hydrated = service.lexical.hydrate_authorized_dense_candidates(
            actor, hits, limit=limit
        )
        return tuple(
            hit
            for hit in hydrated
            if service.lexical._same_canonical_evidence(
                canonical[hit.evidence.chunk_id], hit.evidence
            )
        )
