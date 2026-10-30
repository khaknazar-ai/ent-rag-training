"""Dependency-free domain objects for the first lifecycle slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ids import deterministic_id, sha256_bytes


class DocumentStatus(StrEnum):
    REGISTERED = "registered"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


class VersionStatus(StrEnum):
    UPLOADED = "uploaded"
    INGESTING = "ingesting"
    READY = "ready"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class Tenant:
    id: str
    name: str

    @classmethod
    def create(cls, slug: str, name: str) -> Tenant:
        return cls(deterministic_id("tenant", slug), name)


@dataclass(frozen=True, slots=True)
class User:
    id: str
    tenant_id: str
    email: str

    @classmethod
    def create(cls, tenant_id: str, email: str) -> User:
        return cls(
            deterministic_id("user", tenant_id, email.casefold()), tenant_id, email
        )


@dataclass(frozen=True, slots=True)
class Group:
    id: str
    tenant_id: str
    name: str

    @classmethod
    def create(cls, tenant_id: str, name: str) -> Group:
        return cls(
            deterministic_id("group", tenant_id, name.casefold()), tenant_id, name
        )


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    tenant_id: str
    name: str
    kind: str = "upload"

    @classmethod
    def create(cls, tenant_id: str, name: str, kind: str = "upload") -> Source:
        return cls(
            deterministic_id("source", tenant_id, kind, name), tenant_id, name, kind
        )


@dataclass(slots=True)
class Document:
    id: str
    tenant_id: str
    source_id: str
    canonical_key: str
    title: str
    status: DocumentStatus = DocumentStatus.REGISTERED
    active_version_id: str | None = None
    ingestion_warnings: tuple[str, ...] = ()

    @classmethod
    def create(
        cls, tenant_id: str, source_id: str, canonical_key: str, title: str
    ) -> Document:
        return cls(
            id=deterministic_id("document", tenant_id, source_id, canonical_key),
            tenant_id=tenant_id,
            source_id=source_id,
            canonical_key=canonical_key,
            title=title,
        )


@dataclass(slots=True)
class DocumentVersion:
    id: str
    document_id: str
    version_number: int
    filename: str
    content_sha256: str
    byte_size: int
    object_key: str
    status: VersionStatus = VersionStatus.UPLOADED

    @classmethod
    def from_content(
        cls,
        document_id: str,
        version_number: int,
        filename: str,
        content: bytes,
        object_key: str,
    ) -> DocumentVersion:
        return cls.from_digest(
            document_id=document_id,
            version_number=version_number,
            filename=filename,
            content_sha256=sha256_bytes(content),
            byte_size=len(content),
            object_key=object_key,
        )

    @classmethod
    def from_digest(
        cls,
        document_id: str,
        version_number: int,
        filename: str,
        content_sha256: str,
        byte_size: int,
        object_key: str,
    ) -> DocumentVersion:
        return cls(
            id=deterministic_id("document-version", document_id, content_sha256),
            document_id=document_id,
            version_number=version_number,
            filename=filename,
            content_sha256=content_sha256,
            byte_size=byte_size,
            object_key=object_key,
        )


@dataclass(frozen=True, slots=True)
class Page:
    id: str
    version_id: str
    page_number: int

    @classmethod
    def create(cls, version_id: str, page_number: int) -> Page:
        return cls(
            deterministic_id("page", version_id, page_number), version_id, page_number
        )


@dataclass(frozen=True, slots=True)
class DocumentElement:
    id: str
    page_id: str
    element_type: str
    ordinal: int
    text: str

    @classmethod
    def create(
        cls, page_id: str, element_type: str, ordinal: int, text: str
    ) -> DocumentElement:
        return cls(
            deterministic_id("element", page_id, element_type, ordinal),
            page_id,
            element_type,
            ordinal,
            text,
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    version_id: str
    element_id: str
    kind: str
    ordinal: int
    text: str
    parent_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        version_id: str,
        element_id: str,
        kind: str,
        ordinal: int,
        text: str,
        parent_id: str | None = None,
    ) -> Chunk:
        chunk_id = deterministic_id(
            "chunk", version_id, element_id, kind, ordinal, text
        )
        return cls(chunk_id, version_id, element_id, kind, ordinal, text, parent_id)
