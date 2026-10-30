"""Explicit document and version state transitions."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Document, DocumentStatus, DocumentVersion, VersionStatus


class InvalidTransition(ValueError):
    """Raised when a lifecycle invariant would be violated."""


_DOCUMENT_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.REGISTERED: frozenset(
        {DocumentStatus.INGESTING, DocumentStatus.DELETING}
    ),
    DocumentStatus.INGESTING: frozenset(
        {DocumentStatus.READY, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.READY: frozenset(
        {DocumentStatus.INGESTING, DocumentStatus.DELETING}
    ),
    DocumentStatus.FAILED: frozenset(
        {DocumentStatus.INGESTING, DocumentStatus.DELETING}
    ),
    DocumentStatus.DELETING: frozenset({DocumentStatus.DELETED}),
    DocumentStatus.DELETED: frozenset(),
}

_VERSION_TRANSITIONS: dict[VersionStatus, frozenset[VersionStatus]] = {
    VersionStatus.UPLOADED: frozenset(
        {VersionStatus.INGESTING, VersionStatus.DELETING}
    ),
    VersionStatus.INGESTING: frozenset(
        {VersionStatus.READY, VersionStatus.FAILED, VersionStatus.DELETING}
    ),
    VersionStatus.READY: frozenset({VersionStatus.ACTIVE, VersionStatus.DELETING}),
    VersionStatus.ACTIVE: frozenset({VersionStatus.SUPERSEDED, VersionStatus.DELETING}),
    VersionStatus.SUPERSEDED: frozenset({VersionStatus.DELETING}),
    VersionStatus.FAILED: frozenset({VersionStatus.INGESTING, VersionStatus.DELETING}),
    VersionStatus.DELETING: frozenset({VersionStatus.DELETED}),
    VersionStatus.DELETED: frozenset(),
}


def transition_document(document: Document, target: DocumentStatus) -> None:
    if target not in _DOCUMENT_TRANSITIONS[document.status]:
        raise InvalidTransition(
            f"Document {document.id}: {document.status} -> {target} is not allowed"
        )
    document.status = target


def transition_version(version: DocumentVersion, target: VersionStatus) -> None:
    if target not in _VERSION_TRANSITIONS[version.status]:
        raise InvalidTransition(
            f"Version {version.id}: {version.status} -> {target} is not allowed"
        )
    version.status = target


def activate_version(
    document: Document,
    version: DocumentVersion,
    versions: Iterable[DocumentVersion],
) -> None:
    """Activate exactly one ready version and supersede the previous active version."""

    if version.document_id != document.id:
        raise InvalidTransition(
            "Cannot activate a version belonging to another document"
        )
    if document.status not in {DocumentStatus.INGESTING, DocumentStatus.READY}:
        raise InvalidTransition(
            f"Document {document.id} is not activatable from {document.status}"
        )
    if version.status is not VersionStatus.READY:
        raise InvalidTransition(
            f"Only READY versions can become ACTIVE; found {version.status}"
        )

    related = [
        candidate for candidate in versions if candidate.document_id == document.id
    ]
    if not any(candidate.id == version.id for candidate in related):
        raise InvalidTransition("Version must be present in the related version set")

    active = [
        candidate for candidate in related if candidate.status is VersionStatus.ACTIVE
    ]
    if len(active) > 1:
        raise InvalidTransition(
            f"Document {document.id} already has multiple ACTIVE versions"
        )
    if document.active_version_id is not None and (
        not active or active[0].id != document.active_version_id
    ):
        raise InvalidTransition("Document active pointer does not match its versions")

    for candidate in related:
        if candidate.id == version.id:
            continue
        if candidate.status is VersionStatus.ACTIVE:
            candidate.status = VersionStatus.SUPERSEDED

    version.status = VersionStatus.ACTIVE
    document.active_version_id = version.id
    document.status = DocumentStatus.READY

    active = [
        candidate for candidate in related if candidate.status is VersionStatus.ACTIVE
    ]
    if len(active) != 1 or active[0].id != version.id:
        raise InvalidTransition("Version activation invariant violated")
