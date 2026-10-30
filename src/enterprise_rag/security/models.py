"""Immutable security context and permission contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PrincipalType(StrEnum):
    USER = "user"
    GROUP = "group"
    TENANT = "tenant"


class Permission(StrEnum):
    READ = "read"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class ActorContext:
    """The authenticated subject supplied to an application use case."""

    tenant_id: str
    user_id: str | None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("Actor tenant id must be non-empty")
        if self.user_id is not None and not self.user_id.strip():
            raise ValueError("Actor user id must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class SecurityScope:
    """Database-resolved principals and authorization revision for one request."""

    tenant_id: str
    user_id: str
    group_ids: tuple[str, ...]
    authorization_revision: int

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.user_id.strip():
            raise ValueError("Security scope requires tenant and user")
        if self.authorization_revision < 0:
            raise ValueError("Authorization revision must be non-negative")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("Security scope group ids must be unique")
        if any(not group_id.strip() for group_id in self.group_ids):
            raise ValueError("Security scope group ids must be non-empty")


@dataclass(frozen=True, slots=True)
class DocumentPermission:
    """Tenant-qualified document permission persisted by the metadata adapter."""

    document_id: str
    tenant_id: str
    principal_type: PrincipalType
    principal_id: str
    permission: Permission
