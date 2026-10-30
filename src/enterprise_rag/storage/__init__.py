"""Storage adapters for the lite and core profiles."""

from .local_object_store import (
    DuplicateObject,
    LocalObjectStore,
    ObjectTooLarge,
    StoredObject,
)

__all__ = ["DuplicateObject", "LocalObjectStore", "ObjectTooLarge", "StoredObject"]
