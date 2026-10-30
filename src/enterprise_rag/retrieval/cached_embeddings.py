"""Tenant/content/space cache around actual provider calls, never authorization."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

from .embeddings import EmbeddingCacheKey, validate_provider_output
from .sqlite_embedding_cache import SQLiteEmbeddingCache


@dataclass(frozen=True)
class CacheStatistics:
    hits: int
    misses: int
    provider_texts: int


class CachedEmbeddings:
    def __init__(self, cache: SQLiteEmbeddingCache):
        self.cache = cache
        self._lock = RLock()
        self._hits = self._misses = self._provider_texts = 0

    @property
    def statistics(self) -> CacheStatistics:
        with self._lock:
            return CacheStatistics(self._hits, self._misses, self._provider_texts)

    def embed(
        self,
        tenant_id,
        provider,
        texts: Sequence[str],
        *,
        before_store: Callable[[], None],
    ):
        if isinstance(texts, (str, bytes)) or not 1 <= len(texts) <= 8:
            raise ValueError("Cached embedding requires1..8 texts")
        space = provider.space
        keys = tuple(
            EmbeddingCacheKey.from_normalized_content(tenant_id, space, text)
            for text in texts
        )
        # This admission lock never holds a database connection/transaction.
        with self._lock:
            # Authorization can change while another request owns admission.
            before_store()
            space.require_compatible(provider.space)
            values: dict[EmbeddingCacheKey, tuple[float, ...]] = {}
            missing: dict[EmbeddingCacheKey, str] = {}
            for key, text in zip(keys, texts, strict=True):
                value = self.cache.get(key)
                if value is None:
                    self._misses += 1
                    missing.setdefault(key, text)
                else:
                    self._hits += 1
                    values[key] = value
            if missing:
                inputs = tuple(missing.values())
                before_store()
                space.require_compatible(provider.space)
                vectors = validate_provider_output(
                    space, inputs, provider.embed(inputs)
                )
                space.require_compatible(provider.space)
                self._provider_texts += len(inputs)
                # Live canonical/actor fence after latency and before persistence.
                before_store()
                for key, vector in zip(missing, vectors, strict=True):
                    values[key] = self.cache.put(key, vector)
            before_store()
            space.require_compatible(provider.space)
            return tuple(values[key] for key in keys)
