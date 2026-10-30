"""Authorized lite retrieval adapters."""

from .dense import DenseHit, ExactVectorIndex, VectorRecord
from .embeddings import (
    EmbeddingCacheKey,
    EmbeddingProvider,
    EmbeddingSpace,
    InMemoryEmbeddingCache,
    validate_embedding_batch,
    validate_provider_output,
    validate_vector,
)
from .fusion import (
    CandidateIdentity,
    FusedCandidate,
    FusionContribution,
    RankedCandidate,
    reciprocal_rank_fusion,
)
from .lexical import LexicalHit, build_literal_fts_query, search_lite
from .parents import (
    ChildCitation,
    ChildReference,
    ExpandedParent,
    ParentExpansionResult,
    ParentRecord,
    expand_parents,
)

__all__ = [
    "CandidateIdentity",
    "ChildCitation",
    "ChildReference",
    "DenseHit",
    "EmbeddingCacheKey",
    "EmbeddingProvider",
    "EmbeddingSpace",
    "ExactVectorIndex",
    "ExpandedParent",
    "FusedCandidate",
    "FusionContribution",
    "InMemoryEmbeddingCache",
    "LexicalHit",
    "ParentExpansionResult",
    "ParentRecord",
    "RankedCandidate",
    "VectorRecord",
    "build_literal_fts_query",
    "expand_parents",
    "reciprocal_rank_fusion",
    "search_lite",
    "validate_embedding_batch",
    "validate_provider_output",
    "validate_vector",
]
