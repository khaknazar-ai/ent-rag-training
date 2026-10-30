"""Bounded loopback Qdrant REST adapter; references only, never source quotes."""

import json
import math
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

from enterprise_rag.persistence.sqlite_dense_publication import digest
from enterprise_rag.retrieval.dense import DenseHit
from enterprise_rag.retrieval.embeddings import validate_vector


class QdrantError(RuntimeError):
    """Explicit backend failure; no lexical/local-vector fallback."""


class QdrantDenseRetriever:
    schema = "canonical-reference-v1"

    def __init__(self, endpoint, namespace, space, *, transport=None):
        url = urlsplit(endpoint)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.path not in ("", "/")
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError("Qdrant requires an explicit loopback HTTP port")
        namespace = str(UUID(namespace))
        if space.metric != "cosine" or not 1 <= space.dimension <= 1024:
            raise ValueError("Unsupported Qdrant embedding space")
        self.space = space
        self.space_key = digest(space.compatibility_key)
        self.collection = f"rag_{namespace.replace('-', '')}_{self.space_key[:24]}_v1"
        self.configuration = {
            "endpoint": endpoint.rstrip("/"),
            "namespace": namespace,
            "space": list(space.compatibility_key),
            "schema": self.schema,
        }
        self.client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            timeout=10,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    def close(self):
        self.client.close()

    def _request(self, method, suffix="", body=None):
        try:
            with self.client.stream(
                method, f"/collections/{self.collection}{suffix}", json=body
            ) as response:
                response.raise_for_status()
                data = bytearray()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > 4 * 1024 * 1024:
                        raise QdrantError("Qdrant response exceeds budget")
                value = json.loads(data)
                if value.get("status") != "ok" or "result" not in value:
                    raise QdrantError("Malformed Qdrant response")
                return value["result"]
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
            raise QdrantError("Qdrant request failed") from error

    def initialize(self):
        # Explicit creation only. Query/reopen never recreates a lost collection.
        self._request(
            "PUT",
            body={
                "vectors": {
                    "dense": {"size": self.space.dimension, "distance": "Cosine"}
                },
                "shard_number": 1,
                "replication_factor": 1,
                "on_disk_payload": True,
            },
        )
        for key in ("tenant_id", "run_id", "space", "schema", "generation"):
            self._request(
                "PUT",
                "/index?wait=true",
                {"field_name": key, "field_schema": "keyword"},
            )
        self.check_schema()

    def check_schema(self):
        result = self._request("GET")
        try:
            vectors = result["config"]["params"]["vectors"]
            valid = (
                set(vectors) == {"dense"}
                and vectors["dense"]["size"] == self.space.dimension
                and vectors["dense"]["distance"] == "Cosine"
            )
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise QdrantError("Qdrant named vector schema mismatch")

    def point(
        self,
        tenant_id,
        document_id,
        version_id,
        run_id,
        chunk_id,
        content_hash,
        generation,
    ):
        payload = dict(
            tenant_id=tenant_id,
            document_id=document_id,
            document_version_id=version_id,
            run_id=run_id,
            chunk_id=chunk_id,
            content_sha256=content_hash,
            generation=generation,
            space=self.space_key,
            schema=self.schema,
        )
        return {
            "id": str(uuid5(NAMESPACE_URL, digest([self.configuration, payload]))),
            "payload": payload,
        }

    def upsert(self, points, vectors):
        if not 1 <= len(points) <= 8 or len(points) != len(vectors):
            raise ValueError("Qdrant upsert requires1..8 complete points")
        body = []
        for point, vector in zip(points, vectors, strict=True):
            vector = validate_vector(vector, expected_dimension=self.space.dimension)
            if math.hypot(*vector) == 0:
                raise ValueError("Zero cosine vector")
            body.append({**point, "vector": {"dense": vector}})
        result = self._request("PUT", "/points?wait=true", {"points": body})
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise QdrantError("Qdrant write not completed")

    def verify(self, points, *, expected_vectors=None):
        if not 1 <= len(points) <= 1024:
            raise ValueError("Qdrant verification membership exceeds budget")
        received = []
        for start in range(0, len(points), 8):
            batch = points[start : start + 8]
            result = self._request(
                "POST",
                "/points",
                {
                    "ids": [p["id"] for p in batch],
                    "with_payload": True,
                    "with_vector": True,
                },
            )
            if not isinstance(result, list) or len(result) != len(batch):
                raise QdrantError("Qdrant generation membership incomplete")
            by_id = {r["id"]: r for r in result}
            if set(by_id) != {p["id"] for p in batch}:
                raise QdrantError("Qdrant generation membership differs")
            for offset, point in enumerate(batch):
                remote = by_id[point["id"]]
                if remote.get("payload") != point["payload"]:
                    raise QdrantError("Qdrant reference payload mismatch")
                vectors = remote.get("vector")
                if not isinstance(vectors, dict) or set(vectors) != {"dense"}:
                    raise QdrantError("Qdrant vector schema mismatch")
                vector = validate_vector(
                    vectors["dense"], expected_dimension=self.space.dimension
                )
                if not math.isclose(math.hypot(*vector), 1, abs_tol=1e-5):
                    raise QdrantError("Qdrant vector is not normalized")
                if expected_vectors is not None:
                    expected = expected_vectors[start + offset]
                    norm = math.hypot(*expected)
                    if norm == 0 or any(
                        abs(a - b / norm) > 1e-5
                        for a, b in zip(vector, expected, strict=True)
                    ):
                        raise QdrantError("Qdrant stored vector mismatch")
                received.append({**point, "vector": vector})
        return digest(received)

    def search(self, vector, points, *, tenant_id, limit):
        if not points:
            return ()
        if (
            not 1 <= len(points) <= 1024
            or type(limit) is not int
            or not 1 <= limit <= 20
        ):
            raise ValueError("Qdrant query exceeds budget")
        vector = validate_vector(vector, expected_dimension=self.space.dimension)
        by_id = {p["id"]: p for p in points}
        if len(by_id) != len(points) or any(
            p["payload"]["tenant_id"] != tenant_id for p in points
        ):
            raise ValueError("Invalid authorized Qdrant membership")
        self.check_schema()
        result = self._request(
            "POST",
            "/points/query",
            {
                "query": vector,
                "using": "dense",
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [
                        {"has_id": list(by_id)},
                        {"key": "tenant_id", "match": {"value": tenant_id}},
                        {"key": "space", "match": {"value": self.space_key}},
                        {"key": "schema", "match": {"value": self.schema}},
                        {
                            "key": "run_id",
                            "match": {
                                "any": sorted({p["payload"]["run_id"] for p in points})
                            },
                        },
                        {
                            "key": "generation",
                            "match": {
                                "any": sorted(
                                    {p["payload"]["generation"] for p in points}
                                )
                            },
                        },
                    ]
                },
            },
        )
        rows = result.get("points") if isinstance(result, dict) else None
        if not isinstance(rows, list) or len(rows) != min(limit, len(points)):
            raise QdrantError("Malformed Qdrant query result")
        hits, seen = [], set()
        for rank, row in enumerate(rows, 1):
            point = by_id.get(row.get("id"))
            if (
                point is None
                or row["id"] in seen
                or row.get("payload") != point["payload"]
            ):
                raise QdrantError(
                    "Qdrant returned an unauthorized or changed reference"
                )
            seen.add(row["id"])
            score = row.get("score")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise QdrantError("Invalid Qdrant score")
            p = point["payload"]
            hits.append(
                DenseHit(
                    p["chunk_id"],
                    p["document_id"],
                    p["document_version_id"],
                    p["run_id"],
                    score,
                    rank,
                )
            )
        return tuple(hits)
