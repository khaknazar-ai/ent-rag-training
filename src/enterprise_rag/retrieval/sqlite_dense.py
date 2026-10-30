"""Explicit immutable SQLite vector snapshots for bounded local retrieval.

This stores references/vectors, never document text or authorization facts.
Ready snapshots cannot be edited; canonical state is rechecked at query time.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .dense import VectorRecord
from .embeddings import EmbeddingSpace, validate_vector

_APP_ID = 0x45524456
_VERSION = 1
MAX_RECORDS = 2048
MAX_DIMENSION = 1024
MAX_VECTOR_BYTES = 32768
MAX_INDEX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CanonicalVector:
    reference: VectorRecord
    content_sha256: str

    def __post_init__(self) -> None:
        digest = self.content_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Canonical content hash must be lowercase SHA256")


class DenseSnapshotError(ValueError):
    """An explicit incompatible, incomplete or corrupt index refusal."""


def _space_json(space: EmbeddingSpace) -> str:
    if space.metric != "cosine" or not 1 <= space.dimension <= MAX_DIMENSION:
        raise ValueError("Local dense snapshot requires cosine and1..1024 dimensions")
    if any(
        len(value.encode()) > 512
        for value in (
            space.provider_id,
            space.model_revision,
            space.preprocessing_revision,
        )
    ):
        raise ValueError("Dense embedding identity exceeds its byte budget")
    return json.dumps(asdict(space), sort_keys=True, separators=(",", ":"))


class SQLiteDenseSnapshot:
    def __init__(self, path: Path, space: EmbeddingSpace) -> None:
        self.path = Path(path).resolve()
        self.space = space
        self._space = _space_json(space)
        if not self.path.is_file():
            raise FileNotFoundError("An existing private dense snapshot is required")
        with closing(self._connect()) as db:
            self._meta(db)

    @classmethod
    def create(
        cls, path: Path, space: EmbeddingSpace, *, max_records: int = 1024
    ) -> SQLiteDenseSnapshot:
        encoded = _space_json(space)
        if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
            raise ValueError("Dense snapshot capacity must be1..2048")
        target = Path(path).resolve()
        with target.open("xb"):
            pass
        with closing(sqlite3.connect(target, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "CREATE TABLE dense_meta (singleton INTEGER PRIMARY KEY "
                    "CHECK(singleton=1), space TEXT NOT NULL, capacity INTEGER NOT NULL, "
                    "state TEXT NOT NULL CHECK(state IN ('building','ready')), "
                    "records_digest TEXT)"
                )
                db.execute(
                    "CREATE TABLE dense_vectors (chunk TEXT PRIMARY KEY, tenant TEXT "
                    "NOT NULL, document TEXT NOT NULL, version TEXT NOT NULL, run TEXT "
                    "NOT NULL, content_hash TEXT NOT NULL, vector_json TEXT NOT NULL "
                    "CHECK(length(vector_json)<=32768), vector_hash TEXT NOT NULL)"
                )
                db.execute(
                    "INSERT INTO dense_meta VALUES(1,?,?,'building',NULL)",
                    (encoded, max_records),
                )
                db.execute(f"PRAGMA application_id={_APP_ID}")
                db.execute(f"PRAGMA user_version={_VERSION}")
                db.commit()
            except Exception:
                db.rollback()
                raise
        return cls(target, space)

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        wal = self.path.with_name(self.path.name + "-wal")
        if (
            self.path.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
            > MAX_INDEX_BYTES
        ):
            raise DenseSnapshotError("Dense snapshot file budget exceeded")
        db = sqlite3.connect(
            self.path.as_uri() + ("?mode=rw" if write else "?mode=ro"),
            uri=True,
            timeout=2,
            isolation_level=None,
        )
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA trusted_schema=OFF")
        return db

    def _meta(self, db: sqlite3.Connection) -> sqlite3.Row:
        if (
            db.execute("PRAGMA application_id").fetchone()[0] != _APP_ID
            or db.execute("PRAGMA user_version").fetchone()[0] != _VERSION
        ):
            raise DenseSnapshotError("Unknown dense snapshot format")
        try:
            rows = db.execute(
                "SELECT singleton, CASE WHEN typeof(space)='text' AND "
                "length(CAST(space AS BLOB))<=4096 THEN space END AS space, "
                "capacity, CASE WHEN typeof(state)='text' AND "
                "length(CAST(state AS BLOB))<=16 THEN state END AS state, "
                "CASE WHEN typeof(records_digest)='text' AND "
                "length(CAST(records_digest AS BLOB))<=64 THEN records_digest END "
                "AS records_digest FROM dense_meta LIMIT 2"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise DenseSnapshotError("Invalid dense snapshot metadata") from error
        if (
            len(rows) != 1
            or rows[0]["singleton"] != 1
            or rows[0]["space"] != self._space
            or type(rows[0]["capacity"]) is not int
            or not 1 <= rows[0]["capacity"] <= MAX_RECORDS
            or rows[0]["state"] not in {"building", "ready"}
        ):
            raise DenseSnapshotError("Incompatible dense snapshot identity or capacity")
        return rows[0]

    def put(self, item: CanonicalVector) -> None:
        ref = item.reference
        if any(
            not isinstance(value, str)
            or not value
            or len(value.encode()) > 512
            or "\0" in value
            for value in (
                ref.chunk_id,
                ref.tenant_id,
                ref.document_id,
                ref.document_version_id,
                ref.run_id,
            )
        ):
            raise ValueError("Dense reference identity exceeds its byte policy")
        vector = validate_vector(ref.vector, expected_dimension=self.space.dimension)
        raw = json.dumps(vector, separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > MAX_VECTOR_BYTES:
            raise ValueError("Dense vector exceeds its serialized budget")
        values = (
            ref.chunk_id,
            ref.tenant_id,
            ref.document_id,
            ref.document_version_id,
            ref.run_id,
            item.content_sha256,
            raw,
            sha256(raw.encode()).hexdigest(),
        )
        with closing(self._connect(write=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                meta = self._meta(db)
                self._preflight_rows(db, meta["capacity"])
                if meta["state"] != "building":
                    raise DenseSnapshotError("Ready dense snapshots are immutable")
                existing = db.execute(
                    "SELECT * FROM dense_vectors WHERE chunk=?", (ref.chunk_id,)
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != values:
                        raise DenseSnapshotError("Conflicting immutable dense record")
                else:
                    count = db.execute("SELECT count(*) FROM dense_vectors").fetchone()[
                        0
                    ]
                    if count >= meta["capacity"]:
                        raise DenseSnapshotError("Dense snapshot capacity exhausted")
                    db.execute(
                        "INSERT INTO dense_vectors VALUES(?,?,?,?,?,?,?,?)", values
                    )
                db.commit()
            except Exception:
                db.rollback()
                raise

    def seal(self, *, expected_records: int) -> None:
        if (
            type(expected_records) is not int
            or not 1 <= expected_records <= MAX_RECORDS
        ):
            raise ValueError("Expected dense record count must be1..2048")
        with closing(self._connect(write=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                meta = self._meta(db)
                count = self._preflight_rows(db, meta["capacity"])
                if count != expected_records:
                    raise DenseSnapshotError("Dense snapshot is incomplete")
                rows = db.execute(
                    "SELECT * FROM dense_vectors ORDER BY chunk LIMIT ?",
                    (meta["capacity"] + 1,),
                ).fetchall()
                if len(rows) != expected_records or len(rows) > meta["capacity"]:
                    raise DenseSnapshotError("Dense snapshot is incomplete")
                digest = self._records_digest(rows)
                if meta["state"] == "ready" and meta["records_digest"] != digest:
                    raise DenseSnapshotError("Ready dense snapshot checksum mismatch")
                db.execute(
                    "UPDATE dense_meta SET state='ready',records_digest=? WHERE singleton=1",
                    (digest,),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise

    def _records_digest(self, rows: list[sqlite3.Row]) -> str:
        digest = sha256()
        for row in rows:
            digest.update(json.dumps(tuple(row), separators=(",", ":")).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    def _preflight_rows(self, db: sqlite3.Connection, capacity: int) -> int:
        count = db.execute("SELECT count(*) FROM dense_vectors").fetchone()[0]
        if count > capacity:
            raise DenseSnapshotError("Dense snapshot row capacity exceeded")
        identity_checks = " OR ".join(
            f"typeof({field})!='text' OR length(CAST({field} AS BLOB)) NOT BETWEEN 1 AND 512 OR instr({field},char(0))!=0"
            for field in ("chunk", "tenant", "document", "version", "run")
        )
        invalid = db.execute(
            "SELECT 1 FROM dense_vectors WHERE "
            + identity_checks
            + " OR typeof(vector_json)!='text' OR length(CAST(vector_json AS BLOB)) NOT BETWEEN 1 AND 32768 "
            "OR instr(vector_json,char(0))!=0 OR typeof(content_hash)!='text' "
            "OR length(CAST(content_hash AS BLOB))!=64 OR content_hash GLOB '*[^0-9a-f]*' "
            "OR typeof(vector_hash)!='text' OR length(CAST(vector_hash AS BLOB))!=64 "
            "OR vector_hash GLOB '*[^0-9a-f]*' LIMIT 1"
        ).fetchone()
        if invalid is not None:
            raise DenseSnapshotError(
                "Dense snapshot field type or byte budget exceeded"
            )
        return count

    def records(self, tenant_id: str) -> tuple[CanonicalVector, ...]:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Dense tenant identity is required")
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            meta = self._meta(db)
            if meta["state"] != "ready":
                raise DenseSnapshotError("Dense snapshot is not ready")
            self._preflight_rows(db, meta["capacity"])
            rows = db.execute(
                "SELECT * FROM dense_vectors ORDER BY chunk LIMIT ?",
                (meta["capacity"] + 1,),
            ).fetchall()
            if (
                len(rows) > meta["capacity"]
                or self._records_digest(rows) != meta["records_digest"]
            ):
                raise DenseSnapshotError(
                    "Dense snapshot record set changed or exceeded cap"
                )
        items = []
        for row in rows:
            raw = row["vector_json"]
            if (
                not isinstance(raw, str)
                or len(raw.encode()) > MAX_VECTOR_BYTES
                or sha256(raw.encode()).hexdigest() != row["vector_hash"]
            ):
                raise DenseSnapshotError("Dense vector checksum or budget mismatch")
            if row["tenant"] != tenant_id:
                continue
            try:
                vector = validate_vector(
                    json.loads(raw), expected_dimension=self.space.dimension
                )
                ref = VectorRecord(
                    row["chunk"],
                    row["tenant"],
                    row["document"],
                    row["version"],
                    row["run"],
                    vector,
                )
                items.append(CanonicalVector(ref, row["content_hash"]))
            except (ValueError, TypeError, AttributeError) as error:
                raise DenseSnapshotError("Malformed dense vector record") from error
        return tuple(items)
