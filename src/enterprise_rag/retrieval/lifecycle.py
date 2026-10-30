"""Bounded actor-scoped publication of immutable, derived dense generations."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from enterprise_rag.retrieval.sqlite_dense import (
    DenseSnapshotError,
    SQLiteDenseSnapshot,
)

APP_ID = 0x45524C43
GENERATION = re.compile(r"generation-[0-9a-f]{32}\.sqlite3\Z")


class DenseLifecycleError(DenseSnapshotError):
    """Busy, stale, incomplete or over-budget managed view; no stale fallback."""


class DenseSnapshotLifecycle:
    def __init__(
        self,
        directory: Path,
        *,
        max_records: int = 1024,
        max_generations: int = 32,
        max_bytes: int = 128 * 1024 * 1024,
    ):
        if (
            type(max_records) is not int
            or not 1 <= max_records <= 2048
            or type(max_generations) is not int
            or not 1 <= max_generations <= 32
            or type(max_bytes) is not int
            or not 65536 <= max_bytes <= 128 * 1024 * 1024
        ):
            raise ValueError("Invalid bounded dense lifecycle policy")
        path = Path(directory)
        if path.is_symlink():
            raise DenseLifecycleError("Managed index directory must not be a symlink")
        self.directory = path.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_records, self.max_generations, self.max_bytes = (
            max_records,
            max_generations,
            max_bytes,
        )
        self.catalog = self.directory / "catalog.sqlite3"
        self._budget()
        try:
            with self.catalog.open("xb"):
                pass
        except FileExistsError:
            pass
        else:
            with closing(sqlite3.connect(self.catalog, isolation_level=None)) as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE active (actor TEXT PRIMARY KEY, signature TEXT NOT NULL, generation TEXT, space TEXT NOT NULL)"
                )
                db.execute(f"PRAGMA application_id={APP_ID}")
                db.execute("PRAGMA user_version=1")
                db.commit()
        with closing(self._connect()) as db:
            self._check(db)

    def _budget(self):
        total = generations = entries = 0
        with os.scandir(self.directory) as listing:
            for entry in listing:
                entries += 1
                if (
                    entries > 37
                    or entry.is_symlink()
                    or not entry.is_file(follow_symlinks=False)
                ):
                    raise DenseLifecycleError(
                        "Managed index directory contains unexpected entries"
                    )
                if GENERATION.fullmatch(entry.name):
                    generations += 1
                elif entry.name not in {
                    "catalog.sqlite3",
                    "catalog.sqlite3-journal",
                    "catalog.sqlite3-wal",
                    "catalog.sqlite3-shm",
                    "build.lock",
                }:
                    raise DenseLifecycleError(
                        "Managed index directory contains an unknown file"
                    )
                total += entry.stat(follow_symlinks=False).st_size
        if total > self.max_bytes or generations > self.max_generations:
            raise DenseLifecycleError("Managed dense disk/generation budget exceeded")
        return total, generations

    @contextmanager
    def _admit_build(self):
        # Cross-process admission covers the whole reconciliation, including
        # native work outside SQL transactions. Future generation bytes cannot
        # be double-admitted against the same remaining budget.
        marker = self.directory / "build.lock"
        try:
            stream = marker.open("xb")
        except FileExistsError as error:
            raise DenseLifecycleError("Managed generation builder busy") from error
        try:
            yield
        finally:
            stream.close()
            marker.unlink()

    def _connect(self):
        if self.catalog.is_symlink() or self.catalog.stat().st_size > 2 * 1024 * 1024:
            raise DenseLifecycleError("Managed catalog exceeds its file policy")
        db = sqlite3.connect(
            self.catalog.as_uri() + "?mode=rw",
            uri=True,
            isolation_level=None,
            timeout=0,
        )
        db.execute("PRAGMA trusted_schema=OFF")
        return db

    def _check(self, db):
        if (
            db.execute("PRAGMA application_id").fetchone()[0] != APP_ID
            or db.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise DenseLifecycleError("Foreign managed catalog")
        if db.execute("SELECT count(*) FROM active").fetchone()[0] > 16:
            raise DenseLifecycleError("Managed actor capacity exceeded")

    @staticmethod
    def _key(actor):
        return sha256(
            json.dumps([actor.tenant_id, actor.user_id], separators=(",", ":")).encode()
        ).hexdigest()

    def _row(self, db, key):
        self._check(db)
        row = db.execute(
            "SELECT CASE WHEN typeof(signature)='text' AND length(CAST(signature AS BLOB))=64 THEN signature END, CASE WHEN typeof(generation)='text' AND length(CAST(generation AS BLOB))=51 THEN generation END, CASE WHEN typeof(space)='text' AND length(CAST(space AS BLOB)) BETWEEN 1 AND 2048 AND instr(space,char(0))=0 THEN space END, generation IS NULL FROM active WHERE actor=?",
            (key,),
        ).fetchone()
        if row is not None and (
            row[0] is None
            or row[2] is None
            or any(c not in "0123456789abcdef" for c in row[0])
            or (row[1] is None and not row[3])
            or (row[1] is not None and not GENERATION.fullmatch(row[1]))
        ):
            raise DenseLifecycleError("Invalid managed generation pointer")
        return row

    def current(self, actor):
        """Diagnostic pointer only; serving always reconciles and reauthorizes."""
        self._budget()
        with closing(self._connect()) as db:
            row = self._row(db, self._key(actor))
        if row is None or row[1] is None:
            return None
        from enterprise_rag.retrieval.embeddings import EmbeddingSpace

        return SQLiteDenseSnapshot(
            self.directory / row[1], EmbeddingSpace(**json.loads(row[2]))
        )

    def _view(self, lexical, actor):
        scope = lexical.metadata.resolve_security_scope(actor)
        if scope is None:
            return (), sha256(b"unauthorized").hexdigest()
        try:
            ids = lexical.metadata.list_authorized_dense_chunk_ids(
                scope, limit=self.max_records
            )
        except ValueError as error:
            raise DenseLifecycleError("Canonical dense view exceeds policy") from error
        rows = []
        for offset in range(0, len(ids), 100):
            rows.extend(
                lexical.metadata.get_authorized_evidence_for_candidates(
                    scope, ids[offset : offset + 100]
                )
            )
        if tuple(row.chunk_id for row in rows) != ids:
            raise DenseLifecycleError(
                "Canonical membership changed or failed hydration"
            )
        digest = sha256(
            json.dumps(
                [
                    asdict(scope),
                    [
                        (
                            r.chunk_id,
                            r.document_id,
                            r.document_version_id,
                            r.run_id,
                            r.title,
                            r.filename,
                            r.page_numbers,
                            sha256(r.text.encode()).hexdigest(),
                        )
                        for r in rows
                    ],
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ids, digest

    def reconcile(self, lexical, provider, actor, *, embedding_cache=None):
        """Rebuild changed views; never expose a partial or superseded generation."""
        from enterprise_rag.application.semantic import LocalSemanticService

        key = self._key(actor)
        space = json.dumps(
            asdict(provider.space), sort_keys=True, separators=(",", ":")
        )
        try:
            self._budget()
            with self._admit_build(), closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                row = self._row(db, key)
                ids, signature = self._view(lexical, actor)
                if signature == sha256(b"unauthorized").hexdigest():
                    db.execute("DELETE FROM active WHERE actor=?", (key,))
                    db.commit()
                    return None
                if row is not None and row[0] == signature and row[2] == space:
                    if bool(ids) != (row[1] is not None):
                        raise DenseLifecycleError(
                            "Generation pointer contradicts membership"
                        )
                    db.commit()
                    return (
                        None
                        if row[1] is None
                        else SQLiteDenseSnapshot(
                            self.directory / row[1], provider.space
                        )
                    )
                # Persist invalidation before building. Failure cannot restore
                # an old serving pointer, while its immutable file is retained.
                db.execute("DELETE FROM active WHERE actor=?", (key,))
                db.commit()
                db.execute("BEGIN IMMEDIATE")
                self._check(db)
                if db.execute("SELECT count(*) FROM active").fetchone()[0] >= 16:
                    raise DenseLifecycleError("Managed actor capacity exhausted")
                total, generations = self._budget()
                name, snapshot = None, None
                if ids:
                    if (
                        generations >= self.max_generations
                        or total + len(ids) * 32768 + 65536 > self.max_bytes
                    ):
                        raise DenseLifecycleError(
                            "Managed generation retention budget exhausted"
                        )
                    name = "generation-" + uuid4().hex + ".sqlite3"
                    snapshot = SQLiteDenseSnapshot.create(
                        self.directory / name,
                        provider.space,
                        max_records=self.max_records,
                    )
                    # No catalog or canonical write transaction spans inference.
                    db.commit()
                    LocalSemanticService(
                        lexical, provider, snapshot, embedding_cache=embedding_cache
                    ).index_authorized_chunks(actor, ids)
                    db.execute("BEGIN IMMEDIATE")
                if self._view(lexical, actor) != (ids, signature):
                    raise DenseLifecycleError(
                        "Canonical view changed during generation build"
                    )
                self._budget()
                if space != json.dumps(
                    asdict(provider.space), sort_keys=True, separators=(",", ":")
                ):
                    raise DenseLifecycleError("Embedding space changed during build")
                concurrent = self._row(db, key)
                if concurrent is not None:
                    if concurrent[0] != signature or concurrent[2] != space:
                        raise DenseLifecycleError(
                            "Concurrent generation identity changed"
                        )
                    if bool(ids) != (concurrent[1] is not None):
                        raise DenseLifecycleError(
                            "Concurrent pointer contradicts membership"
                        )
                    db.commit()
                    return (
                        None
                        if concurrent[1] is None
                        else SQLiteDenseSnapshot(
                            self.directory / concurrent[1], provider.space
                        )
                    )
                if db.execute("SELECT count(*) FROM active").fetchone()[0] >= 16:
                    raise DenseLifecycleError("Managed actor capacity exhausted")
                db.execute(
                    "INSERT OR REPLACE INTO active VALUES(?,?,?,?)",
                    (key, signature, name, space),
                )
                db.commit()
                return snapshot
        except sqlite3.Error as error:
            raise DenseLifecycleError(
                "Managed catalog unavailable or writer busy"
            ) from error
