"""Native publication fence for the opt-in external dense backend.

Remote I/O never runs inside a native transaction. A durable reservation limits
partial generations too; a verified manifest is bound to the current worker lease.
"""

import json
from hashlib import sha256


class DensePublicationError(RuntimeError):
    pass


def digest(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def content_digest(db, run_id):
    rows = db.execute(
        "SELECT id,content_sha256,text FROM text_chunks WHERE run_id=? ORDER BY id",
        (run_id,),
    ).fetchmany(1025)
    if not 1 <= len(rows) <= 1024 or any(
        len(row[2].encode()) > 65536 or sha256(row[2].encode()).hexdigest() != row[1]
        for row in rows
    ):
        raise DensePublicationError("Dense canonical membership invalid or over budget")
    return digest([(row[0], row[1]) for row in rows]), len(rows)


def activation_verified(db, lease):
    binding = db.execute("SELECT configuration FROM dense_backend_binding").fetchone()
    if binding is None:
        return True
    row = db.execute(
        """SELECT p.*, r.execution_configuration_sha256 AS current_config
        FROM dense_publications p JOIN ingestion_runs r ON r.id=p.run_id
        WHERE p.run_id=?""",
        (lease.job.run_id,),
    ).fetchone()
    if row is None:
        return False
    signature, count = content_digest(db, lease.job.run_id)
    return (
        row["backend_key"] == digest(json.loads(binding[0]))
        and row["content_digest"] == signature
        and row["point_count"] == count
        and row["configuration_sha256"] == row["current_config"]
        and row["lease_owner"] == lease.worker_id
        and row["lease_token"] == lease.lease_token
        and isinstance(row["remote_digest"], str)
        and len(row["remote_digest"]) == 64
    )


class SQLiteDensePublication:
    def __init__(self, metadata, configuration):
        self.metadata = metadata
        self.configuration = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        )
        self.key = digest(configuration)
        with metadata._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT configuration FROM dense_backend_binding"
            ).fetchone()
            if row is not None and row[0] != self.configuration:
                raise DensePublicationError(
                    "Dense backend/model space binding mismatch"
                )
            if row is None:
                # Never silently adopt a legacy lexical database or invent its vectors.
                if db.execute("SELECT 1 FROM ingestion_runs LIMIT 1").fetchone():
                    raise DensePublicationError(
                        "Legacy runs require an explicit migration"
                    )
                db.execute(
                    "INSERT INTO dense_backend_binding VALUES (1,?)",
                    (self.configuration,),
                )

    def _fence(self, db, lease, now):
        row = db.execute(
            """SELECT r.execution_configuration_sha256
            FROM ingestion_jobs j JOIN ingestion_runs r ON r.id=j.run_id
            JOIN document_versions v ON v.id=r.document_version_id
            JOIN documents d ON d.id=v.document_id
            WHERE j.id=? AND j.run_id=? AND j.tenant_id=? AND j.status='running'
            AND j.lease_owner=? AND j.lease_token=? AND j.lease_expires_at>?
            AND r.status='running' AND d.status NOT IN ('deleting','deleted')""",
            (
                lease.job.id,
                lease.job.run_id,
                lease.job.tenant_id,
                lease.worker_id,
                lease.lease_token,
                now,
            ),
        ).fetchone()
        binding = db.execute(
            "SELECT configuration FROM dense_backend_binding"
        ).fetchone()
        if (
            row is None
            or not row[0]
            or binding is None
            or binding[0] != self.configuration
        ):
            raise DensePublicationError("Dense publication lease/configuration lost")
        return row[0]

    def reserve(self, lease, now):
        with self.metadata._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            config = self._fence(db, lease, now)
            signature, count = content_digest(db, lease.job.run_id)
            epoch = db.execute(
                "SELECT point_count FROM dense_write_epochs WHERE run_id=? AND lease_owner=? AND lease_token=?",
                (lease.job.run_id, lease.worker_id, lease.lease_token),
            ).fetchone()
            if epoch is None:
                generations, points = db.execute(
                    "SELECT count(*), coalesce(sum(point_count),0) FROM dense_write_epochs"
                ).fetchone()
                if generations >= 32 or points + count > 8192:
                    raise DensePublicationError(
                        "Dense generation/point capacity exceeded"
                    )
                db.execute(
                    "INSERT INTO dense_write_epochs VALUES (?,?,?,?)",
                    (lease.job.run_id, lease.worker_id, lease.lease_token, count),
                )
            elif epoch[0] != count:
                raise DensePublicationError("Dense epoch membership changed")
            row = db.execute(
                "SELECT * FROM dense_publications WHERE run_id=?", (lease.job.run_id,)
            ).fetchone()
            if row is not None:
                if (
                    row["backend_key"],
                    row["content_digest"],
                    row["configuration_sha256"],
                ) != (self.key, signature, config):
                    raise DensePublicationError("Dense generation identity changed")
            else:
                db.execute(
                    "INSERT INTO dense_publications VALUES (?,?,?,?,?,?,?,NULL)",
                    (
                        lease.job.run_id,
                        self.key,
                        signature,
                        count,
                        config,
                        lease.worker_id,
                        lease.lease_token,
                    ),
                )
            return signature

    def verify_lease(self, lease, now):
        with self.metadata._connection() as db:
            self._fence(db, lease, now)

    def seal(self, lease, signature, remote_digest, now):
        with self.metadata._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            config = self._fence(db, lease, now)
            current, _ = content_digest(db, lease.job.run_id)
            if current != signature:
                raise DensePublicationError("Dense canonical content changed")
            if (
                db.execute(
                    "SELECT 1 FROM dense_write_epochs WHERE run_id=? AND lease_owner=? AND lease_token=?",
                    (lease.job.run_id, lease.worker_id, lease.lease_token),
                ).fetchone()
                is None
            ):
                raise DensePublicationError("Dense write epoch is not reserved")
            result = db.execute(
                """UPDATE dense_publications SET remote_digest=?,lease_owner=?,lease_token=?
                WHERE run_id=? AND backend_key=? AND content_digest=? AND configuration_sha256=?""",
                (
                    remote_digest,
                    lease.worker_id,
                    lease.lease_token,
                    lease.job.run_id,
                    self.key,
                    signature,
                    config,
                ),
            )
            if result.rowcount != 1:
                raise DensePublicationError("Dense reservation missing")

    def manifest(self, run_id):
        with self.metadata._connection() as db:
            row = db.execute(
                "SELECT * FROM dense_publications WHERE run_id=? AND backend_key=?",
                (run_id, self.key),
            ).fetchone()
            if row is None or row["remote_digest"] is None:
                raise DensePublicationError("Dense generation is not verified")
            return dict(row)
