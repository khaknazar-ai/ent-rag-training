"""Prepare and verify a disposable local Docker-lite demo without starting Docker.

Run from any directory: python -B scripts/prepare_lite.py prepare
The generated .env contains a secret; this command never prints its value.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.bootstrap import build_services
from enterprise_rag.domain.documents import Tenant, User
from enterprise_rag.settings import Settings

DEMO_TENANT_SLUG = "local-demo"
DEMO_TENANT_NAME = "Local Demo"
DEMO_ADMIN_EMAIL = "demo-admin@example.invalid"
DEMO_VIEWER_EMAIL = "demo-viewer@example.invalid"
SAMPLE_PATH = Path("examples/lite-smoke.txt")
TOKEN_PLACEHOLDER = "# Restrict access before writing a bearer token.\n"
COMPOSE_OVERRIDES = (
    "RAG_DOCKER_DATA_DIR",
    "RAG_DOCKER_ARTIFACTS_DIR",
    "RAG_API_TOKEN",
    "RAG_DEMO_TENANT_ID",
    "RAG_DEMO_USER_ID",
    "RAG_DEMO_VIEWER_TOKEN",
    "RAG_DEMO_VIEWER_USER_ID",
)


def _paths(root: Path) -> tuple[Path, Path, Path, Path]:
    root = root.resolve()
    data_dir = root / ".local-data"
    artifacts_dir = root / ".local-artifacts"
    return root / ".env", data_dir, artifacts_dir, data_dir / "enterprise_rag.sqlite3"


def _acl_command(path: Path, *, check_only: bool) -> list[str]:
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(Path(__file__).with_name("protect_lite_env.ps1")),
        "-Path",
        str(path),
    ]
    if check_only:
        command.append("-CheckOnly")
    return command


def _restrict_acl(path: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            _acl_command(path, check_only=False), capture_output=True, text=True
        )
        if result.returncode != 0:
            raise OSError(
                "Could not restrict the local demo token file ACL: "
                + result.stderr.strip()
            )
    else:
        path.chmod(0o600)


def _acl_is_restricted(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        return (
            subprocess.run(
                _acl_command(path, check_only=True), capture_output=True, text=True
            ).returncode
            == 0
        )
    return path.stat().st_mode & 0o077 == 0


def _compose_matches(root: Path, values: dict[str, str]) -> bool:
    env_file, data_dir, artifacts_dir, _database_path = _paths(root)
    if not env_file.is_file():
        return False
    try:
        result = subprocess.run(
            [
                "docker-compose",
                "--env-file",
                str(env_file),
                "-f",
                str(root / "compose.yaml"),
                "--profile",
                "lite",
                "config",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    try:
        services = json.loads(result.stdout)["services"]
        api_environment = services["api"]["environment"]
        mounts = {
            name: {
                volume["target"]: Path(volume["source"]).resolve()
                for volume in services[name]["volumes"]
            }
            for name in ("api", "worker")
        }
    except (KeyError, TypeError, ValueError):
        return False
    return all(
        mounted.get("/data") == data_dir and mounted.get("/artifacts") == artifacts_dir
        for mounted in mounts.values()
    ) and all(
        api_environment.get(name) == values.get(name)
        for name in ("RAG_API_TOKEN", "RAG_DEMO_TENANT_ID", "RAG_DEMO_USER_ID")
    )


def _write_demo_env(env_file: Path, tenant: Tenant, admin: User, viewer: User) -> None:
    """Never put token bytes on disk until the file ACL is restricted."""

    if env_file.exists():
        if env_file.read_text(encoding="utf-8") != TOKEN_PLACEHOLDER:
            raise FileExistsError("Refusing to replace an existing demo configuration")
    else:
        descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(TOKEN_PLACEHOLDER)
    _restrict_acl(env_file)
    content = (
        "# Local, disposable Docker-lite demo; do not commit or share.\n"
        "RAG_PROFILE=lite\n"
        "RAG_DOCKER_DATA_DIR=./.local-data\n"
        "RAG_DOCKER_ARTIFACTS_DIR=./.local-artifacts\n"
        "RAG_API_PORT=8000\n"
        f"RAG_API_TOKEN={secrets.token_hex(32)}\n"
        f"RAG_DEMO_TENANT_ID={tenant.id}\n"
        f"RAG_DEMO_USER_ID={admin.id}\n"
        f"RAG_DEMO_VIEWER_TOKEN={secrets.token_hex(32)}\n"
        f"RAG_DEMO_VIEWER_USER_ID={viewer.id}\n"
    )
    with env_file.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(root: Path) -> dict[str, object]:
    """Create a fresh, explicitly local demo; never replace prior state."""

    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Preparation root must be an existing directory")
    if not (root / "compose.yaml").is_file() or not (root / SAMPLE_PATH).is_file():
        raise ValueError("Preparation root needs compose.yaml and the smoke fixture")
    overridden = [name for name in COMPOSE_OVERRIDES if os.environ.get(name)]
    if overridden:
        raise ValueError(
            "Unset Docker-lite environment overrides before preparation: "
            + ", ".join(overridden)
        )
    env_file, data_dir, artifacts_dir, database_path = _paths(root)
    existing = [
        str(path)
        for path in (env_file, data_dir, artifacts_dir)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to replace existing local state: " + ", ".join(existing)
        )

    tenant = Tenant.create(DEMO_TENANT_SLUG, DEMO_TENANT_NAME)
    admin = User.create(tenant.id, DEMO_ADMIN_EMAIL)
    viewer = User.create(tenant.id, DEMO_VIEWER_EMAIL)
    settings = Settings(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        database_url=f"sqlite:///{database_path.as_posix()}",
    )
    services = build_services(settings)
    services.metadata.ensure_tenant(tenant)
    services.metadata.upsert_user(admin)
    services.metadata.upsert_user(viewer)
    services.metadata.grant_tenant_admin(tenant.id, admin.id)

    _write_demo_env(env_file, tenant, admin, viewer)

    result = check(root)
    if not result["local_preparation_complete"]:
        raise RuntimeError("Local preparation verification failed; inspect local state")
    return result


def resume(root: Path) -> dict[str, object]:
    """Finish only an untouched partial demo bootstrap without deleting data."""

    root = root.resolve()
    env_file, data_dir, artifacts_dir, database_path = _paths(root)
    if (
        not data_dir.is_dir()
        or not artifacts_dir.is_dir()
        or not database_path.is_file()
    ):
        raise ValueError("No complete partial demo database to resume")
    tenant = Tenant.create(DEMO_TENANT_SLUG, DEMO_TENANT_NAME)
    admin = User.create(tenant.id, DEMO_ADMIN_EMAIL)
    viewer = User.create(tenant.id, DEMO_VIEWER_EMAIL)
    with sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True) as connection:
        known_admin = connection.execute(
            "SELECT 1 FROM tenant_admins WHERE tenant_id = ? AND user_id = ?",
            (tenant.id, admin.id),
        ).fetchone()
        known_viewer = connection.execute(
            "SELECT 1 FROM users WHERE tenant_id = ? AND id = ?",
            (tenant.id, viewer.id),
        ).fetchone()
        foreign_tenants = connection.execute(
            "SELECT COUNT(*) FROM tenants WHERE id != ?", (tenant.id,)
        ).fetchone()[0]
        foreign_users = connection.execute(
            "SELECT COUNT(*) FROM users WHERE id NOT IN (?, ?)",
            (admin.id, viewer.id),
        ).fetchone()[0]
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if known_admin is None or foreign_tenants or foreign_users or documents:
        raise ValueError("Partial database is not an untouched local demo")
    if known_viewer is None:
        services = build_services(
            Settings(
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                database_url=f"sqlite:///{database_path.as_posix()}",
            )
        )
        services.metadata.upsert_user(viewer)
    _write_demo_env(env_file, tenant, admin, viewer)
    result = check(root)
    if not result["local_preparation_complete"]:
        raise RuntimeError("Resumed preparation but local verification failed")
    return result


def check(root: Path) -> dict[str, object]:
    """Inspect local prep without opening Docker or printing the API token."""

    root = root.resolve()
    env_file, data_dir, artifacts_dir, database_path = _paths(root)
    values: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value
    token = values.get("RAG_API_TOKEN", "")
    viewer_token = values.get("RAG_DEMO_VIEWER_TOKEN", "")
    config_valid = (
        values.get("RAG_PROFILE") == "lite"
        and values.get("RAG_DOCKER_DATA_DIR") == "./.local-data"
        and values.get("RAG_DOCKER_ARTIFACTS_DIR") == "./.local-artifacts"
        and values.get("RAG_API_PORT") == "8000"
        and len(token) == 64
        and all(character in "0123456789abcdef" for character in token)
        and len(viewer_token) == 64
        and all(character in "0123456789abcdef" for character in viewer_token)
        and token != viewer_token
        and bool(values.get("RAG_DEMO_VIEWER_USER_ID"))
    )
    admin_valid = False
    viewer_valid = False
    if database_path.is_file() and config_valid:
        try:
            with sqlite3.connect(
                f"{database_path.as_uri()}?mode=ro", uri=True
            ) as connection:
                admin_valid = (
                    connection.execute(
                        """
                        SELECT 1
                        FROM users AS u
                        JOIN tenant_admins AS a
                            ON a.user_id = u.id AND a.tenant_id = u.tenant_id
                        WHERE u.tenant_id = ? AND u.id = ?
                        """,
                        (
                            values.get("RAG_DEMO_TENANT_ID"),
                            values.get("RAG_DEMO_USER_ID"),
                        ),
                    ).fetchone()
                    is not None
                )
                viewer_valid = (
                    connection.execute(
                        """
                        SELECT 1 FROM users AS u
                        WHERE u.tenant_id = ? AND u.id = ? AND u.id != ?
                          AND NOT EXISTS (
                              SELECT 1 FROM tenant_admins AS a
                              WHERE a.tenant_id = u.tenant_id AND a.user_id = u.id
                          )
                        """,
                        (
                            values.get("RAG_DEMO_TENANT_ID"),
                            values.get("RAG_DEMO_VIEWER_USER_ID"),
                            values.get("RAG_DEMO_USER_ID"),
                        ),
                    ).fetchone()
                    is not None
                )
        except sqlite3.Error:
            admin_valid = False
            viewer_valid = False
    checks = {
        "compose_file": (root / "compose.yaml").is_file(),
        "smoke_document": (root / SAMPLE_PATH).is_file(),
        "env_configured": bool(config_valid),
        "env_acl_restricted": _acl_is_restricted(env_file),
        "compose_uses_prepared_state": _compose_matches(root, values),
        "data_dir": data_dir.is_dir(),
        "artifacts_dir": artifacts_dir.is_dir(),
        "admin_provisioned": admin_valid,
        "viewer_provisioned": viewer_valid,
    }
    return {
        "local_preparation_complete": all(checks.values()),
        "checks": checks,
        "docker_runtime_tested": False,
    }


def rotate_token(root: Path) -> dict[str, object]:
    """Restrict ACL first, then replace both local demo bearer tokens."""

    root = root.resolve()
    env_file, _data_dir, _artifacts_dir, _database_path = _paths(root)
    if not env_file.is_file():
        raise FileNotFoundError("No local demo .env exists to repair")
    _restrict_acl(env_file)
    lines = env_file.read_text(encoding="utf-8").splitlines()
    token_names = ("RAG_API_TOKEN", "RAG_DEMO_VIEWER_TOKEN")
    matching = {
        name: [line for line in lines if line.startswith(f"{name}=")]
        for name in token_names
    }
    if any(len(matching[name]) != 1 for name in token_names):
        raise ValueError("A local demo token entry is missing or ambiguous")
    replacements = {name: f"{name}={secrets.token_hex(32)}" for name in token_names}
    content = "\n".join(
        replacements.get(line.partition("=")[0], line) for line in lines
    )
    with env_file.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _restrict_acl(env_file)
    result = check(root)
    if not result["local_preparation_complete"]:
        raise RuntimeError("Token rotation completed but local verification failed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "check", "resume", "rotate-token")
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.root)
        elif args.action == "resume":
            result = resume(args.root)
        elif args.action == "rotate-token":
            result = rotate_token(args.root)
        else:
            result = check(args.root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["local_preparation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
