"""FastAPI transport for the explicitly configured local demo."""

from .app import StaticTokenAuthenticator, create_app

__all__ = ["StaticTokenAuthenticator", "create_app"]
