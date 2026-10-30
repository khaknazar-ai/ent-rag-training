"""Read-only current-server identity check for explicitly configured raw semantics.

This is not a generation authorization grant. The caller still owns whether to
invoke the adapter. Construction pins local manifests; verify fetches metadata
only, never starts a server, loads a model, retries, or accepts redirects.
"""

from __future__ import annotations

import json
import urllib.request
from hashlib import sha256
from pathlib import Path

from enterprise_rag.models.ollama import _NoRedirect, _unique_json_object
from enterprise_rag.models.semantic_prompt import (
    SemanticBudgetRefused,
    canonical_json,
    validate_binding,
)
from enterprise_rag.models.tokenizer_assets import PinnedAssets
from enterprise_rag.settings import Settings


class OllamaSemanticRuntimeVerifier:
    """Trusted dependency with optional metadata-fetch double for offline tests.

    The first successful check binds full metadata for this verifier's lifetime;
    later calls must return identical metadata. A historical parity receipt or
    `network_identity_verified` boolean is never accepted as current identity.
    """

    def __init__(
        self, counter, base_url, *, metadata_fetch=None, expected_version="0.40.6"
    ):
        self.counter = counter
        self.binding = counter.binding
        validate_binding(self.binding)
        Settings(
            llm_provider="ollama",
            llm_base_url=base_url,
            llm_model=self.binding.model_tag,
        )
        self.url = base_url.rstrip("/") + "/api/generate"
        self.base_url = base_url.rstrip("/")
        if not isinstance(expected_version, str) or not expected_version:
            raise SemanticBudgetRefused("semantic_runtime_version_unpinned")
        self.expected_version = expected_version
        self._fetch = metadata_fetch or self._metadata
        self._identity_sha256 = None
        self._failed = False
        counter.validate()
        native = counter.native
        wrapper = json.loads(
            native.assets.read(native.config.manifest, 65536),
            object_pairs_hook=_unique_json_object,
        )
        references = [
            layer
            for layer in wrapper["layers"]
            if layer["mediaType"]
            == "application/vnd.docker.distribution.manifest.v2+json"
        ]
        if len(references) != 1:
            raise SemanticBudgetRefused("semantic_original_manifest_unavailable")
        reference = references[0]
        digest = reference["digest"]
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(c not in "0123456789abcdef" for c in digest[7:])
        ):
            raise SemanticBudgetRefused("semantic_original_manifest_invalid")
        self.api_manifest_sha256 = digest[7:]
        original_path = Path(native.config.model).parent / ("sha256-" + digest[7:])
        self.assets = PinnedAssets({original_path: digest[7:]})
        try:
            raw = self.assets.read(original_path, 65536)
            expected = dict(wrapper)
            expected["layers"] = [
                layer for layer in wrapper["layers"] if layer is not reference
            ]
            model_layers = [
                layer
                for layer in expected["layers"]
                if layer["mediaType"] == "application/vnd.ollama.image.model"
            ]
            if (
                len(raw) != reference["size"]
                or json.loads(raw, object_pairs_hook=_unique_json_object) != expected
                or len(model_layers) != 1
                or model_layers[0]["digest"] != "sha256:" + self.binding.model_digest
            ):
                raise SemanticBudgetRefused("semantic_manifest_chain_mismatch")
        except BaseException:
            self.close()
            raise

    def _metadata(self, path, payload):
        if path not in {"/api/version", "/api/tags", "/api/show"}:
            raise SemanticBudgetRefused("semantic_metadata_target_invalid")
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else canonical_json(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        with opener.open(request, timeout=10) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise SemanticBudgetRefused("semantic_metadata_size_limit")
        return json.loads(raw, object_pairs_hook=_unique_json_object)

    def verify(self, url, binding, request_bytes):
        try:
            if self._failed or url != self.url or binding != self.binding:
                raise ValueError("Runtime binding changed")
            self.counter.validate()
            self.assets.validate()
            request = json.loads(request_bytes)
            if (
                request["model"] != binding.model_tag
                or request["raw"] is not True
                or request["options"]["num_ctx"] != binding.context_tokens
                or type(request["options"]["num_gpu"]) is not int
                or request["options"]["num_gpu"] != 0
            ):
                raise ValueError("Runtime request changed")
            version = self._fetch("/api/version", None)
            if version.get("version") != self.expected_version:
                raise ValueError("Current server version differs")
            tags = self._fetch("/api/tags", None)
            candidates = [
                item
                for item in tags["models"]
                if item.get("name") == binding.model_tag
                or item.get("model") == binding.model_tag
            ]
            if (
                len(candidates) != 1
                or candidates[0]["digest"].removeprefix("sha256:")
                != self.api_manifest_sha256
            ):
                raise ValueError("Current server manifest differs")
            show = self._fetch(
                "/api/show", {"model": binding.model_tag, "verbose": False}
            )
            info = show["model_info"]
            maximum = info.get("qwen35.context_length")
            if (
                info.get("general.architecture") != "qwen35"
                or type(maximum) is not int
                or maximum < binding.context_tokens
            ):
                raise ValueError("Server context capacity is unconfirmed")
            identity = sha256(
                canonical_json(
                    {
                        "endpoint": self.base_url,
                        "version": version,
                        "tag": candidates[0],
                        "show": show,
                        "model_digest": binding.model_digest,
                        "wrapper_manifest_sha256": binding.manifest_sha256,
                        "api_manifest_sha256": self.api_manifest_sha256,
                        "context_tokens": binding.context_tokens,
                    }
                ).encode()
            ).hexdigest()
            if self._identity_sha256 is not None and identity != self._identity_sha256:
                raise ValueError("Current server identity changed")
            self.counter.validate()
            self.assets.validate()
            self._identity_sha256 = identity
            return identity
        except Exception:
            self._failed = True
            raise SemanticBudgetRefused(
                "semantic_runtime_identity_unverified"
            ) from None

    def close(self):
        self._failed = True
        self.assets.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
