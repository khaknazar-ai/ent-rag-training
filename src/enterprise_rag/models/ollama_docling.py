"""Explicit small Docling conversion model; no generic QA prompt or retry."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from enterprise_rag.application.answer import (
    ModelProtocolError,
    ModelUnavailable,
)
from enterprise_rag.models.docling_decode import (
    require_docling_core,
)
from enterprise_rag.models.docling_grounding import DoclingGroundingModel
from enterprise_rag.models.docling_runtime import DoclingDecoderRuntime
from enterprise_rag.models.ollama import _unique_json_object
from enterprise_rag.models.ollama_visual import (
    OllamaVisualGroundingModel,
)

DOCLING_MODEL = "ibm/granite-docling:258m"
DOCLING_CONTRACT = "docling-doctags-task-label-v2"


class OllamaDoclingGroundingModel(DoclingGroundingModel):
    """Use the official conversion instruction, then select an exact OCR label.

    This is a separate opt-in contract: CPU2, ctx4096, output1024, 512px,
    timeout180, keep_alive0. Construction does not download or run a model.
    """

    timeout_seconds = 180.0

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        expected_digest: str,
        decoder: DoclingDecoderRuntime | None = None,
    ):
        if model != DOCLING_MODEL:
            raise ValueError("Docling mode requires the explicit Granite-Docling model")
        if not isinstance(expected_digest, str):
            raise ValueError("Docling mode requires a pinned model digest")
        if decoder is None:
            require_docling_core()
        self.decoder = decoder
        # Reuse the existing local-only endpoint, digest and identity HTTP checks.
        self.identity = OllamaVisualGroundingModel(
            base_url,
            model,
            expected_digest=expected_digest,
            timeout_seconds=180,
            keep_alive="0",
            image_max_edge=512,
        )
        self.options = {
            "temperature": 0,
            "num_gpu": 0,
            "num_thread": 2,
            "num_ctx": 4096,
            "num_predict": 1024,
        }

    def _convert(self, image: bytes, instruction: str, deadline: float) -> str:
        self.identity._verify_model_pin(deadline)
        payload = {
            "model": DOCLING_MODEL,
            "stream": False,
            "keep_alive": "0",
            "options": dict(self.options),
            "messages": [
                {
                    "role": "user",
                    "content": instruction,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
        }
        request = urllib.request.Request(
            self.identity.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.identity._opener.open(
                request, timeout=self._remaining(deadline)
            ) as response:
                raw = response.read(65_537)
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            raise ModelUnavailable("Local Docling conversion failed") from error
        self.identity._verify_model_pin(deadline)
        try:
            envelope = json.loads(raw, object_pairs_hook=_unique_json_object)
            if (
                len(raw) > 65_536
                or not isinstance(envelope, dict)
                or envelope.get("done") is not True
                or envelope.get("done_reason") != "stop"
            ):
                raise ValueError("Incomplete conversion")
            content = envelope["message"]["content"]
            if not isinstance(content, str) or len(content.encode("utf-8")) > 16_384:
                raise ValueError("Invalid DocTags content size")
            return content
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
            raise ModelProtocolError("Invalid Docling response") from error
