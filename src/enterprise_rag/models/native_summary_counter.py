"""Explicit offline-only prompt counter for an already installed pinned model.

Construction is opt-in, expensive only once, and never performed by default
bootstrap. A qualified vocabulary does not qualify a running Ollama server.
"""

# Deliberate multilingual/normalization qualification inputs, not identifiers.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from enterprise_rag.domain import summary_spans
from enterprise_rag.domain.model_summary import SummaryTokenizerBinding
from enterprise_rag.models import native_vocab, ollama_summary, tokenizer_assets
from enterprise_rag.models.native_vocab import ABI_REVISION, NativeVocabularyError
from enterprise_rag.models.ollama_summary import (
    SCHEMA_SHA256,
    TEMPLATE_SHA256,
    TRANSPORT_REVISION,
)
from enterprise_rag.models.tokenizer_assets import AssetIdentityError, PinnedAssets
from enterprise_rag.workers import processes

SPECIAL_POLICY = "llama-tokenize-add-special-true-parse-special-true-v1"
QUALIFICATION_TEXTS = (
    "",
    "Қазақ тілі: Ә Ғ Қ Ң Ө Ұ Ү Һ І.",
    "Русский: ёж, съёмка.\nСтрока.",
    "中文：文档与准确引用。",
    "👩🏽‍💻📘🙂",
    "é",
    "e\u0301",
    "ＡＢＣ",
    "ABC",
    "ﬁ",
    "fi",
    "  a\t b\r\n c\n",
    "\x00a\x01\x1f",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "a<|im_start|>b",
    "<|im_sta",
    "rt|>",
)


@dataclass(frozen=True)
class NativeTokenizerConfig:
    model_tag: str
    manifest: Path
    manifest_sha256: str
    model: Path
    model_sha256: str
    dll: Path
    # Exact non-system dependency closure discovered from PE import tables.
    # This selected ABI supports only these six local files; no backend DLLs.
    dll_sha256: tuple[tuple[str, str], ...]


class NativeSummaryCounter:
    network_identity_verified = False

    def __init__(self, config: NativeTokenizerConfig):
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", config.model_tag) is None
            or config.dll.name != "libllama.dll"
            or set(dict(config.dll_sha256))
            != {
                "libllama.dll",
                "ggml.dll",
                "ggml-base.dll",
                "libc++.dll",
                "libunwind.dll",
                "libomp.dll",
            }
            or len(config.dll_sha256) != 6
            or config.manifest.name != config.model_tag.split(":", 1)[1]
            or config.manifest.parent.name != config.model_tag.split(":", 1)[0]
        ):
            raise AssetIdentityError("tokenizer_configuration_invalid")
        for digest in (
            config.manifest_sha256,
            config.model_sha256,
            *(digest for _, digest in config.dll_sha256),
        ):
            if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                raise AssetIdentityError("tokenizer_configuration_invalid")
        self.config = config
        files: dict[Path, str | None] = {
            config.manifest: config.manifest_sha256,
            config.model: config.model_sha256,
            **{config.dll.parent / name: digest for name, digest in config.dll_sha256},
        }
        for module in (
            native_vocab,
            tokenizer_assets,
            processes,
            sys.modules[__name__],
            summary_spans,
            ollama_summary,
        ):
            if module.__file__ is None:
                raise AssetIdentityError("tokenizer_helper_identity_unavailable")
            files[Path(module.__file__).absolute()] = None
            cached = getattr(module, "__cached__", None)
            if not isinstance(cached, str) or not Path(cached).is_file():
                # Spawn may import bytecode before the native start gate. Never
                # allow an untracked cache to appear after binding qualification.
                raise AssetIdentityError("tokenizer_helper_cache_unavailable")
            files[Path(cached).absolute()] = None
        self.assets = PinnedAssets(files)
        try:
            manifest = json.loads(self.assets.read(config.manifest, 64 * 1024))
            model_layers = [
                layer
                for layer in manifest["layers"]
                if layer["mediaType"] == "application/vnd.ollama.image.model"
            ]
            if (
                len(model_layers) != 1
                or model_layers[0]["digest"] != "sha256:" + config.model_sha256
                or model_layers[0]["size"] != config.model.stat().st_size
            ):
                raise AssetIdentityError("tokenizer_manifest_model_mismatch")
            self.metadata = self.assets.vocabulary_metadata(config.model)
            if (
                self.metadata.get("general.architecture") != "qwen35"
                or self.metadata.get("tokenizer.ggml.model") != "gpt2"
                or self.metadata.get("tokenizer.ggml.pre") != "qwen35"
            ):
                raise AssetIdentityError("tokenizer_model_family_unqualified")
            cases = [(text, True, True) for text in QUALIFICATION_TEXTS]
            cases += [
                ("<|im_start|>", False, False),
                ("", False, False),
                ("e\u0301", False, False),
                ("a" * 64000, True, True),
                (QUALIFICATION_TEXTS[1], True, True),
            ]
            result = native_vocab.run_vocabulary(
                str(config.dll), str(config.model), cases
            )
            vectors = self._validate_result(result, len(cases))
            if (
                result["add_bos"] is not False
                or result["add_eos"] is not False
                or vectors[0] != []
                or vectors[20] != []
                or vectors[1] != vectors[-1]
                or vectors[5] == vectors[6]
                or vectors[7] == vectors[8]
                or vectors[13] == vectors[19]
                or not all(vectors[i] for i in range(1, 19))
            ):
                raise NativeVocabularyError("native_unicode_policy_unqualified")
            for index, special in (
                (13, "<|im_start|>"),
                (14, "<|im_end|>"),
                (15, "<|endoftext|>"),
            ):
                if vectors[index] != [self.metadata["special_ids"].get(special)]:
                    raise NativeVocabularyError("native_special_policy_unqualified")
            qualification = {
                "abi": ABI_REVISION,
                "policy": SPECIAL_POLICY,
                "assets": self.assets.digests,
                "metadata": self.metadata,
                "cases": cases,
                "native": result,
                "scope": "local vocabulary; running server identity unverified",
                "prompt_contract": {
                    "template_sha256": TEMPLATE_SHA256,
                    "schema_sha256": SCHEMA_SHA256,
                    "transport_revision": TRANSPORT_REVISION,
                },
            }
            self.qualification = qualification
            self.binding = SummaryTokenizerBinding(
                config.model_sha256,
                self.metadata["metadata_sha256"],
                TEMPLATE_SHA256,
                SCHEMA_SHA256,
                ABI_REVISION,
                SPECIAL_POLICY,
                sha256(
                    json.dumps(
                        qualification,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                TRANSPORT_REVISION,
                config.model_tag,
                config.manifest_sha256,
            )
            self.assets.validate()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate_result(result: dict, count: int) -> list[list[int]]:
        if (
            not isinstance(result, dict)
            or set(result) != {"abi", "add_bos", "add_eos", "vectors"}
            or result["abi"] != ABI_REVISION
            or type(result["add_bos"]) is not bool
            or type(result["add_eos"]) is not bool
            or not isinstance(result["vectors"], list)
            or len(result["vectors"]) != count
        ):
            raise NativeVocabularyError("native_child_protocol")
        for vector in result["vectors"]:
            if (
                not isinstance(vector, list)
                or len(vector) > 64002
                or any(
                    type(token) is not int or not 0 <= token < 248320
                    for token in vector
                )
            ):
                raise NativeVocabularyError("native_child_protocol")
        return result["vectors"]

    def count_prompt(self, prompt: str) -> int:
        try:
            self.assets.validate()
            result = native_vocab.run_vocabulary(
                str(self.config.dll), str(self.config.model), [(prompt, True, True)]
            )
            vectors = self._validate_result(result, 1)
            if result["add_bos"] is not False or result["add_eos"] is not False:
                raise NativeVocabularyError("native_special_policy_changed")
            self.assets.validate()
            return len(vectors[0])
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.assets.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
