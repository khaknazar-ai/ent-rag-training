"""Pinned local English/multilingual models; CPU-only optional dependencies.

No downloads, remote code, hosted inference or device fallback occurs here.
Long passages use complete overlapping token windows rather than truncation.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha1, sha256
from pathlib import Path

from enterprise_rag.retrieval.embeddings import EmbeddingSpace
from enterprise_rag.retrieval.reranking import RerankCandidate, RerankedCandidate

_MODELS = {
    "multilingual_embedding": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f",
        "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
        128,
    ),
    "embedding": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        "cb202bfe2e3c98645018a6d12f182a434c9d3e02",
        256,
    ),
    "reranker": (
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a",
        "688882a79f44442ddc1f60d70334a7ff5df0fb47",
        512,
    ),
}
MAX_TEXT_CHARS = 16000
MAX_WINDOWS = 16
BATCH_SIZE = 8


class LocalModelBusy(RuntimeError):
    """The configured model already has one in-flight operation."""


class LocalModelStopped(RuntimeError):
    """Cooperative cancellation/deadline, observed between native batches."""


@contextmanager
def _exclusive_cpu(core):
    if not core.lock.acquire(blocking=False):
        raise LocalModelBusy("Local CPU model is busy")
    try:
        yield
    finally:
        core.lock.release()


def _check_stop(deadline: float, stop: threading.Event | None = None) -> None:
    if (stop is not None and stop.is_set()) or time.monotonic() >= deadline:
        raise LocalModelStopped("Local CPU operation stopped between native batches")


@dataclass(frozen=True, slots=True)
class TokenWindow:
    token_start: int
    token_end: int
    source_start: int
    source_end: int


def token_windows(
    offsets: Sequence[tuple[int, int]],
    source_chars: int,
    *,
    payload: int,
    overlap: int = 32,
) -> tuple[TokenWindow, ...]:
    """Return bounded windows covering every token, with original character spans."""
    if (
        type(source_chars) is not int
        or not 0 < source_chars <= MAX_TEXT_CHARS
        or type(payload) is not int
        or not 64 <= payload <= 510
        or type(overlap) is not int
        or not 0 <= overlap < payload
        or not offsets
        or len(offsets) > payload + (MAX_WINDOWS - 1) * (payload - overlap)
    ):
        raise ValueError("Text/token window budget exceeded or invalid")
    previous = -1
    for start, end in offsets:
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= source_chars
            or start < previous
        ):
            raise ValueError("Invalid original tokenizer offsets")
        previous = start
    windows = []
    start = 0
    while start < len(offsets):
        end = min(start + payload, len(offsets))
        windows.append(TokenWindow(start, end, offsets[start][0], offsets[end - 1][1]))
        if end == len(offsets):
            break
        start = end - overlap
    return tuple(windows)


def reject_external_tensors(model) -> None:
    """Inspect protobuf messages without resolving ONNX external-data paths."""

    def visit(message, depth=0):
        if depth > 64:
            raise ValueError("ONNX graph nesting exceeds the local policy")
        if message.DESCRIPTOR.name == "TensorProto":
            if message.external_data or message.data_location == 1:
                raise ValueError("External ONNX tensor data is not permitted")
        for descriptor, value in message.ListFields():
            if descriptor.type != descriptor.TYPE_MESSAGE:
                continue
            if descriptor.is_repeated:
                for nested in value:
                    visit(nested, depth + 1)
            else:
                visit(value, depth + 1)

    visit(model)


class _LocalCpuModel:
    def __init__(self, directory: Path, kind: str) -> None:
        repo, revision, digest, tokenizer_blob, max_tokens = _MODELS[kind]
        multilingual = kind == "multilingual_embedding"
        model_limit = 118453870 if multilingual else 100000000
        tokenizer_limit = 9081518 if multilingual else 2000000
        root = Path(directory).resolve()
        model = root / (
            "onnx/model_quint8_avx2.onnx" if multilingual else "onnx/model.onnx"
        )
        tokenizer_file = root / "tokenizer.json"
        if (
            model.is_symlink()
            or tokenizer_file.is_symlink()
            or not model.is_file()
            or not tokenizer_file.is_file()
            or not model.resolve().is_relative_to(root)
            or not tokenizer_file.resolve().is_relative_to(root)
            or model.stat().st_size > model_limit
            or tokenizer_file.stat().st_size > tokenizer_limit
        ):
            raise ValueError("Pinned local model files are missing or exceed policy")
        with model.open("rb") as stream:
            model_bytes = stream.read(model_limit + 1)
        with tokenizer_file.open("rb") as stream:
            raw = stream.read(tokenizer_limit + 1)
        if len(model_bytes) > model_limit or len(raw) > tokenizer_limit:
            raise ValueError("Local model grew beyond its bounded read policy")
        blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        tokenizer_digest = sha256(raw).hexdigest() if multilingual else blob
        if (
            sha256(model_bytes).hexdigest() != digest
            or tokenizer_digest != tokenizer_blob
        ):
            raise ValueError(
                "Local model/tokenizer differs from pinned official revision"
            )
        try:
            self.np = importlib.import_module("numpy")
            ort = importlib.import_module("onnxruntime")
            onnx = importlib.import_module("onnx")
            tokenizers = importlib.import_module("tokenizers")
        except ImportError as error:
            raise RuntimeError(
                "Install the explicitly configured local ONNX CPU extras"
            ) from error
        parsed = onnx.load_model_from_string(model_bytes)
        reject_external_tensors(parsed)
        del parsed
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Only CPUExecutionProvider is permitted")
        self.tokenizer = tokenizers.Tokenizer.from_str(raw.decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.cls = self.tokenizer.token_to_id("<s>" if multilingual else "[CLS]")
        self.sep = self.tokenizer.token_to_id("</s>" if multilingual else "[SEP]")
        self.pad = self.tokenizer.token_to_id("<pad>" if multilingual else "[PAD]")
        if any(type(x) is not int for x in (self.cls, self.sep, self.pad)):
            raise ValueError("Pinned tokenizer special tokens are missing")
        if multilingual and (self.cls, self.sep, self.pad) != (0, 2, 1):
            raise ValueError("Pinned multilingual special token IDs differ")
        if {x.name for x in self.session.get_inputs()} != {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }:
            raise ValueError("Unsupported ONNX model inputs")
        self.max_tokens = max_tokens
        self.model_revision = f"{repo}@{revision}:{digest}"
        self.tokenizer_sha256 = sha256(raw).hexdigest()
        self.runtime_sha256 = sha256(
            "|".join(
                str(getattr(module, "__version__", "unknown"))
                for module in (self.np, ort, onnx, tokenizers)
            ).encode()
        ).hexdigest()
        self.lock = threading.Lock()
        self.last_window_counts: tuple[int, ...] = ()

    def encode(self, text: str):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise ValueError("Local text model input must contain1..16000 characters")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def infer(self, sequences: Sequence[tuple[list[int], list[int]]]):
        if not sequences or len(sequences) > BATCH_SIZE:
            raise ValueError("Local CPU inference batch must contain1..8 windows")
        width = max(len(ids) for ids, _ in sequences)
        if width > self.max_tokens:
            raise ValueError("Local model token budget exceeded")
        ids = self.np.full((len(sequences), width), self.pad, dtype=self.np.int64)
        masks = self.np.zeros(ids.shape, dtype=self.np.int64)
        types = self.np.zeros(ids.shape, dtype=self.np.int64)
        for i, (tokens, segments) in enumerate(sequences):
            ids[i, : len(tokens)] = tokens
            masks[i, : len(tokens)] = 1
            types[i, : len(tokens)] = segments
        outputs = self.session.run(
            None, {"input_ids": ids, "attention_mask": masks, "token_type_ids": types}
        )
        return outputs, masks


class MiniLMOnnxEmbedding:
    _kind = "embedding"
    _preprocessing = "wordpiece254-overlap32-windowmean-l2-v1"

    def __init__(self, directory: Path) -> None:
        self.core = _LocalCpuModel(directory, self._kind)
        self.space = EmbeddingSpace(
            "onnxruntime-cpu",
            self.core.model_revision,
            384,
            self._preprocessing
            + ":tokenizer="
            + self.core.tokenizer_sha256
            + ":runtime="
            + self.core.runtime_sha256,
        )

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)) or not 1 <= len(texts) <= BATCH_SIZE:
            raise ValueError("Embedding batch must contain1..8 texts")
        np = self.core.np
        vectors = []
        counts = []
        deadline = time.monotonic() + 25
        with _exclusive_cpu(self.core):
            for text in texts:
                _check_stop(deadline)
                encoded = self.core.encode(text)
                windows = token_windows(
                    encoded.offsets, len(text), payload=self.core.max_tokens - 2
                )
                counts.append(len(windows))
                pooled = []
                for offset in range(0, len(windows), BATCH_SIZE):
                    _check_stop(deadline)
                    sequences = []
                    for window in windows[offset : offset + BATCH_SIZE]:
                        tokens = [
                            self.core.cls,
                            *encoded.ids[window.token_start : window.token_end],
                            self.core.sep,
                        ]
                        sequences.append((tokens, [0] * len(tokens)))
                    outputs, mask = self.core.infer(sequences)
                    _check_stop(deadline)
                    token_embeddings = outputs[0]
                    if (
                        token_embeddings.ndim != 3
                        or token_embeddings.shape[:2] != mask.shape
                        or token_embeddings.shape[2] != 384
                    ):
                        raise ValueError("Embedding ONNX output shape mismatch")
                    expanded = mask[..., None]
                    batch = (token_embeddings * expanded).sum(axis=1) / expanded.sum(
                        axis=1
                    ).clip(min=1)
                    pooled.extend(batch)
                vector = np.mean(np.asarray(pooled), axis=0)
                norm = float(np.linalg.norm(vector))
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("Invalid local embedding norm")
                normalized = tuple(float(x) for x in vector / norm)
                if not all(math.isfinite(x) for x in normalized):
                    raise ValueError("Nonfinite local embedding output")
                vectors.append(normalized)
            self.core.last_window_counts = tuple(counts)
        return tuple(vectors)


class MultilingualMiniLMOnnxEmbedding(MiniLMOnnxEmbedding):
    """Pinned multilingual Unigram tokenizer/BERT ONNX, official128-token cap.

    Explicit full-text window mean and L2 extension, not silent truncation or
    an assertion of equivalent quality to the original single-window pipeline.
    """

    _kind = "multilingual_embedding"
    _preprocessing = "unigram126-overlap32-cls0-sep2-pad1-maskmean-windowmean-l2-v1"


class MiniLMOnnxReranker:
    def __init__(self, directory: Path) -> None:
        self.core = _LocalCpuModel(directory, "reranker")
        self._admission = threading.Lock()

    async def rerank(
        self, query: str, candidates: tuple[RerankCandidate, ...], top_n: int
    ) -> tuple[RerankedCandidate, ...]:
        if not self._admission.acquire(blocking=False):
            raise LocalModelBusy("Local reranker is busy")
        stop = threading.Event()
        deadline = time.monotonic() + 25

        result: Future = Future()

        def worker():
            try:
                if result.set_running_or_notify_cancel():
                    try:
                        result.set_result(
                            self._score(query, candidates, top_n, stop, deadline)
                        )
                    except BaseException as error:
                        result.set_exception(error)
            finally:
                self._admission.release()

        # Start one owned thread synchronously under admission. Executor queue
        # cancellation during loop shutdown cannot strand the admission lock.
        try:
            threading.Thread(
                target=worker, name="rag-local-reranker", daemon=True
            ).start()
        except BaseException:
            self._admission.release()
            raise
        task = asyncio.wrap_future(result)
        # Cancellation stops subsequent batches; native calls already running
        # finish in the owned thread even if the originating event loop closes.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            stop.set()
            task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
            raise

    def _score(self, query, candidates, top_n, stop, deadline):
        if (
            not 1 <= len(candidates) <= 100
            or type(top_n) is not int
            or not 1 <= top_n <= len(candidates)
        ):
            raise ValueError("Invalid local reranker candidate budget")
        scored = []
        counts = []
        with _exclusive_cpu(self.core):
            _check_stop(deadline, stop)
            question = self.core.encode(query)
            if not 1 <= len(question.ids) <= 96:
                raise ValueError("Local reranker query exceeds96 wordpieces")
            payload = 512 - len(question.ids) - 3
            for candidate in candidates:
                _check_stop(deadline, stop)
                encoded = self.core.encode(candidate.text)
                windows = token_windows(
                    encoded.offsets, len(candidate.text), payload=payload
                )
                counts.append(len(windows))
                scores: list[float] = []
                for offset in range(0, len(windows), BATCH_SIZE):
                    _check_stop(deadline, stop)
                    sequences = []
                    for window in windows[offset : offset + BATCH_SIZE]:
                        prefix = [self.core.cls, *question.ids, self.core.sep]
                        suffix = [
                            *encoded.ids[window.token_start : window.token_end],
                            self.core.sep,
                        ]
                        sequences.append(
                            (prefix + suffix, [0] * len(prefix) + [1] * len(suffix))
                        )
                    outputs, _ = self.core.infer(sequences)
                    _check_stop(deadline, stop)
                    logits = outputs[0]
                    if logits.shape != (len(sequences), 1):
                        raise ValueError("Reranker ONNX output shape mismatch")
                    scores.extend(float(x) for x in logits[:, 0])
                if not all(math.isfinite(x) for x in scores):
                    raise ValueError("Nonfinite local reranker output")
                scored.append(RerankedCandidate(candidate.candidate_id, max(scores)))
            self.core.last_window_counts = tuple(counts)
        return tuple(sorted(scored, key=lambda row: -row.score)[:top_n])
