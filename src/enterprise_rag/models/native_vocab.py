"""Child-only Win64 llama.cpp b11669 vocabulary ABI; never creates a context.

ABI: ggml-org/llama.cpp/b11669/include/llama.h, model defaults and early
vocab-only return: src/llama-model.cpp and src/llama.cpp at the same tag.
Only the six vocabulary/model-lifetime functions below are bound.
"""

from __future__ import annotations

import ctypes as c
import json
import multiprocessing as mp
import os
import struct
from math import isfinite
from multiprocessing.connection import Connection
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any

MAX_INPUT_BYTES = 64000
MAX_BATCH_BYTES = 256000
MAX_RESULT_BYTES = 2 * 1024 * 1024
MEMORY_BYTES = 512 * 1024 * 1024
ABI_REVISION = "llama.cpp-b11669-win64-v1"


class NativeVocabularyError(ValueError):
    """A safe diagnostic code, without native logs or document content."""


class ModelParams(c.Structure):
    _fields_ = [
        ("devices", c.c_void_p),
        ("tensor_buft_overrides", c.c_void_p),
        ("n_gpu_layers", c.c_int32),
        ("split_mode", c.c_int32),
        ("load_mode", c.c_int32),
        ("lazy_mode", c.c_int32),
        ("main_gpu", c.c_int32),
        ("tensor_split", c.c_void_p),
        ("progress_callback", c.c_void_p),
        ("progress_callback_user_data", c.c_void_p),
        ("kv_overrides", c.c_void_p),
        ("vocab_only", c.c_bool),
        ("check_tensors", c.c_bool),
        ("use_extra_bufts", c.c_bool),
        ("no_host", c.c_bool),
        ("no_alloc", c.c_bool),
        ("load_mtp", c.c_bool),
    ]


def validate_abi() -> None:
    expected = [0, 8, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 73, 74, 75, 76, 77]
    if (
        os.name != "nt"
        or struct.calcsize("P") != 8
        or c.sizeof(ModelParams) != 80
        or c.alignment(ModelParams) != 8
        or [getattr(ModelParams, field[0]).offset for field in ModelParams._fields_]
        != expected
    ):
        raise NativeVocabularyError("native_abi_platform_mismatch")


def validate_inputs(cases: list[tuple[str, bool, bool]]) -> None:
    if not 1 <= len(cases) <= 32:
        raise NativeVocabularyError("native_batch_limit")
    total = 0
    for text, add_special, parse_special in cases:
        if (
            type(text) is not str
            or type(add_special) is not bool
            or type(parse_special) is not bool
        ):
            raise NativeVocabularyError("native_input_type")
        size = len(text.encode("utf-8", errors="strict"))
        if size > MAX_INPUT_BYTES:
            raise NativeVocabularyError("native_input_limit")
        total += size
    if total > MAX_BATCH_BYTES:
        raise NativeVocabularyError("native_batch_limit")


def _native(dll: str, model_path: str, cases: list[tuple[str, bool, bool]]) -> dict:
    validate_abi()
    validate_inputs(cases)
    # Absolute DLL + its own directory and System32 only. No PATH/cwd or backend
    # discovery. Dependencies are pinned by the parent for the whole operation.
    api = c.CDLL(dll, winmode=0x100 | 0x800)
    signatures: tuple[tuple[str, list[Any], Any], ...] = (
        ("llama_model_default_params", [], ModelParams),
        ("llama_model_load_from_file", [c.c_char_p, ModelParams], c.c_void_p),
        ("llama_model_get_vocab", [c.c_void_p], c.c_void_p),
        ("llama_model_free", [c.c_void_p], None),
        ("llama_vocab_get_add_bos", [c.c_void_p], c.c_bool),
        ("llama_vocab_get_add_eos", [c.c_void_p], c.c_bool),
        (
            "llama_tokenize",
            [
                c.c_void_p,
                c.c_char_p,
                c.c_int32,
                c.POINTER(c.c_int32),
                c.c_int32,
                c.c_bool,
                c.c_bool,
            ],
            c.c_int32,
        ),
    )
    for name, args, result in signatures:
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    params = api.llama_model_default_params()
    expected = [
        None,
        None,
        -1,
        1,
        -1,
        1,
        0,
        None,
        None,
        None,
        None,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    if [getattr(params, field[0]) for field in ModelParams._fields_] != expected:
        raise NativeVocabularyError("native_abi_defaults_mismatch")
    devices = (c.c_void_p * 1)(None)
    params.devices = c.cast(devices, c.c_void_p)
    params.n_gpu_layers = 0
    params.split_mode = 0
    params.load_mode = 0
    params.lazy_mode = 0
    params.vocab_only = True
    params.use_extra_bufts = False
    params.no_alloc = True
    model = api.llama_model_load_from_file(model_path.encode("utf-8"), params)
    if not model:
        raise NativeVocabularyError("native_vocab_only_load_failed")
    try:
        vocab = api.llama_model_get_vocab(model)
        if not vocab:
            raise NativeVocabularyError("native_vocab_missing")
        vectors = []
        for text, add_special, parse_special in cases:
            raw = text.encode("utf-8", errors="strict")
            # Byte fallback cannot need more than one token per byte + BOS/EOS.
            # A negative return is an error; never retry with guessed capacity.
            capacity = len(raw) + 2
            tokens = (c.c_int32 * capacity)()
            count = api.llama_tokenize(
                vocab, raw, len(raw), tokens, capacity, add_special, parse_special
            )
            if not 0 <= count <= capacity:
                raise NativeVocabularyError("native_token_capacity_failed")
            vector = list(tokens[:count])
            if any(token < 0 for token in vector):
                raise NativeVocabularyError("native_token_invalid")
            vectors.append(vector)
        return {
            "abi": ABI_REVISION,
            "add_bos": bool(api.llama_vocab_get_add_bos(vocab)),
            "add_eos": bool(api.llama_vocab_get_add_eos(vocab)),
            "vectors": vectors,
        }
    finally:
        api.llama_model_free(model)


def _child(
    sender: Connection, gate: Connection, dll: str, model: str, cases: list
) -> None:
    try:
        if not gate.poll(10) or gate.recv_bytes(1) != b"1":
            return
        # Do not collect unbounded C logs or leak document text to parent stdout.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                outcome: dict = {"ok": _native(dll, model, cases)}
            except NativeVocabularyError as exc:
                outcome = {"error": str(exc)}
            except Exception:
                outcome = {"error": "native_child_failed"}
            raw = json.dumps(outcome, separators=(",", ":")).encode()
            if len(raw) > MAX_RESULT_BYTES:
                raw = b'{"error":"native_output_limit"}'
            sender.send_bytes(raw)
    finally:
        sender.close()
        gate.close()


def run_vocabulary(
    dll: str,
    model: str,
    cases: list[tuple[str, bool, bool]],
    *,
    timeout_seconds: float = 20.0,
) -> dict:
    """Own one isolated child; no retries or native imports in the API process."""
    validate_abi()
    validate_inputs(cases)
    if not isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
        raise NativeVocabularyError("native_deadline_invalid")
    from enterprise_rag.workers.processes import _WindowsJob

    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    gate_reader, gate_writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_child, args=(sender, gate_reader, dll, model, cases)
    )
    process.daemon = True
    received: Queue[bytes | Exception] = Queue(maxsize=1)

    def receive() -> None:
        try:
            received.put(receiver.recv_bytes(MAX_RESULT_BYTES))
        except Exception as exc:
            received.put(exc)

    job = None
    reader = None
    started = False
    try:
        job = _WindowsJob(memory_bytes=MEMORY_BYTES)
        deadline = monotonic() + timeout_seconds
        process.start()
        started = True
        sender.close()
        gate_reader.close()
        job.assign_handle(process.sentinel)
        gate_writer.send_bytes(b"1")
        gate_writer.close()
        reader = Thread(target=receive, daemon=True)
        reader.start()
        try:
            raw = received.get(timeout=max(0, deadline - monotonic()))
        except Empty:
            raise NativeVocabularyError("native_child_timeout") from None
        process.join(max(0, deadline - monotonic()))
        if process.is_alive():
            raise NativeVocabularyError("native_child_timeout")
        if process.exitcode != 0 or not isinstance(raw, bytes):
            raise NativeVocabularyError("native_child_crash_or_ipc")
        outcome = json.loads(raw)
        if set(outcome) == {"error"}:
            raise NativeVocabularyError(outcome["error"])
        if set(outcome) != {"ok"}:
            raise NativeVocabularyError("native_child_protocol")
        return outcome["ok"]
    except (OSError, EOFError, ValueError) as exc:
        if isinstance(exc, NativeVocabularyError):
            raise
        raise NativeVocabularyError("native_process_unavailable") from None
    finally:
        sender.close()
        gate_reader.close()
        gate_writer.close()
        confirmed = True
        try:
            if job is not None:
                job.close(timeout_seconds=2)
        except OSError:
            confirmed = False
        if started:
            if process.is_alive():
                process.kill()
            process.join(2)
            if process.is_alive():
                confirmed = False
            else:
                process.close()
        else:
            process.close()
        if reader is not None:
            reader.join(2)
            confirmed = confirmed and not reader.is_alive()
        receiver.close()
        if not confirmed:
            raise NativeVocabularyError("native_cleanup_unconfirmed")
