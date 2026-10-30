"""Pinned llama.cpp multimodal CLI; no API serving or special-token filtering."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from enterprise_rag.application.answer import ModelProtocolError, ModelUnavailable
from enterprise_rag.models.docling_grounding import DoclingGroundingModel
from enterprise_rag.models.docling_runtime import DoclingDecoderRuntime
from enterprise_rag.workers.processes import OwnedChild, _WindowsJob
from enterprise_rag.workers.temporary import (
    OwnedTemporaryDirectory as _CLITemporaryDirectory,
)

LLAMA_DOCLING_CONTRACT = "docling-llama-cpp-special-v1"
MODEL_MEMORY_BYTES = 2 * 1024**3
_SLOT = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cli_output(stdout: bytes, stderr: bytes) -> str:
    """Pinned upstream counters distinguish bounded completion from token limit.

    generate_response evaluates emitted non-EOG tokens one at a time. Reject
    conservatively at1023 runs; a full1024-token loop cannot become an answer
    merely because its final bytes happen to form valid markup.
    """
    if len(stdout) > 16_384 or len(stderr) > 512 * 1024:
        raise ModelProtocolError("Docling CLI output limit exceeded")
    try:
        text, diagnostic = stdout.decode("utf-8"), stderr.decode("utf-8")
        counts = re.findall(
            r"^llama_perf_context_print:\s+eval time\s*=\s*[0-9.]+ ms /\s*(\d+) runs",
            diagnostic,
            re.M,
        )
        prompts = re.findall(
            r"^llama_perf_context_print:\s+prompt eval time\s*=\s*[0-9.]+ ms /\s*(\d+) tokens",
            diagnostic,
            re.M,
        )
        if len(counts) != 1 or len(prompts) != 1:
            raise ValueError("Missing unambiguous completion counters")
        generated, prompt = int(counts[0]), int(prompts[0])
        if (
            not 0 < generated < 1023
            or not 0 < prompt <= 4096
            or prompt + generated > 4096
        ):
            raise ValueError("Incomplete or over-budget completion")
        if not text.strip():
            raise ValueError("Empty conversion")
        return text.strip()
    except (UnicodeError, ValueError) as error:
        raise ModelProtocolError("Docling CLI completion is unqualified") from error


class LlamaCppDoclingModel(DoclingGroundingModel):
    def __init__(
        self,
        runtime_manifest: Path,
        model: Path,
        model_sha256: str,
        projector: Path,
        projector_sha256: str,
        temp_root: Path,
        decoder: DoclingDecoderRuntime,
        *,
        observer: Callable[[dict], None] | None = None,
    ):
        if os.name != "nt":
            raise ValueError("The verified CPU CLI profile is Windows-only")
        if (
            not all(
                path.is_absolute()
                for path in (runtime_manifest, model, projector, temp_root)
            )
            or not temp_root.is_dir()
        ):
            raise ValueError("CLI runtime requires existing absolute paths")
        if runtime_manifest.stat().st_size > 65_536:
            raise ValueError("CLI runtime manifest exceeds its limit")
        manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("tag") != "b11669"
            or not isinstance(manifest.get("files"), dict)
            or not 1 <= len(manifest["files"]) <= 128
        ):
            raise ValueError("Unsupported CLI runtime manifest")
        self.executable = runtime_manifest.parent / "llama-mtmd-cli.exe"
        # The manifest is kept beside the verified runtime, independent of cwd.
        self.assets = []
        for name, digest in manifest["files"].items():
            path = (runtime_manifest.parent / name).resolve()
            if (
                not path.is_relative_to(runtime_manifest.parent.resolve())
                or path.is_symlink()
            ):
                raise ValueError("Runtime asset escapes its configured directory")
            self.assets.append((path, digest))
        self.assets.extend(((model, model_sha256), (projector, projector_sha256)))
        if self.executable not in {path for path, _ in self.assets}:
            raise ValueError("Runtime manifest omits the multimodal CLI")
        for path, digest in self.assets:
            if (
                not path.is_file()
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("Invalid pinned runtime asset")
        self.model, self.projector, self.temp_root, self.decoder = (
            model,
            projector,
            temp_root,
            decoder,
        )
        self.observer = observer
        self.options = {
            "num_gpu": 0,
            "num_thread": 2,
            "num_ctx": 4096,
            "num_predict": 1024,
        }

    def _verify_assets(self):
        for path, digest in self.assets:
            if path.is_symlink() or _sha256(path) != digest:
                raise ModelUnavailable("Pinned CLI runtime/model identity changed")

    def command(self, image: Path, instruction: str) -> tuple[str, ...]:
        if instruction not in {
            "Convert chart to table.",
            "Convert this page to docling.",
        }:
            raise ValueError("Unsupported conversion instruction")
        return (
            str(self.executable),
            "--model",
            str(self.model),
            "--mmproj",
            str(self.projector),
            "--image",
            str(image),
            "--prompt",
            instruction,
            "--threads",
            "2",
            "--threads-batch",
            "2",
            "--ctx-size",
            "4096",
            "--n-predict",
            "1024",
            "--device",
            "none",
            "--n-gpu-layers",
            "0",
            "--no-mmproj-offload",
            "--mmproj-device",
            "none",
            "--no-op-offload",
            "--fit",
            "off",
            "--no-warmup",
            "--temp",
            "0",
            "--seed",
            "0",
            "--no-escape",
            "--perf",
            "--no-log-prefix",
            "--no-log-timestamps",
            "--offline",
            "--verbosity",
            "4",
        )

    def _convert(self, image: bytes, instruction: str, deadline: float) -> str:
        if not _SLOT.acquire(blocking=False):
            raise ModelUnavailable("Local Docling CLI slot is busy")
        try:
            self._verify_assets()
            self._remaining(deadline)
            with _CLITemporaryDirectory(
                prefix="docling-cli-", dir=self.temp_root
            ) as temporary:
                root = Path(temporary)
                picture, output, diagnostic = (
                    root / "page.png",
                    root / "output.txt",
                    root / "stderr.txt",
                )
                picture.write_bytes(image)
                command = self.command(picture, instruction)
                # Native work starts only after OwnedChild has assigned its job.
                wrapper = """import os,sys,subprocess
assert os.environ.get('RAG_WORKER_START_GATE')=='1'
assert sys.stdin.buffer.read(1)==b'1'
with open(sys.argv[1],'wb') as out,open(sys.argv[2],'wb') as err:
    try:
        p=subprocess.run(sys.argv[3:],stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=0x08000000 if os.name=='nt' else 0)
        code=p.returncode
    except OSError as error:
        err.write(('native_launch_error:'+str(getattr(error,'winerror',None) or error.errno)).encode())
        code=1
sys.exit(code)
"""
                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("LLAMA_") and key != "GGML_BACKEND_PATH"
                }
                environment.update(
                    OMP_NUM_THREADS="2",
                    OMP_THREAD_LIMIT="2",
                    OPENBLAS_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                )
                owner = (
                    _WindowsJob(memory_bytes=MODEL_MEMORY_BYTES)
                    if os.name == "nt"
                    else None
                )
                try:
                    if self.observer is not None:
                        self.observer(
                            {"phase": "start", "command": command, "image": image}
                        )
                    with open(os.devnull, "wb") as quiet:
                        with OwnedChild(
                            (
                                sys.executable,
                                "-I",
                                "-B",
                                "-c",
                                wrapper,
                                str(output),
                                str(diagnostic),
                                *command,
                            ),
                            environment=environment,
                            output=quiet,
                            owner_job=owner,
                        ) as child:
                            while child.poll() is None:
                                self._remaining(deadline)
                                if (
                                    output.exists() and output.stat().st_size > 16_384
                                ) or (
                                    diagnostic.exists()
                                    and diagnostic.stat().st_size > 512 * 1024
                                ):
                                    raise ModelProtocolError(
                                        "CLI output limit exceeded"
                                    )
                                time.sleep(0.02)
                            if (output.exists() and output.stat().st_size > 16_384) or (
                                diagnostic.exists()
                                and diagnostic.stat().st_size > 512 * 1024
                            ):
                                raise ModelProtocolError("CLI output limit exceeded")
                            stdout = output.read_bytes() if output.exists() else b""
                            stderr = (
                                diagnostic.read_bytes() if diagnostic.exists() else b""
                            )
                            if self.observer is not None:
                                self.observer(
                                    {
                                        "phase": "result",
                                        "stdout": stdout,
                                        "stderr": stderr,
                                        "exit_code": child.poll(),
                                    }
                                )
                            if child.poll() != 0:
                                raise ModelUnavailable("Docling CLI process failed")
                    self._verify_assets()
                    return validate_cli_output(stdout, stderr)
                finally:
                    if owner is not None:
                        owner.close(timeout_seconds=2)
        finally:
            _SLOT.release()
