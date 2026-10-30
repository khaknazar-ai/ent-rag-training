"""Explicit lite/lexical parent demo factory; no inference or network at startup."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path

from enterprise_rag.api.app import StaticTokenAuthenticator
from enterprise_rag.api.app import create_app as create_api
from enterprise_rag.application.lexical_parent import LexicalParentRetrieval
from enterprise_rag.application.semantic_answer import SemanticAnswerService
from enterprise_rag.application.visual_answer import VisualAnswerService
from enterprise_rag.bootstrap import (
    _history_model_configuration,
    build_services,
    with_pdf_regions,
)
from enterprise_rag.models.docling_runtime import DoclingDecoderRuntime
from enterprise_rag.models.llama_cpp_docling import (
    LLAMA_DOCLING_CONTRACT,
    MODEL_MEMORY_BYTES,
    LlamaCppDoclingModel,
)
from enterprise_rag.models.native_summary_counter import (
    NativeSummaryCounter,
    NativeTokenizerConfig,
)
from enterprise_rag.models.ollama_docling import (
    DOCLING_CONTRACT,
    OllamaDoclingGroundingModel,
)
from enterprise_rag.models.ollama_grounded import OllamaGroundedAnswerModel
from enterprise_rag.models.ollama_visual import OllamaVisualGroundingModel
from enterprise_rag.models.raster_geometry import (
    RASTER_CONTRACT,
    RasterGeometryModel,
    RasterGeometryRuntime,
)
from enterprise_rag.models.semantic_prompt import NativeSemanticCounter
from enterprise_rag.models.semantic_runtime import OllamaSemanticRuntimeVerifier
from enterprise_rag.security import ActorContext
from enterprise_rag.settings import Settings


def read_tokenizer_config(path: Path) -> NativeTokenizerConfig:
    if path.stat().st_size > 65536:
        raise ValueError("Tokenizer configuration exceeds its limit")
    values = json.loads(path.read_text(encoding="utf-8"))
    for name in ("manifest", "model", "dll"):
        values[name] = Path(values[name])
    values["dll_sha256"] = tuple(tuple(pair) for pair in values["dll_sha256"])
    return NativeTokenizerConfig(**values)


def _demo_authenticator_from_env() -> StaticTokenAuthenticator:
    """Build explicit local demo identities without changing default API auth."""

    tenant_id = os.environ.get("RAG_DEMO_TENANT_ID", "")
    admin_token = os.environ.get("RAG_API_TOKEN", "")
    admin_user_id = os.environ.get("RAG_DEMO_USER_ID", "")
    viewer_token = os.environ.get("RAG_DEMO_VIEWER_TOKEN", "")
    viewer_user_id = os.environ.get("RAG_DEMO_VIEWER_USER_ID", "")
    admin_values = (admin_token, tenant_id, admin_user_id)
    viewer_values = (viewer_token, viewer_user_id)
    if any(admin_values) and not all(admin_values):
        raise ValueError("Configure the complete local demo admin token identity")
    if any(viewer_values) and not all(viewer_values):
        raise ValueError("Configure both local demo viewer token and user ID")
    if all(viewer_values) and not all(admin_values):
        raise ValueError("Configure the local demo admin identity before the viewer")
    if all(viewer_values) and viewer_user_id == admin_user_id:
        raise ValueError("Demo admin and viewer identities must be distinct")

    tokens: dict[str, ActorContext] = {}
    if all(admin_values):
        tokens[admin_token] = ActorContext(tenant_id=tenant_id, user_id=admin_user_id)
    if all(viewer_values):
        if viewer_token in tokens:
            raise ValueError("Demo admin and viewer tokens must be distinct")
        tokens[viewer_token] = ActorContext(tenant_id=tenant_id, user_id=viewer_user_id)
    return StaticTokenAuthenticator(tokens)


def _demo_pdf_regions_from_env(services):
    """Opt into bounded page rendering only with explicit local paths."""

    renderer = os.environ.get("RAG_DEMO_PDF_RENDERER", "").strip()
    temp_root = os.environ.get("RAG_DEMO_PDF_RENDER_TEMP", "").strip()
    if not renderer and not temp_root:
        return services
    if not renderer or not temp_root:
        raise ValueError(
            "Configure both RAG_DEMO_PDF_RENDERER and RAG_DEMO_PDF_RENDER_TEMP"
        )
    renderer_path = Path(renderer)
    temp_path = Path(temp_root)
    if (
        not renderer_path.is_absolute()
        or not renderer_path.is_file()
        or not temp_path.is_absolute()
        or not temp_path.is_dir()
    ):
        raise ValueError(
            "Demo PDF renderer must be an existing absolute executable path and "
            "render temp must be an existing absolute directory"
        )
    return with_pdf_regions(
        services,
        renderer_command=(str(renderer_path),),
        temp_root=temp_path,
    )


def create_lite_app():
    """Start the explicit model-free demo with optional ruled-table page answers."""
    settings = Settings.from_env()
    if settings.profile != "lite" or settings.llm_provider != "none":
        raise ValueError(
            "The Lite demo requires RAG_PROFILE=lite and RAG_LLM_PROVIDER=none"
        )
    services = _demo_pdf_regions_from_env(build_services(settings))
    if (
        services.visual_regions is not None
        and services.visual_regions.renderer is not None
        and services.query_history is not None
    ):
        services = replace(
            services,
            visual_answer=VisualAnswerService(
                services.search,
                None,
                services.visual_regions,
                services.query_history,
                model_configuration={
                    "provider": "structured-table",
                    "model": None,
                    "contract": "ruled-table-exact-header-row-v1",
                },
            ),
        )
    return create_api(services, authenticator=_demo_authenticator_from_env())


def create_visual_app():
    """Opt into a pinned local vision model independently of text-model assets."""
    return _create_visual_app(docling=False)


def create_docling_app():
    """Explicit Docling conversion contract; never enabled by the default demo."""
    return _create_visual_app(docling=True)


def _create_visual_app(*, docling: bool):
    settings = Settings.from_env()
    if settings.profile != "lite" or settings.llm_provider != "none":
        raise ValueError("Visual-only startup requires lite and RAG_LLM_PROVIDER=none")
    config_key = "RAG_DEMO_DOCLING_CONFIG" if docling else "RAG_DEMO_VISUAL_CONFIG"
    path = Path(os.environ.get(config_key, ""))
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError(f"Set {config_key} to an existing absolute JSON file")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {"base_url", "model", "digest"}
    if docling:
        required_fields |= {"decoder_python", "decoder_packages", "decoder_temp"}
    if not isinstance(configuration, dict) or set(configuration) != required_fields:
        raise ValueError("Visual configuration has missing or unknown fields")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in configuration.values()
    ):
        raise ValueError("Visual configuration fields must be nonempty strings")
    model: OllamaDoclingGroundingModel | OllamaVisualGroundingModel
    if docling:
        model = OllamaDoclingGroundingModel(
            configuration["base_url"],
            configuration["model"],
            expected_digest=configuration["digest"],
            decoder=DoclingDecoderRuntime(
                Path(configuration["decoder_python"]),
                Path(configuration["decoder_packages"]),
                Path(configuration["decoder_temp"]),
            ),
        )
    else:
        model = OllamaVisualGroundingModel(
            configuration["base_url"],
            configuration["model"],
            expected_digest=configuration["digest"],
            image_max_edge=384,
            timeout_seconds=min(settings.visual_timeout_seconds, 180),
            keep_alive="0",
        )
    services = _demo_pdf_regions_from_env(build_services(settings))
    if (
        services.visual_regions is None
        or services.visual_regions.renderer is None
        or services.query_history is None
    ):
        raise ValueError(
            "Visual-only startup requires the explicit PDF renderer and query history"
        )
    services = replace(
        services,
        visual_answer=VisualAnswerService(
            services.search,
            model,
            services.visual_regions,
            services.query_history,
            model_configuration={
                "provider": "ollama-docling" if docling else "ollama-visual",
                "model": configuration["model"],
                "digest": configuration["digest"],
                "contract": DOCLING_CONTRACT
                if docling
                else "visual-page-citation-v4-conditional-fit384",
                "table_answering": "ruled-table-exact-header-row-v1",
                "image_preprocessing": "trim-white-page-margins-fit512-v1"
                if docling
                else "trim-white-page-margins-fit384-v1",
                "image_max_edge": 512 if docling else 384,
                "num_gpu": 0,
                "num_thread": 2,
                "num_ctx": 4096,
                "num_predict": 1024 if docling else 256,
                "timeout_seconds": 180 if docling else model.timeout_seconds,
                "keep_alive": "0",
                "identity_check": "tags-before-after-within-request-deadline-v1",
                **(
                    {
                        "decoder": "docling-core-2.104.0-owned10s-1GiB-v1",
                        "diagram_geometry": "persisted-ocr-inside-docling-flowchart-v1",
                    }
                    if docling
                    else {}
                ),
            },
        ),
    )
    return create_api(services, authenticator=_demo_authenticator_from_env())


def create_docling_cli_app():
    """Explicit owned CPU CLI, with no shared Ollama process or startup load."""
    settings = Settings.from_env()
    if settings.profile != "lite" or settings.llm_provider != "none":
        raise ValueError("Docling CLI requires lite and RAG_LLM_PROVIDER=none")
    path = Path(os.environ.get("RAG_DEMO_DOCLING_CLI_CONFIG", ""))
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("Configure an existing absolute Docling CLI JSON file")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "runtime_manifest",
        "model_path",
        "model_sha256",
        "projector_path",
        "projector_sha256",
        "inference_temp",
        "decoder_python",
        "decoder_packages",
        "decoder_temp",
    }
    if (
        not isinstance(configuration, dict)
        or set(configuration) != required
        or any(
            not isinstance(value, str) or not value.strip()
            for value in configuration.values()
        )
    ):
        raise ValueError("Invalid Docling CLI configuration")
    decoder = DoclingDecoderRuntime(
        Path(configuration["decoder_python"]),
        Path(configuration["decoder_packages"]),
        Path(configuration["decoder_temp"]),
    )
    model = LlamaCppDoclingModel(
        Path(configuration["runtime_manifest"]),
        Path(configuration["model_path"]),
        configuration["model_sha256"],
        Path(configuration["projector_path"]),
        configuration["projector_sha256"],
        Path(configuration["inference_temp"]),
        decoder,
    )
    services = _demo_pdf_regions_from_env(build_services(settings))
    if (
        services.visual_regions is None
        or services.visual_regions.renderer is None
        or services.query_history is None
    ):
        raise ValueError("Docling CLI requires the explicit PDF renderer and history")
    services = replace(
        services,
        visual_answer=VisualAnswerService(
            services.search,
            model,
            services.visual_regions,
            services.query_history,
            model_configuration={
                "provider": "llama-cpp-docling",
                "contract": LLAMA_DOCLING_CONTRACT,
                "model_sha256": configuration["model_sha256"],
                "projector_sha256": configuration["projector_sha256"],
                "runtime": "llama.cpp-b11669-win-cpu-x64",
                "special_tokens": "common_token_to_piece-special-true",
                "num_gpu": 0,
                "num_thread": 2,
                "num_ctx": 4096,
                "num_predict": 1024,
                "image_max_edge": 512,
                "timeout_seconds": 180,
                "memory_bytes": MODEL_MEMORY_BYTES,
                "lifetime": "owned-per-request",
                "decoder": "docling-core-2.104.0-owned10s-1GiB-v1",
                "diagram_geometry": "persisted-ocr-inside-docling-flowchart-v1",
            },
        ),
    )
    return create_api(services, authenticator=_demo_authenticator_from_env())


def create_raster_app():
    """Explicit bounded OpenCV + existing OCR; no generative model startup."""
    settings = Settings.from_env()
    if settings.profile != "lite" or settings.llm_provider != "none":
        raise ValueError("Raster geometry requires lite and llm_provider=none")
    path = Path(os.environ.get("RAG_DEMO_RASTER_CONFIG", ""))
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("Set an existing absolute RAG_DEMO_RASTER_CONFIG")
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) != {"python", "packages", "temp"}
        or any(not isinstance(v, str) or not v.strip() for v in config.values())
    ):
        raise ValueError("Invalid raster geometry configuration")
    runtime = RasterGeometryRuntime(
        *(Path(config[k]) for k in ("python", "packages", "temp"))
    )
    services = _demo_pdf_regions_from_env(build_services(settings))
    if (
        services.visual_regions is None
        or services.visual_regions.renderer is None
        or services.query_history is None
    ):
        raise ValueError(
            "Raster geometry requires the explicit PDF renderer and history"
        )
    services = replace(
        services,
        visual_answer=VisualAnswerService(
            services.search,
            RasterGeometryModel(runtime),
            services.visual_regions,
            services.query_history,
            model_configuration={
                "provider": "opencv-persisted-ocr",
                "contract": RASTER_CONTRACT,
                "opencv": "4.13.0",
                "num_gpu": 0,
                "num_thread": 2,
                "timeout_seconds": 10,
                "memory_bytes": 1024**3,
                "llm_vlm_calls": 0,
                "bar_numbers": "pixel-heights-only-not-data-values",
            },
        ),
    )
    return create_api(services, authenticator=_demo_authenticator_from_env())


def create_app():
    """uvicorn enterprise_rag.api.demo:create_app --factory, explicitly opt-in.

    Requires ordinary RAG_API_* actor settings, RAG_LLM_PROVIDER=ollama and a
    RAG_PARENT_TOKENIZER_CONFIG file pinning already installed assets. Default
    API/bootstrap remains unchanged. Optional RAG_DEMO_PDF_RENDERER and
    RAG_DEMO_PDF_RENDER_TEMP paths enable page rendering on explicit UI request.
    No service, weights or download is started.
    """
    path = os.environ.get("RAG_PARENT_TOKENIZER_CONFIG", "")
    if not path.strip():
        raise ValueError("Set RAG_PARENT_TOKENIZER_CONFIG for the parent demo")
    config = read_tokenizer_config(Path(path))
    settings = Settings.from_env()
    if settings.llm_provider != "ollama" or settings.llm_model != config.model_tag:
        raise ValueError("Configure Ollama with the exact pinned tokenizer model tag")
    services = _demo_pdf_regions_from_env(build_services(settings))
    resources = ExitStack()
    try:
        native = resources.enter_context(NativeSummaryCounter(config))
        counter = resources.enter_context(
            NativeSemanticCounter(native, context_tokens=4096)
        )
        verifier = resources.enter_context(
            OllamaSemanticRuntimeVerifier(counter, settings.llm_base_url)
        )
        model = OllamaGroundedAnswerModel(
            settings.llm_base_url,
            settings.llm_model,
            counter=counter,
            runtime_verifier=verifier,
            timeout_seconds=min(settings.llm_timeout_seconds, 90),
            context_tokens=4096,
            output_tokens=256,
            keep_alive="5m",
        )
        configuration = _history_model_configuration(model)
        configuration.update(
            digest=counter.binding.model_digest,
            contract="grounding-parent-raw-v1",
            tokenizer_sha256=counter.binding.tokenizer_sha256,
            template_sha256=counter.binding.template_sha256,
        )
        services = replace(
            services,
            semantic_answer=SemanticAnswerService(
                LexicalParentRetrieval(services.search),
                model,
                history=services.query_history,
                model_configuration=configuration,
            ),
        )
        if (
            services.visual_regions is not None
            and services.visual_regions.renderer is not None
            and services.query_history is not None
        ):
            visual_model = OllamaVisualGroundingModel(
                settings.llm_base_url,
                settings.llm_model,
                timeout_seconds=min(settings.visual_timeout_seconds, 180),
                context_tokens=4096,
                output_tokens=256,
                keep_alive="5m",
            )
            visual_configuration = {
                "provider": "ollama-visual",
                "model": settings.llm_model,
                "digest": counter.binding.model_digest,
                "contract": "visual-page-citation-v4-conditional",
                "table_answering": "ruled-table-exact-header-row-v1",
                "image_preprocessing": "trim-white-page-margins-v1",
                "num_gpu": 0,
                "num_thread": 2,
                "num_ctx": 4096,
                "num_predict": 256,
                "timeout_seconds": min(settings.visual_timeout_seconds, 180),
            }
            services = replace(
                services,
                visual_answer=VisualAnswerService(
                    services.search,
                    visual_model,
                    services.visual_regions,
                    services.query_history,
                    model_configuration=visual_configuration,
                ),
            )
        app = create_api(services, authenticator=_demo_authenticator_from_env())
    except BaseException:
        resources.close()
        raise
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with previous(application) as state:
                yield state
        finally:
            resources.close()

    app.router.lifespan_context = lifespan
    return app
