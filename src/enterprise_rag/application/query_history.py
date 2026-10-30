"""Short persistence boundaries around existing answer operations."""

from __future__ import annotations

import asyncio
import json
import time
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

from enterprise_rag.application.context import CONTEXT_PACKING_REVISION
from enterprise_rag.application.quote_matching import QUOTE_MATCHING_REVISION
from enterprise_rag.ports.query_history import (
    QUERY_TRACE_STAGES,
    QueryCompletion,
    QueryHistoryUnavailable,
    QueryRequest,
)
from enterprise_rag.retrieval.query_processing import QUERY_REVISION

_QUERY_STAGE_TIMINGS: ContextVar[dict[str, float] | None] = ContextVar(
    "enterprise_rag_query_stage_timings", default=None
)


@contextmanager
def measure_query_stage(name: str):
    """Accumulate actual wall time for one named stage in the active query."""
    if name not in QUERY_TRACE_STAGES:
        raise ValueError("Unknown query trace stage")
    timings = _QUERY_STAGE_TIMINGS.get()
    if timings is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def history_request(query, normalized, strategy, model_configuration):
    if any(
        not isinstance(text, str) or not text.strip() or len(text) > 500
        for text in (query, normalized)
    ):
        raise ValueError("Question and retrieval input must contain1..500 characters")
    configuration = {
        "history_contract": "query-history-v1",
        "context_revision": CONTEXT_PACKING_REVISION,
        "quote_revision": QUOTE_MATCHING_REVISION,
        "query_revision": QUERY_REVISION,
        "hierarchy_context_revision": "native-heading-ranges-v1"
        if strategy.startswith("semantic:")
        else None,
        "model": model_configuration
        or {"provider": "injected-unidentified", "model": None},
    }
    return QueryRequest(
        query,
        normalized,
        unicodedata.normalize("NFKC", normalized)
        .replace("\u201c", '"')
        .replace("\u201d", '"'),
        strategy,
        json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )


def _complete(history, actor, query_id, outcome):
    try:
        history.complete(actor, query_id, outcome)
    except Exception:
        # Commit acknowledgement can fail after the commit. Read before any
        # further mutation: never overwrite an already committed terminal state.
        try:
            saved = history.get(actor, query_id)
        except Exception:
            saved = None
        if saved is not None and saved.status == outcome.status:
            left = replace(saved.result, query_run_id=None) if saved.result else None
            right = (
                replace(outcome.result, query_run_id=None) if outcome.result else None
            )
            expected_latencies = {
                "total_seconds": outcome.elapsed_seconds,
                **(outcome.stage_latencies or {}),
            }
            if (
                left == right
                and saved.parent_context_refs == outcome.parent_context_refs
                and saved.error_code == outcome.error_code
                and saved.stage_latencies == expected_latencies
            ):
                return
        # Only a confirmed unfinished start permits one bounded failure write.
        # An unreadable DB retains running/unknown state and fails explicitly.
        if (
            saved is not None
            and saved.status == "running"
            and outcome.status != "failed"
        ):
            try:
                history.complete(
                    actor,
                    query_id,
                    QueryCompletion(
                        "failed", outcome.elapsed_seconds, None, "history_write_failed"
                    ),
                )
            except Exception:
                pass
        raise QueryHistoryUnavailable(query_id) from None


def _outcome(value, elapsed, semantic, stage_latencies):
    result = value.result if semantic else value
    status = (
        "evidence_only"
        if result is None
        else "abstained"
        if result.abstained
        else "answered"
    )
    from enterprise_rag.retrieval.parents import parent_context_references as references

    expansion = getattr(value, "parent_expansion", None) if semantic else None
    return QueryCompletion(
        status,
        elapsed,
        result,
        parent_context_refs=references(expansion) if expansion else (),
        stage_latencies=dict(stage_latencies),
    )


def _identified(value, query_id, semantic):
    if semantic:
        return replace(
            value,
            query_run_id=query_id,
            result=replace(value.result, query_run_id=query_id)
            if value.result is not None
            else None,
        )
    return replace(value, query_run_id=query_id)


def record_answer(history, actor, request, operation):
    query_id = history.start(actor, request)
    started = time.perf_counter()
    stage_latencies: dict[str, float] = {}
    trace_token = _QUERY_STAGE_TIMINGS.set(stage_latencies)
    try:
        result = operation()
    except Exception as error:
        _QUERY_STAGE_TIMINGS.reset(trace_token)
        _complete(
            history,
            actor,
            query_id,
            QueryCompletion(
                "failed",
                time.perf_counter() - started,
                None,
                "execution_failed",
                stage_latencies=dict(stage_latencies),
            ),
        )
        vars(error)["query_run_id"] = query_id
        raise
    _QUERY_STAGE_TIMINGS.reset(trace_token)
    _complete(
        history,
        actor,
        query_id,
        _outcome(result, time.perf_counter() - started, False, stage_latencies),
    )
    return _identified(result, query_id, False)


async def _settle_history(task):
    """Wait for the same bounded adapter call; cancellation never retries I/O."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _cancelled_history(
    history, actor, query_id, elapsed, error, stage_latencies=None
):
    vars(error)["query_run_id"] = query_id
    try:
        await _settle_history(
            asyncio.create_task(
                asyncio.to_thread(
                    _complete,
                    history,
                    actor,
                    query_id,
                    QueryCompletion(
                        "failed",
                        elapsed,
                        None,
                        "execution_failed",
                        stage_latencies=dict(stage_latencies or {}),
                    ),
                )
            )
        )
    except Exception:
        # Cancellation must still reach the caller. A failed/unreadable store
        # is explicitly unconfirmed; never invent a successful terminal write.
        vars(error)["query_history_unconfirmed"] = True


async def _async_complete(history, actor, query_id, outcome):
    call = asyncio.create_task(
        asyncio.to_thread(_complete, history, actor, query_id, outcome)
    )
    try:
        await asyncio.shield(call)
    except asyncio.CancelledError as error:
        vars(error)["query_run_id"] = query_id
        try:
            await _settle_history(call)
        except Exception:
            vars(error)["query_history_unconfirmed"] = True
        # An already committed outcome survives cancellation of its awaiter.
        raise error from None


async def record_semantic_answer(history, actor, request, operation):
    start = asyncio.create_task(asyncio.to_thread(history.start, actor, request))
    try:
        query_id = await asyncio.shield(start)
    except asyncio.CancelledError as error:
        try:
            query_id = await _settle_history(start)
        except Exception:
            vars(error).update(query_run_id=None, query_history_unconfirmed=True)
        else:
            await _cancelled_history(history, actor, query_id, 0.0, error)
        raise error from None
    started = time.perf_counter()
    stage_latencies: dict[str, float] = {}
    trace_token = _QUERY_STAGE_TIMINGS.set(stage_latencies)
    try:
        result = await operation()
    except asyncio.CancelledError as error:
        _QUERY_STAGE_TIMINGS.reset(trace_token)
        await _cancelled_history(
            history,
            actor,
            query_id,
            time.perf_counter() - started,
            error,
            stage_latencies,
        )
        raise error from None
    except Exception as error:
        _QUERY_STAGE_TIMINGS.reset(trace_token)
        await _async_complete(
            history,
            actor,
            query_id,
            QueryCompletion(
                "failed",
                time.perf_counter() - started,
                None,
                "execution_failed",
                stage_latencies=dict(stage_latencies),
            ),
        )
        vars(error)["query_run_id"] = query_id
        raise
    _QUERY_STAGE_TIMINGS.reset(trace_token)
    await _async_complete(
        history,
        actor,
        query_id,
        _outcome(result, time.perf_counter() - started, True, stage_latencies),
    )
    return _identified(result, query_id, True)
