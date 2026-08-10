"""Automatic HTTP capture.

Node has exactly one place to instrument — the global `fetch`. Python has none,
so this module wraps the two clients a Python voice agent actually issues
provider calls through:

* `httpx` — what the OpenAI/Azure OpenAI SDK uses.
* `aiohttp` — what most LiveKit streaming plugins (Deepgram, Sarvam, ElevenLabs)
  use for their REST calls.

Both wrappers are no-ops unless there is an ambient session *and* the request
URL matches a configured endpoint rule, so unrelated traffic from the host
application is never touched, timed or recorded.

The patches are process-wide and reference counted. A class attribute can only
hold one wrapper, so binding it to a single observer would mean the second
`VaaniObserver` in a process silently captured nothing and the first one's
`uninstall()` blinded everyone. Instead the wrapper resolves the observer from
the ambient session at call time, which is also the only correct answer when two
observers are active on different tasks.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from ._context import current_context
from ._payload import bounded_body, bounded_text, safe_error

logger = logging.getLogger("vaani_observer")

# A streaming response must not be drained to capture it; doing so would hold
# the first token back until the model finished. Anything larger than this
# multiple of the payload limit is summarised rather than read.
_MAX_DECLARED_BODY_MULTIPLIER = 4

_lock = threading.Lock()
_refcount = 0
_undo: List[Tuple[Any, str, Any]] = []
_targets: List[str] = []


class HttpInstrumentation:
    """A handle on the process-wide patches, reference counted per observer."""

    def __init__(self) -> None:
        self._installed = True
        _acquire_patches()

    @property
    def targets(self) -> List[str]:
        return list(_targets)

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        _release_patches()


def install_http_instrumentation(observer: Any) -> HttpInstrumentation:
    # `observer` is intentionally unused: the wrappers resolve the observer from
    # the ambient session, so every active observer is honoured.
    return HttpInstrumentation()


# ------------------------------------------------------------------- patching


def _acquire_patches() -> None:
    global _refcount
    with _lock:
        _refcount += 1
        if _refcount == 1:
            _patch_httpx()
            _patch_aiohttp()


def _release_patches() -> None:
    global _refcount
    with _lock:
        _refcount -= 1
        if _refcount > 0:
            return
        _refcount = 0
        for owner, name, original in reversed(_undo):
            setattr(owner, name, original)
        _undo.clear()
        _targets.clear()


def _register(owner: Any, name: str, original: Any, wrapper: Any) -> None:
    wrapper._vaani_patched = True
    setattr(owner, name, wrapper)
    _undo.append((owner, name, original))
    _targets.append(f"{owner.__name__}.{name}")


def _patch_httpx() -> None:
    try:
        import httpx
    except ImportError:  # pragma: no cover - optional dependency
        return

    original_async = httpx.AsyncClient.send

    async def send(self, request, **kwargs):  # noqa: ANN001
        operation, observer = _begin(str(request.url), "http")
        if operation is None:
            return await original_async(self, request, **kwargs)
        _attach_request(observer, operation, _httpx_request_body(request))
        try:
            response = await original_async(self, request, **kwargs)
        except BaseException as error:  # noqa: BLE001 - re-raised untouched
            operation.end(status="error", error=safe_error(error))
            raise
        _finish_httpx(observer, operation, response, kwargs.get("stream", False))
        return response

    _register(httpx.AsyncClient, "send", original_async, send)

    original_sync = httpx.Client.send

    def send_sync(self, request, **kwargs):  # noqa: ANN001
        operation, observer = _begin(str(request.url), "http")
        if operation is None:
            return original_sync(self, request, **kwargs)
        _attach_request(observer, operation, _httpx_request_body(request))
        try:
            response = original_sync(self, request, **kwargs)
        except BaseException as error:  # noqa: BLE001 - re-raised untouched
            operation.end(status="error", error=safe_error(error))
            raise
        _finish_httpx(observer, operation, response, kwargs.get("stream", False))
        return response

    _register(httpx.Client, "send", original_sync, send_sync)


def _patch_aiohttp() -> None:
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - optional dependency
        return

    original = aiohttp.ClientSession._request

    async def _request(self, method, str_or_url, **kwargs):  # noqa: ANN001
        operation, observer = _begin(str(str_or_url), "http")
        if operation is None:
            return await original(self, method, str_or_url, **kwargs)
        body = kwargs.get("data")
        if body is None:
            body = kwargs.get("json")
        _attach_request(observer, operation, body)
        try:
            response = await original(self, method, str_or_url, **kwargs)
        except BaseException as error:  # noqa: BLE001 - re-raised untouched
            operation.end(status="error", error=safe_error(error))
            raise
        payload: Dict[str, Any] = {"status": _safe(lambda: response.status)}
        if observer.options["capture"]["http_bodies"]:
            payload["body"] = await _aiohttp_body(observer, response)
        operation.end(
            status="ok" if _status_of(response, "status") < 400 else "error",
            response=payload,
            payload_bounded=observer.options["capture"]["http_bodies"],
        )
        return response

    _register(aiohttp.ClientSession, "_request", original, _request)


# ----------------------------------------------------------------- internals


def _begin(url: str, transport: str):
    """Start an operation for `url`, plus its observer, or `(None, None)`.

    The observer is resolved from the ambient session rather than captured at
    patch time, so every active observer in the process is honoured.
    """
    context = current_context()
    session = context.session if context else None
    if session is None:
        return None, None
    observer = session._observer
    if context.endpoint_id:
        rule = observer.rule_for(context.endpoint_id)
    else:
        try:
            rule = observer.classify_url(url)
        except TypeError as error:
            # An unparseable URL is simply not one of ours. Ambiguous *rules*
            # deliberately raise through to the caller instead: that is a
            # configuration bug the developer has to fix, and swallowing it
            # would silently mis-attribute every call to that endpoint.
            logger.debug("vaani: url classification skipped for %r (%s)", url, error)
            return None, None
    if not rule:
        return None, None
    operation = session.start_operation(
        type=rule["type"],
        endpoint_id=rule["id"],
        provider=rule.get("provider"),
        model=rule.get("model"),
        transport=transport,
        turn_id=context.turn_id,
    )
    return operation, observer


def _safe(thunk: Any, default: Any = None) -> Any:
    """Capture must never be the reason a caller's HTTP request fails."""
    try:
        return thunk()
    except BaseException as error:  # noqa: BLE001 - degradation, not a crash
        logger.debug("vaani: http capture step failed (%s)", error)
        return default


def _status_of(response: Any, attribute: str) -> int:
    return _safe(lambda: int(getattr(response, attribute)), 0) or 0


def _httpx_request_body(request: Any) -> Any:
    """Read a buffered httpx request body, never a streaming one.

    `httpx.Request.content` raises `RequestNotRead` for a streaming body, and
    consuming that stream here would corrupt the request being sent.
    """
    return _safe(lambda: request.content)


def _attach_request(observer: Any, operation: Any, body: Any) -> None:
    if not observer.options["capture"]["http_bodies"]:
        return
    limit = observer.options["capture"]["payload_max_bytes"]
    _safe(lambda: operation.event("request_body_captured"))
    _safe(lambda: operation.set_request({"body": bounded_body(body, limit)}, bounded=True))


def _finish_httpx(observer: Any, operation: Any, response: Any, streaming: bool) -> None:
    operation.end(
        status="ok" if _status_of(response, "status_code") < 400 else "error",
        response=_response_payload(observer, response, streaming),
        # The body is bounded on its own; re-bounding the wrapper would
        # collapse the whole response into a truncation marker.
        payload_bounded=observer.options["capture"]["http_bodies"],
    )


def _response_payload(observer: Any, response: Any, streaming: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": _safe(lambda: response.status_code)}
    if not observer.options["capture"]["http_bodies"]:
        return payload
    limit = observer.options["capture"]["payload_max_bytes"]
    if streaming or not _safe(lambda: response.is_closed, True):
        # Reading a streamed body here would delay the caller's first token,
        # which is exactly the latency this SDK exists to measure.
        payload["body"] = {"_capture_skipped": "Streaming response body."}
        return payload
    text = _safe(lambda: response.text)
    payload["body"] = (
        bounded_text(text, limit)
        if text is not None
        else {"_capture_skipped": "Response body was not readable."}
    )
    return payload


async def _aiohttp_body(observer: Any, response: Any) -> Any:
    """Read a bounded aiohttp body without ever waiting on an open stream.

    aiohttp returns before the body arrives, so a blind `await response.read()`
    would hold the caller until the last byte — fatal for a streamed LLM or TTS
    response. A body is therefore read only when the response declares a
    `Content-Length` that fits the capture bound: chunked, SSE and oversized
    responses are reported as skipped instead. `read()` caches into the response,
    so the application can still consume the payload afterwards.
    """
    limit = observer.options["capture"]["payload_max_bytes"]
    buffered = _safe(lambda: response._body)
    if buffered is None:
        content_type = _safe(lambda: response.headers.get("content-type"), "") or ""
        if "event-stream" in content_type:
            return {"_capture_skipped": "Streaming response body."}
        declared = _safe(lambda: response.headers.get("content-length"))
        if declared is None:
            return {"_capture_skipped": "Streaming or chunked response body."}
        size = _safe(lambda: int(declared))
        if size is None:
            return {"_capture_skipped": "Streaming or chunked response body."}
        if size > limit * _MAX_DECLARED_BODY_MULTIPLIER:
            return {"_truncated": True, "_original_bytes": size, "_preview": ""}
        buffered = await _safe_await(response.read())
        if buffered is None:
            return {"_capture_skipped": "Response body was not readable."}
    text = _safe(lambda: bytes(buffered).decode("utf-8", errors="replace"))
    if text is None:
        return {"_capture_skipped": "Response body was not readable."}
    return bounded_text(text, limit)


async def _safe_await(awaitable: Any) -> Any:
    try:
        return await awaitable
    except BaseException as error:  # noqa: BLE001 - degradation, not a crash
        logger.debug("vaani: http body capture failed (%s)", error)
        return None
