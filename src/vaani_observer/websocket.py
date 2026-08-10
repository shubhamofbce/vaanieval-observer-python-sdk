"""Provider-neutral websocket observation.

The Node SDK attaches to a `ws` EventEmitter. Python websocket clients are not
event emitters, so the equivalent here wraps the send/receive coroutines of the
two shapes a voice agent meets in practice:

* `aiohttp.ClientWebSocketResponse` — Deepgram, Sarvam, Cartesia, ElevenLabs.
* `websockets` client protocols — `send()` / `recv()`.

Only lifecycle, direction, kind and byte counts are recorded. Frame contents and
authentication headers are never stored, because a provider socket carries both
the API key and the raw speech of the call.

The span is `connection` scoped: a streaming socket normally stays open for the
whole call, so its duration is call length and must never be read as per-turn
latency. Per-turn work is recorded explicitly through `session.start_turn()`.
"""

from __future__ import annotations

import asyncio
import inspect
import weakref
import logging
import threading
from typing import Any, List, Optional, Tuple

from ._context import current_context
from ._payload import BYTES_TYPES, safe_error

__all__ = [
    "observe_websocket",
    "WebSocketHandle",
    "WebSocketInstrumentation",
    "install_websocket_instrumentation",
]

logger = logging.getLogger("vaani_observer")

# Only the lowest-level primitive of each client is wrapped. Real clients build
# their convenience methods on top of each other -- aiohttp's `send_json` calls
# `send_str`, and `receive_str` calls `receive` -- so wrapping the whole family
# would count one frame two or three times and inflate every byte total on the
# dashboard.
_SEND_PRIMITIVES = ("send_str", "send_bytes")
_SEND_FALLBACKS = ("send",)
_RECEIVE_PRIMITIVES = ("receive", "recv")


def _byte_count(value: Any) -> int:
    """Frame size, counting text in UTF-8 bytes. Unmeasurable frames count zero."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, BYTES_TYPES):
        return len(bytes(value))
    return 0


def _kind(value: Any) -> str:
    return "binary" if isinstance(value, BYTES_TYPES) else "text"


class _InertHandle:
    """What an unclassified socket gets: an object that does nothing, safely."""

    operation = None

    def record_sent(self, payload: Any) -> None:
        return None

    def record_received(self, payload: Any) -> None:
        return None

    def close(self, close_code: Any = None) -> None:
        return None

    def detach(self, status: str = "cancelled") -> None:
        return None

    def __enter__(self) -> "_InertHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.detach()


class WebSocketHandle:
    """Byte accounting and lifecycle for one provider socket."""

    def __init__(self, session: Any, socket: Any, operation: Any) -> None:
        self._session = session
        self._socket = socket
        self.operation = operation
        self._sent = 0
        self._received = 0
        self._closed = False
        self._patched: list[tuple[str, Any]] = []
        self._wrap()

    # ------------------------------------------------------------ accounting

    def record_sent(self, payload: Any) -> None:
        count = _byte_count(payload)
        self._sent += count
        self.operation.event(
            "sent_frame",
            direction="outbound",
            kind=_kind(payload),
            byte_count=count,
            total_byte_count=self._sent,
        )

    def record_received(self, payload: Any) -> None:
        count = _byte_count(payload)
        self._received += count
        self.operation.event(
            "received_frame",
            direction="inbound",
            kind=_kind(payload),
            byte_count=count,
            total_byte_count=self._received,
        )

    def record_error(self, error: BaseException) -> None:
        # A socket torn down because the call ended is not a transport fault.
        # Reporting shutdown as an error would make every healthy call look
        # like it lost its provider connection.
        if isinstance(error, asyncio.CancelledError):
            self._finish(
                status="cancelled",
                response={
                    "sent_bytes": self._sent,
                    "received_bytes": self._received,
                },
            )
            return
        self._finish(status="error", error=safe_error(error))

    def close(self, close_code: Any = None) -> None:
        self._finish(
            status="ok",
            response={
                "close_code": close_code,
                "sent_bytes": self._sent,
                "received_bytes": self._received,
            },
        )

    def detach(self, status: str = "cancelled") -> None:
        """Unwrap the socket and, if its span is still open, close it.

        `cancelled` is the default because an unwrapped-but-open socket really
        did stop being observed mid-flight. A call that ends normally passes
        `ok` instead: a streaming STT socket is *expected* to stay open for the
        whole call and be torn down at the end, and reporting that as cancelled
        made every healthy call look like it had lost its provider connection.
        """
        for name, original in reversed(self._patched):
            try:
                delattr(self._socket, name)
            except (AttributeError, TypeError):
                try:
                    setattr(self._socket, name, original)
                except (AttributeError, TypeError):
                    pass
        self._patched.clear()
        _forget_observed(self._socket)
        self._finish(
            status=status,
            response={"sent_bytes": self._sent, "received_bytes": self._received},
        )

    def __enter__(self) -> "WebSocketHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self.record_error(exc)
        self.detach()

    # -------------------------------------------------------------- internals

    def _forget(self) -> None:
        try:
            self._session.forget_socket(self)
        except AttributeError:
            pass

    def _finish(self, status: str, response: Optional[dict] = None, error: Optional[dict] = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._forget()
        self.operation.end(status=status, response=response or {}, error=error)

    def _wrap(self) -> None:
        self.operation.event("connected")
        primitives = [
            name for name in _SEND_PRIMITIVES if callable(getattr(self._socket, name, None))
        ]
        if not primitives:
            primitives = [
                name for name in _SEND_FALLBACKS if callable(getattr(self._socket, name, None))
            ]
        for name in primitives:
            self._wrap_send(name)
        for name in _RECEIVE_PRIMITIVES:
            if callable(getattr(self._socket, name, None)):
                self._wrap_receive(name)
                break
        self._wrap_close()

    def _install(self, name: str, replacement: Any, original: Any) -> None:
        try:
            setattr(self._socket, name, replacement)
        except (AttributeError, TypeError):
            # Slotted or frozen socket objects cannot be patched in place. The
            # span still records lifecycle; callers wanting frame counts on such
            # a socket use record_sent()/record_received() directly.
            return
        self._patched.append((name, original))

    def _wrap_send(self, name: str) -> None:
        original = getattr(self._socket, name, None)
        if original is None or not callable(original):
            return
        handle = self

        if inspect.iscoroutinefunction(original):

            # Bytes are only counted once the provider accepted them, and a
            # failed send ends the span the same way a failed receive does --
            # otherwise a socket that died mid-call is reported as healthy with
            # a byte total that was never actually on the wire.
            async def wrapped(payload: Any = None, *args: Any, **kwargs: Any) -> Any:
                try:
                    result = await original(payload, *args, **kwargs)
                except BaseException as error:  # noqa: BLE001 - re-raised untouched
                    handle.record_error(error)
                    raise
                handle.record_sent(payload)
                return result

        else:

            def wrapped(payload: Any = None, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                try:
                    result = original(payload, *args, **kwargs)
                except BaseException as error:  # noqa: BLE001 - re-raised untouched
                    handle.record_error(error)
                    raise
                handle.record_sent(payload)
                return result

        self._install(name, wrapped, original)

    def _wrap_receive(self, name: str) -> None:
        original = getattr(self._socket, name, None)
        if original is None or not callable(original):
            return
        handle = self

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                message = await original(*args, **kwargs)
            except BaseException as error:  # noqa: BLE001 - re-raised untouched
                handle.record_error(error)
                raise
            handle.record_received(_message_payload(message))
            return message

        if not inspect.iscoroutinefunction(original):
            return
        self._install(name, wrapped, original)

    def _wrap_close(self) -> None:
        original = getattr(self._socket, "close", None)
        if original is None or not callable(original):
            return
        handle = self

        if inspect.iscoroutinefunction(original):

            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = await original(*args, **kwargs)
                except BaseException as error:  # noqa: BLE001 - re-raised untouched
                    handle.record_error(error)
                    raise
                handle.close(_close_code(handle._socket, kwargs))
                return result

        else:

            def wrapped(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:  # noqa: BLE001 - re-raised untouched
                    handle.record_error(error)
                    raise
                handle.close(_close_code(handle._socket, kwargs))
                return result

        self._install("close", wrapped, original)


def _message_payload(message: Any) -> Any:
    """Unwrap an aiohttp `WSMessage` to the bytes that were actually on the wire."""
    data = getattr(message, "data", None)
    return data if data is not None else message


def _close_code(socket: Any, kwargs: dict) -> Any:
    if "code" in kwargs:
        return kwargs["code"]
    return getattr(socket, "close_code", None)


def observe_websocket(
    observer: Any,
    socket: Any,
    session: Any = None,
    url: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> Any:
    """Attach neutral lifecycle and frame accounting to an existing socket."""
    if session is None:
        context = current_context()
        session = context.session if context else None
    if session is None:
        raise ValueError(
            "observe_websocket needs a session or an active session.context() scope."
        )
    if endpoint_id is not None:
        rule = observer.rule_for(endpoint_id)
    elif url is not None:
        try:
            rule = observer.classify_url(url)
        except (TypeError, ValueError):
            rule = None
    else:
        rule = None
    if not rule:
        return _InertHandle()
    # A socket can now be reached twice -- once by the aiohttp patch and once by
    # an app that also hands it over explicitly. Observing it twice would double
    # every byte total, so the first attachment wins.
    if _already_observed(socket):
        return _InertHandle()
    operation = session.start_operation(
        type=rule["type"],
        endpoint_id=rule["id"],
        provider=rule.get("provider"),
        model=rule.get("model"),
        transport="websocket",
        scope="connection",
    )
    handle = WebSocketHandle(session, socket, operation)
    _mark_observed(socket)
    return handle


# Sockets that reject attribute assignment still need de-duplication, so
# identity is remembered out-of-band. The registry is weak so it never keeps a
# closed socket alive.
_OBSERVED: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _already_observed(socket: Any) -> bool:
    if getattr(socket, "_vaani_observed", False):
        return True
    try:
        return socket in _OBSERVED
    except TypeError:
        return False


def _mark_observed(socket: Any) -> None:
    try:
        socket._vaani_observed = True
        return
    except (AttributeError, TypeError):
        pass
    try:
        _OBSERVED.add(socket)
    except TypeError:
        pass


def _forget_observed(socket: Any) -> None:
    """Allow a detached socket to be observed again by a later session."""
    try:
        del socket._vaani_observed
    except (AttributeError, TypeError):
        pass
    try:
        _OBSERVED.discard(socket)
    except TypeError:
        pass


# ------------------------------------------------------- ambient auto-observation
#
# `observe_websocket` is the explicit door, and it is the only one the Node SDK
# has: the app owns its provider sockets there and hands them over. A Python
# agent framework does not -- LiveKit's STT and TTS plugins open their own
# aiohttp sockets deep inside the plugin -- so without this the manifest would
# advertise `websocket_instrumentation: active` while never producing a single
# connection span. Patching the one coroutine every aiohttp websocket is born
# from is what makes that claim true.

_ws_lock = threading.Lock()
_ws_refcount = 0
_ws_undo: List[Tuple[Any, str, Any]] = []
_ws_targets: List[str] = []


class WebSocketInstrumentation:
    """A handle on the process-wide websocket patches, reference counted."""

    def __init__(self) -> None:
        self._installed = True
        _acquire_ws_patches()

    @property
    def targets(self) -> List[str]:
        return list(_ws_targets)

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        _release_ws_patches()


def install_websocket_instrumentation(observer: Any) -> WebSocketInstrumentation:
    # `observer` is unused for the same reason as the HTTP side: the wrapper
    # resolves the observer from the ambient session, so every active observer
    # is honoured rather than only the one that installed first.
    return WebSocketInstrumentation()


def _acquire_ws_patches() -> None:
    global _ws_refcount
    with _ws_lock:
        _ws_refcount += 1
        if _ws_refcount == 1:
            _patch_aiohttp_ws()


def _release_ws_patches() -> None:
    global _ws_refcount
    with _ws_lock:
        _ws_refcount -= 1
        if _ws_refcount > 0:
            return
        _ws_refcount = 0
        for owner, name, original in reversed(_ws_undo):
            setattr(owner, name, original)
        _ws_undo.clear()
        _ws_targets.clear()


def _patch_aiohttp_ws() -> None:
    try:
        import aiohttp
    except ImportError:
        return
    owner = getattr(aiohttp, "ClientSession", None)
    # `_ws_connect` is the coroutine that actually returns the socket.
    # `ws_connect` only wraps it in a context manager, so patching this one
    # covers both `await ws_connect(...)` and `async with ws_connect(...)`.
    original = getattr(owner, "_ws_connect", None)
    if owner is None or original is None or getattr(original, "_vaani_patched", False):
        return

    async def wrapped(self: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        socket = await original(self, url, *args, **kwargs)
        try:
            context = current_context()
            session = context.session if context else None
            if session is not None and not session.ended:
                handle = observe_websocket(
                    session._observer, socket, session=session, url=str(url)
                )
                # Nobody else owns this socket, so the session has to finalize
                # its span if the call ends before the socket closes. If the
                # call is already ending, close the span here instead of losing
                # it.
                if not session.register_socket(handle):
                    handle.detach()
        except Exception as error:  # noqa: BLE001 - capture must never break a connect
            logger.debug("vaani: websocket auto-observation skipped (%s)", error)
        return socket

    wrapped._vaani_patched = True
    setattr(owner, "_ws_connect", wrapped)
    _ws_undo.append((owner, "_ws_connect", original))
    _ws_targets.append("ClientSession._ws_connect")
