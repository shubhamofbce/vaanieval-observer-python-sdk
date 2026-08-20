"""Session, turn and operation recording.

Direct port of `nodejs-sdk/src/session.js`. The on-disk package it produces —
`manifest.json`, `events.jsonl`, `call.audio` — is byte
compatible with the Node SDK's, so the same dashboard ingests either without
knowing which language produced the call.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Mapping, Optional, Set

from ._context import ObserverContext, reset_context, set_context
from ._audio import TEMP_TRACK_FILES, compose_stereo
from ._diagnostics import warn_once
from ._payload import (
    BYTES_TYPES,
    bounded_payload,
    json_bytes,
    sha256,  # re-exported: the upload path checksums objects with it
)
from ._writer import SpoolWriter
from ._version import __version__

logger = logging.getLogger("vaani_observer")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .observer import VaaniObserver

__all__ = ["Session", "Operation", "Turn", "FinalizedSession", "sha256"]

SDK = {"name": "@vaanieal/observer", "language": "python", "version": __version__}

OPERATION_TYPES = ("stt", "llm", "tts", "tool")
TRACKS = ("caller", "agent")


@dataclass(frozen=True)
class FinalizedSession:
    """What a finalized local package is: an id, a directory and its manifest."""

    session_id: str
    directory: str
    manifest: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:  # convenience for dict-style access
        return getattr(self, key)


def pcm_duration_ms(byte_length: int, fmt: Mapping[str, Any]) -> Optional[float]:
    """Playback duration of a raw PCM chunk, or None when the format is unknown."""
    if fmt.get("encoding") != "pcm_s16le":
        return None
    rate = fmt.get("sample_rate_hz")
    channels = fmt.get("channels")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        return None
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        return None
    return _number((byte_length / (rate * channels * 2)) * 1000)


def _number(value: float) -> Any:
    """Serialize an integral float as an int, the way `JSON.stringify` does.

    Keeps the Python package byte-identical to the Node one for the common case
    of whole-millisecond frame durations.
    """
    return int(value) if isinstance(value, float) and value.is_integer() else value


class Operation:
    """One unit of provider work: an STT, LLM, TTS or tool span.

    Nothing is written until `end()`. An operation that is never ended is
    deliberately absent from the package rather than persisted as a span with no
    outcome, which would silently skew every duration percentile.
    """

    __slots__ = ("_session", "_event", "_done")

    def __init__(self, session: "Session", event: Dict[str, Any]) -> None:
        self._session = session
        self._event = event
        self._done = False

    @property
    def ended(self) -> bool:
        return self._done

    @property
    def turn_id(self) -> Optional[str]:
        return self._event["turn_id"]

    def event(self, name: str, data: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        """Record a milestone.

        Repeated milestones of the same name accumulate (`count`, first
        `occurred_at_ms`, latest `last_at_ms`) instead of overwriting, so a
        high-frequency transport keeps useful timing without emitting one event
        per frame.
        """
        if self._done:
            return
        payload: Dict[str, Any] = dict(data or {})
        payload.update(kwargs)
        at = payload.pop("occurred_at_ms", None)
        if at is None:
            at = self._session.now()
        previous = self._event["milestones"].get(name)
        if previous:
            merged = {**previous, **payload}
            merged["occurred_at_ms"] = previous["occurred_at_ms"]
            merged["last_at_ms"] = at
            merged["count"] = previous.get("count", 1) + 1
            self._event["milestones"][name] = merged
        else:
            self._event["milestones"][name] = {
                "occurred_at_ms": at,
                "last_at_ms": at,
                "count": 1,
                **payload,
            }

    def set_turn(self, turn_id: Any) -> None:
        if not self._done:
            self._event["turn_id"] = None if turn_id is None else str(turn_id)

    def set_request(self, request: Mapping[str, Any], bounded: bool = False) -> None:
        if self._done:
            return
        self._event["request"] = dict(request) if bounded else self._session.bound(request)

    def sample(
        self,
        name: str,
        data: Optional[Mapping[str, Any]] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> None:
        """Retain a bounded series of low-frequency observations.

        STT partial transcripts are the motivating case: they are worth keeping
        for a latency timeline, but an unbounded list would turn an audio stream
        into an unbounded event stream.
        """
        if self._done or not name:
            return
        samples = self._event.setdefault("samples", {})
        bucket = samples.setdefault(name, {"items": [], "truncated": False})
        if len(bucket["items"]) >= limit:
            bucket["truncated"] = True
            return
        payload: Dict[str, Any] = dict(data or {})
        payload.update(kwargs)
        at = payload.pop("occurred_at_ms", None)
        if at is None:
            at = self._session.now()
        bucket["items"].append({"occurred_at_ms": at, **self._session.bound(payload)})

    def end(
        self,
        status: str = "ok",
        response: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
        ended_at_ms: Optional[int] = None,
        payload_bounded: bool = False,
    ) -> None:
        """Close the span and write it. A second call is ignored."""
        if self._done:
            return
        self._done = True
        event = self._event
        event["ended_at_ms"] = self._session.now() if ended_at_ms is None else ended_at_ms
        event["duration_ms"] = max(0, event["ended_at_ms"] - event["started_at_ms"])
        event["status"] = status
        payload = dict(response or {})
        event["response"] = payload if payload_bounded else self._session.bound(payload)
        event["error"] = dict(error) if error is not None else None
        self._session._write_event(event)


class _InertOperation(Operation):
    """Returned once the session has ended, so integrations need no null checks."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips Operation.__init__
        self._session = None  # type: ignore[assignment]
        self._event = {}
        self._done = True

    @property
    def turn_id(self) -> Optional[str]:
        return None

    def event(self, name, data=None, **kwargs) -> None:  # noqa: D102, ANN001
        return None

    def set_turn(self, turn_id) -> None:  # noqa: D102, ANN001
        return None

    def set_request(self, request, bounded: bool = False) -> None:  # noqa: D102, ANN001
        return None

    def sample(self, name, data=None, limit: int = 100, **kwargs) -> None:  # noqa: D102, ANN001
        return None

    def end(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return None


class Turn:
    """One unit of conversational work. Groups operations by `turn_id`."""

    __slots__ = ("id", "_session", "_ended")

    def __init__(self, session: "Session", turn_id: Optional[Any] = None) -> None:
        self.id = str(turn_id) if turn_id is not None else str(uuid.uuid4())
        self._session = session
        self._ended = False

    @property
    def ended(self) -> bool:
        return self._ended

    def start_operation(self, **kwargs: Any) -> Operation:
        kwargs["turn_id"] = self.id
        return self._session.start_operation(**kwargs)

    def context(self):
        """Scope ambient instrumentation to this turn."""
        return self._session.with_turn(self.id)

    def end(self) -> None:
        self._ended = True
        self._session._turns.discard(self)


class Session:
    """A single recorded call, spooled to its own directory."""

    def __init__(self, observer: "VaaniObserver", **input: Any) -> None:
        self._observer = observer
        self._started = time.monotonic()
        self._ended = False
        self._turns: Set[Turn] = set()
        # Handles for sockets nobody else owns, so `end()` can finalize them.
        self._open_sockets: Set[Any] = set()
        self._ending = False
        self._tracks: Dict[str, Dict[str, Any]] = {}
        self._track_timeline_ends: Dict[str, float] = {}
        self._pending_captures: Set[Awaitable[Any]] = set()
        # A loop-neutral future. `Session` may be constructed on one loop (or
        # none) and awaited on another — a worker restart, a test, a shutdown
        # hook — and an asyncio.Future would be permanently bound to whichever
        # loop happened to touch it first.
        self._completion: "concurrent.futures.Future[FinalizedSession]" = (
            concurrent.futures.Future()
        )
        self._result: Optional[FinalizedSession] = None
        self._lock = threading.Lock()

        self.id = str(input.get("session_id") or uuid.uuid4())
        self.agent_id = input.get("agent_id")
        self.metadata = dict(input.get("metadata") or {})
        self.started_at = _utc_now_iso()
        self.directory = os.path.join(observer.options["spool_directory"], self.id)
        instrumentations = observer.options["instrumentations"]
        self.capture_status: Dict[str, Any] = {
            "events_complete": True,
            "audio_complete": True,
            # `events_complete` and `audio_complete` describe *transport*: they
            # mean "everything we were handed was written". They cannot see a
            # stage the SDK was never told about, so a provider that emitted no
            # metrics produced a package missing 100% of that stage while both
            # flags stayed true. `coverage_complete` describes *observability*:
            # whether what we recorded accounts for what demonstrably happened.
            "coverage_complete": True,
            "http_instrumentation": "active" if instrumentations["http"] else "disabled",
            "websocket_instrumentation": "active" if instrumentations["websocket"] else "disabled",
            "dropped_event_count": 0,
            "dropped_audio_chunk_count": 0,
        }
        self._writer = SpoolWriter(
            self.directory,
            on_error=self._on_write_error,
            on_drop=self._on_write_dropped,
            strict=observer.options["strict"],
        )

    # ------------------------------------------------------------ the clock

    def now(self) -> int:
        """Milliseconds since the session started, on one monotonic clock.

        Every timestamp in the package is relative to this, so a timeline stays
        coherent even when the host's wall clock steps.
        """
        return round((time.monotonic() - self._started) * 1000)

    # ---------------------------------------------------------- the context

    def context(self):
        """`with session.context():` attributes ambient provider calls here."""
        return self._observer.context(self)

    def with_endpoint(self, endpoint_id: str, turn_id: Optional[Any] = None):
        """Force a specific configured endpoint for the work in this scope."""
        if not any(rule["id"] == endpoint_id for rule in self._observer.endpoint_rules):
            raise ValueError(f"Unknown endpoint: {endpoint_id}")
        return self._observer.context(self, endpoint_id=endpoint_id, turn_id=turn_id)

    def with_turn(self, turn_id: Any):
        """Tag auto-instrumented work in this scope with `turn_id`."""
        return self._observer.context(self, turn_id=None if turn_id is None else str(turn_id))

    def bind(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a callback so it runs with this session as its ambient context."""
        session = self

        if asyncio.iscoroutinefunction(handler):

            async def async_bound(*args: Any, **kwargs: Any) -> Any:
                # The scope has to be installed inside the coroutine, but it
                # must still inherit an enclosing endpoint/turn the way the
                # synchronous path does.
                existing = self._observer.current_context()
                inherited = existing if existing and existing.session is session else None
                token = set_context(
                    ObserverContext(
                        session=session,
                        endpoint_id=inherited.endpoint_id if inherited else None,
                        turn_id=inherited.turn_id if inherited else None,
                    )
                )
                try:
                    return await handler(*args, **kwargs)
                finally:
                    reset_context(token)

            return async_bound

        def bound(*args: Any, **kwargs: Any) -> Any:
            with session.context():
                return handler(*args, **kwargs)

        return bound

    def defer_capture(self, awaitable: Awaitable[Any]) -> None:
        """Hold `end()` open for post-response capture work such as body reads."""
        if self._ended:
            return
        task = asyncio.ensure_future(awaitable)
        self._pending_captures.add(task)
        task.add_done_callback(self._pending_captures.discard)

    # ----------------------------------------------------------- structure

    def start_turn(self, turn_id: Optional[Any] = None) -> Turn:
        turn = Turn(self, turn_id)
        self._turns.add(turn)
        return turn

    def start_operation(
        self,
        type: str,  # noqa: A002 - mirrors the Node option name
        turn_id: Optional[Any] = None,
        scope: str = "turn",
        endpoint_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        transport: str = "manual",
        started_at_ms: Optional[int] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> Operation:
        if self._ended:
            return _InertOperation()
        if type not in OPERATION_TYPES:
            raise TypeError("Operation type must be stt, llm, tts, or tool.")
        event = {
            "event_id": str(uuid.uuid4()),
            "session_id": self.id,
            "turn_id": None if turn_id is None else str(turn_id),
            "scope": scope,
            "type": type,
            "endpoint_id": endpoint_id,
            "provider": provider,
            "model": model,
            "transport": transport,
            "started_at_ms": self.now() if started_at_ms is None else started_at_ms,
            "ended_at_ms": None,
            "duration_ms": None,
            "status": "in_progress",
            "milestones": {},
            "request": self.bound(dict(request or {})),
            "response": {},
            "error": None,
        }
        return Operation(self, event)

    # --------------------------------------------------------------- audio

    def record_inbound_audio(
        self, chunk: Any, format: Optional[Mapping[str, Any]] = None  # noqa: A002
    ) -> bool:
        """Record caller audio on the right channel of the finalized recording."""
        return self._record_audio("caller", chunk, format)

    def record_outbound_audio(
        self, chunk: Any, format: Optional[Mapping[str, Any]] = None  # noqa: A002
    ) -> bool:
        """Record agent audio on the left channel of the finalized recording."""
        return self._record_audio("agent", chunk, format)

    def _record_audio(self, track: str, chunk: Any, fmt: Optional[Mapping[str, Any]]) -> bool:
        if self._ended or not self._observer.options["capture"]["audio"]:
            return False
        if not isinstance(chunk, BYTES_TYPES):
            raise TypeError("Audio chunk must be bytes, bytearray or memoryview.")
        fmt = dict(fmt or {})
        normalized = {
            "encoding": fmt.get("encoding"),
            "sample_rate_hz": fmt.get("sample_rate_hz"),
            "channels": fmt.get("channels"),
        }
        if normalized["encoding"] != "pcm_s16le":
            raise ValueError("Audio capture requires pcm_s16le input.")
        if (
            not isinstance(normalized["sample_rate_hz"], int)
            or isinstance(normalized["sample_rate_hz"], bool)
            or normalized["sample_rate_hz"] <= 0
        ):
            raise ValueError("Audio capture requires a valid sample_rate_hz.")
        if (
            not isinstance(normalized["channels"], int)
            or isinstance(normalized["channels"], bool)
            or normalized["channels"] <= 0
        ):
            raise ValueError("Audio capture requires a valid channel count.")
        previous = self._tracks.get(track)
        if previous is not None and previous != normalized:
            raise ValueError(f"{track} audio format cannot change within a session.")
        self._tracks[track] = normalized

        data = bytes(chunk)
        received_at = fmt.get("timestamp_ms")
        if received_at is None:
            received_at = self.now()
        duration_ms = pcm_duration_ms(len(data), normalized)
        # TTS often arrives in a burst even though it will be played in real
        # time. Advancing the agent track's clock by PCM duration keeps the
        # pauses before and between replies instead of compressing them away.
        if track == "agent" and duration_ms is not None:
            occurred_at = max(received_at, self._track_timeline_ends.get(track, 0))
        else:
            occurred_at = received_at
        if duration_ms is not None:
            self._track_timeline_ends[track] = occurred_at + duration_ms

        self._writer.submit(TEMP_TRACK_FILES[track], data)
        event: Dict[str, Any] = {
            "kind": "audio_chunk",
            "track": track,
            "occurred_at_ms": occurred_at,
            "byte_length": len(data),
        }
        if duration_ms is not None:
            event["duration_ms"] = duration_ms
        self._write_event(event)
        return True

    # ------------------------------------------------------------ raw events

    def record_websocket_event(self, **input: Any) -> None:
        """Record neutral socket lifecycle data from any websocket integration."""
        if self._ended:
            return
        occurred_at = input.pop("occurred_at_ms", None)
        self._write_event(
            {
                "kind": "websocket",
                "session_id": self.id,
                "occurred_at_ms": self.now() if occurred_at is None else occurred_at,
                **input,
            }
        )

    def bound(self, value: Any) -> Any:
        return bounded_payload(value, self._observer.options["capture"]["payload_max_bytes"])

    def _write_event(self, event: Mapping[str, Any]) -> None:
        if self._ended:
            return
        try:
            line = json_bytes(event) + b"\n"
        except (TypeError, ValueError):
            line = json_bytes({"kind": "capture_error", "occurred_at_ms": self.now()}) + b"\n"
        self._writer.submit("events.jsonl", line)

    def _on_write_error(self, filename: str, error: BaseException) -> None:
        """Degrade capture_status for the artefact that actually failed.

        Called on the writer thread. The session is finalized only after that
        thread has been joined, so the manifest always sees the final counts.
        """
        self._degrade(filename)

    def _on_write_dropped(self, filename: str) -> None:
        """A bounded-queue drop. Same accounting as a failed write."""
        self._degrade(filename)

    def degrade_audio(self) -> None:
        """Record that a caller-supplied audio chunk was not captured.

        Integrations tap frames off the media path and must swallow their own
        failures — a stalled recorder is worse than a lossy one. But the
        manifest then has to say so, or the package reports `audio_complete`
        while silently missing audio, and every duration and alignment derived
        from it is quietly wrong.

        Reachable from two threads -- an integration's event loop and the
        writer -- so the counter takes the lock every other mutation of shared
        state in this class takes. An unlocked `+= 1` under-reports, and the
        whole point of the counter is to be believed.
        """
        with self._lock:
            self.capture_status["audio_complete"] = False
            self.capture_status["dropped_audio_chunk_count"] += 1

    def report_coverage_gap(self, stage: str, reason: str, **facts: Any) -> None:
        """Record that something observably happened which we did not capture.

        This is the difference between "we wrote everything we were given" and
        "what we wrote accounts for the call". An integration that can *prove*
        a stage ran -- because it taped the audio that stage produced -- and
        finds no span for it must say so here, so the gap travels with the
        package as a fact instead of being invisible.

        Deliberately additive and never self-clearing: a gap once observed is
        part of the call's permanent record.
        """
        with self._lock:
            self.capture_status["coverage_complete"] = False
            gaps = self.capture_status.setdefault("coverage_gaps", [])
            gaps.append({"stage": stage, "reason": reason, **facts})

    def report_capture_measurement(self, **facts: Any) -> None:
        """Record what the recorder itself measured about the call.

        Distinct from a coverage gap, and the reason zero spans is not one
        answer but two. An empty call can mean the capture failed or it can mean
        the agent never spoke -- the single most consequential voice-agent
        failure there is -- and only the recorder's own audio tap can tell those
        apart. Publishing the measurement lets the console name the real story
        instead of pointing an operator at the SDK.

        Never affects `*_complete`: this is evidence about the call, not a
        verdict about the capture.
        """
        with self._lock:
            measured = self.capture_status.setdefault("measured", {})
            measured.update(facts)

    def _degrade(self, filename: str) -> None:
        if filename in TEMP_TRACK_FILES.values():
            self.degrade_audio()
            return
        with self._lock:
            self.capture_status["events_complete"] = False
            self.capture_status["dropped_event_count"] += 1

    # ---------------------------------------------------------- finalization

    @property
    def finished(self) -> "asyncio.Future[FinalizedSession]":
        """Resolves with the finalized package. Awaitable from any loop, twice."""
        return asyncio.wrap_future(self._completion)

    @property
    def ended(self) -> bool:
        return self._ended

    async def ready(self) -> None:
        """Wait until the spool directory exists. Rarely needed outside tests."""
        await asyncio.to_thread(self._writer.wait_ready)

    async def end(self, outcome: str = "unknown") -> FinalizedSession:
        """Close every open span, flush the spool and publish the manifest."""
        if self._ended:
            return await self.finished

        if self._pending_captures:
            await asyncio.gather(*list(self._pending_captures), return_exceptions=True)
        # A socket the app never handed us -- one the auto-instrumentation
        # attached to -- has nobody else to close it. Its span has to be
        # finalized here, while the writer is still open, or the whole
        # connection record is silently dropped.
        with self._lock:
            # Instrumentation runs on whatever thread opened the socket, so
            # registration has to be closed atomically: otherwise a handle
            # added between here and `_ended` would never be finalized.
            self._ending = True
            sockets = list(self._open_sockets)
            self._open_sockets.clear()
        for handle in sockets:
            try:
                # A socket still open when the call ends normally was closed by
                # teardown, not cancelled mid-flight.
                handle.detach(status="ok" if outcome == "completed" else "cancelled")
            except Exception as error:  # noqa: BLE001 - finalization is best effort
                warn_once("socket-finalize",
                          "vaani: socket finalization failed (%s)", error)
        with self._lock:
            if self._ended:
                return await self.finished
            # Set before the writer is closed: every record() path checks this
            # first, so nothing can be accepted once finalization has begun.
            self._ended = True
        for turn in list(self._turns):
            turn.end()

        try:
            write_error = await asyncio.to_thread(self._writer.close)
            if self._writer.ready_error is not None:
                raise self._writer.ready_error
            if self._observer.options["strict"] and write_error is not None:
                raise write_error
            duration_ms = self.now()
            audio = await asyncio.to_thread(
                compose_stereo, self.directory, self._tracks, duration_ms
            )
            manifest = self._manifest(outcome, duration_ms, audio)
            await asyncio.to_thread(_publish_manifest, self.directory, manifest)
            self._result = FinalizedSession(
                session_id=self.id, directory=self.directory, manifest=manifest
            )
            self._resolve(self._result)
        except BaseException as error:  # noqa: BLE001 - reported through `finished`
            self._reject(error)
        return await self.finished

    def register_socket(self, handle: Any) -> bool:
        """Take ownership of a socket span, unless finalization already began.

        Returns False when the caller must finalize the handle itself, so a
        socket opened while the call was ending still produces a span rather
        than being silently dropped.
        """
        with self._lock:
            if self._ending or self._ended:
                return False
            self._open_sockets.add(handle)
            return True

    def forget_socket(self, handle: Any) -> None:
        with self._lock:
            self._open_sockets.discard(handle)

    def _resolve(self, value: FinalizedSession) -> None:
        if not self._completion.done():
            self._completion.set_result(value)

    def _reject(self, error: BaseException) -> None:
        if not self._completion.done():
            self._completion.set_exception(error)

    def _manifest(
        self,
        outcome: str,
        duration_ms: int,
        audio: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "sdk": dict(SDK),
            "session_id": self.id,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "duration_ms": duration_ms,
            "outcome": outcome,
            "capture_status": dict(self.capture_status),
            "audio": {"call": dict(audio)} if audio is not None else {},
        }


def _publish_manifest(directory: str, manifest: Mapping[str, Any]) -> None:
    """Write then rename, so a reader never sees a half-written manifest."""
    temporary = os.path.join(directory, "manifest.json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, os.path.join(directory, "manifest.json"))


def _utc_now_iso() -> str:
    import datetime as _dt

    return (
        _dt.datetime.now(_dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
