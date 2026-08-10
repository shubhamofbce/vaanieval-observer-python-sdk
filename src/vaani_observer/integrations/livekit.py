"""Vaani observability for a LiveKit Agents `AgentSession`.

This is the Python counterpart of the Node agent's `CallRecorder`, expressed
against LiveKit's own event surface instead of a bespoke orchestrator.

Two span scopes are kept deliberately separate, exactly as in the Node
integration:

* ``connection`` — one span per provider socket, for transport health. A
  streaming STT socket stays open for the whole call, so that span is
  call-length by definition and must never be read as per-turn latency.
* ``turn`` — one STT, one or more LLM, one TTS and N tool operations per
  conversational turn, all carrying the same turn id. This is the unit the
  dashboard charts.

Turn identity comes from LiveKit's ``speech_id`` (``SpeechHandle.id``), which is
stamped onto LLM, TTS and EOU metrics. A user utterance is recorded before any
speech handle exists, so the utterance opens a turn and the first agent speech
that follows adopts it. That is the same "the orchestrator owns turn
boundaries" rule the Node recorder follows -- nothing here re-implements
endpointing.

Nothing in this module is on the media path except the two audio taps, which do
a single bounded queue append per frame.

Known limits of what LiveKit exposes:

* ``connection`` spans are produced by the SDK's ambient websocket
  instrumentation, which can only attach to a socket opened while the session
  context is active. A plugin that pools connections (Sarvam's TTS prewarms
  one) may therefore synthesize over a socket opened before recording began, so
  its connection span can show zero bytes even though its turn spans are
  complete. Turn-level TTS timing is unaffected: it comes from metrics, not from
  the socket.
* ``user_input_transcribed`` carries neither word timings nor confidence, so
  ``words_recorded`` and ``confidence_recorded`` are always false. That is a
  platform limit, not a gap in this port.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vaani_observer.livekit")

# Cleanup of a provider iterator is best-effort: a stalled provider must never
# be the reason a call fails to shut down.
_CLOSE_TIMEOUT_S = 5.0

__all__ = [
    "VaaniLiveKitRecorder",
    "observe_agent_session",
    "VaaniAudioTapMixin",
    "STT_ENDPOINT_ID",
    "LLM_ENDPOINT_ID",
    "TTS_ENDPOINT_ID",
]

STT_ENDPOINT_ID = "stt"
LLM_ENDPOINT_ID = "llm"
TTS_ENDPOINT_ID = "tts"

#: Partial transcripts are worth keeping for a latency timeline, but an audio
#: stream would otherwise become an unbounded event stream.
_PARTIAL_SAMPLE_LIMIT = 100

# LiveKit tags every session error with the component that raised it. Without
# this routing an LLM timeout would close the STT and TTS spans of the same
# turn, reporting a healthy transcription as a failure.
_ERROR_TARGETS: Dict[str, tuple] = {
    "stt_error": ("stt",),
    "llm_error": ("llm",),
    "tts_error": ("tts",),
    # A realtime model *is* the STT, LLM and TTS stage at once.
    "realtime_model_error": ("stt", "llm", "tts"),
}
_ALL_ERROR_TARGETS = ("stt", "llm", "tts")


def _present(**fields: Any) -> Dict[str, Any]:
    """Drop absent fields.

    `JSON.stringify` omits `undefined`, so the Node package never carries a null
    milestone field. `json.dumps` writes `null`, which the dashboard renders as
    a measured zero-ish value rather than "not reported".
    """
    return {key: value for key, value in fields.items() if value is not None}


def _ms(seconds: Any) -> Optional[int]:
    """LiveKit reports seconds as floats; the package timeline is integer ms."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return round(value * 1000)


def _positive_ms(seconds: Any) -> int:
    value = _ms(seconds)
    return value if value and value > 0 else 0


def _back_dated(now: int, duration_ms: int) -> tuple:
    """Place a span of known duration ending now, without a negative start.

    LiveKit reports a stage's duration after the fact, so the span has to be
    back-dated. Early in a call `now` can be smaller than the duration -- the
    greeting's TTS starts before the session clock has advanced that far -- and
    naively clamping the start alone would silently shrink the measured
    duration to nothing.
    """
    started = max(0, now - duration_ms)
    return started, started + duration_ms


class _TurnState:
    """Everything open for one conversational turn."""

    __slots__ = ("id", "turn", "stt", "stt_ended_at", "stt_response", "llm", "tts",
                 "tts_response", "tts_ended_at", "tools", "audio_bytes",
                 "audio_first_at_ms", "finished")

    def __init__(self, turn_id: str, turn: Any) -> None:
        self.id = turn_id
        self.turn = turn
        # STT and TTS spans stay open until the turn closes, so LiveKit's
        # `ChatMessage.metrics` -- which arrives last and is the most accurate
        # per-turn latency source there is -- can still be folded into them.
        # Their real end timestamps are recorded here and passed to `end()`, so
        # holding the object open never inflates a measured duration.
        self.stt: Any = None
        self.stt_ended_at: Optional[int] = None
        self.stt_response: Dict[str, Any] = {}
        self.llm: List[Any] = []
        self.tts: Any = None
        self.tts_response: Dict[str, Any] = {}
        self.tts_ended_at: Optional[int] = None
        self.tools: Dict[str, Any] = {}
        self.audio_bytes = 0
        # When the caller first heard this reply. Recorded on the turn rather
        # than the TTS span because the frames arrive before the span exists.
        self.audio_first_at_ms: Optional[int] = None
        self.finished = False


class _PendingStt:
    """The provider-neutral STT payload, captured before a turn id exists."""

    __slots__ = ("started_at_ms", "first_partial_at_ms", "ended_at_ms", "partials", "truncated",
                 "language", "metrics")

    def __init__(self) -> None:
        self.started_at_ms: Optional[int] = None
        self.first_partial_at_ms: Optional[int] = None
        self.ended_at_ms: Optional[int] = None
        self.partials: List[Dict[str, Any]] = []
        self.truncated = False
        self.language: Optional[str] = None
        self.metrics: Dict[str, Any] = {}


class VaaniLiveKitRecorder:
    """Turns one `AgentSession` into a Vaani recording.

    The recorder is inert when no observer is configured, so an agent can be
    written against it unconditionally and observability stays a deployment
    decision rather than a code path.
    """

    def __init__(
        self,
        observer: Any = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        capture_transcripts: bool = True,
        upload: bool = False,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        self._observer = observer
        self._upload = upload
        self._capture_transcripts = capture_transcripts
        self._input_format = {
            "encoding": "pcm_s16le",
            "sample_rate_hz": int(input_sample_rate),
            "channels": int(channels),
        }
        self._output_format = {
            "encoding": "pcm_s16le",
            "sample_rate_hz": int(output_sample_rate),
            "channels": int(channels),
        }
        self.call = None
        if observer is not None:
            try:
                self.call = observer.start_session(agent_id=agent_id, metadata=metadata or {})
            except Exception as error:  # noqa: BLE001 - never block the call
                logger.error("vaani: start_session failed (%s); recording disabled", error)

        # `_turns` is an index, not the population: one state is registered
        # under both its own id and the LiveKit speech id that adopted it.
        # `_all_turns` is the deduplicated list to iterate and count.
        self._turns: Dict[str, _TurnState] = {}
        self._all_turns: List[_TurnState] = []
        self._pending_turn: Optional[_TurnState] = None
        self._current_turn: Optional[_TurnState] = None
        self._pending_stt = _PendingStt()
        # STT metrics are emitted per provider request, not per utterance, so a
        # turn whose metric arrived during an earlier utterance would otherwise
        # be written with no provider or model -- which the dashboard reports as
        # "not recorded by SDK" and refuses to price.
        self._stt_identity: Dict[str, Any] = {}
        self._outcome: Optional[str] = None
        self._turn_counter = 0
        self._sockets: List[Any] = []
        self._attached: List[Any] = []

    # --------------------------------------------------------------- factory

    @classmethod
    def from_env(cls, **overrides: Any) -> "VaaniLiveKitRecorder":
        """Build from `VAANI_*` environment variables, mirroring the Node agent.

        Returns an inert recorder when `VAANI_ENABLED` is off or the observer
        cannot be configured: a misconfigured recorder must never be the reason
        a call fails to start.
        """
        if not _env_bool("VAANI_ENABLED", False):
            return cls(None, **overrides)
        try:
            from .. import VaaniObserver

            endpoints = overrides.pop("endpoints", None) or []
            observer = VaaniObserver(
                endpoint=os.environ.get("VAANI_ENDPOINT", "http://localhost:8000"),
                api_key=os.environ.get("VAANI_API_KEY", "local-dev"),
                spool_directory=os.path.abspath(
                    os.environ.get("VAANI_SPOOL_DIR", ".vaani-spool")
                ),
                capture={
                    "audio": _env_bool("VAANI_CAPTURE_AUDIO", True),
                    "http_bodies": _env_bool("VAANI_CAPTURE_HTTP_BODIES", True),
                    "stt_content": _env_bool("VAANI_CAPTURE_STT_CONTENT", True),
                    "payload_max_bytes": _env_int("VAANI_PAYLOAD_MAX_BYTES", 16 * 1024),
                },
                endpoints=endpoints,
            )
        except Exception as error:  # noqa: BLE001 - observability is optional
            logger.error("vaani: disabled, failed to configure observer — %s", error)
            return cls(None, **overrides)
        overrides.setdefault("agent_id", os.environ.get("VAANI_AGENT_ID", "livekit-agent"))
        overrides.setdefault("capture_transcripts", _env_bool("VAANI_CAPTURE_STT_CONTENT", True))
        overrides.setdefault("upload", _env_bool("VAANI_UPLOAD", True))
        return cls(observer, **overrides)

    # -------------------------------------------------------------- lifecycle

    @property
    def enabled(self) -> bool:
        return self.call is not None

    def attach(self, session: Any) -> "VaaniLiveKitRecorder":
        """Subscribe to an `AgentSession`. Safe to call on an inert recorder."""
        if self.call is None:
            return self
        handlers = {
            "user_state_changed": self._on_user_state,
            "user_input_transcribed": self._on_transcript,
            "conversation_item_added": self._on_conversation_item,
            "function_tools_executed": self._on_tools_executed,
            # Deprecated in LiveKit 1.6 in favour of `session_usage_updated`
            # (usage) plus `ChatMessage.metrics` (latency), but it is still the
            # only source of per-stage duration, TTFT/TTFB and token counts.
            # Both are subscribed to, so the spans survive its removal with
            # only the token counts degrading.
            "metrics_collected": self._on_metrics,
            "session_usage_updated": self._on_usage,
            "speech_created": self._on_speech_created,
            "error": self._on_error,
            "close": self._on_close,
        }
        for name, handler in handlers.items():
            try:
                session.on(name, _guard(handler))
            except Exception as error:  # noqa: BLE001 - version drift is survivable
                logger.debug("vaani: cannot subscribe to %r (%s)", name, error)
        self._attached.append(session)
        return self

    async def finish(self, outcome: Optional[str] = None) -> None:
        """Finalize the local package and, when configured, upload it."""
        call = self.call
        if call is None:
            return
        # The close event knows why the call really ended; an explicit outcome
        # from the caller still wins, and "completed" is only the default when
        # nothing observed a reason at all.
        outcome = outcome or self._outcome or "completed"
        self.call = None
        try:
            self.finalize_open_spans(call, outcome=outcome)
            finalized = await call.end(outcome=outcome)
            logger.info(
                "vaani: recorded %s (%sms, %d turns) → %s",
                finalized.session_id,
                finalized.manifest["duration_ms"],
                len(self._all_turns),
                finalized.directory,
            )
            if self._upload and self._observer is not None:
                result = await self._observer.upload_package(finalized)
                logger.info(
                    "vaani: uploaded %s status=%s operations=%s",
                    result.get("session_id"),
                    result.get("status"),
                    result.get("operation_count"),
                )
        except Exception as error:  # noqa: BLE001 - a failed upload is not a failed call
            logger.warning("vaani: finish/upload failed — %s", error)

    def finalize_open_spans(self, call: Any = None, outcome: Optional[str] = None) -> None:
        """Close anything still open so a dropped call cannot leak a span."""
        for state in self._all_turns:
            self._end_stt(state, "ok" if state.stt_ended_at is not None else "cancelled")
            for operation in state.llm:
                if not operation.ended:
                    operation.end(status="cancelled")
            self._end_tts(state, "cancelled")
            for operation in state.tools.values():
                if not operation.ended:
                    operation.end(status="cancelled")
            state.tools.clear()
            if not state.finished:
                state.finished = True
                state.turn.end()
        # A provider socket that survived to the end of a completed call was
        # closed by teardown; only an abnormal ending really cancelled it.
        socket_status = "ok" if outcome == "completed" else "cancelled"
        for handle in self._sockets:
            try:
                handle.detach(status=socket_status)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        self._sockets.clear()

    def observe_socket(self, socket: Any, url: Optional[str] = None,
                       endpoint_id: Optional[str] = None) -> Any:
        """Record lifecycle and byte accounting for one provider socket."""
        if self.call is None or self._observer is None or socket is None:
            return None
        try:
            handle = self._observer.observe_websocket(
                socket, session=self.call, url=url, endpoint_id=endpoint_id
            )
        except Exception as error:  # noqa: BLE001 - transport health is optional
            logger.debug("vaani: observe_websocket failed (%s)", error)
            return None
        self._sockets.append(handle)
        return handle

    @property
    def _current_turn_id_for_test(self) -> Optional[str]:
        return self._current_turn.id if self._current_turn else None

    def turn_context(self):
        """Scope ambient provider calls to the turn currently being served.

        Without this the Azure/Deepgram/Sarvam requests the plugins issue are
        captured on a task whose context knows the session but not the turn, and
        every automatically instrumented span lands with `turn_id: null` -- so
        the dashboard cannot chart LLM latency per turn.
        """
        import contextlib

        state = self._current_turn
        if self.call is None or state is None:
            return contextlib.nullcontext()
        return self.call.with_turn(state.id)

    # ------------------------------------------------------------ audio taps

    def tap_input_frame(self, frame: Any) -> None:
        """Record one caller PCM frame. Called from `Agent.stt_node`."""
        self._tap(frame, inbound=True)

    def tap_output_frame(self, frame: Any) -> None:
        """Record one agent PCM frame. Called from `Agent.tts_node`."""
        self._tap(frame, inbound=False)
        state = self._current_turn
        if state is None or self.call is None:
            return
        count = _frame_bytes(frame) or 0
        if not count:
            return
        state.audio_bytes += count
        # The frames of a reply are synthesized *before* LiveKit reports the TTS
        # metrics that open the span, so attaching the milestone only when the
        # span already exists dropped it on every real turn -- and with it the
        # dashboard's headline "time to first audio". The first frame's time is
        # therefore held on the turn and stamped onto the span whenever it
        # appears, here or in `_record_tts`.
        if state.audio_first_at_ms is None:
            state.audio_first_at_ms = self.call.now()
        self._mark_first_audio(state)

    def _mark_first_audio(self, state: "_TurnState") -> None:
        """Stamp first-audio timing onto the turn's TTS span, once it exists.

        Milestones merge rather than overwrite, so the first `occurred_at_ms`
        survives every later frame while `total_byte_count` keeps climbing.
        """
        operation = state.tts
        if operation is None or operation.ended or state.audio_first_at_ms is None:
            return
        operation.event(
            "audio_chunk",
            occurred_at_ms=state.audio_first_at_ms,
            total_byte_count=state.audio_bytes,
        )

    def _tap(self, frame: Any, inbound: bool) -> None:
        call = self.call
        if call is None:
            return
        data = _frame_data(frame)
        if data is None:
            return
        # An `rtc.AudioFrame` carries its own rate; trusting it rather than the
        # configured default is what keeps the dashboard's duration maths right
        # when a plugin resamples.
        fmt = dict(self._input_format if inbound else self._output_format)
        rate = getattr(frame, "sample_rate", None)
        channels = getattr(frame, "num_channels", None)
        if isinstance(rate, int) and rate > 0:
            fmt["sample_rate_hz"] = rate
        if isinstance(channels, int) and channels > 0:
            fmt["channels"] = channels
        try:
            if inbound:
                call.record_inbound_audio(data, fmt)
            else:
                call.record_outbound_audio(data, fmt)
        except Exception as error:  # noqa: BLE001 - audio loss beats audio stall
            logger.debug("vaani: audio frame dropped (%s)", error)

    # ---------------------------------------------------------------- events

    def _on_user_state(self, event: Any) -> None:
        state = getattr(event, "new_state", None)
        if state == "speaking":
            if self._pending_stt.started_at_ms is None:
                self._pending_stt.started_at_ms = self.call.now()
        elif state in ("listening", "away") and self._pending_stt.started_at_ms is not None:
            self._pending_stt.ended_at_ms = self.call.now()

    def _on_transcript(self, event: Any) -> None:
        text = (getattr(event, "transcript", None) or "").strip()
        if not text:
            return
        at = self.call.now()
        pending = self._pending_stt
        if pending.started_at_ms is None:
            # Some providers never report a VAD start; the first partial is the
            # earliest defensible start for the STT span.
            pending.started_at_ms = at
        pending.language = getattr(event, "language", None) or pending.language
        if not getattr(event, "is_final", False):
            if pending.first_partial_at_ms is None:
                pending.first_partial_at_ms = at
            previous = pending.partials[-1]["transcript"] if pending.partials else None
            if previous != text:
                if len(pending.partials) < _PARTIAL_SAMPLE_LIMIT:
                    pending.partials.append({"occurred_at_ms": at, "transcript": text})
                else:
                    pending.truncated = True
            return
        self._close_user_turn(text, at)

    def _close_user_turn(self, text: str, at: int, reason: str = "endpoint") -> None:
        """A final transcript ends the user's half of the turn and opens ours."""
        pending = self._pending_stt
        pending.metrics = {**self._stt_identity, **pending.metrics}
        self._pending_stt = _PendingStt()
        state = self._new_turn()
        self._pending_turn = state
        self._current_turn = state
        started_at_ms = pending.started_at_ms if pending.started_at_ms is not None else at
        operation = state.turn.start_operation(
            type="stt",
            endpoint_id=STT_ENDPOINT_ID,
            # The dashboard prices and groups by provider/model, and reports
            # "not recorded by SDK" when they are missing, so they are lifted
            # off the STT metric rather than left in the response payload.
            provider=pending.metrics.get("provider"),
            model=pending.metrics.get("model"),
            transport="websocket" if pending.metrics.get("streamed") else "http",
            started_at_ms=started_at_ms,
            request={
                "sample_rate_hz": self._input_format["sample_rate_hz"],
                "language": pending.language,
                "model": pending.metrics.get("model"),
                "streamed": pending.metrics.get("streamed"),
            },
        )
        operation.event("speech_started", occurred_at_ms=started_at_ms)
        if pending.first_partial_at_ms is not None:
            operation.event("first_partial", occurred_at_ms=pending.first_partial_at_ms)
        operation.event(
            "speech_ended",
            occurred_at_ms=pending.ended_at_ms if pending.ended_at_ms is not None else at,
        )
        # LiveKit does not report *why* a transcript went final, but the
        # dashboard groups finalization latency by reason, so the turn detector
        # that actually ended the turn is named rather than left null.
        operation.event("final_transcript", occurred_at_ms=at, final_reason=reason)
        operation.event("speech_final", occurred_at_ms=at)
        if self._capture_transcripts:
            for sample in pending.partials:
                operation.sample("partial", sample, limit=_PARTIAL_SAMPLE_LIMIT)
            if pending.truncated:
                operation.event("partial_samples_truncated", limit=_PARTIAL_SAMPLE_LIMIT)
        response: Dict[str, Any] = (
            {"transcript": text, "language": pending.language, "final_reason": reason}
            if self._capture_transcripts
            else {"char_count": len(text), "final_reason": reason}
        )
        response["audio_ms"] = pending.metrics.get("audio_ms")
        state.stt = operation
        state.stt_response = response
        state.stt_ended_at = at

    def _on_speech_created(self, event: Any) -> None:
        """Bind a LiveKit speech handle to the turn it is replying to."""
        speech_id = _speech_id(getattr(event, "speech_handle", None))
        if speech_id is None:
            return
        if speech_id in self._turns:
            self._current_turn = self._turns[speech_id]
            return
        state = self._pending_turn
        if state is None:
            # `say()` and the opening greeting produce a turn with no user
            # speech at all.
            state = self._new_turn()
        self._pending_turn = None
        self._turns[speech_id] = state
        self._current_turn = state

    def _on_usage(self, event: Any) -> None:
        """Cumulative session usage, kept on the manifest rather than a span.

        It is a running total, not per-turn work, so recording it as an
        operation would double-count every token already on an LLM span.
        """
        usage = getattr(event, "usage", None)
        if usage is None or self.call is None:
            return
        payload = _plain(usage)
        if payload is None:
            logger.debug("vaani: session usage not serialisable (%r)", type(usage))
            return
        self.call.metadata["usage"] = payload

    def _on_metrics(self, event: Any) -> None:
        metrics = getattr(event, "metrics", None)
        kind = getattr(metrics, "type", None)
        if kind == "llm_metrics":
            self._record_llm(metrics)
        elif kind == "tts_metrics":
            self._record_tts(metrics)
        elif kind == "stt_metrics":
            self._record_stt_metrics(metrics)
        elif kind == "eou_metrics":
            self._record_eou(metrics)

    def _record_llm(self, metrics: Any) -> None:
        state = self._state_for(getattr(metrics, "speech_id", None))
        if state is None:
            return
        duration = _positive_ms(getattr(metrics, "duration", 0))
        started_at, ended_at = _back_dated(self.call.now(), duration)
        operation = state.turn.start_operation(
            type="llm",
            endpoint_id=LLM_ENDPOINT_ID,
            provider=_provider(metrics),
            model=_model(metrics),
            transport="http",
            started_at_ms=started_at,
            request={"request_id": getattr(metrics, "request_id", None)},
        )
        ttft = _ms(getattr(metrics, "ttft", None))
        if ttft is not None and ttft >= 0:
            operation.event("first_token", occurred_at_ms=started_at + ttft)
        operation.end(
            status="cancelled" if getattr(metrics, "cancelled", False) else "ok",
            response=_present(
                prompt_tokens=getattr(metrics, "prompt_tokens", None),
                prompt_cached_tokens=getattr(metrics, "prompt_cached_tokens", None),
                completion_tokens=getattr(metrics, "completion_tokens", None),
                total_tokens=getattr(metrics, "total_tokens", None),
                tokens_per_second=getattr(metrics, "tokens_per_second", None),
                ttft_ms=ttft,
            ),
            ended_at_ms=ended_at,
        )
        state.llm.append(operation)

    def _record_tts(self, metrics: Any) -> None:
        state = self._state_for(getattr(metrics, "speech_id", None))
        if state is None:
            return
        duration = _positive_ms(getattr(metrics, "duration", 0))
        started_at, ended_at = _back_dated(self.call.now(), duration)
        operation = state.tts
        if operation is None or operation.ended:
            operation = state.turn.start_operation(
                type="tts",
                endpoint_id=TTS_ENDPOINT_ID,
                provider=_provider(metrics),
                model=_model(metrics),
                transport="websocket" if getattr(metrics, "streamed", False) else "http",
                started_at_ms=started_at,
                request={"sample_rate_hz": self._output_format["sample_rate_hz"]},
            )
            state.tts = operation
        operation.event("speak", _present(char_count=getattr(metrics, "characters_count", None)))
        # Frames already synthesized for this reply were counted before this
        # span existed; stamp their timing on now that there is somewhere to put it.
        self._mark_first_audio(state)
        ttfb = _ms(getattr(metrics, "ttfb", None))
        if ttfb is not None and ttfb >= 0:
            operation.event("first_byte", occurred_at_ms=started_at + ttfb)
        state.tts_ended_at = ended_at
        state.tts_response.update(
            _present(
                audio_ms=_ms(getattr(metrics, "audio_duration", None)),
                characters_count=getattr(metrics, "characters_count", None),
                ttfb_ms=ttfb,
                segment_id=getattr(metrics, "segment_id", None),
            )
        )
        state.tts_response["cancelled"] = bool(getattr(metrics, "cancelled", False))

    def _record_stt_metrics(self, metrics: Any) -> None:
        """STT metrics arrive without a speech id, so they decorate the pending span."""
        self._stt_identity = _present(
            provider=_provider(metrics),
            model=_model(metrics),
            streamed=getattr(metrics, "streamed", None),
        )
        self._pending_stt.metrics = {
            "audio_ms": _ms(getattr(metrics, "audio_duration", None)),
            **self._stt_identity,
        }

    def _record_eou(self, metrics: Any) -> None:
        state = self._state_for(getattr(metrics, "speech_id", None))
        target = state.stt if state is not None else None
        if target is None or target.ended:
            return
        target.event(
            "end_of_utterance",
            _present(
                end_of_utterance_delay_ms=_ms(getattr(metrics, "end_of_utterance_delay", None)),
                transcription_delay_ms=_ms(getattr(metrics, "transcription_delay", None)),
            ),
        )

    def _on_conversation_item(self, event: Any) -> None:
        """LiveKit's own per-turn latency report is the best source there is.

        It arrives after the stage metrics, so it is folded into the spans the
        dashboard already charts rather than recorded as a synthetic operation:
        the package format only knows stt, llm, tts and tool.
        """
        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        state = self._current_turn
        if state is None:
            return
        metrics = getattr(item, "metrics", None) or {}
        text = (getattr(item, "text_content", None) or "").strip()
        if role == "user":
            if state.stt is not None and not state.stt.ended:
                state.stt.event(
                    "turn_report",
                    _present(
                        transcription_delay_ms=_ms(metrics.get("transcription_delay")),
                        end_of_turn_delay_ms=_ms(metrics.get("end_of_turn_delay")),
                        on_user_turn_completed_delay_ms=_ms(
                            metrics.get("on_user_turn_completed_delay")
                        ),
                    ),
                )
            return
        if role != "assistant":
            return
        if state.tts is not None and not state.tts.ended:
            state.tts.event(
                "turn_report",
                _present(
                    e2e_latency_ms=_ms(metrics.get("e2e_latency")),
                    playback_latency_ms=_ms(metrics.get("playback_latency")),
                    tts_ttfb_ms=_ms(metrics.get("tts_node_ttfb")),
                    # `llm_node_ttfs` measures the LLM -> TTS handoff, and the
                    # LLM span is already closed by the time this report
                    # arrives, so it goes on the span it actually gates.
                    llm_ttfs_ms=_ms(metrics.get("llm_node_ttfs")),
                    llm_ttft_ms=_ms(metrics.get("llm_node_ttft")),
                    llm_tokens_per_second=metrics.get("llm_node_tps"),
                ),
            )
            if self._capture_transcripts and text:
                state.tts_response["text"] = text
            else:
                state.tts_response["char_count"] = len(text)
        self._finish_turn(state)

    def _on_tools_executed(self, event: Any) -> None:
        state = self._current_turn
        if state is None:
            return
        try:
            pairs = event.zipped()
        except Exception:  # noqa: BLE001 - version drift is survivable
            pairs = list(
                zip(
                    getattr(event, "function_calls", []) or [],
                    getattr(event, "function_call_outputs", []) or [],
                )
            )
        for call, output in pairs:
            operation = state.turn.start_operation(
                type="tool",
                transport="internal",
                request={
                    "name": getattr(call, "name", None),
                    "input": getattr(call, "arguments", None),
                },
            )
            is_error = bool(getattr(output, "is_error", False)) if output is not None else False
            operation.end(
                status="error" if is_error else "ok",
                response={"result": getattr(output, "output", None) if output else None},
            )

    # LiveKit's own vocabulary for why a session closed. Reporting every call
    # as "completed" would make the call-level success rate -- the headline
    # reliability number -- permanently 100%, however badly the call went.
    _OUTCOMES = {
        "error": "failed",
        "job_shutdown": "abandoned",
        "participant_disconnected": "completed",
        "user_initiated": "completed",
        "task_completed": "completed",
    }

    def _on_close(self, event: Any) -> None:
        call = self.call
        if call is None:
            # Finalization won the shutdown race; there is nothing left to
            # annotate and the outcome is already written.
            return
        reason = getattr(event, "reason", None)
        reason = getattr(reason, "value", reason)
        # A reason this version does not know about must not be optimistically
        # called a success: a future LiveKit failure mode would then be counted
        # as a healthy call. The raw reason is kept so it can be classified.
        self._outcome = self._OUTCOMES.get(str(reason), "unknown")
        call.metadata["close_reason"] = str(reason)
        error = getattr(event, "error", None)
        if error is not None:
            self._outcome = "failed"
            call.metadata["close_error"] = str(
                getattr(error, "message", None) or type(error).__name__
            )

    def fail(self, error: BaseException) -> None:
        """Record that the call could not be run at all."""
        self._outcome = "failed"
        if self.call is not None:
            self.call.metadata["failure"] = f"{type(error).__name__}: {error}"

    def _on_error(self, event: Any) -> None:
        """A session error closes the spans of the component that actually failed.

        The package format has no "error" operation type, so the failure is
        attributed to the spans it interrupted instead of inventing one the
        dashboard cannot chart. LiveKit tags the error with the stage that
        raised it (``llm_error``, ``stt_error``, ``tts_error``), so only that
        stage is failed: an LLM timeout must not mark the turn's completed
        transcription as a failed STT, which would corrupt every STT quality
        and reliability metric downstream.
        """
        state = self._current_turn
        if state is None:
            return
        error = getattr(event, "error", event)
        message = str(error)
        kinds = _error_targets(event, error)
        targets: List[Any] = []
        if "stt" in kinds:
            targets.append(state.stt)
        if "tts" in kinds:
            targets.append(state.tts)
        if "llm" in kinds:
            targets.extend(state.llm)
        payload = _present(
            message=message,
            recoverable=getattr(error, "recoverable", None),
        )
        for operation in targets:
            if operation is not None and not operation.ended:
                operation.end(status="error", error=payload)
        logger.warning("vaani: livekit session error on %s — %s", state.id, message)

    # ------------------------------------------------------------- internals

    def _new_turn(self) -> _TurnState:
        self._turn_counter += 1
        turn_id = f"turn-{self._turn_counter}"
        state = _TurnState(turn_id, self.call.start_turn(turn_id))
        self._turns[turn_id] = state
        self._all_turns.append(state)
        return state

    def _state_for(self, speech_id: Optional[str]) -> Optional[_TurnState]:
        """Resolve the turn a metric belongs to, adopting the pending one once."""
        if self.call is None:
            return None
        if speech_id is None:
            return self._current_turn
        existing = self._turns.get(speech_id)
        if existing is not None:
            self._current_turn = existing
            return existing
        state = self._pending_turn or self._new_turn()
        self._pending_turn = None
        self._turns[speech_id] = state
        self._current_turn = state
        return state

    def _end_stt(self, state: _TurnState, status: str) -> None:
        if state.stt is None or state.stt.ended:
            return
        state.stt.end(
            status=status,
            response=state.stt_response,
            ended_at_ms=state.stt_ended_at,
        )

    def _end_tts(self, state: _TurnState, status: str) -> None:
        if state.tts is None or state.tts.ended:
            return
        rate = self._output_format["sample_rate_hz"] or 1
        response = dict(state.tts_response)
        response.setdefault("audio_bytes", state.audio_bytes)
        response.setdefault("audio_ms", round((state.audio_bytes / 2 / rate) * 1000))
        if response.pop("cancelled", False) and status == "ok":
            status = "cancelled"
        state.tts.end(status=status, response=response, ended_at_ms=state.tts_ended_at)

    def _finish_turn(self, state: _TurnState) -> None:
        if state.finished:
            return
        state.finished = True
        self._end_stt(state, "ok")
        self._end_tts(state, "ok")
        state.turn.end()
        if self._current_turn is state:
            self._current_turn = None


class VaaniAudioTapMixin:
    """Adds caller and agent PCM capture to an `Agent` via the node hooks.

    `stt_node` and `tts_node` are LiveKit's supported extension points, so this
    tees the exact frames the pipeline uses without reaching into private io
    plumbing that changes between releases.

    Mix it in *before* `Agent` and set `self.vaani = <recorder>`::

        class MyAgent(VaaniAudioTapMixin, Agent):
            ...
    """

    vaani: Optional[VaaniLiveKitRecorder] = None

    async def stt_node(self, audio, model_settings):  # noqa: ANN001, D102
        recorder = self.vaani

        async def tapped():
            async for frame in audio:
                if recorder is not None:
                    recorder.tap_input_frame(frame)
                yield frame

        async for event in _aiter(super().stt_node(tapped(), model_settings)):
            yield event

    async def llm_node(self, chat_ctx, tools, model_settings):  # noqa: ANN001
        """Run the LLM inside the turn's observer scope.

        The provider request is issued from a task whose context was captured
        when the session started, which knows the session but not the turn.
        Entering the turn here is what gives the automatically instrumented HTTP
        span a `turn_id` -- the same job the Node integration's `instrumentLLM`
        wrapper does.
        """
        recorder = self.vaani
        source = super().llm_node(chat_ctx, tools, model_settings)
        async for chunk in _scoped(recorder, source):
            yield chunk

    async def tts_node(self, text, model_settings):  # noqa: ANN001, D102
        recorder = self.vaani
        source = super().tts_node(text, model_settings)
        async for frame in _scoped(recorder, source):
            if recorder is not None:
                recorder.tap_output_frame(frame)
            yield frame


async def _scoped(recorder: Optional[VaaniLiveKitRecorder], source: Any):
    """Advance `source` with the turn scope installed for each step.

    An async generator body runs in the *consumer's* context, so wrapping the
    generator's creation -- or wrapping the `async for` -- would either do
    nothing or leak the scope out to the caller across every `yield`. Entering
    it around each `__anext__` is what puts the provider request inside the
    scope and nothing else, which is exactly what the Node integration does by
    wrapping `iterator.next()`.
    """
    import inspect

    if inspect.isawaitable(source):
        source = await source
    if source is None:
        return
    # The provider's own iterator has to be closed, not a wrapper around it:
    # closing an adapter generator that is suspended at its own `yield` does
    # not propagate `aclose()` inwards, which would leave the provider's
    # request, socket and background tasks alive after the consumer walked away.
    iterator = source.__aiter__()
    try:
        while True:
            scope = recorder.turn_context() if recorder is not None else _null()
            with scope:
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    return
            yield item
    finally:
        await _close_quietly(iterator)


async def _close_quietly(iterator: Any) -> None:
    """Close a provider iterator without letting cleanup mask the real failure.

    This runs on the unwind of whatever ended the stream -- often a cancellation
    at shutdown. A provider whose cleanup raises, or stalls waiting on a socket
    the peer already dropped, must not become the exception the caller sees or
    the reason the process fails to exit. The cleanup is therefore run as a task
    that is *cancelled* on timeout rather than shielded and abandoned, so a
    stalled provider cannot keep running unreferenced after we stop waiting.
    """
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    task = asyncio.ensure_future(_awaited(aclose()))
    try:
        await asyncio.wait([task], timeout=_CLOSE_TIMEOUT_S)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if not task.done():
        task.cancel()
        # Consume the cancellation so the loop never reports it as unretrieved.
        try:
            await task
        except BaseException as error:  # noqa: BLE001 - cleanup is best effort
            logger.debug("vaani: provider iterator cleanup timed out (%s)", error)
        return
    error = task.exception()
    if error is not None:
        # A `CancelledError` raised *by the provider's own cleanup* is the
        # provider's business, not a cancellation of this task, so it is logged
        # like any other cleanup failure instead of being propagated.
        logger.debug("vaani: provider iterator cleanup failed (%s)", error)


async def _awaited(value: Any) -> Any:
    return await value


def _null():
    import contextlib

    return contextlib.nullcontext()


async def _aiter(source: Any):
    """LiveKit nodes may return an iterable, a coroutine yielding one, or None."""
    import inspect

    if inspect.isawaitable(source):
        source = await source
    if source is None:
        return
    async for item in source:
        yield item


def observe_agent_session(
    session: Any,
    recorder: Optional[VaaniLiveKitRecorder] = None,
    **options: Any,
) -> VaaniLiveKitRecorder:
    """Attach a recorder to an `AgentSession`, building one from env if needed."""
    recorder = recorder or VaaniLiveKitRecorder.from_env(**options)
    return recorder.attach(session)


# --------------------------------------------------------------- small helpers


def _guard(handler: Any) -> Any:
    """Event handlers run on LiveKit's loop; a raised error would kill the call."""

    def wrapped(event: Any) -> None:
        try:
            handler(event)
        except Exception as error:  # noqa: BLE001 - recording is best effort
            logger.debug("vaani: handler %s failed (%s)", getattr(handler, "__name__", "?"), error)

    return wrapped


def _speech_id(handle: Any) -> Optional[str]:
    value = getattr(handle, "id", None)
    return str(value) if value else None


def _plain(value: Any) -> Any:
    """Reduce a LiveKit usage object to something `json.dumps` accepts.

    LiveKit mixes pydantic models and plain dataclasses across versions --
    `AgentSessionUsage` is a dataclass, so assuming `model_dump()` silently
    dropped usage from every manifest.
    """
    import dataclasses

    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump())
        except Exception:  # noqa: BLE001 - fall through to the generic paths
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name, None)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _provider(metrics: Any) -> Optional[str]:
    metadata = getattr(metrics, "metadata", None)
    return getattr(metadata, "model_provider", None) or getattr(metrics, "label", None)


def _model(metrics: Any) -> Optional[str]:
    metadata = getattr(metrics, "metadata", None)
    return getattr(metadata, "model_name", None)


def _error_targets(event: Any, error: Any) -> tuple:
    """Which stages a session error belongs to.

    `ErrorEvent.error` is an `STTError`/`LLMError`/`TTSError` pydantic model
    whose `type` literal names the stage, which is the reliable signal. The
    `source` component is the fallback for versions or fakes that omit it, and
    an unrecognised error still fails every open span so a genuine outage is
    never recorded as a clean call.
    """
    kind = getattr(error, "type", None)
    if isinstance(kind, str) and kind in _ERROR_TARGETS:
        return _ERROR_TARGETS[kind]
    source = getattr(event, "source", None)
    if source is not None:
        names = [source] if isinstance(source, str) else [
            base.__name__ for base in type(source).__mro__
        ]
        lowered = {str(name).lower() for name in names}
        if "realtimemodel" in lowered:
            return _ERROR_TARGETS["realtime_model_error"]
        for stage in _ALL_ERROR_TARGETS:
            if stage in lowered:
                return (stage,)
    return _ALL_ERROR_TARGETS
    metadata = getattr(metrics, "metadata", None)
    return getattr(metadata, "model_name", None)


def _frame_data(frame: Any) -> Optional[bytes]:
    data = getattr(frame, "data", None)
    if data is None:
        return frame if isinstance(frame, (bytes, bytearray, memoryview)) else None
    try:
        return bytes(memoryview(data).cast("B"))
    except (TypeError, ValueError):
        try:
            return bytes(data)
        except (TypeError, ValueError):
            return None


def _frame_bytes(frame: Any) -> Optional[int]:
    data = _frame_data(frame)
    return len(data) if data is not None else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default
