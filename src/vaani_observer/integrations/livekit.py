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
from typing import Any, Dict, List, Mapping, Optional

from .._diagnostics import warn_once

logger = logging.getLogger("vaani_observer.livekit")

# Cleanup of a provider iterator is best-effort: a stalled provider must never
# be the reason a call fails to shut down.
_CLOSE_TIMEOUT_S = 5.0

# Whole-upload budget when the caller does not set one. It exists because the
# usual caller is a LiveKit shutdown hook racing `shutdown_process_timeout`
# (10s by default, and the docs ask for 120s): with no budget the retry policy
# can spend minutes on a single leg and be killed anyway, having made the tail
# worse than no retries at all. Pass `upload_timeout=0` for no budget.
DEFAULT_FINISH_UPLOAD_TIMEOUT_S = 60.0

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
# How far rendered audio may fall short of synthesized audio before the reply
# is treated as truncated. Frames still draining when the last one is tapped,
# and rounding at both ends, put a small honest gap on every healthy reply.
_PLAYOUT_TOLERANCE_MS = 250
# The most measured agent speech a whole call may write off as boundary jitter
# before the capture is reported incomplete. A per-turn floor cannot answer
# "did this call lose data", because that answer is a sum: enough sub-floor
# tails add up to seconds of speech with no operation behind them, which is
# precisely the failure this audit round was opened to fix. An absolute bound
# rather than a share of the call, because jitter comes from the number of turn
# boundaries, not from how long the call ran -- a proportional allowance lets an
# hour-long call hide over a minute of speech and still look complete. Whatever
# is written off is published as `measured.tail_written_off_ms`.
_TAIL_WRITE_OFF_CAP_MS = 1000
# Stages whose measurements describe one reply, so a metric that cannot name
# its reply cannot be published. `stt`/`eou` describe the caller's utterance
# and legitimately carry no reply identity.
_REPLY_SCOPED_STAGES = frozenset({"llm", "tts"})

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


class _OutputStream:
    """Identifies one `tts_node` invocation, and the reply it is rendering.

    `speech_id` is the reply's identity taken from LiveKit's own speech-handle
    context, so ownership is *proved* rather than inferred from what happened
    to be speaking. `owner` caches the turn once that id resolves. The pending
    buffers hold output produced before the turn exists, so nothing has to be
    guessed at or dropped in that window.
    """

    __slots__ = ("owner", "speech_id", "pending_bytes", "pending_text",
                 "pending_first_at_ms", "closed", "ownership_inferred")

    def __init__(self) -> None:
        self.owner: Any = None
        self.speech_id: Optional[str] = None
        self.pending_bytes: int = 0
        self.pending_text: List[str] = []
        self.pending_first_at_ms: Optional[int] = None
        self.closed: bool = False
        # Set when the stream had no speech-handle context to name its reply,
        # so whatever it carries was placed by timing. Held here rather than
        # announced at once: a stream that never yields a frame placed nothing.
        self.ownership_inferred: bool = False


def _current_speech_handle() -> Any:
    """The `SpeechHandle` that owns the code calling this, if LiveKit exposes it.

    LiveKit sets `_SpeechHandleContextVar` before `asyncio.create_task` for a
    speech, so every coroutine of that speech -- including `tts_node` -- reads
    back the handle that owns it. Verified on a live call: every `tts_node`
    invocation reported exactly the handle of a preceding `speech_created`,
    including the opening greeting, whose ownership no other signal can prove.

    That matters because every timing-based rule for "whose audio is this"
    is defeated by a real interleaving. The same live call emitted three
    consecutive `speech_created` events with no rendering in between, so
    "the newest speech" and "the one that is speaking" were both wrong.

    The symbol is private, so this degrades to `None` on any version that
    moves it; the caller then falls back to pinning at invocation time.
    """
    try:
        from livekit.agents.voice.agent_activity import (  # type: ignore
            _SpeechHandleContextVar,
        )
    except Exception:  # pragma: no cover - depends on the installed version
        return None
    try:
        return _SpeechHandleContextVar.get(None)
    except Exception:  # pragma: no cover - defensive
        return None


_TEXT_MATCH_MIN_CHARS = 24


def _text_matches(left: str, right: str) -> bool:
    """Whether two renderings of the same reply are the same words.

    The tape off `tts_node` is what we asked to be spoken; `forwarded_text` is
    what LiveKit committed. They differ in whitespace and in how much of an
    interrupted reply each saw, so one being a prefix of the other counts.
    """
    a = " ".join(left.split()).casefold()
    b = " ".join(right.split()).casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    # A prefix is only evidence when it is long enough to identify a reply.
    # Replies routinely open with the same few words ("Sure,", "Of course"),
    # and an interrupted tape can be exactly that short, so a bare
    # `startswith` hands a confident answer to the least distinguishable case.
    if min(len(a), len(b)) < _TEXT_MATCH_MIN_CHARS:
        return False
    return a.startswith(b) or b.startswith(a)


def _pcm16_ms(byte_count: int, sample_rate_hz: int, channels: int = 1) -> int:
    """Milliseconds of 16-bit PCM in `byte_count` bytes.

    The single definition of the bytes-to-duration conversion, so a duration
    quoted next to a byte count is always that byte count and not a different
    measurement that happens to share a name. `channels` is part of the
    denominator for the same reason it is in `session.pcm_duration_ms`: on a
    stereo track, ignoring it reports every duration at twice the truth while
    the audio track's own duration stays right, and the package contradicts
    itself.
    """
    if not byte_count or sample_rate_hz <= 0 or channels <= 0:
        return 0
    return round((byte_count / 2 / channels / sample_rate_hz) * 1000)


def _span_ms(started_at: Any, ended_at: Any) -> Optional[int]:
    """The length of a wall-clock window, when both ends were reported.

    LiveKit reports these as `time.time()` floats, which cannot be compared to
    the recorder's monotonic call clock -- but their difference is a duration,
    and durations are clock-agnostic.
    """
    if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
        return None
    if not isinstance(ended_at, (int, float)) or isinstance(ended_at, bool):
        return None
    if ended_at < started_at:
        return None
    return round((ended_at - started_at) * 1000)


class _TurnState:
    """Everything open for one conversational turn."""

    __slots__ = ("id", "turn", "stt", "stt_ended_at", "stt_response", "llm", "tts",
                 "tts_response", "tts_ended_at", "tools", "audio_bytes",
                 "audio_first_at_ms", "finished", "tts_derived", "tts_ttfa_ms",
                 "reply_complete", "tts_text", "tts_reconstructed", "speech_handle",
                 "reply_source",
                 "open_streams", "awaiting_stream_close", "unscoped_audio_bytes",
                "unscoped_audio_bytes_at_publish", "publication_snapshot_taken",
                 "seq", "finished_at_seq",
                 "tts_was_derived",
                 "awaiting_reply_item",
                 "published_played_ms")

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
        # True when this turn's TTS span was reconstructed rather than measured.
        # A derived span is an estimate standing in for a missing metric, so a
        # measured metric arriving later must correct it rather than be
        # recorded a second time alongside it.
        self.tts_derived = False
        # Permanent: this span's identity and start time were reconstructed,
        # whatever arrived afterwards.
        self.tts_was_derived = False
        # Time to first audio, when there is a basis for it. `None` means "not
        # known", which is reported as absent -- never as zero, which would
        # claim the caller heard the reply the instant it was requested.
        self.tts_ttfa_ms: Optional[int] = None
        # LiveKit has committed the reply's text, but the audio is still
        # draining to the caller: the turn stays open so the rest of it is
        # still attributed here.
        self.reply_complete = False
        # The words handed to `tts_node`, teed chunk by chunk. This is the only
        # record of the agent's speech that exists for every reply, including
        # one generated before the first user turn.
        self.tts_text: List[str] = []
        # True when nothing in the pipeline reported this reply and the span
        # exists only because we taped the frames. Distinct from `tts_derived`,
        # which also covers spans rebuilt from a reported conversation item.
        self.tts_reconstructed = False
        # Milliseconds of agent speech this turn's TTS span actually reported.
        self.published_played_ms = 0
        # The LiveKit speech handle that produced this reply. Kept so an
        # assistant conversation item can be matched to the speech that made it
        # by identity rather than by guessing from timing.
        self.speech_handle: Any = None
        # How LiveKit created this turn's reply, or `None` while it has no
        # reply at all. Decides which provider stages this turn could ever be
        # the claimant for. marker:r9-reply-source
        self.reply_source: Optional[str] = None
        # How many `tts_node` generators are still able to render for this
        # reply. A span closed while one is open publishes a duration shorter
        # than the audio the caller heard, and the frames that arrive after it
        # land on a turn nothing can absorb them into.
        self.open_streams: int = 0
        self.awaiting_stream_close: bool = False
        # Audio that arrived through a tap carrying no stream token, so its
        # owner was resolved by timing rather than proved. Only these
        # milliseconds are eligible for the turn-boundary write-off.
        self.unscoped_audio_bytes: int = 0
        # How much of that had arrived by the time the span was published.
        # Everything before publication is already inside `played_ms`, so only
        # the difference can be part of an unexplained residual.
        self.unscoped_audio_bytes_at_publish: int = 0
        # Whether that snapshot was ever taken. A span closed by the error path
        # never publishes its accounting, and a zero snapshot there reads as
        # "all of this turn's untokenized audio arrived after publication" --
        # which would forgive audio no span accounts for.
        self.publication_snapshot_taken: bool = False
        # Creation and completion order, used to tell a metric that plausibly
        # belongs to this turn from one that plausibly belongs to a reply which
        # ended after this turn had already begun.
        self.seq: int = 0
        self.finished_at_seq: int = 0
        # Stages for which this turn has already received a provider metric.
        # A reply that has one is no longer waiting for one, which is what
        # makes an unidentified metric attributable to somebody else.
        # Retired from rendering, but its span is held open until LiveKit
        # commits the reply's text so the transcript is not lost.
        self.awaiting_reply_item = False


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
        model_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._observer = observer
        #: Why recording is off, when it is off despite being asked for.
        self.last_error: Optional[str] = None
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
                self.last_error = f"{type(error).__name__}: {error}"

        # `_turns` is an index, not the population: one state is registered
        # under both its own id and the LiveKit speech id that adopted it.
        # `_all_turns` is the deduplicated list to iterate and count.
        self._turns: Dict[str, _TurnState] = {}
        self._all_turns: List[_TurnState] = []
        self._pending_turn: Optional[_TurnState] = None
        # Streams that named their speech before that speech was registered.
        self._unpinned_streams: Dict[str, List["_OutputStream"]] = {}
        self._current_turn: Optional[_TurnState] = None
        # The turn whose reply is being synthesized, which is *not* the same as
        # the turn being served: a caller who speaks while the agent is still
        # talking advances `_current_turn` mid-reply. Attributing agent PCM to
        # `_current_turn` therefore credited the interrupting turn -- measured on
        # a real call as a TTS span reporting 18920 ms of audio and 480 bytes.
        self._speaking_turn: Optional[_TurnState] = None
        # The reply the most recent `speech_created` superseded. An assistant
        # conversation item that arrives just after a new speech opens usually
        # belongs to this one, not to the speech that has not spoken yet.
        self._retired_reply: Optional[_TurnState] = None
        self._pending_stt = _PendingStt()
        # STT metrics are emitted per provider request, not per utterance, so a
        # turn whose metric arrived during an earlier utterance would otherwise
        # be written with no provider or model -- which the dashboard reports as
        # "not recorded by SDK" and refuses to price.
        self._stt_identity: Dict[str, Any] = {}
        self._outcome: Optional[str] = None
        self._turn_counter = 0
        self._sockets: List[Any] = []
        # (session, event name, wrapped handler). The *wrapped* reference has to
        # be kept or `session.off()` is impossible: LiveKit matches listeners by
        # identity, and what was registered is the wrapper, not the bound method.
        self._attached: List[tuple] = []
        # Failure classes already reported. Recording is best effort, but the
        # first occurrence of each distinct failure must be visible: three
        # different real bugs used to present identically as "it records
        # nothing" because every one of them was logged at DEBUG.
        self._warned: "set[str]" = set()
        # An explicit label always wins over anything reported or sniffed: the
        # operator knows what they deployed.
        self._model_overrides: Dict[str, str] = dict(model_overrides or {})
        self._sniffed_models: Dict[str, str] = {}
        # What each plugin says it is, read off the component itself. Only ever
        # a *fallback*: a span built from a provider metric takes its identity
        # from that metric. This exists so a span we had to derive -- because no
        # metric arrived at all -- still names its provider and model instead of
        # rendering as "not recorded by SDK" and dropping out of cost reporting.
        self._component_identity: Dict[str, Dict[str, str]] = {}
        #: Every agent PCM byte this recorder taped, whether or not a turn was
        #: open to attribute it to. Independent evidence of whether the agent
        #: spoke at all.
        self._agent_audio_bytes = 0
        # The rate a provider actually used, learned from the first tapped
        # frame. The constructor values are only a default for spans that are
        # written before any audio has been seen.
        self._observed_rates: Dict[str, int] = {}
        # Duration maths must use the same denominator as the audio track's own,
        # or the manifest disagrees with itself on a non-mono track.
        self._observed_channels: Dict[str, int] = {}
        # Whether the agent's audio can be measured at all. Established when the
        # recorder is bound to an agent, *not* inferred from frames arriving:
        # `tts_node` only runs when there is something to say, so an agent that
        # is correctly wired and simply never speaks would otherwise be
        # indistinguishable from one that was never wired -- and those two need
        # opposite responses from whoever reads the call.
        self._agent_audio_tapped = False
        # An agent was handed to us, so "nothing was measured" is a wiring
        # fault we can name rather than a caller who never asked to be measured.
        self._agent_bound = False
        # Turns whose reply was already published when a further provider
        # metric arrived: the later segments' duration and billable character
        # count are not in any span, and that has to reach the manifest.
        self._late_metric_turns: Set[str] = set()
        self._unidentified_metrics: Dict[str, int] = {}
        # True once any stream had to be bound by timing because LiveKit's
        # speech-handle context was unavailable.
        self._stream_ownership_inferred: bool = False
        self._tail_written_off_ms: int = 0
        self._tail_written_off_turns: List[str] = []

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
            from ..observer import upload_options_from_env

            endpoints = overrides.pop("endpoints", None) or _env_endpoints()
            # Patching httpx and aiohttp costs something on every request and
            # buys nothing without rules to match against: `_begin` opens a span
            # only on a rule hit. Leaving them on with an empty rule set is how
            # an adopter concludes "auto HTTP capture is broken" when in fact it
            # was never asked to capture anything.
            #
            # Defaults are deliberately *not* shipped. A rule of type "llm"
            # pointed at api.openai.com would open a second LLM operation
            # alongside the one derived from `metrics_collected`, double-counting
            # tokens and latency in every aggregate. Connection-level capture is
            # therefore opt-in via VAANI_ENDPOINTS, and is for endpoints LiveKit
            # metrics do not already describe.
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
                instrumentations={"http": bool(endpoints), "websocket": bool(endpoints)},
                # Without this the documented upload-tuning table is unreachable
                # from the only constructor the docs show, and every value an
                # operator sets is silently discarded.
                upload=upload_options_from_env(),
            )
        except Exception as error:  # noqa: BLE001 - observability is optional
            logger.error("vaani: disabled, failed to configure observer — %s", error)
            recorder = cls(None, **overrides)
            recorder.last_error = f"{type(error).__name__}: {error}"
            return recorder
        overrides.setdefault("agent_id", os.environ.get("VAANI_AGENT_ID", "livekit-agent"))
        overrides.setdefault("capture_transcripts", _env_bool("VAANI_CAPTURE_STT_CONTENT", True))
        overrides.setdefault("upload", _env_bool("VAANI_UPLOAD", True))
        # Fail at startup, not after the first call is already lost.
        spool_error = observer.preflight()
        if spool_error is not None:
            logger.error("vaani: disabled, %s", spool_error)
            recorder = cls(None, **overrides)
            recorder.last_error = spool_error
            return recorder
        recorder = cls(observer, **overrides)
        # The recorder stays inert rather than raising, because a misconfigured
        # recorder must never be the reason a call fails to start. Staying
        # *silent* about it is a different thing, and is how an adopter ends up
        # with a deployment that records nothing and never finds out.
        if not recorder.enabled:
            logger.error(
                "vaani: VAANI_ENABLED is set but recording is OFF — %s. "
                "No call will be recorded until this is fixed.",
                recorder.last_error or "the observer could not start a session",
            )
        return recorder

    # -------------------------------------------------------------- lifecycle

    @property
    def enabled(self) -> bool:
        return self.call is not None

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        """Report the first occurrence of a failure class, then stay quiet.

        Recording is best effort, so a failing handler must not spam a
        production log on every event. But swallowing everything at DEBUG is
        why three distinct bugs all presented to adopters as "it just records
        nothing". The first of each class is a warning; the rest are DEBUG.
        """
        if key in self._warned:
            logger.debug("vaani: " + message, *args)
            return
        self._warned.add(key)
        logger.warning("vaani: " + message, *args)

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
            # (usage) plus `ChatMessage.metrics` (latency). It remains the only
            # source of per-stage duration, TTFT/TTFB and token counts, and
            # `_record_llm` is reached from here and nowhere else, so LLM spans
            # depend on it. `_on_conversation_item` builds the fallback.
            "metrics_collected": self._on_metrics,
            "session_usage_updated": self._on_usage,
            "speech_created": self._on_speech_created,
            "error": self._on_error,
            "close": self._on_close,
        }
        for name, handler in handlers.items():
            wrapped = _guard(self, name, handler)
            try:
                session.on(name, wrapped)
            except Exception as error:  # noqa: BLE001 - version drift is survivable
                self._warn_once(
                    f"subscribe:{name}", "cannot subscribe to %r (%s)", name, error
                )
                continue
            self._attached.append((session, name, wrapped))
        self._sniff_models(session)
        return self

    def _sniff_models(self, session: Any) -> None:
        """Learn each stage's real model from the plugin instance.

        `metrics.metadata.model_name` reports the plugin's `model` *option*,
        which for `openai.LLM.with_azure(azure_deployment=...)` is left at its
        `"gpt-4o"` default while the real deployment goes to the client. Every
        Azure agent therefore reports `gpt-4o` whatever it is running, and the
        dashboard presents that as fact.
        """
        for kind in ("stt", "llm", "tts"):
            try:
                component = getattr(session, kind, None)
                name = _sniff_model(component)
                identity = _component_identity(component)
            except Exception:  # noqa: BLE001 - a plugin property may raise
                # This reads private attributes of third-party plugins, so it is
                # the code most likely to meet something that throws. Observing
                # the call must never be what ends it.
                continue
            if name:
                self._sniffed_models[kind] = name
            if identity:
                self._component_identity[kind] = identity

    def _model_for(self, kind: str, metrics: Any) -> Optional[str]:
        override = self._model_overrides.get(kind)
        if override:
            return override
        return self._sniffed_models.get(kind) or _model(metrics)

    def _detach(self) -> None:
        """Unsubscribe every handler this recorder registered.

        Without this the handlers outlive the recording: `finish()` releases the
        call while the session keeps emitting, and every later event used to
        dereference `None`.
        """
        for session, name, wrapped in self._attached:
            off = getattr(session, "off", None)
            if off is None:
                continue
            try:
                off(name, wrapped)
            except Exception as error:  # noqa: BLE001 - teardown must not raise
                logger.debug("vaani: cannot unsubscribe from %r (%s)", name, error)
        self._attached.clear()

    async def finish(self, outcome: Optional[str] = None,
                     timeout: Optional[float] = None) -> None:
        """Finalize the local package and, when configured, upload it.

        `timeout` bounds the *upload* only, and defaults to
        `DEFAULT_FINISH_UPLOAD_TIMEOUT_S` rather than to "wait forever": the
        usual caller is a shutdown hook on a clock. Pass `0` to remove the
        budget. Finalization always completes, so the package is on the spool
        and recoverable even when the network leg is abandoned — which is what
        makes running this from a job shutdown hook survivable. See
        `python -m vaani_observer.drain`.
        """
        call = self.call
        if call is None:
            return
        # The close event knows why the call really ended; an explicit outcome
        # from the caller still wins, and "completed" is only the default when
        # nothing observed a reason at all.
        outcome = outcome or self._outcome or "completed"
        # Releasing the call first makes every in-flight handler a clean no-op,
        # which matters because unsubscribing is not atomic with respect to
        # events LiveKit has already dispatched.
        self.call = None
        self._detach()
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
        except Exception as error:  # noqa: BLE001 - a failed call is still a call
            # Losing a recording is the worst thing that can happen here, so it
            # is reported at ERROR and remembered on the recorder rather than
            # logged once at WARNING and forgotten.
            self.last_error = f"{type(error).__name__}: {error}"
            logger.error(
                "vaani: finalization failed, this call was NOT recorded — %s. "
                "The partial package is at %s.",
                error, getattr(call, "directory", "the spool"),
            )
            return
        if not (self._upload and self._observer is not None):
            return
        if timeout is None:
            timeout = DEFAULT_FINISH_UPLOAD_TIMEOUT_S
        try:
            result = await self._observer.upload_package(
                finalized, timeout=timeout if timeout > 0 else None
            )
            # Leave a receipt, or a drain sidecar polling the same spool will
            # re-ship this package -- full audio payload -- on every tick for
            # the rest of the worker's life.
            _mark_delivered(finalized, result, self._observer)
            logger.info(
                "vaani: uploaded %s status=%s operations=%s",
                result.get("session_id"),
                result.get("status"),
                result.get("operation_count"),
            )
        except Exception as error:  # noqa: BLE001 - a failed upload is not a failed call
            # The package is already written, so this is recoverable — but only
            # if the operator knows the spool now holds something that needs
            # draining, and that on ephemeral storage it will not survive.
            logger.warning(
                "vaani: upload failed, package retained at %s — %s. "
                "Run `python -m vaani_observer.drain` to retry.",
                finalized.directory,
                error,
            )

    def finalize_open_spans(self, call: Any = None, outcome: Optional[str] = None) -> None:
        """Close anything still open so a dropped call cannot leak a span."""
        for state in self._all_turns:
            self._end_stt(state, "ok" if state.stt_ended_at is not None else "cancelled")
            for operation in state.llm:
                if not operation.ended:
                    # A reply LiveKit reported as delivered is not cancelled
                    # merely because the call has since ended: the turn is only
                    # still open because its playout window was being measured.
                    operation.end(status="ok" if state.reply_complete else "cancelled")
            # Deliberately not keyed on `reply_complete`. A reply can be fully
            # rendered and fully measured and still never commit an item -- a
            # `say(add_to_chat_ctx=False)`, or a room that disconnects between
            # playout ending and the commit -- and marking those `cancelled`
            # is the false-barge-in defect this round fixed, reintroduced one
            # frame higher. `_end_tts` weighs the actual evidence: the
            # provider's cancelled flag, `item.interrupted`, and played versus
            # synthesized duration.
            self._end_tts(state, "ok")
            for operation in state.tools.values():
                if not operation.ended:
                    operation.end(status="cancelled")
            state.tools.clear()
            if not state.finished:
                state.finished = True
                state.finished_at_seq = self._turn_counter
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
        self._audit_coverage(call)

    def _audit_coverage(self, call: Any = None) -> None:
        """Refuse to report a call as fully captured when it demonstrably is not.

        This recorder taps every frame that goes through `tts_node`, so it holds
        independent proof of whether the agent spoke -- proof that does not
        depend on the TTS plugin emitting anything. A turn that rendered audio
        but carries no TTS span is a measured gap, and the package must say so:
        a 100% undercount of the agent's talk time behind a green status is
        worse than no number at all, because it is believed.

        `_derive_tts` closes the known cause of this. The audit stays because
        the next cause will not be known in advance, and the point is that it
        cannot be silent.
        """
        call = call or self.call
        report = getattr(call, "report_coverage_gap", None)
        if report is None:
            return
        rate = self._rate_for("agent") or 1
        channels = self._channels_for("agent") or 1
        self._report_agent_audio(call, rate, channels)
        # An agent that was never tapped cannot be audited at all, and the
        # audit below would clear it: with no frames counted, `measured_ms` is
        # zero and nothing is ever unattributed. That is the wrong answer to a
        # different question. "Every millisecond we taped is on a span" is
        # trivially true when we taped nothing, and reporting it as a complete
        # capture tells an operator their numbers are trustworthy at the one
        # moment they are least able to be. A correctly wired agent that simply
        # never spoke stays distinguishable, because its tap *was* installed.
        if not self._agent_audio_tapped:
            # `_agent_bound` selects the *advice*, never whether to speak up.
            # Gating the gap on it meant a recorder attached without `agent=`
            # at all -- no tap, nothing measured, `agent_audio_ms: 0` -- passed
            # the audit silently, because with no frames counted nothing can
            # ever be unattributed. "Every millisecond we taped is on a span"
            # is trivially true when we taped nothing, and it is the least
            # deserving call on the platform to be showing a green status.
            logger.warning(
                "vaani: no agent audio tap is active, so this call's agent "
                "speech was never measured and its capture cannot be verified. "
                "%s",
                "Mix VaaniAudioTapMixin into your Agent ahead of the framework "
                "base class (class MyAgent(VaaniAudioTapMixin, Agent))."
                if self._agent_bound else
                "Pass agent=<your Agent> to observe_agent_session() so the "
                "recorder can measure what the agent actually rendered.",
            )
            try:
                report(
                    "tts",
                    "no agent audio tap was active, so capture could not be verified",
                    agent_audio_tapped=False,
                )
            except Exception as error:  # noqa: BLE001 - teardown must not raise
                logger.debug("vaani: cannot record coverage gap (%s)", error)
            return
        # A reply that is still draining when the caller starts talking leaves a
        # few hundred milliseconds on the next turn. That is boundary jitter
        # between two clocks, not a stage that failed to report -- and
        # downgrading an otherwise complete call for it would train operators
        # to ignore the one status that means "a number here is missing".
        turns = [
            s for s in self._all_turns
            if s.tts is None
            and (s.tts_text
                 or _pcm16_ms(s.audio_bytes, rate, channels) > _PLAYOUT_TOLERANCE_MS)
        ]
        # The totals are the load-bearing check. Comparing per-turn only finds
        # audio that landed on a turn, so frames rendered while no turn was
        # open at all -- a greeting spoken before the session reported any
        # speech, anything after the last turn was retired -- were invisible to
        # it: measured talk time exceeded the sum of the spans and the call
        # still reported itself fully captured. That is the audit's exact
        # complaint, so the invariant is stated the way a reader would state
        # it: every millisecond we taped is on a span, or we say it is not.
        attributed_ms = sum(s.published_played_ms for s in self._all_turns)
        # Audio on a turn that never replied, below the floor for deriving a
        # span: the tail of the previous reply still draining when the caller
        # barged in. Subtracted by name rather than folded into the gap,
        # because we know exactly where these milliseconds went -- reporting
        # them as missing would flag healthy calls and teach an operator to
        # ignore the one status that means a number is really absent.
        #
        # Two limits, because the per-turn floor is the wrong shape for a
        # question whose answer is a sum. A turn is only eligible if nothing
        # about it looks like a real reply -- words taped off `tts_node` mean
        # the agent was *given something to say*, which no drain tail ever is
        # -- and the write-off is then capped, because twelve barge-ins each
        # leaving 240ms wrote off 2.88 seconds of measured speech against zero
        # TTS spans and still reported the call fully captured. That is the
        # audit's P0-A signature exactly: audio with no operations behind a
        # green status.
        # Eligibility asks whether the turn ever published what its span
        # accounted for, not whether it has a span. Note what this is *not*
        # doing: a published turn cannot carry a forgivable residual at all,
        # because untokenized audio only ever lands on the turn that is still
        # rendering and publication happens when that turn is retired, so the
        # cap below is already zero for it. The live case is the error path,
        # which ends spans directly and publishes nothing -- see below.
        tail_turns = []
        tail_ms = 0
        for s in self._all_turns:
            if s.tts_text and s.tts is None:
                # Words were taped for it, so this is a reply that lost its
                # span, not a tail. Writing it off would hide the failure.
                continue
            residual = max(
                0, _pcm16_ms(s.audio_bytes, rate, channels) - s.published_played_ms)
            # Only audio whose owner was *inferred* is forgivable. A tokenized
            # stream names its reply, so a residual on one is not boundary
            # jitter -- it is a lifecycle defect, and writing it off would
            # excuse exactly the class of bug this allowance keeps being asked
            # to cover.
            if s.tts is not None and not s.publication_snapshot_taken:
                # marker:r8-no-writeoff-without-publication
                # A span exists but was ended without publishing what it
                # accounted for -- the error path ends operations directly and
                # does exactly that. `published_played_ms` is then zero, which
                # reads as "this reply published nothing, so all of its audio
                # is post-publication residual" and forgives a whole reply's
                # worth of speech that no span accounts for.
                #
                # A turn with no span at all is different and stays eligible:
                # that is the drain tail this allowance was written for, where
                # nothing was published because there was nothing to publish.
                continue
            # marker:r7-residual-post-publish
            # The *post-publication* unscoped audio, not the turn's lifetime
            # total: audio tapped before the span closed is already counted in
            # `played_ms` and cannot be part of this residual. Using the total
            # let a turn's early untokenized frames buy forgiveness for later
            # identified-stream frames -- a lifecycle defect written off as
            # boundary jitter, which is the one thing this allowance must
            # never do.
            residual = min(residual, _pcm16_ms(
                max(0, s.unscoped_audio_bytes - s.unscoped_audio_bytes_at_publish),
                rate, channels))
            if residual:
                tail_ms += residual
                tail_turns.append(s.id)
        measured_ms = _pcm16_ms(self._agent_audio_bytes, rate, channels)
        # Bounded absolutely, not as a share of the call. A fraction of the
        # call length is the wrong shape twice over: boundary jitter is a
        # property of how many turn boundaries there were, not of how long the
        # call ran, and scaling it means an hour-long call can hide 72 seconds
        # of measured speech behind a green status -- the audit's P0-A, reached
        # by arithmetic instead of by a bug. The cap is also *published*, so
        # once the number is visible the exact value stops being load-bearing.
        tail_ms = min(tail_ms, _TAIL_WRITE_OFF_CAP_MS)
        self._tail_written_off_ms = tail_ms
        self._tail_written_off_turns = tail_turns
        unattributed_ms = max(0, measured_ms - attributed_ms - tail_ms)
        # Over-attribution is audited too. Reporting more speech than was
        # rendered is the same class of defect as reporting less -- and it is
        # the one a double-counted span produces, so an audit blind to it
        # cannot catch the failure most likely to flatter the numbers.
        overattributed_ms = max(0, attributed_ms - measured_ms)
        note = getattr(call, "report_capture_measurement", None)
        if note is not None:
            try:
                # Published whether or not it changes the verdict: a write-off
                # that is invisible is indistinguishable from data that was
                # never lost, which is the complaint this audit opened with.
                derived_ms = self._derived_agent_audio_ms(rate, channels)
                note(tail_written_off_ms=tail_ms,
                     tail_write_off_cap_ms=_TAIL_WRITE_OFF_CAP_MS,
                     tail_written_off_turn_ids=tail_turns,
                     # Published unconditionally, including when it sits under
                     # the tolerance that keeps it out of the verdict. A
                     # threshold that is applied but never shown is a second,
                     # invisible write-off on top of the first.
                     unattributed_agent_audio_ms=unattributed_ms,
                     unattributed_tolerance_ms=_PLAYOUT_TOLERANCE_MS,
                     # How much of the agent's speech is described by spans the
                     # recorder rebuilt rather than the provider measured.
                     # Deliberately a published *fact* and not a term in
                     # `coverage_complete`: that flag answers "did this call
                     # lose data", and a reconstructed span lost none -- every
                     # millisecond is attributed and each span names its
                     # source. Folding it in would fire the flag on nearly
                     # every Deepgram `aura-2` call, where roughly three
                     # replies in four emit no metric (60% of a measured live
                     # call), and a status that is red on healthy calls is one
                     # operators learn to skip past.
                     derived_tts_share_pct=(
                         round(derived_ms * 100 / measured_ms) if measured_ms else 0),
                     # Whether every reply's audio was bound to it by identity
                     # or some of it by timing. The fallback is sound, but it
                     # is a weaker claim, and a reader auditing a per-turn
                     # number deserves to know which one it rests on -- the
                     # audit's recurring complaint is not wrong numbers so much
                     # as numbers that do not say how sure they are.
                     stream_ownership=(
                         "inferred" if self._stream_ownership_inferred else "proved"),
                     )
            except Exception as error:  # noqa: BLE001 - teardown must not raise
                logger.debug("vaani: cannot record measurement (%s)", error)
        if self._late_metric_turns:
            # The reply was already published when a further provider metric
            # arrived, so that segment's duration and billable characters are
            # in no span. Only a process log said so, which nothing downstream
            # can read: a bill computed from this package is low while the page
            # calls the call complete.
            try:
                report(
                    "tts",
                    "provider metrics arrived after the reply was published, so "
                    "later segments' duration and character counts are missing",
                    turn_ids=sorted(self._late_metric_turns),
                )
            except Exception as error:  # noqa: BLE001 - teardown must not raise
                logger.debug("vaani: cannot record coverage gap (%s)", error)
        for stage, dropped in sorted(self._unidentified_metrics.items()):
            # Dropped on purpose: see `_unidentified_metric_turn`. Saying so is
            # the difference between a missing provider measurement and a
            # confidently wrong one, and only the package can carry that. The
            # gap names the stage that was lost -- reporting every one of them
            # as "tts" sent a reader looking for audio that was never missing.
            try:
                report(
                    stage,
                    f"{dropped} {stage} metric(s) carried no speech_id while "
                    "more than one reply was in flight and were dropped rather "
                    "than published on a turn that may not have produced them",
                )
            except Exception as error:  # noqa: BLE001 - teardown must not raise
                logger.debug("vaani: cannot record coverage gap (%s)", error)
        if (unattributed_ms <= _PLAYOUT_TOLERANCE_MS
                and overattributed_ms <= _PLAYOUT_TOLERANCE_MS and not turns):
            return
        if overattributed_ms > _PLAYOUT_TOLERANCE_MS:
            logger.warning(
                "vaani: tts operations report %dms more agent speech than was "
                "rendered (%dms reported, %dms taped); this call is reported as "
                "an incomplete capture rather than as healthy.",
                overattributed_ms, attributed_ms, measured_ms,
            )
            try:
                report(
                    "tts",
                    "tts operations report more agent speech than was rendered",
                    reported_agent_audio_ms=attributed_ms,
                    measured_agent_audio_ms=measured_ms,
                    overattributed_agent_audio_ms=overattributed_ms,
                )
            except Exception as error:  # noqa: BLE001 - teardown must not raise
                logger.debug("vaani: cannot record coverage gap (%s)", error)
            if unattributed_ms <= _PLAYOUT_TOLERANCE_MS and not turns:
                return
        logger.warning(
            "vaani: %dms of agent audio is not attributed to any tts operation "
            "(%d turn(s) rendered audio with no span); this call is reported as "
            "an incomplete capture rather than as healthy. Check that the TTS "
            "plugin emits metrics_collected.",
            unattributed_ms, len(turns),
        )
        try:
            report(
                "tts",
                "agent audio was rendered that no tts operation accounts for",
                turn_count=len(turns),
                unattributed_agent_audio_ms=unattributed_ms,
                # Naming the turns is the difference between "some audio is
                # missing" and a lead an operator can actually pull on.
                turn_ids=[s.id for s in turns],
            )
        except Exception as error:  # noqa: BLE001 - teardown must not raise
            logger.debug("vaani: cannot record coverage gap (%s)", error)

    def _derived_agent_audio_ms(self, rate: int, channels: int) -> int:
        return sum(
            _pcm16_ms(state.audio_bytes, rate, channels)
            for state in self._all_turns
            if state.tts is not None and (state.tts_derived or state.tts_was_derived)
        )

    def _report_agent_audio(self, call: Any, rate: int, channels: int = 1) -> None:
        """Publish how much audio the agent actually produced.

        Recorded on every call, including -- especially -- the ones where the
        answer is zero. A call with no operations reads as a broken recorder,
        and an operator sent to debug the SDK never learns that their agent was
        mute for a minute, which is the failure that actually cost them the
        caller. The tap runs in `tts_node`, so this number is measured, not
        inferred from any span.
        """
        note = getattr(call, "report_capture_measurement", None)
        if note is None:
            return
        rendered = self._agent_audio_bytes
        # How much of the TTS accounting exists only because we rebuilt it from
        # our own tape. Published because a coverage audit that counts a
        # reconstructed span as proof of coverage is grading a paper it wrote
        # itself: without this number, "100% of the agent's speech is
        # accounted for" and "no stage of the pipeline reported any of it" are
        # indistinguishable to the reader.
        reconstructed = [
            state for state in self._all_turns
            if state.tts is not None and state.tts_reconstructed
        ]
        # Broader, and the number a reader actually needs: every TTS span whose
        # timings came from anywhere other than the provider's own
        # `tts_metrics`. On a real Deepgram `aura-2` call 3 of 4 replies emit no
        # metric, so a manifest reporting only `reconstructed_op_count: 0` --
        # true, because those three *were* announced by
        # `conversation_item_added` -- told the reader every span was measured
        # when three quarters of them were estimates. Silence about an estimate
        # is the same failure class as a wrong number, and harder to catch.
        derived = [
            state for state in self._all_turns
            if state.tts is not None and (state.tts_derived or state.tts_was_derived)
        ]
        try:
            note(
                agent_audio_ms=_pcm16_ms(rendered, rate, channels),
                agent_audio_bytes=rendered,
                agent_audio_sample_rate_hz=rate,
                agent_audio_channels=channels,
                reconstructed_op_count=len(reconstructed),
                reconstructed_agent_audio_ms=sum(
                    _pcm16_ms(state.audio_bytes, rate, channels)
                    for state in reconstructed
                ),
                derived_tts_op_count=len(derived),
                derived_tts_agent_audio_ms=sum(
                    _pcm16_ms(state.audio_bytes, rate, channels)
                    for state in derived
                ),
                # Whether the measurement was possible at all. Without this a
                # reader cannot tell "your agent was silent" from "you never
                # bound `agent=`, so nothing was ever measured" -- and the
                # console would state the first with total confidence.
                agent_audio_tapped=self._agent_audio_tapped,
            )
        except Exception as error:  # noqa: BLE001 - teardown must not raise
            logger.debug("vaani: cannot record audio measurement (%s)", error)

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
            self._warn_once(
                "observe-socket", "observe_websocket failed (%s)", error
            )
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

    def note_audio_tap_installed(self, agent: Any = None) -> None:
        """Record that this recorder is positioned to measure the agent.

        The mixin overrides `tts_node`, so an agent carrying it and holding a
        reference to this recorder will route every rendered frame here. That
        is knowable at wire-up, and knowing it is what lets a zero be reported
        as "your agent was silent" rather than "nothing was measured".
        """
        if agent is not None and getattr(agent, "vaani", None) is not self:
            return
        if agent is not None:
            self._agent_bound = True
        if agent is not None and not _tap_is_active(agent):
            # Bound, but the tapping `tts_node` is not the one Python will
            # call, so no frame can reach us. `isinstance` is not enough:
            # `class Wrong(Agent, VaaniAudioTapMixin)` passes it while `Agent`
            # wins the MRO and the tap never runs -- and reporting a
            # measurement then is the same false claim by a subtler route.
            self._warn_once(
                "no-tap-mixin",
                "vaani: agent is bound but its tts_node is not VaaniAudioTapMixin's, "
                "so no agent audio can be captured. Declare the mixin *first*: "
                "`class MyAgent(VaaniAudioTapMixin, Agent)`.",
            )
            return
        self._agent_audio_tapped = True

    def tap_output_frame(self, frame: Any,
                         stream: "Optional[_OutputStream]" = None) -> None:
        """Record one agent PCM frame. Called from `Agent.tts_node`."""
        # Also set here, so a caller who wires `agent.vaani` by hand rather than
        # through `observe_agent_session` is still measured rather than
        # reported as unmeasurable.
        self._agent_audio_tapped = True
        self._tap(frame, inbound=False)
        count = _frame_bytes(frame) or 0
        # Counted before any turn attribution, and independently of it. A
        # greeting spoken before the first turn opens belongs to no turn, and
        # crediting only attributed frames would report "the agent never spoke"
        # about a call that opened with the agent speaking.
        if count and self.call is not None:
            self._agent_audio_bytes += count
        state = self._stream_turn(stream)
        if state is None or self.call is None:
            if state is None and count and stream is not None and self.call is not None:
                stream.pending_bytes += count
                if stream.pending_first_at_ms is None:
                    # Held with the bytes: "time to first audio" is measured
                    # from this instant, so recovering the audio without it
                    # would publish the reply's latency against a later frame.
                    stream.pending_first_at_ms = self.call.now()
            return
        if not count:
            return
        state.audio_bytes += count
        if stream is None or stream.ownership_inferred:
            # marker:r8-mark-at-credit
            # Marked where the audio is actually credited, so the flag reports
            # attribution that happened rather than attribution that might.
            self._mark_ownership_inferred()
        if stream is None:
            # No token, so this turn was chosen by timing. Track it separately:
            # it is the only audio a turn-boundary write-off may forgive.
            state.unscoped_audio_bytes += count
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
        if state.tts_derived and state.tts_ttfa_ms is None:
            # A derived span has no measured request time, so its start was
            # anchored at the first frame. Stamping the milestone there would
            # publish "time to first audio: 0ms" for every reply on a plugin
            # that emits no metrics. A missing number is read as unknown; a
            # zero is read as instant, and would be charted as such.
            return
        operation.event(
            "audio_chunk",
            occurred_at_ms=state.audio_first_at_ms,
            total_byte_count=state.audio_bytes,
        )

    def _rate_for(self, track: str) -> int:
        """The rate actually observed on a track, falling back to the default.

        A span written before any frame has been tapped can only report the
        configured default; once a frame has been seen, the default is a guess
        we no longer need. Deepgram at 16 kHz used to produce spans claiming
        24 kHz purely because that is the constructor default.
        """
        observed = self._observed_rates.get(track)
        if observed:
            return observed
        fmt = self._input_format if track == "caller" else self._output_format
        return int(fmt["sample_rate_hz"])

    def _channels_for(self, track: str) -> int:
        """The channel count actually observed on a track, as `_rate_for`."""
        observed = self._observed_channels.get(track)
        if observed:
            return observed
        fmt = self._input_format if track == "caller" else self._output_format
        return int(fmt.get("channels") or 1)

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
        track = "caller" if inbound else "agent"
        self._observed_rates.setdefault(track, int(fmt["sample_rate_hz"]))
        self._observed_channels.setdefault(track, int(fmt.get("channels") or 1))
        try:
            if inbound:
                call.record_inbound_audio(data, fmt)
            else:
                call.record_outbound_audio(data, fmt)
        except Exception as error:  # noqa: BLE001 - audio loss beats audio stall
            # The frame is gone either way, but the manifest must not go on
            # claiming the recording is complete. A package that is quietly
            # missing audio while `audio_complete` stays true is worse than one
            # that admits the gap, because every number derived from it is
            # trusted.
            try:
                call.degrade_audio()
            except Exception:  # noqa: BLE001 - accounting must not raise
                pass
            self._warn_once(
                f"audio-drop:{track}",
                "%s audio frame dropped and the recording is now incomplete (%s)",
                track,
                error,
            )

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
                "sample_rate_hz": self._rate_for("caller"),
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
        # `source` is `"say"` or `"generate_reply"`. An unknown value is read
        # as a generated reply, which is the conservative reading: it keeps
        # the turn eligible to claim an LLM measurement rather than silently
        # narrowing the field on a build whose events say something new.
        source = getattr(event, "source", None) or "generate_reply"
        if speech_id in self._turns:
            self._turns[speech_id].speech_handle = getattr(event, "speech_handle", None)
            self._turns[speech_id].reply_source = source
            self._current_turn = self._turns[speech_id]
            self._retire_reply(keep=self._turns[speech_id])
            self._speaking_turn = self._turns[speech_id]
            return
        state = self._pending_turn
        if state is None:
            # `say()` and the opening greeting produce a turn with no user
            # speech at all.
            state = self._new_turn()
        self._pending_turn = None
        self._turns[speech_id] = state
        state.speech_handle = getattr(event, "speech_handle", None)
        state.reply_source = source
        # A `tts_node` invocation can render a whole short reply and return
        # before this event is dispatched. Its stream named this speech from
        # the start, so nothing about it was ever ambiguous -- it was only
        # waiting for the turn to exist. marker:r9-resolve-deferred
        for deferred in self._unpinned_streams.pop(speech_id, []):
            self._pin_stream(deferred, state)
        self._current_turn = state
        # A new reply supersedes the previous one, so whatever was still
        # draining has now stopped: this is the point at which the last reply's
        # playout is known to be over and its span can be closed honestly.
        self._retire_reply(keep=state)
        self._speaking_turn = state
        self._harvest_speech_text(getattr(event, "speech_handle", None), state)

    def open_output_stream(self) -> "_OutputStream":
        """A handle identifying one `tts_node` invocation.

        One call to `tts_node` renders exactly one reply, so the frames it
        yields belong to that reply for as long as it runs. Resolving each
        frame against the recorder's *global* idea of who is speaking breaks
        that: `say()` creates a speech and LiveKit emits `speech_created` for it
        while the previous reply is still yielding, so the older reply's
        remaining audio -- and its words -- were credited to a reply that had
        not made a sound. The total is conserved, so the corruption reports as a
        fully-covered call while two turns' talk time, and every latency and
        cost figure derived from them, are wrong in opposite directions.

        The reply is identified from LiveKit's speech-handle context rather
        than from whoever is speaking when output first appears. Resolving on
        the first output is a trap: `tts_node`'s first frame waits on the LLM's
        first token, and a `say()` queued inside that window moves the global
        "who is speaking" before the pin happens -- binding the *whole* reply
        to the wrong turn for the rest of the generator.
        """
        stream = _OutputStream()
        stream.speech_id = _speech_id(_current_speech_handle())
        if stream.speech_id is None:
            # No context to prove ownership with. Pinning here, at invocation
            # time, is still strictly better than pinning at first output:
            # invocation happens before the LLM round-trip that a competing
            # `say()` can slip into. Recorded, because a reader cannot
            # otherwise tell an attribution that was proved from one that was
            # inferred, and those do not deserve the same confidence.
            # Recorded on the stream, not on the call. An invocation that is
            # cancelled before yielding a frame placed no audio by timing, and
            # downgrading a call for a stream that never spoke would make the
            # flag mean "something might have happened" instead of "a number on
            # this page rests on a guess".
            stream.ownership_inferred = True
            self._pin_stream(stream, self._rendering_turn())
        else:
            # Pinned now, not on first output. `tts_node` can be invoked a full
            # LLM round-trip before it yields anything, and until its stream is
            # counted the turn looks idle -- so the next `speech_created`
            # retires and closes a reply that is still about to speak.
            if self._pin_stream(stream, self._turns.get(stream.speech_id)) is None:
                self._defer_stream(stream)
        return stream

    def _defer_stream(self, stream: "_OutputStream") -> None:
        """Hold a stream until the speech it named is registered.

        Waiting is not the same as losing. Whatever the stream buffers in this
        window is provably owned -- it carries the speech id -- so the only
        question is when the turn appears, and until this existed the answer
        was "when the next frame arrives", which for a reply that has already
        finished rendering is never. marker:r9-defer
        """
        if stream.speech_id is None:
            return
        waiting = self._unpinned_streams.setdefault(stream.speech_id, [])
        if stream not in waiting:
            waiting.append(stream)

    def _pin_stream(self, stream: "_OutputStream",
                    owner: "Optional[_TurnState]") -> "Optional[_TurnState]":
        """Bind a stream to its reply, once, and count it as open.

        Idempotent by design: ownership is decided at most once per `tts_node`
        invocation, so a stream can never be counted twice against a turn nor
        moved to another turn after output has been credited to the first.
        """
        if stream.owner is not None or owner is None:
            return stream.owner
        stream.owner = owner
        if not stream.closed:
            owner.open_streams += 1
        self._flush_stream_buffer(stream, owner)
        return owner

    def _mark_ownership_inferred(self) -> None:
        """Record that some audio was bound to its reply by timing.

        Called from both routes that can do so -- a build without the
        speech-handle context, and a frame tapped with no stream token at all
        -- because the manifest's claim is about the *call*, not about which
        code path made it. Reporting `proved` while a tail was placed by
        timing is the failure class this field exists to prevent.
        """
        if self._stream_ownership_inferred:
            return
        self._stream_ownership_inferred = True
        self._warn_once(
            "stream_ownership_inferred",
            "vaani: some agent audio is bound to its reply by timing rather "
            "than by identity, because this livekit-agents build does not "
            "expose the speech-handle context or the frame was tapped without "
            "one; per-turn talk time and cost can move between adjacent "
            "replies (manifest: capture_status.measured."
            "stream_ownership='inferred')",
        )

    def _stream_turn(self, stream: "Optional[_OutputStream]") -> "Optional[_TurnState]":
        if stream is None:
            # No token at all: whichever reply is rendering *now* gets it. That
            # is timing, and a reply's own tail draining past the next reply's
            # start is credited to the wrong one -- so the call may not claim
            # its per-turn audio was proved.
            return self._rendering_turn()  # marker:r8-streamturn-plain
        if stream.owner is not None:
            return stream.owner
        if stream.speech_id is not None:
            # Identity, not timing. `None` here means the turn has not been
            # registered yet, which is a wait -- never a licence to guess.
            owner = self._pin_stream(stream, self._turns.get(stream.speech_id))
            if owner is None:
                self._defer_stream(stream)
            return owner
        # A stream that opened before any turn existed and had no handle to
        # name: attach it as soon as a turn appears.
        return self._pin_stream(stream, self._rendering_turn())

    def close_output_stream(self, stream: "Optional[_OutputStream]") -> None:
        """One `tts_node` generator has stopped producing.

        Retirement is deferred while a reply's own generator is still open,
        because a reply is not over when the *next* one is authorized -- an
        interrupted reply keeps draining frames the caller already heard.
        Closing then publishes a span shorter than the audio it played, and
        the frames that arrive afterwards land on a turn whose span is
        immutable, where they read as unattributed audio rather than as the
        speech they are.
        """
        if stream is None or stream.closed:
            return
        stream.closed = True
        owner = stream.owner
        if owner is None:
            return
        if owner.open_streams > 0:
            owner.open_streams -= 1
        if owner.open_streams or owner.finished:
            return
        if owner.awaiting_stream_close:
            owner.awaiting_stream_close = False
            if owner.awaiting_reply_item:
                # Still owed its transcript; the item's arrival closes it.
                return
            self._finish_turn(owner)

    def _flush_stream_buffer(self, stream: "_OutputStream",
                             owner: "_TurnState") -> None:
        """Credit output produced before this stream's turn was known.

        Without this, everything rendered in that window is counted in the
        call's total but on no turn, which surfaces as unattributed audio and
        a missing transcript for the reply the caller heard first.
        """
        if stream.pending_bytes:
            if stream.ownership_inferred:
                # marker:r8-flush-marks
                self._mark_ownership_inferred()
            owner.audio_bytes += stream.pending_bytes
            stream.pending_bytes = 0
            if owner.audio_first_at_ms is None:
                owner.audio_first_at_ms = stream.pending_first_at_ms
            self._mark_first_audio(owner)
        if stream.pending_text:
            owner.tts_text.extend(stream.pending_text)
            stream.pending_text = []

    def _rendering_turn(self) -> "Optional[_TurnState]":
        """The turn that owns what is coming out of `tts_node` right now.

        Both the frames and the words of a reply arrive through the same node
        call, so they must be attributed by the same rule or they disagree:
        resolving the text more strictly than the audio produced spans that
        carried a measured `played_ms` and no record of what was said, which is
        the audit's "the agent's words are never recorded" on the very turns
        that prove the agent spoke.

        Credits the reply being spoken, not whoever is talking now -- a caller
        who interrupts rotates `_current_turn` onto a new turn while the
        previous reply is still draining.
        """
        state = self._speaking_turn
        if state is None or state.finished:
            state = self._current_turn
        return state

    def tap_output_text(self, chunk: Any,
                        stream: "Optional[_OutputStream]" = None) -> None:
        """Record one chunk of the text being synthesized, from `tts_node`.

        The words handed to the TTS node are the only source of the agent's
        speech that is present for *every* reply. LiveKit emits no
        `conversation_item_added` for a reply generated before the first user
        turn, and populates `SpeechHandle.chat_items` from that same path, so
        an agent that opens the call by speaking -- the overwhelmingly common
        design -- had its greeting rendered, measured and charted while the
        transcript showed nothing for it. "The agent's words are never
        recorded" was the audit's second headline defect, and a transcript that
        silently omits the first thing the caller heard is that defect wearing
        a smaller hat.

        This is the same tactic that fixed the audio accounting: tee what
        actually flows through the pipeline rather than trusting a stage to
        announce it.
        """
        if not isinstance(chunk, str) or not chunk:
            return
        state = self._stream_turn(stream)
        if state is None:
            if stream is not None:
                # Held, not dropped: the words of a reply rendered before its
                # turn was registered are the transcript of the first thing
                # the caller heard.
                stream.pending_text.append(chunk)
            return
        state.tts_text.append(chunk)

    def _harvest_speech_text(self, handle: Any, state: "_TurnState") -> None:
        """Take the reply's words off the speech handle when it finishes.

        LiveKit emits no `conversation_item_added` for a reply generated before
        the first user turn, so an agent that opens the call by speaking -- the
        overwhelmingly common design -- had its greeting rendered, measured and
        charted while the transcript showed nothing for it. "The agent's words
        are never recorded" was the audit's second headline defect, and a
        transcript that silently omits the first thing the caller heard is the
        same defect wearing a smaller hat.

        `SpeechHandle.chat_items` is the public record of what this speech
        produced, and it is populated for the greeting. It is read in a done
        callback so an interrupted reply still reports the words that were
        actually spoken.
        """
        if handle is None or not self._capture_transcripts:
            return
        register = getattr(handle, "add_done_callback", None)
        if not callable(register):
            return

        def _take(completed: Any) -> None:
            try:
                if state.tts_response.get("text"):
                    # `conversation_item_added` already reported this reply and
                    # its `forwarded_text` is the better record: it is what
                    # reached the caller, not what the model produced.
                    return
                parts = [
                    (getattr(item, "text_content", None) or "").strip()
                    for item in (getattr(completed, "chat_items", None) or [])
                    if getattr(item, "role", None) == "assistant"
                ]
                text = " ".join(part for part in parts if part).strip()
                if not text:
                    return
                state.tts_response["text"] = text
                state.tts_response["char_count"] = len(text)
                # Named so a reader can tell this came from the speech handle
                # rather than from the forwarded-text event, which is the one
                # that proves the words were actually played.
                state.tts_response["text_source"] = "speech_handle"
            except Exception as error:  # noqa: BLE001 - a transcript must never break a call
                logger.debug("vaani: cannot read speech chat items (%s)", error)

        try:
            register(_take)
        except Exception as error:  # noqa: BLE001
            logger.debug("vaani: cannot observe speech completion (%s)", error)

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
        self._relabel_usage(payload)
        self.call.metadata["usage"] = payload

    def _relabel_usage(self, payload: Any) -> None:
        """Apply the resolved model name to the session usage rollup.

        LiveKit reports the plugin's `model` *option* here, which for
        `openai.LLM.with_azure(azure_deployment=...)` is left at its default of
        `"gpt-4o"`. The per-turn spans already correct this, but the rollup is
        what anyone aggregating cost or usage by model reads -- so leaving it
        meant the manifest confidently attributed a whole session's tokens to a
        model that was never called. The reported value is kept alongside
        rather than discarded, because it is evidence of what the SDK was told.
        """
        if not isinstance(payload, dict):
            return
        entries = payload.get("model_usage")
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("type") or "").replace("_usage", "")
            resolved = self._model_overrides.get(kind) or self._sniffed_models.get(kind)
            reported = entry.get("model")
            if not resolved or resolved == reported:
                continue
            entry["model"] = resolved
            if reported:
                entry["reported_model"] = reported

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
        state = self._state_for(getattr(metrics, "speech_id", None), stage="llm")
        if state is None:
            return
        duration = _positive_ms(getattr(metrics, "duration", 0))
        started_at, ended_at = _back_dated(self.call.now(), duration)
        operation = state.turn.start_operation(
            type="llm",
            endpoint_id=LLM_ENDPOINT_ID,
            provider=_provider(metrics),
            model=self._model_for("llm", metrics),
            transport="http",
            started_at_ms=started_at,
            request=_present(
                request_id=getattr(metrics, "request_id", None),
                # Keep what the plugin claimed when we overrode it, so the
                # correction is auditable rather than a silent substitution.
                reported_model=_reported_model(_model(metrics),
                                               self._model_for("llm", metrics)),
            ),
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
        state = self._state_for(getattr(metrics, "speech_id", None), stage="tts")
        if state is None:
            return
        duration = _positive_ms(getattr(metrics, "duration", 0))
        started_at, ended_at = _back_dated(self.call.now(), duration)
        operation = state.tts
        if operation is not None and operation.ended:
            # This turn's reply was already published, and a metric for it has
            # only now arrived. Opening a second span here would record the
            # reply twice: `_end_tts` attributes the turn's *whole* byte count
            # to whatever span it closes, so the same audio would be reported
            # under both -- doubling talk time and cost -- and the second one
            # would close as `cancelled`, which the dashboard counts as a
            # barge-in on a turn that was never interrupted.
            #
            # Applies to measured spans as well as derived ones. A genuinely
            # multi-segment turn loses the later segments' provider counts,
            # which is a real cost, but it is a gap rather than a number that
            # is confidently wrong in the direction that flatters the product.
            self._warn_once(
                "tts-late-metrics",
                "vaani: tts_metrics arrived after this turn's reply was "
                "closed; the metric is discarded rather than recorded as a "
                "second reply, which would double the turn's reported talk "
                "time. Later segments of a multi-segment reply are not "
                "separately reported.",
            )
            self._late_metric_turns.add(state.id)
            return
        if operation is None:
            operation = state.turn.start_operation(
                type="tts",
                endpoint_id=TTS_ENDPOINT_ID,
                provider=_provider(metrics),
                model=self._model_for("tts", metrics),
                transport="websocket" if getattr(metrics, "streamed", False) else "http",
                started_at_ms=started_at,
                request={"sample_rate_hz": self._rate_for("agent")},
            )
            state.tts = operation
            state.tts_derived = False
        elif state.tts_derived:
            # The measurement arrived while the estimate was still open, so the
            # span stops being an estimate: the provider's own numbers replace
            # the reconstructed ones below.
            state.tts_derived = False
            # `tts_was_derived` is permanent. `tts_derived` means "still an
            # estimate", and clearing it is what lets the provider's numbers
            # replace the reconstructed ones -- but the span's provider, model
            # and start time were fixed when it was opened and stay
            # reconstructed forever. Counting it as fully measured would drop
            # it out of `derived_tts_op_count` and out of the dashboard's
            # estimate warning, which is the disclosure the whole reconstruction
            # story rests on.
            state.tts_was_derived = True
            state.tts_response.pop("audio_ms", None)
            # `estimated` deliberately survives. The provider's timings replace
            # the reconstructed ones below, but `provider`, `model`,
            # `transport`, `started_at_ms` and `derived_from` were fixed when
            # the span was opened and cannot be rewritten -- so clearing the
            # flag would present a span still carrying reconstructed identity
            # and a reconstructed start as fully measured.
            state.tts_response["estimated_fields"] = "provider,model,started_at"
            operation.event("measured_metrics_arrived")
        operation.event("speak", _present(char_count=getattr(metrics, "characters_count", None)))
        # Frames already synthesized for this reply were counted before this
        # span existed; stamp their timing on now that there is somewhere to put it.
        self._mark_first_audio(state)
        ttfb = _ms(getattr(metrics, "ttfb", None))
        if ttfb is not None and ttfb >= 0:
            operation.event("first_byte", occurred_at_ms=started_at + ttfb)
        state.tts_ended_at = ended_at
        # One reply can be synthesized as several segments, and LiveKit's own
        # usage collector sums every `TTSMetrics` it sees -- these are additive
        # measurements, not restatements. Overwriting them reported the last
        # segment as though it were the whole reply: two segments totalling
        # 3000ms and 30 characters were published as 2000ms and 20, so both the
        # provider duration and the billable character count came out low
        # behind a healthy status. Undercounting what a customer is charged for
        # is the least forgivable direction for this number to be wrong in.
        segment_audio_ms = _ms(getattr(metrics, "audio_duration", None))
        if segment_audio_ms is not None:
            state.tts_response["audio_ms"] = (
                (state.tts_response.get("audio_ms") or 0) + segment_audio_ms
            )
        characters = getattr(metrics, "characters_count", None)
        if isinstance(characters, (int, float)):
            state.tts_response["characters_count"] = (
                (state.tts_response.get("characters_count") or 0) + characters
            )
        segment_id = getattr(metrics, "segment_id", None)
        if segment_id is not None:
            # Every segment named, in order, rather than only whichever
            # happened to report last.
            seen = state.tts_response.get("segment_id")
            state.tts_response["segment_id"] = (
                segment_id if not seen
                else (seen if segment_id in str(seen).split(",")
                      else f"{seen},{segment_id}")
            )
            state.tts_response["segment_count"] = len(
                str(state.tts_response["segment_id"]).split(",")
            )
        if ttfb is not None:
            # The first segment's time to first byte is the reply's: later
            # segments start while the caller is already listening.
            state.tts_response.setdefault("ttfb_ms", ttfb)
        # Any segment reporting a cancellation cancels the reply.
        state.tts_response["cancelled"] = bool(
            state.tts_response.get("cancelled")
        ) or bool(getattr(metrics, "cancelled", False))

    def _record_stt_metrics(self, metrics: Any) -> None:
        """STT metrics arrive without a speech id, so they decorate the pending span."""
        self._stt_identity = _present(
            provider=_provider(metrics),
            model=self._model_for("stt", metrics),
            streamed=getattr(metrics, "streamed", None),
        )
        self._pending_stt.metrics = {
            "audio_ms": _ms(getattr(metrics, "audio_duration", None)),
            **self._stt_identity,
        }

    def _record_eou(self, metrics: Any) -> None:
        # User-scoped: an end-of-utterance measurement describes the caller's
        # turn, not a reply, so it legitimately carries no reply identity.
        state = self._state_for(getattr(metrics, "speech_id", None), stage="eou")
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
        if role == "assistant":
            # Identity first: the speech handle that produced this item is the
            # authoritative answer, and it is available on every path that
            # emits one.
            state = self._turn_owning_item(item)
            resolved_exactly = state is not None
            if state is None:
                state, resolved_exactly = self._reply_turn(
                    (getattr(item, "text_content", None) or "").strip())
        else:
            state, resolved_exactly = self._current_turn, True
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
        # Derive the TTS span *before* folding in the report below: on a plugin
        # that emits no `tts_metrics` this is the only thing that will ever
        # create one, and without it the report -- and the agent's own words --
        # have nowhere to land and are discarded.
        self._derive_tts(state, metrics, text, item, resolved_exactly)
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
            # What the agent actually said. `text_content` here is LiveKit's
            # `forwarded_text` -- the words that reached the caller, not the
            # words the LLM produced -- so on an interrupted reply it is the
            # truthful record of what was heard.
            #
            # The character count is recorded whether or not content capture is
            # on, because "we saw 179 characters go by" is a fact worth keeping
            # even under a policy that forbids storing them.
            if text:
                if self._capture_transcripts:
                    state.tts_response["text"] = text
                state.tts_response["char_count"] = len(text)
            # LiveKit knows whether the reply was cut off; the TTS plugin's own
            # `cancelled` flag does not survive every interruption path.
            if getattr(item, "interrupted", None):
                state.tts_response["interrupted"] = True
        # Independent of the TTS span: an agent can emit tts_metrics while its
        # LLM emits none, and that turn still deserves an LLM span.
        self._derive_llm(state, metrics, resolved_exactly)
        # Deliberately *not* `_finish_turn`. `conversation_item_added` fires
        # when the reply's text is committed, not when the caller has finished
        # hearing it -- measured at 0.6s into a 9s reply. Closing the turn here
        # ended the TTS span almost immediately and, worse, left every
        # subsequent frame of that reply belonging to no open turn at all: a
        # 49s call attributed 6.5s of the 33s the agent actually spoke.
        #
        # The turn is instead marked complete and retired when the reply is
        # superseded (`speech_created`) or the call ends, by which time the
        # whole reply has been rendered and can be measured.
        state.reply_complete = True
        if state.awaiting_reply_item:
            # The words this span was being held open for have arrived, so it
            # can be closed with them on it.
            state.awaiting_reply_item = False
            if state.open_streams:
                # The text commits well before playout ends -- measured at
                # ~0.6s into a 9s reply on a live call -- so closing here
                # would cut the span short by most of the reply.
                state.awaiting_stream_close = True
                return
            self._finish_turn(state)

    def _retire_reply(self, keep: Optional[_TurnState] = None) -> None:
        """Close the reply that is no longer the one being spoken.

        Called when a new speech handle opens and at teardown. A turn whose
        text is committed but whose audio is still draining is left alone, so
        the frames still arriving are attributed to the reply that produced
        them.
        """
        state = self._speaking_turn
        if state is None or state is keep or state.finished:
            return
        # NOTE: since deferral, `_retired_reply` may point at a turn that is
        # still *open*. It means "the reply that was just superseded", not "a
        # closed turn", and anything added later that assumes the latter will
        # write to a live span.
        self._retired_reply = state
        if state.audio_bytes and not state.reply_complete:
            # This reply rendered audio but LiveKit has not committed its text
            # yet. Closing the span now makes its transcript unrecoverable: the
            # operation is already emitted by the time the item arrives, so the
            # agent's own words -- the audit's second headline defect -- are
            # dropped for exactly the replies most likely to matter, the ones
            # the caller interrupted.
            #
            # Retiring is only about *attribution*: frames now follow
            # `_speaking_turn`, which the caller reassigns immediately after
            # this, so leaving the span open costs nothing in accuracy. The
            # byte count is already frozen, so `ended_at` is computed from the
            # same audio whenever the close finally happens, and `Turn.end()`
            # records no timestamp at all. `finalize_open_spans` closes
            # anything still open at the end of the call, so nothing can leak.
            state.awaiting_reply_item = True
            if state.open_streams:
                state.awaiting_stream_close = True
            return
        if state.open_streams:
            # Its own generator is still rendering: the caller is still
            # hearing this reply.
            state.awaiting_stream_close = True
            return
        self._finish_turn(state)

    def _turn_owning_item(self, item: Any) -> "Optional[_TurnState]":
        """The turn whose speech handle actually produced this item.

        LiveKit appends the item to `SpeechHandle.chat_items` before it emits
        `conversation_item_added`, so the speech that made a reply can be
        identified rather than inferred. Everything else here is a heuristic
        over timing, and every heuristic this file has tried has been broken by
        a real interleaving: a `say()` queued behind an active reply commits
        its text before its first frame, so "the new speech has not rendered
        yet" points at exactly the wrong turn; and a reply retired with its
        words already taped fails a "does it have text yet" test. Identity has
        no such failure mode.
        """
        item_id = getattr(item, "id", None)
        if item_id is None:
            return None
        for state in self._all_turns:
            handle = state.speech_handle
            if handle is None:
                continue
            try:
                items = list(getattr(handle, "chat_items", None) or ())
            except Exception:  # noqa: BLE001 - a handle is not ours to trust
                continue
            for candidate in items:
                if getattr(candidate, "id", None) == item_id:
                    return state
        return None

    def _reply_turn(self, item_text: str = "") -> "tuple[Optional[_TurnState], bool]":
        """The turn an assistant conversation item belongs to, and how sure we are.

        `conversation_item_added(assistant)` is emitted after the reply has
        finished playing out -- seconds after the LLM metric that opened its
        span. A caller who speaks in that window rotates `_current_turn` onto a
        brand-new turn, so resolving by "current" folds the reply's report, and
        any span derived from it, into a turn that never called an LLM.

        `_speaking_turn` is set from `speech_created` and is an exact answer.
        The event carries no speech id, so when there is no live speaking turn
        the only remaining candidate is `_current_turn` -- returned with
        `False`, because callers that would *create* data from it should not.
        """
        state = self._speaking_turn
        if state is not None and not state.finished:
            prior = self._retired_reply
            if (not state.audio_bytes
                    and prior is not None
                    and prior is not state
                    and prior.audio_bytes
                    and not prior.reply_complete):
                owner = self._claimant(state, prior, item_text)
                if owner is None:
                    # Genuinely undecidable -- see `_claimant`. Both readings
                    # are equally consistent with every event seen, so writing
                    # either one publishes a transcript that is wrong half the
                    # time with nothing on the page to say so. The words are
                    # dropped and the gap is declared instead: a missing
                    # transcript is recoverable by a reader, a confidently
                    # misfiled one is not.
                    self._note_ambiguous_reply(prior, state)
                    return None, False
                if owner is not prior:
                    return owner, True
                # The item arrived in the window between a new speech opening
                # and that speech rendering its first frame, and the reply it
                # superseded has audio but no words yet. Crediting the new
                # speech would give a turn that has not spoken the previous
                # reply's transcript and a `reply_complete` it has not earned
                # -- a fabricated turn next to a mute one, from a single event
                # arriving a few milliseconds late.
                #
                # Only taken when the live speech could have proved the item
                # was its own and did not -- see `_can_disprove_ownership`.
                # Without that negative proof this branch guesses, and because
                # a deferred reply's spans are still open the guess is *written*
                # rather than dropped: a live call would publish the new
                # reply's words on the previous turn. A transcript on the wrong
                # turn is worse than a missing one, because nothing on the page
                # marks it as suspect.
                return prior, True
            return state, True
        return self._current_turn, False

    def _claimant(self, state: "_TurnState", prior: "_TurnState",
                  item_text: str) -> "Optional[_TurnState]":
        """Decide between a superseded reply and the speech that replaced it.

        A new speech that has rendered nothing, and a previous reply that
        rendered audio but has no words yet, produce *event-for-event
        identical* streams whether the arriving item is the old reply's late
        report or the new reply's own -- committing text before the first frame
        is ordinary for `say()` and for non-streaming TTS. Timing cannot
        separate them, so this looks only for actual evidence, in order of
        strength, and returns None when there is none.
        """
        if self._can_disprove_ownership(state):
            # LiveKit appends an assistant item to `SpeechHandle.chat_items`
            # before emitting the session event. A handle that exposes the list
            # and does not contain this item positively did not produce it:
            # negative proof, not inference. This is the live path.
            return prior
        if not item_text:
            return None
        # Second-best evidence, and independent of LiveKit internals: we taped
        # what was handed to `tts_node`. Both candidates are compared, because
        # a new reply can render text before its first frame -- so "the prior
        # tape matches" is only evidence if the current one does not. Matching
        # a prefix of a *longer* current reply ("Sure" against "Sure, one
        # moment") is exactly how a confident wrong answer gets produced.
        matched = [
            candidate for candidate in (prior, state)
            if _text_matches("".join(candidate.tts_text).strip(), item_text)
        ]
        if len(matched) == 1:
            return matched[0]
        return None

    def _note_ambiguous_reply(self, prior: "_TurnState",
                              state: "_TurnState") -> None:
        report = getattr(self.call, "report_coverage_gap", None)
        self._warn_once(
            "reply-attribution-ambiguous",
            "vaani: an assistant message could not be attributed to a "
            "specific reply, so its transcript was dropped rather than "
            "guessed. Mix VaaniAudioTapMixin into your Agent so replies can "
            "be matched by their text.",
        )
        if report is None:
            return
        try:
            report(
                "tts",
                "an assistant message matched no reply and was dropped rather "
                "than attributed to the wrong turn",
                turn_ids=[prior.id, state.id],
            )
        except Exception as error:  # noqa: BLE001 - teardown must not raise
            logger.debug("vaani: cannot record coverage gap (%s)", error)

    def _can_disprove_ownership(self, state: "_TurnState") -> bool:
        """Whether this turn's speech could have claimed an item and did not.

        LiveKit appends an assistant item to `SpeechHandle.chat_items` before
        emitting the session event, so when a handle exposes that list, an item
        missing from it is positively *not* that speech's -- a negative proof,
        not an inference. When the handle exposes no such list there is nothing
        to conclude from, and a reply's own item committing before its first
        frame is ordinary for `say()` and for non-streaming TTS, so the two
        cases are indistinguishable by timing alone. In that situation the
        speech that is actually rendering keeps its own report.
        """
        handle = getattr(state, "speech_handle", None)
        if isinstance(getattr(handle, "chat_items", None), (list, tuple)):
            return True
        if handle is not None:
            # Naming the real cause matters: the fallback's own warning tells
            # the operator to install the audio tap, which is sound advice for
            # a call that has no other evidence but is not why *this* one lost
            # its proof.
            self._warn_once(
                "speech-handle-no-chat-items",
                "vaani: this LiveKit version's SpeechHandle exposes no "
                "chat_items, so replies cannot be matched to their speech by "
                "identity and interrupted replies may be reported as gaps. "
                "Upgrade to livekit-agents>=1.2.",
            )
        return False

    def _derive_llm(self, state: Any, metrics: Mapping[str, Any],
                    resolved_exactly: bool = True) -> None:
        """Reconstruct an LLM span when `metrics_collected` never delivered one.

        `_record_llm` runs from `metrics_collected` and nowhere else, so an
        agent whose LLM plugin emits no metrics -- a custom `llm_node`, a
        provider without a metrics implementation, a version where the event
        moved -- previously produced a package with zero LLM spans and no
        indication anything was missing. `conversation_item_added` carries the
        session's own per-turn LLM timings, which is less than the plugin's
        report but far more than nothing.

        The span is explicitly marked derived: an approximate number a reader
        knows is approximate is useful, while one they believe is measured is
        worse than a gap.
        """
        if state.finished:
            # The turn is closed and its `turn.end()` has already run. Opening
            # a span on it now attaches this reply's latency to a turn that had
            # finished reporting -- the misattribution the caller was trying to
            # avoid, one frame lower down. When routing sent us here there is
            # nothing left to write to, so the report is dropped, which is what
            # "dropped rather than misfiled" has to mean to be true.
            return
        if state.llm:
            return
        if not resolved_exactly and state.tts is None:
            # We could not tell which turn this reply belongs to, and the
            # candidate shows no sign of ever having replied. Deriving here
            # would invent a span on a turn that never called an LLM -- and a
            # confidently wrong number is worse than a gap.
            return
        ttft = _ms(metrics.get("llm_node_ttft"))
        ttfs = _ms(metrics.get("llm_node_ttfs"))
        if ttft is None and ttfs is None:
            return
        self._warn_once(
            "llm-derived",
            "vaani: no llm_metrics for this turn; deriving the LLM span from "
            "conversation_item_added. Timings are approximate and token counts "
            "are unavailable. Check that the LLM plugin emits metrics_collected.",
        )
        # `llm_node_ttfs` (first token handed to TTS) is the closest thing to
        # an end that this event exposes; `ttft` is the floor when it is absent.
        duration = max(0, ttfs if ttfs is not None else (ttft or 0))
        started_at, ended_at = _back_dated(self.call.now(), duration)
        operation = state.turn.start_operation(
            type="llm",
            endpoint_id=LLM_ENDPOINT_ID,
            model=self._model_for("llm", None),
            transport="http",
            started_at_ms=started_at,
            request=_present(derived_from="conversation_item_added"),
        )
        if ttft is not None and ttft >= 0:
            operation.event("first_token", occurred_at_ms=started_at + ttft)
        operation.end(
            status="ok",
            response=_present(
                ttft_ms=ttft,
                tokens_per_second=metrics.get("llm_node_tps"),
                # Absent, not zero. Reporting zero tokens would corrupt every
                # cost and throughput aggregate built on top of this package.
                estimated=True,
            ),
            ended_at_ms=ended_at,
        )
        state.llm.append(operation)

    def _derive_tts(self, state: Any, metrics: Mapping[str, Any], text: str,
                    item: Any, resolved_exactly: bool = True,
                    derived_from: str = "conversation_item_added") -> None:
        """Reconstruct a TTS span when `metrics_collected` never delivered one.

        `_record_tts` runs from `metrics_collected` and nowhere else, and a TTS
        plugin only emits that metric when the provider closes the segment it
        is measuring: `livekit.plugins.deepgram` emits on `SpeechMetadata`, and
        a segment ended by an interruption, a socket close, or a provider that
        simply never sends it produces no metric at all. The result was a call
        with real, audible agent speech and zero TTS spans -- and, because the
        turn report and the agent's transcript were only folded into an
        *existing* span, no record of what was said either.

        A voice agent's own speech is the one thing an observability tool for
        voice agents cannot be missing, so it is reconstructed from evidence
        that does not depend on the plugin:

        * `started_speaking_at`/`stopped_speaking_at` -- the session's own
          playout window, measured at the audio output rather than the codec.
        * the frames this recorder taped off `tts_node`, which is a direct
          measurement of what was rendered and is available even when LiveKit
          reports no timings.

        The span is marked derived and estimated for the same reason the LLM
        one is: an approximate number a reader knows is approximate is useful,
        while one they believe was measured is worse than a gap.
        """
        if state.tts is not None:
            return
        if not resolved_exactly and not state.audio_bytes:
            # We could not tell which turn this reply belongs to, and the
            # candidate shows no sign of having spoken. Deriving here would
            # attach speech to a turn that was silent.
            return
        if not state.audio_bytes and not text:
            # Nothing was rendered and nothing was said. There is no reply here
            # to describe, and inventing an empty span would turn a quiet turn
            # into a fabricated one.
            return
        rate = self._rate_for("agent") or 1
        channels = self._channels_for("agent") or 1
        played_ms = _pcm16_ms(state.audio_bytes, rate, channels)
        # Prefer the session's playout window: it spans the whole reply as the
        # caller heard it, including the tail still draining when the last
        # frame was tapped. Both ends are wall clock, so the *difference* is
        # sound even though the values cannot be compared to a monotonic clock.
        spoke_ms = _span_ms(metrics.get("started_speaking_at"),
                            metrics.get("stopped_speaking_at"))
        # Whether the playout window was actually reported decides how much of
        # this span may be treated as measured. Without it the only honest
        # thing to do is leave the duration to `_end_tts`, which runs once the
        # whole reply has drained and can count the bytes that really flowed.
        # A window is only "measured" if the frames we counted corroborate it.
        # `conversation_item_added` fires when the reply's text commits, ~0.6s
        # into a 9s reply, and a window reflecting that instant -- or a
        # zero-width one, which `_span_ms` reports as 0 rather than None --
        # would be published as `audio_ms` and as the span's duration for a
        # reply the caller heard in full. `played_ms` is a direct count of
        # frames that flowed, so a provider window shorter than it is already
        # disproved by evidence in hand.
        measured_window = (
            spoke_ms is not None
            and spoke_ms > 0
            and spoke_ms >= played_ms - _PLAYOUT_TOLERANCE_MS
        )
        if not measured_window:
            spoke_ms = played_ms
        self._warn_once(
            "tts-derived",
            "vaani: no tts_metrics for this turn; deriving the TTS span from "
            "conversation_item_added and captured audio. Timings are "
            "approximate and provider character counts are unavailable. Check "
            "that the TTS plugin emits metrics_collected.",
        )
        # The first tapped frame is this recorder's own measurement of when the
        # caller started hearing the reply, and is preferred over back-dating
        # from "now", which would absorb the delay before this event fired.
        started_at = state.audio_first_at_ms
        ttfb = _ms(metrics.get("tts_node_ttfb"))
        if started_at is None:
            started_at, _ = _back_dated(self.call.now(), spoke_ms)
        elif ttfb is not None and ttfb >= 0:
            # The span must start when the request was made, not when the audio
            # arrived: anchoring it at the first frame made time-to-first-audio
            # come out as exactly 0ms on every derived span, which is not a
            # slightly-wrong number but a physically impossible one.
            started_at = max(0, started_at - ttfb)
        identity = self._component_identity.get("tts", {})
        operation = state.turn.start_operation(
            type="tts",
            endpoint_id=TTS_ENDPOINT_ID,
            provider=identity.get("provider"),
            model=self._model_for("tts", None) or identity.get("model"),
            transport="websocket",
            started_at_ms=started_at,
            request=_present(sample_rate_hz=rate, derived_from=derived_from),
        )
        state.tts = operation
        state.tts_derived = True
        state.tts_ttfa_ms = ttfb if (ttfb is not None and ttfb >= 0) else None
        if measured_window:
            # Measured from the first frame, not from the back-shifted start:
            # `started_at` was moved `ttfb` earlier so time-to-first-audio is
            # not an impossible zero, and adding the playout window to *that*
            # ends the span `ttfb` before the last frame was actually rendered.
            anchor = state.audio_first_at_ms
            if anchor is None:
                anchor = started_at
            state.tts_ended_at = max(started_at, anchor + spoke_ms)
        # Stamp the first-audio milestone now that there is a span to hold it.
        self._mark_first_audio(state)
        if ttfb is not None and ttfb >= 0:
            operation.event("first_byte", occurred_at_ms=started_at + ttfb)
        state.tts_response.update(_present(
            # Only claim a synthesized duration when the session actually
            # reported the playout window. Otherwise `_end_tts` fills it from
            # the bytes that reached the caller, once they all have.
            audio_ms=spoke_ms if measured_window else None,
            ttfb_ms=ttfb,
            # Absent, not zero: the provider never told us how many characters
            # it billed for, and a zero would silently deflate cost aggregates.
            estimated=True,
        ))

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
        state.seq = self._turn_counter
        self._turns[turn_id] = state
        self._all_turns.append(state)
        return state

    def _state_for(self, speech_id: Optional[str], *,
                   stage: str) -> Optional[_TurnState]:
        """Resolve the turn a metric belongs to, adopting the pending one once."""
        if self.call is None:
            return None
        if speech_id is None:
            # Deliberately *not* recovered from the speech-handle context here.
            # That context proves ownership inside `tts_node`, which runs as
            # part of the speech; a metric is delivered on the session's event
            # loop, where whatever handle happens to be current belongs to
            # whichever reply is speaking now -- during a barge-in, the wrong
            # one. Reusing it would dress a guess up as identity.
            return self._unidentified_metric_turn(stage)
        existing = self._turns.get(speech_id)
        if existing is not None:
            self._current_turn = existing
            return existing
        state = self._pending_turn or self._new_turn()
        self._pending_turn = None
        self._turns[speech_id] = state
        self._current_turn = state
        return state

    def _could_emit(self, state: "_TurnState", stage: str) -> bool:
        """Whether this turn could be the source of a `stage` measurement.

        Two exclusions, both provable from LiveKit's own events rather than
        from timing -- which is the whole point, since timing is exactly what
        this code refuses to rely on:

        * A turn with no `reply_source` never had a speech created for it, so
          `tts_node` never ran and no LLM ever answered on its behalf. A
          second final transcript arriving before the reply produces such a
          turn, and it can never be retired, because retirement follows the
          *speaking* turn and this one never speaks. Left in the running it
          became a permanent second live candidate, and one duplicated
          endpoint stopped the call from accepting another provider metric
          for the rest of its life.
        * `session.say()` speaks text it was handed. No LLM runs for it, so it
          cannot be the claimant for an LLM measurement -- yet the opening
          greeting is retired by the very speech that creates the first real
          reply, so its `finished_at_seq` always lands at-or-after that
          reply's and it out-argued every unnamed LLM metric on the most
          common shape a LiveKit agent has.

        It still counts as a TTS claimant: it does speak, and a late metric
        from its synthesis is genuinely ambiguous with the next reply's.
        """
        if state.reply_source is None:
            return False
        return not (stage == "llm" and state.reply_source == "say")

    def _unidentified_metric_turn(self, stage: str) -> "Optional[_TurnState]":
        """Where a provider metric with no reply identity may be published.

        `speech_id` is `str | None` by contract upstream, and a metric that
        omits it -- with nothing in the speech context either -- carries no
        proof of ownership at all. Publishing it on `_current_turn` is the
        same mistake every other attribution path here has been corrected for:
        during a barge-in the current turn is the *new* one, so the previous
        reply's provider, model, character count and TTFB are published as a
        fully *measured* span on a reply that never produced them -- while the
        reply that did is downgraded to reconstructed. Cost is billed off
        `characters_count`, so this is a wrong number wearing the badge of a
        measured one.

        The rule is stage-specific because the stages mean different things:

        * `stt` and `eou` describe the *caller's* utterance. They have no reply
          to name and never did, so the current turn is their correct home and
          refusing them would delete measurements that were never ambiguous.
        * `llm` and `tts` describe one reply. For these the turn must be
          provable, and "it is the only turn still open" is not proof: a reply
          that finished after this one began is an equally good candidate for a
          late metric, which is exactly how a barge-in moves a span onto the
          reply that interrupted it.

        When the reply cannot be established the metric is dropped and the loss
        is disclosed, which costs a provider measurement and keeps the numbers
        that remain true.
        """
        if stage not in _REPLY_SCOPED_STAGES:
            return self._current_turn
        # Only turns that could have produced this stage are in the running --
        # as claimants and, just as importantly, as rivals. A turn that cannot
        # emit the measurement is not evidence of ambiguity, and counting it
        # as one refuses metrics that were never in doubt.
        claimants = [s for s in self._all_turns
                     if self._could_emit(s, stage)]  # marker:r9-eligible
        live = [s for s in claimants if not s.finished]
        candidate = live[0] if len(live) == 1 else None
        if candidate is not None and any(
                t is not candidate and t.finished
                and t.finished_at_seq >= candidate.seq
                for t in claimants):  # marker:r8-no-tolerance
            # A reply ended at or after this one started, so a metric arriving
            # now is at least as likely to be its trailing measurement as this
            # turn's. There is no unambiguous answer -- only a coin flip that
            # would be published as fact.
            #
            # "It already received a metric" is not the exemption it looks
            # like. `llm` and `tts` metrics are *additive*: one reply emits one
            # per synthesis segment or per tool-call round, and `_record_tts`
            # sums them. Nor is "its measurements already account for its
            # audio": a tolerance wide enough to absorb ordinary provider
            # disagreement is also wide enough to hide a short segment, and a
            # barged-in reply reports more synthesized audio than it played, so
            # it looks complete no matter how much is missing.
            #
            # The cost is real -- the reply loses a provider character count,
            # and cost is billed off that -- but the span survives, rebuilt
            # from the tape and marked estimated, and the gap says what was
            # lost. The alternative is a character count on the wrong reply,
            # which is a wrong invoice that looks exactly like a right one.
            #
            # Worth being clear about when this happens at all: LiveKit stamps
            # `speech_id` from its own speech-handle context, so a metric
            # arrives unnamed only when the framework that owns the speech
            # could not name it either. A guess made from timing is not better
            # information than the one the framework declined to give.
            candidate = None
        if candidate is not None:
            return candidate
        if not live and self._current_turn is not None:
            # Nothing is open: this is a trailing metric for the reply that
            # just ended, and `_record_tts`/`_record_llm` already recognise and
            # disclose a metric that arrives after its span was published.
            return self._current_turn
        self._unidentified_metrics[stage] = self._unidentified_metrics.get(stage, 0) + 1
        self._warn_once(
            "unidentified_metric_%s" % stage,
            "vaani: a %s metric arrived with no speech_id and no speech "
            "context while %d replies were in flight; it was dropped rather "
            "than published on a turn that may not have produced it",
            stage,
            len(live),
        )
        return None

    def _end_stt(self, state: _TurnState, status: str) -> None:
        if state.stt is None or state.stt.ended:
            return
        state.stt.end(
            status=status,
            response=state.stt_response,
            ended_at_ms=state.stt_ended_at,
        )

    def _end_tts(self, state: _TurnState, status: str) -> None:
        if (self._capture_transcripts and state.tts_text
                and not state.tts_response.get("text")):
            # Nothing announced this reply's words, but we taped them off
            # `tts_node` on their way to the provider. Marked with their source
            # so a reader can tell this is what we asked to be spoken, not the
            # `forwarded_text` that proves what the caller actually heard.
            text = "".join(state.tts_text).strip()
            if text:
                state.tts_response["text"] = text
                state.tts_response["char_count"] = len(text)
                state.tts_response["text_source"] = "tts_node"
        if (state.tts is None and state.audio_bytes
                and _pcm16_ms(state.audio_bytes, self._rate_for("agent") or 1,
                              self._channels_for("agent") or 1)
                > _PLAYOUT_TOLERANCE_MS):
            # Above the floor, because the tail of a reply still draining when
            # the caller barged in lands on the *next* turn, and deriving a
            # span from it would report an 80ms "reply" on a turn where the
            # agent never spoke. Those spans are worse than nothing: they
            # inflate the denominator of every TTS rate, and the ones that
            # close as `cancelled` are counted by the dashboard as barge-ins,
            # re-inflating the exact metric this round was asked to fix.
            #
            # A reply cut off before its text was committed, or one generated
            # before the first user turn, emits neither `TTSMetrics` nor
            # `conversation_item_added` -- so nothing ever opened a span for
            # it, while the caller demonstrably heard it because we taped the
            # frames. Left alone this is the audit's headline defect in
            # miniature: real agent speech, no TTS operation, and every latency
            # and cost number computed as though it never happened.
            #
            # Derived here rather than in `_finish_turn` because a turn still
            # open when the call ends is closed by `finalize_open_spans`
            # instead, and the last reply of a call is exactly the one most
            # likely to be cut off.
            self._derive_tts(state, {}, "", None,
                             derived_from="captured_agent_audio")
            state.tts_reconstructed = state.tts is not None
        if state.tts is None or state.tts.ended:
            return
        rate = self._rate_for("agent") or 1
        response = dict(state.tts_response)
        # `audio_bytes` is what was rendered, and `played_ms` is that same
        # quantity expressed as time -- the two are always the same
        # measurement, so they are always derived from one another. `audio_ms`
        # is the provider's separate claim about what it synthesized.
        response.setdefault("audio_bytes", state.audio_bytes)
        response["audio_bytes_sample_rate_hz"] = rate
        # `audio_ms` and the taped bytes measure two different things, and
        # conflating them is how a reader ends up trusting a number that is not
        # what they think it is:
        #
        #   audio_ms  -- what the TTS provider says it synthesized.
        #   played_ms -- what actually flowed through `tts_node` to the caller.
        #
        # On an interrupted reply the second is legitimately smaller, and that
        # gap is the useful part: it is how much of the answer the caller never
        # heard. Reporting only one of them, or silently overwriting one with
        # the other, throws that away.
        played_ms = _pcm16_ms(state.audio_bytes, rate, self._channels_for("agent") or 1)
        if state.audio_bytes:
            response.setdefault("played_ms", played_ms)
        # Only stand in for the provider when it reported nothing at all.
        response.setdefault("audio_ms", played_ms)
        # A reply is only "cancelled" if the caller was actually cut short. The
        # TTS plugin raises its own flag when the *synthesis stream* is torn
        # down, which happens routinely at the clean end of a reply, so trusting
        # it alone reported healthy turns as cancelled. LiveKit's own
        # `ChatMessage.interrupted` is the authoritative signal, and rendered
        # audio falling short of synthesized audio is the corroborating one.
        plugin_cancelled = bool(response.pop("cancelled", False))
        interrupted = bool(response.pop("interrupted", False))
        audio_ms = response.get("audio_ms")
        truncated = (
            isinstance(audio_ms, (int, float))
            and state.audio_bytes > 0
            and played_ms < audio_ms - _PLAYOUT_TOLERANCE_MS
        )
        if status == "ok" and (interrupted or (plugin_cancelled and truncated)):
            status = "cancelled"
        elif (
            status == "ok"
            and plugin_cancelled
            and state.audio_bytes == 0
            and self._agent_audio_bytes > 0
        ):
            # Nothing at all reached the caller. That is the *most* complete
            # interruption there is, but `truncated` cannot see it because it
            # needs rendered audio to compare against. The `audio_bytes > 0`
            # requirement exists only to avoid trusting the plugin when no tap
            # is installed -- and audio tapped elsewhere on this call proves the
            # tap works, so the plugin is believed here.
            status = "cancelled"
            response["played_ms"] = 0
        if (isinstance(audio_ms, (int, float)) and state.audio_bytes > 0
                and played_ms > audio_ms + _PLAYOUT_TOLERANCE_MS):
            # The caller heard more than the provider says it synthesized. This
            # is the multi-segment case where only one segment's `TTSMetrics`
            # arrived: on a live call a 5670ms reply carried `audio_ms: 3040`,
            # a 46% undercount of the billable quantity.
            #
            # Both numbers stay -- `audio_ms` is still what the invoice will be
            # based on -- but the disagreement is published rather than left for
            # a reader to notice by dividing two fields. Silence here is how a
            # cost dashboard ends up confidently low, which is the failure
            # direction that flatters the product and so the one to be loudest
            # about.
            response["provider_audio_ms_undercount_ms"] = int(played_ms - audio_ms)
            self._warn_once(
                "tts-provider-undercount",
                "vaani: the TTS provider reported less synthesized audio than "
                "was actually rendered to the caller (%dms reported vs %dms "
                "played). Provider character and duration counts for this call "
                "understate usage; `played_ms` is measured from the audio "
                "itself and is the reliable figure.",
                int(audio_ms), int(played_ms),
            )
        if plugin_cancelled and not (interrupted or truncated) and status != "cancelled":
            # Keep the disagreement rather than hiding it: the reply played to
            # completion, so it is reported as such, but the fact that the
            # plugin said otherwise is the sort of thing an operator chasing a
            # provider bug needs to be able to see.
            response["provider_reported_cancelled"] = True
        ended_at = state.tts_ended_at
        if state.tts_derived and state.audio_first_at_ms is not None:
            # A derived span is closed when its reply is superseded, which can
            # be long after the caller stopped hearing it -- at the end of a
            # call, arbitrarily so. Ending it at the extent of the audio that
            # actually played keeps the timeline bar the length of the reply
            # instead of the length of the silence that followed it.
            #
            # This *overrides* any window carried on `tts_ended_at`, because
            # for a derived span that window came from `conversation_item_added`
            # -- which fires when the reply's text commits, measured at ~0.6s
            # into a 9s reply, not when the caller stopped hearing it. Its
            # `stopped_speaking_at` therefore describes only the fraction that
            # had played by then. On a live call this closed three of four
            # replies at roughly half their true length: an 8.7s answer
            # published as a 4.4s span, with every synthesis-rate and
            # cost-per-second figure computed from it wrong by the same factor,
            # and nothing on the page to say so.
            #
            # The frames are a direct measurement of what flowed to the caller
            # and are counted until the reply is superseded, so they are both
            # the later and the better evidence. An interrupted reply is
            # correctly *shortened* by this too: the caller only heard the
            # frames that were rendered.
            if state.audio_bytes or ended_at is None:
                ended_at = state.audio_first_at_ms + played_ms
        elif state.audio_first_at_ms is not None and state.audio_bytes:
            # A *measured* span closes when `TTSMetrics` arrives, which is when
            # synthesis finished -- and Deepgram synthesizes far faster than
            # realtime, so the caller is still listening well after that. On a
            # live call this published a 3080ms greeting as a 2318ms span.
            #
            # A span whose duration is shorter than the audio it says it played
            # is not defensible on a page an operator uses to answer "how long
            # was my agent talking": the timeline bar contradicts the number
            # inside it, and anything integrating duration undercounts. Extend
            # to cover the playout, never shorten -- an interrupted reply's
            # metrics-arrival end is already the later of the two and stays.
            playout_end = state.audio_first_at_ms + played_ms
            if ended_at is None or playout_end > ended_at:
                ended_at = playout_end
                response["ended_at_source"] = "played_audio"
        # What this span actually published, kept so the coverage audit can
        # reconcile the emitted spans against the tape rather than against the
        # recorder's own bookkeeping -- which would let a double-counted or
        # fabricated span pass unnoticed, because it is the very thing under
        # audit that decides whether a turn counts as attributed.
        state.published_played_ms = response.get("played_ms") or 0
        state.unscoped_audio_bytes_at_publish = state.unscoped_audio_bytes
        state.publication_snapshot_taken = True  # marker:r8-publish-flag
        state.tts.end(status=status, response=response, ended_at_ms=ended_at)

    def _finish_turn(self, state: _TurnState) -> None:
        if state.finished:
            return
        state.finished = True
        state.finished_at_seq = self._turn_counter
        self._end_stt(state, "ok")
        self._end_tts(state, "ok")
        state.turn.end()
        if self._current_turn is state:
            self._current_turn = None


def _tap_is_active(agent: Any) -> bool:
    """Whether the mixin's tapping nodes are the ones Python will actually call.

    Class membership only proves the mixin is *somewhere* in the bases. With
    `class Wrong(Agent, VaaniAudioTapMixin)` the framework's own `tts_node`
    wins method resolution, every frame bypasses the tap, and an agent that
    talked for a minute measures zero -- reported, without this check, as "your
    agent never spoke".
    """
    try:
        resolved = type(agent).tts_node
        return getattr(resolved, "__func__", resolved) is VaaniAudioTapMixin.tts_node
    except Exception:  # noqa: BLE001 - a wiring check must never break a call
        return False


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
        _warn_untapped(recorder, "stt_node")

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
        _warn_untapped(recorder, "tts_node")
        # One stream per invocation, shared by the words and the frames so both
        # land on the same reply. Without it a `say()` queued mid-reply moves
        # the recorder's global "who is speaking" and the rest of this
        # generator is credited to the new speech.
        stream = recorder.open_output_stream() if recorder is not None else None

        async def tapped():
            async for chunk in text:
                if recorder is not None:
                    recorder.tap_output_text(chunk, stream)
                yield chunk

        try:
            source = super().tts_node(tapped(), model_settings)
            async for frame in _scoped(recorder, source):
                if recorder is not None:
                    recorder.tap_output_frame(frame, stream)
                yield frame
        finally:
            # Reached on interruption too, which is the case that matters: the
            # reply is over exactly when its generator stops, not when the next
            # speech is authorized.
            if recorder is not None:
                recorder.close_output_stream(stream)


def _warn_untapped(recorder: Optional[VaaniLiveKitRecorder], node: str) -> None:
    """Say so the first time audio flows past an unwired tap.

    The mixin is inert without `agent.vaani`, and being inert looks exactly
    like working: the call records, the spans are there, the manifest is
    valid -- and `call.audio` is empty. Warning at wire-up time cannot catch
    this, because the mixin can be installed on an agent that is never given
    a recorder. Warning here, where the frames actually go nowhere, can.
    """
    if recorder is not None:
        return
    warn_once(
        "untapped-audio",
        "vaani: %s is running through VaaniAudioTapMixin but agent.vaani is "
        "not set, so NO audio is being captured. Call "
        "observe_agent_session(session, recorder, agent=agent) or assign "
        "agent.vaani = recorder before the session starts.",
        node,
    )


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
    *,
    agent: Any = None,
    job_ctx: Any = None,
    upload_timeout: Optional[float] = None,
    **options: Any,
) -> VaaniLiveKitRecorder:
    """Attach a recorder to an `AgentSession` and wire its whole lifecycle.

    Pass `agent` and `job_ctx` and the integration is complete — this is the
    supported way to use it, because the two things a caller has to remember
    are exactly the two that fail silently when forgotten:

    * `agent.vaani = recorder`, without which no audio is ever captured.
    * finalizing at *job shutdown*. `AgentSession.start()` returns when the
      session starts, not when the call ends, so finalizing after it truncates
      the recording mid-call.
    """
    recorder = recorder or VaaniLiveKitRecorder.from_env(**options)
    recorder.attach(session)
    if agent is not None:
        agent.vaani = recorder
        recorder.note_audio_tap_installed(agent)
    elif recorder.enabled:
        # Audio is the single most expensive thing to lose and the easiest to
        # forget, because nothing else about the recording looks wrong.
        recorder._warn_once(
            "no-agent",
            "recorder is enabled but no agent was bound — pass agent=... to "
            "observe_agent_session() or set agent.vaani, or no audio is captured",
        )
    if job_ctx is not None and recorder.enabled:
        add = getattr(job_ctx, "add_shutdown_callback", None)
        if add is None:
            recorder._warn_once(
                "no-shutdown-hook",
                "job context has no add_shutdown_callback; call finish() yourself",
            )
        else:
            # LiveKit inspects the callback's arity and passes the shutdown
            # *reason* as the first positional argument when it accepts one.
            # `recorder.finish` accepts one — so registering it directly would
            # feed a raw LiveKit reason string in as the package `outcome`,
            # which is a closed vocabulary. Keep the reason as metadata and let
            # the close event decide the outcome.
            async def _finish_at_shutdown(reason: str = "") -> None:
                if reason and recorder.call is not None:
                    recorder.call.metadata.setdefault("shutdown_reason", str(reason))
                await recorder.finish(timeout=upload_timeout)

            add(_finish_at_shutdown)
    return recorder


# --------------------------------------------------------------- small helpers


def _guard(recorder: "VaaniLiveKitRecorder", name: str, handler: Any) -> Any:
    """Wrap an event handler so it can neither kill the call nor outlive it.

    Two separate jobs:

    * Handlers run on LiveKit's loop, so a raised error would take the call
      down. Recording is best effort and must never do that.
    * Handlers outlive the recording. `finish()` releases the call while the
      session is still emitting — a job shutdown hook races the session's own
      teardown, and `off()` is not atomic with respect to events LiveKit has
      already dispatched. A handler that fires after `finish()` is therefore a
      clean no-op rather than an `AttributeError` on a `None` call.
    """

    def wrapped(event: Any) -> None:
        if recorder.call is None:
            return
        try:
            handler(event)
        except Exception as error:  # noqa: BLE001 - recording is best effort
            recorder._warn_once(
                f"handler:{name}", "handler %s failed (%s)", name, error
            )

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


def _sniff_model(component: Any) -> Optional[str]:
    """The model a provider is really using, when its label disagrees.

    `openai.LLM.with_azure(azure_deployment="gpt-5-mini")` leaves the plugin's
    `model` option at its `"gpt-4o"` default and hands the deployment to the
    Azure client instead, so the metric reports `gpt-4o`. The deployment name
    is read back off the client because it is the only place the truth exists.

    Deliberately private-attribute access, and deliberately best effort: a
    wrong-but-confident model label corrupts every per-model cost and latency
    number downstream, so it is worth reaching for. If the plugin changes shape
    this returns `None` and the reported label is used unchanged.
    """
    client = getattr(component, "_client", None)
    deployment = getattr(client, "_azure_deployment", None)
    if isinstance(deployment, str) and deployment.strip():
        return deployment.strip()
    return None


def _component_identity(component: Any) -> Dict[str, str]:
    """The provider and model a plugin reports about itself.

    `TTS`/`STT`/`LLM` expose `provider` and `model` properties, which is the
    only identity available for a span that had to be derived because the
    plugin emitted no metric to carry one. Best effort by design: a plugin that
    raises from these properties yields no identity rather than no recording.
    """
    identity: Dict[str, str] = {}
    for field in ("provider", "model"):
        value = getattr(component, field, None)
        if isinstance(value, str) and value.strip():
            identity[field] = value.strip()
    return identity


def _reported_model(reported: Optional[str], resolved: Optional[str]) -> Optional[str]:
    """Keep the provider's own label only when we replaced it."""
    if reported and resolved and reported != resolved:
        return reported
    return None


def _mark_delivered(finalized: Any, response: Any, observer: Any = None) -> None:
    """Record that this package has already been shipped, and to where.

    `drain` treats a receipt as "nothing to do here". Without one, a drain
    sidecar -- which the docs recommend running alongside the worker -- cannot
    tell a package that was just uploaded in-process from one that was never
    uploaded at all, and re-ships every recording it sees.

    The endpoint matters because this is where receipts actually come from in
    production: almost every package is shipped in-process by `finish()`, and
    the drain only sees the leftovers. A receipt with no destination recorded
    would let a spool that moved backends look fully delivered to the new one.

    The observer is read here rather than at the call site so that a custom or
    stubbed observer without `options` cannot make a *successful* upload look
    like a failed one -- bookkeeping must never be able to fail the call.
    """
    try:
        from ..drain import _acknowledge

        options = getattr(observer, "options", None)
        endpoint = None
        if isinstance(options, Mapping):
            endpoint = options.get("endpoint")
        else:
            endpoint = getattr(options, "endpoint", None)
        _acknowledge(finalized, response, purge=False, endpoint=endpoint)
        # Seed the spool's destination ledger here too. `drain` was the only
        # writer, but in production almost every package is shipped in-process
        # by `finish()` and the drain only ever sees leftovers -- so a busy,
        # healthy worker built an empty ledger. `_endpoint_change()` then had
        # no known destination to compare against, and the guardrail that is
        # supposed to stop a typo'd endpoint from receiving a spool full of raw
        # call audio waved it through without asking.
        from .._spool import remember_destination

        spool = os.path.dirname(os.path.abspath(finalized.directory))
        remember_destination(spool, endpoint)
    except Exception as error:  # noqa: BLE001 - bookkeeping, never a call failure
        logger.debug("vaani: could not write the upload receipt — %s", error)


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


def _env_endpoints() -> List[Dict[str, Any]]:
    """Connection-capture rules from `VAANI_ENDPOINTS`, a JSON array.

    Each entry is `{"id", "type", "url"}` with an optional `"match"`, e.g.::

        VAANI_ENDPOINTS='[{"id":"deepgram","type":"stt",
                           "url":"wss://api.deepgram.com","match":"origin"}]'

    A malformed value disables connection capture rather than the whole
    recorder, and says so: losing transport spans is recoverable, losing the
    call is not.
    """
    raw = os.environ.get("VAANI_ENDPOINTS", "").strip()
    if not raw:
        return []
    import json

    try:
        parsed = json.loads(raw)
    except ValueError as error:
        logger.error("vaani: VAANI_ENDPOINTS is not valid JSON, ignoring it — %s", error)
        return []
    if not isinstance(parsed, list):
        logger.error("vaani: VAANI_ENDPOINTS must be a JSON array, ignoring it")
        return []
    return parsed
