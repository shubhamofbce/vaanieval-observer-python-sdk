"""The LiveKit Agents integration, driven by fake LiveKit events.

These tests deliberately do not import `livekit.agents`: the recorder only ever
reads attributes off the event objects, so duck-typed stand-ins pin the exact
contract without pulling a heavyweight optional dependency into the suite.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import operations, read_events
from vaani_observer import VaaniObserver
from vaani_observer.integrations.livekit import (
    VaaniAudioTapMixin,
    VaaniLiveKitRecorder,
    observe_agent_session,
)


class FakeAgentSession:
    """The slice of `AgentSession` the recorder touches: `on(name, handler)`."""

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, name: str, handler) -> None:  # noqa: ANN001
        self.handlers.setdefault(name, []).append(handler)

    def off(self, name: str, handler) -> None:  # noqa: ANN001
        registered = self.handlers.get(name, [])
        if handler in registered:
            registered.remove(handler)

    def emit(self, name: str, event: Any) -> None:
        for handler in self.handlers.get(name, []):
            handler(event)


class FakeFrame:
    """An `rtc.AudioFrame` stand-in."""

    def __init__(self, data: bytes, sample_rate: int = 24000, channels: int = 1) -> None:
        self.data = memoryview(data)
        self.sample_rate = sample_rate
        self.num_channels = channels


def agent_frame(duration_ms: int, sample_rate: int = 24000) -> "FakeFrame":
    """A frame of exactly `duration_ms` of 16-bit mono PCM."""
    return FakeFrame(b"\x00" * (duration_ms * sample_rate * 2 // 1000), sample_rate=sample_rate)


def transcript(text: str, is_final: bool, language: str | None = "hi-IN"):
    return SimpleNamespace(
        transcript=text, is_final=is_final, language=language, item_id=None, speaker_id=None
    )


def llm_metrics(speech_id: str, **overrides):
    return SimpleNamespace(
        metrics=SimpleNamespace(
            type="llm_metrics",
            label="azure",
            request_id="req-1",
            duration=overrides.pop("duration", 0.8),
            ttft=overrides.pop("ttft", 0.3),
            cancelled=False,
            completion_tokens=20,
            prompt_tokens=100,
            prompt_cached_tokens=0,
            total_tokens=120,
            tokens_per_second=25.0,
            speech_id=speech_id,
            metadata=SimpleNamespace(model_name="gpt-4o", model_provider="azure"),
            **overrides,
        )
    )


def tts_metrics(speech_id: str, **overrides):
    return SimpleNamespace(
        metrics=SimpleNamespace(
            type="tts_metrics",
            label="sarvam",
            request_id="tts-1",
            ttfb=overrides.pop("ttfb", 0.25),
            duration=overrides.pop("duration", 1.1),
            audio_duration=overrides.pop("audio_duration", 2.0),
            cancelled=overrides.pop("cancelled", False),
            characters_count=42,
            streamed=True,
            segment_id="seg-1",
            speech_id=speech_id,
            metadata=SimpleNamespace(model_name="bulbul:v3", model_provider="sarvam"),
            **overrides,
        )
    )


def stt_metrics(**overrides):
    return SimpleNamespace(
        metrics=SimpleNamespace(
            type="stt_metrics",
            label="deepgram",
            request_id="stt-1",
            duration=0.0,
            audio_duration=overrides.pop("audio_duration", 3.5),
            streamed=True,
            metadata=SimpleNamespace(model_name="nova-3", model_provider="deepgram"),
            **overrides,
        )
    )


_ITEM_SEQ = [0]


def chat_item(role: str, text: str, metrics: dict | None = None, interrupted: bool = False,
              item_id: str | None = None):
    _ITEM_SEQ[0] += 1
    return SimpleNamespace(item=SimpleNamespace(
        id=item_id or f"item-{_ITEM_SEQ[0]}",
        role=role, text_content=text, metrics=metrics or {}, interrupted=interrupted,
    ))


class FakeSpeechHandle:
    """The slice of `SpeechHandle` the recorder reads.

    `chat_items` is how LiveKit reports which items a speech produced, and it
    is populated before `conversation_item_added` is emitted -- which is what
    lets an item be matched to its speech by identity instead of by guessing
    from timing.
    """

    def __init__(self, handle_id: str):
        self.id = handle_id
        self.chat_items: list = []

    def add_done_callback(self, callback):  # pragma: no cover - not used here
        self._done = callback


def speech_created(handle, source: str = "generate_reply"):
    return SimpleNamespace(speech_handle=handle, source=source)


@pytest.fixture
def recorder(tmp_path):
    made: list[VaaniLiveKitRecorder] = []

    def factory(**options):
        observer = VaaniObserver(
            spool_directory=str(tmp_path / f"spool-{len(made)}"),
            instrumentations={"http": False},
        )
        made.append(VaaniLiveKitRecorder(observer, agent_id="trip-planner", **options))
        return made[-1]

    return factory


async def run_one_turn(rec: VaaniLiveKitRecorder, session: FakeAgentSession, speech_id="speech-1"):
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("goa", False))
    session.emit("user_input_transcribed", transcript("goa ki flight", False))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit("user_input_transcribed", transcript("goa ki flight kitne ki hai", True))
    session.emit("conversation_item_added", chat_item("user", "goa ki flight kitne ki hai",
                                                      {"transcription_delay": 0.12}))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id=speech_id), source="generate_reply"),
    )
    session.emit("metrics_collected", llm_metrics(speech_id))
    # A real agent carries the tap, so its frames are always measured. Without
    # this the fixture describes a call nothing could verify, and the audit is
    # right to refuse to call that one healthy.
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics(speech_id))
    session.emit(
        "conversation_item_added",
        chat_item("assistant", "Goa ki flight 6000 rupaye", {"e2e_latency": 1.4,
                                                             "llm_node_ttft": 0.3}),
    )


# --------------------------------------------------------------- one full turn


async def test_records_a_full_turn_as_correlated_stt_llm_and_tts_spans(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    await rec.finish()

    events = operations(read_events(_dir(rec)))
    by_type = {event["type"]: event for event in events}
    assert set(by_type) == {"stt", "llm", "tts"}
    turn_ids = {event["turn_id"] for event in events}
    assert len(turn_ids) == 1, "every stage of one turn must share a turn id"
    assert all(event["scope"] == "turn" for event in events)
    assert all(event["status"] == "ok" for event in events)


def _dir(rec):
    return rec._finalized.directory


def _manifest_of(rec):
    return rec._finalized.manifest


async def _finalized(rec):
    await rec.finish()
    return _manifest_of(rec)


# The recorder does not expose its finalized package, so capture it in finish().
@pytest.fixture(autouse=True)
def _capture_finalized(monkeypatch):
    original = VaaniLiveKitRecorder.finish

    async def finish(self, outcome=None, timeout=None):
        call = self.call
        await original(self, outcome, timeout)
        if call is not None:
            self._finalized = await call.finished

    monkeypatch.setattr(VaaniLiveKitRecorder, "finish", finish)


async def test_the_stt_span_carries_the_dashboard_milestones(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    await rec.finish()

    stt = _by_type(rec, "stt")
    # The dashboard reads these by name to build the utterance timeline.
    for name in ("speech_started", "first_partial", "speech_ended", "final_transcript",
                 "speech_final"):
        assert name in stt["milestones"], name
    assert stt["milestones"]["speech_started"]["occurred_at_ms"] <= stt["started_at_ms"] + 1
    assert stt["response"]["transcript"] == "goa ki flight kitne ki hai"
    assert stt["response"]["language"] == "hi-IN"
    assert stt["response"]["audio_ms"] == 3500
    assert stt["samples"]["partial"]["items"][0]["transcript"] == "goa"
    assert stt["milestones"]["turn_report"]["transcription_delay_ms"] == 120


async def test_the_llm_span_is_back_dated_from_its_metrics(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    await rec.finish()

    llm = _by_type(rec, "llm")
    assert llm["duration_ms"] == 800, "duration must come from the metric, not wall clock"
    assert llm["milestones"]["first_token"]["occurred_at_ms"] == llm["started_at_ms"] + 300
    assert llm["response"]["total_tokens"] == 120
    assert llm["response"]["ttft_ms"] == 300
    assert llm["provider"] == "azure"
    assert llm["model"] == "gpt-4o"


async def test_the_tts_span_carries_audio_accounting_and_the_turn_report(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    await rec.finish()

    tts = _by_type(rec, "tts")
    # 1100ms is how long *synthesis* took; the caller was still listening to
    # the 2000ms it produced. The span covers the reply the caller heard, and
    # the synthesis figure stays reachable through `ttfb_ms` and the
    # milestones, so neither number is lost.
    assert tts["duration_ms"] == 2000
    assert tts["response"]["ended_at_source"] == "played_audio"
    assert tts["milestones"]["first_byte"]["occurred_at_ms"] == tts["started_at_ms"] + 250
    assert tts["response"]["audio_ms"] == 2000
    assert tts["milestones"]["turn_report"]["e2e_latency_ms"] == 1400
    assert tts["response"]["text"] == "Goa ki flight 6000 rupaye"


async def test_a_fully_rendered_reply_is_not_cancelled_just_because_the_plugin_said_so(recorder):
    """The TTS plugin raises `cancelled` when its synthesis stream is torn
    down, which also happens at the clean end of a reply. Trusting it alone
    reported healthy turns as cancelled, so the caller-visible evidence
    decides and the plugin's claim is kept as a separate, visible fact."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1", cancelled=True))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()
    tts = _by_type(rec, "tts")
    assert tts["status"] == "ok"
    assert tts["response"]["provider_reported_cancelled"] is True


async def test_an_interrupted_reply_is_cancelled(recorder):
    """LiveKit's own `ChatMessage.interrupted` is authoritative."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1"))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}, interrupted=True))
    await rec.finish()
    assert _by_type(rec, "tts")["status"] == "cancelled"


async def test_a_reply_cut_short_mid_playout_is_cancelled(recorder):
    """The provider synthesized 2000ms; only 500ms reached the caller."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1", cancelled=True))
    rec.tap_output_frame(agent_frame(500))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()
    tts = _by_type(rec, "tts")
    assert tts["status"] == "cancelled"
    assert tts["response"]["played_ms"] == 500
    assert tts["response"]["audio_ms"] == 2000


async def test_a_reply_is_recorded_even_when_the_tts_plugin_emits_no_metrics(recorder):
    """P0-A. `_record_tts` runs from `metrics_collected` and nowhere else, and a
    TTS plugin only emits that metric when the provider closes the segment --
    Deepgram on `SpeechMetadata`, which an interruption or a socket close can
    skip entirely. Calls with real, audible agent speech recorded zero TTS
    spans and still reported every turn `ok`, so 100% of the agent's talk time
    went missing behind a green check.

    Here the agent speaks 1200ms and LiveKit reports the playout window, but no
    `tts_metrics` ever arrives."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(1200))
    session.emit("conversation_item_added", chat_item(
        "assistant", "Goa ki flight 6000 rupaye",
        {"started_speaking_at": 1000.0, "stopped_speaking_at": 1002.5,
         "tts_node_ttfb": 0.4},
    ))
    await rec.finish()
    tts = _by_type(rec, "tts")
    assert tts is not None, "the agent spoke; a span must exist"
    assert tts["request"]["derived_from"] == "conversation_item_added"
    # The playout window LiveKit measured, not the frames we happened to tape.
    assert tts["response"]["audio_ms"] == 2500
    assert tts["response"]["played_ms"] == 1200
    assert tts["response"]["estimated"] is True
    assert tts["milestones"]["first_byte"]["occurred_at_ms"] == tts["started_at_ms"] + 400


async def test_a_derived_reply_records_what_the_agent_said(recorder):
    """P0-B. The agent's words were only ever written onto an *existing* TTS
    span, so the same missing metric that lost the span silently discarded the
    transcript with it -- leaving the product unable to show what its own agent
    said."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(800))
    session.emit("conversation_item_added",
                 chat_item("assistant", "Namaste, main aapki madad karunga", {}))
    await rec.finish()
    tts = _by_type(rec, "tts")
    assert tts["response"]["text"] == "Namaste, main aapki madad karunga"
    assert tts["response"]["char_count"] == 33


async def test_a_derived_span_is_not_invented_for_a_turn_that_never_spoke(recorder):
    """A derived span must describe observed speech, never fill a gap. No audio
    was rendered and nothing was said, so there is nothing to report."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("conversation_item_added", chat_item("assistant", "", {}))
    await rec.finish()
    assert not _all_of_type(rec, "tts")


async def test_a_real_tts_metric_still_wins_over_the_derived_one(recorder):
    """The fallback must never duplicate or displace a measured span."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1"))
    rec.tap_output_frame(agent_frame(1200))
    session.emit("conversation_item_added", chat_item(
        "assistant", "hi", {"started_speaking_at": 1000.0, "stopped_speaking_at": 1009.0},
    ))
    await rec.finish()
    spans = _all_of_type(rec, "tts")
    assert len(spans) == 1
    assert "derived_from" not in spans[0].get("request", {})
    assert spans[0]["response"]["audio_ms"] == 2000


async def test_late_tts_metrics_correct_the_derived_span_instead_of_adding_one(
    recorder,
):
    """The ordering the fallback exists for must not record the reply twice.

    A plugin that emits `tts_metrics` late -- at teardown, after LiveKit has
    already committed the reply's text -- is the same plugin the derived span
    exists for. Opening a second operation for it doubles that turn's cost and
    duration, and the phantom span closes as `cancelled`, which the dashboard
    charts as a barge-in on a turn nobody interrupted.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(1200))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    # The reply is over and its turn has been retired before the plugin finally
    # reports. This is the exact sequence observed on Deepgram's aura-2.
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1"))
    await rec.finish()
    spans = _all_of_type(rec, "tts")
    for_first_turn = [s for s in spans if s["turn_id"] == spans[0]["turn_id"]]
    assert len(for_first_turn) == 1, "one reply must never be recorded as two"
    assert for_first_turn[0]["status"] != "cancelled"


async def test_a_derived_span_never_claims_zero_time_to_first_audio(recorder):
    """A derived span with no measured request time must report no TTFA.

    The span is anchored at the first tapped frame, so stamping the first-audio
    milestone there yields `tts_first_audio_ms: 0` -- not an approximation but
    a physically impossible number, charted as if the caller heard the reply
    the instant they stopped speaking.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(1200))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["request"]["derived_from"] == "conversation_item_added"
    milestones = span.get("milestones") or {}
    chunk = milestones.get("audio_chunk")
    assert chunk is None or chunk["occurred_at_ms"] > span["started_at_ms"], (
        "a derived span must report an unknown first-audio time, never a zero one"
    )


async def test_a_derived_span_reports_the_whole_reply_not_just_its_first_frames(
    recorder,
):
    """`conversation_item_added` fires mid-reply, so it cannot end the turn.

    LiveKit commits the reply's text when it is forwarded, seconds before the
    caller has heard it. Closing the turn there froze the TTS span at whatever
    had rendered so far and left the rest of the reply belonging to no open
    turn at all -- a 100% green call reporting a fraction of the agent's speech.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(200))
    session.emit("conversation_item_added", chat_item("assistant", "a long reply", {}))
    # The rest of the reply drains *after* the text was committed.
    for _ in range(9):
        rec.tap_output_frame(agent_frame(1000))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["played_ms"] == 9200
    assert span["response"]["audio_ms"] == 9200
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_ms"] == 9200
    assert span["response"]["audio_bytes"] == measured["agent_audio_bytes"], (
        "every frame the agent rendered must be attributed to the reply that made it"
    )


async def test_a_reply_the_caller_heard_none_of_is_reported_as_cancelled(recorder):
    """The most complete interruption there is must not read as healthy.

    When the barge-in lands before a single frame reaches `tts_node`, there is
    no rendered audio to compare against the provider's claim, so the
    "was it truncated" test cannot fire. Audio tapped elsewhere on the call
    proves the tap works, which is the only reason that test was guarded.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(500))
    session.emit("conversation_item_added", chat_item("assistant", "first", {}))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s2", cancelled=True))
    session.emit("conversation_item_added", chat_item("assistant", "cut off", {}))
    await rec.finish()
    spans = _all_of_type(rec, "tts")
    cut = [s for s in spans if s["response"].get("char_count") == len("cut off")]
    assert cut, spans
    assert cut[0]["status"] == "cancelled"
    assert cut[0]["response"]["played_ms"] == 0


async def test_a_stereo_agent_track_is_not_reported_at_twice_its_duration(recorder):
    """Duration maths must use the same denominator as the audio track's own."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    # One second of 24kHz 16-bit *stereo*: twice the bytes of the mono second.
    rec.tap_output_frame(FakeFrame(b"\x00" * (24000 * 2 * 2), channels=2))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_channels"] == 2
    assert measured["agent_audio_ms"] == 1000, "stereo bytes are not double the time"


async def test_every_span_that_measured_audio_also_records_what_was_said(recorder):
    """The frames and the words of a reply arrive through the same `tts_node`
    call, so resolving one more strictly than the other produces spans that
    carry a measured `played_ms` and no record of what was said -- the audit's
    "the agent's words are never recorded" on the very turns that prove the
    agent spoke."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    # Synthesis starts before the session reports the speech handle, so there
    # is no bound speaking turn when the words and the frames arrive.
    session.emit("user_input_transcribed", transcript("hello", True))
    rec.tap_output_text("first reply")
    rec.tap_output_frame(agent_frame(2000))
    # No `conversation_item_added`: the tape is the only possible source, so
    # the assertion below can only pass if the words were attributed to the
    # same turn as the frames they were spoken with.
    await rec.finish()
    for span in _all_of_type(rec, "tts"):
        response = span["response"]
        if response.get("played_ms"):
            assert response.get("text"), (
                f"{span['turn_id']} measured {response['played_ms']}ms of "
                "speech but recorded none of the words"
            )


async def test_a_queued_say_never_steals_the_active_replys_report(recorder):
    """`speech_created` means "queued", not "now speaking". A `say()` raised
    while a reply is playing commits its own text before its first frame, so
    every rule of the form "the new speech has not rendered yet, therefore this
    item belongs to the old one" points at exactly the wrong turn.

    Matching the item against the speech handle that produced it has no such
    failure mode, which is the whole reason to prefer identity over timing."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    reply = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(reply))
    rec.tap_output_frame(agent_frame(2500))
    # A `say()` is queued while the reply is still playing.
    interjection = FakeSpeechHandle("say1")
    session.emit("speech_created", speech_created(interjection, source="say"))
    # The `say()`'s own words commit before it has rendered a single frame.
    said = chat_item("assistant", "ONE MOMENT PLEASE", {})
    interjection.chat_items.append(said.item)
    session.emit("conversation_item_added", said)
    rec.tap_output_frame(agent_frame(900))
    await rec.finish()
    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    interjection_span = spans["turn-2"]
    assert (interjection_span.get("response") or {}).get("text") == "ONE MOMENT PLEASE", (
        "the say()'s words belong to the say(), not to the reply it interrupted"
    )
    assert (spans["turn-1"].get("response") or {}).get("text") != "ONE MOMENT PLEASE"


async def test_a_late_item_is_matched_to_its_speech_not_to_the_clock(recorder):
    """The mirror case: the item belongs to the reply that has *finished*, and
    a new speech has already opened. Identity gets both right; no rule based on
    "has the new turn rendered yet" can."""
    rec = recorder()
    session = FakeAgentSession(); rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    first = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(first))
    rec.tap_output_frame(agent_frame(3000))
    session.emit("user_input_transcribed", transcript("wait", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    # The second reply is already speaking, so no timing heuristic can help:
    # "the new turn has not rendered yet" is false, and only the handle that
    # actually produced the item can say where it belongs.
    rec.tap_output_frame(agent_frame(800))
    late = chat_item("assistant", "the first reply", {})
    first.chat_items.append(late.item)
    session.emit("conversation_item_added", late)
    await rec.finish()
    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert (spans["turn-1"].get("response") or {}).get("text") == "the first reply"
    assert (spans["turn-2"].get("response") or {}).get("text") != "the first reply", (
        "the second reply did not say the first reply's words"
    )


async def test_the_prior_reply_rule_still_works_with_transcripts_off(recorder):
    """The fallback heuristic must not quietly disable itself under a privacy
    setting. Keying it on stored transcript text did exactly that: with
    `capture_transcripts=False` the text is never written, so the test was
    permanently true and every late item was rerouted to the previous reply --
    stripping the current turn of its LLM span, its character count and its
    turn report, and reporting `coverage_complete: true` over the top."""
    rec = recorder(capture_transcripts=False)
    session = FakeAgentSession(); rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("s1"))
    session.emit("conversation_item_added", chat_item("assistant", "first", {}))
    # Second turn: its own item arrives before its first frame.
    session.emit("user_input_transcribed", transcript("again", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    session.emit("conversation_item_added", chat_item(
        "assistant", "second", {"llm_node_ttft": 0.3}))
    rec.tap_output_frame(agent_frame(1500))
    await rec.finish()
    llm_turns = [o["turn_id"] for o in _all_of_type(rec, "llm")]
    assert "turn-2" in llm_turns, (
        "turn 2's own reply report was routed to turn 1, so turn 2 has no LLM span"
    )


async def test_enough_small_tails_are_a_capture_failure(recorder):
    """A per-turn floor cannot answer "did this call lose data", because that
    answer is a sum. Twelve barge-ins each leaving 240ms wrote off 2.9s of
    measured agent speech against *zero* TTS spans and still reported the call
    fully captured -- the audit's P0-A signature exactly: audio with no
    operations behind a green status."""
    rec = recorder()
    session = FakeAgentSession(); rec.attach(session)
    for index in range(12):
        session.emit("user_input_transcribed", transcript(f"u{index}", True))
        session.emit(
            "speech_created",
            SimpleNamespace(speech_handle=SimpleNamespace(id=f"s{index}"),
                            source="generate_reply"),
        )
        rec.tap_output_frame(agent_frame(240))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert len(_all_of_type(rec, "tts")) == 0
    assert capture["coverage_complete"] is False, (
        "2.9s of measured agent speech with zero TTS operations is not a "
        "complete capture, however small each individual piece was"
    )


async def test_a_short_reply_with_words_is_never_written_off_as_a_tail(recorder):
    """A drain tail is audio with no words behind it. A 120ms reply that the
    agent was actually *given something to say* is a real reply, and treating
    it as jitter loses a real turn -- an earcon, a backchannel, a one-word
    acknowledgement."""
    rec = recorder()
    session = FakeAgentSession(); rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_text("mm-hm")
    rec.tap_output_frame(agent_frame(120))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False
    gap = capture["coverage_gaps"][0]
    assert gap["turn_ids"] == ["turn-1"], (
        "the turn the agent was given words to speak is the one to name"
    )
    assert gap["unattributed_agent_audio_ms"] == 120, (
        "the 120ms was not written off as boundary jitter: the agent was given "
        "something to say, which no drain tail ever is"
    )


async def test_multi_segment_replies_add_up(recorder):
    """One reply can be synthesized as several segments, and LiveKit's own
    usage collector sums every `TTSMetrics` -- these are additive measurements,
    not restatements. Overwriting them published the last segment as the whole
    reply, so both the provider duration and the *billable character count*
    came out low behind a healthy status."""
    rec = recorder()
    session = FakeAgentSession(); rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(3000))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=1.0))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=2.0))
    session.emit("conversation_item_added", chat_item("assistant", "a reply", {}))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["audio_ms"] == 3000, "1000ms + 2000ms of synthesis"
    assert span["response"]["characters_count"] == 84, "42 + 42 billable characters"


async def test_an_agent_that_was_never_tapped_is_not_called_fully_captured(recorder):
    """With no frames counted the coverage audit clears the call trivially:
    nothing was measured, so nothing can be unattributed. "Every millisecond we
    taped is on a span" is vacuously true when we taped nothing, and reporting
    that as a complete capture tells an operator their numbers are trustworthy
    at the one moment they are least able to be."""
    rec = recorder()

    class _Framework:
        async def tts_node(self, text, model_settings):  # noqa: ANN001
            yield None

    class Wrong(_Framework, VaaniAudioTapMixin):
        pass

    agent = Wrong()
    agent.vaani = rec
    rec.note_audio_tap_installed(agent)
    session = FakeAgentSession(); rec.attach(session)
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False
    assert capture["coverage_gaps"][0]["agent_audio_tapped"] is False


async def test_a_derived_span_is_as_long_as_the_reply_the_caller_heard(recorder):
    """Found on a live call, not in a fixture: three of four replies published
    a span roughly half the length of the answer the recording proves was
    spoken -- an 8.7s reply reported as 4.4s.

    `conversation_item_added` fires when the reply's *text* commits, which on a
    real call is ~0.6s into a 9s answer, so its `stopped_speaking_at` describes
    only the fraction that had played by then. Using it as the span's end makes
    every synthesis-rate and cost-per-second figure derived from that span
    wrong by the same factor, with nothing on the page to say so. The frames
    are counted until the reply is superseded, so they are both the later and
    the better evidence."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    # 600ms has played when the text commits and the item reports its window.
    rec.tap_output_frame(agent_frame(600))
    session.emit("conversation_item_added", chat_item(
        "assistant", "a long reply",
        {"started_speaking_at": 1000.0, "stopped_speaking_at": 1000.6},
    ))
    # The rest of the reply drains afterwards.
    rec.tap_output_frame(agent_frame(8100))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["played_ms"] == 8700
    assert span["duration_ms"] == 8700, (
        f"the caller heard 8700ms; the span published {span['duration_ms']}ms"
    )


async def test_the_manifest_says_how_many_spans_were_estimates(recorder):
    """On a real Deepgram `aura-2` call 3 replies in 4 emit no `tts_metrics`,
    so their spans are rebuilt from `conversation_item_added`. A manifest that
    reports only the narrower "nothing in the pipeline reported this" count
    truthfully says `reconstructed_op_count: 0` -- and thereby tells the reader
    every span was measured when three quarters were estimates. Silence about
    an estimate is the same failure class as a wrong number, and harder to
    catch."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    # Reply 1: the provider reports it.
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("s1"))
    session.emit("conversation_item_added", chat_item("assistant", "measured", {}))
    # Reply 2: no metric at all, only the conversation item.
    session.emit("user_input_transcribed", transcript("again", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(3000))
    session.emit("conversation_item_added", chat_item("assistant", "estimated", {}))
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["derived_tts_op_count"] == 1
    assert measured["derived_tts_agent_audio_ms"] == 3000
    # The narrower count stays narrow: this reply *was* announced.
    assert measured["reconstructed_op_count"] == 0


async def test_a_late_reply_report_never_lands_on_the_next_speech(recorder):
    """`conversation_item_added` fires when the reply's text commits, and a
    caller who barges in during that window opens a new speech first. Crediting
    the new speech gives a turn that has not spoken a word the previous reply's
    transcript and a `reply_complete` it never earned -- a fabricated turn
    beside a mute one, from one event arriving a few milliseconds late."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(3000))
    # The caller barges in; a new speech opens before the first reply's item
    # has been committed, and it has not rendered a single frame yet.
    session.emit("user_input_transcribed", transcript("wait", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    session.emit("conversation_item_added",
                 chat_item("assistant", "the first reply", {}))
    await rec.finish()
    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert "turn-2" not in spans, (
        "turn 2 rendered no audio at all; a span for it is fabricated"
    )
    assert (spans["turn-1"].get("response") or {}).get("played_ms") == 3000


async def test_a_reply_is_never_reported_twice(recorder):
    """`_end_tts` attributes the turn's *whole* byte count to whatever span it
    closes, so any second span for one reply reports the same audio again --
    doubling talk time and cost -- and closes as `cancelled`, which the
    dashboard's `is_interrupted` counts as a barge-in that never happened.
    `finalize_open_spans` re-runs `_end_tts` over every turn, including ones
    already closed mid-call, so the idempotence is load-bearing on every call
    with more than one turn.

    This is an invariant test, not a single-guard test: the property is held up
    by `_derive_tts`'s `state.tts is not None` early return, `_end_tts`'s
    `state.tts is None` derivation precondition and its `state.tts.ended`
    return, and `_record_tts`'s ended-span return. Reverting any one of them
    alone leaves the invariant intact; reverting the two derivation guards
    together produces a third span for two replies, which is what this test
    catches."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("s1"))
    session.emit("conversation_item_added", chat_item("assistant", "a reply", {}))
    # A new turn closes the first one mid-call; `finish()` then walks every
    # turn again.
    session.emit("user_input_transcribed", transcript("again", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(1000))
    session.emit("conversation_item_added", chat_item("assistant", "second", {}))
    # A stray metric for the first reply arrives after it was published.
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=99.0))
    await rec.finish()
    spans = _all_of_type(rec, "tts")
    assert len(spans) == 2, f"2 replies produced {len(spans)} spans"
    played = sum((o.get("response") or {}).get("played_ms") or 0 for o in spans)
    assert played == 3000, f"3000ms was rendered but {played}ms was reported"
    assert [o["status"] for o in spans] == ["ok", "ok"]


async def test_reporting_more_speech_than_was_rendered_is_never_healthy(recorder):
    """Over-attribution is the failure a double-counted span produces, so an
    audit blind to it cannot catch the defect most likely to flatter the
    numbers."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(2000))
    session.emit("conversation_item_added", chat_item("assistant", "a reply", {}))
    # Simulate a span that published more speech than the tape holds.
    rec._all_turns[0].audio_bytes *= 3
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False
    gap = capture["coverage_gaps"][0]
    assert gap["overattributed_agent_audio_ms"] > 0


async def test_a_mixin_that_loses_the_mro_is_not_called_measured(recorder):
    """`class Wrong(Agent, VaaniAudioTapMixin)` passes `isinstance` while the
    framework's own `tts_node` wins method resolution and the tap never runs.
    Calling that zero "measured" reports a talking agent as silent."""
    rec = recorder()

    class _Framework:
        async def tts_node(self, text, model_settings):  # noqa: ANN001
            yield None

    class Wrong(_Framework, VaaniAudioTapMixin):
        pass

    agent = Wrong()
    agent.vaani = rec
    rec.note_audio_tap_installed(agent)
    session = FakeAgentSession()
    rec.attach(session)
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_tapped"] is False

    class Right(VaaniAudioTapMixin, _Framework):
        pass

    rec2 = recorder()
    right = Right()
    right.vaani = rec2
    rec2.note_audio_tap_installed(right)
    session2 = FakeAgentSession()
    rec2.attach(session2)
    await rec2.finish()
    assert _manifest_of(rec2)["capture_status"]["measured"]["agent_audio_tapped"] is True


async def test_a_playout_window_the_frames_disprove_is_not_treated_as_measured(
    recorder,
):
    """`conversation_item_added` fires when the text commits, ~0.6s into a 9s
    reply. A window reflecting that instant would be published as the reply's
    duration for speech the caller heard in full."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(9000))
    session.emit("conversation_item_added", chat_item(
        "assistant", "a long reply",
        {"started_speaking_at": 1000.0, "stopped_speaking_at": 1000.6},
    ))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["played_ms"] == 9000
    assert span["duration_ms"] == 9000, (
        "a 600ms window is disproved by the 9000ms of frames already counted"
    )


async def test_a_reply_measured_but_never_committed_is_not_called_cancelled(
    recorder,
):
    """A `say(add_to_chat_ctx=False)`, or a room that disconnects between
    playout ending and the commit, produces a fully rendered reply with no
    conversation item. Marking it cancelled is a fabricated barge-in."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(4000))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=2.0))
    # The call ends without any conversation item.
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["status"] == "ok", (
        "played 4000ms against a 2000ms synthesized claim is not evidence of "
        "an interruption"
    )


async def test_the_greeting_text_is_taped_off_the_tts_node(recorder):
    """The words handed to `tts_node` are the only record of the agent's
    speech present for every reply. LiveKit announces nothing for a greeting
    generated before the first user turn, so without this tap the first thing
    the caller heard is missing from the transcript."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_text("Hi, thanks ")
    rec.tap_output_text("for calling.")
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["text"] == "Hi, thanks for calling."
    assert span["response"]["text_source"] == "tts_node"


async def test_taped_text_never_overrides_the_words_the_caller_heard(recorder):
    """`forwarded_text` proves what actually reached the caller; the tapped
    text is only what we asked to be spoken. On an interrupted reply they
    differ, and overwriting the former would turn a truthful record of a
    cut-off reply into a claim the whole thing was heard."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_text("Hi, how can I help you today?")
    rec.tap_output_frame(agent_frame(2000))
    session.emit("conversation_item_added", chat_item("assistant", "Hi, how c", {}))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["text"] == "Hi, how c"
    assert span["response"].get("text_source") != "tts_node"


async def test_tapped_text_is_withheld_when_content_capture_is_off(recorder):
    """A policy that forbids storing prompts must not be defeated by the tap."""
    rec = recorder(capture_transcripts=False)
    session = FakeAgentSession()
    rec.attach(session)
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_text("Hi, thanks for calling.")
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert "text" not in span["response"]


async def test_the_opening_greeting_appears_in_the_transcript(recorder):
    """LiveKit emits no `conversation_item_added` for a reply generated before
    the first user turn, so an agent that opens by speaking had its greeting
    rendered, measured and charted while the transcript showed nothing for it.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    handle = _DoneHandle("greeting")
    session.emit("speech_created",
                 SimpleNamespace(speech_handle=handle, source="generate_reply"))
    rec.tap_output_frame(agent_frame(2000))
    handle.finish([SimpleNamespace(role="assistant", text_content="Hi, how can I help?")])
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["text"] == "Hi, how can I help?"
    assert span["response"]["char_count"] == 19
    assert span["response"]["text_source"] == "speech_handle"


async def test_the_forwarded_text_wins_over_the_speech_handle(recorder):
    """`conversation_item_added` carries what actually reached the caller; the
    handle carries what the model produced. On an interrupted reply those
    differ, and the played words are the truthful record."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    handle = _DoneHandle("s1")
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created",
                 SimpleNamespace(speech_handle=handle, source="generate_reply"))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("conversation_item_added", chat_item("assistant", "Hi, how c", {}))
    handle.finish([SimpleNamespace(role="assistant",
                                   text_content="Hi, how can I help you today?")])
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["text"] == "Hi, how c", (
        "the words the caller heard must not be overwritten by the words the "
        "model produced"
    )
    assert "text_source" not in span["response"]


async def test_a_reply_cut_off_before_its_text_committed_still_gets_a_tts_span(
    recorder,
):
    """An interruption before `conversation_item_added` emits no TTS metric and
    no conversation item, so nothing opened a span -- while the caller
    demonstrably heard the reply, because we taped the frames. That is the
    audit's headline defect in miniature."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(4000))
    # The caller barges in: a new reply supersedes this one, which never
    # committed any text.
    session.emit("user_input_transcribed", transcript("stop", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(1000))
    session.emit("conversation_item_added", chat_item("assistant", "ok", {}))
    await rec.finish()
    spans = _all_of_type(rec, "tts")
    assert len(spans) == 2, "the interrupted reply must be recorded, not dropped"
    played = sum((o.get("response") or {}).get("played_ms") or 0 for o in spans)
    assert played == 5000, f"all rendered audio must be attributed, got {played}"
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True


async def test_a_derived_span_ends_when_the_reply_did_not_when_the_turn_closed(
    recorder,
):
    """A derived span is closed when its reply is superseded, which on the last
    reply of a call is the end of the call. Left at that, the timeline draws a
    35-second bar for an 11-second answer."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(2000))
    session.emit("conversation_item_added", chat_item("assistant", "a reply", {}))
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["response"]["played_ms"] == 2000
    assert span["duration_ms"] == 2000, (
        "a derived span must be as long as the reply, not as long as the wait "
        "before something closed it"
    )


async def test_a_few_hundred_ms_of_playout_tail_is_not_a_capture_failure(recorder):
    """A reply still draining when the caller speaks lands a little audio on
    the next turn. That is boundary jitter, not a stage that failed to report,
    and downgrading a complete call for it teaches operators to ignore the
    status that means a number is actually missing."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(2000))
    session.emit("conversation_item_added", chat_item("assistant", "a reply", {}))
    # The caller interrupts; the tail of the previous reply lands on the new
    # turn, which then never replies at all.
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s2"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(120))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True
    assert "coverage_gaps" not in capture
    spans = _all_of_type(rec, "tts")
    assert len(spans) == 1, (
        "a 120ms drain tail is not a reply: fabricating a span for it inflates "
        "the denominator of every TTS rate, and a phantom that closes as "
        "cancelled is counted by the dashboard as a barge-in"
    )
    assert spans[0]["turn_id"] == "turn-1"


async def test_a_derived_reply_names_its_provider_and_model(recorder):
    """A derived span that cannot name its provider drops out of cost and
    latency reporting entirely, which is a quieter version of the same bug."""
    rec = recorder()
    session = FakeAgentSession()
    session.tts = SimpleNamespace(provider="deepgram", model="aura-2-thalia-en")
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(600))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()
    tts = _by_type(rec, "tts")
    assert tts["provider"] == "deepgram"
    assert tts["model"] == "aura-2-thalia-en"


async def test_a_reply_with_no_metrics_and_no_item_is_recovered_not_merely_flagged(
    recorder,
):
    """The recorder taped the agent's audio, so it can prove the agent spoke.
    When nothing else in the pipeline reported the reply, that tape is enough
    to reconstruct the span -- which is strictly better than reporting a gap,
    because the operator gets the reply rather than a warning about it."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    rec.tap_output_frame(agent_frame(1500))
    # No tts_metrics and no conversation_item_added: the agent spoke and
    # nothing in the pipeline ever told us about it.
    await rec.finish()
    span = _by_type(rec, "tts")
    assert span["request"]["derived_from"] == "captured_agent_audio"
    assert span["response"]["played_ms"] == 1500
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True


async def test_agent_audio_belonging_to_no_turn_is_never_reported_as_healthy(recorder):
    """The backstop for the *next* cause of a missing stage. Auditing per-turn
    only finds audio that landed on a turn, so frames rendered while no turn
    was open were invisible: measured talk time exceeded the sum of the spans
    and the call still called itself fully captured -- the audit's exact
    complaint."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    # The agent speaks before the session reports any speech at all, so there
    # is no turn to credit and no span to derive.
    rec.tap_output_frame(agent_frame(1500))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["measured"]["agent_audio_ms"] == 1500
    assert capture["coverage_complete"] is False
    gap = capture["coverage_gaps"][0]
    assert gap["stage"] == "tts"
    assert gap["unattributed_agent_audio_ms"] == 1500


async def test_a_fully_captured_call_reports_complete_coverage(recorder):
    """The audit must not cry wolf on a healthy call, or it will be ignored."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True
    assert "coverage_gaps" not in capture


async def test_a_mute_agent_is_reported_as_measured_silence_not_as_a_capture_gap(
    recorder,
):
    """Zero operations has two causes and they need opposite responses.

    An agent that never speaks is the most consequential voice failure there
    is, and it produces exactly the same empty console as a broken recorder.
    The tap runs in `tts_node`, so `agent_audio_ms == 0` is a measurement: it
    sends the operator to their agent instead of to our SDK.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("are you there", True))
    # Wired correctly, and simply never asked to speak. `tts_node` only runs
    # when there is something to say, so this must be established at wire-up:
    # inferring it from arriving frames makes a correctly-wired mute agent
    # indistinguishable from a mis-wired one, which is the whole distinction.
    rec.note_audio_tap_installed()
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["measured"]["agent_audio_ms"] == 0
    assert capture["measured"]["agent_audio_tapped"] is True
    assert capture["coverage_complete"] is True, "silence is not a capture failure"
    assert "coverage_gaps" not in capture


async def test_an_agent_without_the_tap_mixin_is_not_reported_as_measured(recorder):
    """Binding `agent=` is necessary but not sufficient.

    Without `VaaniAudioTapMixin` the node hooks are never overridden, so no
    frame can reach the recorder. Treating "an agent was bound" as proof of
    measurement would make the same false claim by a different route.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    class PlainAgent:
        vaani = None

    agent = PlainAgent()
    agent.vaani = rec
    rec.note_audio_tap_installed(agent)
    session.emit("user_input_transcribed", transcript("are you there", True))
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_tapped"] is False


async def test_an_unmeasured_call_is_never_reported_as_a_silent_agent(recorder):
    """Zero audio with no tap installed is not evidence about the agent.

    `tap_output_frame` only runs when the caller bound `agent=` and mixed in
    `VaaniAudioTapMixin`. Publishing the same `agent_audio_ms: 0` for that case
    lets the console tell an operator who mis-wired the SDK that their agent
    was mute -- stated in the product's most confident language, about the one
    thing they would act on immediately.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("are you there", True))
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_ms"] == 0
    assert measured["agent_audio_tapped"] is False, (
        "a call with no audio tap must not be presented as a measurement of the agent"
    )


async def test_audio_spoken_before_any_turn_still_counts_as_the_agent_speaking(
    recorder,
):
    """A greeting belongs to no turn, and must not read as silence.

    Turn attribution starts at the caller's first transcript, so an agent that
    opens the call has already spoken by the time any turn exists. Counting
    only attributed frames would publish `agent_audio_ms: 0` for a call that
    began with the agent talking -- the exact false story this measurement
    exists to prevent.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.tap_output_frame(agent_frame(1200))
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_ms"] == 1200


async def test_the_measured_agent_audio_travels_with_every_call(recorder):
    """Published on healthy calls too, or the console cannot tell zero from absent."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.tap_output_frame(agent_frame(2000))
    await run_one_turn(rec, session)
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["agent_audio_ms"] >= 2000
    assert measured["agent_audio_sample_rate_hz"] > 0
    assert measured["agent_audio_bytes"] == measured["agent_audio_ms"] * (
        measured["agent_audio_sample_rate_hz"] // 1000
    ) * 2


# ------------------------------------------------------------------- turns


async def test_two_turns_do_not_share_a_turn_id(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session, speech_id="speech-1")
    await run_one_turn(rec, session, speech_id="speech-2")
    await rec.finish()
    events = operations(read_events(_dir(rec)))
    assert len({event["turn_id"] for event in events}) == 2


async def test_a_greeting_with_no_user_speech_still_gets_a_turn(recorder):
    """`say()` produces a speech handle with no preceding utterance."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="greet"), source="say"),
    )
    session.emit("metrics_collected", tts_metrics("greet"))
    session.emit("conversation_item_added", chat_item("assistant", "Namaste", {}))
    await rec.finish()
    events = operations(read_events(_dir(rec)))
    assert [event["type"] for event in events] == ["tts"]
    assert events[0]["turn_id"] is not None
    # A greeting never went through STT; a zero-length STT span there would
    # poison the STT latency distribution.
    assert not any(event["type"] == "stt" for event in events)


async def test_tool_calls_are_recorded_against_the_current_turn(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("book it", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit(
        "function_tools_executed",
        SimpleNamespace(
            function_calls=[SimpleNamespace(name="search_flights", arguments='{"to":"GOI"}',
                                            call_id="c1")],
            function_call_outputs=[SimpleNamespace(output='{"price":6000}', is_error=False,
                                                   call_id="c1")],
            zipped=lambda: [
                (
                    SimpleNamespace(name="search_flights", arguments='{"to":"GOI"}', call_id="c1"),
                    SimpleNamespace(output='{"price":6000}', is_error=False, call_id="c1"),
                )
            ],
        ),
    )
    await rec.finish()
    tool = _by_type(rec, "tool")
    assert tool["request"]["name"] == "search_flights"
    assert tool["response"]["result"] == '{"price":6000}'
    assert tool["status"] == "ok"
    assert tool["turn_id"] == _by_type(rec, "stt")["turn_id"]


async def test_a_failing_tool_is_recorded_as_an_error(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("book it", True))
    session.emit(
        "function_tools_executed",
        SimpleNamespace(
            zipped=lambda: [
                (
                    SimpleNamespace(name="broken", arguments="{}", call_id="c1"),
                    SimpleNamespace(output="boom", is_error=True, call_id="c1"),
                )
            ]
        ),
    )
    await rec.finish()
    assert _by_type(rec, "tool")["status"] == "error"


# ------------------------------------------------------------------- audio


async def test_audio_taps_write_one_dashboard_playable_stereo_recording(recorder):
    rec = recorder()
    rec.tap_input_frame(FakeFrame(b"\x01\x02" * 240))
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hi", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1"))
    rec.tap_output_frame(FakeFrame(b"\x03\x04" * 240))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()

    audio = rec._finalized.manifest["audio"]
    assert set(audio) == {"call"}
    track = audio["call"]
    assert track["file"] == "call.audio"
    assert track["encoding"] == "pcm_s16le"
    assert track["channels"] == 2
    assert track["channel_layout"] == {"left": "agent", "right": "caller"}
    assert isinstance(track["sample_rate_hz"], int) and track["sample_rate_hz"] > 0
    assert os.path.getsize(os.path.join(rec._finalized.directory, track["file"])) >= 960


async def test_the_tts_span_marks_first_audio_even_though_frames_precede_metrics(recorder):
    """The dashboard's headline "time to first audio" reads this milestone.

    LiveKit synthesizes the reply *before* it reports the TTS metrics that open
    the span, so a recorder that only stamps the milestone when the span already
    exists drops it on every real turn and leaves the KPI permanently blank.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hi", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    # The real order: audio first, metrics afterwards.
    rec.tap_output_frame(FakeFrame(b"\x03\x04" * 240))
    rec.tap_output_frame(FakeFrame(b"\x03\x04" * 240))
    session.emit("metrics_collected", tts_metrics("s1"))
    session.emit("conversation_item_added", chat_item("assistant", "hi", {}))
    await rec.finish()

    tts = _by_type(rec, "tts")
    milestone = tts["milestones"].get("audio_chunk")
    assert milestone is not None, "first-audio milestone was never recorded"
    assert milestone["occurred_at_ms"] <= tts["ended_at_ms"]
    assert milestone["total_byte_count"] == 960
    assert tts["response"]["audio_bytes"] == 960


async def test_a_frame_uses_its_own_sample_rate_not_the_configured_default(recorder):
    rec = recorder(input_sample_rate=24000)
    rec.tap_input_frame(FakeFrame(b"\x00" * 320, sample_rate=16000))
    await rec.finish()
    assert rec._finalized.manifest["audio"]["call"]["sample_rate_hz"] == 16000


# ------------------------------------------------------------- degradation


async def test_an_inert_recorder_accepts_every_call(tmp_path):
    """Observability off must be a deployment choice, not a code path."""
    rec = VaaniLiveKitRecorder(None)
    session = FakeAgentSession()
    rec.attach(session)
    assert rec.enabled is False
    await run_one_turn(rec, session)
    rec.tap_input_frame(FakeFrame(b"\x00" * 10))
    rec.tap_output_frame(FakeFrame(b"\x00" * 10))
    rec.finalize_open_spans()
    assert rec.observe_socket(object()) is None
    await rec.finish()


async def test_a_handler_error_never_escapes_into_the_call(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    # An event shaped nothing like LiveKit's must not take the call down.
    session.emit("metrics_collected", SimpleNamespace(metrics=None))
    session.emit("user_input_transcribed", SimpleNamespace())
    session.emit("conversation_item_added", SimpleNamespace(item=None))
    await rec.finish()
    assert rec._finalized.manifest["capture_status"]["events_complete"] is True


async def test_finalize_closes_a_turn_abandoned_mid_flight(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", llm_metrics("s1"))
    # No assistant item ever arrives: the caller hung up.
    await rec.finish(outcome="abandoned")
    events = operations(read_events(_dir(rec)))
    assert events, "an abandoned call must still produce spans"
    assert all(event["ended_at_ms"] is not None for event in events)
    assert all(event["status"] != "in_progress" for event in events)
    assert rec._finalized.manifest["outcome"] == "abandoned"


async def test_a_session_error_fails_only_the_stage_that_raised_it(recorder):
    """An LLM timeout must not report the turn's completed transcription as a failed STT."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("metrics_collected", llm_metrics("s1"))
    session.emit(
        "error",
        SimpleNamespace(
            error=SimpleNamespace(
                type="llm_error", label="livekit.plugins.openai.llm.LLM",
                error=RuntimeError("provider down"), recoverable=True,
            ),
            source="llm",
        ),
    )
    await rec.finish(outcome="failed")
    assert _by_type(rec, "stt")["status"] == "ok"


async def test_an_stt_error_fails_the_stt_span(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit(
        "error",
        SimpleNamespace(
            error=SimpleNamespace(
                type="stt_error", label="deepgram",
                error=RuntimeError("socket reset"), recoverable=False,
            ),
            source="stt",
        ),
    )
    await rec.finish(outcome="failed")
    stt = _by_type(rec, "stt")
    assert stt["status"] == "error"
    assert "socket reset" in stt["error"]["message"]
    assert stt["error"]["recoverable"] is False


async def test_an_untyped_session_error_still_fails_every_open_span(recorder):
    """An unrecognised failure must never be recorded as a clean call."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("error", SimpleNamespace(error=RuntimeError("provider down"), source=None))
    await rec.finish(outcome="failed")
    stt = _by_type(rec, "stt")
    assert stt["status"] == "error"
    assert "provider down" in stt["error"]["message"]


# --------------------------------------------------------------- the mixin


async def test_the_audio_tap_mixin_tees_both_directions(recorder):
    rec = recorder()

    class Base:
        async def stt_node(self, audio, model_settings):  # noqa: ANN001
            async for frame in audio:
                yield f"heard:{len(bytes(frame.data))}"

        async def tts_node(self, text, model_settings):  # noqa: ANN001
            yield FakeFrame(b"\x09" * 100)

    class Tapped(VaaniAudioTapMixin, Base):
        pass

    agent = Tapped()
    agent.vaani = rec

    async def frames():
        yield FakeFrame(b"\x01" * 200)

    heard = [event async for event in agent.stt_node(frames(), None)]
    spoken = [frame async for frame in agent.tts_node(iter([]), None)]
    await rec.finish()

    assert heard == ["heard:200"], "the tap must pass frames through unchanged"
    assert len(spoken) == 1
    directory = rec._finalized.directory
    assert os.path.getsize(os.path.join(directory, "call.audio")) >= 400


async def test_observe_agent_session_is_inert_without_the_env_flag(monkeypatch):
    monkeypatch.delenv("VAANI_ENABLED", raising=False)
    rec = observe_agent_session(FakeAgentSession())
    assert rec.enabled is False
    await rec.finish()


class _DoneHandle:
    """A LiveKit `SpeechHandle` as far as this recorder uses it."""

    def __init__(self, id: str):
        self.id = id
        self.chat_items = []
        self._callbacks = []

    def add_done_callback(self, callback):
        self._callbacks.append(callback)

    def finish(self, items):
        self.chat_items = items
        for callback in self._callbacks:
            callback(self)


def _by_type(rec: VaaniLiveKitRecorder, kind: str) -> dict:
    matches = [event for event in operations(read_events(_dir(rec))) if event["type"] == kind]
    assert matches, f"no {kind} operation was recorded"
    return matches[0]


def _all_of_type(rec: VaaniLiveKitRecorder, kind: str) -> list:
    return [event for event in operations(read_events(_dir(rec))) if event["type"] == kind]


async def test_the_stt_span_names_its_provider_and_model(recorder):
    """The dashboard prices by provider/model and warns when they are absent."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    await rec.finish()
    stt = _by_type(rec, "stt")
    assert stt["provider"] == "deepgram"
    assert stt["model"] == "nova-3"
    assert stt["transport"] == "websocket"


async def test_absent_latency_fields_are_omitted_not_written_as_null(recorder):
    """Node omits `undefined`; a null reads as a measured zero on the charts."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hi", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    session.emit("metrics_collected", tts_metrics("s1"))
    session.emit("conversation_item_added",
                 chat_item("assistant", "hello", {"e2e_latency": 1.0}))
    await rec.finish()
    report = _by_type(rec, "tts")["milestones"]["turn_report"]
    assert report["e2e_latency_ms"] == 1000
    assert "playback_latency_ms" not in report
    assert "llm_ttft_ms" not in report


async def test_a_turn_is_counted_once_despite_being_indexed_twice(recorder):
    """A turn is keyed by both its own id and the speech id that adopted it."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session, speech_id="speech-1")
    assert len(rec._all_turns) == 1
    assert len(rec._turns) == 2, "the double index is what makes lookup work"
    rec.finalize_open_spans()
    await rec.finish()


async def test_session_usage_is_recorded_on_the_manifest_not_as_a_span(recorder):
    """It is a running total; a span per update would double-count tokens."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    usage = SimpleNamespace(model_dump=lambda: {"llm_prompt_tokens": 310})
    session.emit("session_usage_updated", SimpleNamespace(usage=usage))
    await rec.finish()
    assert rec._finalized.manifest["metadata"]["usage"] == {"llm_prompt_tokens": 310}
    assert not operations(read_events(_dir(rec)))


async def test_the_llm_node_scope_gives_ambient_http_spans_a_turn_id(recorder):
    """Without this every auto-instrumented provider span lands with turn_id null."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hi", True))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    seen = []

    class Base:
        async def llm_node(self, chat_ctx, tools, model_settings):  # noqa: ANN001
            # Stands in for the provider call the plugin makes here.
            seen.append(rec._observer.current_context())
            yield "token"

    class Scoped(VaaniAudioTapMixin, Base):
        pass

    agent = Scoped()
    agent.vaani = rec
    assert [chunk async for chunk in agent.llm_node(None, [], None)] == ["token"]
    await rec.finish()

    assert seen and seen[0] is not None
    assert seen[0].turn_id == rec._current_turn_id_for_test
    # The scope must not leak out to whoever consumed the generator.
    assert rec._observer.current_context() is None


async def test_stt_identity_survives_a_turn_whose_metric_arrived_earlier(recorder):
    """STT metrics fire per provider request, not per utterance."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_input_transcribed", transcript("first", True))
    # No second stt_metrics event for the next utterance.
    session.emit("user_input_transcribed", transcript("second", True))
    await rec.finish()
    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert len(stt_ops) == 2
    assert all(event["provider"] == "deepgram" for event in stt_ops)
    assert all(event["model"] == "nova-3" for event in stt_ops)


async def test_abandoning_a_node_stream_closes_the_provider_iterator(recorder):
    """A consumer that walks away must not leave the provider's request alive."""
    rec = recorder()
    closed = []

    class Base:
        async def llm_node(self, chat_ctx, tools, model_settings):  # noqa: ANN001
            try:
                yield "one"
                yield "two"
            finally:
                closed.append(True)

    class Scoped(VaaniAudioTapMixin, Base):
        pass

    agent = Scoped()
    agent.vaani = rec
    stream = agent.llm_node(None, [], None)
    assert await stream.__anext__() == "one"
    await stream.aclose()
    await rec.finish()
    assert closed == [True], "the provider generator was never finalized"


async def test_a_provider_whose_cleanup_fails_does_not_break_the_caller(recorder):
    rec = recorder()

    class Exploding:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("provider cleanup exploded")

    from vaani_observer.integrations.livekit import _scoped

    assert [item async for item in _scoped(rec, Exploding())] == []
    await rec.finish()


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("error", "failed"),
        ("job_shutdown", "abandoned"),
        ("participant_disconnected", "completed"),
        ("user_initiated", "completed"),
        ("task_completed", "completed"),
        ("something_new", "unknown"),
    ],
)
async def test_the_close_reason_decides_the_outcome(recorder, reason, expected):
    """Reporting every call as completed would peg call success rate at 100%."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("close", SimpleNamespace(reason=reason, error=None))
    manifest = await _finalized(rec)
    assert manifest["outcome"] == expected
    assert manifest["metadata"]["close_reason"] == reason


async def test_a_close_carrying_an_error_is_always_a_failure(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit(
        "close",
        SimpleNamespace(reason="user_initiated", error=SimpleNamespace(message="llm died")),
    )
    manifest = await _finalized(rec)
    assert manifest["outcome"] == "failed"
    assert manifest["metadata"]["close_error"] == "llm died"


async def test_an_explicit_outcome_still_wins_over_the_observed_one(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("close", SimpleNamespace(reason="job_shutdown", error=None))
    await rec.finish(outcome="abandoned_by_caller")
    assert _manifest_of(rec)["outcome"] == "abandoned_by_caller"


async def test_session_usage_from_a_dataclass_reaches_the_manifest(recorder):
    """LiveKit mixes pydantic models and dataclasses; usage must survive both."""
    import dataclasses

    @dataclasses.dataclass
    class ModelUsage:
        model: str
        input_tokens: int

    @dataclasses.dataclass
    class AgentSessionUsage:
        model_usage: list

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit(
        "session_usage_updated",
        SimpleNamespace(usage=AgentSessionUsage(model_usage=[ModelUsage("gpt-5-mini", 42)])),
    )
    manifest = await _finalized(rec)
    assert manifest["metadata"]["usage"] == {
        "model_usage": [{"model": "gpt-5-mini", "input_tokens": 42}]
    }


async def test_a_span_is_never_shorter_than_the_audio_it_says_it_played(recorder):
    """A measured span closes when `TTSMetrics` arrives -- when *synthesis*
    finished. Deepgram synthesizes far faster than realtime, so the caller is
    still listening well after that: on a live call a 3080ms greeting was
    published as a 2318ms span. A timeline bar shorter than the audio inside it
    contradicts itself, and anything integrating duration undercounts the
    agent's talk time, which is the exact number this product exists to sell."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(3080))
    # Metrics arrive while the caller is still hearing the reply.
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=3.08))
    await rec.finish()
    span = _by_type(rec, "tts")
    played = (span.get("response") or {}).get("played_ms")
    assert played == 3080
    assert span["duration_ms"] >= played, (
        f"the span lasts {span['duration_ms']}ms but reports playing "
        f"{played}ms of audio; the caller heard the longer number"
    )


async def test_a_provider_that_reports_less_audio_than_played_is_flagged(recorder):
    """Multi-segment replies where only one segment's `TTSMetrics` arrives make
    the provider's `audio_ms` understate the billable quantity -- live, a 5670ms
    reply carried `audio_ms: 3040`, a 46% undercount. Both numbers are kept, but
    the disagreement is published: a cost dashboard that is confidently low is
    the failure direction that flatters the product."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(5670))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=3.04))
    await rec.finish()
    response = _by_type(rec, "tts").get("response") or {}
    assert response["played_ms"] == 5670
    assert response["audio_ms"] == 3040, "the provider's own claim is preserved"
    assert response.get("provider_audio_ms_undercount_ms") == 2630, (
        "the 2630ms the provider never accounted for is not published, so a "
        "reader billing off audio_ms cannot tell they are 46% low"
    )


async def test_words_that_could_belong_to_either_reply_are_not_guessed(recorder):
    """A superseded reply that rendered audio without words, and a new speech
    that has rendered nothing, produce event-for-event identical streams whether
    an arriving item is the old reply's late report or the new one's own --
    committing text before the first frame is ordinary for `say()`. With no
    handle identity and no taped text there is no evidence, and publishing
    either reading is wrong half the time with nothing marking it suspect."""
    rec = recorder(capture_transcripts=False)
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created",
                 SimpleNamespace(speech_handle=SimpleNamespace(id="s1"),
                                 source="generate_reply"))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("user_input_transcribed", transcript("wait", True))
    session.emit("speech_created",
                 SimpleNamespace(speech_handle=SimpleNamespace(id="s2"),
                                 source="generate_reply"))
    session.emit("conversation_item_added",
                 chat_item("assistant", "WHOSE WORDS ARE THESE", {}))
    await rec.finish()
    for span in _all_of_type(rec, "tts"):
        assert "WHOSE WORDS ARE THESE" != (span.get("response") or {}).get("text"), (
            f"{span['turn_id']} was given words no evidence says it spoke"
        )
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False, (
        "dropping a transcript is a gap and must not be reported as a "
        "complete capture"
    )
    assert any(g.get("turn_ids") == ["turn-1", "turn-2"]
               for g in capture["coverage_gaps"]), capture["coverage_gaps"]


async def test_taped_words_decide_which_reply_an_item_belongs_to(recorder):
    """When handle identity is unavailable, what we taped on its way to
    `tts_node` still settles it: the arriving words either are the previous
    reply's words or they are not."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created",
                 SimpleNamespace(speech_handle=SimpleNamespace(id="s1"),
                                 source="generate_reply"))
    rec.tap_output_text("the first reply")
    rec.tap_output_frame(agent_frame(2000))
    session.emit("user_input_transcribed", transcript("wait", True))
    session.emit("speech_created",
                 SimpleNamespace(speech_handle=SimpleNamespace(id="s2"),
                                 source="generate_reply"))
    session.emit("conversation_item_added",
                 chat_item("assistant", "the first reply", {}))
    await rec.finish()
    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    response = spans["turn-1"].get("response") or {}
    assert response.get("text") == "the first reply"
    # `forwarded_text` is proof of what the caller heard; the tape is only what
    # we asked to be spoken. Resolving the item is what upgrades one to the
    # other, so the source is what distinguishes a matched item from a dropped
    # one -- the words alone would read the same either way.
    assert response.get("text_source") != "tts_node", (
        "the committed words never landed, so this reply is only evidenced by "
        "what we asked for, not by what was delivered"
    )
    assert "turn-2" not in spans, "turn 2 rendered nothing; a span for it is fabricated"
    assert _manifest_of(rec)["capture_status"]["coverage_complete"] is True, (
        "the taped text settled the ownership question, so nothing was dropped"
    )


async def test_a_queued_say_never_steals_the_audio_still_being_rendered(recorder):
    """`say()` creates a speech and LiveKit emits `speech_created` for it while
    the previous reply is *still yielding frames*. Attributing every frame to
    the recorder's global idea of who is speaking hands the rest of the older
    reply's audio, and its words, to a reply that has not made a sound. The
    total is conserved, so the call reports fully covered while two turns' talk
    time -- and every latency and cost figure derived from it -- are wrong in
    opposite directions."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    first = rec.open_output_stream()
    rec.tap_output_text("the long answer", first)
    rec.tap_output_frame(agent_frame(2000), first)
    # `say()` is queued while the first reply is mid-flight.
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2"),
                                                  source="say"))
    second = rec.open_output_stream()
    # The first reply keeps rendering: it was authorized first and is still
    # draining through its own `tts_node` call.
    rec.tap_output_frame(agent_frame(1500), first)
    rec.tap_output_text(" one moment", second)
    rec.tap_output_frame(agent_frame(400), second)
    await rec.finish()
    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert (spans["turn-1"].get("response") or {}).get("played_ms") == 3500, (
        "the 1500ms rendered after say() was queued belongs to the reply whose "
        "generator produced it"
    )
    assert (spans["turn-2"].get("response") or {}).get("played_ms") == 400
    assert "the long answer" in (spans["turn-1"].get("response") or {})["text"]
    assert "the long answer" not in (spans["turn-2"].get("response") or {})["text"]


async def test_the_reply_that_owns_a_tts_node_call_keeps_its_audio_across_an_llm_stall(
        recorder, monkeypatch):
    """The first frame out of `tts_node` waits on the LLM's first token, so
    pinning ownership at the first *output* resolves it hundreds of
    milliseconds after the node was invoked -- the very window a filler
    `say()` is queued in. Pinned there, the whole reply binds to the wrong
    turn: the reply the caller actually heard gets no span at all, and the
    filler's span publishes two replies' words concatenated.

    LiveKit sets its speech-handle context before creating the speech task, so
    `tts_node` can name its own reply. That is proof, not a timing guess."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    # `tts_node` for s1 is invoked here -- before its LLM has produced a token.
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()

    # A filler `say()` is queued inside the LLM stall and renders immediately.
    h2 = FakeSpeechHandle("s2")
    session.emit("speech_created", speech_created(h2, source="say"))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h2)
    second = rec.open_output_stream()
    rec.tap_output_text("one moment", second)
    rec.tap_output_frame(agent_frame(400), second)

    # Only now does s1's LLM produce and its reply render.
    rec.tap_output_text("the real answer", first)
    rec.tap_output_frame(agent_frame(3000), first)
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert set(spans) == {"turn-1", "turn-2"}, (
        "the reply that was stalled behind the LLM must still get its own span"
    )
    assert (spans["turn-1"].get("response") or {}).get("played_ms") == 3000
    assert (spans["turn-2"].get("response") or {}).get("played_ms") == 400
    assert "the real answer" in (spans["turn-1"]["response"])["text"]
    assert "the real answer" not in (spans["turn-2"]["response"])["text"]


async def test_output_rendered_before_its_turn_is_registered_is_held_not_dropped(
        recorder, monkeypatch):
    """A reply can start rendering before the recorder has seen the speech that
    owns it. Resolving that to "no turn" and returning loses the audio and the
    words of the first thing the caller heard, while still counting the bytes
    in the call total -- which surfaces as unattributed audio rather than as a
    transcript."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    handle = FakeSpeechHandle("s1")
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: handle)
    stream = rec.open_output_stream()
    rec.tap_output_text("good morning", stream)
    rec.tap_output_frame(agent_frame(1200), stream)
    # The speech is only reported to the recorder afterwards.
    session.emit("speech_created", speech_created(handle))
    rec.tap_output_frame(agent_frame(300), stream)
    await rec.finish()

    span = _by_type(rec, "tts")
    assert (span.get("response") or {}).get("played_ms") == 1500, (
        "audio rendered before the turn was registered belongs to that turn"
    )
    assert "good morning" in (span["response"])["text"]
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True, capture.get("coverage_gaps")


async def test_ownership_falls_back_to_invocation_time_when_livekit_hides_the_handle(
        recorder, monkeypatch):
    """The speech-handle context is a private LiveKit symbol. When it is not
    available the stream must still pin at *invocation* time, not at first
    output -- invocation happens before the LLM round-trip a competing `say()`
    slips into."""
    from vaani_observer.integrations import livekit as lk

    monkeypatch.setattr(lk, "_current_speech_handle", lambda: None)
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    first = rec.open_output_stream()
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2"), source="say"))
    second = rec.open_output_stream()
    rec.tap_output_frame(agent_frame(400), second)
    rec.tap_output_frame(agent_frame(3000), first)
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert (spans["turn-1"].get("response") or {}).get("played_ms") == 3000
    assert (spans["turn-2"].get("response") or {}).get("played_ms") == 400


async def test_a_reply_is_not_closed_while_its_own_generator_is_still_rendering(
        recorder, monkeypatch):
    """A reply's text commits long before its playout ends -- ~0.6s into a 9s
    reply on a live call -- and the next speech can be authorized in that
    window. Closing the span on either signal publishes a duration shorter
    than the audio the caller heard, and every frame that arrives afterwards
    lands on a turn whose span is already immutable, where it reads as
    unattributed audio instead of as speech."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()
    rec.tap_output_frame(agent_frame(2000), first)
    item = chat_item("assistant", "the whole answer")
    h1.chat_items.append(item.item)
    session.emit("conversation_item_added", item)          # text commits early
    h2 = FakeSpeechHandle("s2")
    session.emit("speech_created", speech_created(h2, source="say"))  # next speech
    # s1's generator is still draining the reply the caller is hearing.
    rec.tap_output_frame(agent_frame(1300), first)
    rec.close_output_stream(first)
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert (spans["turn-1"].get("response") or {}).get("played_ms") == 3300, (
        "the span must cover every millisecond its own generator rendered"
    )
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True, capture.get("coverage_gaps")


async def test_a_provider_metric_with_no_speech_id_is_dropped_rather_than_misattributed(
        recorder, monkeypatch):
    """`speech_id` is optional upstream. Publishing such a metric on the
    current turn is wrong precisely when it matters: during a barge-in the
    current turn is the *new* reply, so the previous reply's provider, model,
    TTFB and billable character count are published as a fully *measured* span
    on a reply that never produced them -- while the reply that did is
    downgraded to reconstructed. Cost is billed off `characters_count`."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()
    rec.tap_output_text("the long answer", first)
    rec.tap_output_frame(agent_frame(2000), first)
    # The caller barges in: a new turn opens and becomes the current one.
    h2 = FakeSpeechHandle("s2")
    session.emit("speech_created", speech_created(h2))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h2)
    second = rec.open_output_stream()
    rec.tap_output_text("ok", second)
    rec.tap_output_frame(agent_frame(500), second)
    # s1's metric finally lands, with no speech_id to prove whose it is.
    session.emit("metrics_collected", tts_metrics(None))
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    second_response = spans["turn-2"].get("response") or {}
    assert second_response.get("characters_count") is None, (
        "an unidentified provider metric must not bill turn-2 for another "
        f"reply's characters: {second_response}"
    )
    assert second_response.get("played_ms") == 500
    assert spans["turn-2"].get("request", {}).get("provider") in (None, "unknown"), (
        spans["turn-2"].get("request")
    )
    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps") or []
    assert any("speech_id" in str(g) for g in gaps), (
        "dropping the metric silently is the same class of failure as "
        f"misattributing it: {gaps}"
    )


async def test_a_short_shared_opening_never_decides_which_reply_owns_a_message(
        recorder, monkeypatch):
    """The text ladder compared only the *prior* reply's tape, so a new reply
    that rendered text before its first frame lost every tie: an interrupted
    tape reading "Sure," matches a different reply beginning "Sure, one
    moment" by prefix, and the message is filed under the wrong turn with full
    confidence. Replies routinely share their opening words, so the prefix
    rule needs both a competitor check and a length floor."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    # No chat_items anywhere, so the negative-proof rung cannot decide.
    h1 = SimpleNamespace(id="s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()
    rec.tap_output_text("Sure,", first)          # interrupted after two words
    rec.tap_output_frame(agent_frame(600), first)
    h2 = SimpleNamespace(id="s2")
    session.emit("speech_created", speech_created(h2, source="say"))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h2)
    second = rec.open_output_stream()
    rec.tap_output_text("Sure, one moment", second)
    session.emit("conversation_item_added", chat_item("assistant", "Sure, one moment"))
    rec.tap_output_frame(agent_frame(700), second)
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    turn_one_text = (spans["turn-1"].get("response") or {}).get("text") or ""
    assert "one moment" not in turn_one_text, (
        "turn-1 was interrupted after 'Sure,' -- the longer reply is not its "
        f"transcript: {turn_one_text!r}"
    )
    turn_two_text = (spans["turn-2"].get("response") or {}).get("text") or ""
    assert "one moment" in turn_two_text, (
        "the reply that taped these exact words must keep them; dropping the "
        "transcript because a short shared opening also matched is the same "
        f"defect pointed the other way: {turn_two_text!r}"
    )
    # Only one reply's tape matches the item, so ownership *is* decidable and
    # the call must not be marked ambiguous. Declaring a gap here would make a
    # healthy call look lossy -- the cost of resolving the tie by refusing to.
    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps") or []
    assert not any("attribut" in str(g) for g in gaps), (
        f"exactly one reply spoke these words; that is not undecidable: {gaps}"
    )


async def test_a_short_interrupted_tape_never_claims_a_longer_reply(recorder,
                                                                    monkeypatch):
    """The competitor check saves the case where both replies have taped text.
    It cannot save this one: the new reply has rendered nothing yet, so the
    interrupted tape `"Sure,"` is the *only* match for `"Sure, one moment"` and
    a bare prefix rule hands it a confident wrong answer. Replies routinely
    open with the same few words, and an interrupted tape is routinely that
    short, so a prefix is only evidence when it is long enough to identify a
    reply."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    # No chat_items anywhere, so the negative-proof rung cannot decide.
    h1 = SimpleNamespace(id="s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()
    rec.tap_output_text("Sure,", first)          # interrupted after two words
    rec.tap_output_frame(agent_frame(600), first)
    h2 = SimpleNamespace(id="s2")
    session.emit("speech_created", speech_created(h2, source="say"))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h2)
    second = rec.open_output_stream()
    # The second reply has rendered nothing yet -- no text and no audio -- so
    # it is the only other candidate and it cannot compete on text. The item
    # must arrive in this window, because once the new reply has audio the
    # attribution is no longer in question.
    session.emit("conversation_item_added", chat_item("assistant", "Sure, one moment"))
    rec.tap_output_frame(agent_frame(700), second)
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    turn_one_text = (spans["turn-1"].get("response") or {}).get("text") or ""
    assert "one moment" not in turn_one_text, (
        "turn-1 was interrupted after 'Sure,' -- a five-character opening is "
        f"not proof that the longer reply is its transcript: {turn_one_text!r}"
    )
    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps") or []
    assert any("attribut" in str(g) for g in gaps), (
        f"an undecidable transcript must be declared, not silently dropped: {gaps}"
    )


async def test_drain_frames_landing_on_a_turn_with_no_span_are_written_off(
        recorder):
    """The ordinary drain tail: a fragment of the previous reply resolves by
    timing onto a turn that never speaks and so never opens a span at all.
    This must stay forgivable. It is deliberately the *narrower* of the two
    write-off cases -- it was once named for the published-span case and
    tested neither, since 120ms sits under the playout tolerance and no span
    is ever built here."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    # A legacy tap with no stream token: the first reply's last fragment
    # resolves by timing onto the next turn, which never replies.
    rec.tap_output_frame(agent_frame(120))
    await rec.finish()

    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is True, capture.get("coverage_gaps")
    measured = capture["measured"]
    assert measured["tail_written_off_ms"] >= 120, measured
    assert measured["tail_written_off_turn_ids"], measured
    assert "unattributed_agent_audio_ms" in measured, (
        "the residual must be published even when it sits under the tolerance"
    )


async def test_a_greeting_never_blocks_the_next_replys_own_llm_metric(recorder):
    """The refusal rule asks "did another reply finish after this one began",
    which is stage-blind. `session.say()` synthesises text that was handed to
    it -- no LLM ever runs for it -- so a greeting cannot be the claimant for
    an LLM measurement, yet it was counted as one. Since the opening greeting
    is retired by the very speech that creates the first real reply, its
    `finished_at_seq` always lands at-or-after that reply's, and so *every*
    unnamed LLM metric on the first generated reply of an agent that greets
    was dropped -- the overwhelmingly common LiveKit shape."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    greeting = FakeSpeechHandle("greet")
    session.emit("speech_created", speech_created(greeting, source="say"))
    session.emit("user_input_transcribed", transcript("kya haal hai", True))
    reply = FakeSpeechHandle("s2")
    session.emit("speech_created", speech_created(reply, source="generate_reply"))
    session.emit("metrics_collected", llm_metrics(None))
    rec.tap_output_frame(agent_frame(900))
    session.emit("conversation_item_added", chat_item("assistant", "theek hoon"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = [op for op in ops if op["type"] == "llm"]
    assert llm, "the reply's own LLM metric was refused because a say() greeting existed"
    assert llm[0]["turn_id"] == rec._turns["s2"].turn.id
    assert llm[0]["response"].get("total_tokens") == 120, llm[0]["response"]
    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps", [])
    assert not [g for g in gaps if g.get("stage") == "llm"], gaps


async def test_a_superseded_user_turn_never_becomes_a_phantom_claimant(recorder):
    """Two final transcripts before any reply -- an endpoint that fires twice,
    which LiveKit answers with a single reply -- replaced `_pending_turn`
    while the first state stayed live in `_all_turns` forever. It can never be
    retired, because retirement runs off the *speaking* turn and this one
    never spoke. From then on every unnamed LLM or TTS metric saw two live
    turns and was refused for the rest of the call: one dropped endpoint
    silently stopped the call from measuring anything again."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("goa ki", True))
    session.emit("user_input_transcribed", transcript("flight kitne ki hai", True))
    reply = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(reply))
    session.emit("metrics_collected", llm_metrics(None))
    rec.tap_output_frame(agent_frame(900))
    session.emit("conversation_item_added", chat_item("assistant", "dus hazaar"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = [op for op in ops if op["type"] == "llm"]
    assert llm, "a superseded pending turn stayed live and refused the reply's metric"
    assert llm[0]["turn_id"] == rec._turns["s1"].turn.id


async def test_a_stream_that_closes_before_its_turn_exists_still_delivers(
        recorder, monkeypatch):
    """`tts_node` can be entered, render a short reply and return before
    LiveKit's `speech_created` for it is dispatched. The stream knew its
    speech id the whole time, so nothing was ever ambiguous -- but the buffer
    was only flushed by a *later* frame, and there is no later frame. The
    reply's audio and its words were dropped from the turn and resurfaced as
    unattributed audio: a complete reply, provably owned, lost to ordering."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    handle = FakeSpeechHandle("s1")
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: handle)
    stream = rec.open_output_stream()
    rec.tap_output_frame(agent_frame(700), stream)
    rec.tap_output_text("namaste", stream)
    rec.close_output_stream(stream)
    session.emit("speech_created", speech_created(handle))
    await rec.finish()

    state = rec._turns["s1"]
    assert state.audio_bytes > 0, (
        "the reply's audio never reached the turn that provably owned it"
    )
    assert "namaste" in "".join(state.tts_text), state.tts_text
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["unattributed_agent_audio_ms"] == 0, measured


async def test_a_trailing_metric_never_lands_on_a_turn_that_cannot_produce_it(
        recorder):
    """Eligibility guarded the candidate search but not the fallback beneath
    it. When no eligible reply is open the recorder treats an unnamed metric
    as the trailing measurement of the reply that just ended and publishes it
    on `_current_turn` -- without asking whether that turn could have produced
    it. A `say()` greeting following a generated reply therefore collected the
    reply's LLM tokens, latency and cost as a fully measured span, on a turn
    where no model was ever called, with no gap to say so."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session)
    greeting = FakeSpeechHandle("g1")
    session.emit("speech_created", speech_created(greeting, source="say"))
    rec.tap_output_frame(agent_frame(600))
    session.emit("conversation_item_added", chat_item("assistant", "aur kuch"))
    session.emit("metrics_collected", llm_metrics(None))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    say_turn = rec._turns["g1"].turn.id
    stray = [o for o in ops if o["type"] == "llm" and o["turn_id"] == say_turn]
    assert not stray, (
        "a say() turn was billed for an LLM call it never made: %r" % stray
    )


async def test_a_stream_is_resolved_even_when_a_metric_registered_its_turn(
        recorder, monkeypatch):
    """A metric that *does* carry its speech id registers the turn under that
    id. When `speech_created` then arrives it takes the already-registered
    branch, which returned without ever looking at the streams waiting for
    that speech. A reply that finished rendering before its event was
    dispatched stayed stranded, and its audio and words never reached it."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    handle = FakeSpeechHandle("s1")
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: handle)
    stream = rec.open_output_stream()
    rec.tap_output_frame(agent_frame(700), stream)
    rec.tap_output_text("namaste", stream)
    rec.close_output_stream(stream)
    # The named metric registers the turn before LiveKit dispatches the event.
    session.emit("metrics_collected", llm_metrics("s1"))
    session.emit("speech_created", speech_created(handle))
    await rec.finish()

    state = rec._turns["s1"]
    assert state.audio_bytes > 0, "the reply's audio never reached its own turn"
    assert "namaste" in "".join(state.tts_text), state.tts_text


async def test_a_generator_entered_before_its_speech_event_is_not_retired(recorder,
                                                                          monkeypatch):
    """`tts_node` is entered before it yields, and on a busy loop it can be
    entered before LiveKit dispatches the `speech_created` for the very speech
    it is rendering. The stream could not be counted against a turn that did
    not exist yet, so the reply looked idle: the next speech retired it and
    closed its span, and everything it went on to say landed on an operation
    that was already ended. Registering the stream at invocation only helps if
    something resolves it once the turn appears."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    stream = rec.open_output_stream()  # entered; the speech event has not landed
    session.emit("speech_created", speech_created(h1))
    turn_one = rec._turns["s1"]
    assert turn_one.open_streams == 1, (
        "the running generator was never counted against its reply"
    )
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    assert turn_one.finished is False, (
        "a reply whose generator is still running was retired by the next speech"
    )
    rec.tap_output_frame(agent_frame(700), stream)
    rec.tap_output_text("namaste", stream)
    rec.close_output_stream(stream)
    await rec.finish()

    assert turn_one.audio_bytes > 0
    assert "namaste" in "".join(turn_one.tts_text), turn_one.tts_text


async def test_a_reply_is_not_retired_while_its_generator_waits_on_the_llm(
        recorder, monkeypatch):
    """The stream was counted against its turn on first *output*, not on
    invocation. `tts_node` is entered a full LLM round-trip before it yields,
    so for that entire window the reply looked idle: the next `speech_created`
    retired it and closed its span, and everything it went on to say was
    published against the wrong reply -- or against a span that was already
    ended and could take no more."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()  # invoked; the LLM has not answered yet

    turn_one = rec._turns["s1"]
    assert turn_one.open_streams == 1, (
        "the stream must be counted at invocation, or the reply looks idle "
        "for the whole LLM round-trip"
    )
    h2 = FakeSpeechHandle("s2")
    session.emit("speech_created", speech_created(h2, source="say"))
    assert turn_one.finished is False, (
        "a reply whose generator is still running was retired by the next "
        "speech, so its remaining audio had nowhere to go"
    )
    rec.tap_output_text("the real answer", first)
    rec.tap_output_frame(agent_frame(3000), first)
    rec.close_output_stream(first)
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert (spans["turn-1"].get("response") or {}).get("played_ms") == 3000, spans


async def test_the_timing_fallback_also_holds_its_reply_open(recorder, monkeypatch):
    """The deferral was wired only into the identity path. On a livekit-agents
    build that hides the speech context the stream was attached directly and
    never counted, so close-deferral -- the fix for the defect above -- was
    silently inactive on exactly the versions that need it most."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: None)
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    stream = rec.open_output_stream()

    turn_one = rec._turns["s1"]
    assert turn_one.open_streams == 1, "the fallback path must count its stream too"
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    assert turn_one.finished is False
    rec.tap_output_frame(agent_frame(1200), stream)
    rec.close_output_stream(stream)
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["stream_ownership"] == "inferred", (
        "a call whose attribution rests on timing must say so; a reader "
        f"cannot otherwise tell it from one that was proved: {measured}"
    )


async def test_a_call_whose_streams_were_all_identified_says_so(recorder, monkeypatch):
    """The provenance field is only worth publishing if it can be `proved`."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    h1 = FakeSpeechHandle("s1")
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(h1))
    stream = rec.open_output_stream()
    rec.tap_output_text("an answer of a reasonable length", stream)
    rec.tap_output_frame(agent_frame(1000), stream)
    rec.close_output_stream(stream)
    session.emit("metrics_collected", tts_metrics("s1"))
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["stream_ownership"] == "proved", measured


async def test_an_unnamed_metric_is_dropped_but_its_reply_is_still_published(recorder):
    """Once a second reply exists, a metric that cannot name its reply is
    genuinely ambiguous: `llm` and `tts` metrics are additive, so the reply
    that just ended may still owe one. The metric is refused.

    What must survive is the *reply*. The span is rebuilt from the tape and the
    transcript, marked estimated, and the loss is named -- so the failure mode
    is a disclosed missing character count, never a missing reply (the audit's
    P0-A) and never a character count billed to the wrong turn."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("one", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(1000))
    session.emit("metrics_collected", tts_metrics(None))
    session.emit("conversation_item_added", chat_item("assistant", "the first answer"))
    session.emit("user_input_transcribed", transcript("two", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    rec.tap_output_frame(agent_frame(1000))
    session.emit("metrics_collected", tts_metrics(None, ttfb=0.4))
    session.emit("conversation_item_added", chat_item("assistant", "the second answer"))
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    assert len(spans) == 2, "a refused metric must never cost a reply its span"
    first = spans["turn-1"].get("response") or {}
    assert first.get("characters_count") is not None, (
        f"the opening reply had no competitor and must keep its metric: {first}"
    )
    second = spans["turn-2"].get("response") or {}
    assert second.get("characters_count") is None, (
        "this metric could equally have been the previous reply's second "
        f"segment; publishing it here bills one reply for another: {second}"
    )
    assert second.get("text") == "the second answer", second
    assert second.get("played_ms") == 1000, (
        f"the reply's duration is measured from its own audio: {second}"
    )
    assert second.get("estimated") is True, (
        f"a rebuilt span must not present itself as a measured one: {second}"
    )
    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps") or []
    assert any(g.get("stage") == "tts" and "speech_id" in str(g) for g in gaps), (
        f"the refusal must be declared, not silent: {gaps}"
    )


async def test_a_dropped_metric_reports_the_stage_it_came_from(recorder, monkeypatch):
    """Every dropped metric was reported as a `tts` gap, whatever it was. An
    operator reading "tts metrics were dropped" goes looking for missing audio;
    the LLM timings that actually went missing are not where they look."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    first = rec.open_output_stream()
    rec.tap_output_text("the long answer", first)
    rec.tap_output_frame(agent_frame(2000), first)
    h2 = FakeSpeechHandle("s2")
    session.emit("speech_created", speech_created(h2))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h2)
    second = rec.open_output_stream()
    rec.tap_output_frame(agent_frame(500), second)
    session.emit("metrics_collected", llm_metrics(None))
    await rec.finish()

    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps") or []
    assert any(g.get("stage") == "llm" and "speech_id" in str(g) for g in gaps), (
        f"the gap must name the stage whose measurement was lost: {gaps}"
    )


async def test_audio_on_an_identified_stream_is_never_written_off_as_jitter(recorder,
                                                                           monkeypatch):
    """The write-off exists for frames whose owner was guessed from timing at a
    turn boundary. Applying it to a stream that *names* its reply turns the
    allowance into a blindfold: a lifecycle bug that strands part of a
    tokenized reply is precisely what it would hide, and hiding it is how a
    100% undercount ends up behind a green checkmark."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    stream = rec.open_output_stream()
    rec.tap_output_text("a first answer of a reasonable length", stream)
    rec.tap_output_frame(agent_frame(2000), stream)
    session.emit("metrics_collected", tts_metrics("s1"))
    rec.close_output_stream(stream)
    session.emit("conversation_item_added",
                 chat_item("assistant", "a first answer of a reasonable length"))
    # The next reply starts, so the first one's span is published and closed.
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    # More frames now arrive on the very same stream: they name their reply, so
    # this is a lifecycle defect and not the boundary jitter the allowance was
    # written for.
    rec.tap_output_frame(agent_frame(120), stream)
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["tail_written_off_ms"] == 0, (
        "audio on a stream that named its reply is not boundary jitter: "
        f"{measured}"
    )
    assert measured["unattributed_agent_audio_ms"] >= 120, measured


async def test_a_replys_second_segment_is_never_added_to_the_reply_after_it(recorder):
    """TTS metrics are additive: one reply emits one per synthesis segment and
    `_record_tts` sums them. So "this reply already got a metric" is not proof
    that it got its last one -- and treating it as proof handed a reply's
    second segment to the reply that followed it. Nothing is dropped, so no
    coverage gap appears: two wrong per-turn durations and character counts
    behind a fully healthy call, which is the exact failure class this module
    exists to refuse."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("one", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    # Three seconds spoken, but only the first segment measured so far.
    rec.tap_output_frame(agent_frame(3000))
    session.emit("metrics_collected", tts_metrics(None, audio_duration=1.0))
    session.emit("conversation_item_added", chat_item("assistant", "the first answer"))
    session.emit("user_input_transcribed", transcript("two", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    rec.tap_output_frame(agent_frame(400))
    # The first reply's remaining segment, arriving late and unnamed.
    session.emit("metrics_collected", tts_metrics(None, audio_duration=2.0))
    await rec.finish()

    spans = {o["turn_id"]: o for o in _all_of_type(rec, "tts")}
    second = spans["turn-2"].get("response") or {}
    assert (second.get("audio_ms") or 0) < 2000, (
        "turn-2 spoke 400ms and was credited with the previous reply's "
        f"segment: {second}"
    )
    assert second.get("characters_count") is None, (
        f"turn-2 was billed for characters another reply synthesized: {second}"
    )
    gaps = _manifest_of(rec)["capture_status"].get("coverage_gaps") or []
    assert any(g.get("stage") == "tts" for g in gaps), (
        f"a metric that could not name its reply must be declared: {gaps}"
    )


async def test_early_untokenized_audio_never_excuses_a_later_stranded_frame(recorder,
                                                                           monkeypatch):
    """The write-off was capped by the turn's *lifetime* unscoped audio, but
    every frame taped before the span closed is already inside `played_ms` and
    cannot be part of the residual. So a turn that once took an untokenized
    frame bought itself an allowance it could spend later -- on audio that
    named its reply. That is a lifecycle defect wearing boundary jitter's
    clothes."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    stream = rec.open_output_stream()
    rec.tap_output_text("a first answer of a reasonable length", stream)
    rec.tap_output_frame(agent_frame(2000), stream)
    # An untokenized frame, before the span is published: already accounted for.
    rec.tap_output_frame(agent_frame(500))
    session.emit("metrics_collected", tts_metrics("s1"))
    rec.close_output_stream(stream)
    session.emit("conversation_item_added",
                 chat_item("assistant", "a first answer of a reasonable length"))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    # Stranded afterwards on the stream that names its reply.
    rec.tap_output_frame(agent_frame(500), stream)
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["tail_written_off_ms"] == 0, (
        "the residual is identified-stream audio; an earlier untokenized frame "
        f"is not a licence to forgive it: {measured}"
    )
    assert measured["unattributed_agent_audio_ms"] >= 500, measured


async def test_audio_placed_without_a_stream_token_is_reported_as_inferred(recorder):
    """`stream_ownership` answers "was per-turn audio bound by identity or by
    timing", and a frame tapped with no stream at all is bound by timing --
    whichever reply is rendering *now* takes it, so a reply's tail draining
    past the next one's start lands on the wrong turn. Reporting that as
    `proved` tells a reader the per-turn split is trustworthy when the audit's
    whole complaint is numbers that do not say how sure they are."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(1000))
    session.emit("conversation_item_added", chat_item("assistant", "an answer"))
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["stream_ownership"] == "inferred", (
        f"this audio was placed by timing, not by identity: {measured}"
    )


async def test_a_span_ended_by_an_error_cannot_forgive_its_own_audio(recorder):
    """The write-off measures a residual against what a span published. A span
    ended by the error path never publishes that accounting, so its baseline is
    zero -- which reads as "every millisecond of this reply arrived after it
    was published" and forgives a whole reply's worth of speech that no
    operation accounts for, behind a complete-looking call."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    session.emit("metrics_collected", tts_metrics("s1"))
    # Untokenized audio, so it would otherwise be eligible for the allowance.
    rec.tap_output_frame(agent_frame(800))
    session.emit("error", SimpleNamespace(error=RuntimeError("tts socket reset"),
                                          source=None))
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["tail_written_off_ms"] == 0, (
        "a span that published no accounting offers nothing to measure a "
        f"residual against; forgiving it hides a whole reply: {measured}"
    )
    assert measured["unattributed_agent_audio_ms"] >= 800, measured


async def test_a_stream_that_never_speaks_does_not_downgrade_the_call(recorder,
                                                                     monkeypatch):
    """`stream_ownership` says whether per-turn numbers rest on identity or on
    timing. A `tts_node` invocation that is cancelled before yielding a frame
    placed no audio at all, so it moved no number -- reporting `inferred` for
    it would make the flag mean "something might have happened", and a flag
    that fires on healthy calls is one operators learn to ignore."""
    from vaani_observer.integrations import livekit as lk

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    h1 = FakeSpeechHandle("s1")
    session.emit("speech_created", speech_created(h1))
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: h1)
    spoken = rec.open_output_stream()
    rec.tap_output_frame(agent_frame(1000), spoken)
    # A second invocation with no context to name its reply -- cancelled before
    # it renders anything.
    monkeypatch.setattr(lk, "_current_speech_handle", lambda: None)
    silent = rec.open_output_stream()
    rec.close_output_stream(silent)
    session.emit("conversation_item_added", chat_item("assistant", "an answer"))
    await rec.finish()

    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["stream_ownership"] == "proved", (
        f"no audio was placed by timing, so nothing here was inferred: {measured}"
    )


async def test_a_recorder_that_measured_nothing_is_never_called_fully_captured(recorder):
    """With no audio tap, "every millisecond we taped is on a span" is
    trivially true because nothing was taped. Gating that check on whether an
    agent was passed meant a recorder attached without `agent=` at all -- no
    tap, `agent_audio_ms: 0`, nothing independently verified -- sailed through
    the audit green, which is the least deserving call on the platform to be
    showing a healthy status."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)  # no agent=, so no tap is ever installed
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    session.emit("metrics_collected", tts_metrics("s1"))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False
    assert any(g.get("agent_audio_tapped") is False
               for g in capture["coverage_gaps"]), capture["coverage_gaps"]


async def test_the_write_off_does_not_grow_with_the_length_of_the_call(recorder):
    """Scaling the boundary-jitter allowance with call duration is the wrong
    shape: jitter comes from how many turn boundaries there were, not from how
    long the call ran. At 2% an hour-long call could bury over a minute of
    measured speech behind a green status -- the audit's headline defect
    reached by arithmetic rather than by a bug."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(600000))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=600.0))
    # Ten barge-in tails, each under the per-turn floor, on a ten-minute call.
    for index in range(10):
        session.emit("user_input_transcribed", transcript(f"q{index}", True))
        session.emit("speech_created", speech_created(FakeSpeechHandle(f"t{index}")))
        rec.tap_output_frame(agent_frame(240))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["measured"]["tail_written_off_ms"] <= 1000, (
        "the write-off must not scale with the call"
    )
    assert capture["coverage_complete"] is False, (
        "2.4s of measured speech sits behind no span; a ten-minute call must "
        "not be allowed a larger allowance than a one-minute call"
    )


async def test_the_manifest_says_how_much_of_the_call_was_reconstructed(recorder):
    """`coverage_complete` answers "did this call lose data", and a rebuilt span
    lost none. The share that was rebuilt is a separate fact, and it has to be
    a number rather than a status bit -- roughly three Deepgram replies in four
    emit no metric, so folding it into the status would fire on healthy
    calls."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(1000))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=1.0))
    session.emit("user_input_transcribed", transcript("more", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    rec.tap_output_frame(agent_frame(3000))  # no metrics: rebuilt
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["derived_tts_share_pct"] == 75, measured
    assert _manifest_of(rec)["capture_status"]["coverage_complete"] is True


async def test_a_span_the_provider_only_confirmed_late_is_still_marked_estimated(recorder):
    """Late metrics replace a rebuilt span's timings, but its provider, model
    and start time were fixed when it was opened and stay reconstructed. Letting
    it out of the estimate count would quietly present it as fully measured --
    and the disclosure is the whole basis on which reconstruction is honest."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("conversation_item_added",
                 chat_item("assistant", "a rebuilt reply", {}))
    # The provider's metric turns up while the estimate is still open.
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=2.0))
    await rec.finish()
    measured = _manifest_of(rec)["capture_status"]["measured"]
    assert measured["derived_tts_op_count"] == 1, (
        "the span was opened from a reconstruction and its identity still is "
        "one, whatever arrived afterwards"
    )
    assert _by_type(rec, "tts")["response"]["estimated_fields"] == (
        "provider,model,started_at")


async def test_metrics_that_arrive_too_late_to_record_are_declared_missing(recorder):
    """A later segment's metric arriving after the reply was published is
    discarded, because recording it would report the reply twice. That is the
    right call and it is also a hole in the billable character count -- a
    process log cannot be read by anything downstream, so a bill computed from
    this package would be low while the page called the call complete."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=2.0))
    # The reply's text commits, which closes its span for good.
    session.emit("conversation_item_added", chat_item("assistant", "done", {}))
    session.emit("user_input_transcribed", transcript("next", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s2")))
    # The first reply's second segment reports after its span was published.
    session.emit("metrics_collected", tts_metrics("s1", audio_duration=1.5))
    await rec.finish()
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False
    assert any("later segments" in g.get("reason", "")
               for g in capture["coverage_gaps"]), capture["coverage_gaps"]


# ------------------------------------------------- preemptive reply generation


async def _preemptive_reply(session, rec, speech_id, *, final_at_end=True):
    """The event order LiveKit produces with `preemptive_generation` enabled.

    The reply is generated from a *predicted* end of turn, so `speech_created`
    arrives while the caller is still speaking -- before the final transcript.
    """
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("goa ki", False))
    session.emit("speech_created", speech_created(SimpleNamespace(id=speech_id)))
    if final_at_end:
        session.emit("metrics_collected", stt_metrics())
        session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
        session.emit("user_input_transcribed", transcript("goa ki flight kitne ki hai", True))


async def test_a_preemptively_generated_reply_stays_in_the_turn_it_answers(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await _preemptive_reply(session, rec, "speech-p1")
    session.emit("metrics_collected", llm_metrics("speech-p1"))
    rec.tap_output_frame(agent_frame(900))
    session.emit("metrics_collected", tts_metrics("speech-p1"))
    session.emit("conversation_item_added", chat_item("assistant", "chhah hazaar"))
    await rec.finish()

    states = rec._all_turns
    assert len(states) == 1, "one exchange must not be split across two turns"
    only = states[0]
    assert only.stt is not None, "the caller's words belong to the turn that answered them"
    assert only.llm, "the reply's LLM measurement belongs to the same turn"
    assert only.tts is not None
    assert only.audio_bytes > 0


async def test_a_cancelled_preemptive_attempt_leaves_no_phantom_turn(recorder):
    """LiveKit cancels and regenerates a preemptive reply as the transcript
    grows (`max_retries` is 3). A cancelled attempt never speaks and never
    reports a metric, so it must not surface as a turn of its own."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("goa", False))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-try1")))
    session.emit("user_input_transcribed", transcript("goa ki flight", False))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-try2")))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_input_transcribed", transcript("goa ki flight kitne ki hai", True))
    session.emit("metrics_collected", llm_metrics("speech-try2"))
    rec.tap_output_frame(agent_frame(900))
    session.emit("metrics_collected", tts_metrics("speech-try2"))
    await rec.finish()

    assert len(rec._all_turns) == 1, "cancelled attempts must not each become a turn"


async def test_a_preemptive_turn_is_released_once_its_utterance_lands(recorder):
    """The merge is keyed to the utterance the reply was generated from. Once
    that utterance has been claimed, the *next* one must open its own turn --
    otherwise a single preemptive reply would swallow the rest of the call.

    (A partial that never reaches a final is not a separate utterance: LiveKit
    replaces partials in place, so the final that eventually arrives closes the
    same one. That case merges by design, not by accident.)"""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await _preemptive_reply(session, rec, "speech-p1")
    session.emit("metrics_collected", llm_metrics("speech-p1"))
    rec.tap_output_frame(agent_frame(900))
    session.emit("metrics_collected", tts_metrics("speech-p1"))
    # a second utterance, answered the ordinary way
    await run_one_turn(rec, session, speech_id="speech-2")
    await rec.finish()

    states = rec._all_turns
    assert len(states) == 2, "the next utterance must not join the preemptive turn"
    assert all(st.stt is not None for st in states)
    assert all(st.llm for st in states)


async def test_back_to_back_preemptive_replies_stay_in_separate_turns(recorder):
    """`preemptive_generation` is on by default, so *every* exchange can take
    this path. The second reply's speech is created while the first turn is
    still the pending one, and it must not join it."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    for n, sid in enumerate(("speech-p1", "speech-p2")):
        await _preemptive_reply(session, rec, sid)
        session.emit("metrics_collected", llm_metrics(sid))
        rec.tap_output_frame(agent_frame(600))
        session.emit("metrics_collected", tts_metrics(sid))
    await rec.finish()

    states = rec._all_turns
    assert len(states) == 2, "two exchanges must stay two turns"
    for st in states:
        assert st.stt is not None and st.llm and st.tts is not None


async def test_a_greeting_before_the_caller_speaks_still_opens_its_own_turn(recorder):
    """The opening greeting is not preemptive -- nobody has spoken yet -- so it
    must not swallow the caller's first utterance."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-hello")))
    rec.tap_output_frame(agent_frame(400))
    session.emit("metrics_collected", tts_metrics("speech-hello"))
    await run_one_turn(rec, session, speech_id="speech-1")
    await rec.finish()

    states = rec._all_turns
    assert len(states) == 2
    assert states[0].stt is None, "the greeting answers nothing"
    assert states[1].stt is not None, "the caller's first utterance keeps its own turn"
