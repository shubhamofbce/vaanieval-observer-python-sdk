"""The LiveKit Agents integration, driven by fake LiveKit events.

These tests deliberately do not import `livekit.agents`: the recorder only ever
reads attributes off the event objects, so duck-typed stand-ins pin the exact
contract without pulling a heavyweight optional dependency into the suite.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import time
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import operations, read_events
from vaani_observer import VaaniObserver
from vaani_observer.integrations import livekit as livekit_integration
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

    def generate_reply(self, *, input_modality: str = "text",
                       handle_id: str = "app-reply"):
        """What `AgentSession.generate_reply()` does that the recorder can see.

        The event is emitted synchronously inside the call and the handle is
        returned, and `input_modality` is passed through unchanged -- so with
        `"audio"` every field matches LiveKit's own automatic answer.
        """
        handle = FakeSpeechHandle(handle_id, modality=input_modality)
        self.emit("speech_created", speech_created(handle))
        return handle


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

    def __init__(self, handle_id: str, scheduled: bool = True,
                 modality: str = "audio"):
        self.id = handle_id
        self.chat_items: list = []
        # `SpeechHandle.create` defaults to audio input details, and the
        # automatic reply to a completed turn passes them explicitly.
        # `AgentSession.generate_reply()` defaults to "text", which is what
        # separates an application's own reply from the framework's.
        self.input_details = SimpleNamespace(modality=modality)
        # LiveKit schedules a generated reply in the same synchronous frame as
        # `speech_created` (`agent_activity.py:1575` -> `_mark_scheduled`), so
        # by the time anything reads the handle an ordinary reply says True.
        # Preemptive generation is the exception: it passes
        # `schedule_speech=False` and stays unscheduled until the predicted
        # turn is validated, which is what tells the two apart.
        self.scheduled = scheduled

    def add_done_callback(self, callback):  # pragma: no cover - not used here
        self._done = callback


def speech_created(handle, source: str = "generate_reply",
                   user_initiated: bool = True):
    # A realtime model's server-side generation is the one that reports
    # `user_initiated=False`; every other speech LiveKit creates reports True.
    return SimpleNamespace(speech_handle=handle, source=source,
                           user_initiated=user_initiated)


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
    # The recogniser's websocket meter, not this turn's speech.
    assert stt["response"]["provider_metered_audio_ms"] == 3500
    assert "audio_ms" not in stt["response"]
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
    # LiveKit commits the first utterance before the second one starts, which
    # is what makes these two turns rather than one utterance delivered as two
    # finals. Without the commit they would -- correctly -- share a span.
    session.emit("conversation_item_added", chat_item("user", "first"))
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
    """A committed utterance that LiveKit never answered -- it can decline to
    reply -- left `_pending_turn` replaced while the first state stayed live in
    `_all_turns` forever. It can never be retired, because retirement runs off
    the *speaking* turn and this one never spoke. From then on every unnamed
    LLM or TTS metric saw two live turns and was refused for the rest of the
    call: one unanswered utterance silently stopped the call from measuring
    anything again."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("goa ki", True))
    # The commit is what makes these two turns. Without it LiveKit would merge
    # them into one user message, and the recorder follows that boundary.
    session.emit("conversation_item_added", chat_item("user", "goa ki"))
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


async def test_a_reply_that_never_made_a_sound_is_not_reported_as_missing_audio(recorder):
    """The call ends while the agent is still being spoken: its words were
    taped, no audio was rendered, and no TTS metric arrived. That is a missing
    *rendering*, and reporting it as unaccounted-for audio sent an operator
    looking for something that never existed -- alongside a payload that said
    `unattributed_agent_audio_ms: 0` in the same breath."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_one_turn(rec, session, speech_id="speech-1")
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_input_transcribed", transcript("aur hotel", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-2")))
    session.emit("metrics_collected", llm_metrics("speech-2"))
    # `tts_node` began emitting the reply's words, then the room closed before
    # a single frame was rendered and before LiveKit added the chat item.
    stream = rec.open_output_stream()
    stream.speech_id = "speech-2"
    rec.tap_output_text("hotel bhi dekh lete hain", stream=stream)
    rec.close_output_stream(stream)
    await rec.finish()

    capture = _manifest_of(rec)["capture_status"]
    gaps = capture.get("coverage_gaps") or []
    silent = [g for g in gaps if "never rendered as audio" in g["reason"]]
    assert silent, f"the unrendered reply must be named for what it is: {gaps}"
    assert silent[0]["turn_ids"] == ["turn-2"]
    assert "unattributed_agent_audio_ms" not in silent[0], (
        "a reply that made no sound has no unattributed audio to report")
    assert not [g for g in gaps if "audio was rendered that no tts" in g["reason"]], (
        "no audio was rendered, so that gap must not fire")


async def test_a_filler_said_while_the_caller_talks_does_not_swallow_their_turn(recorder):
    """`say()` reports `source="say"` and is spoken *at* the caller -- a
    "let me look that up" filler while they are still talking answers nothing.
    Only a generated reply can be preemptive, so the filler must keep its own
    turn and leave the caller's utterance to the reply that answers it."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("goa ki", False))
    # a filler, spoken over the caller
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-say"), source="say"))
    rec.tap_output_frame(agent_frame(300))
    session.emit("metrics_collected", tts_metrics("speech-say"))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit("user_input_transcribed", transcript("goa ki flight kitne ki hai", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-1")))
    session.emit("metrics_collected", llm_metrics("speech-1"))
    rec.tap_output_frame(agent_frame(800))
    session.emit("metrics_collected", tts_metrics("speech-1"))
    await rec.finish()

    states = rec._all_turns
    assert len(states) == 2, "the filler must not merge with the caller's turn"
    assert states[0].stt is None, "a filler answers nothing"
    assert not states[0].llm, "`say()` runs no LLM"
    assert states[1].stt is not None and states[1].llm, (
        "the caller's words belong to the reply that answered them")


# ------------------------------------------------- one utterance, many finals


async def test_one_utterance_delivered_as_several_finals_is_a_single_turn(recorder):
    """LiveKit's boundary is the commit, so pre-commit finals share a turn.

    A provider is free to end a transcript at every sentence. LiveKit merges
    those into one user message and answers it once; recording a turn per
    final invented two turns whose caller was never answered.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("Thanks.", True))
    session.emit("user_input_transcribed", transcript("That is all I needed.", True))
    session.emit("user_input_transcribed", transcript("Goodbye.", True))
    session.emit("conversation_item_added",
                 chat_item("user", "Thanks. That is all I needed. Goodbye."))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-1")))
    session.emit("metrics_collected", llm_metrics("speech-1"))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("speech-1"))
    await rec.finish()
    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert len(stt_ops) == 1
    stt = _by_type(rec, "stt")
    assert stt["response"]["transcript"] == "Thanks. That is all I needed. Goodbye."
    assert stt["response"]["final_segments"] == 3


async def test_a_committed_utterance_never_absorbs_the_next_one(recorder):
    """Once LiveKit commits the message, the next final starts a new turn."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("first", True))
    session.emit("conversation_item_added", chat_item("user", "first"))
    session.emit("user_input_transcribed", transcript("second", True))
    session.emit("conversation_item_added", chat_item("user", "second"))
    await rec.finish()
    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert len({op["turn_id"] for op in stt_ops}) == 2


async def test_an_answered_turn_never_absorbs_the_next_utterance(recorder):
    """A turn that already has a reply is finished collecting the caller."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("first", True))
    # Frames, not a metric: `speech_created` clears the pending turn, so audio
    # arriving on its own is the way a still-pending turn can already have
    # spoken -- and a turn that has spoken is done collecting the caller.
    rec.tap_output_frame(agent_frame(2000))
    session.emit("user_input_transcribed", transcript("second", True))
    await rec.finish()
    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert len({op["turn_id"] for op in stt_ops}) == 2


async def test_a_reply_the_caller_talks_past_keeps_its_utterance_whole(recorder):
    """The shape a real LiveKit call actually produces.

    LiveKit answers a *provisional* end of turn, so it creates a reply speech
    after every final transcript and cancels the ones the caller talks past.
    Only the last attempt survives to say anything. Recorded naively that is
    three turns: two whose caller was never answered, and one reply with no
    question -- which is what a live call produced before this.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("Thanks.", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="cancelled-1")))
    session.emit("user_input_transcribed", transcript("That is all I needed.", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="cancelled-2")))
    session.emit("user_input_transcribed", transcript("Goodbye.", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="spoken")))
    session.emit("conversation_item_added",
                 chat_item("user", "Thanks. That is all I needed. Goodbye."))
    session.emit("metrics_collected", llm_metrics("spoken"))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("spoken"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    stt_ops = [op for op in ops if op["type"] == "stt"]
    assert len(stt_ops) == 1, "the caller's one message was recorded as several turns"
    llm_ops = [op for op in ops if op["type"] == "llm"]
    assert llm_ops[0]["turn_id"] == stt_ops[0]["turn_id"], \
        "the reply landed in a different turn from the question it answered"
    stt = _by_type(rec, "stt")
    assert stt["response"]["transcript"] == "Thanks. That is all I needed. Goodbye."


async def test_a_filler_between_the_commit_and_the_next_final_keeps_turns_apart(recorder):
    """The commit is reported against the current turn, not the collecting one.

    A `say()` spoken between the caller's message and its commit makes its own
    turn current, so the commit lands there. The caller's message is committed
    all the same, and whatever they say next is a new turn.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_input_transcribed", transcript("first", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="filler"), source="say"))
    session.emit("conversation_item_added", chat_item("user", "first"))
    session.emit("user_input_transcribed", transcript("second", True))
    await rec.finish()
    stt_ops = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"]
    assert len({op["turn_id"] for op in stt_ops}) == 2


class StopResponse(Exception):
    """LiveKit's own signal, by name, for an utterance the agent ignores."""


class FakeAgent:
    def __init__(self, raises: BaseException | None = None) -> None:
        self.raises = raises
        self.calls: list[Any] = []

    async def on_user_turn_completed(self, turn_ctx, new_message):
        self.calls.append(new_message)
        if self.raises is not None:
            raise self.raises
        return "agent-result"


@pytest.mark.asyncio
async def test_an_utterance_the_agent_ignores_is_not_prepended_to_the_next_one(recorder):
    """`StopResponse` ends a user turn that is never committed.

    LiveKit has already returned `True` from `on_end_of_turn` by then, so it
    clears the accumulated transcript and the next final starts a fresh message
    -- but no `conversation_item_added` is ever emitted for the ignored one.
    Keyed only on that event, the recording merged the two: one turn whose
    transcript carried words the agent was told to discard.
    """
    rec = recorder()
    session = FakeAgentSession()
    agent = FakeAgent(raises=StopResponse())
    rec.attach(session)
    rec.watch_agent(agent)

    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is my card number?", True))
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(None, chat_item("user", "What is my card number?"))

    session.emit("user_input_transcribed", transcript("What is the cheapest fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the cheapest fare?"))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-1")))
    session.emit("metrics_collected", llm_metrics("speech-1"))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("speech-1"))
    await rec.finish()

    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert len(stt_ops) == 2, "the ignored utterance is a turn of its own"
    texts = [op["response"]["transcript"] for op in stt_ops]
    assert texts == ["What is my card number?", "What is the cheapest fare?"]
    assert stt_ops[0]["response"]["reply_skipped"] == "stop_response"
    assert "reply_skipped" not in stt_ops[1]["response"]


@pytest.mark.asyncio
async def test_watching_an_agent_never_changes_what_it_returns_or_raises(recorder):
    """The hook wraps a method the application wrote, so it has to be invisible."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    agent = FakeAgent()
    rec.watch_agent(agent)
    assert await agent.on_user_turn_completed(None, chat_item("user", "hi")) == "agent-result"
    assert len(agent.calls) == 1

    failing = FakeAgent(raises=ValueError("boom"))
    rec.watch_agent(failing)
    with pytest.raises(ValueError, match="boom"):
        await failing.on_user_turn_completed(None, chat_item("user", "hi"))

    # Wiring the recorder twice must not stack wrappers: each layer would book
    # the same turn again and the chain would grow for the life of the call.
    wrapped = agent.on_user_turn_completed
    rec.watch_agent(agent)
    rec.watch_agent(agent)
    assert agent.on_user_turn_completed is wrapped
    assert await agent.on_user_turn_completed(None, chat_item("user", "hi")) == "agent-result"
    assert len(agent.calls) == 2
    await rec.finish()


@pytest.mark.asyncio
async def test_a_reply_created_after_the_commit_still_answers_its_own_question(recorder):
    """The reply speech and the committed user item can arrive in either order.

    Releasing the collecting turn on the commit made adoption depend on that
    ordering, and on the losing order the caller's words sat in a turn with no
    reply while the reply sat in a turn with no question.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("Thanks.", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-1")))
    session.emit("user_input_transcribed", transcript("That is all I needed.", True))
    session.emit("conversation_item_added",
                 chat_item("user", "Thanks. That is all I needed."))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-2")))
    session.emit("metrics_collected", llm_metrics("speech-2"))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("metrics_collected", tts_metrics("speech-2"))
    await rec.finish()

    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert len(stt_ops) == 1
    turn_id = stt_ops[0]["turn_id"]
    kinds = {e["type"] for e in operations(read_events(_dir(rec))) if e["turn_id"] == turn_id}
    assert {"stt", "llm", "tts"} <= kinds, "the reply belongs to the question it answered"


@pytest.mark.asyncio
async def test_a_filler_said_over_the_caller_never_becomes_their_answer(recorder):
    """`say()` is spoken *at* the caller, not in answer to them.

    Adopting it would bill its audio to the caller's unfinished message. The
    filler does end that message's span, so the utterance is recorded as two
    turns where LiveKit had one -- but every word survives, in the turn that
    heard it, and the filler stays out of both.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("Thanks.", True))
    # The provisional reply to that final takes the pending turn, so the filler
    # arrives at the branch that adopts an unclaimed speech.
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-1")))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1"))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_input_transcribed", transcript("That is all I needed.", True))
    session.emit("conversation_item_added",
                 chat_item("user", "Thanks. That is all I needed."))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    stt_ops = [e for e in ops if e["type"] == "stt"]
    # Not one turn -- but nothing is lost and nothing is attributed twice.
    assert [op["response"]["transcript"] for op in stt_ops] == [
        "Thanks.", "That is all I needed."
    ]
    spoken = [e for e in ops if e["type"] == "tts"]
    assert len(spoken) == 1
    heard = {op["turn_id"] for op in stt_ops}
    assert spoken[0]["turn_id"] not in heard, "the filler is not the caller's answer"


async def test_a_cancelled_preemptive_reply_never_swallows_the_rest_of_the_question(
        recorder):
    """LiveKit answers a provisional end of turn, so an LLM call can be billed
    against a message the caller has not finished saying. Once a turn carries
    that cost it is no longer collecting, and the rest of the utterance opens a
    turn of its own -- LiveKit committed one message, we record two.

    That split is accepted, but only on terms: every word the caller said is
    still recorded, on the turn that heard it, and the tokens are billed once,
    to the audio that actually caused them. Merging instead would have to write
    into a span that is already published, which loses the words silently.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("Book me a flight", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("s1")))
    session.emit("metrics_collected", llm_metrics("s1"))
    session.emit("user_input_transcribed", transcript("to Berlin please", True))
    session.emit("conversation_item_added",
                 chat_item("user", "Book me a flight to Berlin please"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    said = [op["response"]["transcript"] for op in ops if op["type"] == "stt"]
    assert said == ["Book me a flight", "to Berlin please"], (
        f"the caller's words must survive the split: {said}"
    )
    billed = [op for op in ops if op["type"] == "llm"]
    assert len(billed) == 1, f"the preemptive call is billed once: {billed}"
    heard_first = [op["turn_id"] for op in ops
                   if op["type"] == "stt"][0]
    assert billed[0]["turn_id"] == heard_first, (
        "the tokens belong to the audio that caused them, not to the "
        f"continuation: {billed[0]['turn_id']} vs {heard_first}"
    )


async def test_every_billing_tick_the_recogniser_sends_is_counted(recorder):
    """Deepgram meters a stream in *increments* -- one every five seconds and a
    remainder on close -- and LiveKit forwards each as its own metric. Keeping
    only the latest published four seconds for a nine-second stretch: an
    understatement of the provider's own billing number, on a healthy call."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("metrics_collected", stt_metrics(audio_duration=5.0))
    session.emit("metrics_collected", stt_metrics(audio_duration=4.0))
    session.emit("user_input_transcribed", transcript("hello there", True))
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"][0]
    assert stt["response"]["provider_metered_audio_ms"] == 9000, stt["response"]


async def test_a_filler_spoken_first_never_takes_the_callers_turn(recorder):
    """The filler is the *first* speech after the caller's final, so nothing
    has claimed the pending turn yet. Letting it in bills "let me check that"
    to the caller's question and leaves the real answer in a turn with no
    question in it -- both numbers wrong, on a call reporting full coverage."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("what is my balance", True))
    session.emit("speech_created", speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    session.emit("speech_created", speech_created(FakeSpeechHandle("answer")))
    session.emit("metrics_collected", tts_metrics("answer", audio_duration=2.0))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    heard = [op for op in ops if op["type"] == "stt"][0]
    on_the_question = [op["response"]["audio_ms"] for op in ops
                       if op["type"] == "tts" and op["turn_id"] == heard["turn_id"]]
    assert on_the_question == [2000], (
        "the question is answered by the generated reply and billed for it "
        f"alone -- the filler is not part of the answer: {on_the_question}"
    )
    elsewhere = [op["response"]["audio_ms"] for op in ops
                 if op["type"] == "tts" and op["turn_id"] != heard["turn_id"]]
    assert elsewhere == [400], f"the filler is still recorded, apart: {elsewhere}"


async def test_a_scripted_answer_still_belongs_to_the_question(recorder):
    """The other half of the rule. An agent that answers with `say()` and
    raises StopResponse produces no generated reply at all, so refusing every
    `say()` would detach every scripted answer from its question."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("what are your hours", True))
    session.emit("conversation_item_added", chat_item("user", "what are your hours"))
    session.emit("speech_created", speech_created(FakeSpeechHandle("scripted"), source="say"))
    session.emit("metrics_collected", tts_metrics("scripted"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    heard = [op for op in ops if op["type"] == "stt"][0]
    spoke = [op for op in ops if op["type"] == "tts"][0]
    assert spoke["turn_id"] == heard["turn_id"], (
        "a scripted reply is still this question's answer"
    )


async def test_a_turn_callback_that_crashes_is_not_recorded_as_a_mute_agent(recorder):
    """LiveKit catches any other exception from `on_user_turn_completed`, logs
    it and returns: no commit, no reply. Unlabelled that is a transcript with
    no answer and nothing saying why -- indistinguishable from a TTS that
    never sounded, which needs the opposite response."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    agent = FakeAgent(raises=ValueError("policy lookup failed"))
    rec.watch_agent(agent)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("is my card blocked", True))
    with pytest.raises(ValueError):
        await agent.on_user_turn_completed(None, None)
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"][0]
    assert stt["response"]["reply_skipped"] == "callback_error", stt["response"]


async def test_a_stop_response_subclass_is_still_a_stop_response(recorder):
    """LiveKit catches it by inheritance, so an agent that subclasses it to
    carry a reason is ignoring the utterance exactly as much."""
    class PolicyStop(StopResponse):
        pass

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    agent = FakeAgent(raises=PolicyStop())
    rec.watch_agent(agent)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("read me the pin", True))
    with pytest.raises(PolicyStop):
        await agent.on_user_turn_completed(None, None)
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"][0]
    assert stt["response"]["reply_skipped"] == "stop_response", stt["response"]


async def test_an_agent_reused_for_a_second_call_is_watched_by_that_call(recorder):
    """The wrapper's closure holds the recorder that installed it. An agent
    object reused for the next call still carries the finished recorder's
    wrapper, so treating "already wrapped" as "already watched" left the new
    call with no turn watch and merged the ignored utterance into the next."""
    first = recorder()
    agent = FakeAgent(raises=StopResponse())
    first.attach(FakeAgentSession())
    first.watch_agent(agent)
    await first.finish()

    second = recorder()
    session = FakeAgentSession()
    second.attach(session)
    second.watch_agent(agent)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("ignore this one", True))
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(None, None)
    session.emit("user_input_transcribed", transcript("but not this one", True))
    await second.finish()

    ops = operations(read_events(_dir(second)))
    said = [op["response"]["transcript"] for op in ops if op["type"] == "stt"]
    assert said == ["ignore this one", "but not this one"], said
    skipped = [op["response"].get("reply_skipped") for op in ops if op["type"] == "stt"]
    assert skipped[0] == "stop_response", skipped


async def test_an_agent_handed_off_mid_call_is_instrumented_too(recorder, tmp_path):
    """LiveKit supports replacing the agent mid-session. The replacement gets
    neither `agent.vaani` nor the turn watch, so the rest of the call captures
    no agent audio at all -- and nothing about the recording looks wrong,
    because the audio that is missing was never announced."""
    class SessionWithHandoff(FakeAgentSession):
        def __init__(self) -> None:
            super().__init__()
            self.handed_to: list[Any] = []

        def update_agent(self, agent, *args, **kwargs):
            self.handed_to.append(agent)
            return "livekit-result"

    rec = recorder()
    session = SessionWithHandoff()
    first = FakeAgent()
    observe_agent_session(session, rec, agent=first)

    replacement = FakeAgent()
    assert session.update_agent(replacement) == "livekit-result", (
        "following the handoff must not change what LiveKit returns"
    )
    assert session.handed_to == [replacement], "LiveKit still performs the handoff"
    assert replacement.vaani is rec, "the replacement must be able to tap audio"
    assert getattr(replacement.on_user_turn_completed, "_vaani_watch", None) is rec
    await rec.finish()


@pytest.mark.asyncio
async def test_a_turn_we_split_says_so_instead_of_quietly_skewing_the_average(recorder):
    """LiveKit committed one message; we recorded two. Say which two.

    We split only when keeping the caller's words requires it, and the split is
    defensible. What is not defensible is letting it disappear: per-turn latency
    and answer attribution stop matching the framework's history at that point,
    and an average over the two halves is not the average anyone thinks they are
    reading. The link back is what lets a reader put the halves together again.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("Thanks.", True))
    session.emit("speech_created", speech_created(SimpleNamespace(id="speech-1")))
    # The filler ends -- and publishes -- the span carrying "Thanks.".
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("user_input_transcribed", transcript("That is all I needed.", True))
    session.emit("conversation_item_added",
                 chat_item("user", "Thanks. That is all I needed."))
    await rec.finish()

    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    assert [op["response"]["transcript"] for op in stt_ops] == [
        "Thanks.", "That is all I needed."
    ], "the split itself is the premise of this test"
    first, second = stt_ops

    # The second half must name the first, and say why it could not be merged.
    assert second["response"].get("continues_turn") == first["turn_id"], (
        "a reader given only the second half cannot tell it is half of "
        "anything unless we point at the other half"
    )
    assert second["response"].get("split_reason") == "earlier_words_already_published"
    # And an ordinary opening turn must not claim to continue anything.
    assert "continues_turn" not in first["response"]


@pytest.mark.asyncio
async def test_usage_metered_after_the_final_bills_the_caller_who_earned_it(recorder):
    """OpenAI meters *after* the transcript is final. That is not the next caller.

    The recogniser queues the final transcript first and the usage for it second
    (`plugins/openai/stt.py:895`), by which time the pending span has already
    been swapped out. Adding it there charged one caller's speech to whoever
    spoke next -- wrong in a direction no downstream check can catch, because
    both turns still look complete. Token counts were dropped outright.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("First caller speaking.", True))
    # Usage for the message that just went final, with nothing yet spoken after it.
    session.emit("metrics_collected",
                 stt_metrics(audio_duration=2.0, input_tokens=140, output_tokens=12))
    session.emit("conversation_item_added", chat_item("user", "First caller speaking."))
    session.emit("user_input_transcribed", transcript("Second thing entirely.", True))
    session.emit("conversation_item_added", chat_item("user", "Second thing entirely."))
    await rec.finish()

    stt_ops = [e for e in operations(read_events(_dir(rec))) if e["type"] == "stt"]
    first = next(op for op in stt_ops
                 if op["response"]["transcript"] == "First caller speaking.")
    second = next(op for op in stt_ops
                  if op["response"]["transcript"] == "Second thing entirely.")

    assert first["response"]["provider_metered_audio_ms"] == 2000, (
        "the audio was metered against the message that had just gone final"
    )
    assert not second["response"].get("provider_metered_audio_ms"), (
        "the second caller never incurred this; billing them is the actual defect"
    )
    # And the tokens the provider charges for must survive at all.
    assert first["response"]["input_tokens"] == 140
    assert first["response"]["output_tokens"] == 12
    # Say that the number arrived out of band, so a reader can audit the choice.
    assert first["response"]["metered_after_final"] is True


@pytest.mark.asyncio
async def test_a_filler_said_inside_the_callback_does_not_pass_as_the_answer(recorder):
    """Nothing at the time a `say()` is made can tell a filler from an answer.

    Said inside `on_user_turn_completed`, the caller's message is already
    committed, so the "is the message still open" rule that catches an
    interjection cannot help: LiveKit's `SpeechCreatedEvent` carries no intent
    (`events.py:474`). What settles it is what happens next -- if a generated
    reply follows, the filler was not the answer. Reported as one number, a
    400ms "let me check" plus a 2000ms answer reads as a 2400ms answer, and
    every reply-duration statistic drifts up by however chatty the filler is.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    # Said from inside the callback: after the commit, before the real answer.
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    # LiveKit then generates the answer it was always going to give.
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-1")))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    tts = [op for op in ops if op["type"] == "tts"]
    assert len(tts) == 1, "both played on the caller's turn; that part was right"
    response = tts[0]["response"]

    # The total may stay 2400ms -- the caller really did hear that much -- but
    # it must not be presentable as the answer's own duration.
    assert response.get("reply_includes_filler") is True
    assert response.get("filler_audio_ms") == 400, (
        "without this the answer is indistinguishable from a 2400ms one"
    )
    assert response["audio_ms"] == 2400


@pytest.mark.asyncio
async def test_a_handoff_to_an_untapped_agent_stops_certifying_the_call(recorder):
    """The first agent's tap must not vouch for the one that replaced it.

    `agent_audio_tapped` is sticky on purpose: it answers "was capture ever
    possible", which is what separates a silent agent from an unbound one. But
    a handoff to an agent whose `tts_node` is not the mixin's means everything
    said from that point is unrecorded, and the sticky flag let the earlier
    agent's tap certify that silence -- the SDK logged the loss and the manifest
    called the call complete in the same breath.
    """
    class _Framework:
        async def tts_node(self, text, model_settings):  # noqa: ANN001
            yield None

    class Tapped(VaaniAudioTapMixin, _Framework):
        pass

    class Untapped(_Framework, VaaniAudioTapMixin):
        # `Agent` wins the MRO, so the tapping node is never the one called.
        pass

    rec = recorder()
    first = Tapped()
    first.vaani = rec
    rec.note_audio_tap_installed(first)
    session = FakeAgentSession()
    rec.attach(session)

    replacement = Untapped()
    rec.bind_agent(replacement)
    await rec.finish()

    capture = _manifest_of(rec)["capture_status"]
    # Capture *was* possible once; that stays true and stays useful.
    assert capture["measured"]["agent_audio_tapped"] is True
    assert capture["coverage_complete"] is False, (
        "known loss the manifest was reporting as a complete capture"
    )
    assert any(gap.get("agent_tap_lost_on_handoff")
               for gap in capture["coverage_gaps"])


@pytest.mark.asyncio
async def test_reusing_an_agent_does_not_stack_a_wrapper_per_call(recorder):
    """A shared agent is reused precisely where call volume is highest.

    Each wrapper closes over the previous one and over the recorder that
    installed it, so wrapping afresh every call grows the callback chain and
    pins every finished recorder in memory -- unbounded, on the deployments
    least able to absorb it.
    """
    agent = FakeAgent()
    original = agent.on_user_turn_completed

    recorders = []
    for _ in range(4):
        rec = recorder()
        recorders.append(rec)
        rec.watch_agent(agent)

    watcher = agent.on_user_turn_completed
    assert getattr(watcher, "_vaani_watch", None) is recorders[-1], (
        "the live call must be the one being watched"
    )
    # Exactly one layer, wrapping the agent's own method -- not the previous
    # wrapper, and not three dead recorders behind it.
    assert getattr(watcher, "_vaani_wrapped", None) == original
    for rec in recorders:
        await rec.finish()


@pytest.mark.asyncio
async def test_the_answers_words_are_not_filed_under_a_filler_that_spoke_last(recorder):
    """Found on a live call: two turns, both captioned with the answer's words.

    `conversation_item_added(assistant)` carries no speech id, so the reply is
    resolved to whichever turn is currently speaking. A filler created after the
    answer makes *itself* the speaking turn, and the answer's transcript is
    filed under a turn that only ever said "let me check that for you" -- while
    the real answer keeps its own copy. A reader sees the same sentence twice,
    once against audio that never contained it.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-1")))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    # The filler is created after the answer, so it is the live speaking turn.
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("conversation_item_added",
                 chat_item("assistant", "The cheapest fare is 120 dollars."))
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec))) if op["type"] == "tts"]
    captioned = [op for op in tts
                 if "cheapest fare" in (op["response"].get("text") or "")]
    # The reply these words belong to was published before they arrived, so
    # there is nowhere honest left to put them. The filler is not an answer of
    # last resort: it was handed its own script and never said this.
    assert not any(op["response"].get("audio_ms") == 400 for op in captioned), (
        "the answer's transcript filed against audio that never contained it"
    )
    # Dropped, but never quietly: a reader can see that words went missing.
    capture = _manifest_of(rec)["capture_status"]
    assert capture["coverage_complete"] is False
    assert any("dropped rather than attributed" in (gap.get("reason") or "")
               for gap in capture["coverage_gaps"])


@pytest.mark.asyncio
async def test_a_finished_filler_turn_does_not_adopt_the_next_callers_reply(recorder):
    """Being adoptable must expire when the next caller starts talking.

    Letting a real answer adopt a turn whose only reply so far was a filler is
    what stops the answer opening a turn with no question in it. But a
    `say()`-only turn stays filler-only forever, and LiveKit generates replies
    *preemptively* from a predicted end of turn -- so the second caller's reply
    can be created while the first caller's turn is still the collecting one.
    Adopted backwards, two exchanges are recorded as one: the second caller's
    words end up in a turn answered before they spoke, and the first caller's
    question is credited with an answer to someone else's.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    # A scripted, LLM-free answer: the turn is now filler-only and complete.
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="scripted-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("scripted-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    # The next caller begins speaking. No final yet, so the first caller's turn
    # is still `_collecting_turn`.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    # LiveKit generates that caller's reply preemptively, before their final.
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-2")))
    session.emit("metrics_collected", tts_metrics("answer-2", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("user_input_transcribed", transcript("And to Boston?", True))
    session.emit("conversation_item_added", chat_item("user", "And to Boston?"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    by_turn: dict[str, set] = {}
    for op in ops:
        by_turn.setdefault(op["turn_id"], set()).add(op["type"])
    first = [op for op in ops
             if op["type"] == "stt"
             and "fare" in (op["response"].get("transcript") or "")]
    assert first, "the first caller's turn must still exist"
    second_reply = [op for op in ops
                    if op["type"] == "tts" and op["response"].get("audio_ms") == 2000]
    assert second_reply, "the second caller's reply must be recorded"
    assert second_reply[0]["turn_id"] != first[0]["turn_id"], (
        "the second caller's reply was adopted into the first caller's turn"
    )
    assert second_reply[0]["response"].get("reply_attribution") == "inferred", (
        "keeping the reply separate here is a judgement call, not a proof, and "
        "the span has to say which it is"
    )


@pytest.fixture
def public_generate_reply(monkeypatch, tmp_path):
    """Stand the fake session's method in for `AgentSession.generate_reply`.

    `livekit-agents` is not installed here -- these tests exist so the recorder
    can be driven without it -- so the code object the stack is matched against
    is pointed at the fake, and a directory stands in for the installed package.
    What is under test is the rule: the public method on the stack, and who the
    frame above it belongs to. The same rule is exercised against the real
    `AgentSession` and a real captured bound method in `verify_real_handle.py`.
    """
    root = tmp_path / "livekit" / "agents"
    (root / "voice").mkdir(parents=True)
    monkeypatch.setattr(livekit_integration, "_GENERATE_REPLY_CODE",
                        FakeAgentSession.generate_reply.__code__)
    monkeypatch.setattr(livekit_integration, "_LIVEKIT_ROOTS",
                        (os.path.join(str(root.parent.resolve()), ""),))

    def framework_caller(session, _from=None, _deferred=False,
                         _run_from=None, **kwargs):
        """A call to the public method made from inside the installed package.

        `_deferred` hands the call to the interpreter instead of making it
        directly, so the frames read package -> interpreter -> public method,
        which is what a deferred callback, a task wrapper or a decorator
        produces in an installed LiveKit.
        """
        source = ("import contextlib\n"
                  "def call(session, kwargs, deferred, run_from):\n"
                  "    if run_from is not None:\n"
                  "        return run_from(session, kwargs)\n"
                  "    if not deferred:\n"
                  "        return session.generate_reply(**kwargs)\n"
                  "    with contextlib.ExitStack() as stack:\n"
                  "        stack.callback(session.generate_reply, **kwargs)\n")
        namespace: dict = {}
        default = root / "voice" / "agent_activity.py"
        path = str((_from or default).resolve())
        exec(compile(source, path, "exec"), namespace)
        outer = None
        if _run_from is not None:
            inner: dict = {}
            exec(compile("def ask(session, kwargs):\n"
                         "    return session.generate_reply(**kwargs)\n",
                         str(_run_from.resolve()), "exec"), inner)
            outer = inner["ask"]
        return namespace["call"](session, kwargs, _deferred, outer)

    # `AgentSession.run()` lives in the package but is only ever reached from
    # an adopter's code, so its frame must not be read as the framework's.
    entry_namespace: dict = {}
    exec(compile("def run(session, kwargs):\n"
                 "    return session.generate_reply(**kwargs)\n",
                 str((root / "voice" / "agent_session.py").resolve()), "exec"),
         entry_namespace)
    monkeypatch.setattr(livekit_integration, "_APPLICATION_ENTRY_CODES",
                        (entry_namespace["run"].__code__,))
    # `AgentActivity._generate_reply()` is the function that actually emits
    # `speech_created`. LiveKit reaches it directly for the automatic answer to
    # a completed turn, and the public method reaches it for an application
    # call -- so it is the anchor the stack always has, even when the public
    # method has been replaced.
    emitter_namespace: dict = {}
    exec(compile("def emit_reply(session, handle_id, make):\n"
                 "    session.emit('speech_created', make(handle_id))\n",
                 str((root / "voice" / "agent_activity.py").resolve()), "exec"),
         emitter_namespace)
    monkeypatch.setattr(livekit_integration, "_EMITTING_REPLY_CODE",
                        emitter_namespace["emit_reply"].__code__)

    def reach_emitter(session, handle_id, *, _from):
        """Call the emitting function from a frame in the given file."""
        namespace: dict = {}
        exec(compile("def call(session, handle_id, make, emit):\n"
                     "    return emit(session, handle_id, make)\n",
                     str(_from.resolve()), "exec"), namespace)
        return namespace["call"](
            session, handle_id,
            lambda hid: speech_created(FakeSpeechHandle(hid)),
            emitter_namespace["emit_reply"])

    # A stand-in for the public method is recognised by name *and* receiver,
    # so the class the receiver must be an instance of has to be known here
    # too -- the real one is captured when `AgentSession` is resolved.
    monkeypatch.setattr(livekit_integration, "_AGENT_SESSION_CLASS",
                        FakeAgentSession)
    framework_caller.package_root = root.parent
    framework_caller.entry_caller = entry_namespace["run"]
    framework_caller.reach_emitter = reach_emitter
    framework_caller.livekit_file = root / "voice" / "agent_activity.py"
    return framework_caller


@pytest.mark.asyncio
async def test_a_reply_the_application_asked_for_admits_it_might_answer_the_caller(
        recorder, public_generate_reply):
    """The one thing this must never be is silent.

    `AgentSession.generate_reply()` reaches the same `_generate_reply` as
    LiveKit's automatic answer and passes `input_modality` straight through, so
    an application asking for audio produces an event identical in every field:
    scheduled, `user_initiated`, audio input details. Reading the event alone
    cannot separate them, and an application may call it to answer the caller or
    to say something unrelated. The recorder reads the call site off the stack
    instead -- but knowing the call happened is not knowing what it meant.

    The recorder keeps such a reply separate, because merging two exchanges
    makes a caller's words appear answered before they were spoken. But that is
    a guess, and a reader deciding a latency from it is owed that.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    # Someone speaks while the filler plays, and the application generates a
    # reply of its own. Nothing here says which of the two it is for.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    # Asking for audio input details, which makes the event identical in every
    # field to the automatic answer. Only the call itself is different.
    session.generate_reply(input_modality="audio", handle_id="answer-2")
    session.emit("metrics_collected", tts_metrics("answer-2", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec)))
           if op["type"] == "tts" and op["response"].get("audio_ms") == 2000]
    assert tts, "the reply must be recorded whichever turn it lands in"
    response = tts[0]["response"]
    assert response.get("reply_attribution") == "inferred"
    assert "generate_reply" in (response.get("reply_attribution_reason") or ""), (
        "the reason must name what could not be distinguished, so a reader can "
        "judge it rather than take our word for it"
    )


@pytest.mark.asyncio
async def test_a_contested_turns_token_count_carries_the_caveat_with_no_tts_span(recorder):
    """The tokens are the expensive number, and they can outlive the TTS span.

    A reply can report LLM metrics and then be interrupted before any TTS
    metric, audio frame or assistant item exists. Carrying the caveat only on
    the TTS response meant that in exactly that case the turn published an `ok`
    LLM operation, with prompt and completion tokens on it, and nothing
    anywhere saying its ownership was a judgement call.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-2")))
    # LLM reports; the reply is then cut off, so no TTS metric ever arrives.
    session.emit("metrics_collected", llm_metrics("answer-2"))
    await rec.finish()

    llm = [op for op in operations(read_events(_dir(rec))) if op["type"] == "llm"]
    assert llm, "the LLM measurement must still be published"
    assert llm[0]["response"].get("reply_attribution") == "inferred", (
        "an uncertain turn's tokens must not read as settled just because the "
        "reply never got as far as speaking"
    )


@pytest.mark.asyncio
async def test_an_ordinary_interim_does_not_detach_the_answer_from_its_question(
        recorder):
    """A reply LiveKit scheduled answers the turn it was scheduled for.

    A preflight transcript and an ordinary interim arrive as the same public
    event, so while a filler is playing there is no way to tell a reply meant
    for the *next* caller from this caller's own delayed answer. Refusing both
    filed measured tokens in a turn with no question in it, and left the
    question reading as unanswered: one exchange described wrongly twice, with
    the cost on the wrong side of it.

    The handle settles it. An ordinary reply is scheduled in the same
    synchronous frame as `speech_created`, a preemptive one is not, so the
    decision only has to wait until this handler returns. It is deliberately
    read here *before* the loop gets a slice -- the LLM metric is emitted
    immediately -- because the answer must not depend on which arrives first.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("question one", True))
    session.emit("conversation_item_added", chat_item("user", "question one"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    # An ordinary interim, indistinguishable from a preflight in the events.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("um", False))
    # Caller one's own answer, scheduled the moment it was created.
    session.emit("speech_created", speech_created(FakeSpeechHandle("old-answer")))
    session.emit("metrics_collected", llm_metrics("old-answer"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    stt = next(op for op in ops if op["type"] == "stt"
               and "question one" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["response"].get("total_tokens"), (
        "guard: this test is meaningless unless the tokens were measured"
    )
    assert llm["turn_id"] == stt["turn_id"], (
        "the tokens LiveKit measured for this reply must be filed against the "
        "question it was scheduled to answer, not a turn with no question in it"
    )
    assert llm["response"].get("reply_attribution") is None, (
        "the handle said who this reply belongs to, so the turn must not be "
        "marked as a judgement"
    )


@pytest.mark.asyncio
async def test_a_preemptive_reply_over_a_filler_still_gets_its_own_turn(recorder):
    """The other half of the same reading, which must not regress.

    An unscheduled reply was generated from a *predicted* end of the speech
    that arrived over the filler, so it answers the next caller. Merging it
    backwards would report their words as answered before they were spoken.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("question one", True))
    session.emit("conversation_item_added", chat_item("user", "question one"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("and to Boston", False))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("next-answer", scheduled=False)))
    session.emit("metrics_collected", llm_metrics("next-answer"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    stt = next(op for op in ops if op["type"] == "stt"
               and "question one" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["turn_id"] != stt["turn_id"], (
        "a reply predicted for the next caller must not be merged into the "
        "previous caller's turn"
    )
    assert llm["response"].get("reply_attribution") is None, (
        "the handle said this reply was preemptive, so keeping it separate is a "
        "reading rather than a judgement and must not be hedged"
    )


@pytest.mark.asyncio
async def test_a_realtime_models_own_reply_is_not_merged_into_the_previous_caller(recorder):
    """Scheduled does not mean "answers the turn that just finished".

    A realtime model generates server-side as it transcribes. LiveKit surfaces
    that as a `speech_created` with `user_initiated=False`, scheduled at once
    (`agent_activity.py:2007` on 1.7.0, `:1983` on 1.6.10) -- and it answers the
    speech being transcribed *now*, not the previous one. The framework itself
    notes at `:1978-1989` that a provider may withhold the final transcript
    until that reply has finished generating, so the reply legitimately arrives
    before the words that prompted it.

    Reading `scheduled` alone would merge it backwards into the previous
    caller's turn and, worse, report that as certain.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("question one", True))
    session.emit("conversation_item_added", chat_item("user", "question one"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("and to Boston", False))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("realtime-answer"),
                                user_initiated=False))
    session.emit("metrics_collected", llm_metrics("realtime-answer"))
    # The provider releases the final transcript only now, after its reply.
    session.emit("user_input_transcribed", transcript("and to Boston", True))
    session.emit("conversation_item_added", chat_item("user", "and to Boston"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    first = next(op for op in ops if op["type"] == "stt"
                 and "question one" in (op["response"].get("transcript") or ""))
    second = next(op for op in ops if op["type"] == "stt"
                  and "and to Boston" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["turn_id"] != first["turn_id"], (
        "a realtime reply generated for the caller now speaking must not be "
        "merged into the previous caller's turn"
    )
    assert llm["turn_id"] == second["turn_id"], (
        "it belongs with the words that prompted it, which arrive after it"
    )
    assert llm["response"].get("reply_attribution") is None, (
        "the event said this reply was not user-initiated, so placing it is a "
        "reading rather than a judgement and must not be hedged"
    )


@pytest.mark.asyncio
async def test_attaching_the_same_recorder_twice_does_not_record_the_call_twice(
        recorder):
    """Subscribing again is a mistake a restart or a retry makes for you.

    Every handler was registered a second time, so one final transcript ran
    `_on_transcript` twice and the turn published the caller's words doubled --
    `"hello hello"` for one `"hello"` -- while the manifest still reported the
    capture complete. Anything derived from the transcript, including an
    evaluation of what the caller asked for, is then wrong with no sign of it.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("conversation_item_added", chat_item("user", "hello"))
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"]
    assert len(stt) == 1, "the same utterance was recorded once per subscription"
    assert stt[0]["response"]["transcript"] == "hello", (
        "the caller's words were doubled by a second copy of the handler"
    )


@pytest.mark.asyncio
async def test_a_livekit_plugin_asking_for_a_reply_is_not_the_application(
        recorder, public_generate_reply):
    """`livekit` is a namespace package, and plugins are separate packages in it.

    Matching only the `livekit/agents` directory read a plugin's own call to
    `generate_reply` as the application's -- a detached turn under a caveat,
    with the caller's question one turn behind it. That is the same
    misattribution stack inspection replaced, so the whole namespace path is
    what a LiveKit-owned caller is matched against.
    """
    plugin = public_generate_reply.package_root / "plugins" / "openai" / "llm.py"
    plugin.parent.mkdir(parents=True)
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    public_generate_reply(session, _from=plugin, input_modality="audio",
                          handle_id="answer-2")
    session.emit("metrics_collected", tts_metrics("answer-2", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("user_input_transcribed", transcript("And to Boston?", True))
    session.emit("conversation_item_added", chat_item("user", "And to Boston?"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "tts"
               and op["response"].get("audio_ms") == 2000)
    first = next(op for op in ops if op["type"] == "stt"
                 and "What is the fare?" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] != first["turn_id"]
    assert "LiveKit asked for this reply itself" in (
        llm["response"].get("reply_attribution_reason") or ""), (
        "a plugin's own call was not recorded as LiveKit's own"
    )


@pytest.mark.asyncio
async def test_a_reply_livekit_asked_for_through_the_stdlib_is_still_livekits(
        recorder, public_generate_reply):
    """The frame above the public method is not always the one that decided.

    A deferred callback, a task wrapper or a decorator puts a standard-library
    frame there instead; `ExitStack` stands for all of them here because it is
    the one that is trivially reproducible. Read literally, that frame belongs
    to no package, so every such call answered "the application" -- LiveKit's
    own reply detached from the caller and published under a caveat, which is
    the misattribution stack inspection was introduced to end. Interpreter
    frames are stepped over until a frame that belongs to somebody is found,
    and "application" stays the answer when none does, because that only ever
    adds a caveat.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))

    public_generate_reply(session, _deferred=True,
                          input_modality="audio", handle_id="answer-2")
    session.emit("metrics_collected", tts_metrics("answer-2", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    session.emit("user_input_transcribed", transcript("And to Boston?", True))
    session.emit("conversation_item_added", chat_item("user", "And to Boston?"))
    await rec.finish()

    reply = next(op for op in operations(read_events(_dir(rec)))
                 if op["type"] == "tts"
                 and op["response"].get("audio_ms") == 2000)
    assert "LiveKit asked for this reply itself" in (
        reply["response"].get("reply_attribution_reason") or ""), (
        "a reply LiveKit asked for indirectly was not recorded as its own"
    )


@pytest.mark.asyncio
async def test_an_installed_application_is_not_mistaken_for_the_interpreter(
        recorder, public_generate_reply, monkeypatch, tmp_path):
    """An adopter whose agent is installed, not run from a checkout.

    Installed packages live *under* the interpreter's library directory, so
    treating that whole directory as plumbing stepped over the application's
    own frame and kept walking -- reaching LiveKit's frame above it and
    reporting the reply as the framework's, with the caveat dropped. A reply
    the application asked for, published as settled fact.
    """
    library = tmp_path / "lib" / "python3.14"
    app = library / "site-packages" / "my_agent" / "flows.py"
    app.parent.mkdir(parents=True)
    monkeypatch.setattr(livekit_integration, "_STDLIB_ROOTS",
                        (os.path.join(str(library.resolve()), ""),))
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))

    # LiveKit calls into the installed application, which asks for the reply.
    public_generate_reply(session, _run_from=app, input_modality="audio",
                          handle_id="answer-2")
    session.emit("metrics_collected", tts_metrics("answer-2", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()

    reply = next(op for op in operations(read_events(_dir(rec)))
                 if op["type"] == "tts"
                 and op["response"].get("audio_ms") == 2000)
    assert reply["response"].get("reply_attribution") == "inferred", (
        "an installed application's own reply was published as the "
        "framework's, with no caveat"
    )


@pytest.mark.asyncio
async def test_a_project_directory_beside_the_package_is_still_the_application(
        recorder, public_generate_reply):
    """`livekit_helpers/` sits beside `livekit/` and its path starts the same.

    Comparing the caller's filename against a root with no trailing separator
    made every module whose directory merely *began* with the package path read
    as LiveKit's own -- and a reply the application asked for was then placed
    with no caveat at all, which is the one outcome the caveat exists to
    prevent.
    """
    root = public_generate_reply.package_root
    beside = root.parent / (root.name + "_helpers") / "replies.py"
    beside.parent.mkdir(parents=True)
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    public_generate_reply(session, _from=beside, input_modality="audio",
                          handle_id="answer-2")
    session.emit("metrics_collected", tts_metrics("answer-2", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()

    llm = next(op for op in operations(read_events(_dir(rec)))
               if op["type"] == "tts" and op["response"].get("audio_ms") == 2000)
    assert llm["response"].get("reply_attribution") == "inferred", (
        "an application module beside the package was taken for the framework"
    )


@pytest.mark.asyncio
async def test_a_reply_the_application_asked_for_through_livekits_own_api(
        recorder, public_generate_reply):
    """`AgentSession.run()` is the adopter's decision in LiveKit's file.

    `run(user_input=...)` is a public entry point that forwards to the public
    `generate_reply` (`agent_session.py:823`). Read by filename alone the frame
    above the reply belongs to LiveKit, so the reply was called framework speech
    and silently joined to the *next* spoken caller's turn -- putting a
    programmatic run's tokens and cost on a caller who never asked for them, and
    saying so with no caveat. There is no caller speech behind such a reply at
    all, so it belongs in a turn of its own.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    # Somebody starts talking over the filler, and before their words go final
    # the application drives a programmatic run.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    public_generate_reply.entry_caller(session, {"handle_id": "run-reply"})
    session.emit("metrics_collected", llm_metrics("run-reply"))
    session.emit("user_input_transcribed", transcript("And to Boston?", True))
    session.emit("conversation_item_added", chat_item("user", "And to Boston?"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "llm")
    later = next(op for op in ops if op["type"] == "stt"
                 and "And to Boston?" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] != later["turn_id"], (
        "a programmatic run's tokens were billed to a caller who never asked "
        "for them, because the frame above the reply was LiveKit's own file"
    )


@pytest.mark.asyncio
async def test_a_tool_reply_arriving_after_livekit_gave_up_is_not_asserted_away(
        recorder):
    """Five seconds bound LiveKit's wait, not the provider's generation.

    `agent_activity.py:4278` waits on a shielded future and then merely stops
    tracking it (`:4284-4286`); the chat context that prompts the provider was
    already updated at `:4291-4295`. So a slow provider can emit the generation
    the tool was owed after the deadline. Calling that ordinary in-flight speech
    moved its tokens onto the next caller and said so with no caveat -- the tool
    exchange then reads as having had no model answer at all.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("book it", True))
    session.emit("conversation_item_added", chat_item("user", "book it"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("function_tools_executed", tools_executed())
    # Past the window LiveKit waits, with the provider's answer still to come.
    rec._current_turn.tool_reply_deadline_ms = rec.call.now() - 1
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello?", False))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("late-reply"),
                                user_initiated=False))
    session.emit("metrics_collected", llm_metrics("late-reply"))
    await rec.finish()

    llm = next(op for op in operations(read_events(_dir(rec)))
               if op["type"] == "llm")
    assert llm["response"].get("reply_attribution") == "inferred", (
        "a generation arriving after LiveKit stopped waiting was placed as "
        "though its origin were known"
    )
    assert "stopped waiting" in (
        llm["response"].get("reply_attribution_reason") or ""), (
        "the caveat must say the tool's answer may have arrived late"
    )


@pytest.mark.asyncio
async def test_a_stack_that_cannot_be_read_is_not_proof_of_an_automatic_answer(
        recorder, monkeypatch):
    """Failing to look is not the same fact as looking and finding nothing.

    `None` meant "the public method is not on the stack", which is LiveKit's own
    automatic answer to a completed turn -- merged backwards into it with no
    caveat. A blocked `sys._getframe`, an unreachable code object or a walk that
    ran out of budget returned the same `None`, so an unrelated application
    reply was published as the certain answer to whoever spoke last.
    """
    # The code object cannot be reached on this build, which is one of the
    # three ways the lookup fails; a blocked `sys._getframe` and a walk that
    # ran out of budget reach the same sentinel and the same branch below.
    monkeypatch.setattr(livekit_integration, "_GENERATE_REPLY_CODE",
                        livekit_integration._CALL_SITE_UNAVAILABLE)

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    session.generate_reply(handle_id="app-reply")
    session.emit("metrics_collected", llm_metrics("app-reply"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "llm")
    question = next(op for op in ops if op["type"] == "stt"
                    and "What is the fare?" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] != question["turn_id"], (
        "a reply whose origin could not be read was merged into the caller's "
        "turn as though it were LiveKit's answer to them"
    )
    assert llm["response"].get("reply_attribution") == "inferred", (
        "a failure to read the stack must be disclosed, not resolved silently"
    )


@pytest.mark.asyncio
async def test_two_threads_attaching_at_once_still_subscribe_once(recorder):
    """Idempotence has to hold when both callers arrive together.

    A retry and a reconnect racing each other could both find nothing attached
    before either had finished subscribing, register every handler twice, and
    publish the caller's words doubled -- with the manifest still reporting the
    capture complete, so nothing downstream could tell it from a caller who
    repeated themselves. The sleep only widens the window that already exists.
    """
    import threading, time

    class SlowSession(FakeAgentSession):
        def on(self, name, handler):
            time.sleep(0.02)
            super().on(name, handler)

    rec = recorder()
    session = SlowSession()
    threads = [threading.Thread(target=rec.attach, args=(session,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {len(handlers) for handlers in session.handlers.values()} == {1}, (
        "a concurrent attach registered a second copy of every handler"
    )
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("conversation_item_added", chat_item("user", "hello"))
    rec.note_audio_tap_installed()
    await rec.finish()

    stt = next(op for op in operations(read_events(_dir(rec))) if op["type"] == "stt")
    assert stt["response"]["transcript"] == "hello", (
        "a concurrent attach published the caller's words twice"
    )


@pytest.mark.asyncio
async def test_a_session_reused_for_a_second_call_is_recorded_again(recorder):
    """Refusing a repeat subscription must not refuse a legitimate one.

    A worker that keeps one `AgentSession` and records consecutive calls on it
    attaches, finishes, and attaches again. `finish()` unsubscribes, so the
    second attach is not a duplicate -- and a guard that remembered the session
    forever would have silently recorded an empty second call while the rest of
    the package looked healthy.
    """
    session = FakeAgentSession()
    first_recorder = recorder()
    first_recorder.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("first call", True))
    session.emit("conversation_item_added", chat_item("user", "first call"))
    await first_recorder.finish()

    second_recorder = recorder()
    second_recorder.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("second call", True))
    session.emit("conversation_item_added", chat_item("user", "second call"))
    await second_recorder.finish()

    for rec, expected in ((first_recorder, "first call"),
                          (second_recorder, "second call")):
        stt = [op for op in operations(read_events(_dir(rec)))
               if op["type"] == "stt"]
        assert [op["response"]["transcript"] for op in stt] == [expected], (
            "a session reused for a second call recorded the wrong words, or "
            "none at all"
        )


@pytest.mark.asyncio
async def test_livekits_own_call_to_the_public_method_is_not_the_applications(
        recorder, public_generate_reply):
    """LiveKit calls `AgentSession.generate_reply()` itself, in six places.

    On 1.7.0 it is used to commit a realtime turn manually
    (`agent_activity.py:1693`), to retry a structured output
    (`run_result.py:292`), to answer an asynchronous tool result
    (`tool_executor.py:599`), and by the IVR activity (`ivr_activity.py:53`,
    `:83`). Counting calls therefore reported the framework's own replies as the
    application's: a detached turn under a caveat, with the caller's final
    transcript opening yet another turn after it, so the tokens and the question
    ended up two turns apart.

    Every one of those routes answers something already under way, which is the
    same shape as a realtime generation: its own turn, which the final
    transcript then joins. The frame above the call says which it is.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("question one", True))
    session.emit("conversation_item_added", chat_item("user", "question one"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("question", False))
    public_generate_reply(session, input_modality="audio", handle_id="answer-2")
    session.emit("metrics_collected", llm_metrics("answer-2"))
    # LiveKit committed the turn, so the final transcript follows its reply.
    session.emit("user_input_transcribed", transcript("question two", True))
    session.emit("conversation_item_added", chat_item("user", "question two"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    first = next(op for op in ops if op["type"] == "stt"
                 and "question one" in (op["response"].get("transcript") or ""))
    second = next(op for op in ops if op["type"] == "stt"
                  and "question two" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["turn_id"] != first["turn_id"], (
        "the framework's reply was merged into the turn the filler was holding "
        "open, which is the one caller it cannot be for"
    )
    assert llm["turn_id"] == second["turn_id"], (
        "the reply and the question it answers were recorded two turns apart, "
        "so the turn has tokens with no question and the question has no answer"
    )
    assert "LiveKit asked for this reply itself" in (
        llm["response"].get("reply_attribution_reason") or ""), (
        "the framework's own call was not recorded as LiveKit's own"
    )


def tools_executed(name: str = "search_flights") -> Any:
    call = SimpleNamespace(name=name, arguments='{"to":"GOI"}', call_id="c1")
    output = SimpleNamespace(output='{"price":6000}', is_error=False, call_id="c1")
    return SimpleNamespace(function_calls=[call], function_call_outputs=[output],
                           zipped=lambda: [(call, output)])


@pytest.mark.asyncio
async def test_a_tool_result_that_asks_for_no_reply_does_not_claim_the_next_one(
        recorder):
    """`function_tools_executed` is not proof that a reply is owed.

    A tool returning `StopResponse` sets `reply_required` False on its output,
    and `has_tool_reply` is the OR of those (`voice/events.py:441,447`). Reading
    the event's mere presence left the flag standing after a result that asked
    for nothing, so the next caller's realtime generation was adopted backwards
    as the tool's answer -- putting its tokens on an exchange that ended before
    the caller spoke.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("stop talking", True))
    session.emit("conversation_item_added", chat_item("user", "stop talking"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    event = tools_executed()
    event.has_tool_reply = False
    session.emit("function_tools_executed", event)
    # A new caller, and the realtime model's reply to them.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", False))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("answer-2"), user_initiated=False))
    session.emit("metrics_collected", llm_metrics("answer-2"))
    session.emit("user_input_transcribed", transcript("are you there", True))
    session.emit("conversation_item_added", chat_item("user", "are you there"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    first = next(op for op in ops if op["type"] == "stt"
                 and "stop talking" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["turn_id"] != first["turn_id"], (
        "a reply was adopted into a turn whose tool result had explicitly "
        "asked for no reply"
    )


@pytest.mark.asyncio
async def test_a_tool_reply_that_never_arrives_stops_claiming_later_replies(
        recorder):
    """An interrupted tool execution emits the event and answers nothing.

    So does a realtime model configured without `auto_tool_reply_generation`,
    which continues on the existing speech handle and emits no `speech_created`
    that could clear the flag (`agent_activity.py:4305-4328`). The flag then
    stood for the rest of the call and the next caller's reply was adopted
    backwards. LiveKit stops waiting after five seconds
    (`agent_activity.py:4278`); so does this.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("book it", True))
    session.emit("conversation_item_added", chat_item("user", "book it"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("function_tools_executed", tools_executed())
    # No reply comes. Long after LiveKit gave up, someone else speaks.
    turn = rec._current_turn
    turn.tool_reply_deadline_ms = rec.call.now() - 1
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", False))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("answer-2"), user_initiated=False))
    session.emit("metrics_collected", llm_metrics("answer-2"))
    session.emit("user_input_transcribed", transcript("still there", True))
    session.emit("conversation_item_added", chat_item("user", "still there"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    first = next(op for op in ops if op["type"] == "stt"
                 and "book it" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["turn_id"] != first["turn_id"], (
        "a tool result whose answer never came kept claiming replies for the "
        "rest of the call"
    )


@pytest.mark.asyncio
async def test_a_realtime_reply_owed_to_a_tool_result_is_not_given_to_the_next_caller(
        recorder):
    """`user_initiated=False` has two causes, and they point opposite ways.

    LiveKit emits the identical scheduled, audio, not-user-initiated event both
    for a realtime model answering the speech being transcribed now and for its
    automatic reply after a tool result -- which answers the turn that ran the
    tool. `_on_generation_created` builds the same handle for both and only then
    consumes its private pending-tool marker, so the event cannot be read.

    Treating every such reply as generated for the speech in flight hands the
    tool's tokens and response latency to whoever spoke over it, and leaves the
    turn that called the tool with no answer. What the recorder can see is that
    a tool result on this turn is still owed a reply, so the reply stays with
    it -- and says that the placement was a reading, because a caller talking
    over the tool call could have prompted it instead.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("book the flight", True))
    session.emit("conversation_item_added", chat_item("user", "book the flight"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("function_tools_executed", tools_executed())
    # Someone speaks over the tool call, and the model's post-tool reply lands.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("actually wait", False))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("tool-answer"),
                                user_initiated=False))
    session.emit("metrics_collected", llm_metrics("tool-answer"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    asked = next(op for op in ops if op["type"] == "stt"
                 and "book the flight" in (op["response"].get("transcript") or ""))
    llm = next(op for op in ops if op["type"] == "llm")
    assert llm["turn_id"] == asked["turn_id"], (
        "the answer owed to this turn's tool result was handed to whoever "
        "spoke over it, leaving the turn that called the tool unanswered"
    )
    assert llm["response"].get("reply_attribution") == "inferred", (
        "a caller talking over the tool call could have prompted this reply "
        "instead, and the event says nothing either way"
    )


@pytest.mark.asyncio
async def test_a_contested_turns_derived_llm_span_carries_the_caveat_too(recorder):
    """The caveat is documented as covering that turn's spans, not one of them.

    When no LLM plugin emits `metrics_collected`, the span is reconstructed
    from `conversation_item_added` instead. That path publishes an `ok` LLM
    operation with a first-token latency on it, and it was the one publisher
    the disclosure never reached -- so an adopter comparing per-turn LLM
    latency saw a clean number for a reply that may answer a different turn.
    There are no tokens here, but a latency drawn from the wrong turn is wrong
    in exactly the same way, for free.

    The reply here comes from a build whose speech handle does not report
    whether it is scheduled, which is the only case left where the ownership
    really is a judgement -- and therefore the case where the disclosure has to
    reach every publisher.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    # No `scheduled` on this handle: an older build, where the reply's owner
    # cannot be read and the turn has to say so.
    answer = SimpleNamespace(id="answer-2", chat_items=[])
    assert not hasattr(answer, "scheduled"), (
        "guard: this test only covers the fallback while the handle stays silent"
    )
    session.emit("speech_created", speech_created(answer))
    # No `metrics_collected` for the LLM at all: the span has to be derived.
    # The item is linked to its speech the way LiveKit links it, so the reply
    # is matched by identity and the derived span is actually reached.
    item = chat_item("assistant", "It is thirty dollars.", {"llm_node_ttft": 0.3})
    answer.chat_items.append(item.item)
    session.emit("conversation_item_added", item)
    await rec.finish()

    llm = [op for op in operations(read_events(_dir(rec))) if op["type"] == "llm"]
    assert llm, "the derived LLM span must still be published"
    assert llm[0]["response"].get("estimated") is True, (
        "guard: this test is meaningless unless it exercised the derived path"
    )
    assert llm[0]["response"].get("reply_attribution") == "inferred", (
        "a derived span on a contested turn must disclose the same doubt the "
        "measured one does, or the caveat depends on which plugin you use"
    )


@pytest.mark.asyncio
async def test_a_swallowed_handler_error_does_not_hedge_the_next_innocent_turn(recorder):
    """`_guard` keeps the call alive, which must not corrupt later turns.

    The contested marker used to live on the recorder, set in one place and
    cleared in another. `_guard` deliberately swallows a handler exception and
    lets recording continue, so a transient failure in between left the marker
    set, and the next unrelated utterance was published carrying someone else's
    uncertainty.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))

    original = rec._new_turn
    calls = {"n": 0}

    def explode():
        calls["n"] += 1
        raise RuntimeError("transient bookkeeping failure")

    rec._new_turn = explode
    # Swallowed by `_guard`, exactly as a real transient failure would be.
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-2")))
    rec._new_turn = original
    assert calls["n"] == 1, "the test must actually exercise the failure path"

    # A later, entirely unrelated utterance.
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="greeting"), source="say"))
    session.emit("metrics_collected", tts_metrics("greeting", audio_duration=1.0))
    rec.tap_output_frame(agent_frame(1000))
    await rec.finish()

    for op in operations(read_events(_dir(rec))):
        if op.get("response", {}).get("audio_ms") == 1000:
            assert "reply_attribution" not in op["response"], (
                "this turn was never contested; the caveat leaked onto it"
            )


@pytest.mark.asyncio
async def test_an_uncontested_reply_is_not_labelled_a_guess(recorder):
    """A caveat on every turn is a caveat nobody reads.

    With no partial arriving during the filler there is nothing to be uncertain
    about, and the span must not carry the disclosure.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-1")))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()

    for op in operations(read_events(_dir(rec))):
        assert "reply_attribution" not in op.get("response", {}), (
            "nothing was contested, so nothing should be hedged"
        )


@pytest.mark.asyncio
async def test_a_filler_is_declared_even_when_its_metrics_have_not_arrived(recorder):
    """`say()` emits `speech_created` first and schedules its TTS after.

    Waiting for the filler's metrics before marking the reply meant losing a
    race that LiveKit's own ordering makes routine (`agent_activity.py:1435`):
    the generated reply is created before the filler is measured, so nothing
    marked the span and the filler's audio folded into the answer as if the
    answer had simply taken longer. Its duration may legitimately be unknown at
    this instant; that it happened at all is not.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    # No `tts_metrics` for the filler yet -- the answer wins the race.
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-1")))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec))) if op["type"] == "tts"]
    assert len(tts) == 1
    response = tts[0]["response"]
    assert response.get("reply_includes_filler") is True, (
        "a filler that ran before its meter reported is still a filler"
    )
    assert response.get("filler_audio_ms_unknown") is True, (
        "an admitted unknown, not a silent omission"
    )


@pytest.mark.asyncio
async def test_a_late_filler_segment_is_counted_as_the_fillers_own(recorder):
    """One `say()` can be synthesized in several segments.

    LiveKit finalises TTS metrics per segment (`tts.py:683`), so a filler
    straddling the answer's creation had only its already-landed segments
    snapshotted -- and every later one was quietly added to the answer's share,
    which is the inflation this whole mechanism exists to prevent. Segments are
    now attributed by the speech that produced them, whenever they arrive, and a
    late one also resolves a duration that was unknown at creation time.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    # First filler segment lands before the answer is created...
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.2))
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-1")))
    # ...and the second lands after.
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.2))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec))) if op["type"] == "tts"]
    assert len(tts) == 1
    response = tts[0]["response"]
    assert response["audio_ms"] == 2400, "the caller heard all of it either way"
    assert response.get("filler_audio_ms") == 400, (
        "the late segment was credited to the answer, overstating it by 200ms"
    )
    assert response.get("filler_audio_ms_unknown") is None


@pytest.mark.asyncio
async def test_the_answer_does_not_erase_the_fillers_words(recorder):
    """Both utterances commit an assistant item onto the one shared span.

    They deliberately share a span because the caller heard both. Assigning the
    text meant the answer's item overwrote the filler's, so a span that
    honestly reported 2400ms of audio reported only the answer's words -- and
    `char_count`, which is kept even when content capture is off, understated
    by exactly the filler.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    filler = FakeSpeechHandle("filler-1")
    session.emit("speech_created", speech_created(filler, source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    answer = FakeSpeechHandle("answer-1")
    session.emit("speech_created", speech_created(answer))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    filler_item = chat_item("assistant", "Let me check that.")
    filler.chat_items.append(filler_item.item)
    session.emit("conversation_item_added", filler_item)
    answer_item = chat_item("assistant", "The fare is 120 dollars.")
    answer.chat_items.append(answer_item.item)
    session.emit("conversation_item_added", answer_item)
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec))) if op["type"] == "tts"]
    assert len(tts) == 1
    response = tts[0]["response"]
    assert "Let me check that." in (response.get("text") or ""), (
        "the filler's words were overwritten by the answer's"
    )
    assert "The fare is 120 dollars." in (response.get("text") or "")


@pytest.mark.asyncio
async def test_stt_tokens_reported_before_the_final_are_published(recorder):
    """A recogniser that meters mid-utterance reports tokens there too.

    The closing path copied only the metered audio out of the pending span, so
    every token counted before the final was dropped. The turn then showed
    metered seconds and no tokens at all -- two billing numbers from one
    provider that could never be reconciled against each other.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", False))
    session.emit("metrics_collected",
                 stt_metrics(audio_duration=2.0, input_tokens=140, output_tokens=12))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"]
    assert len(stt) == 1
    response = stt[0]["response"]
    assert response.get("provider_metered_audio_ms") == 2000
    assert response.get("input_tokens") == 140, "tokens counted before the final"
    assert response.get("output_tokens") == 12


@pytest.mark.asyncio
async def test_post_final_metering_publishes_the_subtractable_amount(recorder):
    """Routing it to the closed turn is right; hiding how much is not.

    A recogniser that meters per utterance (OpenAI) puts real speech after the
    final; one that meters the connection (Deepgram, every five seconds off
    frames streamed) puts inter-turn silence there. Nothing in the payload
    tells them apart, so the amount that arrived after the caller stopped
    talking is published rather than guessed at.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    # The meter fires after the final, onto an untouched pending span.
    session.emit("metrics_collected", stt_metrics(audio_duration=5.0))
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"]
    response = stt[0]["response"]
    assert response.get("metered_after_final") is True
    assert response.get("metered_after_final_ms") == 5000, (
        "without the amount, silence and speech are indistinguishable"
    )
    # The whole meter arrived after the caller stopped. That is a fact about
    # arrival and is recorded as one -- it is deliberately NOT read as evidence
    # that the meter is connection-scoped, because OpenAI sends a true
    # per-utterance meter in exactly this position.
    assert response.get("metered_arrival") == "after_final"
    assert response.get("metering_scope") == "unknown"


@pytest.mark.asyncio
async def test_the_recorder_never_infers_billing_semantics_from_arrival_time(recorder):
    """Timing cannot tell you what a provider's meter measures.

    OpenAI emits an item's final transcript and then that same item's usage,
    computed from the item's own start and end -- a per-utterance meter that
    arrives entirely after the final. Deepgram pushes every streamed frame into
    a five-second collector and emits each tick, so connection-scoped audio
    routinely reports while the caller is still speaking. Any rule that reads
    scope off arrival is therefore wrong in both directions, and wrong silently,
    on a number people bill from.

    So arrival is published as arrival, and scope says it does not know.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    # A meter landing mid-speech -- the shape Deepgram's five-second tick makes
    # even though its meter is connection-scoped.
    session.emit("metrics_collected", stt_metrics(audio_duration=3.0))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"]
    response = stt[0]["response"]
    assert response.get("provider_metered_audio_ms") == 3000
    assert response.get("metered_arrival") == "before_final", (
        "arrival is observable and must still be reported"
    )
    assert response.get("metering_scope") == "unknown", (
        "a meter arriving mid-speech is not thereby an utterance meter"
    )
    assert "provider" in (response.get("metering_scope_note") or ""), (
        "the payload must carry the caveat, not leave it to documentation"
    )
    assert response.get("metered_after_final") is not True


@pytest.mark.asyncio
async def test_a_meter_split_across_the_final_is_reported_as_split(recorder):
    """A connection meter that ticks either side of the final is the common case.

    Deepgram's five-second collector will routinely have some of its audio
    before a final transcript and some after. Reporting only `before_final` or
    only `after_final` would misdescribe it, so the third value has to exist and
    has to be reachable -- an unreachable enum member is a lie in the schema.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("metrics_collected", stt_metrics(audio_duration=3.0))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    # A second tick lands after the final, so the meter straddles it.
    session.emit("metrics_collected", stt_metrics(audio_duration=2.0))
    await rec.finish()

    stt = [op for op in operations(read_events(_dir(rec))) if op["type"] == "stt"]
    response = stt[0]["response"]
    total = response.get("provider_metered_audio_ms")
    after = response.get("metered_after_final_ms")
    assert total and after and 0 < after < total, (
        f"this scenario must produce a genuinely split meter, got {after}/{total}"
    )
    assert response.get("metered_arrival") == "straddles_final"
    assert response.get("metering_scope") == "unknown", (
        "straddling the final says nothing about what is being metered either"
    )


@pytest.mark.asyncio
async def test_a_call_that_starts_untapped_is_not_certified_by_a_later_tap(recorder):
    """The loss is the same loss whichever order the handoff went in.

    Remembering only tapped->untapped left the mirror image certifying itself:
    a call whose *first* agent could not be tapped, handed off to one that
    could, ends with `agent_audio_tapped: True` -- which suppresses the
    never-tapped gap -- and no record of loss. Everything the opening agent
    said is missing, and the manifest calls the capture complete.
    """
    class _Framework:
        async def tts_node(self, text, model_settings):  # noqa: ANN001
            yield None

    class Tapped(VaaniAudioTapMixin, _Framework):
        pass

    class Untapped(_Framework, VaaniAudioTapMixin):
        pass

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    # The call opens with an agent we cannot hear.
    rec.bind_agent(Untapped())
    # It is then handed off to one we can.
    later = Tapped()
    later.vaani = rec
    rec.note_audio_tap_installed(later)
    await rec.finish()

    capture = _manifest_of(rec)["capture_status"]
    assert capture["measured"]["agent_audio_tapped"] is True
    assert capture["coverage_complete"] is False, (
        "the opening agent's speech was never captured"
    )
    assert any(gap.get("agent_tap_lost_on_handoff")
               for gap in capture["coverage_gaps"])


@pytest.mark.asyncio
async def test_a_cough_does_not_detach_a_delayed_answer_from_its_question(recorder):
    """VAD firing is not the same fact as another caller having spoken.

    A callback can issue a filler, await, and return only after some noise has
    opened a VAD interval; LiveKit then creates *that* caller's reply, after the
    callback returns (`agent_activity.py:2547`). Gating on VAD start refused the
    adoption and filed the answer in a turn of its own -- exactly the detached
    answer with no question in it that this rule exists to prevent. A preemptive
    reply is generated from a transcript, never from VAD alone, so a partial is
    what marks a genuine next caller.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(SimpleNamespace(id="filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    # A cough. VAD opens an interval; no transcript ever comes of it.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    # The awaited callback returns and the real answer is generated.
    session.emit("speech_created", speech_created(SimpleNamespace(id="answer-1")))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    rec.tap_output_frame(agent_frame(2000))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    question = [op for op in ops
                if op["type"] == "stt"
                and "fare" in (op["response"].get("transcript") or "")]
    answer = [op for op in ops
              if op["type"] == "tts" and op["response"].get("audio_ms") == 2400]
    assert question and answer, "both must be recorded"
    assert answer[0]["turn_id"] == question[0]["turn_id"], (
        "the answer was detached from the question a cough came between"
    )


@pytest.mark.asyncio
async def test_an_answer_contained_in_the_filler_does_not_erase_it(recorder):
    """Containment means already recorded, not supersedes.

    A short answer can be a substring of a chatty filler. Treating that as "no
    new words" and then falling through to a plain assignment replaced the
    filler's whole transcript with the fragment, while `char_count` still
    described both -- text and count disagreeing about the same span.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    filler = FakeSpeechHandle("filler-1")
    session.emit("speech_created", speech_created(filler, source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    answer = FakeSpeechHandle("answer-1")
    session.emit("speech_created", speech_created(answer))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    filler_item = chat_item("assistant", "The fare is ready")
    filler.chat_items.append(filler_item.item)
    session.emit("conversation_item_added", filler_item)
    # A one-word answer that happens to appear inside the filler.
    answer_item = chat_item("assistant", "fare")
    answer.chat_items.append(answer_item.item)
    session.emit("conversation_item_added", answer_item)
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec))) if op["type"] == "tts"]
    response = tts[0]["response"]
    assert "The fare is ready" in (response.get("text") or ""), (
        "the filler's words were replaced by a fragment of themselves"
    )
    assert "The fare is ready fare" == (response.get("text") or ""), (
        "the caller heard two separate utterances and both belong on the span"
    )
    assert response.get("char_count") == len("The fare is ready") + len("fare"), (
        "text and char_count must describe the same speech"
    )


@pytest.mark.asyncio
async def test_the_same_item_delivered_twice_is_only_counted_once(recorder):
    """Identity, not words, is what makes an utterance a repeat.

    The span has to answer "have I already recorded this" without deciding it
    from the text, because two utterances can legitimately share their words --
    an agent that says "one moment" twice said it twice. LiveKit gives every
    chat item an id, so a redelivery of one item is distinguishable from two
    items that happen to read alike.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("Are you there?", True))
    session.emit("conversation_item_added", chat_item("user", "Are you there?"))
    filler = FakeSpeechHandle("filler-1")
    session.emit("speech_created", speech_created(filler, source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    answer = FakeSpeechHandle("answer-1")
    session.emit("speech_created", speech_created(answer))
    session.emit("metrics_collected", tts_metrics("answer-1", audio_duration=2.0))
    filler_item = chat_item("assistant", "One moment", item_id="item-same")
    filler.chat_items.append(filler_item.item)
    session.emit("conversation_item_added", filler_item)
    # The very same item delivered a second time.
    session.emit("conversation_item_added", filler_item)
    # A different item whose words are identical -- genuinely spoken twice.
    twin = chat_item("assistant", "One moment", item_id="item-other")
    answer.chat_items.append(twin.item)
    session.emit("conversation_item_added", twin)
    await rec.finish()

    tts = [op for op in operations(read_events(_dir(rec))) if op["type"] == "tts"]
    response = tts[0]["response"]
    assert response.get("char_count") == len("One moment") * 2, (
        "a redelivered item must not add, and a genuine repeat must"
    )
    assert (response.get("text") or "") == "One moment One moment", (
        "the agent said it twice, so the transcript says it twice"
    )


@pytest.mark.asyncio
async def test_a_reply_from_a_replaced_public_method_is_not_the_last_callers_answer(
        recorder, public_generate_reply, tmp_path):
    """A missing base code object proves absence, not innocence.

    `AgentActivity._generate_reply()` (`agent_activity.py:1506`) is what emits
    this event, and LiveKit reaches it directly for the automatic answer to a
    completed turn (`:2574`). A subclass or wrapper that stands in for the
    public `generate_reply` reaches the same function with an identical event
    and an identical handle -- only the base code object is gone. Matching on
    that code object alone therefore read an application's own reply as
    LiveKit's automatic answer, merged it backwards into the previous caller's
    turn, and billed them its tokens with no caveat at all.
    """
    application = tmp_path / "app" / "tracing_session.py"
    application.parent.mkdir(parents=True)
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    public_generate_reply.reach_emitter(session, "wrapped-reply", _from=application)
    session.emit("metrics_collected", llm_metrics("wrapped-reply"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "llm")
    question = next(op for op in ops if op["type"] == "stt"
                    and "What is the fare?" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] != question["turn_id"], (
        "a reply made through a replaced public method was published as the "
        "certain answer to whoever spoke last"
    )
    assert llm["response"].get("reply_attribution") == "inferred", (
        "the substitution must be disclosed, not resolved silently"
    )
    assert "standing in for" in (llm["response"].get("reply_attribution_reason") or ""), (
        "the caveat must say what it actually found on the stack"
    )


@pytest.mark.asyncio
async def test_livekits_own_automatic_answer_still_belongs_to_the_turn_it_answers(
        recorder, public_generate_reply):
    """The common case must not acquire a caveat to fix the rare one.

    LiveKit reaches the emitting function directly for the automatic answer to
    a completed turn, and that reply really does answer the caller who just
    finished speaking. Separating it from a replaced public method has to be
    done by *who called the emitter*, not by the emitter being reached at all --
    otherwise every ordinary answer in every call is hedged and the attribution
    stops meaning anything.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    # The same contested moment as the test above: an interim arrives over the
    # filler, so the reply's origin is what decides whose turn it lands in.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("um", False))
    public_generate_reply.reach_emitter(
        session, "auto-reply", _from=public_generate_reply.livekit_file)
    session.emit("metrics_collected", llm_metrics("auto-reply"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "llm")
    question = next(op for op in ops if op["type"] == "stt"
                    and "What is the fare?" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] == question["turn_id"], (
        "LiveKit's own automatic answer was detached from the question it answered"
    )
    assert llm["response"].get("reply_attribution") is None, (
        "an answer whose origin the stack established must not be hedged"
    )


@pytest.mark.asyncio
async def test_a_livekit_build_whose_modules_moved_is_not_read_as_an_automatic_answer(
        recorder, monkeypatch):
    """No livekit at all and a livekit that moved are different facts.

    Import failure was one answer: "there is no such public method, so its
    absence from the stack is a fact about the call". That holds when
    `livekit.agents` is not in the process at all. It does not hold when the
    package is right there and only this module moved -- then a real session is
    generating real replies and the SDK simply cannot read them, which is the
    one case that must not resolve to "LiveKit's answer to whoever spoke last".
    """
    monkeypatch.setattr(livekit_integration, "_GENERATE_REPLY_CODE", None)
    monkeypatch.setattr(livekit_integration, "_livekit_agents_present",
                        lambda: True)

    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    session.generate_reply(handle_id="moved-reply")
    session.emit("metrics_collected", llm_metrics("moved-reply"))
    await rec.finish()

    llm = next(op for op in operations(read_events(_dir(rec)))
               if op["type"] == "llm")
    assert llm["response"].get("reply_attribution") == "inferred", (
        "a build this SDK cannot read reported its replies as certain"
    )


@pytest.mark.asyncio
async def test_the_framework_caveat_names_a_retry_of_a_run_that_already_exists(
        recorder, public_generate_reply):
    """A reissue for an existing run must not be billed to the next caller.

    LiveKit's internal calls were all placed with the speech in flight, which
    also makes the next final transcript join that turn.
    `RunResult._maybe_retry_output()` (`run_result.py:292`) and the realtime
    fallback adapter (`realtime_fallback_adapter.py:394` and `:445`) are not
    answering anybody speaking: they reissue a reply for a run that already
    exists. Placing them in flight handed a retry's tokens, latency and cost to
    whoever spoke next, and put that person's words on the retry's turn -- an
    exchange neither of them had. The reissue is kept in a turn of its own, and
    the caveat names the two call sites so an adopter can recognise their case.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    retry_from = public_generate_reply.package_root / "agents" / "voice" / "run_result.py"
    public_generate_reply(session, handle_id="retry-reply", _from=retry_from)
    session.emit("metrics_collected", llm_metrics("retry-reply"))
    # The person who spoke over the filler now finishes. Their words are their
    # own; the reissue was never an answer to them.
    session.emit("user_input_transcribed", transcript("And to Boston?", True))
    session.emit("conversation_item_added", chat_item("user", "And to Boston?"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "llm")
    later = next(op for op in ops if op["type"] == "stt"
                 and "Boston" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] != later["turn_id"], (
        "a reply reissued for an existing run was placed with the speech in "
        "flight, so the next caller's turn was charged for it and their "
        "transcript was recorded as the question it answered"
    )
    reason = llm["response"].get("reply_attribution_reason") or ""
    assert "run that already exists" in reason and "run_result.py:292" in reason, (
        "the caveat does not name the reissue, so an adopter whose reply was "
        "retried for an existing run cannot tell that it applies to them"
    )


@pytest.mark.asyncio
async def test_attaching_while_finish_releases_the_call_leaves_nothing_behind(
        recorder):
    """The check and the transition it guards have to be under one lock.

    `attach()` read `self.call` before taking the lock, so a thread could see a
    live call, wait, and register all nine handlers *after* `finish()` had
    released the call and run an empty `_detach()`. `_guard` makes them inert,
    so nothing is corrupted -- but nothing removes them either, and a long-lived
    session keeps the finished recorder and every turn it holds for as long as
    it lives. Repeating the race accumulates both.
    """
    import threading

    rec = recorder()
    session = FakeAgentSession()
    rec.note_audio_tap_installed()
    # Held here so the attaching thread reaches the lock and waits on it. The
    # lock is reentrant, so `finish()` on this thread still proceeds -- which is
    # exactly the interleaving being pinned.
    rec._attach_lock.acquire()
    try:
        attaching = threading.Thread(target=rec.attach, args=(session,))
        attaching.start()
        time.sleep(0.05)
        await rec.finish()
    finally:
        rec._attach_lock.release()
    attaching.join(timeout=5)

    assert rec.call is None, "guard: this test is meaningless unless finish ran"
    assert session.handlers == {}, (
        "handlers were registered on a live session after the recorder had "
        "finished, and nothing is left to remove them"
    )


@pytest.mark.asyncio
async def test_a_cancelled_replys_caveat_does_not_follow_the_turn_it_left_behind(
        recorder, public_generate_reply):
    """Reusing an empty turn must not mean inheriting its doubts.

    A framework reply that LiveKit cancels before any metric, audio or tool
    leaves its turn empty, and the next preemptive reply reuses that turn on
    purpose so cancelled attempts do not surface as phantoms. But the caveat
    stored on it described the *cancelled* reply. Left in place, it published a
    doubt about a reply whose own handle had established where it belongs --
    the same state leak the per-reply flag exists to prevent, in the other
    direction.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    # LiveKit asks for a reply itself, which is placed with the speech in
    # flight and carries the caveat that its internal callers are not all
    # answers to it. It is then cancelled before it measures anything at all,
    # so its turn stays empty -- and stays reusable.
    public_generate_reply(session, handle_id="framework-reply")
    # The contested decision is taken one loop slice later.
    await asyncio.sleep(0)
    left_behind = rec._preemptive_turn
    assert left_behind is not None and left_behind.reply_attribution is not None, (
        "guard: this test is meaningless unless the cancelled reply left a "
        "caveat on an empty turn for the next one to inherit"
    )
    # The ordinary preemptive reply that follows reuses that empty turn. Its
    # own handle says where it belongs, so it has nothing to be hedged about.
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("known-reply", scheduled=False)))
    assert rec._preemptive_turn is left_behind, (
        "guard: this test is meaningless unless the empty turn was reused"
    )
    session.emit("metrics_collected", llm_metrics("known-reply"))
    session.emit("user_input_transcribed", transcript("And to Boston?", True))
    session.emit("conversation_item_added", chat_item("user", "And to Boston?"))
    await rec.finish()

    llm = next(op for op in operations(read_events(_dir(rec)))
               if op["type"] == "llm")
    assert llm["response"].get("reply_attribution") is None, (
        "a reply whose own handle established its origin carried a cancelled "
        "reply's caveat"
    )
    assert llm["response"].get("reply_attribution_reason") is None, (
        "the cancelled reply's reason outlived the reply it described"
    )


def test_a_warning_is_not_prefixed_with_the_package_name_twice():
    """`_warn_once` adds the prefix, so a message must not carry its own.

    Nine call sites passed a message that already began with `vaani: ` into the
    helper that prepends it, and every one of those warnings reached adopters
    reading `vaani: vaani: ...`. It is cosmetic, but it is the first thing an
    adopter sees when something has gone wrong with their recording, and it
    reads as though the product cannot keep track of its own output.
    """
    import ast

    source = pathlib.Path(livekit_integration.__file__).read_text()
    doubled = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name != "_warn_once":
            continue
        for argument in node.args:
            value = getattr(argument, "value", None)
            if isinstance(value, str) and value.startswith("vaani: "):
                doubled.append((node.lineno, value[:60]))
    assert doubled == [], (
        f"messages carry the prefix the helper already adds: {doubled}"
    )


@pytest.mark.asyncio
async def test_an_override_that_reaches_past_the_emitter_is_not_the_last_answer(
        recorder, public_generate_reply):
    """The last silent merge: no base method, no emitting frame, no anchor.

    A tracing or custom `AgentSession` subclass can override the public method
    and reach a *different* protected API, so neither the base code object nor
    `AgentActivity._generate_reply()` is on the stack. That is indistinguishable
    from LiveKit's automatic answer by every other means, and reading it as one
    merged the application's reply backwards into the previous caller's turn
    with nothing recorded about it -- the one direction a failure to identify
    must never take.

    A frame calling itself `generate_reply` on an `AgentSession` is weaker
    evidence than either anchor, which is why it is consulted last. It is still
    enough to know the reply was not the automatic one.
    """
    rec = recorder()

    class OverriddenSession(FakeAgentSession):
        def generate_reply(self, **kwargs):  # noqa: D401 - stands in
            handle_id = kwargs.pop("handle_id", "override-reply")
            self.emit("speech_created",
                      speech_created(FakeSpeechHandle(handle_id)))

    session = OverriddenSession()
    rec.attach(session)
    rec.note_audio_tap_installed()
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("What is the fare?", True))
    session.emit("conversation_item_added", chat_item("user", "What is the fare?"))
    session.emit("speech_created",
                 speech_created(FakeSpeechHandle("filler-1"), source="say"))
    session.emit("metrics_collected", tts_metrics("filler-1", audio_duration=0.4))
    rec.tap_output_frame(agent_frame(400))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("And to Boston?", False))
    session.generate_reply(handle_id="override-reply")
    session.emit("metrics_collected", llm_metrics("override-reply"))
    await rec.finish()

    ops = operations(read_events(_dir(rec)))
    llm = next(op for op in ops if op["type"] == "llm")
    asked = next(op for op in ops if op["type"] == "stt"
                 and "fare" in (op["response"].get("transcript") or ""))
    assert llm["turn_id"] != asked["turn_id"], (
        "a reply from a replaced public method was merged into the previous "
        "caller's turn, so their exchange reports tokens they never prompted"
    )
    assert llm["response"].get("reply_attribution") == "inferred", (
        "the reply was separated on a guess, and a reader deciding a latency "
        "from it is owed that"
    )
