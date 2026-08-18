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


def chat_item(role: str, text: str, metrics: dict | None = None):
    return SimpleNamespace(item=SimpleNamespace(role=role, text_content=text, metrics=metrics or {}))


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
    assert tts["duration_ms"] == 1100
    assert tts["milestones"]["first_byte"]["occurred_at_ms"] == tts["started_at_ms"] + 250
    assert tts["response"]["audio_ms"] == 2000
    assert tts["milestones"]["turn_report"]["e2e_latency_ms"] == 1400
    assert tts["response"]["text"] == "Goa ki flight 6000 rupaye"


async def test_a_cancelled_tts_metric_marks_the_span_cancelled(recorder):
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
    assert _by_type(rec, "tts")["status"] == "cancelled"


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


def _by_type(rec: VaaniLiveKitRecorder, kind: str) -> dict:
    matches = [event for event in operations(read_events(_dir(rec))) if event["type"] == kind]
    assert matches, f"no {kind} operation was recorded"
    return matches[0]


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
