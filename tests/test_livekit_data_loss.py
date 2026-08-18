"""Regressions for the data-loss findings in the LiveKit integration review.

Kept apart from `test_livekit_integration.py` because that file describes what
the recorder *does*; this one pins the ways it used to lose a recording without
saying so.

Twenty-one of these fail against the pre-fix source (`git show
HEAD~1:src/vaani_observer/integrations/livekit.py`) and are proofs of a specific
past bug. The remaining seven guard behaviour that did not exist before the fix
and therefore cannot fail against it:

    test_a_second_finish_is_a_no_op
    test_a_derived_llm_span_never_duplicates_a_measured_one
    test_no_llm_span_is_invented_on_a_turn_that_never_replied
    test_an_explicit_zero_timeout_still_means_no_budget
    test_a_plugin_whose_model_property_raises_does_not_kill_the_recording
    test_played_ms_is_absent_rather_than_zero_when_no_audio_was_taped
    test_receipt_bookkeeping_can_never_fail_a_successful_upload

That distinction is recorded because an earlier version of this file claimed
all of them failed pre-fix. They did -- but on a `TypeError` from the autouse
fixture passing a `timeout` argument `finish()` did not yet accept, not on
anything they asserted. A verification that cannot fail is not a verification.
See `_capture_finalized`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from conftest import operations, read_events
from test_livekit_integration import (
    FakeAgentSession,
    FakeFrame,
    chat_item,
    llm_metrics,
    stt_metrics,
    transcript,
    tts_metrics,
)
from vaani_observer import VaaniObserver
from vaani_observer.integrations.livekit import (
    VaaniLiveKitRecorder,
    observe_agent_session,
)


@pytest.fixture(autouse=True)
def _capture_finalized(monkeypatch):
    """Keep the finalized package reachable after `finish()` releases the call."""
    import inspect

    original = VaaniLiveKitRecorder.finish
    # `finish()` gained a `timeout` parameter as part of this work. Passing it
    # unconditionally made *every* test in this file raise `TypeError` against
    # pre-fix source, so "it fails before the fix" was true of all of them for
    # a reason unrelated to what they assert. Adapting to the signature is what
    # makes the pre-fix run a real check rather than a guaranteed one.
    accepts_timeout = "timeout" in inspect.signature(original).parameters

    async def finish(self, outcome=None, timeout=None):
        call = self.call
        if accepts_timeout:
            await original(self, outcome, timeout)
        else:
            await original(self, outcome)
        if call is not None:
            self._finalized = await call.finished

    monkeypatch.setattr(VaaniLiveKitRecorder, "finish", finish)


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


# ------------------------------------------------------- P0-1: events after finish


async def test_events_after_finish_are_inert_rather_than_an_attributeerror(recorder, caplog):
    """The documented `finally: await recorder.finish()` fires mid-call.

    `start()` returns when the session starts, not when it ends, so the session
    keeps emitting long after `finish()` released the call. Every one of those
    events used to dereference `None` and be swallowed at DEBUG.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    # Snapshot the handlers LiveKit already holds. `off()` is not atomic with
    # respect to events LiveKit has *already dispatched*, so a handler firing
    # after `finish()` is a real race, not a hypothetical one -- and it is the
    # only thing that still exercises this path once unsubscribe works.
    dispatched = {
        name: list(handlers) for name, handlers in session.handlers.items() if handlers
    }
    assert dispatched, "attach() must have registered handlers"
    await rec.finish()

    with caplog.at_level(logging.DEBUG, logger="vaani_observer.livekit"):
        for name, event in (
            ("user_state_changed", SimpleNamespace(new_state="speaking")),
            ("user_input_transcribed", transcript("still talking", True)),
            ("metrics_collected", llm_metrics("speech-late")),
            ("conversation_item_added", chat_item("assistant", "late")),
        ):
            for handler in dispatched.get(name, ()):
                handler(event)

    # Asserting on the *absence of a swallowed-failure record* rather than on
    # the string "AttributeError". The pre-fix code logged
    # "'NoneType' object has no attribute 'now'" -- which never contains the
    # class name -- at DEBUG, so both of the original assertions held on the
    # buggy code and the test proved nothing.
    failures = [r for r in caplog.records if "failed" in r.getMessage()]
    assert not failures, f"handlers must be inert after finish(), got: {failures}"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_finish_unsubscribes_every_handler_it_registered(recorder):
    """Leaving handlers attached is what made post-finish events possible."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    assert any(session.handlers.values())

    await rec.finish()

    assert not any(session.handlers.values()), "finish() must unsubscribe"


async def test_a_second_finish_is_a_no_op(recorder):
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await rec.finish()
    await rec.finish()  # must not raise


# ------------------------------ P0-1: the shutdown hook, not a try/finally


async def test_observe_agent_session_finishes_from_the_shutdown_hook(recorder):
    """The fix for the truncated-call bug: finish when the job ends."""
    rec = recorder()
    session = FakeAgentSession()
    callbacks: list = []
    job_ctx = SimpleNamespace(add_shutdown_callback=callbacks.append)

    observe_agent_session(session, rec, job_ctx=job_ctx)
    assert callbacks, "a shutdown callback must be registered"

    await run_turn(session)
    assert rec.call is not None, "the call must still be open while the job runs"

    await callbacks[0]()
    assert rec.call is None


async def test_the_livekit_shutdown_reason_is_not_used_as_the_call_outcome(recorder):
    """`add_shutdown_callback` passes its reason positionally.

    A callback that accepts one argument is handed the raw reason string, so
    registering `recorder.finish` directly would have written arbitrary LiveKit
    text into `outcome`, which is a closed vocabulary the dashboard groups by.
    """
    rec = recorder()
    session = FakeAgentSession()
    callbacks: list = []
    job_ctx = SimpleNamespace(add_shutdown_callback=callbacks.append)
    observe_agent_session(session, rec, job_ctx=job_ctx)
    await run_turn(session)
    call = rec.call

    await callbacks[0]("job terminated by the worker")

    finalized = await call.finished
    assert finalized.manifest["outcome"] != "job terminated by the worker"
    assert finalized.manifest["outcome"] in (None, "completed", "failed", "abandoned")


# -------------------------------------------------- P1-4: LLM spans without metrics


async def test_llm_spans_survive_an_agent_that_emits_no_llm_metrics(recorder):
    """`_record_llm` has one call site, so losing metrics lost every LLM span."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("conversation_item_added", chat_item("user", "hello"))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="s1"), source="generate_reply"),
    )
    # Deliberately no `metrics_collected` at all.
    session.emit(
        "conversation_item_added",
        chat_item("assistant", "hi there",
                  {"e2e_latency": 1.4, "llm_node_ttft": 0.3, "llm_node_ttfs": 0.5}),
    )
    await rec.finish()

    llm = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "llm"]
    assert len(llm) == 1, "a derived LLM span must exist"
    assert llm[0]["request"]["derived_from"] == "conversation_item_added"
    assert llm[0]["response"]["estimated"] is True
    assert llm[0]["response"]["ttft_ms"] == 300
    # A derived span must never invent token counts.
    assert "total_tokens" not in llm[0]["response"]


async def test_a_derived_llm_span_never_duplicates_a_measured_one(recorder):
    """When metrics do arrive, the fallback must stay out of the way."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_turn(session)
    await rec.finish()

    llm = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "llm"]
    assert len(llm) == 1
    assert llm[0]["response"]["total_tokens"] == 120
    assert "derived_from" not in llm[0].get("request", {})


# ------------------------------------------------------------ P1-5: model labels


async def test_an_azure_deployment_name_beats_the_plugins_default_model(recorder):
    """`with_azure` defaults `model="gpt-4o"`; reporting it verbatim is a lie."""
    rec = recorder()
    session = FakeAgentSession()
    session.llm = SimpleNamespace(
        _client=SimpleNamespace(_azure_deployment="gpt-5-mini-deployment")
    )
    rec.attach(session)
    await run_turn(session)
    await rec.finish()

    llm = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "llm"][0]
    assert llm["model"] == "gpt-5-mini-deployment"
    # The plugin's claim is kept as evidence, not silently discarded.
    assert llm["request"]["reported_model"] == "gpt-4o"


async def test_an_explicit_model_override_wins_over_everything(recorder):
    rec = recorder(model_overrides={"llm": "gpt-5-mini"})
    session = FakeAgentSession()
    session.llm = SimpleNamespace(_client=SimpleNamespace(_azure_deployment="ignored"))
    rec.attach(session)
    await run_turn(session)
    await rec.finish()

    llm = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "llm"][0]
    assert llm["model"] == "gpt-5-mini"


# ------------------------------------------------------- P1-7: real sample rates


async def test_the_span_reports_the_sample_rate_actually_observed(recorder):
    """It used to report the constructor default no matter what arrived."""
    rec = recorder(input_sample_rate=24000)
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    rec.tap_input_frame(FakeFrame(b"\x01\x00" * 160, sample_rate=16000))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    await rec.finish()

    stt = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "stt"][0]
    assert stt["request"]["sample_rate_hz"] == 16000


# ------------------------------------------- P1-8: a degraded recording says so


async def test_dropped_audio_marks_the_manifest_incomplete(recorder):
    """A manifest claiming `audio_complete` over a gap is the worst outcome."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    rec.tap_input_frame(FakeFrame(b"\x01\x00" * 160, sample_rate=24000))
    # A mid-call format change is rejected by the session writer.
    rec.tap_input_frame(FakeFrame(b"\x01\x00" * 160, sample_rate=48000))
    await rec.finish()

    capture = rec._finalized.manifest.get("capture_status") or {}
    assert capture.get("audio_complete") is False


# --------------------------------------------- P2-9: a disabled recorder is loud


def test_enabled_but_broken_configuration_is_reported_at_error(monkeypatch, caplog):
    monkeypatch.setenv("VAANI_ENABLED", "true")
    monkeypatch.setenv("VAANI_ENDPOINT", "not a url at all")
    monkeypatch.setattr(
        "vaani_observer.VaaniObserver.__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad endpoint")),
    )
    with caplog.at_level(logging.ERROR, logger="vaani_observer.livekit"):
        rec = VaaniLiveKitRecorder.from_env()

    assert rec.enabled is False
    assert rec.last_error is not None
    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_connection_capture_stays_off_until_endpoints_are_configured(monkeypatch):
    """Patching httpx with no rules is pure cost and captures nothing."""
    monkeypatch.setenv("VAANI_ENABLED", "true")
    monkeypatch.delenv("VAANI_ENDPOINTS", raising=False)
    rec = VaaniLiveKitRecorder.from_env()
    assert rec._observer.options["instrumentations"] == {"http": False, "websocket": False}

    monkeypatch.setenv(
        "VAANI_ENDPOINTS",
        '[{"id":"deepgram","type":"stt","url":"wss://api.deepgram.com","match":"origin"}]',
    )
    rec = VaaniLiveKitRecorder.from_env()
    try:
        assert rec._observer.options["instrumentations"]["http"] is True
        assert [rule["id"] for rule in rec._observer.endpoint_rules] == ["deepgram"]
    finally:
        # These patches are process-global; leaving them installed would leak
        # into every test that runs afterwards.
        rec._observer.uninstall_instrumentation()


def test_malformed_endpoint_json_disables_capture_without_killing_the_recorder(
    monkeypatch, caplog
):
    monkeypatch.setenv("VAANI_ENABLED", "true")
    monkeypatch.setenv("VAANI_ENDPOINTS", "{not json")
    with caplog.at_level(logging.ERROR, logger="vaani_observer.livekit"):
        rec = VaaniLiveKitRecorder.from_env()
    assert rec._observer.endpoint_rules == []
    assert "VAANI_ENDPOINTS" in caplog.text


# ----------------------------------------------------------------------- helpers


async def run_turn(session: FakeAgentSession, speech_id: str = "speech-1") -> None:
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit("conversation_item_added", chat_item("user", "hello"))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id=speech_id), source="generate_reply"),
    )
    session.emit("metrics_collected", llm_metrics(speech_id))
    session.emit("metrics_collected", tts_metrics(speech_id))
    session.emit(
        "conversation_item_added",
        chat_item("assistant", "hi", {"e2e_latency": 1.4, "llm_node_ttft": 0.3}),
    )


# ------------------------------------ reviewer: turn rotation misattributes replies


async def test_a_reply_is_credited_to_the_turn_that_spoke_it_not_the_next_one(recorder):
    """`conversation_item_added(assistant)` lands after full audio playout.

    A caller who barges in during that window rotates `_current_turn` onto a
    brand-new turn, so resolving the reply by "current" folded the report -- and
    any span derived from it -- into a turn that never called an LLM.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    # Turn 1: the caller speaks and the agent starts replying.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="speech-1"), source="generate_reply"),
    )
    session.emit("metrics_collected", llm_metrics("speech-1"))
    session.emit("metrics_collected", tts_metrics("speech-1"))

    # Turn 2 opens and closes while turn 1's reply is still playing out.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("actually wait", True))
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))

    # Only now does turn 1's assistant item arrive.
    session.emit(
        "conversation_item_added",
        chat_item("assistant", "hi", {"e2e_latency": 1.4, "llm_node_ttft": 0.3}),
    )
    await rec.finish()

    events = read_events(rec._finalized.directory)
    spans = operations(events)
    tts = [event for event in spans if event["type"] == "tts"]
    assert len(tts) == 1
    # The report must sit on the TTS span that actually produced the audio,
    # and nowhere else.
    assert "turn_report" in tts[0]["milestones"]
    assert tts[0]["milestones"]["turn_report"]["e2e_latency_ms"] == 1400
    assert [span["type"] for span in spans if "turn_report" in span["milestones"]] == ["tts"]


async def test_no_llm_span_is_invented_on_a_turn_that_never_replied(recorder):
    """With no speech to attribute to, deriving would fabricate a measurement."""
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    # A turn with no `speech_created` at all: nothing identifies a reply.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit(
        "conversation_item_added",
        chat_item("assistant", "hi", {"llm_node_ttft": 0.3, "llm_node_ttfs": 0.5}),
    )
    await rec.finish()

    llm = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "llm"]
    assert llm == [], "a turn with no synthesized reply must not gain an LLM span"


async def test_agent_audio_is_credited_to_the_turn_being_spoken(recorder):
    """Overlapping turns made `audio_ms` and `audio_bytes` disagree by ~1900x.

    Output frames were credited to `_current_turn`, which the interrupting turn
    had already claimed, so the speaking turn's span reported a duration with
    almost none of its bytes.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("metrics_collected", stt_metrics())
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="speech-1"), source="generate_reply"),
    )

    # The caller barges in, rotating `_current_turn`, while the agent speaks on.
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("wait", True))
    for _ in range(10):
        rec.tap_output_frame(FakeFrame(b"\x01\x00" * 240, sample_rate=24000))

    session.emit("metrics_collected", tts_metrics("speech-1"))
    await rec.finish()

    tts = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "tts"][0]
    assert tts["response"]["audio_bytes"] == 10 * 480


# --------------------------------------------- reviewer: the finish upload budget


async def test_finish_bounds_the_upload_by_default(recorder, monkeypatch):
    """`timeout=None` used to mean "wait forever" inside a 10s shutdown hook."""
    rec = recorder()
    seen: dict = {}

    class FakeObserver:
        async def upload_package(self, finalized, timeout=None):
            seen["timeout"] = timeout
            return {"session_id": finalized.session_id, "status": "ready"}

    rec._observer = FakeObserver()
    rec._upload = True
    session = FakeAgentSession()
    rec.attach(session)
    await run_turn(session)
    await rec.finish()

    assert seen["timeout"] == pytest.approx(60.0)


async def test_an_explicit_zero_timeout_still_means_no_budget(recorder):
    rec = recorder()
    seen: dict = {}

    class FakeObserver:
        async def upload_package(self, finalized, timeout=None):
            seen["timeout"] = timeout
            return {"session_id": finalized.session_id, "status": "ready"}

    rec._observer = FakeObserver()
    rec._upload = True
    session = FakeAgentSession()
    rec.attach(session)
    await run_turn(session)
    await rec.finish(timeout=0)

    assert seen["timeout"] is None


async def test_a_package_uploaded_in_process_is_not_shipped_again_by_the_drain(recorder):
    """Without a receipt a drain sidecar re-uploads every package on every tick."""
    import os

    from vaani_observer.drain import RECEIPT_NAME, pending_packages

    rec = recorder()

    class FakeObserver:
        async def upload_package(self, finalized, timeout=None):
            return {"session_id": finalized.session_id, "status": "ready"}

    spool = rec._observer.options["spool_directory"]
    rec._observer = FakeObserver()
    rec._upload = True
    session = FakeAgentSession()
    rec.attach(session)
    await run_turn(session)
    await rec.finish()

    assert os.path.exists(os.path.join(rec._finalized.directory, RECEIPT_NAME))
    assert pending_packages(spool) == []


# ------------------------------------ reviewer: a raising plugin property is fatal


async def test_a_plugin_whose_model_property_raises_does_not_kill_the_recording(recorder):
    """Sniffing runs at attach; an exception there used to abort the whole call."""

    class ExplodingLLM:
        # `_sniff_model` reaches for the Azure client here. Third-party plugins
        # are free to make that a property that raises.
        @property
        def _client(self):
            raise RuntimeError("provider not configured")

    rec = recorder()
    session = FakeAgentSession()
    session.llm = ExplodingLLM()
    rec.attach(session)  # must not raise
    await run_turn(session)
    await rec.finish()

    assert rec._finalized.manifest["duration_ms"] is not None


# ------------------------ reviewer: an unwired audio tap is indistinguishable from OK


async def test_an_unwired_audio_tap_says_so_the_first_time_audio_flows(caplog):
    """`agent.vaani = None` records a valid package with an empty `call.audio`.

    Nothing about the result looks wrong -- spans are present, the manifest is
    valid -- so the warning has to happen where the frames are dropped.
    """
    from vaani_observer._diagnostics import reset_warnings
    from vaani_observer.integrations.livekit import VaaniAudioTapMixin

    reset_warnings()

    class BaseAgent:
        async def tts_node(self, text, model_settings):
            yield FakeFrame(b"\x01\x00" * 240)

    class Agent(VaaniAudioTapMixin, BaseAgent):
        pass

    agent = Agent()
    assert agent.vaani is None

    with caplog.at_level(logging.WARNING, logger="vaani_observer"):
        async for _ in agent.tts_node("hi", None):
            pass

    assert any(
        record.levelno >= logging.WARNING and "NO audio is being captured" in record.getMessage()
        for record in caplog.records
    ), "an unwired tap must be reported"


async def test_the_unwired_warning_does_not_repeat_per_frame(caplog):
    from vaani_observer._diagnostics import reset_warnings
    from vaani_observer.integrations.livekit import VaaniAudioTapMixin

    reset_warnings()

    class BaseAgent:
        async def tts_node(self, text, model_settings):
            for _ in range(50):
                yield FakeFrame(b"\x01\x00" * 240)

    class Agent(VaaniAudioTapMixin, BaseAgent):
        pass

    with caplog.at_level(logging.WARNING, logger="vaani_observer"):
        for _ in range(3):
            async for _frame in Agent().tts_node("hi", None):
                pass

    warnings = [
        record for record in caplog.records
        if record.levelno >= logging.WARNING and "NO audio is being captured" in record.getMessage()
    ]
    assert len(warnings) == 1


# ---------------- e2e: synthesized duration and delivered audio are not the same number


async def test_an_interrupted_reply_reports_what_the_caller_actually_heard(recorder):
    """`audio_ms` is what the provider synthesized; the tape is what was played.

    On a barge-in these legitimately differ, and the gap is the useful part --
    it is how much of the answer the caller never heard. Measured live: a span
    reporting 23200ms with 5570ms of taped audio.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)

    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_input_transcribed", transcript("hello", True))
    session.emit("user_state_changed", SimpleNamespace(new_state="listening"))
    session.emit(
        "speech_created",
        SimpleNamespace(speech_handle=SimpleNamespace(id="speech-1"), source="generate_reply"),
    )
    # A long reply, cut off after a quarter of a second of audio.
    for _ in range(25):
        rec.tap_output_frame(FakeFrame(b"\x01\x00" * 240, sample_rate=24000))
    session.emit(
        "metrics_collected",
        tts_metrics("speech-1", audio_duration=8.0, cancelled=True),
    )
    await rec.finish()

    tts = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "tts"][0]
    assert tts["status"] == "cancelled"
    # The provider's number survives untouched...
    assert tts["response"]["audio_ms"] == 8000
    # ...and what the caller actually heard is reported separately.
    assert tts["response"]["played_ms"] == 250
    assert tts["response"]["audio_bytes"] == 25 * 480


async def test_played_ms_is_absent_rather_than_zero_when_no_audio_was_taped(recorder):
    """Zero would read as "the caller heard silence"; absent reads as "unknown".

    Guards new behaviour rather than proving a past bug: pre-fix there was no
    `played_ms` at all, so its absence held trivially.
    """
    rec = recorder()
    session = FakeAgentSession()
    rec.attach(session)
    await run_turn(session)  # spans, but no audio tapped at all
    await rec.finish()

    tts = [event for event in operations(read_events(rec._finalized.directory))
           if event["type"] == "tts"][0]
    assert "played_ms" not in tts["response"]
    assert tts["response"]["audio_ms"] == 2000  # the provider's, unchanged


async def test_the_usage_rollup_reports_the_real_model_not_the_plugin_default(recorder):
    """Spans were corrected; the session usage rollup was not.

    Found on a live Azure call: every LLM span said `gpt-5-mini` while
    `metadata.usage.model_usage[0].model` still said `gpt-4o`, which is what
    anyone aggregating cost or tokens by model actually reads.
    """
    rec = recorder(model_overrides={"llm": "gpt-5-mini"})
    session = FakeAgentSession()
    rec.attach(session)
    session.emit("session_usage_updated", SimpleNamespace(usage={
        "model_usage": [
            {"type": "llm_usage", "model": "gpt-4o", "input_tokens": 562},
            {"type": "tts_usage", "model": "aura-2-thalia-en", "characters_count": 694},
        ]
    }))
    await rec.finish()

    usage = rec._finalized.manifest["metadata"]["usage"]["model_usage"]
    assert usage[0]["model"] == "gpt-5-mini"
    assert usage[0]["reported_model"] == "gpt-4o", "the reported value is evidence"
    assert usage[0]["input_tokens"] == 562, "the numbers must be untouched"
    # Nothing was resolved for TTS, so it must be left exactly as reported.
    assert usage[1]["model"] == "aura-2-thalia-en"
    assert "reported_model" not in usage[1]


async def test_the_in_process_receipt_records_which_backend_received_the_call(recorder):
    """Receipts written by finish() are the ones that exist in production.

    The drain only ever sees leftovers, so scoping receipts to their endpoint
    is inert unless this path records it too -- a spool that moved backends
    would still look fully delivered to the new one.
    """
    import json as _json
    import os

    from vaani_observer.drain import RECEIPT_NAME

    rec = recorder()

    class FakeObserver:
        options = {"endpoint": "https://prod.example.com"}

        async def upload_package(self, finalized, timeout=None):
            return {"session_id": finalized.session_id, "status": "ready"}

    rec._observer = FakeObserver()
    rec._upload = True
    session = FakeAgentSession()
    rec.attach(session)
    await rec.finish()

    path = os.path.join(rec._finalized.directory, RECEIPT_NAME)
    assert os.path.exists(path), "finish() must leave a receipt"
    assert _json.loads(open(path).read())["endpoint"] == "https://prod.example.com"


async def test_receipt_bookkeeping_can_never_fail_a_successful_upload(recorder):
    """An observer without `options` must not turn a delivered call into a loss."""
    rec = recorder()

    class BareObserver:
        async def upload_package(self, finalized, timeout=None):
            return {"session_id": finalized.session_id, "status": "ready"}

    rec._observer = BareObserver()
    rec._upload = True
    session = FakeAgentSession()
    rec.attach(session)
    await rec.finish()

    assert rec._finalized is not None
