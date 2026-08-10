"""Operations, turns, milestones, samples and ambient context.

Mirrors nodejs-sdk/test/session-operations.test.js.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from conftest import operations, read_events

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def test_fills_operation_defaults_for_a_minimally_specified_operation(new_observer):
    session = new_observer().start_session(session_id="call-1")
    session.start_operation(type="llm").end()
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["session_id"] == "call-1"
    assert event["turn_id"] is None
    assert event["scope"] == "turn"
    assert event["endpoint_id"] is None
    assert event["provider"] is None
    assert event["model"] is None
    assert event["transport"] == "manual"
    assert event["status"] == "ok"
    assert event["request"] == {}
    assert event["response"] == {}
    assert event["milestones"] == {}
    assert event["error"] is None
    assert UUID_RE.match(event["event_id"])


async def test_carries_provider_model_transport_and_request_metadata(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(
        type="tts",
        endpoint_id="tts-main",
        provider="acme",
        model="voice-1",
        transport="websocket",
        request={"voice": "ana"},
    )
    operation.end(status="ok", response={"audio_bytes": 42})
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["endpoint_id"] == "tts-main"
    assert event["provider"] == "acme"
    assert event["model"] == "voice-1"
    assert event["transport"] == "websocket"
    assert event["request"] == {"voice": "ana"}
    assert event["response"] == {"audio_bytes": 42}


async def test_accepts_tool_operations_and_rejects_unsupported_types(new_observer):
    session = new_observer().start_session()
    tool = session.start_operation(type="tool", request={"name": "search", "input": {"city": "Pune"}})
    tool.end(response={"result": "ok"})
    for bad in ["http", "LLM", None, ""]:
        with pytest.raises(TypeError):
            session.start_operation(type=bad)
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["type"] == "tool"


async def test_never_writes_an_event_for_an_operation_that_was_never_ended(new_observer):
    session = new_observer().start_session()
    session.start_operation(type="stt").event("partial")
    finalized = await session.end()
    assert operations(read_events(finalized.directory)) == []


async def test_records_milestones_with_a_timestamp_and_merges_supplied_data(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    operation.event("first_token")
    operation.event("chunk", {"index": 1, "text_available": True})
    operation.end()
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert isinstance(event["milestones"]["first_token"]["occurred_at_ms"], int)
    assert event["milestones"]["chunk"]["index"] == 1
    assert event["milestones"]["chunk"]["text_available"] is True


async def test_accepts_milestone_data_as_keyword_arguments(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    operation.event("chunk", index=3)
    operation.end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["milestones"]["chunk"]["index"] == 3


async def test_lets_a_milestone_payload_override_the_recorded_timestamp(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    operation.event("first_token", {"occurred_at_ms": 7})
    operation.end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["milestones"]["first_token"]["occurred_at_ms"] == 7


async def test_repeated_milestones_accumulate_first_last_and_count(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    operation.event("chunk", {"index": 1, "occurred_at_ms": 10})
    operation.event("chunk", {"index": 2, "occurred_at_ms": 40})
    operation.end()
    finalized = await session.end()
    milestone = operations(read_events(finalized.directory))[0]["milestones"]["chunk"]
    assert milestone == {"occurred_at_ms": 10, "last_at_ms": 40, "count": 2, "index": 2}


async def test_records_bounded_operation_samples_independently_from_milestones(new_observer):
    session = new_observer(capture={"payload_max_bytes": 64}).start_session()
    operation = session.start_operation(type="stt")
    operation.sample("partial", {"occurred_at_ms": 10, "transcript": "hello"}, limit=2)
    operation.sample("partial", {"occurred_at_ms": 20, "transcript": "hello there"}, limit=2)
    operation.sample("partial", {"occurred_at_ms": 30, "transcript": "ignored"}, limit=2)
    operation.end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["samples"]["partial"] == {
        "items": [
            {"occurred_at_ms": 10, "transcript": "hello"},
            {"occurred_at_ms": 20, "transcript": "hello there"},
        ],
        "truncated": True,
    }


async def test_ignores_milestones_recorded_after_the_operation_ended(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    operation.end()
    operation.event("too_late")
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["milestones"] == {}


async def test_ignores_a_second_end_so_an_operation_is_written_exactly_once(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    operation.end(status="ok")
    operation.end(status="error", error={"name": "Error", "message": "late"})
    finalized = await session.end()
    written = operations(read_events(finalized.directory))
    assert len(written) == 1
    assert written[0]["status"] == "ok"
    assert written[0]["error"] is None


async def test_records_a_non_negative_duration_from_the_session_clock(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="stt")
    await asyncio.sleep(0.012)
    operation.end()
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["started_at_ms"] >= 0
    assert event["duration_ms"] == event["ended_at_ms"] - event["started_at_ms"]
    assert event["duration_ms"] >= 10


async def test_back_dates_a_span_whose_start_is_only_known_in_hindsight(new_observer):
    session = new_observer().start_session()
    await asyncio.sleep(0.02)
    session.start_operation(type="stt", started_at_ms=3).end(ended_at_ms=25)
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert (event["started_at_ms"], event["ended_at_ms"], event["duration_ms"]) == (3, 25, 22)


async def test_clamps_a_negative_duration_to_zero(new_observer):
    session = new_observer().start_session()
    session.start_operation(type="stt", started_at_ms=100).end(ended_at_ms=40)
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["duration_ms"] == 0


async def test_preserves_an_error_result_verbatim(new_observer):
    session = new_observer().start_session()
    session.start_operation(type="llm").end(
        status="error", error={"name": "TypeError", "message": "boom"}
    )
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["status"] == "error"
    assert event["error"] == {"name": "TypeError", "message": "boom"}


async def test_returns_an_inert_operation_once_the_session_has_ended(new_observer):
    session = new_observer().start_session()
    finalized = await session.end()
    operation = session.start_operation(type="not-a-valid-type")
    operation.event("x")
    operation.sample("y", {"a": 1})
    operation.set_turn("t")
    operation.set_request({"a": 1})
    operation.end(status="ok")
    assert operation.ended is True
    assert operations(read_events(finalized.directory)) == []


async def test_bounds_an_oversized_request_and_response_payload(new_observer):
    session = new_observer(capture={"payload_max_bytes": 32}).start_session()
    session.start_operation(type="llm", request={"prompt": "x" * 500}).end(
        response={"text": "y" * 500}
    )
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    for payload in (event["request"], event["response"]):
        assert payload["_truncated"] is True
        assert payload["_original_bytes"] > 32
        assert len(payload["_preview"]) <= 32


async def test_replaces_an_unserializable_payload_with_a_capture_error(new_observer):
    session = new_observer().start_session()
    session.start_operation(type="llm", request={"socket": object()}).end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["request"] == {
        "_capture_error": "Payload is not JSON serializable."
    }


async def test_set_request_replaces_the_request_payload(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm", request={"a": 1})
    operation.set_request({"b": 2})
    operation.end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["request"] == {"b": 2}


# ------------------------------------------------------------------- turns


async def test_correlates_operations_started_through_a_turn(new_observer):
    session = new_observer().start_session()
    turn = session.start_turn("turn-7")
    turn.start_operation(type="stt").end()
    turn.start_operation(type="llm", turn_id="ignored-override").end()
    turn.end()
    finalized = await session.end()
    assert [e["turn_id"] for e in operations(read_events(finalized.directory))] == ["turn-7", "turn-7"]


async def test_generates_a_turn_id_when_one_is_not_supplied(new_observer):
    session = new_observer().start_session()
    first = session.start_turn()
    second = session.start_turn()
    assert UUID_RE.match(first.id)
    assert first.id != second.id
    await session.end()


async def test_marks_a_turn_ended_explicitly_and_again_when_the_session_ends(new_observer):
    session = new_observer().start_session()
    closed = session.start_turn()
    still_open = session.start_turn()
    closed.end()
    assert closed.ended is True
    assert still_open.ended is False
    await session.end()
    assert still_open.ended is True


async def test_still_records_operations_started_from_a_turn_that_has_ended(new_observer):
    session = new_observer().start_session()
    turn = session.start_turn("turn-1")
    turn.end()
    turn.start_operation(type="llm").end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["turn_id"] == "turn-1"


async def test_coerces_a_non_string_turn_id(new_observer):
    session = new_observer().start_session()
    session.start_turn(7).start_operation(type="llm").end()
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["turn_id"] == "7"


# -------------------------------------------------------------- ws events


async def test_records_neutral_websocket_lifecycle_events(new_observer):
    session = new_observer().start_session(session_id="call-9")
    session.record_websocket_event(phase="open", endpoint_id="stt", byte_count=10)
    finalized = await session.end()
    event = read_events(finalized.directory)[0]
    assert event["kind"] == "websocket"
    assert event["session_id"] == "call-9"
    assert event["phase"] == "open"
    assert event["byte_count"] == 10
    assert isinstance(event["occurred_at_ms"], int)


async def test_lets_a_websocket_event_payload_override_the_generated_timestamp(new_observer):
    session = new_observer().start_session()
    session.record_websocket_event(phase="close", occurred_at_ms=99)
    finalized = await session.end()
    assert read_events(finalized.directory)[0]["occurred_at_ms"] == 99


async def test_drops_websocket_events_recorded_after_the_session_ended(new_observer):
    session = new_observer().start_session()
    finalized = await session.end()
    session.record_websocket_event(phase="close")
    assert read_events(finalized.directory) == []


# ------------------------------------------------------------------ context


async def test_bind_runs_a_handler_inside_the_session_context(new_observer):
    vaani = new_observer()
    session = vaani.start_session()

    def handler(a, b):
        assert vaani.current_context().session is session
        return a + b

    assert session.bind(handler)(2, 3) == 5
    assert vaani.current_context() is None
    await session.end()


async def test_bind_runs_an_async_handler_inside_the_session_context(new_observer):
    vaani = new_observer()
    session = vaani.start_session()

    async def handler():
        await asyncio.sleep(0)
        return vaani.current_context().session

    assert await session.bind(handler)() is session
    assert vaani.current_context() is None
    await session.end()


async def test_scopes_an_endpoint_id_onto_the_active_context_and_validates_it(new_observer):
    vaani = new_observer(endpoints=[{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    session = vaani.start_session()
    with pytest.raises(ValueError, match="Unknown endpoint: missing"):
        session.with_endpoint("missing")
    with session.with_endpoint("llm"):
        assert vaani.current_context().endpoint_id == "llm"
    assert vaani.current_context() is None
    await session.end()


async def test_a_nested_scope_inherits_the_enclosing_endpoint_id(new_observer):
    vaani = new_observer(endpoints=[{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    session = vaani.start_session()
    with session.with_endpoint("llm"):
        with session.context():
            assert vaani.current_context().endpoint_id == "llm"
            assert vaani.current_context().session is session
    await session.end()


async def test_a_turn_scope_keeps_the_enclosing_endpoint_id(new_observer):
    vaani = new_observer(endpoints=[{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    session = vaani.start_session()
    with session.with_endpoint("llm"):
        with session.with_turn("turn-3"):
            context = vaani.current_context()
            assert (context.endpoint_id, context.turn_id) == ("llm", "turn-3")
    await session.end()


async def test_a_turn_context_manager_tags_ambient_operations(new_observer):
    session = new_observer().start_session()
    turn = session.start_turn("turn-9")
    with turn.context():
        assert session._observer.current_context().turn_id == "turn-9"
    await session.end()


async def test_the_context_does_not_leak_between_concurrent_tasks(new_observer):
    vaani = new_observer()
    first = vaani.start_session(session_id="a")
    second = vaani.start_session(session_id="b")
    seen = {}

    async def work(session, key):
        with session.context():
            await asyncio.sleep(0.01)
            seen[key] = vaani.current_context().session.id

    await asyncio.gather(work(first, "a"), work(second, "b"))
    assert seen == {"a": "a", "b": "b"}
    await asyncio.gather(first.end(), second.end())
