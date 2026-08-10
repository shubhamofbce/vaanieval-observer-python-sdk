"""Session creation, manifest publication and finalization.

Mirrors nodejs-sdk/test/session-lifecycle.test.js.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid

import pytest

from conftest import PCM, read_events, read_manifest
from vaani_observer import VaaniObserver

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def settle() -> None:
    await asyncio.sleep(0)


async def test_generates_a_session_id_and_directory_when_none_is_supplied(new_observer):
    vaani = new_observer()
    session = vaani.start_session()
    assert UUID_RE.match(session.id)
    assert session.directory == os.path.join(vaani.options["spool_directory"], session.id)
    assert session.agent_id is None
    assert session.metadata == {}
    assert session.started_at.endswith("Z")
    await session.end()


async def test_creates_the_spool_directory_eagerly(new_observer):
    session = new_observer().start_session()
    await session.ready()
    assert os.path.isdir(session.directory)
    await session.end()


async def test_writes_a_complete_manifest_for_a_finalized_session(new_observer):
    vaani = new_observer()
    session = vaani.start_session(
        session_id="call-1", agent_id="support", metadata={"env": "test", "region": "in"}
    )
    result = await session.end(outcome="completed")
    assert (result.session_id, result.directory) == ("call-1", session.directory)
    manifest = read_manifest(result.directory)
    assert manifest["schema_version"] == "1.0"
    assert manifest["sdk"] == {"name": "@vaanieal/observer", "language": "python", "version": "0.1.0"}
    assert manifest["session_id"] == "call-1"
    assert manifest["agent_id"] == "support"
    assert manifest["metadata"] == {"env": "test", "region": "in"}
    assert manifest["outcome"] == "completed"
    assert manifest["duration_ms"] >= 0
    assert manifest["audio"] == {}


async def test_defaults_the_outcome_to_unknown(new_observer):
    finalized = await new_observer().start_session().end()
    assert read_manifest(finalized.directory)["outcome"] == "unknown"


async def test_reports_instrumentation_state_through_capture_status(new_observer):
    enabled = new_observer(instrumentations={"http": False, "websocket": True})
    disabled = new_observer(instrumentations={"http": False, "websocket": False})
    first = await enabled.start_session().end()
    second = await disabled.start_session().end()
    assert first.manifest["capture_status"] == {
        "events_complete": True,
        "audio_complete": True,
        "http_instrumentation": "disabled",
        "websocket_instrumentation": "active",
        "dropped_event_count": 0,
        "dropped_audio_chunk_count": 0,
    }
    assert second.manifest["capture_status"]["websocket_instrumentation"] == "disabled"


async def test_publishes_the_manifest_atomically_and_leaves_no_temporary_file(new_observer):
    session = new_observer().start_session()
    session.record_inbound_audio(b"\x01", PCM)
    finalized = await session.end()
    assert sorted(os.listdir(finalized.directory)) == [
        "call.audio",
        "events.jsonl",
        "manifest.json",
    ]


async def test_is_idempotent_a_second_end_resolves_to_the_first_package(new_observer):
    session = new_observer().start_session()
    first = await session.end(outcome="completed")
    second = await session.end(outcome="failed")
    assert second is first
    assert read_manifest(first.directory)["outcome"] == "completed"


async def test_finished_resolves_with_the_same_package_that_end_returns(new_observer):
    session = new_observer().start_session()
    result = await session.end()
    assert await session.finished is result


async def test_drops_every_kind_of_record_written_after_the_session_ended(new_observer):
    session = new_observer().start_session()
    operation = session.start_operation(type="llm")
    finalized = await session.end()
    operation.end(status="ok")
    session.record_websocket_event(phase="close")
    session.record_inbound_audio(b"\x01", PCM)
    await settle()
    assert read_events(finalized.directory) == []


async def test_flush_waits_for_every_in_flight_session(new_observer):
    vaani = new_observer()
    first = vaani.start_session()
    second = vaani.start_session()
    flushed = False

    async def flush() -> None:
        nonlocal flushed
        await vaani.flush()
        flushed = True

    task = asyncio.ensure_future(flush())
    await settle()
    assert flushed is False
    await first.end()
    await second.end()
    await task
    assert flushed is True
    await vaani.flush()


async def test_flush_resolves_when_no_session_was_ever_started(new_observer):
    await new_observer().flush()


async def test_degrades_capture_status_instead_of_raising_when_a_write_fails(new_observer):
    session = new_observer().start_session()
    await session.ready()
    # A directory where the audio track file belongs makes the append fail.
    os.mkdir(os.path.join(session.directory, ".caller.audio.tmp"))
    session.record_inbound_audio(b"\x01", PCM)
    finalized = await session.end()
    status = read_manifest(finalized.directory)["capture_status"]
    assert status["audio_complete"] is False
    assert status["dropped_audio_chunk_count"] >= 1


async def test_degrades_events_complete_when_the_event_log_write_fails(new_observer):
    session = new_observer().start_session()
    await session.ready()
    os.mkdir(os.path.join(session.directory, "events.jsonl"))
    session.start_operation(type="llm").end()
    finalized = await session.end()
    status = read_manifest(finalized.directory)["capture_status"]
    assert status["events_complete"] is False
    assert status["dropped_event_count"] >= 1
    assert status["audio_complete"] is True


async def test_drops_and_counts_writes_once_the_queue_is_saturated():
    """A stalled disk must cost bounded memory, not the whole agent process."""
    import tempfile
    import threading as _threading

    from vaani_observer._writer import SpoolWriter

    errors: list = []
    drops: list = []
    release = _threading.Event()
    with tempfile.TemporaryDirectory() as directory:
        writer = SpoolWriter(
            os.path.join(directory, "s"),
            on_error=lambda name, error: errors.append(name),
            on_drop=drops.append,
            max_queued_writes=2,
        )
        writer.wait_ready(1.0)
        original_append = writer._append
        writer._append = lambda *a: (release.wait(2.0), original_append(*a))  # type: ignore
        accepted = [writer.submit(".caller.audio.tmp", b"\x01") for _ in range(64)]
        release.set()
        writer.close(timeout=5.0)

    assert False in accepted, "a bounded queue must refuse writes when saturated"
    assert len(drops) == accepted.count(False)
    assert all(name == ".caller.audio.tmp" for name in drops)
    assert not errors


async def test_propagates_write_failures_from_end_in_strict_mode(new_observer):
    session = new_observer(strict=True).start_session()
    await session.ready()
    os.mkdir(os.path.join(session.directory, ".caller.audio.tmp"))
    session.record_inbound_audio(b"\x01", PCM)
    with pytest.raises(OSError):
        await session.end()
    with pytest.raises(OSError):
        await session.finished
    assert not os.path.exists(os.path.join(session.directory, "manifest.json"))


async def test_fails_the_session_when_its_spool_directory_cannot_be_created(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    vaani = VaaniObserver(spool_directory=str(blocker), instrumentations={"http": False})
    session = vaani.start_session()
    session.record_inbound_audio(b"\x01", PCM)
    with pytest.raises((NotADirectoryError, FileExistsError)):
        await session.end()
    assert session.capture_status["events_complete"] is False


async def test_keeps_concurrent_sessions_isolated_on_disk(new_observer):
    vaani = new_observer()
    first = vaani.start_session(session_id="a")
    second = vaani.start_session(session_id="b")
    first.record_inbound_audio(b"\x01", PCM)
    second.record_inbound_audio(b"\x02\x02", PCM)
    one, two = await asyncio.gather(first.end(), second.end())
    assert one.directory != two.directory
    assert read_events(one.directory)[0]["byte_length"] == 1
    assert read_events(two.directory)[0]["byte_length"] == 2


async def test_preserves_event_ordering_under_interleaved_writes(new_observer):
    session = new_observer().start_session()
    for index in range(25):
        session.record_inbound_audio(bytes(index + 1), PCM)
    finalized = await session.end()
    events = read_events(finalized.directory)
    assert [event["byte_length"] for event in events] == list(range(1, 26))


async def test_uses_a_supplied_session_id_verbatim(new_observer):
    session = new_observer().start_session(session_id="call 1+a")
    finalized = await session.end()
    assert finalized.session_id == "call 1+a"
    assert os.path.basename(finalized.directory) == "call 1+a"


async def test_the_session_clock_is_monotonic_and_starts_at_zero(new_observer):
    session = new_observer().start_session()
    first = session.now()
    await asyncio.sleep(0.02)
    second = session.now()
    assert 0 <= first <= 5
    assert second >= first + 15
    await session.end()


async def test_finished_is_awaitable_from_a_different_event_loop(new_observer):
    """A session may outlive the loop it was created on (worker restarts, tests)."""
    session = new_observer().start_session()
    finalized = await session.end()

    def on_another_loop():
        return asyncio.run(_await_finished(session))

    assert (await asyncio.to_thread(on_another_loop)).directory == finalized.directory


async def _await_finished(session):
    return await session.finished


async def test_finished_can_be_awaited_more_than_once(new_observer):
    session = new_observer().start_session()
    first = await session.end()
    assert (await session.finished).session_id == first.session_id
    assert (await session.finished).session_id == first.session_id


async def test_flush_waits_for_a_session_that_is_still_finalizing(new_observer):
    observer = new_observer()
    session = observer.start_session()
    ending = asyncio.create_task(session.end())
    await asyncio.sleep(0)
    await observer.flush()
    assert session._completion.done()
    await ending
