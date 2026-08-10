"""Single stereo call recording: agent left, caller right."""

from __future__ import annotations

import os
import struct

import pytest

from conftest import PCM, read_events, read_manifest, read_track


async def test_writes_one_timeline_aligned_stereo_recording(new_observer):
    session = new_observer().start_session()
    session.record_inbound_audio(struct.pack("<2h", 1000, 2000), {**PCM, "timestamp_ms": 0})
    session.record_outbound_audio(struct.pack("<2h", 3000, 4000), {**PCM, "timestamp_ms": 0})
    finalized = await session.end()

    samples = struct.unpack("<4h", read_track(finalized.directory, "call")[:8])
    assert samples == (3000, 1000, 4000, 2000)
    assert sorted(os.listdir(finalized.directory)) == ["call.audio", "events.jsonl", "manifest.json"]
    assert finalized.manifest["audio"] == {
        "call": {
            "file": "call.audio",
            "encoding": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 2,
            "channel_layout": {"left": "agent", "right": "caller"},
        }
    }


async def test_uses_silence_for_a_missing_side(new_observer):
    session = new_observer().start_session()
    session.record_inbound_audio(struct.pack("<h", 1234), {**PCM, "timestamp_ms": 0})
    finalized = await session.end()
    assert struct.unpack("<2h", read_track(finalized.directory, "call")[:4]) == (0, 1234)


async def test_resamples_both_sides_to_the_higher_source_rate(new_observer):
    session = new_observer().start_session()
    session.record_inbound_audio(
        struct.pack("<2h", 1000, 2000),
        {"encoding": "pcm_s16le", "sample_rate_hz": 8000, "channels": 1, "timestamp_ms": 0},
    )
    session.record_outbound_audio(
        struct.pack("<2h", 3000, 4000),
        {"encoding": "pcm_s16le", "sample_rate_hz": 24000, "channels": 1, "timestamp_ms": 0},
    )
    finalized = await session.end()
    assert finalized.manifest["audio"]["call"]["sample_rate_hz"] == 24000
    samples = struct.unpack("<8h", read_track(finalized.directory, "call")[:16])
    assert samples[::2] == (3000, 4000, 0, 0)
    assert samples[1::2] == (1000, 1333, 1667, 2000)


@pytest.mark.parametrize("chunk", [b"\x07\x08", bytearray(b"\x07\x08"), memoryview(b"\x07\x08")])
async def test_accepts_every_byte_container(new_observer, chunk):
    session = new_observer().start_session()
    assert session.record_inbound_audio(chunk, {**PCM, "timestamp_ms": 0}) is True
    finalized = await session.end()
    assert read_track(finalized.directory, "call")[2:4] == b"\x07\x08"


@pytest.mark.parametrize("chunk", ["abc", 42, None, {}, [1, 2]])
async def test_rejects_chunk_types_that_are_not_byte_containers(new_observer, chunk):
    session = new_observer().start_session()
    with pytest.raises(TypeError):
        session.record_inbound_audio(chunk, PCM)
    await session.end()


@pytest.mark.parametrize(
    "fmt,error",
    [
        ({"encoding": "opus", "sample_rate_hz": 48000, "channels": 1}, "pcm_s16le"),
        ({"encoding": "pcm_s16le", "channels": 1}, "sample_rate_hz"),
        ({"encoding": "pcm_s16le", "sample_rate_hz": 16000}, "channel"),
    ],
)
async def test_requires_playable_pcm_metadata(new_observer, fmt, error):
    session = new_observer().start_session()
    with pytest.raises(ValueError, match=error):
        session.record_inbound_audio(b"\x00\x00", fmt)
    await session.end()


async def test_emits_audio_chunk_events_with_source_identity(new_observer):
    session = new_observer().start_session()
    session.record_inbound_audio(b"\x01\x02", {**PCM, "timestamp_ms": 1234})
    finalized = await session.end()
    event = read_events(finalized.directory)[0]
    assert event == {
        "kind": "audio_chunk",
        "track": "caller",
        "occurred_at_ms": 1234,
        "byte_length": 2,
        "duration_ms": 0.0625,
    }


async def test_keeps_outbound_pcm_on_a_real_time_playback_clock(new_observer):
    session = new_observer().start_session()
    frame = bytes(320)
    session.record_outbound_audio(frame, {**PCM, "timestamp_ms": 50})
    session.record_outbound_audio(frame, {**PCM, "timestamp_ms": 51})
    finalized = await session.end()
    events = [event for event in read_events(finalized.directory) if event.get("track") == "agent"]
    assert [event["occurred_at_ms"] for event in events] == [50, 60]


async def test_does_not_advance_the_caller_clock_by_pcm_duration(new_observer):
    session = new_observer().start_session()
    frame = bytes(320)
    session.record_inbound_audio(frame, {**PCM, "timestamp_ms": 50})
    session.record_inbound_audio(frame, {**PCM, "timestamp_ms": 51})
    finalized = await session.end()
    events = [event for event in read_events(finalized.directory) if event.get("track") == "caller"]
    assert [event["occurred_at_ms"] for event in events] == [50, 51]


async def test_locks_the_audio_format_per_source(new_observer):
    session = new_observer().start_session()
    session.record_outbound_audio(b"\x00\x00", PCM)
    with pytest.raises(ValueError, match="agent audio format cannot change"):
        session.record_outbound_audio(b"\x00\x00\x00\x00", {**PCM, "channels": 2})
    await session.end()


async def test_drops_audio_entirely_when_capture_is_disabled(new_observer):
    session = new_observer(capture={"audio": False}).start_session()
    assert session.record_inbound_audio("not-bytes", {}) is False
    finalized = await session.end()
    assert finalized.manifest["audio"] == {}
    assert not os.path.exists(os.path.join(finalized.directory, "call.audio"))
    assert read_events(finalized.directory) == []


async def test_stops_accepting_audio_after_the_session_ends(new_observer):
    session = new_observer().start_session()
    finalized = await session.end()
    assert session.record_inbound_audio(b"\x00\x00", PCM) is False
    assert not os.path.exists(os.path.join(finalized.directory, "call.audio"))
    assert read_manifest(finalized.directory)["audio"] == {}
