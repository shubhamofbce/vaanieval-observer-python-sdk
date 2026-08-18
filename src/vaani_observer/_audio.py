"""Finalize temporary PCM tracks into one timeline-aligned stereo recording."""

from __future__ import annotations

import json
import os
import sys
from array import array
from typing import Any, Mapping

TEMP_TRACK_FILES = {
    "caller": ".caller.audio.tmp",
    "agent": ".agent.audio.tmp",
}
OUTPUT_FILE = "call.audio"


def _fit(samples: "array[int]", frames: int) -> "array[int]":
    """`samples` as exactly `frames` entries: truncated, or zero-padded.

    Extended-slice assignment on an `array` requires an exact length match, so
    a track that ran short (the caller stopped talking first) has to be padded
    rather than checked per index.
    """
    if len(samples) == frames:
        return samples
    if len(samples) > frames:
        return samples[:frames]
    padded = array("h", samples)
    padded.frombytes(b"\x00" * (2 * (frames - len(samples))))
    return padded


def compose_stereo(
    directory: str,
    tracks: Mapping[str, Mapping[str, Any]],
    duration_ms: int,
) -> dict[str, Any] | None:
    """Write agent-left/caller-right s16le PCM and remove temporary mono tracks."""
    if not tracks:
        return None

    for track, fmt in tracks.items():
        if fmt.get("encoding") != "pcm_s16le":
            raise ValueError(f"{track} audio must use pcm_s16le for stereo finalization.")
        rate = fmt.get("sample_rate_hz")
        channels = fmt.get("channels")
        if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
            raise ValueError(f"{track} audio is missing a valid sample rate.")
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError(f"{track} audio is missing a valid channel count.")

    output_rate = max(fmt["sample_rate_hz"] for fmt in tracks.values())
    events = _audio_events(directory)
    rendered = {
        track: _render_track(directory, track, fmt, events.get(track, []), output_rate)
        for track, fmt in tracks.items()
    }
    minimum_frames = round(max(0, duration_ms) * output_rate / 1000)
    frames = max(minimum_frames, *(len(samples) for samples in rendered.values()))
    caller = rendered.get("caller", array("h"))
    agent = rendered.get("agent", array("h"))
    # Interleaving one sample at a time is ~7s of pure Python on a maximum
    # length call, spent inside a shutdown window that is ten seconds wide by
    # default -- so the recording could be killed before the upload it is
    # blocking had begun. Extended-slice assignment does the identical work in
    # C, ~57x faster.
    # `array("h").frombytes(b"\x00" * frames * 4)` allocates a full-size `bytes`
    # temporary alongside the array: measured 273 MB peak versus 132 MB on a
    # 23-minute call, inside a shutdown window. Repetition builds the same
    # zeroed buffer in one allocation, and is faster besides.
    stereo = array("h", [0]) * (frames * 2)
    stereo[0::2] = _fit(agent, frames)
    stereo[1::2] = _fit(caller, frames)
    if sys.byteorder != "little":
        stereo.byteswap()
    with open(os.path.join(directory, OUTPUT_FILE), "wb") as handle:
        stereo.tofile(handle)
    for filename in TEMP_TRACK_FILES.values():
        try:
            os.remove(os.path.join(directory, filename))
        except FileNotFoundError:
            pass
        except (IsADirectoryError, PermissionError):
            path = os.path.join(directory, filename)
            if not os.path.isdir(path):
                raise
            os.rmdir(path)
    return {
        "file": OUTPUT_FILE,
        "encoding": "pcm_s16le",
        "sample_rate_hz": output_rate,
        "channels": 2,
        "channel_layout": {"left": "agent", "right": "caller"},
    }


def _audio_events(directory: str) -> dict[str, list[tuple[int, int]]]:
    path = os.path.join(directory, "events.jsonl")
    result: dict[str, list[tuple[int, int]]] = {}
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return result
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            track = event.get("track")
            occurred_at = event.get("occurred_at_ms")
            size = event.get("byte_length")
            if (
                event.get("kind") == "audio_chunk"
                and track in TEMP_TRACK_FILES
                and isinstance(occurred_at, (int, float))
                and isinstance(size, int)
            ):
                result.setdefault(track, []).append(
                    (max(0, round(occurred_at)), max(0, size))
                )
    return result


def _render_track(
    directory: str,
    track: str,
    fmt: Mapping[str, Any],
    events: list[tuple[int, int]],
    output_rate: int,
) -> array:
    path = os.path.join(directory, TEMP_TRACK_FILES[track])
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, IsADirectoryError):
        raw = b""
    if not events and raw:
        events = [(0, len(raw))]

    source_rate = fmt["sample_rate_hz"]
    channels = fmt["channels"]
    output = array("h")
    offset = 0
    for occurred_at, size in events:
        data = raw[offset : offset + size]
        offset += len(data)
        mono = _decode_mono(data, channels)
        if not mono:
            continue
        start = round(occurred_at * output_rate / 1000)
        resampled = _resample(mono, source_rate, output_rate)
        required = start + len(resampled)
        if required > len(output):
            output.extend(0 for _ in range(required - len(output)))
        output[start:required] = resampled
    if offset < len(raw):
        mono = _decode_mono(raw[offset:], channels)
        start = len(output)
        resampled = _resample(mono, source_rate, output_rate)
        output.extend(resampled)
    return output


def _decode_mono(data: bytes, channels: int) -> array:
    usable = len(data) - (len(data) % (channels * 2))
    samples = array("h")
    samples.frombytes(data[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    if channels == 1:
        return samples
    mono = array("h")
    for index in range(0, len(samples), channels):
        mono.append(round(sum(samples[index : index + channels]) / channels))
    return mono


def _resample(samples: array, source_rate: int, output_rate: int) -> array:
    if source_rate == output_rate:
        return array("h", samples)
    output_length = max(1, round(len(samples) * output_rate / source_rate))
    if len(samples) < 2:
        return array("h", [samples[0] if samples else 0]) * output_length
    output = array("h")
    for index in range(output_length):
        position = index * source_rate / output_rate
        left = min(len(samples) - 1, int(position))
        right = min(len(samples) - 1, left + 1)
        fraction = position - left
        output.append(round(samples[left] + (samples[right] - samples[left]) * fraction))
    return output
