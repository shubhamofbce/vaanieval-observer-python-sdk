"""Regressions for the upload path: timeouts, retries, budget and compression.

The review's second P0 was that a recording could be lost between "finalized on
disk" and "stored in the backend", with nothing in the transport to prevent it:
no socket timeout, a `retries` option that was configured and never read, and an
upload large enough that a job shutdown hook was killed part-way through.


Sixteen of these fail against the pre-fix source (`git show
HEAD:src/vaani_observer/observer.py`). Six cannot, because they guard
behaviour that did not exist before -- compression, and `retries` being read
at all -- so "nothing was compressed" and "nothing was retried" held trivially:

    test_a_permanent_rejection_is_not_retried
    test_retries_are_exhausted_with_a_message_naming_the_object
    test_a_small_object_is_sent_as_is
    test_compression_can_be_turned_off
    test_nothing_is_compressed_for_a_backend_that_never_said_it_could
    test_an_unknown_advertised_encoding_is_not_used
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

from conftest import PCM, RecordingTransport, empty_response, json_response
from vaani_observer import VaaniObserver, sha256

UPLOAD_URLS = {
    "events.jsonl": "https://objects.example.com/events",
    "call.audio": "https://objects.example.com/call",
}


def uploader(**options):
    options.setdefault("endpoint", "https://ingest.example.com")
    options.setdefault("api_key", "test-key")
    options.setdefault("instrumentations", {"http": False})
    return VaaniObserver(**options)


async def package(new_observer, *, audio_bytes: bytes = b"\x01\x00", session_id="call-1"):
    session = new_observer().start_session(session_id=session_id, agent_id="support")
    session.record_inbound_audio(audio_bytes, {**PCM, "timestamp_ms": 0})
    session.record_outbound_audio(audio_bytes, {**PCM, "timestamp_ms": 0})
    session.start_operation(type="llm").end()
    return await session.end(outcome="completed")


def ok_handler(call, index):
    if index == 1:
        return json_response({"upload_urls": UPLOAD_URLS, "accepted_encodings": ["gzip"]}, 201)
    return json_response({"status": "ready"})


def legacy_handler(call, index):
    """A backend predating gzip ingest: it advertises no encodings at all."""
    if index == 1:
        return json_response({"upload_urls": UPLOAD_URLS}, 201)
    return json_response({"status": "ready"})


# ------------------------------------------------------------------- timeouts


async def test_every_request_carries_a_socket_timeout(new_observer):
    """`urlopen` with no timeout waits forever, and a shutdown hook cannot."""
    finalized = await package(new_observer)
    observer = uploader()
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    assert transport.calls, "the upload must have happened"
    assert all(call["timeout"] and call["timeout"] > 0 for call in transport.calls)


async def test_the_configured_timeout_bounds_the_handshake(new_observer):
    """`timeout_s` applies as-is to requests that carry no body."""
    finalized = await package(new_observer)
    observer = uploader(upload={"timeout_s": 5.0, "retries": 0})
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    control = [call for call in transport.calls if len(call.get("body") or b"") < 4096]
    assert control, "the protocol has small control-plane requests"
    # A few hundred bytes at the default floor is well under a millisecond.
    assert all(5.0 <= call["timeout"] < 5.01 for call in control)


async def test_a_large_body_is_given_more_time_than_the_handshake(new_observer):
    """`urlopen`'s timeout caps the *whole* send, not idle time.

    CPython's `sock_sendall` takes its deadline once and applies it across the
    entire send loop, so a flat timeout makes every object above
    `timeout_s x throughput` permanently unshippable -- the drain would hit the
    identical cap on every retry, forever.
    """
    finalized = await package(new_observer)
    observer = uploader(
        upload={"timeout_s": 5.0, "retries": 0, "compress": False,
                "min_throughput_bps": 1000}
    )
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    for call in transport.calls:
        size = len(call.get("body") or b"")
        assert call["timeout"] == pytest.approx(5.0 + size / 1000)
    assert any(len(call.get("body") or b"") > 0 for call in transport.calls)


async def test_the_throughput_floor_can_be_turned_off(new_observer):
    finalized = await package(new_observer)
    observer = uploader(
        upload={"timeout_s": 5.0, "retries": 0, "min_throughput_bps": 0}
    )
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    assert {call["timeout"] for call in transport.calls} == {5.0}


async def test_an_override_written_before_timeout_existed_still_uploads(new_observer, caplog):
    """`request()` is a documented extension point; breaking it lost recordings."""
    import logging

    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 3})
    seen = []

    def legacy_request(method, url, headers, body):  # no `timeout` parameter
        seen.append(method)
        return ok_handler({"method": method, "url": url, "body": body}, len(seen))

    observer.request = legacy_request

    with caplog.at_level(logging.WARNING, logger="vaani_observer"):
        result = await observer.upload_package(finalized)

    assert result["status"] == "ready"
    assert seen, "the legacy transport must actually be called"
    assert "does not accept `timeout`" in caplog.text


# -------------------------------------------------------------------- retries


async def test_a_transient_failure_is_retried_rather_than_losing_the_call(new_observer):
    """`upload.retries` was configured and read by nothing at all."""
    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 3, "timeout_s": 1.0})
    attempts = {"count": 0}

    def handler(call, index):
        if call["method"] == "POST" and call["url"].endswith("/v1/sessions"):
            attempts["count"] += 1
            if attempts["count"] < 3:
                return empty_response(503)
            return json_response({"upload_urls": UPLOAD_URLS}, 201)
        return json_response({"status": "ready"})

    RecordingTransport(handler).install(observer)
    await observer.upload_package(finalized)

    assert attempts["count"] == 3, "the 503s must have been retried"


async def test_a_dropped_connection_is_retried(new_observer):
    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 2, "timeout_s": 1.0})
    seen = {"count": 0}

    def handler(call, index):
        if call["method"] == "POST" and call["url"].endswith("/v1/sessions"):
            seen["count"] += 1
            if seen["count"] == 1:
                raise OSError("connection reset by peer")
            return json_response({"upload_urls": UPLOAD_URLS}, 201)
        return json_response({"status": "ready"})

    RecordingTransport(handler).install(observer)
    await observer.upload_package(finalized)

    assert seen["count"] == 2


async def test_a_permanent_rejection_is_not_retried(new_observer):
    """Retrying a 401 only burns the shutdown budget it needs to fail inside."""
    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 5, "timeout_s": 1.0})
    transport = RecordingTransport(lambda call, index: empty_response(401)).install(observer)

    with pytest.raises(RuntimeError, match="Session creation failed: HTTP 401"):
        await observer.upload_package(finalized)

    assert len(transport.calls) == 1


async def test_retries_are_exhausted_with_a_message_naming_the_object(new_observer):
    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 1, "timeout_s": 0.01})

    def handler(call, index):
        if index == 1:
            return json_response({"upload_urls": UPLOAD_URLS}, 201)
        return empty_response(500)

    RecordingTransport(handler).install(observer)
    with pytest.raises(RuntimeError, match="Upload failed for events.jsonl: HTTP 500"):
        await observer.upload_package(finalized)


# --------------------------------------------------------------------- budget


async def test_the_upload_budget_is_enforced_across_the_whole_upload(new_observer):
    """A shutdown hook has a hard deadline; the upload must respect it."""
    import time

    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 0, "timeout_s": 30.0})

    def handler(call, index):
        time.sleep(0.05)
        if index == 1:
            return json_response({"upload_urls": UPLOAD_URLS}, 201)
        return json_response({"status": "ready"})

    RecordingTransport(handler).install(observer)
    with pytest.raises(TimeoutError, match="retained locally"):
        await observer.upload_package(finalized, timeout=0.06)

    # The package is still on disk, which is the whole point: over budget must
    # mean "ship it later", never "lose it".
    assert os.path.exists(os.path.join(finalized.directory, "manifest.json"))


async def test_the_per_request_timeout_never_exceeds_the_remaining_budget(new_observer):
    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 0, "timeout_s": 30.0})
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized, timeout=2.0)

    assert all(call["timeout"] <= 2.0 for call in transport.calls)


# ---------------------------------------------------------------- compression


async def test_a_large_object_is_gzipped_in_transit(new_observer):
    """Raw PCM is ~94 KB/s; shipping it uncompressed is what runs out of time."""
    finalized = await package(new_observer, audio_bytes=b"\x01\x00" * 60000)
    observer = uploader(upload={"retries": 0})
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    puts = [call for call in transport.calls if call["method"] == "PUT"]
    audio = [call for call in puts if call["url"].endswith("/call")][0]
    assert audio["headers"].get("content-encoding") == "gzip"
    assert len(audio["body"]) < 60000, "the body must actually be smaller"


async def test_the_digest_describes_the_object_not_the_compressed_body(new_observer):
    """The receiver stores decompressed bytes, so it verifies decompressed bytes."""
    finalized = await package(new_observer, audio_bytes=b"\x01\x00" * 60000)
    observer = uploader(upload={"retries": 0})
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    puts = [call for call in transport.calls if call["method"] == "PUT"]
    audio = [call for call in puts if call["url"].endswith("/call")][0]
    original = gzip.decompress(audio["body"])
    with open(os.path.join(finalized.directory, "call.audio"), "rb") as handle:
        on_disk = handle.read()
    assert original == on_disk

    complete = [call for call in transport.calls
                if call["method"] == "POST" and call["url"].endswith("/complete")][0]
    declared = json.loads(complete["body"])["objects"]["call.audio"]
    assert declared["sha256"] == sha256(on_disk)
    assert declared["byte_size"] == len(on_disk)


async def test_a_small_object_is_sent_as_is(new_observer):
    """Compressing a few hundred bytes costs CPU and saves nothing."""
    finalized = await package(new_observer)
    observer = uploader(upload={"retries": 0})
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    puts = [call for call in transport.calls if call["method"] == "PUT"]
    assert all("content-encoding" not in call["headers"] for call in puts)


async def test_compression_can_be_turned_off(new_observer):
    finalized = await package(new_observer, audio_bytes=b"\x01\x00" * 60000)
    observer = uploader(upload={"retries": 0, "compress": False})
    transport = RecordingTransport(ok_handler).install(observer)

    await observer.upload_package(finalized)

    puts = [call for call in transport.calls if call["method"] == "PUT"]
    assert all("content-encoding" not in call["headers"] for call in puts)


async def test_nothing_is_compressed_for_a_backend_that_never_said_it_could(new_observer):
    """Measured against a real pre-gzip dashboard: every upload was lost.

    It accepts the compressed body with a `204`, stores it verbatim, and only
    fails at `/complete` with a digest mismatch -- by which point the shutdown
    hook has already returned. Compression must therefore be offered by the
    server, never assumed by the client.
    """
    finalized = await package(new_observer, audio_bytes=b"\x01\x00" * 60000)
    observer = uploader(upload={"retries": 0})
    transport = RecordingTransport(legacy_handler).install(observer)

    await observer.upload_package(finalized)

    puts = [call for call in transport.calls if call["method"] == "PUT"]
    assert puts, "the upload must have happened"
    assert all("content-encoding" not in call["headers"] for call in puts)
    audio = [call for call in puts if call["url"].endswith("/call")][0]
    with open(os.path.join(finalized.directory, "call.audio"), "rb") as handle:
        assert audio["body"] == handle.read()


async def test_an_unknown_advertised_encoding_is_not_used(new_observer):
    """`accepted_encodings: ["br"]` must not be read as "compression is fine"."""
    finalized = await package(new_observer, audio_bytes=b"\x01\x00" * 60000)
    observer = uploader(upload={"retries": 0})

    def handler(call, index):
        if index == 1:
            return json_response({"upload_urls": UPLOAD_URLS, "accepted_encodings": ["br"]}, 201)
        return json_response({"status": "ready"})

    transport = RecordingTransport(handler).install(observer)

    await observer.upload_package(finalized)

    puts = [call for call in transport.calls if call["method"] == "PUT"]
    assert all("content-encoding" not in call["headers"] for call in puts)


# ------------------------------------------------------------------- from_env


def test_from_env_returns_none_rather_than_guessing_a_destination(monkeypatch):
    monkeypatch.delenv("VAANI_ENDPOINT", raising=False)
    monkeypatch.delenv("VAANI_API_KEY", raising=False)
    assert VaaniObserver.from_env() is None


def test_from_env_reads_the_documented_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("VAANI_ENDPOINT", "https://ingest.example.com")
    monkeypatch.setenv("VAANI_API_KEY", "key-1")
    monkeypatch.setenv("VAANI_SPOOL_DIR", str(tmp_path / "spool"))
    observer = VaaniObserver.from_env(instrumentations={"http": False, "websocket": False})
    assert observer is not None
    assert observer.options["endpoint"] == "https://ingest.example.com"
    assert observer.options["spool_directory"] == str(tmp_path / "spool")


# ---------------------------------------------------------- reachability


def test_upload_tuning_is_reachable_from_the_environment(monkeypatch):
    """The docs' tuning table was unreachable from the documented constructor.

    `from_env` built the observer itself and forwarded no `upload` mapping, so
    every value an operator set from the tuning table was silently discarded.
    """
    monkeypatch.setenv("VAANI_ENDPOINT", "http://localhost:8000")
    monkeypatch.setenv("VAANI_API_KEY", "k")
    monkeypatch.setenv("VAANI_UPLOAD_TIMEOUT_S", "120")
    monkeypatch.setenv("VAANI_UPLOAD_RETRIES", "5")
    monkeypatch.setenv("VAANI_UPLOAD_COMPRESS", "off")

    observer = VaaniObserver.from_env()

    assert observer is not None
    assert observer.options["upload"]["timeout_s"] == 120.0
    assert observer.options["upload"]["retries"] == 5
    assert observer.options["upload"]["compress"] is False


def test_an_explicit_argument_still_beats_the_environment(monkeypatch):
    monkeypatch.setenv("VAANI_ENDPOINT", "http://localhost:8000")
    monkeypatch.setenv("VAANI_API_KEY", "k")
    monkeypatch.setenv("VAANI_UPLOAD_TIMEOUT_S", "120")

    observer = VaaniObserver.from_env(upload={"timeout_s": 7.0})

    assert observer is not None
    assert observer.options["upload"]["timeout_s"] == 7.0


def test_a_malformed_upload_variable_is_reported_not_swallowed(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("VAANI_ENDPOINT", "http://localhost:8000")
    monkeypatch.setenv("VAANI_API_KEY", "k")
    monkeypatch.setenv("VAANI_UPLOAD_TIMEOUT_S", "two minutes")

    with caplog.at_level(logging.WARNING, logger="vaani_observer"):
        observer = VaaniObserver.from_env()

    assert observer is not None
    assert observer.options["upload"]["timeout_s"] == 30.0
    assert "VAANI_UPLOAD_TIMEOUT_S" in caplog.text
