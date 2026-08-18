"""The create -> upload -> complete package protocol.

Mirrors nodejs-sdk/test/upload-package.test.js.
"""

from __future__ import annotations

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
    # Retries are exercised deliberately in the retry tests below; everywhere
    # else they would only add sleeps and obscure the call sequence under test.
    options.setdefault("upload", {"retries": 0})
    return VaaniObserver(**options)


async def full_package(new_observer, session_id="call-1"):
    """A finalized package containing events.jsonl plus one stereo recording."""
    session = new_observer().start_session(session_id=session_id, agent_id="support")
    session.record_inbound_audio(b"\x01\x00", {**PCM, "timestamp_ms": 0})
    session.record_outbound_audio(b"\x03\x00", {**PCM, "timestamp_ms": 0})
    session.start_operation(type="llm").end()
    return await session.end(outcome="completed")


async def events_only_package(new_observer):
    """A finalized package with events.jsonl but no audio files at all."""
    session = new_observer().start_session(session_id="call-2")
    session.start_operation(type="stt").end()
    return await session.end()


def default_handler(call, index):
    if index == 1:
        return json_response({"session_id": "call-1", "upload_urls": UPLOAD_URLS}, 201)
    if call["method"] == "PUT":
        return empty_response(204)
    return json_response({"session_id": "call-1", "status": "ready", "operation_count": 1}, 202)


async def test_refuses_to_upload_without_an_endpoint_and_api_key(new_observer):
    finalized = await full_package(new_observer)
    for observer in (uploader(endpoint=None), uploader(api_key=None), VaaniObserver(instrumentations={"http": False})):
        with pytest.raises(ValueError, match="endpoint and api_key are required"):
            await observer.upload_package(finalized)


async def test_performs_create_per_object_upload_and_complete_in_order(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    result = await observer.upload_package(finalized)
    assert result == {"session_id": "call-1", "status": "ready", "operation_count": 1}
    assert [f"{c['method']} {c['url']}" for c in transport.calls] == [
        "POST https://ingest.example.com/v1/sessions",
        "PUT https://objects.example.com/events",
        "PUT https://objects.example.com/call",
        "POST https://ingest.example.com/v1/sessions/call-1/complete",
    ]


async def test_sends_the_manifest_auth_and_idempotency_headers_on_control_calls(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    create, complete = transport.calls[0], transport.calls[-1]
    assert json.loads(create["body"]) == finalized.manifest
    for call in (create, complete):
        assert call["headers"]["content-type"] == "application/json"
        assert call["headers"]["idempotency-key"] == "call-1"
        assert "test-key" in call["headers"]["authorization"]
    # The signed URL is the credential for an object PUT; re-sending the API key
    # would leak it to whatever object store the backend points at.
    assert "authorization" not in transport.calls[1]["headers"]


async def test_reports_the_byte_size_and_sha256_of_every_uploaded_object(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    objects = json.loads(transport.calls[-1]["body"])["objects"]
    assert list(objects) == ["events.jsonl", "call.audio"]
    for name, info in objects.items():
        with open(os.path.join(finalized.directory, name), "rb") as handle:
            data = handle.read()
        assert info["byte_size"] == len(data)
        assert info["sha256"] == sha256(data)
    assert objects["call.audio"]["byte_size"] >= 4


async def test_uploads_the_exact_file_bytes(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    call = next(c for c in transport.calls if c["url"].endswith("/call"))
    assert call["body"][:4] == b"\x03\x00\x01\x00"


async def test_normalizes_a_trailing_slash_on_the_configured_endpoint(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader(endpoint="https://ingest.example.com/")
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    assert transport.calls[0]["url"] == "https://ingest.example.com/v1/sessions"


async def test_url_encodes_the_session_id_in_the_complete_path(new_observer):
    finalized = await full_package(new_observer, session_id="call 1+a")
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    assert transport.calls[-1]["url"] == "https://ingest.example.com/v1/sessions/call%201%2Ba/complete"


async def test_skips_objects_that_were_never_written_to_disk(new_observer):
    finalized = await events_only_package(new_observer)
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    assert [c["method"] for c in transport.calls] == ["POST", "PUT", "POST"]
    assert list(json.loads(transport.calls[-1]["body"])["objects"]) == ["events.jsonl"]


async def test_completes_with_no_objects_when_the_package_directory_is_empty(new_observer):
    finalized = await new_observer().start_session().end()
    observer = uploader()
    transport = RecordingTransport(default_handler).install(observer)

    await observer.upload_package(finalized)
    assert [c["method"] for c in transport.calls] == ["POST", "POST"]
    assert json.loads(transport.calls[-1]["body"])["objects"] == {}


async def test_fails_when_session_creation_is_rejected(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()
    transport = RecordingTransport(lambda call, index: json_response({"detail": "nope"}, 400)).install(observer)

    with pytest.raises(RuntimeError, match="Session creation failed: HTTP 400"):
        await observer.upload_package(finalized)
    assert len(transport.calls) == 1


async def test_fails_when_the_backend_omits_an_upload_url_for_a_file_that_exists(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()

    def handler(call, index):
        if index == 1:
            return json_response({"upload_urls": {"events.jsonl": UPLOAD_URLS["events.jsonl"]}}, 201)
        return empty_response(204)

    RecordingTransport(handler).install(observer)
    with pytest.raises(RuntimeError, match="did not provide an upload URL for call.audio"):
        await observer.upload_package(finalized)


async def test_fails_when_the_create_response_carries_no_upload_urls_at_all(new_observer):
    finalized = await events_only_package(new_observer)
    observer = uploader()
    RecordingTransport(lambda call, index: json_response({"session_id": "call-2"}, 201)).install(observer)

    with pytest.raises(RuntimeError, match="did not provide an upload URL for events.jsonl"):
        await observer.upload_package(finalized)


async def test_fails_and_stops_when_an_object_upload_is_rejected(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()

    def handler(call, index):
        if index == 1:
            return json_response({"upload_urls": UPLOAD_URLS}, 201)
        return empty_response(500)

    transport = RecordingTransport(handler).install(observer)
    with pytest.raises(RuntimeError, match="Upload failed for events.jsonl: HTTP 500"):
        await observer.upload_package(finalized)
    assert len(transport.calls) == 2


async def test_fails_when_completion_is_rejected(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()

    def handler(call, index):
        if index == 1:
            return json_response({"upload_urls": UPLOAD_URLS}, 201)
        if call["method"] == "PUT":
            return empty_response(204)
        return json_response({"detail": "checksum"}, 400)

    RecordingTransport(handler).install(observer)
    with pytest.raises(RuntimeError, match="Session completion failed: HTTP 400"):
        await observer.upload_package(finalized)


async def test_propagates_transport_errors_from_the_network_layer(new_observer):
    finalized = await full_package(new_observer)
    observer = uploader()

    def handler(call, index):
        raise ConnectionError("upload failed")

    RecordingTransport(handler).install(observer)
    with pytest.raises(ConnectionError, match="upload failed"):
        await observer.upload_package(finalized)


async def test_propagates_unexpected_filesystem_errors_instead_of_skipping(new_observer, tmp_path):
    finalized = await events_only_package(new_observer)
    observer = uploader()
    RecordingTransport(default_handler).install(observer)
    broken_directory = tmp_path / "broken"
    broken_directory.mkdir()
    os.mkdir(broken_directory / "events.jsonl")
    broken = type(finalized)(
        session_id=finalized.session_id,
        directory=str(broken_directory),
        manifest=finalized.manifest,
    )

    with pytest.raises(IsADirectoryError):
        await observer.upload_package(broken)


def test_sha256_produces_the_canonical_digest_of_the_given_bytes():
    assert sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
