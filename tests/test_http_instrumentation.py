"""Automatic HTTP capture through httpx and aiohttp.

Mirrors nodejs-sdk/test/fetch-instrumentation.test.js. Python has no global
`fetch`, so the instrumentation patches the two clients a voice agent uses;
these tests exercise both and always restore the originals.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
import httpx
import pytest
from aiohttp import web

from conftest import operations, read_events
from vaani_observer import VaaniObserver

ENDPOINTS = [
    {"id": "llm", "type": "llm", "url": "https://api.example.com/v1/chat"},
    {"id": "tts", "type": "tts", "url": "https://api.example.com/v1/speak"},
]


@pytest.fixture
def instrumented(tmp_path):
    """An observer with live instrumentation, always uninstalled afterwards."""
    created = []

    def factory(**options):
        options.setdefault("spool_directory", str(tmp_path / f"spool-{len(created)}"))
        options.setdefault("endpoints", ENDPOINTS)
        observer = VaaniObserver(**options)
        created.append(observer)
        return observer

    yield factory
    for observer in created:
        observer.uninstall_instrumentation()


def client(handler=None, **kwargs) -> httpx.AsyncClient:
    def default(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler or default), **kwargs)


# ------------------------------------------------------------------ patching


def test_patches_the_http_clients_when_the_instrumentation_is_enabled(instrumented):
    original = httpx.AsyncClient.send
    observer = instrumented()
    assert httpx.AsyncClient.send is not original
    assert "AsyncClient.send" in observer._http_instrumentation.targets
    assert "ClientSession._request" in observer._http_instrumentation.targets
    observer.uninstall_instrumentation()
    assert httpx.AsyncClient.send is original


def test_leaves_the_http_clients_untouched_when_the_instrumentation_is_disabled(instrumented):
    original = httpx.AsyncClient.send
    instrumented(instrumentations={"http": False})
    assert httpx.AsyncClient.send is original


# --------------------------------------------------------------------- httpx


async def test_passes_calls_through_untouched_when_there_is_no_active_session(instrumented):
    instrumented()
    async with client() as http:
        response = await http.get("https://api.example.com/v1/chat")
    assert response.status_code == 200


async def test_records_an_operation_for_a_classified_call_inside_a_session(instrumented):
    observer = instrumented()
    session = observer.start_session()
    async with client() as http:
        with session.context():
            await http.get("https://api.example.com/v1/chat/completions?key=secret")
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["type"] == "llm"
    assert event["endpoint_id"] == "llm"
    assert event["transport"] == "http"
    assert event["status"] == "ok"
    assert event["response"] == {"status": 200}
    assert event["duration_ms"] >= 0


async def test_never_records_the_request_url_headers_or_body(instrumented):
    observer = instrumented()
    session = observer.start_session()
    async with client() as http:
        with session.context():
            await http.post(
                "https://api.example.com/v1/chat",
                headers={"authorization": "Bearer super-secret"},
                json={"prompt": "private text"},
            )
    finalized = await session.end()
    raw = json.dumps(read_events(finalized.directory))
    assert "super-secret" not in raw
    assert "private text" not in raw
    assert "api.example.com" not in raw


async def test_captures_bounded_request_and_response_bodies_when_enabled(instrumented):
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 12})
    session = observer.start_session()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="response body that is longer")

    async with client(handler) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content="request body that is longer")
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["request"]["body"] == {
        "_truncated": True,
        "_original_bytes": 27,
        "_preview": "request body",
    }
    assert event["response"]["body"] == {
        "_truncated": True,
        "_original_bytes": 28,
        "_preview": "response bod",
    }
    assert event["milestones"]["request_body_captured"]["count"] == 1


async def test_keeps_the_recent_conversation_when_a_chat_body_exceeds_the_budget(instrumented):
    # A byte prefix of this body would stop inside the system prompt and carry
    # no conversation at all, which is exactly the capture a reviewer cannot use.
    messages = [
        {"role": "system", "content": "x" * 4000},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "what did I just ask?"},
    ]
    body = json.dumps({"model": "gpt-4o-mini", "messages": messages, "stream": True})
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 1024})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=body)
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]

    assert captured["_truncated"] is True
    assert captured["_original_bytes"] == len(body.encode("utf-8"))
    preview = json.loads(captured["_preview"])
    assert preview["model"] == "gpt-4o-mini"
    assert preview["messages"][-1]["content"] == "what did I just ask?"
    assert any(message["content"] == "first question" for message in preview["messages"])
    assert len(captured["_preview"].encode("utf-8")) <= 1024


async def test_elides_the_oldest_messages_when_even_the_recent_ones_overflow(instrumented):
    messages = [{"role": "user", "content": f"turn {index} " * 20} for index in range(40)]
    body = json.dumps({"model": "gpt-4o-mini", "messages": messages})
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 1024})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=body)
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]

    assert captured["_elided_messages"] > 0
    preview = json.loads(captured["_preview"])
    assert preview["_elided_messages"] == captured["_elided_messages"]
    assert preview["messages"][-1]["content"] == messages[-1]["content"]
    assert len(preview["messages"]) + preview["_elided_messages"] == len(messages)


async def test_trades_away_tool_schemas_rather_than_the_conversation(instrumented):
    # Tool schemas repeat on every call; the exchange never does. A budget that
    # the schemas fill first would leave the reviewer with no conversation.
    tools = [
        {"type": "function", "function": {"name": f"tool_{index}", "description": "d" * 400, "parameters": {"type": "object"}}}
        for index in range(6)
    ]
    messages = [
        {"role": "system", "content": "s" * 500},
        {"role": "user", "content": "the question that matters"},
    ]
    body = json.dumps({"model": "gpt-4o-mini", "messages": messages, "tools": tools})
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 1024})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=body)
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]

    preview = json.loads(captured["_preview"])
    assert "tool schema(s) omitted" in preview["tools"]
    assert preview["messages"][-1]["content"] == "the question that matters"
    assert len(captured["_preview"].encode("utf-8")) <= 1024


async def test_keeps_the_preview_within_the_budget_when_content_needs_heavy_escaping(instrumented):
    # Escaping turns one byte into two or six, so a budget computed against raw
    # bytes overflows. Only the serialized size is a real measure.
    messages = [{"role": "system", "content": '"' * 4000}, {"role": "user", "content": "hi"}]
    body = json.dumps({"model": "m", "messages": messages})
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 2000})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=body)
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]

    assert len(captured["_preview"].encode("utf-8")) <= 2000
    preview = json.loads(captured["_preview"])
    assert preview["messages"][0]["_content_truncated"] is True
    assert preview["messages"][-1]["content"] == "hi"


async def test_elides_a_message_that_is_not_a_mapping_rather_than_spreading_it(instrumented):
    body = json.dumps({"model": "m", "messages": ["x" * 2000, {"role": "user", "content": "hi"}]})
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 500})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=body)
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]

    assert len(captured["_preview"].encode("utf-8")) <= 500
    preview = json.loads(captured["_preview"])
    assert preview["messages"] == [{"role": "user", "content": "hi"}]
    assert preview["_elided_messages"] == 1


async def test_never_splits_a_multi_byte_character_when_shortening(instrumented):
    body = json.dumps({"model": "m", "messages": [{"role": "system", "content": "\u4e2d" * 1000}, {"role": "user", "content": "hi"}]})
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 2000})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=body)
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]

    preview = json.loads(captured["_preview"])
    assert "\ufffd" not in preview["messages"][0]["content"]
    assert set(preview["messages"][0]["content"]) <= {"\u4e2d"}
    assert len(captured["_preview"].encode("utf-8")) <= 2000


async def test_falls_back_to_a_byte_prefix_for_a_body_that_is_not_a_chat_request(instrumented):
    observer = instrumented(capture={"http_bodies": True, "payload_max_bytes": 12})
    session = observer.start_session()

    async with client(lambda request: httpx.Response(200, text="ok")) as http:
        with session.context():
            await http.post("https://api.example.com/v1/chat", content=json.dumps({"audio": "x" * 50}))
    finalized = await session.end()
    captured = operations(read_events(finalized.directory))[0]["request"]["body"]
    assert captured["_preview"] == '{"audio": "x'


async def test_marks_a_non_2xx_response_as_an_error_while_still_returning_it(instrumented):
    observer = instrumented()
    session = observer.start_session()
    async with client(lambda request: httpx.Response(503, text="bad")) as http:
        with session.context():
            response = await http.get("https://api.example.com/v1/chat")
    assert response.status_code == 503
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["status"] == "error"
    assert event["response"] == {"status": 503}


async def test_records_a_transport_failure_and_reraises_the_original_error(instrumented):
    observer = instrumented()
    session = observer.start_session()

    def handler(request: httpx.Request):
        raise httpx.ConnectError("connect failed")

    async with client(handler) as http:
        with session.context():
            with pytest.raises(httpx.ConnectError):
                await http.get("https://api.example.com/v1/chat")
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["status"] == "error"
    assert event["error"] == {"name": "ConnectError", "message": "connect failed"}
    assert "status" not in event["response"]


async def test_ignores_an_unclassified_url_inside_a_session_context(instrumented):
    observer = instrumented()
    session = observer.start_session()
    async with client() as http:
        with session.context():
            await http.get("https://telemetry.example.com/collect")
    finalized = await session.end()
    assert operations(read_events(finalized.directory)) == []


async def test_lets_a_scoped_endpoint_id_override_url_classification(instrumented):
    observer = instrumented()
    session = observer.start_session()
    async with client() as http:
        with session.with_endpoint("tts"):
            await http.get("https://unmapped.example.com/anything")
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["type"] == "tts"
    assert event["endpoint_id"] == "tts"


async def test_tags_the_operation_with_the_scoped_turn_id(instrumented):
    observer = instrumented()
    session = observer.start_session()
    turn = session.start_turn("turn-4")
    async with client() as http:
        with turn.context():
            await http.get("https://api.example.com/v1/chat")
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["turn_id"] == "turn-4"


async def test_records_one_operation_per_concurrent_call(instrumented):
    observer = instrumented()
    session = observer.start_session()
    async with client() as http:
        with session.context():
            await asyncio.gather(
                http.get("https://api.example.com/v1/chat"),
                http.get("https://api.example.com/v1/chat"),
                http.get("https://api.example.com/v1/speak"),
            )
    finalized = await session.end()
    written = operations(read_events(finalized.directory))
    assert sorted(event["type"] for event in written) == ["llm", "llm", "tts"]
    assert len({event["event_id"] for event in written}) == 3


async def test_does_not_record_calls_made_after_the_session_ended(instrumented):
    observer = instrumented()
    session = observer.start_session()
    finalized = await session.end()
    async with client() as http:
        with session.context():
            await http.get("https://api.example.com/v1/chat")
    assert operations(read_events(finalized.directory)) == []


async def test_propagates_an_ambiguous_rule_error_to_the_caller(instrumented):
    observer = instrumented(
        endpoints=[
            {"id": "a", "type": "llm", "url": "https://api.example.com/v1"},
            {"id": "b", "type": "tts", "url": "https://api.example.com/v1"},
        ]
    )
    session = observer.start_session()
    async with client() as http:
        with session.context():
            with pytest.raises(ValueError, match="Ambiguous"):
                await http.get("https://api.example.com/v1/chat")
    await session.end()


async def test_does_not_drain_a_streaming_response_body(instrumented):
    observer = instrumented(capture={"http_bodies": True})
    session = observer.start_session()
    async with client(lambda request: httpx.Response(200, text="a chunked answer")) as http:
        with session.context():
            async with http.stream("GET", "https://api.example.com/v1/chat") as response:
                chunks = [chunk async for chunk in response.aiter_text()]
    assert "".join(chunks) == "a chunked answer"
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["response"]["body"] == {"_capture_skipped": "Streaming response body."}


def test_the_sync_httpx_client_is_instrumented_too(instrumented):
    observer = instrumented()
    session = observer.start_session()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    with httpx.Client(transport=transport) as http:
        with session.context():
            http.get("https://api.example.com/v1/chat")
    # The operation is already recorded; finalization just needs a loop.
    finalized = asyncio.new_event_loop().run_until_complete(session.end())
    assert operations(read_events(finalized.directory))[0]["type"] == "llm"


# ------------------------------------------------------------------- aiohttp


@pytest.fixture
async def aiohttp_server_url():
    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith("/fail"):
            return web.Response(status=503, text="bad")
        if request.path.endswith("/stream"):
            response = web.StreamResponse(
                headers={"content-type": "text/event-stream"}
            )
            await response.prepare(request)
            await response.write(b"data: one\n\n")
            await response.write(b"data: two\n\n")
            await response.write_eof()
            return response
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


async def test_records_an_aiohttp_call_inside_a_session_context(instrumented, aiohttp_server_url):
    observer = instrumented(
        endpoints=[{"id": "stt", "type": "stt", "url": f"{aiohttp_server_url}/v1/listen"}]
    )
    session = observer.start_session()
    async with aiohttp.ClientSession() as http:
        with session.context():
            async with http.post(f"{aiohttp_server_url}/v1/listen", data=b"audio") as response:
                assert response.status == 200
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["type"] == "stt"
    assert event["endpoint_id"] == "stt"
    assert event["transport"] == "http"
    assert event["response"] == {"status": 200}


async def test_marks_a_failed_aiohttp_call_as_an_error(instrumented, aiohttp_server_url):
    observer = instrumented(
        endpoints=[{"id": "stt", "type": "stt", "url": f"{aiohttp_server_url}/v1"}]
    )
    session = observer.start_session()
    async with aiohttp.ClientSession() as http:
        with session.context():
            async with http.get(f"{aiohttp_server_url}/v1/fail") as response:
                assert response.status == 503
    finalized = await session.end()
    assert operations(read_events(finalized.directory))[0]["status"] == "error"


async def test_an_aiohttp_body_is_still_readable_by_the_caller_after_capture(
    instrumented, aiohttp_server_url
):
    """Capture must never consume the payload the application is waiting for."""
    observer = instrumented(
        endpoints=[{"id": "stt", "type": "stt", "url": f"{aiohttp_server_url}/v1"}],
        capture={"http_bodies": True},
    )
    session = observer.start_session()
    async with aiohttp.ClientSession() as http:
        with session.context():
            async with http.get(f"{aiohttp_server_url}/v1/listen") as response:
                assert await response.json() == {"ok": True}
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert json.loads(event["response"]["body"]) == {"ok": True}


async def test_passes_unclassified_aiohttp_calls_through(instrumented, aiohttp_server_url):
    observer = instrumented(
        endpoints=[{"id": "stt", "type": "stt", "url": "https://elsewhere.example.com/v1"}]
    )
    session = observer.start_session()
    async with aiohttp.ClientSession() as http:
        with session.context():
            async with http.get(f"{aiohttp_server_url}/v1/listen") as response:
                assert response.status == 200
    finalized = await session.end()
    assert operations(read_events(finalized.directory)) == []


# -------------------------------------------------- concurrent observers


async def test_a_second_observer_is_instrumented_too(instrumented):
    """A class attribute holds one wrapper; both observers must still capture.

    Binding the patch to whichever observer installed it first would silently
    blind every later observer in the process.
    """
    first = instrumented()
    second = instrumented(
        endpoints=[{"id": "stt", "type": "stt", "url": "https://api.example.com/v1/chat"}]
    )
    a, b = first.start_session(), second.start_session()
    async with client() as http:
        with a.context():
            await http.get("https://api.example.com/v1/chat")
        with b.context():
            await http.get("https://api.example.com/v1/chat")
    first_done, second_done = await a.end(), await b.end()
    assert operations(read_events(first_done.directory))[0]["endpoint_id"] == "llm"
    # The second observer's own rules must be applied, not the first one's.
    assert operations(read_events(second_done.directory))[0]["endpoint_id"] == "stt"
    assert operations(read_events(second_done.directory))[0]["type"] == "stt"


async def test_uninstalling_one_observer_leaves_the_other_instrumented(instrumented):
    first = instrumented()
    second = instrumented()
    first.uninstall_instrumentation()
    session = second.start_session()
    async with client() as http:
        with session.context():
            await http.get("https://api.example.com/v1/chat")
    finalized = await session.end()
    assert len(operations(read_events(finalized.directory))) == 1


def test_the_patches_are_removed_once_every_observer_uninstalls(instrumented):
    original = httpx.AsyncClient.send
    first = instrumented()
    second = instrumented()
    first.uninstall_instrumentation()
    assert httpx.AsyncClient.send is not original
    second.uninstall_instrumentation()
    assert httpx.AsyncClient.send is original


def test_uninstalling_twice_does_not_unbalance_the_refcount(instrumented):
    original = httpx.AsyncClient.send
    first = instrumented()
    second = instrumented()
    first.uninstall_instrumentation()
    first.uninstall_instrumentation()
    assert httpx.AsyncClient.send is not original, "a live observer lost its instrumentation"
    second.uninstall_instrumentation()
    assert httpx.AsyncClient.send is original


# ------------------------------------------------------------- fail open


async def test_a_streaming_request_body_does_not_break_the_call(instrumented):
    """`httpx.Request.content` raises for a streaming body; capture must not."""
    observer = instrumented(capture={"http_bodies": True})
    session = observer.start_session()

    async def chunks():
        yield b"hello "
        yield b"world"

    async with client() as http:
        with session.context():
            response = await http.post("https://api.example.com/v1/chat", content=chunks())
    assert response.status_code == 200
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["status"] == "ok"


async def test_an_sse_response_body_is_never_awaited(instrumented, aiohttp_server_url):
    observer = instrumented(
        endpoints=[{"id": "llm", "type": "llm", "url": f"{aiohttp_server_url}/v1"}],
        capture={"http_bodies": True},
    )
    session = observer.start_session()
    async with aiohttp.ClientSession() as http:
        with session.context():
            async with http.get(f"{aiohttp_server_url}/v1/stream") as response:
                assert response.status == 200
                await response.read()
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["response"]["body"] == {"_capture_skipped": "Streaming response body."}


async def test_a_rule_can_declare_the_provider_and_model_it_identifies(
    instrumented, aiohttp_server_url
):
    """A URL alone never identifies the model, so the rule has to carry it."""
    observer = instrumented(
        endpoints=[
            {
                "id": "stt",
                "type": "stt",
                "url": f"{aiohttp_server_url}/v1",
                "provider": "Deepgram",
                "model": "nova-3",
            }
        ]
    )
    session = observer.start_session()
    async with aiohttp.ClientSession() as http:
        with session.context():
            async with http.post(f"{aiohttp_server_url}/v1/listen", data=b"audio"):
                pass
    finalized = await session.end()
    event = operations(read_events(finalized.directory))[0]
    assert event["provider"] == "Deepgram"
    assert event["model"] == "nova-3"
