"""Websocket observation: lifecycle, direction and byte accounting.

Mirrors nodejs-sdk/test/observe-websocket.test.js, adapted to Python sockets,
which expose send/receive coroutines rather than EventEmitter callbacks.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeSocket, operations, read_events
from vaani_observer import observe_websocket

RULES = [
    {"id": "stt", "type": "stt", "url": "wss://stt.example.com/stream"},
    {"id": "tts", "type": "tts", "url": "wss://tts.example.com/speak"},
]


def setup(new_observer, **options):
    vaani = new_observer(endpoints=RULES, **options)
    return vaani, vaani.start_session(), FakeSocket()


def operation_of(events):
    for event in events:
        if event.get("type") in ("stt", "llm", "tts"):
            return event
    return None


async def test_requires_an_explicit_session_or_an_active_context(new_observer):
    vaani, session, socket = setup(new_observer)
    with pytest.raises(ValueError, match="needs a session"):
        vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
    with session.context():
        vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
    await session.end()


async def test_returns_an_inert_handle_and_records_nothing_for_an_unclassified_url(new_observer):
    vaani, session, socket = setup(new_observer)
    original = socket.send_str
    handle = vaani.observe_websocket(socket, session=session, url="wss://unknown.example.com/stream")
    handle.detach()
    assert socket.send_str == original
    finalized = await session.end()
    assert operation_of(read_events(finalized.directory)) is None


async def test_resolves_the_endpoint_by_id_bypassing_url_classification(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(
        socket, session=session, endpoint_id="tts", url="wss://unknown.example.com/x"
    )
    handle.close(1000)
    finalized = await session.end()
    operation = operation_of(read_events(finalized.directory))
    assert operation["type"] == "tts"
    assert operation["endpoint_id"] == "tts"
    assert operation["transport"] == "websocket"
    assert operation["scope"] == "connection"


async def test_records_nothing_when_the_supplied_endpoint_id_is_unknown(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(
        socket, session=session, endpoint_id="nope", url="wss://stt.example.com/stream"
    )
    handle.detach()
    finalized = await session.end()
    assert operation_of(read_events(finalized.directory)) is None


async def test_captures_the_full_lifecycle_with_byte_accounting_in_both_directions(new_observer):
    vaani, session, socket = setup(new_observer)
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream?lang=en")
    await socket.send_bytes(b"\x01\x02\x03")
    await socket.send_str("hello")
    handle_in = vaani  # noqa: F841 - readability only
    await socket.close(1000)

    finalized = await session.end()
    operation = operation_of(read_events(finalized.directory))
    assert operation["status"] == "ok"
    assert operation["endpoint_id"] == "stt"
    assert operation["response"] == {"close_code": 1000, "sent_bytes": 8, "received_bytes": 0}
    assert operation["milestones"]["connected"]["occurred_at_ms"] >= 0
    sent = operation["milestones"]["sent_frame"]
    assert sent["count"] == 2
    assert sent["direction"] == "outbound"
    assert sent["kind"] == "text"
    assert sent["byte_count"] == 5
    assert sent["total_byte_count"] == 8


async def test_counts_received_frames_through_the_receive_coroutine(new_observer):
    vaani = new_observer(endpoints=RULES)
    session = vaani.start_session()
    socket = FakeSocket(incoming=[b"\x01\x02", "partial-text"])
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    await socket.receive()
    await socket.receive()
    await socket.close(1000)
    finalized = await session.end()
    operation = operation_of(read_events(finalized.directory))
    assert operation["response"]["received_bytes"] == 14
    received = operation["milestones"]["received_frame"]
    assert received["count"] == 2
    assert received["direction"] == "inbound"
    assert received["kind"] == "text"
    assert received["byte_count"] == 12
    assert received["total_byte_count"] == 14


async def test_unwraps_an_aiohttp_style_message_object(new_observer):
    class Message:
        def __init__(self, data):
            self.data = data

    vaani = new_observer(endpoints=RULES)
    session = vaani.start_session()
    socket = FakeSocket(incoming=[Message(b"\x01\x02\x03\x04")])
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    await socket.receive()
    await socket.close(1000)
    finalized = await session.end()
    assert operation_of(read_events(finalized.directory))["response"]["received_bytes"] == 4


async def test_classifies_binary_and_text_frames_by_payload_type(new_observer):
    vaani = new_observer(endpoints=RULES)
    session = vaani.start_session()
    socket = FakeSocket(incoming=[b"\x01\x02"])
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    await socket.send_bytes(b"\x01")
    await socket.receive()
    await socket.close(1000)
    finalized = await session.end()
    operation = operation_of(read_events(finalized.directory))
    assert operation["milestones"]["sent_frame"]["kind"] == "binary"
    assert operation["milestones"]["received_frame"]["kind"] == "binary"


async def test_delegates_to_the_original_send_and_preserves_its_return_value(new_observer):
    vaani, session, socket = setup(new_observer)
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    assert await socket.send_str("ping") == "sent"
    assert socket.sent == ["ping"]
    await socket.close(1000)
    await session.end()


async def test_counts_a_frame_with_no_measurable_length_as_zero_bytes(new_observer):
    vaani, session, socket = setup(new_observer)
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    await socket.send_str(None)
    await socket.close(1000)
    finalized = await session.end()
    assert operation_of(read_events(finalized.directory))["response"] == {
        "close_code": 1000,
        "sent_bytes": 0,
        "received_bytes": 0,
    }


async def test_ends_the_operation_as_an_error_when_the_socket_errors(new_observer):
    class Failing(FakeSocket):
        async def receive(self):
            raise TypeError("handshake failed")

    vaani = new_observer(endpoints=RULES)
    session = vaani.start_session()
    socket = Failing()
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    with pytest.raises(TypeError):
        await socket.receive()
    finalized = await session.end()
    operation = operation_of(read_events(finalized.directory))
    assert operation["status"] == "error"
    assert operation["error"] == {"name": "TypeError", "message": "handshake failed"}


async def test_keeps_the_first_terminal_result_when_close_follows_an_error(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    handle.record_error(RuntimeError("reset"))
    await socket.close(1006)
    finalized = await session.end()
    written = [e for e in read_events(finalized.directory) if e.get("type") == "stt"]
    assert len(written) == 1
    assert written[0]["status"] == "error"


async def test_detach_unwraps_send_and_marks_the_operation_cancelled(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    wrapped = socket.send_str
    handle.detach()
    assert socket.send_str != wrapped
    assert await socket.send_str("after-detach") == "sent"
    assert socket.sent == ["after-detach"]
    finalized = await session.end()
    operation = operation_of(read_events(finalized.directory))
    assert operation["status"] == "cancelled"
    assert "sent_frame" not in operation["milestones"]


async def test_detach_after_close_does_not_overwrite_the_recorded_close_result(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    await socket.close(1000)
    handle.detach()
    finalized = await session.end()
    written = [e for e in read_events(finalized.directory) if e.get("type") == "stt"]
    assert len(written) == 1
    assert written[0]["status"] == "ok"


async def test_detach_twice_is_safe(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    handle.detach()
    handle.detach()
    finalized = await session.end()
    assert len([e for e in read_events(finalized.directory) if e.get("type") == "stt"]) == 1


async def test_tolerates_a_socket_without_send_or_receive(new_observer):
    vaani, session, _ = setup(new_observer)
    bare = type("Bare", (), {})()
    handle = vaani.observe_websocket(bare, session=session, url="wss://stt.example.com/stream")
    handle.detach()
    finalized = await session.end()
    assert operation_of(read_events(finalized.directory))["status"] == "cancelled"


async def test_a_socket_open_at_the_end_of_a_completed_call_is_not_cancelled(new_observer):
    """A streaming STT socket stays open for the whole call by design.

    Reporting that expected teardown as `cancelled` made every healthy call
    look like it had lost its provider connection.
    """
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    session.register_socket(handle)
    await socket.send_str("audio")
    finalized = await session.end(outcome="completed")
    operation = operation_of(read_events(finalized.directory))
    assert operation["status"] == "ok"
    assert operation["response"]["sent_bytes"] > 0


async def test_a_socket_open_when_a_call_is_abandoned_is_still_cancelled(new_observer):
    vaani, session, socket = setup(new_observer)
    handle = vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    session.register_socket(handle)
    finalized = await session.end(outcome="abandoned")
    assert operation_of(read_events(finalized.directory))["status"] == "cancelled"


async def test_used_as_a_context_manager_it_cancels_on_exit(new_observer):
    vaani, session, socket = setup(new_observer)
    with vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream"):
        await socket.send_str("hi")
    finalized = await session.end()
    assert operation_of(read_events(finalized.directory))["status"] == "cancelled"


async def test_observes_two_sockets_on_one_session_independently(new_observer):
    vaani = new_observer(endpoints=RULES)
    session = vaani.start_session()
    first = FakeSocket()
    second = FakeSocket(incoming=[bytes(4)])
    vaani.observe_websocket(first, session=session, url="wss://stt.example.com/stream")
    vaani.observe_websocket(second, session=session, url="wss://tts.example.com/speak")
    await first.send_str("abc")
    await second.receive()
    await first.close(1000)
    await second.close(1001)
    finalized = await session.end()
    events = read_events(finalized.directory)
    stt = next(e for e in events if e.get("type") == "stt")
    tts = next(e for e in events if e.get("type") == "tts")
    assert stt["response"]["sent_bytes"] == 3
    assert stt["response"]["received_bytes"] == 0
    assert tts["response"]["received_bytes"] == 4
    assert tts["response"]["close_code"] == 1001


async def test_a_connection_span_covers_the_socket_lifetime_not_a_turn(new_observer):
    """The scope is what stops call-length socket spans polluting turn latency."""
    vaani, session, socket = setup(new_observer)
    vaani.observe_websocket(socket, session=session, url="wss://stt.example.com/stream")
    turn = session.start_turn("t1")
    turn.start_operation(type="stt").end()
    await socket.close(1000)
    finalized = await session.end()
    events = [e for e in read_events(finalized.directory) if e.get("type") == "stt"]
    scopes = {event["scope"]: event["turn_id"] for event in events}
    assert scopes == {"turn": "t1", "connection": None}


# ------------------------------------------------- real client shapes


async def test_counts_an_aiohttp_frame_once_even_through_send_json(new_observer):
    """`send_json` delegates to `send_str`; wrapping both would double count."""
    import aiohttp
    from aiohttp import web

    async def handler(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            if message.type == aiohttp.WSMsgType.TEXT:
                await socket.send_str("pong")
                break
        return socket

    app = web.Application()
    app.router.add_get("/v1/stream", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    url = f"http://127.0.0.1:{port}/v1/stream"
    try:
        observer = new_observer(
            endpoints=[{"id": "stt", "type": "stt", "url": f"http://127.0.0.1:{port}/v1"}]
        )
        session = observer.start_session()
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(url) as socket:
                handle = observe_websocket(observer, socket, session=session, url=url)
                await socket.send_json({"hello": "world"})
                await socket.receive()
                handle.detach()
        finalized = await session.end()
    finally:
        await runner.cleanup()

    event = operations(read_events(finalized.directory))[0]
    assert event["milestones"]["sent_frame"]["count"] == 1
    assert event["milestones"]["received_frame"]["count"] == 1
    assert event["milestones"]["sent_frame"]["total_byte_count"] == len(
        json.dumps({"hello": "world"}).encode()
    )


async def test_auto_observes_an_aiohttp_socket_opened_inside_a_session_context(new_observer):
    """LiveKit's plugins own their sockets, so nobody can hand one to us."""
    import aiohttp
    from aiohttp import web

    async def handler(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            await socket.send_str("pong:" + message.data)
        return socket

    app = web.Application()
    app.router.add_get("/stream", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/stream"

    vaani = new_observer(endpoints=[{"id": "stt", "type": "stt", "url": url}])
    session = vaani.start_session()
    try:
        async with aiohttp.ClientSession() as http:
            with session.context():
                async with http.ws_connect(url) as socket:
                    await socket.send_str("ping")
                    assert (await socket.receive()).data == "pong:ping"
        finalized = await session.end()
    finally:
        await runner.cleanup()

    event = operation_of(read_events(finalized.directory))
    assert event is not None, "the socket was never observed"
    assert event["scope"] == "connection"
    assert event["transport"] == "websocket"
    assert event["response"]["sent_bytes"] == len("ping")
    assert event["response"]["received_bytes"] == len("pong:ping")


async def test_observing_the_same_socket_twice_does_not_double_count(new_observer):
    """Auto-observation and an explicit hand-over can reach the same socket."""
    vaani, session, socket = setup(new_observer)
    with session.context():
        vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
        vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
        await socket.send_str("hello")
        await socket.close()
    finalized = await session.end()
    events = read_events(finalized.directory)
    connections = [e for e in events if e.get("scope") == "connection"]
    assert len(connections) == 1
    sent = connections[0]["milestones"]["sent_frame"]
    assert sent["count"] == 1
    assert sent["total_byte_count"] == len("hello")


async def test_a_socket_cancelled_at_shutdown_is_not_reported_as_a_transport_error(
    new_observer,
):
    """Every call ends by cancelling its receive loop; that is not a fault."""
    import asyncio

    vaani, session, socket = setup(new_observer)
    with session.context():
        vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
        await socket.send_str("hello")
        socket.fail_receive(asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await socket.receive()
    finalized = await session.end()
    event = operation_of(read_events(finalized.directory))
    assert event["status"] == "cancelled"
    assert "error" not in event or event["error"] is None
    assert event["response"]["sent_bytes"] == len("hello")


async def test_a_socket_still_open_when_the_call_ends_still_gets_a_span(new_observer):
    """An auto-observed socket has no owner to close it, so `end()` must."""
    import aiohttp
    from aiohttp import web

    async def handler(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.receive()
        return socket

    app = web.Application()
    app.router.add_get("/stream", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/stream"

    vaani = new_observer(endpoints=[{"id": "stt", "type": "stt", "url": url}])
    session = vaani.start_session()
    try:
        async with aiohttp.ClientSession() as http:
            with session.context():
                socket = await http.ws_connect(url)
                await socket.send_str("ping")
                # The call ends while the socket is still open.
                finalized = await session.end()
            await socket.close()
    finally:
        await runner.cleanup()

    event = operation_of(read_events(finalized.directory))
    assert event is not None, "the open socket's span was dropped"
    assert event["status"] == "cancelled"
    assert event["milestones"]["sent_frame"]["total_byte_count"] == len("ping")


async def test_a_send_that_the_provider_rejects_ends_the_span_as_an_error(new_observer):
    """A socket that died mid-call must not be reported as healthy."""

    class BrokenSocket(FakeSocket):
        async def send_str(self, payload: str) -> str:
            raise ConnectionResetError("provider went away")

    vaani = new_observer(endpoints=RULES)
    session = vaani.start_session()
    socket = BrokenSocket()
    with session.context():
        vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
        with pytest.raises(ConnectionResetError):
            await socket.send_str("hello")
    finalized = await session.end()
    event = operation_of(read_events(finalized.directory))
    assert event["status"] == "error"
    assert event["error"]["name"] == "ConnectionResetError"
    # Bytes that never reached the wire are not counted.
    assert "sent_frame" not in event["milestones"]


async def test_a_socket_registered_while_the_call_is_ending_is_not_lost(new_observer):
    """Instrumentation runs on whichever thread opened the socket."""
    vaani, session, socket = setup(new_observer)
    with session.context():
        handle = vaani.observe_websocket(socket, url="wss://stt.example.com/stream")
        await socket.send_str("hello")
    assert session.register_socket(handle) is True
    finalized = await session.end()
    # Registration is closed once finalization begins, so a late arrival is
    # handed back to the caller rather than silently dropped.
    assert session.register_socket(object()) is False
    event = operation_of(read_events(finalized.directory))
    assert event["status"] == "cancelled"
    assert event["milestones"]["sent_frame"]["total_byte_count"] == len("hello")


async def test_a_provider_cleanup_that_hangs_does_not_hang_the_call():
    """A stalled provider must not stop the process from exiting."""
    import asyncio

    from vaani_observer.integrations import livekit as livekit_module

    started = asyncio.Event()

    class Hanging:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            started.set()
            await asyncio.sleep(3600)

    monkey = livekit_module._CLOSE_TIMEOUT_S
    livekit_module._CLOSE_TIMEOUT_S = 0.05
    try:
        iterator = Hanging()
        await asyncio.wait_for(livekit_module._close_quietly(iterator), timeout=5)
    finally:
        livekit_module._CLOSE_TIMEOUT_S = monkey
    assert started.is_set()
