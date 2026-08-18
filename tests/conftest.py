"""Shared helpers for the observer test-suite. Contains no tests itself."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pytest

from vaani_observer import VaaniObserver
from vaani_observer.observer import HttpResponse

PCM = {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1}


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path / "spool")


@pytest.fixture
def new_observer(tmp_path):
    """An observer that spools into a throwaway directory and patches no client."""
    created: List[VaaniObserver] = []
    counter = {"n": 0}

    def factory(**options: Any) -> VaaniObserver:
        counter["n"] += 1
        options.setdefault("spool_directory", str(tmp_path / f"spool-{counter['n']}"))
        options.setdefault("instrumentations", {"http": False})
        observer = VaaniObserver(**options)
        created.append(observer)
        return observer

    yield factory
    for observer in created:
        observer.uninstall_instrumentation()


def read_events(directory: str) -> List[Dict[str, Any]]:
    path = os.path.join(directory, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_manifest(directory: str) -> Dict[str, Any]:
    with open(os.path.join(directory, "manifest.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_track(directory: str, track: str) -> bytes:
    with open(os.path.join(directory, f"{track}.audio"), "rb") as handle:
        return handle.read()


def operations(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for event in events if event.get("type") in ("stt", "llm", "tts", "tool")]


class RecordingTransport:
    """Replaces the observer's upload transport with a recording stub."""

    def __init__(self, handler) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._handler = handler

    def __call__(self, method: str, url: str, headers, body, timeout=None) -> HttpResponse:
        call = {
            "method": method,
            "url": url,
            "headers": dict(headers),
            "body": body,
            "timeout": timeout,
        }
        self.calls.append(call)
        return self._handler(call, len(self.calls))

    def install(self, observer: VaaniObserver) -> "RecordingTransport":
        observer.request = self  # type: ignore[method-assign]
        return self


def json_response(body: Any, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, body=json.dumps(body).encode("utf-8"))


def empty_response(status: int = 204) -> HttpResponse:
    return HttpResponse(status=status, body=b"")


class FakeSocket:
    """Minimal duck-typed async socket: send/receive/close, recording payloads."""

    def __init__(self, incoming: Optional[List[Any]] = None) -> None:
        self.sent: List[Any] = []
        self.closed = False
        self.close_code: Optional[int] = None
        self._incoming = list(incoming or [])
        self._receive_error: Optional[BaseException] = None

    def fail_receive(self, error: BaseException) -> None:
        """Make the next receive raise, the way a torn-down socket does."""
        self._receive_error = error

    async def send_str(self, payload: str) -> str:
        self.sent.append(payload)
        return "sent"

    async def send_bytes(self, payload: bytes) -> str:
        self.sent.append(payload)
        return "sent"

    async def receive(self) -> Any:
        if self._receive_error is not None:
            error, self._receive_error = self._receive_error, None
            raise error
        if not self._incoming:
            raise StopAsyncIteration
        return self._incoming.pop(0)

    async def close(self, code: int = 1000) -> bool:
        self.closed = True
        self.close_code = code
        return True
