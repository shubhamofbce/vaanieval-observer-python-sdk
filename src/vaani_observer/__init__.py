"""Vaani Observer — framework-independent voice-call observability for Python.

Local-first: a call is written to a portable session package on disk
(`manifest.json`, `events.jsonl`, `call.audio`) and the media
path never waits on a remote upload.

    from vaani_observer import VaaniObserver

    vaani = VaaniObserver(
        spool_directory="/tmp/vaani",
        endpoints=[{"id": "llm", "type": "llm", "url": "https://llm.example/v1"}],
    )
    session = vaani.start_session(agent_id="support")
    session.record_inbound_audio(pcm, {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1})
    with session.context():
        ...                                   # provider calls are attributed here
    finalized = await session.end(outcome="completed")
    await vaani.upload_package(finalized)     # explicit, post-call, never inline
"""

from ._context import ObserverContext, current_context
from ._payload import sha256
from .observer import VaaniObserver
from .session import FinalizedSession, Operation, Session, Turn
from .websocket import WebSocketHandle, observe_websocket

__all__ = [
    "VaaniObserver",
    "Session",
    "Operation",
    "Turn",
    "FinalizedSession",
    "ObserverContext",
    "WebSocketHandle",
    "observe_websocket",
    "current_context",
    "sha256",
]

from ._version import __version__  # noqa: E402  (re-exported)
