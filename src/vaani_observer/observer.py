"""The observer: configuration, endpoint classification, upload.

Direct port of `nodejs-sdk/src/index.js`. Where Node monkey-patches the global
`fetch`, Python has no single HTTP entry point, so the equivalent auto
instrumentation wraps the two clients a voice agent actually uses — `httpx`
(the OpenAI SDK) and `aiohttp` (most streaming provider plugins). See
`instrumentation.py`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._context import ObserverContext, current_context, use_context
from ._payload import sha256
from .session import FinalizedSession, Session

__all__ = ["VaaniObserver"]

ENDPOINT_TYPES = ("stt", "llm", "tts")
MATCH_STRATEGIES = ("path", "origin", "exact")
PACKAGE_OBJECTS = ("events.jsonl", "call.audio")
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

# A provider's websocket lives at the same origin as its REST API -- Deepgram
# streams from `wss://api.deepgram.com/v1/listen` and transcribes files at
# `https://api.deepgram.com/v1/listen`. Comparing schemes literally would mean a
# rule could only ever classify one of the two, so a config that looks complete
# would silently record no connection spans at all. Matching is therefore done
# on the transport-neutral scheme.
_TRANSPORT_NEUTRAL_SCHEMES = {"ws": "http", "wss": "https"}


def _comparable_scheme(scheme: str) -> str:
    return _TRANSPORT_NEUTRAL_SCHEMES.get(scheme, scheme)


@dataclass(frozen=True)
class HttpResponse:
    """The minimal response shape the upload protocol needs."""

    status: int
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body or b"{}")

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class ParsedUrl:
    """A comparable absolute URL. Normalizes the default port away, as WHATWG does."""

    __slots__ = ("scheme", "host", "path", "query", "href")

    def __init__(self, value: Any) -> None:
        if isinstance(value, ParsedUrl):
            text = value.href
        elif isinstance(value, str):
            text = value
        else:
            raise TypeError(f"Invalid URL: {value!r}")
        parts = urllib.parse.urlsplit(text)
        if not parts.scheme or not parts.netloc:
            raise TypeError(f"Invalid URL: {value!r}")
        self.scheme = parts.scheme.lower()
        host = parts.netloc.lower()
        default = DEFAULT_PORTS.get(self.scheme)
        if default is not None and host.endswith(f":{default}"):
            host = host[: -len(f":{default}")]
        self.host = host
        self.path = parts.path or "/"
        self.query = f"?{parts.query}" if parts.query else ""
        self.href = urllib.parse.urlunsplit(
            (self.scheme, self.host, parts.path, parts.query, "")
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParsedUrl({self.href!r})"


class VaaniObserver:
    """A provider-neutral, local-first observer for voice-agent calls."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        spool_directory: Optional[str] = None,
        capture: Optional[Mapping[str, Any]] = None,
        instrumentations: Optional[Mapping[str, Any]] = None,
        endpoints: Optional[Sequence[Mapping[str, Any]]] = None,
        upload: Optional[Mapping[str, Any]] = None,
        strict: bool = False,
    ) -> None:
        self.options: Dict[str, Any] = {
            "endpoint": endpoint,
            "api_key": api_key,
            "spool_directory": spool_directory
            or os.path.join(os.getcwd(), ".vaani-spool"),
            # Provider transcript content is application-defined and can include
            # sensitive speech. The SDK exposes the policy flag but integrations
            # must explicitly attach STT results to their per-turn operation.
            "capture": {
                "audio": True,
                "http_bodies": False,
                "websocket_text_frames": False,
                "stt_content": False,
                "payload_max_bytes": 16 * 1024,
                **(capture or {}),
            },
            "instrumentations": {"http": True, "websocket": True, **(instrumentations or {})},
            "endpoints": list(endpoints or []),
            "upload": {"retries": 3, **(upload or {})},
            "strict": bool(strict),
        }
        self.endpoint_rules: List[Dict[str, Any]] = _validate_endpoint_rules(
            self.options["endpoints"]
        )
        self._sessions: "set[Session]" = set()
        self._http_instrumentation = None
        self._websocket_instrumentation = None
        if self.options["instrumentations"]["http"]:
            self._install_http_instrumentation()
        if self.options["instrumentations"]["websocket"]:
            self._install_websocket_instrumentation()

    # -------------------------------------------------------------- sessions

    def start_session(self, **input: Any) -> Session:
        session = Session(self, **input)
        self._sessions.add(session)
        # Drop the reference as soon as the package is on disk, so a long-lived
        # observer in a worker process does not accumulate finished sessions.
        session._completion.add_done_callback(
            lambda _future, session=session: self._sessions.discard(session)
        )
        return session

    async def flush(self) -> None:
        """Wait for local session finalization. Does not force a network upload.

        Tracks *completion*, not `end()` being called: a session that is still
        writing its manifest is exactly the one a caller is flushing for.
        Finalization errors propagate, matching Node's rejecting `Promise.all`.
        """
        import asyncio

        pending = [
            session for session in list(self._sessions) if not session._completion.done()
        ]
        if not pending:
            return
        await asyncio.gather(*[session.finished for session in pending])

    # --------------------------------------------------------------- context

    def context(
        self,
        session: Session,
        endpoint_id: Optional[str] = None,
        turn_id: Optional[Any] = None,
    ):
        """Scope ambient work to a session, optionally to an endpoint and turn.

        Values not supplied are inherited from the enclosing scope, so
        `with session.with_turn(t):` inside `with session.with_endpoint(e):`
        keeps both.
        """
        existing = current_context()
        if endpoint_id is None and existing is not None and existing.session is session:
            endpoint_id = existing.endpoint_id
        if turn_id is None and existing is not None and existing.session is session:
            turn_id = existing.turn_id
        return use_context(
            ObserverContext(
                session=session,
                endpoint_id=endpoint_id,
                turn_id=None if turn_id is None else str(turn_id),
            )
        )

    def current_context(self) -> Optional[ObserverContext]:
        return current_context()

    # -------------------------------------------------------- classification

    def classify_url(self, url: Any) -> Optional[Dict[str, Any]]:
        """Return the configured endpoint rule a URL belongs to, if any."""
        candidate = ParsedUrl(url)
        # A rule written for the exact scheme always wins. Only when nothing
        # matches literally does the transport-neutral form apply, so adding
        # websocket coverage can never make a previously working pair of
        # `https://` and `wss://` rules ambiguous.
        matches = [
            rule for rule in self.endpoint_rules if _matches(candidate, rule, exact_scheme=True)
        ]
        if not matches:
            matches = [rule for rule in self.endpoint_rules if _matches(candidate, rule)]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous Vaani endpoint rules for {candidate.scheme}://"
                f"{candidate.host}{candidate.path}"
            )
        return matches[0] if matches else None

    def rule_for(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        for rule in self.endpoint_rules:
            if rule["id"] == endpoint_id:
                return rule
        return None

    # ------------------------------------------------------------ websockets

    def observe_websocket(self, socket: Any, **kwargs: Any):
        from .websocket import observe_websocket

        return observe_websocket(self, socket, **kwargs)

    # ---------------------------------------------------------- HTTP capture

    def _install_http_instrumentation(self) -> None:
        from .instrumentation import install_http_instrumentation

        self._http_instrumentation = install_http_instrumentation(self)

    def _install_websocket_instrumentation(self) -> None:
        from .websocket import install_websocket_instrumentation

        self._websocket_instrumentation = install_websocket_instrumentation(self)

    def uninstall_instrumentation(self) -> None:
        """Restore the original HTTP and websocket client methods. Mainly for tests."""
        if self._http_instrumentation is not None:
            self._http_instrumentation.uninstall()
            self._http_instrumentation = None
        if self._websocket_instrumentation is not None:
            self._websocket_instrumentation.uninstall()
            self._websocket_instrumentation = None

    # --------------------------------------------------------------- upload

    async def upload_package(self, finalized: FinalizedSession) -> Any:
        """Upload a finalized local package using the direct-object protocol."""
        import asyncio

        return await asyncio.to_thread(self.upload_package_sync, finalized)

    def upload_package_sync(self, finalized: FinalizedSession) -> Any:
        if not self.options["endpoint"] or not self.options["api_key"]:
            raise ValueError("endpoint and api_key are required for upload_package().")
        base = self.options["endpoint"].rstrip("/")
        headers = {
            "authorization": f"Bearer {self.options['api_key']}",
            "content-type": "application/json",
            "idempotency-key": finalized.session_id,
        }
        create = self.request(
            "POST",
            f"{base}/v1/sessions",
            headers=headers,
            body=json.dumps(finalized.manifest, ensure_ascii=False).encode("utf-8"),
        )
        if not create.ok:
            raise RuntimeError(f"Session creation failed: HTTP {create.status}")
        upload_urls = (create.json() or {}).get("upload_urls") or {}

        objects: Dict[str, Any] = {}
        for name in PACKAGE_OBJECTS:
            path = os.path.join(finalized.directory, name)
            try:
                with open(path, "rb") as handle:
                    data = handle.read()
            except FileNotFoundError:
                continue
            if not upload_urls.get(name):
                raise RuntimeError(f"Backend did not provide an upload URL for {name}.")
            # Object PUTs deliberately carry no authorization header: the
            # signed URL is the credential, and re-sending the API key would
            # leak it to whatever object store the backend points at.
            response = self.request("PUT", upload_urls[name], headers={}, body=data)
            if not response.ok:
                raise RuntimeError(f"Upload failed for {name}: HTTP {response.status}")
            objects[name] = {"byte_size": len(data), "sha256": sha256(data)}

        complete = self.request(
            "POST",
            f"{base}/v1/sessions/{urllib.parse.quote(finalized.session_id, safe='')}/complete",
            headers=headers,
            body=json.dumps({"objects": objects}).encode("utf-8"),
        )
        if not complete.ok:
            raise RuntimeError(f"Session completion failed: HTTP {complete.status}")
        return complete.json()

    def request(
        self, method: str, url: str, headers: Mapping[str, str], body: Optional[bytes]
    ) -> HttpResponse:
        """The upload transport. Overridable so tests need no live server.

        `urllib` is used rather than an async client on purpose: the upload path
        must not be picked up by this SDK's own HTTP instrumentation, and it
        runs off the call path anyway.
        """
        request = urllib.request.Request(url, data=body, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310 - configured URL
                return HttpResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as error:
            return HttpResponse(status=error.code, body=error.read())


# ----------------------------------------------------------------- helpers


def _validate_endpoint_rules(rules: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: "set[str]" = set()
    validated: List[Dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise TypeError("Each endpoint needs id, type, and url.")
        rule_id = rule.get("id")
        rule_type = rule.get("type")
        url = rule.get("url")
        if not rule_id or not isinstance(rule_id, str):
            raise TypeError("Each endpoint needs id, type, and url.")
        if rule_type not in ENDPOINT_TYPES:
            raise TypeError("Each endpoint needs id, type, and url.")
        if not url:
            raise TypeError("Each endpoint needs id, type, and url.")
        if rule_id in seen:
            raise TypeError(f"Duplicate endpoint id: {rule_id}")
        seen.add(rule_id)
        match = rule.get("match") or "path"
        if match not in MATCH_STRATEGIES:
            raise TypeError(f"Unknown endpoint match strategy: {match}")
        validated.append({**dict(rule), "match": match, "url": ParsedUrl(url)})
    return validated


def _matches(
    candidate: ParsedUrl, rule: Mapping[str, Any], exact_scheme: bool = False
) -> bool:
    target: ParsedUrl = rule["url"]
    if exact_scheme:
        scheme_matches = candidate.scheme == target.scheme
    else:
        scheme_matches = _comparable_scheme(candidate.scheme) == _comparable_scheme(
            target.scheme
        )
    if not scheme_matches or candidate.host != target.host:
        return False
    if rule["match"] == "origin":
        return True
    if rule["match"] == "exact":
        return candidate.path == target.path and candidate.query == target.query
    return candidate.path.startswith(target.path)
