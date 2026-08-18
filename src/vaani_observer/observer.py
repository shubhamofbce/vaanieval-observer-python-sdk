"""The observer: configuration, endpoint classification, upload.

Direct port of `nodejs-sdk/src/index.js`. Where Node monkey-patches the global
`fetch`, Python has no single HTTP entry point, so the equivalent auto
instrumentation wraps the two clients a voice agent actually uses — `httpx`
(the OpenAI SDK) and `aiohttp` (most streaming provider plugins). See
`instrumentation.py`.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._context import ObserverContext, current_context, use_context
from ._diagnostics import warn_once

logger = logging.getLogger("vaani_observer")
from ._payload import sha256
from .session import FinalizedSession, Session

__all__ = ["VaaniObserver"]

ENDPOINT_TYPES = ("stt", "llm", "tts")
MATCH_STRATEGIES = ("path", "origin", "exact")
PACKAGE_OBJECTS = ("events.jsonl", "call.audio")
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

#: Per-request socket timeout. Without one, `urlopen` waits forever on a
#: half-open socket, which in a job shutdown hook means waiting until SIGKILL.
DEFAULT_UPLOAD_TIMEOUT_S = 30.0
#: Assumed worst-case upload throughput, in bytes/second, used to extend the
#: socket timeout in proportion to body size. 128 KB/s is ~1 Mbps: slow enough
#: that a healthy link never notices, fast enough that a genuinely dead socket
#: is still abandoned rather than held until SIGKILL.
DEFAULT_MIN_THROUGHPUT_BPS = 128 * 1024
#: Statuses worth trying again. A 4xx other than these is a permanent answer and
#: retrying it just burns the shutdown budget.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
#: Below this, compression costs more CPU than it saves on the wire. Call audio
#: is far above it; `events.jsonl` usually is too.
_COMPRESS_MIN_BYTES = 64 * 1024
#: Below this much remaining budget, skip compression and start sending. gzip on
#: a large object is seconds of CPU that produce nothing if the deadline expires
#: before a byte reaches the wire.
_COMPRESS_MIN_BUDGET_S = 10.0

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
            "upload": {
                "retries": 3,
                "timeout_s": DEFAULT_UPLOAD_TIMEOUT_S,
                "compress": True,
                **(upload or {}),
            },
            "strict": bool(strict),
        }
        # Resolved lazily on the first send; see `_request_compat`.
        self._request_takes_timeout: Optional[bool] = None
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

    @classmethod
    def from_env(cls, **overrides: Any) -> Optional["VaaniObserver"]:
        """Build an observer from `VAANI_*` variables, or `None` if unconfigured.

        Used by out-of-process tooling such as `python -m vaani_observer.drain`,
        which has no application config to inherit. Returns `None` rather than
        guessing a destination, so a missing variable surfaces as a clear
        "nothing configured" message instead of uploads quietly going nowhere.
        """
        endpoint = overrides.pop("endpoint", None) or os.environ.get("VAANI_ENDPOINT")
        api_key = overrides.pop("api_key", None) or os.environ.get("VAANI_API_KEY")
        if not endpoint or not api_key:
            return None
        spool = overrides.pop("spool_directory", None) or os.environ.get("VAANI_SPOOL_DIR")
        if spool:
            overrides["spool_directory"] = os.path.abspath(spool)
        # Explicit keyword arguments win over the environment, so a caller can
        # still override what a deployment set.
        upload = {**upload_options_from_env(), **(overrides.pop("upload", None) or {})}
        if upload:
            overrides["upload"] = upload
        return cls(endpoint=endpoint, api_key=api_key, **overrides)

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

    async def upload_package(self, finalized: FinalizedSession,
                             timeout: Optional[float] = None) -> Any:
        """Upload a finalized local package using the direct-object protocol.

        `timeout` budgets the upload across every object and retry, not a
        single request. The usual caller is a job shutdown hook with a hard
        kill deadline, so exceeding the budget must raise while the package is
        still intact on the spool rather than be SIGKILLed halfway through a
        transfer.

        It is checked *between* requests and retries, and it also caps each
        socket timeout, so it bounds a dead or slow-to-accept peer. It does not
        bound a peer that keeps trickling bytes: `urlopen` applies its timeout
        per socket operation, so a response arriving one byte at a time renews
        the deadline forever. Treat the budget as "stop starting new work after
        this", not as a hard wall-clock guarantee.
        """
        import asyncio

        return await asyncio.to_thread(self.upload_package_sync, finalized, timeout)

    def upload_package_sync(self, finalized: FinalizedSession,
                            timeout: Optional[float] = None) -> Any:
        import time

        if not self.options["endpoint"] or not self.options["api_key"]:
            raise ValueError("endpoint and api_key are required for upload_package().")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        base = self.options["endpoint"].rstrip("/")
        headers = {
            "authorization": f"Bearer {self.options['api_key']}",
            "content-type": "application/json",
            "idempotency-key": finalized.session_id,
        }
        create = self._send(
            "POST",
            f"{base}/v1/sessions",
            headers=headers,
            body=json.dumps(finalized.manifest, ensure_ascii=False).encode("utf-8"),
            deadline=deadline,
        )
        if not create.ok:
            raise RuntimeError(f"Session creation failed: HTTP {create.status}")
        upload_urls = (create.json() or {}).get("upload_urls") or {}
        # A backend that predates gzip ingest stores a compressed body verbatim
        # and only fails at /complete, when verification catches the digest
        # mismatch -- i.e. the whole recording is lost. So compression is opt-in
        # by the *server*: no advertisement, no compression.
        accepted = (create.json() or {}).get("accepted_encodings") or []
        allow_gzip = isinstance(accepted, list) and "gzip" in accepted

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
            payload, object_headers = self._encode_object(data, allow_gzip, deadline)
            response = self._send(
                "PUT", upload_urls[name], headers=object_headers, body=payload,
                deadline=deadline,
            )
            if not response.ok:
                raise RuntimeError(f"Upload failed for {name}: HTTP {response.status}")
            # The digest always describes the *object*, never the transfer
            # encoding, so the receiver verifies exactly what it stored.
            objects[name] = {"byte_size": len(data), "sha256": sha256(data)}

        complete = self._send(
            "POST",
            f"{base}/v1/sessions/{urllib.parse.quote(finalized.session_id, safe='')}/complete",
            headers=headers,
            body=json.dumps({"objects": objects}).encode("utf-8"),
            deadline=deadline,
        )
        if not complete.ok:
            raise RuntimeError(f"Session completion failed: HTTP {complete.status}")
        return complete.json()

    def _encode_object(self, data: bytes, allow_gzip: bool = False,
                       deadline: Optional[float] = None) -> "tuple[bytes, Dict[str, str]]":
        """Compress an object for transit when it is worth it *and* accepted.

        Call audio is raw `pcm_s16le` -- 94 KB/s at 24 kHz stereo -- and is the
        only reason an upload is ever slow enough to be killed. gzip roughly
        thirds it for speech at zero dependency cost, and the receiver stores
        the decompressed bytes, so nothing downstream that seeks by sample
        offset has to change.

        `allow_gzip` comes from the server's `accepted_encodings`. Compressing
        for a backend that cannot decompress turns a slow upload into a lost
        one, which is strictly worse than the problem being solved.
        """
        if not allow_gzip:
            return data, {}
        if not self.options["upload"].get("compress", True):
            return data, {}
        if len(data) < _COMPRESS_MIN_BYTES:
            return data, {}
        # Compressing 128 MiB is seconds of CPU on a process already being torn
        # down. Spending the last of the budget on it and *then* timing out with
        # nothing on the wire is worse than uploading raw.
        if deadline is not None:
            import time

            if deadline - time.monotonic() < _COMPRESS_MIN_BUDGET_S:
                return data, {}
        import gzip

        packed = gzip.compress(data, compresslevel=6)
        # Incompressible payloads exist; sending a larger body to save nothing
        # would be a pure loss.
        if len(packed) >= len(data):
            return data, {}
        return packed, {"content-encoding": "gzip"}

    def _send(self, method: str, url: str, headers: Mapping[str, str],
              body: Optional[bytes], deadline: Optional[float] = None) -> HttpResponse:
        """`request()` plus the retry policy the options have always promised.

        `upload.retries` was configured and never read, so one dropped packet
        lost a whole recording. Retrying is safe because every leg of the
        protocol is idempotent: creation carries an idempotency key, object
        PUTs are addressed by a stable URL, and completion is a declarative
        statement of what was uploaded.
        """
        import random
        import time

        attempts = max(0, int(self.options["upload"].get("retries", 0))) + 1
        per_request = self._request_timeout(len(body) if body else 0)
        last_error: Optional[BaseException] = None
        last_response: Optional[HttpResponse] = None
        for attempt in range(attempts):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"Upload budget exhausted before {method} {url}; "
                    "the package is retained locally."
                )
            timeout = per_request if remaining is None else min(per_request, remaining)
            try:
                response = self._request_compat(method, url, headers, body, timeout)
            except Exception as error:  # noqa: BLE001 - transport errors are retryable
                last_error = error
                last_response = None
            else:
                if response.ok or response.status not in _RETRYABLE_STATUS:
                    return response
                last_response = response
            if attempt == attempts - 1:
                break
            # Exponential backoff with jitter, so a fleet of agents recovering
            # from one outage does not resend in lockstep.
            delay = min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random() / 2)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
            if delay > 0:
                time.sleep(delay)
        # A response, even a failing one, is more informative than an exception:
        # the caller knows which object it was uploading and says so.
        if last_response is not None:
            return last_response
        raise last_error if last_error is not None else RuntimeError("Upload failed")

    def _request_timeout(self, body_bytes: int) -> float:
        """A socket timeout that scales with how much there is to send.

        `urlopen(timeout=T)` is **not** an idle timeout: CPython's `sock_sendall`
        takes its deadline once and applies it to the whole send loop, so `T`
        caps total transmission time even on a healthy, steadily-progressing
        connection. Measured: 213 KB of a 64 MiB body sent before a 3 s timeout
        fired against a slow-but-live reader.

        A single constant therefore cannot serve both a 200-byte POST and a
        128 MiB PUT. Using one means every recording above
        `timeout_s x throughput` is not merely slow but *permanently*
        unshippable -- the drain, which is the documented recovery path, would
        hit the identical cap on every retry, forever.

        So `timeout_s` bounds the handshake, and the body is given an
        additional allowance at a deliberately pessimistic assumed throughput.
        This only ever governs when a transfer is slower than that floor; it
        never makes a healthy upload wait.
        """
        base = float(self.options["upload"].get("timeout_s", DEFAULT_UPLOAD_TIMEOUT_S))
        if body_bytes <= 0:
            return base
        floor = float(
            self.options["upload"].get("min_throughput_bps", DEFAULT_MIN_THROUGHPUT_BPS)
        )
        if floor <= 0:
            return base
        return base + body_bytes / floor

    def _request_compat(self, method: str, url: str, headers: Mapping[str, str],
                        body: Optional[bytes], timeout: Optional[float]) -> HttpResponse:
        """Call `request()`, tolerating an override written before `timeout`.

        `request()` is documented as the transport override point, so
        subclasses of it exist outside this repository with the original
        four-argument signature. Calling those with `timeout=` raises
        `TypeError`, which `_send` would treat as a retryable transport error:
        four full retries with backoff inside the shutdown window, and then a
        stranded package. A signature change to an advertised extension point
        must not be able to lose a recording.
        """
        if self._request_takes_timeout is None:
            try:
                import inspect

                parameters = inspect.signature(self.request).parameters
                self._request_takes_timeout = "timeout" in parameters or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            except (TypeError, ValueError):  # pragma: no cover - exotic callables
                self._request_takes_timeout = True
        if self._request_takes_timeout:
            return self.request(method, url, headers, body, timeout=timeout)
        warn_once(
            "legacy-request-override",
            "vaani: %s.request() does not accept `timeout`, so uploads cannot be "
            "bounded and may hang until the process is killed. Add "
            "`timeout=None` to its signature.",
            type(self).__name__,
        )
        return self.request(method, url, headers, body)

    def request(
        self, method: str, url: str, headers: Mapping[str, str],
        body: Optional[bytes], timeout: Optional[float] = None,
    ) -> HttpResponse:
        """The upload transport. Overridable so tests need no live server.

        `urllib` is used rather than an async client on purpose: the upload path
        must not be picked up by this SDK's own HTTP instrumentation, and it
        runs off the call path anyway.
        """
        request = urllib.request.Request(url, data=body, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        if timeout is None:
            timeout = self.options["upload"].get("timeout_s", DEFAULT_UPLOAD_TIMEOUT_S)
        try:
            # Without an explicit timeout `urlopen` waits forever on a half-open
            # socket, which inside a shutdown hook means waiting until SIGKILL.
            with urllib.request.urlopen(  # noqa: S310 - configured URL
                request, timeout=timeout
            ) as response:
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


def upload_options_from_env() -> Dict[str, Any]:
    """Read the `upload` tuning options from `VAANI_UPLOAD_*` variables.

    These options were documented as tunable long before they were reachable:
    the only constructor the docs show is `from_env`, which built the observer
    itself and forwarded nothing. An operator following the tuning table set a
    value that was silently discarded. Anything not set is left absent so the
    constructor's own defaults apply -- this never fabricates a value.
    """
    options: Dict[str, Any] = {}
    for key, name, cast in (
        ("timeout_s", "VAANI_UPLOAD_TIMEOUT_S", float),
        ("retries", "VAANI_UPLOAD_RETRIES", int),
        ("min_throughput_bps", "VAANI_UPLOAD_MIN_THROUGHPUT_BPS", int),
    ):
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            options[key] = cast(raw.strip())
        except ValueError:
            # A typo must not silently fall back to a default the operator
            # believes they overrode.
            logger.warning("vaani: ignoring %s=%r, expected a number", name, raw)
    raw = os.environ.get("VAANI_UPLOAD_COMPRESS")
    if raw is not None and raw.strip():
        options["compress"] = raw.strip().lower() not in {"0", "false", "no", "off"}
    return options
