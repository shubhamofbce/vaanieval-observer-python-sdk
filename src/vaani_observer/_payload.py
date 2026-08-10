"""Bounded payload capture.

Provider request and response bodies are application data of unknown size. The
SDK keeps them useful but never lets one of them dominate a session package, so
everything retained goes through a byte budget and is replaced by a truncation
marker once it is exceeded.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

BYTES_TYPES = (bytes, bytearray, memoryview)


def sha256(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _preview(raw: bytes, limit: int) -> str:
    # A byte budget can land mid-codepoint; dropping the partial character is
    # better than persisting a payload that no JSON reader can decode.
    return raw[:limit].decode("utf-8", errors="ignore")


def bounded_payload(value: Any, limit: int) -> Any:
    """Return `value`, or a truncation marker when its JSON form exceeds `limit`."""
    if limit is None or not isinstance(limit, (int, float)) or math.isinf(limit) or limit < 0:
        return value
    try:
        raw = json_bytes(value)
    except (TypeError, ValueError):
        return {"_capture_error": "Payload is not JSON serializable."}
    if len(raw) <= limit:
        return value
    return {
        "_truncated": True,
        "_original_bytes": len(raw),
        "_preview": _preview(raw, int(limit)),
    }


def bounded_text(value: str, limit: int) -> Any:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    structured = bounded_chat_body(value, int(limit))
    if structured is not None:
        return structured
    return {"_truncated": True, "_original_bytes": len(raw), "_preview": _preview(raw, int(limit))}


def _shorten_message(message: Mapping[str, Any], budget: int) -> dict:
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps("" if content is None else content, ensure_ascii=False)
    raw = text.encode("utf-8")

    def build(room: int) -> dict:
        return {
            **message,
            "content": _preview(raw, room),
            "_content_truncated": True,
            "_content_bytes": len(raw),
        }

    # JSON escaping can turn one content byte into six, so the room a budget
    # allows cannot be derived by subtraction. Shrink until the serialized
    # message actually fits, which is the only measure that matters.
    low, high = 0, len(raw)
    while low < high:
        mid = (low + high + 1) // 2
        if len(json_bytes(build(mid))) <= budget:
            low = mid
        else:
            high = mid - 1
    return build(low)


def bounded_chat_body(text: str, limit: int) -> dict | None:
    """Bound a chat-completions body by dropping whole messages, not bytes.

    A byte prefix keeps the system prompt — which is identical on every call —
    and discards the conversation, which is the only part that differs. On a
    real agent the instructions alone can exceed the limit, so the stored
    preview ends mid-sentence inside message zero, no message survives intact,
    and the capture answers none of the questions it was taken to answer.

    Keeping the newest messages instead preserves the exchange that actually
    produced this reply, records how many older ones were elided, and leaves the
    preview as valid JSON so it can still be parsed and read.

    Returns `None` when the body is not a chat request, so the caller falls back
    to the byte prefix.
    """
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    messages = parsed.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    envelope = {**parsed, "messages": []}
    # Room for the elision marker and the separators between kept messages.
    budget = limit - len(json_bytes(envelope)) - 96
    # Tool schemas are boilerplate that repeats on every call of the session,
    # and a large one can crowd out the conversation entirely. The exchange is
    # worth more than the schemas, so trade them away rather than the messages.
    tools = parsed.get("tools")
    if budget <= 256 and isinstance(tools, list) and tools:
        envelope["tools"] = f"[{len(tools)} tool schema(s) omitted to keep the conversation]"
        budget = limit - len(json_bytes(envelope)) - 96
    if budget <= 0:
        return None

    kept: list[Any] = []
    elided = 0
    for index in range(len(messages) - 1, -1, -1):
        cost = len(json_bytes(messages[index])) + 1
        if cost <= budget:
            kept.insert(0, messages[index])
            budget -= cost
            continue
        # The standing instructions are worth a summary even when they do not fit.
        if index == 0 and kept and budget > 256 and isinstance(messages[0], Mapping):
            kept.insert(0, _shorten_message(messages[0], budget))
            budget = 0
            continue
        elided = index + 1
        break
    # A large `tools` block can consume the budget before a single message fits.
    # Falling back to a byte prefix there would lose the newest exchange for the
    # calls that need it most, so keep a shortened version of it instead.
    if not kept and budget > 256 and isinstance(messages[-1], Mapping):
        kept.append(_shorten_message(messages[-1], budget))
        elided = len(messages) - 1
    if not kept:
        return None

    preview = {**envelope, "messages": kept}
    if elided:
        preview["_elided_messages"] = elided
    return {
        "_truncated": True,
        "_original_bytes": len(text.encode("utf-8")),
        "_preview": json_bytes(preview).decode("utf-8"),
        "_elided_messages": elided,
    }


def bounded_body(body: Any, limit: int) -> Any:
    """Bound an outgoing HTTP request body of whichever shape the client used."""
    if body is None:
        return None
    if isinstance(body, str):
        return bounded_text(body, limit)
    if isinstance(body, BYTES_TYPES):
        return bounded_text(bytes(body).decode("utf-8", errors="replace"), limit)
    if isinstance(body, Mapping) or isinstance(body, (list, tuple)):
        return bounded_payload(body, limit)
    return {"_capture_skipped": "Unsupported or streaming request body."}


def safe_error(error: BaseException | None) -> dict:
    """A JSON-safe error record. Never carries a traceback or provider payload."""
    if error is None:
        return {"name": "Error", "message": ""}
    return {"name": type(error).__name__, "message": str(error)}


def drop_none(value: Mapping[str, Any]) -> dict:
    """Drop `None` entries so the manifest matches the Node package byte for byte.

    `JSON.stringify` omits `undefined` object values; `json.dumps` would write
    `null`. Audio format fields are the case that matters, because the dashboard
    treats a present-but-null `encoding` differently from an absent one.
    """
    return {key: item for key, item in value.items() if item is not None}
