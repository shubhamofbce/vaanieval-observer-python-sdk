"""Ambient session context.

The Node SDK uses `AsyncLocalStorage` so that auto-instrumented provider calls
can find the session they belong to without every call site threading it
through. `contextvars.ContextVar` is the direct Python equivalent: it is
inherited by tasks created from the current task, which is exactly the
propagation an asyncio voice agent needs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import Session


@dataclass(frozen=True)
class ObserverContext:
    """The session, endpoint and turn that ambient work should be attributed to."""

    session: "Session"
    endpoint_id: Optional[str] = None
    turn_id: Optional[str] = None


_CURRENT: ContextVar[Optional[ObserverContext]] = ContextVar(
    "vaani_observer_context", default=None
)


def current_context() -> Optional[ObserverContext]:
    return _CURRENT.get()


@contextmanager
def use_context(context: ObserverContext) -> Iterator[ObserverContext]:
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def set_context(context: Optional[ObserverContext]):
    """Set the context without a scope. The caller owns the returned token.

    Needed by `bind()`, which has to install a context inside a coroutine it
    does not control the lifetime of.
    """
    return _CURRENT.set(context)


def reset_context(token) -> None:
    _CURRENT.reset(token)
