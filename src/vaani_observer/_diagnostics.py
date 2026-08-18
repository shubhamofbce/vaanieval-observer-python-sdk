"""Make a silent failure audible exactly once.

Capture is best effort, so no single failure may abort a call and none may spam
a production log on every frame. The previous compromise -- log everything at
DEBUG -- meant that three separate bugs all presented to adopters as "it just
records nothing", with the default log level hiding the only evidence.

The rule here is: the *first* occurrence of each failure class is a WARNING,
because an operator who never sees one has no reason to look. Every repeat is
DEBUG, because after the first the information content is zero.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Set

logger = logging.getLogger("vaani_observer")

_lock = threading.Lock()
_seen: Set[str] = set()


def warn_once(key: str, message: str, *args: Any) -> None:
    """WARNING on the first `key`, DEBUG on every one after it."""
    with _lock:
        first = key not in _seen
        if first:
            _seen.add(key)
    if first:
        logger.warning(message, *args)
    else:
        logger.debug(message, *args)


def reset_warnings() -> None:
    """Forget what has been reported. For tests, and for long-lived workers
    that want a fresh report per job."""
    with _lock:
        _seen.clear()
