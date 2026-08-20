"""Which destinations a spool has been associated with.

A spool holds raw call audio and whatever the caller said. Nothing about a
directory of packages records where those bytes were meant to go, so a drain
pass run with a mistyped `VAANI_ENDPOINT` will happily ship every retained
recording to an arbitrary host and -- because delivery is what authorises
cleanup -- delete the local copies afterwards. The operator sees a successful
drain. That is a data-exfiltration footgun, not a misconfiguration.

This module keeps a plain-text ledger at the spool root naming every endpoint
the spool has been used with. It lives beside the packages rather than inside
them precisely so it survives the purge that removes them, which is the case
that matters: the ledger has to outlive the evidence it protects.

It is a guardrail, not a security control. Anything that can write the spool can
write the ledger, and a first-ever drain has nothing to compare against. What it
buys is that a *change* of destination stops being silent.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

LEDGER_NAME = ".vaani-destinations"

logger = logging.getLogger("vaani_observer.spool")


def normalize_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """Compare destinations the way an operator means them.

    A trailing slash and a change of case in the host are not a different
    server, and treating them as one would make the guard cry wolf on every
    other invocation -- which is how a guard gets disabled.
    """
    if not endpoint:
        return None
    text = str(endpoint).strip().rstrip("/")
    if not text:
        return None
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(text)
        if parts.scheme and parts.netloc:
            # Scheme and host are case-insensitive per RFC 3986; the path is
            # not, and lowercasing it would merge two genuinely different
            # destinations. Without this the guard fired on `Localhost` vs
            # `localhost` -- a false refusal, which is worse than no guard
            # because it stops the drain permanently.
            text = urlunsplit(
                (parts.scheme.lower(), parts.netloc.lower(), parts.path,
                 parts.query, parts.fragment)
            ).rstrip("/")
    except ValueError:  # pragma: no cover - malformed URLs stay as typed
        pass
    return text or None


def ledger_path(spool_directory: str) -> str:
    return os.path.join(spool_directory, LEDGER_NAME)


def known_destinations(spool_directory: str) -> List[str]:
    """Every endpoint this spool has been used with, oldest first.

    An unreadable or absent ledger yields an empty list: the guard's job is to
    catch a *changed* destination, and with no record of a previous one there is
    nothing to warn about. Failing open here is deliberate -- a corrupt ledger
    must not be able to stop recordings from ever shipping.
    """
    try:
        with open(ledger_path(spool_directory), "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, ValueError):
        return []
    seen: List[str] = []
    for line in lines:
        value = normalize_endpoint(line)
        if value and value not in seen:
            seen.append(value)
    return seen


def remember_destination(spool_directory: str, endpoint: Optional[str]) -> None:
    """Record an endpoint as one this spool is associated with.

    Append-only and idempotent. Failure is logged at debug and swallowed: a
    spool that cannot write its ledger is still a spool, and refusing to record
    a call because a guardrail file could not be updated would trade a warning
    for the data loss the warning exists to prevent.
    """
    value = normalize_endpoint(endpoint)
    if not value or value in known_destinations(spool_directory):
        return
    try:
        os.makedirs(spool_directory, exist_ok=True)
        with open(ledger_path(spool_directory), "a", encoding="utf-8") as handle:
            handle.write(value + "\n")
    except OSError as error:  # pragma: no cover - disk-level failure
        logger.debug("Could not record spool destination %s: %s", value, error)
