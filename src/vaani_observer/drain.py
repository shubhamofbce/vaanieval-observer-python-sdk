"""Upload finalized packages that are still sitting on the local spool.

Recording and uploading are deliberately separate steps. A call finalizes to
disk in milliseconds; shipping it can take tens of seconds on a slow link, and
the process that recorded it may not live that long -- a LiveKit worker's
default `shutdown_process_timeout` is ten seconds, and it SIGKILLs whatever is
still running. Anything that fails to upload in time is therefore left intact on
the spool rather than lost, and this module is what ships it afterwards.

Run it as a one-shot after a batch of calls::

    python -m vaani_observer.drain

or as a sidecar next to a long-running worker::

    python -m vaani_observer.drain --watch 60

Both read `VAANI_ENDPOINT`, `VAANI_API_KEY` and `VAANI_SPOOL_DIR` unless
overridden by flags. On a host with ephemeral storage -- LiveKit Cloud among
them -- the spool disappears with the container, so a sidecar there must share a
volume with the worker or the drain has nothing to find.
"""

from __future__ import annotations

import json
import logging
import os
import calendar
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .observer import VaaniObserver, upload_options_from_env
from .session import FinalizedSession

__all__ = ["pending_packages", "drain_spool", "DrainResult", "main"]

logger = logging.getLogger("vaani_observer.drain")

#: Written next to a package once the backend has acknowledged it. Its presence
#: is what makes a second drain pass a no-op instead of a re-upload.
RECEIPT_NAME = "uploaded.json"
MANIFEST_NAME = "manifest.json"

#: How long a package that the backend has already acknowledged is kept on
#: disk before a drain pass removes it. Not zero, because during a migration
#: or an incident the local package is the only evidence available, and a
#: receipt says the bytes were *accepted*, not that anyone has looked at them.
DEFAULT_DELIVERED_TTL_S = 24 * 3600

# A package upload has to be bounded or one stalled peer starves the queue
# behind it: `once()` is synchronous, so in the documented `--watch` sidecar
# form every other pending recording waits. The floor rate means a maximum-size
# 128 MiB package legitimately needs ~17 minutes, so the budget has to clear
# that or slow-but-healthy links would be cut off; 30 minutes leaves headroom
# while still bounding a stall to something an operator can wait out.
DEFAULT_PACKAGE_TIMEOUT_S = 1800.0


@dataclass
class DrainResult:
    """What one pass over the spool did."""

    uploaded: List[str]
    failed: List[str]
    skipped: List[str]
    purged: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def __str__(self) -> str:  # pragma: no cover - operator-facing summary
        return (
            f"{len(self.uploaded)} uploaded, {len(self.failed)} failed, "
            f"{len(self.skipped)} already delivered, {len(self.purged)} purged"
        )


def pending_packages(spool_directory: str) -> List[FinalizedSession]:
    """Every finalized, not-yet-delivered package on the spool, oldest first.

    A package counts as finalized only once `manifest.json` exists, and that
    file is published by an atomic rename, so a call still being written can
    never be picked up half-formed by a concurrent drain.
    """
    return _scan(spool_directory)[0]


def _receipt_covers(directory: str, endpoint: Optional[str]) -> bool:
    """Does the receipt prove *this* backend already has the package?

    A receipt only ever meant "somebody verified the digests". Treating that as
    "delivered" regardless of destination is wrong the moment the endpoint
    moves -- staging to prod, self-host to cloud, a tenant migration. Every old
    receipt would still count, the package would be skipped forever, and once a
    TTL purge exists it would then be deleted from the only machine that had
    it. Scoping the receipt to its endpoint turns that silent loss into a
    re-upload, which is idempotent and therefore free.

    Receipts written before this field existed carry no endpoint. Those are
    honoured as-is: a pre-existing spool must not suddenly re-upload wholesale.
    """
    path = os.path.join(directory, RECEIPT_NAME)
    if not os.path.exists(path):
        return False
    if not endpoint:
        return True
    try:
        with open(path, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError):
        # An unreadable receipt is not evidence of delivery. Re-uploading is
        # idempotent; assuming delivery is not recoverable.
        return False
    if not isinstance(receipt, dict):
        return False
    recorded = receipt.get("endpoint")
    if not recorded:
        return True
    return str(recorded).rstrip("/") == str(endpoint).rstrip("/")


def _scan(spool_directory: str, endpoint: Optional[str] = None) -> "tuple[List[FinalizedSession], List[str]]":
    """Split the spool into what still needs shipping and what already went.

    The delivered half is returned rather than discarded so the drain can say
    which it is. "0 uploaded, 0 failed" on its own reads identically whether
    the spool is empty, fully delivered, or simply the wrong path -- and an
    operator checking that no recordings are stranded needs to tell those
    apart.
    """
    try:
        entries = sorted(os.scandir(spool_directory), key=lambda entry: entry.name)
    except FileNotFoundError:
        return [], []
    packages: List[FinalizedSession] = []
    delivered: List[str] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        manifest_path = os.path.join(entry.path, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            continue
        if _receipt_covers(entry.path, endpoint):
            delivered.append(entry.name)
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning("Skipping unreadable manifest at %s: %s", manifest_path, error)
            continue
        packages.append(
            FinalizedSession(
                session_id=str(manifest.get("session_id") or entry.name),
                directory=entry.path,
                manifest=manifest,
            )
        )
    return packages, delivered


def drain_spool(
    observer: Optional[VaaniObserver] = None,
    *,
    spool_directory: Optional[str] = None,
    timeout: Optional[float] = None,
    purge: bool = True,
    delivered_ttl_s: Optional[float] = DEFAULT_DELIVERED_TTL_S,
) -> DrainResult:
    """Upload every pending package once.

    Failures are recorded and the pass continues: one unshippable call must not
    strand the calls queued behind it. The package stays on the spool, so the
    next pass retries it.
    """
    vaani = observer or VaaniObserver.from_env()
    if vaani is None:
        raise ValueError(
            "No observer configured. Pass one, or set VAANI_ENDPOINT and VAANI_API_KEY."
        )
    directory = spool_directory or vaani.options["spool_directory"]
    endpoint = vaani.options.get("endpoint")
    packages, delivered = _scan(directory, endpoint)
    result = DrainResult(uploaded=[], failed=[], skipped=delivered)
    # A package uploaded in-process by `finish()` gets a receipt and is then
    # skipped by every later pass -- forever. At ~28 MB per 5-minute call that
    # fills the disk of any worker that records all day, which is every worker.
    # The docs promised this cleanup long before anything performed it.
    if purge and delivered_ttl_s is not None:
        result.purged = _purge_delivered(directory, delivered, delivered_ttl_s)
        result.skipped = [name for name in delivered if name not in set(result.purged)]
    for package in packages:
        try:
            response = vaani.upload_package_sync(package, timeout)
        except Exception as error:  # noqa: BLE001 - one bad package must not stop the rest
            logger.warning(
                "Upload failed for %s (kept at %s): %s",
                package.session_id,
                package.directory,
                error,
            )
            result.failed.append(package.session_id)
            continue
        _acknowledge(package, response, purge=purge, endpoint=endpoint)
        logger.info("Uploaded %s", package.session_id)
        result.uploaded.append(package.session_id)
    return result


def _acknowledge(
    package: FinalizedSession, response: Any, *, purge: bool, endpoint: Optional[str] = None
) -> None:
    """Make a delivered package idempotent for future passes.

    Once the backend has verified the digests the local copy is redundant, and
    a worker that records all day will fill its disk if nothing removes it -- a
    5-minute call is ~28 MB of raw PCM. Keeping it is still offered, because
    during a migration the local package is the only evidence available.
    """
    if purge:
        shutil.rmtree(package.directory, ignore_errors=True)
        return
    receipt = {
        "session_id": package.session_id,
        "endpoint": endpoint,
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response if isinstance(response, (dict, list)) else None,
    }
    try:
        with open(os.path.join(package.directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
            json.dump(receipt, handle)
    except OSError as error:  # pragma: no cover - disk-level failure
        # The upload succeeded; failing to write the receipt only risks a
        # harmless idempotent re-upload, so it must not be reported as a loss.
        logger.warning("Uploaded %s but could not write its receipt: %s",
                       package.session_id, error)


def _build_observer(args: Any) -> VaaniObserver:
    options: Dict[str, Any] = {"instrumentations": {"http": False, "websocket": False}}
    if args.endpoint:
        options["endpoint"] = args.endpoint
    if args.api_key:
        options["api_key"] = args.api_key
    if args.spool:
        options["spool_directory"] = args.spool
    if options.get("endpoint") and options.get("api_key"):
        # The drain exists to ship recordings a shutdown hook could not. Giving
        # it the default 30s cap on this path -- and not on the env path -- is
        # how the same package fails identically on every retry, forever.
        options["upload"] = upload_options_from_env()
        return VaaniObserver(**options)
    # Fall back to the environment, then re-apply any explicit overrides so a
    # flag always wins over a variable.
    vaani = VaaniObserver.from_env()
    if vaani is None:
        raise SystemExit(
            "No destination configured. Set VAANI_ENDPOINT and VAANI_API_KEY, "
            "or pass --endpoint and --api-key."
        )
    for key, value in options.items():
        if key != "instrumentations" and value:
            vaani.options[key] = value
    return vaani


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vaani_observer.drain",
        description="Upload finalized VaaniEval packages left on the local spool.",
    )
    parser.add_argument("--spool", default=os.environ.get("VAANI_SPOOL_DIR"),
                        help="Spool directory (default: VAANI_SPOOL_DIR or ./.vaani-spool)")
    parser.add_argument("--endpoint", default=None, help="Ingest endpoint (default: VAANI_ENDPOINT)")
    parser.add_argument("--api-key", default=None, help="API key (default: VAANI_API_KEY)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_PACKAGE_TIMEOUT_S,
                        help="Budget in seconds for each package upload "
                             f"(default {DEFAULT_PACKAGE_TIMEOUT_S:.0f}; 0 disables)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep delivered packages on disk instead of removing them")
    parser.add_argument(
        "--delivered-ttl", type=float, metavar="HOURS",
        default=DEFAULT_DELIVERED_TTL_S / 3600,
        help=(
            "Remove packages the backend already acknowledged once their receipt "
            "is this many hours old (default: 24). Use a negative value to keep "
            "them indefinitely. Ignored with --keep."
        ),
    )
    parser.add_argument("--watch", type=float, metavar="SECONDS", default=None,
                        help="Keep running, re-scanning the spool every SECONDS")
    parser.add_argument("--verbose", action="store_true", help="Log every package")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    observer = _build_observer(args)
    directory = args.spool or observer.options["spool_directory"]

    def once() -> DrainResult:
        result = drain_spool(
            observer,
            spool_directory=directory,
            timeout=args.timeout or None,
            purge=not args.keep,
            delivered_ttl_s=args.delivered_ttl * 3600,
        )
        print(f"{directory}: {result}")
        return result

    if args.watch is None:
        return 0 if once().ok else 1
    try:
        while True:
            once()
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
        return 0


def _purge_delivered(spool_directory: str, delivered: List[str], ttl_s: float) -> List[str]:
    """Remove acknowledged packages once they are older than `ttl_s`.

    Age is taken from the receipt, not the package, because the receipt is
    written when delivery was confirmed -- that is the point from which the
    local copy is redundant.
    """
    if ttl_s < 0:
        return []
    now = time.time()
    purged: List[str] = []
    for name in delivered:
        directory = os.path.join(spool_directory, name)
        age = _delivered_age(os.path.join(directory, RECEIPT_NAME), now)
        if age is None or age < ttl_s:
            continue
        try:
            shutil.rmtree(directory)
        except OSError as error:  # pragma: no cover - disk-level failure
            logger.warning("Could not remove delivered package %s: %s", directory, error)
            continue
        logger.info("Removed delivered package %s (%.1fh old)", name, age / 3600)
        purged.append(name)
    return purged


def _delivered_age(receipt: str, now: float) -> Optional[float]:
    """How long ago a package was delivered.

    The receipt's own `uploaded_at` is authoritative and `mtime` is only a
    fallback, because `mtime` is destroyed by anything that copies a spool:
    `cp -R`, `rsync` without `-a`, a container image build, a backup restore.
    Measured directly -- a `cp -R` of a spool whose receipts were 21 hours old
    produced receipts reading as 0 hours old, so nothing was ever old enough to
    remove and the disk filled anyway, which is the problem this exists to
    solve.

    Deleting here is not data loss: a receipt means the backend verified the
    digests, so the recording is already stored remotely and the local copy is
    redundant. The TTL is a forensics courtesy window, not the only copy.

    A timestamp in the future means a clock moved. That is reported as age zero
    -- keep the package -- rather than allowed to go negative.
    """
    try:
        with open(receipt, "r", encoding="utf-8") as handle:
            stamp = json.load(handle).get("uploaded_at")
        if isinstance(stamp, str):
            parsed = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
            return max(0.0, now - parsed)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    try:
        return max(0.0, now - os.path.getmtime(receipt))
    except OSError:
        return None


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
