"""Off-thread spool writer.

Recording must never block the media path. The Node SDK gets that for free by
chaining `fs/promises` appends onto a promise queue; Python's equivalent is a
single dedicated writer thread per session fed by a bounded queue.

One thread (rather than an executor pool) is deliberate: it preserves the strict
append ordering that `events.jsonl` and the raw audio tracks both depend on,
while keeping every filesystem syscall off the asyncio event loop.

The queue is bounded on purpose. A stalled disk behind an unbounded queue turns
a recording problem into an out-of-memory crash of the voice agent, so past the
high-water mark the writer drops chunks and counts them instead.
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
import weakref
from typing import Callable, Dict, Optional

_STOP = object()

#: Roughly 40 s of 20 ms audio frames plus their events on both tracks. Far more
#: headroom than a healthy disk ever needs, and still a bounded amount of RAM.
DEFAULT_MAX_QUEUED_WRITES = 8192

#: Every live writer, so an interpreter shutdown still flushes what it can.
_LIVE_WRITERS: "weakref.WeakSet" = weakref.WeakSet()


class SpoolWriter:
    """Serialises appends to a session directory on a private thread."""

    def __init__(
        self,
        directory: str,
        on_error: Callable[[str, BaseException], None],
        on_drop: Callable[[str], None],
        strict: bool = False,
        max_queued_writes: int = DEFAULT_MAX_QUEUED_WRITES,
    ) -> None:
        self.directory = directory
        self._on_error = on_error
        self._on_drop = on_drop
        self._strict = strict
        self._queue: "queue.Queue" = queue.Queue(maxsize=max_queued_writes)
        self._handles: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._accepting = True
        self.ready_error: Optional[BaseException] = None
        self.first_error: Optional[BaseException] = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"vaani-spool-{os.path.basename(directory)}", daemon=True
        )
        self._thread.start()
        _LIVE_WRITERS.add(self)

    # ------------------------------------------------------------------ api

    def submit(self, filename: str, data: bytes) -> bool:
        """Queue an append. Returns immediately; failures surface via callbacks.

        The accept decision and the enqueue happen under one lock, so a
        concurrent `close()` can never leave an accepted write stranded behind
        the stop sentinel.
        """
        with self._lock:
            if not self._accepting:
                return False
            try:
                self._queue.put_nowait((filename, bytes(data)))
            except queue.Full:
                self._on_drop(filename)
                return False
        return True

    def wait_ready(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        """Block until the spool directory exists (or failed to be created)."""
        self._ready.wait(timeout)
        return self.ready_error

    def close(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        """Drain the queue, close handles, and return the first error seen."""
        with self._lock:
            if self._accepting:
                self._accepting = False
                # A blocking put: the sentinel must never be dropped, and by
                # this point no further write can be accepted ahead of it.
                self._queue.put(_STOP)
        self._thread.join(timeout)
        _LIVE_WRITERS.discard(self)
        return self.first_error

    # --------------------------------------------------------------- thread

    def _run(self) -> None:
        try:
            os.makedirs(self.directory, exist_ok=True)
        except BaseException as error:  # noqa: BLE001 - recorded, never raised here
            self.ready_error = error
        finally:
            self._ready.set()

        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            filename, data = item
            self._append(filename, data)
        for handle in self._handles.values():
            try:
                handle.close()  # type: ignore[attr-defined]
            except BaseException:  # noqa: BLE001 - nothing useful is left to do
                pass
        self._handles.clear()

    def _append(self, filename: str, data: bytes) -> None:
        # A strict session stops writing after its first failure, matching the
        # Node chain, where a rejected write short-circuits everything queued
        # behind it.
        if self._strict and self.first_error is not None:
            return
        if self.ready_error is not None:
            self._fail(filename, self.ready_error)
            return
        try:
            handle = self._handles.get(filename)
            if handle is None:
                handle = open(os.path.join(self.directory, filename), "ab")
                self._handles[filename] = handle
            handle.write(data)  # type: ignore[attr-defined]
            handle.flush()  # type: ignore[attr-defined]
        except BaseException as error:  # noqa: BLE001 - degradation, not a crash
            self._fail(filename, error)

    def _fail(self, filename: str, error: BaseException) -> None:
        if self.first_error is None:
            self.first_error = error
        self._on_error(filename, error)


@atexit.register
def _drain_live_writers() -> None:
    """Best-effort flush for a process that exits with sessions still open.

    An unfinalized session has no manifest and is not a valid package, but the
    bytes already recorded are still worth keeping for a post-mortem.
    """
    for writer in list(_LIVE_WRITERS):
        try:
            writer.close(timeout=2.0)
        except BaseException:  # noqa: BLE001 - shutdown must not raise
            pass
