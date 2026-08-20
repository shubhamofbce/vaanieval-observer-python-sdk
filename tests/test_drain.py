"""The spool drainer: shipping packages the recording process could not.

Recording and uploading are separate for a reason -- a LiveKit worker's default
`shutdown_process_timeout` is ten seconds and a five-minute call is ~28 MB of
raw PCM -- so a package left behind has to be recoverable out of band.

`drain.py` is a new module, so none of these tests can be run against a pre-fix
version of it: there is no pre-fix version. Their baseline is instead the
behaviour they each describe -- the purge and receipt-scoping tests were
verified by reverting the specific guard they exercise and confirming they fail
for the stated reason, which is the only honest check available here.
"""

from __future__ import annotations

import json
import os

import pytest

from conftest import PCM, RecordingTransport, empty_response, json_response
from vaani_observer import VaaniObserver
from vaani_observer.drain import (
    RECEIPT_NAME,
    EndpointChanged,
    drain_spool,
    pending_packages,
)
from vaani_observer.drain import main as drain_main

UPLOAD_URLS = {
    "events.jsonl": "https://objects.example.com/events",
    "call.audio": "https://objects.example.com/call",
}


def ok_handler(call, index):
    if call["method"] == "POST" and call["url"].endswith("/v1/sessions"):
        return json_response({"upload_urls": UPLOAD_URLS}, 201)
    return json_response({"status": "ready"})


@pytest.fixture
def spool(tmp_path):
    return str(tmp_path / "spool")


@pytest.fixture
def recording(spool):
    """Records `count` finished calls to a shared spool and returns its path."""

    async def record(count: int = 1):
        observer = VaaniObserver(
            spool_directory=spool, instrumentations={"http": False, "websocket": False}
        )
        for index in range(count):
            session = observer.start_session(session_id=f"call-{index}", agent_id="support")
            session.record_inbound_audio(b"\x01\x00", {**PCM, "timestamp_ms": 0})
            session.start_operation(type="llm").end()
            await session.end(outcome="completed")
        return spool

    return record


ENDPOINT = "https://ingest.example.com"


def uploader(spool, **options):
    options.setdefault("endpoint", ENDPOINT)
    options.setdefault("api_key", "test-key")
    options.setdefault("instrumentations", {"http": False, "websocket": False})
    options.setdefault("spool_directory", spool)
    options.setdefault("upload", {"retries": 0})
    return VaaniObserver(**options)


async def test_finds_every_finalized_package_on_the_spool(recording, spool):
    await recording(3)
    found = pending_packages(spool)
    assert [package.session_id for package in found] == ["call-0", "call-1", "call-2"]


def test_an_empty_or_missing_spool_is_not_an_error(tmp_path):
    assert pending_packages(str(tmp_path / "nothing-here")) == []
    assert pending_packages(str(tmp_path)) == []


async def test_ignores_a_directory_whose_manifest_has_not_been_published(recording, spool):
    """A call still being written must never be uploaded half-formed."""
    await recording(1)
    os.remove(os.path.join(spool, "call-0", "manifest.json"))
    assert pending_packages(spool) == []


async def test_uploads_and_then_removes_a_delivered_package(recording, spool):
    await recording(2)
    observer = uploader(spool)
    RecordingTransport(ok_handler).install(observer)

    result = drain_spool(observer, spool_directory=spool)

    assert result.uploaded == ["call-0", "call-1"]
    assert result.failed == []
    # A worker recording all day fills its disk if nothing reclaims it.
    assert pending_packages(spool) == []
    assert not os.path.exists(os.path.join(spool, "call-0"))


async def test_keeping_a_package_records_a_receipt_so_it_is_not_resent(recording, spool):
    await recording(1)
    observer = uploader(spool)
    RecordingTransport(ok_handler).install(observer)

    drain_spool(observer, spool_directory=spool, purge=False)

    receipt_path = os.path.join(spool, "call-0", RECEIPT_NAME)
    assert os.path.exists(receipt_path)
    with open(receipt_path, "r", encoding="utf-8") as handle:
        assert json.load(handle)["session_id"] == "call-0"
    assert pending_packages(spool) == [], "a delivered package must not be re-uploaded"


async def test_a_failed_package_is_kept_for_the_next_pass(recording, spool):
    await recording(1)
    observer = uploader(spool)
    RecordingTransport(lambda call, index: empty_response(500)).install(observer)

    result = drain_spool(observer, spool_directory=spool)

    assert result.failed == ["call-0"]
    assert result.ok is False
    assert [package.session_id for package in pending_packages(spool)] == ["call-0"]


async def test_one_unshippable_package_does_not_strand_the_others(recording, spool):
    await recording(3)
    observer = uploader(spool)

    def handler(call, index):
        if "call-1" in json.dumps(call["body"].decode("utf-8", "replace")):
            return empty_response(500)
        return ok_handler(call, index)

    RecordingTransport(handler).install(observer)
    result = drain_spool(observer, spool_directory=spool)

    assert sorted(result.uploaded) == ["call-0", "call-2"]
    assert result.failed == ["call-1"]


async def test_a_second_pass_over_a_drained_spool_does_nothing(recording, spool):
    await recording(1)
    observer = uploader(spool)
    transport = RecordingTransport(ok_handler).install(observer)

    drain_spool(observer, spool_directory=spool)
    calls_after_first = len(transport.calls)
    result = drain_spool(observer, spool_directory=spool)

    assert result.uploaded == []
    assert len(transport.calls) == calls_after_first


def test_refuses_to_run_without_a_destination(monkeypatch, spool):
    monkeypatch.delenv("VAANI_ENDPOINT", raising=False)
    monkeypatch.delenv("VAANI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VAANI_ENDPOINT"):
        drain_spool(spool_directory=spool)


def test_the_summary_distinguishes_an_empty_spool_from_a_delivered_one(tmp_path):
    """"0 uploaded, 0 failed" alone cannot tell an operator that nothing is stranded."""
    import json
    import os

    from vaani_observer.drain import RECEIPT_NAME, drain_spool

    spool = tmp_path / "spool"
    for name in ("aaa", "bbb"):
        directory = spool / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"session_id": name}))
        (directory / RECEIPT_NAME).write_text(json.dumps({"session_id": name, "endpoint": ENDPOINT}))

    class Observer:
        options = {"spool_directory": str(spool)}

        def upload_package_sync(self, package, timeout=None):  # pragma: no cover
            raise AssertionError("a delivered package must never be re-uploaded")

    result = drain_spool(Observer(), purge=False)

    assert result.uploaded == []
    assert result.failed == []
    assert sorted(result.skipped) == ["aaa", "bbb"]
    assert "2 already delivered" in str(result)

    empty = drain_spool(Observer(), spool_directory=str(tmp_path / "nothing-here"))
    assert empty.skipped == []
    assert "0 already delivered" in str(empty)


# ------------------------------------------------ purging delivered packages


async def test_a_package_delivered_in_process_is_eventually_removed(recording, spool):
    """`finish()` uploads and leaves a receipt; nothing ever removed it.

    Every later pass classified it as "already delivered" and skipped it, so a
    worker recording all day accumulated ~28 MB per five-minute call on disk,
    permanently -- while the docs promised the drain cleaned up.
    """
    import time

    await recording(1)
    directory = os.path.join(spool, os.listdir(spool)[0])
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": ENDPOINT}, handle)
    old = time.time() - 48 * 3600
    os.utime(os.path.join(directory, RECEIPT_NAME), (old, old))

    result = drain_spool(uploader(spool), spool_directory=spool, delivered_ttl_s=3600)

    assert result.purged == [os.path.basename(directory)]
    assert result.skipped == []
    assert not os.path.exists(directory)


async def test_a_freshly_delivered_package_is_kept_for_the_ttl(recording, spool):
    """A receipt means the bytes were accepted, not that anyone has looked."""
    await recording(1)
    directory = os.path.join(spool, os.listdir(spool)[0])
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": ENDPOINT}, handle)

    result = drain_spool(uploader(spool), spool_directory=spool, delivered_ttl_s=24 * 3600)

    assert result.purged == []
    assert result.skipped == [os.path.basename(directory)]
    assert os.path.exists(directory)


async def test_delivered_packages_can_be_kept_indefinitely(recording, spool):
    import time

    await recording(1)
    directory = os.path.join(spool, os.listdir(spool)[0])
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": ENDPOINT}, handle)
    old = time.time() - 90 * 24 * 3600
    os.utime(os.path.join(directory, RECEIPT_NAME), (old, old))

    result = drain_spool(uploader(spool), spool_directory=spool, delivered_ttl_s=None)

    assert result.purged == []
    assert os.path.exists(directory)


# ------------------------------------------------------------- the CLI itself


async def test_the_documented_command_actually_runs(recording, spool, tmp_path):
    """`python -m vaani_observer.drain` is the documented recovery path.

    Every other test in this file imports the module and calls `drain_spool`
    directly, which executes the whole module body first. Running it as
    `__main__` does not: `main()` is invoked from the `if __name__` guard, so
    anything defined *below* that guard does not exist yet when it runs. A
    helper added at the end of the file therefore crashed the CLI with
    `NameError` while all 13 unit tests stayed green -- and the CLI is
    precisely the thing an operator runs when a recording is already at risk.
    """
    import subprocess
    import sys

    await recording(1)
    result = subprocess.run(
        [sys.executable, "-m", "vaani_observer.drain",
         "--spool", spool, "--endpoint", "http://127.0.0.1:9", "--api-key", "k",
         "--verbose"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")},
    )

    # The endpoint is unreachable on purpose, so a non-zero exit is expected.
    # What must NOT happen is the module failing before it does any work.
    assert "NameError" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert "Upload failed" in result.stderr or "Upload failed" in result.stdout


async def test_the_cli_help_lists_every_flag_main_reads(spool):
    """A flag `main()` reads but `argparse` never defined is an AttributeError."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "vaani_observer.drain", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")},
    )

    assert result.returncode == 0, result.stderr
    for flag in ("--spool", "--endpoint", "--api-key", "--timeout",
                 "--keep", "--delivered-ttl", "--watch", "--verbose"):
        assert flag in result.stdout, f"{flag} missing from --help"


async def test_a_copied_spool_does_not_lose_its_delivery_clock(recording, spool):
    """`cp -R` resets receipt mtime, so nothing ever looked old enough to remove.

    Measured on a real spool: receipts 21 hours old read as 0 hours old after a
    copy, so the disk filled anyway -- the exact problem the purge exists to
    solve. The receipt's recorded `uploaded_at` survives a copy, so it is the
    authoritative signal and mtime is only a fallback.
    """
    import shutil
    import time

    await recording(1)
    name = os.listdir(spool)[0]
    long_ago = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 90 * 3600))
    with open(os.path.join(spool, name, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": ENDPOINT, "uploaded_at": long_ago}, handle)

    copied = spool + "-copy"
    shutil.copytree(spool, copied)
    os.utime(os.path.join(copied, name, RECEIPT_NAME), None)  # mtime says "just now"

    result = drain_spool(uploader(copied), spool_directory=copied, delivered_ttl_s=3600)

    assert result.purged == [name], "the receipt's own timestamp must win over mtime"
    assert not os.path.exists(os.path.join(copied, name))


async def test_a_receipt_without_a_timestamp_falls_back_to_mtime(recording, spool):
    """Receipts written by an older SDK have no `uploaded_at`."""
    import time

    await recording(1)
    name = os.listdir(spool)[0]
    with open(os.path.join(spool, name, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": ENDPOINT}, handle)
    old = time.time() - 90 * 3600
    os.utime(os.path.join(spool, name, RECEIPT_NAME), (old, old))

    result = drain_spool(uploader(spool), spool_directory=spool, delivered_ttl_s=3600)

    assert result.purged == [name]


async def test_a_receipt_stamped_in_the_future_is_not_treated_as_ancient(recording, spool):
    """A skewed clock must not delete a package before its window elapses."""
    import time

    await recording(1)
    name = os.listdir(spool)[0]
    future = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 86400))
    with open(os.path.join(spool, name, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": ENDPOINT, "uploaded_at": future}, handle)

    result = drain_spool(uploader(spool), spool_directory=spool, delivered_ttl_s=3600)

    assert result.purged == []
    assert os.path.exists(os.path.join(spool, name))


async def test_a_receipt_from_a_different_backend_does_not_count_as_delivered(recording, spool):
    """Repointing VAANI_ENDPOINT must not silently strand -- or delete -- calls.

    A receipt only ever meant "somebody verified the digests". Once a TTL purge
    exists, honouring an old backend's receipt deletes the package from the
    only machine holding it, during exactly the migration the retention window
    is supposed to protect.
    """
    await recording(1)
    delivered = uploader(spool, endpoint="https://staging.example.com")
    RecordingTransport(ok_handler).install(delivered)
    drain_spool(delivered, spool_directory=spool, purge=False)

    moved = uploader(spool, endpoint="https://prod.example.com")
    transport = RecordingTransport(ok_handler)
    transport.install(moved)
    result = drain_spool(
        moved,
        spool_directory=spool,
        purge=False,
        delivered_ttl_s=0,
        allow_endpoint_change=True,
    )

    assert result.uploaded == ["call-0"], "the new backend never received this call"
    assert result.purged == [], "a package the new backend lacks must never be purged"
    assert os.path.exists(os.path.join(spool, "call-0"))
    with open(os.path.join(spool, "call-0", RECEIPT_NAME), encoding="utf-8") as handle:
        assert json.load(handle)["endpoint"] == "https://prod.example.com"


async def test_a_receipt_from_the_same_backend_is_still_honoured(recording, spool):
    """The endpoint check must not cause a re-upload storm on every pass."""
    await recording(1)
    observer = uploader(spool, endpoint="https://ingest.example.com/")
    RecordingTransport(ok_handler).install(observer)
    drain_spool(observer, spool_directory=spool, purge=False)

    again = uploader(spool, endpoint="https://ingest.example.com")
    transport = RecordingTransport(ok_handler)
    transport.install(again)
    result = drain_spool(again, spool_directory=spool, purge=False)

    assert result.uploaded == [], "a trailing slash is the same destination"
    assert result.skipped == ["call-0"]
    assert transport.calls == []


async def test_a_receipt_written_before_endpoints_were_recorded_is_honoured(recording, spool):
    """An existing spool must not re-upload wholesale on upgrade."""
    await recording(1)
    directory = os.path.join(spool, "call-0")
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "uploaded_at": "2020-01-01T00:00:00Z"}, handle)

    observer = uploader(spool, endpoint="https://ingest.example.com")
    transport = RecordingTransport(ok_handler)
    transport.install(observer)
    result = drain_spool(observer, spool_directory=spool, purge=False)

    assert result.uploaded == []
    assert result.skipped == ["call-0"]


async def test_an_unreadable_receipt_is_re_uploaded_rather_than_assumed_delivered(recording, spool):
    """Re-uploading is idempotent; assuming delivery is not recoverable."""
    await recording(1)
    directory = os.path.join(spool, "call-0")
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        handle.write("{truncated")

    observer = uploader(spool, endpoint="https://ingest.example.com")
    RecordingTransport(ok_handler).install(observer)
    result = drain_spool(observer, spool_directory=spool, purge=False, delivered_ttl_s=0)

    assert result.uploaded == ["call-0"]
    assert result.purged == []


async def test_the_purge_never_deletes_a_call_the_current_backend_lacks(recording, spool):
    """The actual data-loss case, with the purge armed.

    Pre-fix this deleted the package outright: the old backend's receipt made
    it "delivered", an expired TTL made it purgeable, and the call was gone
    from the only machine that had it without ever reaching the new backend.
    """
    await recording(1)
    staging = uploader(spool, endpoint="https://staging.example.com")
    RecordingTransport(ok_handler).install(staging)
    drain_spool(staging, spool_directory=spool, purge=False)
    assert os.path.exists(os.path.join(spool, "call-0"))

    moved = uploader(spool, endpoint="https://prod.example.com")
    transport = RecordingTransport(ok_handler)
    transport.install(moved)
    result = drain_spool(
        moved,
        spool_directory=spool,
        purge=True,
        delivered_ttl_s=0,
        allow_endpoint_change=True,
    )

    assert result.purged == [], "nothing may be purged before prod holds it"
    assert result.uploaded == ["call-0"], "prod must actually receive the call"
    posted = [c["url"] for c in transport.calls if c["method"] == "POST"]
    assert any("prod.example.com" in url for url in posted), posted


async def test_the_purge_spares_a_receipt_that_names_no_backend(recording, spool):
    """The one cell of the matrix that loses data: no endpoint, purge armed.

    Skipping and deleting need opposite defaults. An endpoint-less receipt is
    honoured for skipping, so a pre-existing spool does not re-upload
    wholesale -- but it is not evidence that the backend now being written to
    holds the call, so it must never justify deleting it.
    """
    await recording(1)
    directory = os.path.join(spool, "call-0")
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "uploaded_at": "2020-01-01T00:00:00Z"}, handle)

    observer = uploader(spool, endpoint="https://prod.example.com")
    RecordingTransport(ok_handler).install(observer)
    result = drain_spool(observer, spool_directory=spool, purge=True, delivered_ttl_s=0)

    assert result.purged == [], "an unprovenanced receipt is not evidence of delivery"
    assert result.skipped == ["call-0"], "but it still suppresses a re-upload"
    assert os.path.exists(directory), "the only copy of this call must survive"


async def test_a_receipt_recording_a_null_endpoint_is_not_evidence_either(recording, spool):
    """`endpoint: null` must not read as "delivered to wherever you are now"."""
    await recording(1)
    directory = os.path.join(spool, "call-0")
    with open(os.path.join(directory, RECEIPT_NAME), "w", encoding="utf-8") as handle:
        json.dump({"session_id": "call-0", "endpoint": None,
                   "uploaded_at": "2020-01-01T00:00:00Z"}, handle)

    observer = uploader(spool, endpoint="https://prod.example.com")
    RecordingTransport(ok_handler).install(observer)
    result = drain_spool(observer, spool_directory=spool, purge=True, delivered_ttl_s=0)

    assert result.purged == []
    assert os.path.exists(directory)


async def test_a_delivered_package_is_still_purged_once_its_backend_matches(recording, spool):
    """The stricter purge must not disable retention for the normal case."""
    await recording(1)
    observer = uploader(spool, endpoint="https://ingest.example.com")
    RecordingTransport(ok_handler).install(observer)
    drain_spool(observer, spool_directory=spool, purge=False)

    again = uploader(spool, endpoint="https://ingest.example.com")
    RecordingTransport(ok_handler).install(again)
    result = drain_spool(again, spool_directory=spool, purge=True, delivered_ttl_s=0)

    assert result.purged == ["call-0"]
    assert not os.path.exists(os.path.join(spool, "call-0"))


def test_a_pass_ships_the_longest_waiting_call_first(tmp_path):
    """Session ids are UUIDs, so sorting by directory name is arbitrary.

    On an ephemeral host the container can die partway through a backlog, so
    the order decides which recordings survive.
    """
    spool = tmp_path / "spool"
    for name, started in [
        ("zzz", "2026-01-01T00:00:00.000Z"),
        ("aaa", "2026-03-01T00:00:00.000Z"),
        ("mmm", "2026-02-01T00:00:00.000Z"),
        ("nnn", None),
    ]:
        directory = spool / name
        directory.mkdir(parents=True)
        manifest = {"session_id": name}
        if started:
            manifest["started_at"] = started
        (directory / "manifest.json").write_text(json.dumps(manifest))

    order = [package.session_id for package in pending_packages(str(spool))]

    assert order[:3] == ["zzz", "mmm", "aaa"], "oldest call first, not lowest name"
    assert order[3] == "nnn", "an undated package sorts last, but is never dropped"


# --- Re-pointing the drain must not silently exfiltrate raw audio ------------
#
# A package holds the caller's voice and whatever they said. Delivery is what
# authorises deleting the local copy, so a typo'd VAANI_ENDPOINT ships that to
# an arbitrary host *and* destroys the only other copy, reporting success.


async def test_refuses_to_drain_to_an_endpoint_the_spool_has_never_used(recording, spool):
    await recording(2)
    known = uploader(spool)
    RecordingTransport(ok_handler).install(known)
    drain_spool(known, spool_directory=spool, purge=False)

    await recording(1)
    typo = uploader(spool, endpoint="https://ingest.exmaple.com")
    transport = RecordingTransport(ok_handler)
    transport.install(typo)

    with pytest.raises(EndpointChanged) as caught:
        drain_spool(typo, spool_directory=spool)

    assert transport.calls == [], "nothing may leave before the operator confirms"
    message = str(caught.value)
    assert ENDPOINT in message and "ingest.exmaple.com" in message
    assert "--yes" in message, "the refusal has to name its own remedy"
    assert os.path.isdir(os.path.join(spool, "call-0"))


async def test_a_confirmed_endpoint_change_uploads_but_keeps_the_local_copies(
    recording, spool
):
    """`--yes` authorises the disclosure, not the deletion.

    The operator has had no chance to check that anything arrived at a host
    this spool has never shipped to, so a 2xx from it is not yet evidence
    worth destroying the only local copy on.
    """
    await recording(1)
    known = uploader(spool)
    RecordingTransport(ok_handler).install(known)
    drain_spool(known, spool_directory=spool, purge=False)

    await recording(2)
    moved = uploader(spool, endpoint="https://ingest-2.example.com")
    RecordingTransport(ok_handler).install(moved)

    result = drain_spool(moved, spool_directory=spool, allow_endpoint_change=True)

    assert sorted(result.uploaded) == ["call-0", "call-1"]
    for name in result.uploaded:
        assert os.path.isdir(os.path.join(spool, name)), "local evidence kept"


async def test_the_second_pass_to_a_now_known_endpoint_is_unguarded(recording, spool):
    """A migration must not need the flag forever, or it will be aliased away."""
    await recording(1)
    moved = uploader(spool, endpoint="https://ingest-2.example.com")
    RecordingTransport(ok_handler).install(moved)
    drain_spool(moved, spool_directory=spool, allow_endpoint_change=True)

    await recording(2)
    again = uploader(spool, endpoint="https://ingest-2.example.com")
    RecordingTransport(ok_handler).install(again)

    result = drain_spool(again, spool_directory=spool)

    assert sorted(result.uploaded) == ["call-0", "call-1"]


async def test_a_trailing_slash_is_not_a_different_destination(recording, spool):
    """A guard that cries wolf is a guard that gets disabled."""
    await recording(1)
    first = uploader(spool)
    RecordingTransport(ok_handler).install(first)
    drain_spool(first, spool_directory=spool, purge=False)

    await recording(2)
    same = uploader(spool, endpoint=ENDPOINT + "/")
    RecordingTransport(ok_handler).install(same)

    # call-0 already carries a receipt for the same host written without the
    # slash, so the pass must both proceed unguarded and honour that receipt.
    result = drain_spool(same, spool_directory=spool)

    assert result.uploaded == ["call-1"]
    assert result.skipped == ["call-0"]


async def test_a_first_ever_drain_is_not_blocked(recording, spool):
    """With no prior destination there is no *change* to object to.

    Paired with the control below, which uses the same fixture and differs only
    in having a prior destination on record: without that pair this asserts
    nothing, because a spool that was never drained has no ledger either way.
    """
    await recording(1)
    assert not os.path.exists(os.path.join(spool, ".vaani-destinations"))
    observer = uploader(spool)
    RecordingTransport(ok_handler).install(observer)

    assert drain_spool(observer, spool_directory=spool).uploaded == ["call-0"]


async def test_the_same_spool_is_blocked_once_it_has_a_different_destination(
    recording, spool
):
    """The control for the test above: identical, plus one line of history."""
    await recording(1)
    with open(os.path.join(spool, ".vaani-destinations"), "w", encoding="utf-8") as handle:
        handle.write("https://somewhere.else.example\n")
    observer = uploader(spool)
    RecordingTransport(ok_handler).install(observer)

    with pytest.raises(EndpointChanged):
        drain_spool(observer, spool_directory=spool)


async def test_starting_the_agent_against_a_new_endpoint_does_not_authorise_the_drain(
    recording, spool
):
    """Re-pointing and restarting is how an operator changes destination.

    It is also exactly what a typo does. `VAANI_ENDPOINT` is normally set once
    for both the agent and the drainer, so if the recorder's own startup
    preflight writes the new host into the ledger, the guard is disarmed by the
    very action it exists to question -- and the raw audio ships to the typo'd
    host and is deleted locally, which is the finding verbatim.
    """
    await recording(1)
    with open(os.path.join(spool, ".vaani-destinations"), "w", encoding="utf-8") as handle:
        handle.write("https://ingest.example.com\n")
    # The agent is restarted against the new endpoint and writes a call.
    restarted = VaaniObserver(
        endpoint="https://ingest.exmaple.com",
        api_key="test-key",
        spool_directory=spool,
        instrumentations={"http": False, "websocket": False},
    )
    restarted.preflight()

    observer = uploader(spool, endpoint="https://ingest.exmaple.com")
    RecordingTransport(ok_handler).install(observer)
    with pytest.raises(EndpointChanged):
        drain_spool(observer, spool_directory=spool)


async def test_a_confirmed_migration_is_never_purged_by_the_ttl_on_a_later_pass(
    recording, spool
):
    """`--yes` keeps the local copy; a day later that must still be true.

    The confirmed pass writes a receipt naming the new host, and from the next
    pass onwards the destination is no longer "changed" -- so nothing asks the
    question again and the TTL quietly deletes the only local copy of calls
    nobody has yet confirmed arrived anywhere.
    """
    await recording(1)
    with open(os.path.join(spool, ".vaani-destinations"), "w", encoding="utf-8") as handle:
        handle.write("https://ingest.example.com\n")
    observer = uploader(spool, endpoint="https://ingest.exmaple.com")
    RecordingTransport(ok_handler).install(observer)
    drain_spool(observer, spool_directory=spool, allow_endpoint_change=True)
    package = os.path.join(spool, "call-0")
    assert os.path.exists(package), "a confirmed migration keeps the local copy"

    # A later, unremarkable pass to the same host, with the receipt long past
    # the TTL. Nothing prompts the operator again.
    receipt = os.path.join(package, RECEIPT_NAME)
    old = json.loads(open(receipt, encoding="utf-8").read())
    assert old.get("migration") is True
    # Age is read from the receipt's own `uploaded_at`, so a day has to pass
    # there rather than on the filesystem.
    old["uploaded_at"] = "2020-01-01T00:00:00Z"
    with open(receipt, "w", encoding="utf-8") as handle:
        json.dump(old, handle)
    observer2 = uploader(spool, endpoint="https://ingest.exmaple.com")
    RecordingTransport(ok_handler).install(observer2)
    result = drain_spool(observer2, spool_directory=spool, delivered_ttl_s=1)
    assert result.purged == [], "an unconfirmed migration must not be deleted by a timer"
    assert os.path.exists(package)


def test_a_change_of_host_case_is_not_a_change_of_destination(tmp_path, monkeypatch):
    """A guard that cries wolf is a guard that gets disabled.

    Scheme and host are case-insensitive per RFC 3986, and a hostname
    templated from a different source routinely differs only in case.
    """
    spool = tmp_path / "spool"
    package = spool / "call-0"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({"session_id": "call-0"}))
    (spool / ".vaani-destinations").write_text("http://Localhost:8000\n")
    observer = uploader(str(spool), endpoint="http://localhost:8000/")
    RecordingTransport(ok_handler).install(observer)

    drain_spool(observer, spool_directory=str(spool))


def test_a_refusal_does_not_kill_a_watching_sidecar(tmp_path, monkeypatch):
    """Exiting on refusal makes the guard cause the loss it exists to prevent.

    A supervised sidecar would re-exit forever, nothing would ever drain, and
    the spool would fill unboundedly. A refusal means "not shipping yet".
    """
    spool = tmp_path / "spool"
    package = spool / "call-0"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({"session_id": "call-0"}))
    (spool / ".vaani-destinations").write_text("https://ingest.example.com\n")
    monkeypatch.setenv("VAANI_ENDPOINT", "https://ingest.exmaple.com")
    monkeypatch.setenv("VAANI_API_KEY", "test-key")

    slept: list[float] = []

    def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) >= 2:
            raise KeyboardInterrupt
        return None

    monkeypatch.setattr("vaani_observer.drain.time.sleep", fake_sleep)
    assert drain_main(["--spool", str(spool), "--watch", "1"]) == 0
    assert len(slept) == 2, "the loop must keep running after a refusal"


def test_the_cli_refuses_a_changed_endpoint_with_a_nonzero_status(tmp_path, monkeypatch):
    """A supervisor must not read a refused exfiltration as a clean pass."""
    spool = tmp_path / "spool"
    package = spool / "call-0"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({"session_id": "call-0"}))
    (spool / ".vaani-destinations").write_text("https://ingest.example.com\n")
    monkeypatch.setenv("VAANI_ENDPOINT", "https://ingest.exmaple.com")
    monkeypatch.setenv("VAANI_API_KEY", "test-key")

    assert drain_main(["--spool", str(spool)]) == 2
    assert (package / "manifest.json").exists()


def test_the_drainer_never_patches_the_host_processs_http_clients(tmp_path, monkeypatch):
    """A drainer's only outbound traffic is the upload.

    `from_env` turns the HTTP and WebSocket instrumentation on by default, so
    the env-configured CLI path used to monkeypatch every client in whatever
    process ran it -- a sidecar patching itself to observe its own uploads,
    and, because the patch is global and refcounted, one that outlives the
    pass.
    """
    import httpx

    from vaani_observer.drain import _build_observer

    original = httpx.AsyncClient.send
    monkeypatch.setenv("VAANI_ENDPOINT", "https://ingest.example.com")
    monkeypatch.setenv("VAANI_API_KEY", "test-key")

    class Args:
        endpoint = None
        api_key = None
        spool = str(tmp_path / "spool")

    observer = _build_observer(Args())

    assert httpx.AsyncClient.send is original
    assert observer.options["instrumentations"] == {"http": False, "websocket": False}
