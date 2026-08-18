"""Observer construction, defaults and endpoint-rule validation.

Mirrors nodejs-sdk/test/observer-options.test.js.
"""

from __future__ import annotations

import os

import pytest

from vaani_observer import VaaniObserver
from vaani_observer.observer import ParsedUrl


def observer(**options):
    options.setdefault("instrumentations", {"http": False})
    return VaaniObserver(**options)


def test_applies_documented_defaults():
    vaani = observer()
    assert vaani.options["endpoint"] is None
    assert vaani.options["api_key"] is None
    assert vaani.options["spool_directory"] == os.path.join(os.getcwd(), ".vaani-spool")
    assert vaani.options["capture"] == {
        "audio": True,
        "http_bodies": False,
        "websocket_text_frames": False,
        "stt_content": False,
        "payload_max_bytes": 16 * 1024,
    }
    assert vaani.options["endpoints"] == []
    assert vaani.options["upload"] == {
        "retries": 3,
        "timeout_s": 30.0,
        "compress": True,
    }
    assert vaani.options["strict"] is False
    assert vaani.endpoint_rules == []


def test_merges_partial_capture_and_instrumentation_options():
    vaani = observer(capture={"audio": False}, instrumentations={"http": False, "websocket": False})
    assert vaani.options["capture"] == {
        "audio": False,
        "http_bodies": False,
        "websocket_text_frames": False,
        "stt_content": False,
        "payload_max_bytes": 16 * 1024,
    }
    assert vaani.options["instrumentations"] == {"http": False, "websocket": False}


def test_websocket_instrumentation_stays_enabled_when_only_http_is_disabled():
    assert observer().options["instrumentations"]["websocket"] is True


def test_merges_partial_upload_options_with_the_default_retry_count():
    assert observer(upload={"retries": 0}).options["upload"] == {
        "retries": 0,
        "timeout_s": 30.0,
        "compress": True,
    }
    assert observer(upload={"timeout_s": 10}).options["upload"] == {
        "retries": 3,
        "timeout_s": 10,
        "compress": True,
    }


def test_treats_a_nullish_endpoints_option_as_an_empty_rule_set():
    assert observer(endpoints=None).endpoint_rules == []
    assert observer().endpoint_rules == []


def test_normalizes_endpoint_rules_with_a_parsed_url_and_default_path_match():
    vaani = observer(endpoints=[{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    rule = vaani.endpoint_rules[0]
    assert rule["id"] == "llm"
    assert rule["match"] == "path"
    assert isinstance(rule["url"], ParsedUrl)
    assert rule["url"].href == "https://api.example.com/v1"


def test_preserves_an_explicit_match_strategy_and_extra_rule_fields():
    rule = observer(
        endpoints=[
            {"id": "tts", "type": "tts", "url": "https://tts.example.com/", "match": "origin", "provider": "acme"}
        ]
    ).endpoint_rules[0]
    assert rule["match"] == "origin"
    assert rule["provider"] == "acme"


@pytest.mark.parametrize(
    "endpoints",
    [
        [{"type": "llm", "url": "https://a.example.com"}],
        [{"id": "a", "url": "https://a.example.com"}],
        [{"id": "a", "type": "llm"}],
        [None],
        [{"id": "", "type": "llm", "url": "https://a.example.com"}],
    ],
)
def test_rejects_endpoint_rules_missing_required_fields(endpoints):
    with pytest.raises(TypeError):
        observer(endpoints=endpoints)


@pytest.mark.parametrize("rule_type", ["http", "STT", "vad", "", 1])
def test_rejects_endpoint_types_outside_stt_llm_and_tts(rule_type):
    with pytest.raises(TypeError):
        observer(endpoints=[{"id": "a", "type": rule_type, "url": "https://a.example.com"}])


def test_accepts_each_supported_endpoint_type():
    vaani = observer(
        endpoints=[
            {"id": "a", "type": "stt", "url": "https://a.example.com"},
            {"id": "b", "type": "llm", "url": "https://b.example.com"},
            {"id": "c", "type": "tts", "url": "https://c.example.com"},
        ]
    )
    assert [rule["type"] for rule in vaani.endpoint_rules] == ["stt", "llm", "tts"]


def test_rejects_duplicate_endpoint_ids():
    with pytest.raises(TypeError, match="Duplicate endpoint id: same"):
        observer(
            endpoints=[
                {"id": "same", "type": "llm", "url": "https://a.example.com"},
                {"id": "same", "type": "tts", "url": "https://b.example.com"},
            ]
        )


def test_rejects_an_endpoint_url_that_cannot_be_parsed():
    with pytest.raises(TypeError):
        observer(endpoints=[{"id": "a", "type": "llm", "url": "not-a-url"}])


def test_does_not_mutate_the_caller_supplied_endpoint_objects():
    given = {"id": "a", "type": "llm", "url": "https://a.example.com/v1"}
    observer(endpoints=[given])
    assert given["url"] == "https://a.example.com/v1"
    assert "match" not in given


def test_rule_for_resolves_a_configured_endpoint_by_id():
    vaani = observer(endpoints=[{"id": "llm", "type": "llm", "url": "https://a.example.com/v1"}])
    assert vaani.rule_for("llm")["id"] == "llm"
    assert vaani.rule_for("missing") is None
