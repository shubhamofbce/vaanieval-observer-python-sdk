"""URL to endpoint classification.

Mirrors nodejs-sdk/test/classify-url.test.js.
"""

from __future__ import annotations

import pytest

from vaani_observer import VaaniObserver
from vaani_observer.observer import ParsedUrl


def observer(endpoints):
    return VaaniObserver(endpoints=endpoints, instrumentations={"http": False})


def test_returns_none_when_no_rule_is_configured():
    assert observer([]).classify_url("https://api.example.com/v1/chat") is None


def test_matches_a_path_prefix_and_ignores_the_query_string():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    assert vaani.classify_url("https://api.example.com/v1/chat?token=secret")["id"] == "llm"
    assert vaani.classify_url("https://api.example.com/v1")["id"] == "llm"


def test_does_not_match_a_sibling_path_outside_the_configured_prefix():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com/v1/chat"}])
    assert vaani.classify_url("https://api.example.com/v2/chat") is None


def test_treats_the_prefix_as_a_raw_string_prefix():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    # Documents current behaviour: /v10 shares the /v1 string prefix and matches.
    assert vaani.classify_url("https://api.example.com/v10/chat")["id"] == "llm"


def test_requires_the_protocol_to_match():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    assert vaani.classify_url("http://api.example.com/v1/chat") is None


def test_requires_the_host_including_the_port_to_match():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com:8443/v1"}])
    assert vaani.classify_url("https://api.example.com/v1/chat") is None
    assert vaani.classify_url("https://other.example.com:8443/v1/chat") is None
    assert vaani.classify_url("https://api.example.com:8443/v1/chat")["id"] == "llm"


def test_treats_the_default_port_as_equivalent_to_an_omitted_port():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com:443/v1"}])
    assert vaani.classify_url("https://api.example.com/v1/chat")["id"] == "llm"


def test_matches_every_path_under_an_origin_rule():
    vaani = observer(
        [{"id": "tts", "type": "tts", "url": "https://tts.example.com/ignored/path", "match": "origin"}]
    )
    assert vaani.classify_url("https://tts.example.com/")["id"] == "tts"
    assert vaani.classify_url("https://tts.example.com/anything/else?x=1")["id"] == "tts"
    assert vaani.classify_url("https://other.example.com/ignored/path") is None


def test_requires_both_path_and_query_to_match_an_exact_rule():
    vaani = observer(
        [{"id": "stt", "type": "stt", "url": "https://stt.example.com/v1/listen?model=a", "match": "exact"}]
    )
    assert vaani.classify_url("https://stt.example.com/v1/listen?model=a")["id"] == "stt"
    assert vaani.classify_url("https://stt.example.com/v1/listen?model=b") is None
    assert vaani.classify_url("https://stt.example.com/v1/listen") is None
    assert vaani.classify_url("https://stt.example.com/v1/listen/extra?model=a") is None


def test_raises_when_two_rules_match_the_same_url():
    vaani = observer(
        [
            {"id": "a", "type": "llm", "url": "https://api.example.com/v1"},
            {"id": "b", "type": "tts", "url": "https://api.example.com/v1"},
        ]
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        vaani.classify_url("https://api.example.com/v1/chat")


def test_does_not_raise_when_overlapping_rules_disambiguate_by_path():
    vaani = observer(
        [
            {"id": "chat", "type": "llm", "url": "https://api.example.com/v1/chat"},
            {"id": "speak", "type": "tts", "url": "https://api.example.com/v1/speak"},
        ]
    )
    assert vaani.classify_url("https://api.example.com/v1/chat/completions")["id"] == "chat"
    assert vaani.classify_url("https://api.example.com/v1/speak")["id"] == "speak"


def test_classifies_websocket_urls_by_protocol():
    vaani = observer([{"id": "stt", "type": "stt", "url": "wss://stt.example.com/stream"}])
    assert vaani.classify_url("wss://stt.example.com/stream?lang=en")["id"] == "stt"
    # Plaintext is still a different transport from TLS, so `ws` never matches.
    assert vaani.classify_url("ws://stt.example.com/stream") is None


def test_a_rule_covers_both_the_rest_and_websocket_form_of_one_endpoint():
    """Deepgram streams from wss://…/v1/listen and transcribes at https://…/v1/listen.

    This deliberately diverges from the Node SDK, which compares protocols
    literally: there a config that names the REST origin records no connection
    spans at all, which looks like the provider never opened a socket.
    """
    vaani = observer([{"id": "stt", "type": "stt", "url": "https://api.deepgram.com/v1"}])
    assert vaani.classify_url("wss://api.deepgram.com/v1/listen?model=nova-3")["id"] == "stt"
    assert vaani.classify_url("https://api.deepgram.com/v1/listen")["id"] == "stt"
    # http:// against an https:// rule is still a mismatch.
    assert vaani.classify_url("http://api.deepgram.com/v1/listen") is None


def test_accepts_a_parsed_url_instance_as_well_as_a_string():
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    assert vaani.classify_url(ParsedUrl("https://api.example.com/v1/chat"))["id"] == "llm"


@pytest.mark.parametrize("value", ["/v1/chat", None, 42, ""])
def test_raises_on_input_that_is_not_a_valid_absolute_url(value):
    vaani = observer([{"id": "llm", "type": "llm", "url": "https://api.example.com/v1"}])
    with pytest.raises(TypeError):
        vaani.classify_url(value)


def test_an_exact_scheme_rule_wins_over_the_transport_neutral_fallback():
    """Adding websocket coverage must not make working configs ambiguous."""
    vaani = observer(
        [
            {"id": "rest", "type": "stt", "url": "https://api.example.com/v1"},
            {"id": "stream", "type": "stt", "url": "wss://api.example.com/v1"},
        ]
    )
    assert vaani.classify_url("https://api.example.com/v1/listen")["id"] == "rest"
    assert vaani.classify_url("wss://api.example.com/v1/listen")["id"] == "stream"
