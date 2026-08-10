"""End-to-end check: drive the LiveKit recorder, upload, read the dashboard back.

Not part of the unit suite -- it needs a running dashboard -- but it is the only
thing that proves a Python-produced package renders the same way the Node one
does.

    python scripts/e2e_python_package.py http://127.0.0.1:8077
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vaani_observer import VaaniObserver  # noqa: E402
from vaani_observer.integrations.livekit import VaaniLiveKitRecorder  # noqa: E402

RATE = 24000


def tone(ms: int, hz: float) -> bytes:
    samples = int(RATE * ms / 1000)
    return struct.pack(
        f"<{samples}h",
        *[int(12000 * math.sin(2 * math.pi * hz * n / RATE)) for n in range(samples)],
    )


class Frame:
    def __init__(self, data: bytes) -> None:
        self.data = memoryview(data)
        self.sample_rate = RATE
        self.num_channels = 1


def transcript(text, final, language="hi-IN"):
    return SimpleNamespace(transcript=text, is_final=final, language=language)


def llm(speech_id):
    return SimpleNamespace(metrics=SimpleNamespace(
        type="llm_metrics", label="azure", request_id="r1", duration=0.9, ttft=0.32,
        cancelled=False, completion_tokens=48, prompt_tokens=310, prompt_cached_tokens=0,
        total_tokens=358, tokens_per_second=53.3, speech_id=speech_id,
        metadata=SimpleNamespace(model_name="gpt-4o", model_provider="azure")))


def tts(speech_id):
    return SimpleNamespace(metrics=SimpleNamespace(
        type="tts_metrics", label="sarvam", request_id="t1", ttfb=0.28, duration=1.2,
        audio_duration=2.4, cancelled=False, characters_count=61, streamed=True,
        segment_id="s1", speech_id=speech_id,
        metadata=SimpleNamespace(model_name="bulbul:v3", model_provider="sarvam")))


def stt():
    return SimpleNamespace(metrics=SimpleNamespace(
        type="stt_metrics", label="deepgram", request_id="s1", duration=0.0,
        audio_duration=2.9, streamed=True,
        metadata=SimpleNamespace(model_name="nova-3", model_provider="deepgram")))


def eou(speech_id):
    return SimpleNamespace(metrics=SimpleNamespace(
        type="eou_metrics", timestamp=0.0, end_of_utterance_delay=0.44,
        transcription_delay=0.12, on_user_turn_completed_delay=0.01, speech_id=speech_id))


class Bus:
    """Stands in for `AgentSession.on`."""

    def __init__(self) -> None:
        self.handlers: dict = {}

    def on(self, name, handler):
        self.handlers.setdefault(name, []).append(handler)

    def emit(self, name, event):
        for handler in self.handlers.get(name, []):
            handler(event)


async def main(base_url: str) -> int:
    observer = VaaniObserver(
        endpoint=base_url,
        api_key="local-dev",
        spool_directory=os.path.abspath(".e2e-spool"),
        capture={"stt_content": True, "http_bodies": True},
        endpoints=[
            {"id": "stt", "type": "stt", "url": "https://api.deepgram.com/v1"},
            {"id": "llm", "type": "llm", "url": "https://example.openai.azure.com/openai"},
            {"id": "tts", "type": "tts", "url": "https://api.sarvam.ai"},
        ],
        instrumentations={"http": False},
    )
    recorder = VaaniLiveKitRecorder(
        observer,
        agent_id="livekit-trip-planner",
        metadata={"pipeline": "deepgram->azure-openai->sarvam", "source": "python-e2e"},
        upload=False,
    )
    bus = Bus()
    recorder.attach(bus)
    call = recorder.call
    await call.ready()

    bus.emit("speech_created", SimpleNamespace(
        speech_handle=SimpleNamespace(id="greet"), source="say"))
    bus.emit("metrics_collected", tts("greet"))
    for _ in range(20):
        recorder.tap_output_frame(Frame(tone(20, 220)))
        await asyncio.sleep(0.005)
    bus.emit("conversation_item_added", SimpleNamespace(item=SimpleNamespace(
        role="assistant", text_content="Namaste, main Tara hoon.",
        metrics={"e2e_latency": 0.9, "tts_node_ttfb": 0.28})))

    exchanges = [
        ("goa ki flight kitne ki hai", "Goa ki flight lagbhag 6000 rupaye ki hai."),
        ("hotel bhi bata do", "Goa mein 3500 rupaye per night ka hotel hai."),
    ]
    for index, (heard, said) in enumerate(exchanges, start=1):
        speech_id = f"speech-{index}"
        bus.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
        for _ in range(25):
            recorder.tap_input_frame(Frame(tone(20, 440)))
            await asyncio.sleep(0.005)
        bus.emit("user_input_transcribed", transcript(heard.split()[0], False))
        bus.emit("user_input_transcribed", transcript(" ".join(heard.split()[:3]), False))
        bus.emit("metrics_collected", stt())
        bus.emit("user_state_changed", SimpleNamespace(new_state="listening"))
        bus.emit("user_input_transcribed", transcript(heard, True))
        bus.emit("conversation_item_added", SimpleNamespace(item=SimpleNamespace(
            role="user", text_content=heard,
            metrics={"transcription_delay": 0.12, "end_of_turn_delay": 0.44})))
        bus.emit("speech_created", SimpleNamespace(
            speech_handle=SimpleNamespace(id=speech_id), source="generate_reply"))
        bus.emit("metrics_collected", eou(speech_id))
        bus.emit("metrics_collected", llm(speech_id))
        bus.emit("function_tools_executed", SimpleNamespace(zipped=lambda: [(
            SimpleNamespace(name="search_flights", arguments='{"to":"GOI"}', call_id="c1"),
            SimpleNamespace(output='{"price":6000}', is_error=False, call_id="c1"))]))
        bus.emit("metrics_collected", tts(speech_id))
        for _ in range(30):
            recorder.tap_output_frame(Frame(tone(20, 330)))
            await asyncio.sleep(0.005)
        bus.emit("conversation_item_added", SimpleNamespace(item=SimpleNamespace(
            role="assistant", text_content=said,
            metrics={"e2e_latency": 1.42, "llm_node_ttft": 0.32, "llm_node_tps": 53.3,
                     "tts_node_ttfb": 0.28, "playback_latency": 0.02})))

    await recorder.finish(outcome="completed")
    finalized = await call.finished
    print(f"package  : {finalized.directory}")
    print(f"audio    : {finalized.manifest['audio']}")

    result = await observer.upload_package(finalized)
    print(f"uploaded : {result}")
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
    raise SystemExit(asyncio.run(main(url)))
