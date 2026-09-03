# Vaani Observer for Python

> Local-first observability for real-time voice agents.

Capture the STT, LLM, TTS, tool, WebSocket, and audio activity behind a voice
call without putting a network dependency on its live media path. Vaani
Observer writes a portable session package locally, then uploads it explicitly
after the call for review in the [Vaani Observer Dashboard](https://github.com/shubhamofbce/vaanieval-observer-backend).

![Vaani Observer dashboard](docs/images/dashboard-overview.png)

## What you get

- Provider-neutral operation spans and streaming milestones for STT, LLM, TTS,
  tools, and connection lifetimes.
- Timeline-aligned caller and agent audio, recorded as stereo PCM.
- Safe ambient instrumentation: only traffic inside an active session that
  matches your configured endpoint rules is observed.
- A byte-compatible package format shared with the
  [Node.js SDK](https://github.com/shubhamofbce/vaanieval-observer-nodejs-sdk).

## Quick start

Local-first Python observability for voice-agent calls. It writes a portable
session package (`manifest.json`, `events.jsonl`, `call.audio`) to disk and
never blocks the call path on a remote upload. `call.audio` is timeline-aligned
16-bit stereo PCM with agent audio on the left and caller audio on the right.

The package format is identical to the
[Node.js SDK](https://github.com/shubhamofbce/vaanieval-observer-nodejs-sdk)'s, so the
[dashboard](https://github.com/shubhamofbce/vaanieval-observer-backend) ingests
calls from either runtime without knowing which produced them.

## Instrument a call

```python
from vaani_observer import VaaniObserver

vaani = VaaniObserver(
    spool_directory="/tmp/vaani",
    endpoints=[{"id": "llm", "type": "llm", "url": "https://llm.example/v1"}],
)

session = vaani.start_session(agent_id="support")
session.record_inbound_audio(pcm, {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1})

with session.context():
    await client.post("https://llm.example/v1/chat")   # auto-instrumented

finalized = await session.end(outcome="completed")
await vaani.upload_package(finalized)                  # explicit, post-call
```

## What is different from the Node SDK, and why

| Concern | Node | Python |
| --- | --- | --- |
| Ambient session | `AsyncLocalStorage` | `contextvars.ContextVar`, entered with `with session.context():` |
| HTTP capture | patches the global `fetch` | patches `httpx.Client/AsyncClient.send` and `aiohttp.ClientSession._request` |
| Websockets | `ws` EventEmitter handlers | wraps `send_*`/`receive*`/`close` coroutines (aiohttp, `websockets`) |
| Spool writes | a promise chain over `fs/promises` | one dedicated writer thread per session |

Python has no single HTTP entry point to instrument, so the SDK wraps the two
clients a Python voice agent actually uses. Both wrappers are inert unless there
is an ambient session *and* the URL matches a configured endpoint rule, so
unrelated application traffic is never timed or recorded.

Spool writes go to a private thread rather than the event loop. That keeps every
filesystem syscall off the loop (audio arrives every 20 ms, and a blocking
`write` there is audible) while preserving the strict append ordering that
`events.jsonl` and the raw audio tracks depend on.

## Operations, turns and scopes

`session.start_operation()` records provider-neutral streaming milestones.
Operations carry a `scope`:

* `turn` (the default) — one unit of conversational work, grouped by `turn_id`.
* `connection` — a provider socket's lifetime.

`tool` is supported alongside `stt`, `llm` and `tts` for internal tool steps.

Repeated milestones of the same name accumulate (`count`, `occurred_at_ms` of
the first, `last_at_ms` of the latest) rather than overwriting, so a
high-frequency transport keeps useful timing without one event per frame.
`start_operation(started_at_ms=...)` back-dates a span whose start is only known
in hindsight — for example STT work that began before the turn id was allocated.

`operation.sample(name, data, limit=100)` retains a bounded series (STT partial
transcripts) without turning an audio stream into an unbounded event stream.

## Capture policy

`capture.http_bodies=True` retains HTTP request/response bodies; headers are
never persisted. `capture.payload_max_bytes` caps each retained payload (16 KiB
by default). `capture.stt_content` defaults to `False`: when enabled, the
integration should attach the final transcript, language, confidence, word
timings and a bounded sample of partials to its per-turn STT operation.

Streaming response bodies are never drained to capture them — doing so would
hold back the first token, which is exactly the latency this SDK measures.

## LiveKit Agents integration

```python
from vaani_observer.integrations.livekit import VaaniAudioTapMixin, observe_agent_session

class MyAgent(VaaniAudioTapMixin, Agent):   # tees caller/agent PCM
    ...

recorder = observe_agent_session(agent_session)   # configured from VAANI_* env
agent.vaani = recorder
...
await recorder.finish(outcome="completed")
```

See `src/vaani_observer/integrations/livekit.py`. It maps LiveKit's session
events and metrics onto turns and STT/LLM/TTS/tool operations. The mixin records
both audio tracks through LiveKit's supported `stt_node` / `tts_node` extension
points rather than private io plumbing, which changes between releases.

`VaaniLiveKitRecorder.from_env()` reads `VAANI_ENABLED`, `VAANI_ENDPOINT`,
`VAANI_API_KEY`, `VAANI_SPOOL_DIR`, `VAANI_AGENT_ID`, `VAANI_UPLOAD`,
`VAANI_CAPTURE_AUDIO`, `VAANI_CAPTURE_HTTP_BODIES`, `VAANI_CAPTURE_STT_CONTENT`
and `VAANI_PAYLOAD_MAX_BYTES`. It returns an **inert** recorder when
`VAANI_ENABLED` is off or configuration fails: a misconfigured recorder must
never be the reason a call fails to start. Note that it enables body and
transcript capture by default, where the bare SDK defaults them off.

## Tests

```bash
uv run --extra dev pytest
```

## Scaling implications

* **Audio dominates storage.** Two raw 16-bit PCM tracks are ~64 KB/s of call at
  16 kHz mono. A few thousand calls a day is terabytes a month; production needs
  object storage with retention and, realistically, Opus rather than raw PCM.
* **One writer thread per session** is right for one agent per process, which is
  how LiveKit workers run. A process multiplexing hundreds of concurrent calls
  would need a shared writer pool instead.
* **Post-call upload delays visibility** by the call duration and is lost if the
  host dies mid-call. The spool survives a crash; an unfinalized directory does
  not, and needs a sweeper if that matters.
* **Body capture increases blast radius.** Retained LLM bodies contain the
  conversation. It is off by default and bounded when on, but it is a real
  privacy and storage decision, not a debug toggle.
* **Instrumentation is monkey-patching.** It is reversible and version-tolerant,
  but a major `httpx`/`aiohttp` refactor can silently disable HTTP capture;
  `capture_status.http_instrumentation` is what tells you it was on.
