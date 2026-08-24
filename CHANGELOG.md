# Changelog

Every recorded package stamps its SDK version into the manifest, so the version
here is what a bug report is tied back to. Entries describe what changed about
*the data* — a fix that alters no recorded value is not worth an adopter's time.

## 0.4.0

The whole of this release answers an external audit of the LiveKit integration
and five subsequent independent review passes over the fixes themselves. Nearly
every defect below was a *silent* one: the call completed, the dashboard showed
a green checkmark, and the numbers were wrong or missing with nothing on screen
to say so.

### Data loss — recordings that ended early or never arrived

- **`finish()` no longer truncates a live call.** The documented
  `finally: await recorder.finish()` fired mid-call on livekit-agents 1.x,
  because `start()` returns when the session *starts*. A measured 49-second,
  3-turn call was recorded as 17.5 seconds and 2 turns. Pass `job_ctx=` and the
  recorder closes when the job does.
- **Event handlers are inert after `finish()`.** `finish()` cleared internal
  state while handlers stayed subscribed, and the resulting `AttributeError`
  was swallowed at DEBUG.
- **Upload no longer has to finish inside the shutdown hook.** LiveKit's default
  `shutdown_process_timeout` is 10s; raw 24 kHz stereo PCM is 94 KB/s, so a
  5-minute call could not upload in time at 10 Mbit/s no matter how the timeout
  was tuned. Objects are now gzipped on the wire when the backend advertises
  support and there is budget left to spend on it, `urlopen` has a real
  `timeout`, and the configured retry count is actually read.
- **Upload failures say the path, the cause and the remedy** instead of leaving
  an empty spool directory behind.

### Wrong numbers that looked right

- **TTS spans are no longer lost.** A 130-second call reported 0 TTS operations
  and 4 healthy turns while its own recorded audio proved the agent spoke for
  63.4 seconds — a 100% undercount behind a green checkmark. Reproduced on 3 of
  3 calls before the fix.
- **The agent's words are recorded.** The transcript view was permanently empty
  while the docs promised transcripts.
- **The model label is the model.** `with_azure(azure_deployment="gpt-5-mini")`
  leaves the plugin's `model` at its `"gpt-4o"` default; every span said
  `gpt-4o`. The deployment is now read back off the client, and the provider's
  own label is preserved beside it rather than overwritten.
- **`audio_bytes` and `audio_ms` use the same denominator.** They did not, which
  put cost estimates out by roughly 1000×.
- **A reply is attributed to the turn it answers.** Preemptive replies, filler
  phrases spoken from a callback, barge-in, and turns LiveKit merges or splits
  each had a way of moving a latency, a token count or a caller's words onto
  the wrong turn. Where ownership genuinely cannot be proven the span carries
  `stream_ownership: "inferred"` rather than `"proved"`, so a reader auditing a
  per-turn number knows which claim it rests on.
- **A split exchange counts once.** When the caller's earlier words were already
  published, the SDK records two spans where LiveKit committed one message.
  That is now marked with `continues_turn`, so turn counts and per-turn averages
  do not double.

### Numbers that were honest but unreadable

- **`metering_scope`** states what a recogniser's meter is a measurement *of*:
  `utterance`, `connection`, or `mixed`. On a live 4-turn call, 3 turns reported
  `connection` — the entire published meter arrived *after* the caller stopped
  talking, so read as speech duration those numbers were not slightly wrong,
  they were unrelated. The amount itself is never adjusted; it is still what the
  provider will invoice.
- **`metered_after_final_ms`** publishes the subtractable portion.
- **`provider_metered_audio_ms`** was renamed from `audio_ms`, which read as
  speech duration and showed 5000 ms of "audio" on a 2975 ms span.
- **Coverage is stated, not implied.** A call that started untapped, an agent
  that was never bound, and a mid-call format change each now say so instead of
  certifying themselves complete.

### Install and operation

- **Version is defined once** (`_version.py`), read by `pyproject.toml` and
  stamped into every manifest. Packages previously all claimed `0.1.0` however
  much had changed underneath them.
- **`drain` works end to end**, including endpoint-scoped receipts, and refuses
  to delete a local recording the backend has not proven it holds.
- All four `VAANI_UPLOAD_*` knobs are read.
- Structured logging; failures surface at WARNING rather than DEBUG.

### Known limits, stated deliberately

- A caller who both makes a non-speech sound *and* produces a partial transcript
  before the previous answer arrives is genuinely ambiguous; the SDK picks one
  reading and does not pretend otherwise.
- `metering_scope` is derived from observed timing, not from a table of provider
  behaviour. A table would be silently wrong for any provider missing from it.
- The 128 MiB package cap is a hard ceiling on call length.

## 0.1.0

Initial release. `0.3.0` was tagged in development but, since this SDK is
distributed only from source during the closed beta, `0.4.0` is the first
version after `0.1.0` that an adopter can install.
