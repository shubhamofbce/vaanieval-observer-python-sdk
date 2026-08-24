# Changelog

Every recorded package stamps its SDK version into the manifest, so the version
here is what a bug report is tied back to. Entries describe what changed about
*the data* — a fix that alters no recorded value is not worth an adopter's time.

## 0.5.0

A sixth review pass over the 0.4.0 fixes, which found four further defects of
the same silent kind, and a seventh pass over *those* fixes, which found two
more. One field changed meaning, hence the minor bump.

### A caveat that never reached the expensive number

Attribution uncertainty was written onto the TTS span. A reply can report LLM
token counts and then be interrupted before any TTS metric, audio frame or
assistant item exists — in which case the turn published an `ok` LLM operation,
with prompt and completion tokens on it, and nothing anywhere saying its
ownership was a judgement call. The caveat now belongs to the turn and is
stamped on every operation it publishes.

### A swallowed error could hedge an innocent turn

The contested marker lived on the recorder, set in one place and cleared in
another. Handler exceptions are deliberately swallowed so a call keeps
recording, so a transient failure in between left the marker set and the next,
unrelated utterance was published carrying someone else's uncertainty. It is now
a local variable, which removes the failure mode rather than guarding it.

### Opening a call changed its turn count

The session list counts one LiveKit message committed as two turn rows as a
single exchange. The call view counted rows, so opening a one-turn call showed
two. Both rows are still listed — the inspector needs them — but the headline
and the live rail now count exchanges, and say how many rows continue another.

### A spoken sentence could be dropped from the transcript

The recorder de-duplicated assistant messages by asking whether the new text was
*contained in* what it had already recorded. So a filler of "The fare is ready"
followed by an answer of "fare" discarded the answer: 21 characters spoken, 17
recorded, and `char_count` short by the same amount with nothing to indicate it.
Containment is not identity, and even identical text can legitimately be said
twice. De-duplication is now keyed on the LiveKit `ChatItem.id`, which is unique
per delivered item. A test that had asserted the truncated count — blessing the
defect — was corrected.

### The volume chart disagreed with the KPI directly above it

When one caller's utterance was split across two turns, the KPI counted the pair
as one turn and the timeseries counted it as two. A range landing on the split
boundary could show a KPI of 0 above a chart bar of 1. The chart now applies the
same rule the KPI does. Latency, failures and lag stay per-span, because those
are measured rather than counted.

### `metering_scope` asserted billing semantics it could not know

**Breaking.** 0.4.0 inferred, from whether a provider's usage report arrived
before or after the final transcript, whether it metered one utterance or the
whole connection. That inference is wrong for both providers we checked against
their source: OpenAI emits a genuine *per-utterance* meter after the final,
which we labelled `connection`; Deepgram ticks a *connection-wide* 5-second
collector that routinely fires before the final, which we labelled `utterance`.

`metering_scope` is now always `"unknown"`. What is actually observable is
published as `metered_arrival` (`before_final`, `after_final` or
`straddles_final`), and `metering_scope_note` states the caveat on the payload
itself, so it travels with the number rather than living only in these docs.
**If you read `metering_scope`, you must stop.**

### A reply detached from its turn without saying so

If a partial transcript arrives while a filler is playing, the reply that
follows is kept in a turn of its own. That is the safe choice — merging would
show a caller answered before they spoke — but it is a *judgement*: LiveKit
reports a preflight transcript from a second speaker and an ordinary interim
from the current one through the same event, and preemptive generation is on by
default, so both really occur. The span now carries `reply_attribution:
"inferred"` and a `reply_attribution_reason` naming exactly what could not be
distinguished, and the caveat is attached to the *turn*, so a reply that reports
LLM tokens and is then interrupted before speaking still carries it.

The events do not distinguish the two cases, but the **speech handle does**:
preemptive generation is created with `_generate_reply(schedule_speech=False)`
and stays unscheduled until the prediction is validated, while an ordinary reply
is marked scheduled synchronously. Both are still unscheduled inside the
`speech_created` callback itself, so using this signal means deferring the
binding by an event-loop iteration. That is **open work, not done here** — an
earlier note in this file calling the ambiguity "inherent" was wrong and is
withdrawn.

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
