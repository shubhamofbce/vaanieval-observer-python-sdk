# Changelog

Every recorded package stamps its SDK version into the manifest, so the version
here is what a bug report is tied back to. Entries describe what changed about
*the data* — a fix that alters no recorded value is not worth an adopter's time.

## 0.5.1

An eighth review pass over the 0.5.0 fixes. One release-blocking defect: a
reply's measured token counts could be filed against a turn with no question in
it. No field changed meaning, hence the patch bump.

### A reply is now attributed by asking LiveKit, not by guessing

While a filler is playing, a reply that arrives over a partial transcript is
either this caller's own delayed answer or one predicted for the caller talking
over them. The events cannot tell the two apart: a preflight transcript and an
ordinary interim reach the SDK as the same `user_input_transcribed(is_final=
False)`. Both were therefore refused, and the ordinary case — the common one —
published the reply's measured `total_tokens`, `prompt_tokens` and first-token
latency against a turn that contained no question, while the question itself
read as unanswered. One exchange described wrongly twice, with the cost on the
wrong side of it, under a caveat that admitted the doubt without resolving it.

LiveKit does distinguish them, just not at the moment the SDK first hears about
the reply, and not through a single field. There are four ways a reply is
created and three of them say who they are for: the automatic answer to a
completed user turn is scheduled in the same synchronous frame that created it,
reports `user_initiated` and carries audio input details; preemptive generation
deliberately is not scheduled, and stays that way until the predicted turn is
validated; a realtime model's server-side generation is scheduled at once but is
not `user_initiated`, and it answers the speech still being transcribed — that
provider may withhold the final transcript until its reply exists, so the reply
legitimately precedes the words that prompted it.

The decision now waits the one event-loop iteration that scheduling takes and
then reads all three signals. Automatic answers join the turn they answer.
Preemptive and realtime replies keep their own turn, which the caller's final
transcript then joins, so a caller's words are never reported as answered before
they were spoken.

Scheduling alone is not enough, and reading it alone was itself a defect caught
before release: it is queue state, not provenance. A realtime reply is queued
immediately and would have been merged backwards into the previous caller's
turn, and reported as certain.

Nothing is bound during that iteration, which leaves the previous turn current —
the same turn anything arriving between the filler and the reply would have
landed on anyway. The decision is also settled on demand by whichever of the
metric, the audio stream or the next speech needs the turn first, because
waiting on the loop alone would have made attribution depend on arrival order,
and an operation already written cannot be moved.

Verified against real `SpeechHandle` and `SpeechCreatedEvent` objects on
livekit-agents 1.7.0, and against 1.6.10 source, in all four directions.

### The attribution caveat now means what it says

`reply_attribution: "inferred"` was published for every reply that followed a
filler. It is now recorded only where the ownership genuinely cannot be read:
a reply your own code asked for with `AgentSession.generate_reply()`, which is
indistinguishable from the automatic answer except that its input modality
defaults to text — and which may be answering the caller or saying something
unrelated. Builds too old to report these signals also keep the caveat, rather
than being merged on an assumption. A caveat that appears on correct data
teaches adopters to ignore it; one that disappears on a guess is worse.

Derived LLM spans — reconstructed from `conversation_item_added` when no plugin
emits `metrics_collected` — now carry the caveat too. It was the one publisher
the disclosure never reached, so a per-turn latency from a build with no LLM
metrics looked settled when it was not.

### Split exchanges: which stage was measured twice is now recorded

The console warned that a stage may be double counted by asking whether it had
measured more populations than the range has turns. That proves a double count
but cannot detect one: a split exchange plus one unrelated reply-only turn
leaves the totals level, so the warning disappeared and the panel published full
coverage for a stage measured twice. Which stages a split actually doubled is a
fact about its two halves, so it is now recorded per stage when the halves are
ingested and summed like any other counter.

Two consequences worth stating. Stored calls are rebuilt on upgrade, because the
new columns would otherwise sit at their default and the drift audit — which
recomputes them — would have reported every already-stored split call as
tampered with, permanently. And the split is recorded against the half the
exchange *started* on, so an hour holding only the second half reports neither
the exchange nor its caveat. That is the rule calls already follow when they
span an hour: counted once, where they began. Recording it on the continuation
instead made an hour report zero exchanges and, in the same response, that one
of them carried more than one utterance.

### A call's turn count no longer depends on who is counting

A continuation row whose parent is missing — the parent fell outside the range,
or was never recorded — was counted as a turn by the call rollup and the browser
and subtracted by the rail, the range count, the overview and the hourly
rollup. A reader could open a call listed as zero turns, read one turn in its
headline, and see zero again in range. The rule is now applied once, where the
rows are ingested, so every counter inherits it. Raw `continues_turn` is left
untouched in the operation payload: that is what the SDK observed.

Rows ingested by an earlier release are re-decided on upgrade. The previous
backfill trusted that a turn saying it continues another one had that other one
present, which is exactly the case that fails; and because the schema version
already recorded that the backfill had run, nothing would have revisited them.
An upgraded database showed no turns in the session rail and one everywhere
else, for the same call.

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
