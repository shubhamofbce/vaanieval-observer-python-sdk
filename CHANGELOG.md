# Changelog

Every recorded package stamps its SDK version into the manifest, so the version
here is what a bug report is tied back to. Entries describe what changed about
*the data* — a fix that alters no recorded value is not worth an adopter's time.

## 0.5.4

A twelfth review pass, aimed at the anchor the eleventh release rests on. It
found that the anchor itself can be removed from the stack, that one family of
LiveKit's own calls is billed to the wrong caller, and that a cancelled reply
can hand its caveat to an unrelated one. Two of the five findings are about the
dashboard rather than the SDK.

**A replaced `generate_reply()` looked exactly like LiveKit's automatic answer.**
The call site was matched against the public method's code object, which a
subclass, a wrapper or any tracing layer can replace — and a reply that never
passed through it was read as the automatic answer to the previous caller and
merged backwards into their turn, silently. Replies are now anchored on
`AgentActivity._generate_reply()`, the function that actually emits the event
and is therefore always on the stack, and the decision is taken from *who
called it*: LiveKit itself (`agent_activity.py:2574`) is the automatic answer,
anything else is a stand-in for the public method. Where even that is absent, a
frame calling itself `generate_reply` on an `AgentSession` is enough to know the
reply was not the automatic one. Verified against real `AgentActivity` objects
on livekit-agents 1.7.0.

**A reissued reply was charged to whoever spoke next.**
`RunResult._maybe_retry_output()` (`run_result.py:292`) and the realtime
fallback adapter (`llm/realtime_fallback_adapter.py:394` and `:445`) reissue a
reply for a run that *already exists*. They were placed with the speech in
flight like LiveKit's other internal calls, so the next caller's turn was
charged for the retry's tokens, latency and cost, and that caller's transcript
was recorded as the question it answered — an exchange neither of them had.
Reissues are now kept in a turn of their own and are not joined by the
following transcript.

**A cancelled reply left its caveat behind for the next one.**
When LiveKit's own reply was cancelled before it measured anything, its empty
turn was reused by the next reply — which inherited a `reply_attribution` that
described a call it was not. A reply whose own handle says where it belongs now
clears the caveat when it takes over an abandoned turn.

**`attach()` could register handlers onto a finished recorder.**
The check for a live call was read *outside* the lock that guards the
transition, so a thread could see a live call, wait, and subscribe all nine
handlers after `finish()` had already detached. They are inert, but nothing
removes them, so a long-lived session held the finished recorder and every turn
in it. The check and the transition are now under the same lock on both sides.

**Two dashboard fixes.** The `operations(turns)` index was created *after* the
migration that scans on it, so a backfill on a large archive ran unindexed
(measured 3.9× for 2× input). And the continuation rebuild materialised every
turn in the archive at once (36.9 MiB traced for 150k turns); it now resolves
one call at a time — 0.27 s and 0.1 MiB peak for the same input.

**Warnings no longer read `vaani: vaani:`.** Nine call sites passed a message
that already carried the prefix into a helper that adds it.

## 0.5.3

An eleventh review pass, aimed at the reply-attribution work itself. It found
that the classifier answers a narrower question than it is read as answering,
and that two of the release's own safeguards resolve an unknown into a
confident answer. Nothing here changes what a field means; every change either
moves a measurement to a different turn or adds a caveat that was owed.

**`session.run()` is the adopter's decision, in LiveKit's file.**
`AgentSession.run(user_input=...)` is a public entry point that forwards to the
public `generate_reply` (`agent_session.py:823`). The frame above the reply is
LiveKit's own, so it was read as framework speech and joined silently to the
*next* spoken caller's turn — putting a programmatic run's tokens and cost on
somebody who never asked for them. LiveKit's own entry points are now
recognised, so such a reply is kept in a turn of its own.

**LiveKit's internal calls are not all answers to speech already under way.**
The previous release asserted that they were, on the strength of six call
sites. There are more than a dozen on 1.7.0, and ten of them — the
`beta/workflows/` prompts — ask the caller a question *before* the caller has
said anything. The transcript that follows such a reply is the answer to it,
not the question it answered. The placement is unchanged, because it is still
the likeliest reading, but it now carries a caveat naming the ambiguity instead
of presenting one of the two readings as fact.

**Five seconds bound LiveKit's wait, not the provider's generation.**
`agent_activity.py:4278` waits on a shielded future and then merely stops
tracking it, while the chat context that prompts the provider was already
updated. A slow provider's answer can therefore arrive after the deadline and
still be the one the tool was owed. It was being called ordinary in-flight
speech — moving its tokens to the next caller, with no caveat, so the tool
exchange read as having had no model answer at all. An expired window is now a
third answer, distinct from "no tool is waiting", and resolves to a turn of its
own with the reason stated.

**Failing to read the stack is not evidence of an automatic answer.** A blocked
`sys._getframe`, a public method whose code object cannot be reached and a walk
that ran out of budget all returned the same value as "the public method is
genuinely not on the stack" — which is LiveKit's automatic answer to a
completed turn, merged backwards into it with no caveat. Those are opposite
kinds of fact and are now kept apart; a failure to look is disclosed.

**Attaching twice at once no longer doubles the caller's words.** Checking
whether a session was already attached and subscribing to it were two steps, so
a reconnect racing a startup path could pass the check in both threads and
register every handler twice. One final transcript then published the caller's
words doubled, with the manifest still reporting the capture complete. The
check and the subscription are now one decision.

Verified across nine routes against a real `AgentSession` on livekit-agents
1.7.0, including `session.run()`, a bound method saved before recording
started, and a caller compiled inside the `livekit` package.

## 0.5.2

A tenth review pass, this time aimed squarely at how a reply's owner is
decided. It found that the mechanism introduced in 0.5.1 was wrong in both
directions, and two smaller defects of the same silent kind. No field changed
meaning, hence the patch bump.

### Attribution reads the stack instead of replacing a method

0.5.1 identified a reply your own code asked for by replacing
`session.generate_reply` and counting calls to the replacement. Two things
defeat that.

An application that saved the bound method before recording started — `reply =
session.generate_reply` at setup, or any wrapper holding its own reference —
called straight past the replacement. Its reply was then read as the automatic
answer and merged into whichever turn a filler was holding open, with no caveat:
the token counts and cost of a reply that may have answered nothing were charged
to the caller's question, and nothing on the record said so.

The other direction is more common. LiveKit calls its own public method in six
places on 1.7.0: committing a realtime turn (`agent_activity.py:1693`), retrying
a structured output (`run_result.py:292`), answering an asynchronous tool result
(`tool_executor.py:599`), the IVR activity (`ivr_activity.py:53`, `:83`) and
`agent.py:346`. Every one of those was reported as *your* reply — given its own
turn, under a caveat, with the caller's final transcript opening yet another
turn after it, so the question and its answer sat two turns apart.

The stack is the one thing a captured reference cannot change: the same code
object runs either way, and `speech_created` is emitted synchronously inside the
call, so the frame that asked is still live. The recorder walks the stack for
`AgentSession.generate_reply` and classifies by the filename of the frame above
it — inside the `livekit` package the reply is the framework's own, outside it
is yours, absent entirely it is the automatic answer. Framework replies now join
the turn that follows them rather than the one before, because every internal
route answers something already under way.

"Inside the `livekit` package" means the namespace package's own search path,
not just `livekit/agents`: plugins are separate distributions sharing that
directory, so matching only `agents` would have read a plugin's own call as
yours. The comparison keeps a trailing separator, so a project directory named
`livekit_helpers` beside `livekit` is not taken for a package inside it — that
mistake would have dropped the caveat from a reply you asked for.

The frame directly above the public method is not always the one that decided:
a deferred callback, a task wrapper or a decorator puts an interpreter frame
there. Those are stepped over, up to a small bound, until a frame that belongs
to somebody is reached — but *only* interpreter frames, and installed packages
live under the interpreter's own library directory, so `site-packages` and
`dist-packages` are explicitly excluded. Without that, an adopter whose agent is
installed rather than run from a checkout would have had their own reply read as
LiveKit's, with the caveat dropped. Where nothing can be established the answer
stays "yours", which only ever adds a caveat.

Verified across eight routes against a real `AgentSession` on livekit-agents
1.7.0, including a bound method saved before recording started and a caller
compiled inside the `livekit` package.

### Recording a call twice no longer doubles the caller's words

`attach()` called twice — which a restart, a retry or a defensive `attach()` in
two places all produce — registered every handler a second time. One final
transcript then ran the handler twice and the turn published the caller's words
doubled, `"hello hello"` for one `"hello"`, while the manifest reported the
capture complete. Anything read from the transcript, including an evaluation of
what the caller asked for, was wrong with no sign of it. `attach()` is now
idempotent per session; because `finish()` unsubscribes, a session reused for a
second call is still recorded in full.

### A tool result that wants no reply no longer claims a later one

The flag saying a tool result is owed a reply was set when tools ran and cleared
only when a reply arrived. A result needing none — `StopResponse`, or any result
whose `reply_required` is false — left it set indefinitely, so an unrelated
barge-in minutes later was filed as the answer to that tool call, on the wrong
turn, under a caveat that made it look considered. The flag now honours
`has_tool_reply` (`events.py:447`) and otherwise expires after the same five
seconds LiveKit itself waits for the reply (`agent_activity.py:4278`).

### A call whose halves point at each other keeps its turns

A package claiming turn A continues turn B *and* turn B continues turn A
satisfied "my parent is present" in both directions, so both halves were
subtracted and the call was stored as ready with no turns at all — which also
hid it from the unmeasured-call check, since that only looks at calls with
turns. A cycle has no first half, so neither row is treated as a continuation
now: the call keeps both exchanges and reads them as separate, which is the
honest answer for a package that cannot say which came first. The rule that
decides this also lived in four places, and the session rail and the call page
disagreed about that call; they now share one function.

### A stage that skipped the middle half still says it ran twice

Whether a stage was measured twice within one split exchange was answered by
comparing each half against the half before it. An exchange split twice whose
middle half carries only the caller's words — no reply is owed yet, so no model
runs — is true of no adjacent pair, so the panel showed "2 measured, 1 turn in
range" with nothing to reconcile them. The question is about the whole exchange
and is now asked of the whole exchange.

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

LiveKit distinguishes most of them, just not at the moment the SDK first hears
about the reply, and not through a single field. Preemptive generation
deliberately is not scheduled, and stays that way until the predicted turn is
validated. A realtime model's server-side generation is scheduled at once but is
not `user_initiated` — that provider may withhold the final transcript until its
reply exists, so the reply legitimately precedes the words that prompted it.
Everything else is scheduled and `user_initiated`.

Two routes emit an event identical in *every* field to the automatic answer, so
they are settled by what the recorder observed rather than by what the event
says. A reply your own code asked for is recognised because the call is still on
the stack: `AgentSession.generate_reply()` reaches `speech_created` synchronously
(`agent_session.py:1508-1520`), so the frame that asked for the reply is still
live when the recorder sees it. The recorder walks the stack for that method and
reads the filename of the frame above it — inside the `livekit` package the
reply is the framework's own, outside it is yours, and absent entirely it is the
automatic answer, which never passes through the public method at all. This
holds whatever `input_modality` the call was given, and whether or not your code
saved the bound method before recording started. And `user_initiated=False`
covers both a realtime reply to speech being transcribed now *and* the automatic
reply after a tool result, which answers the turn that ran the tool; those are
told apart by whether the current turn has a tool result still owed a reply,
which LiveKit publishes as `function_tools_executed` before the reply is
generated (`agent_activity.py:4242`, ahead of `:4273`).

The decision now waits the one event-loop iteration that scheduling takes and
then reads every available signal. Automatic answers and post-tool replies join
the turn they answer. Preemptive and realtime replies keep their own turn, which
the caller's final transcript then joins, so a caller's words are never reported
as answered before they were spoken.

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

Verified against a real `AgentSession` and real `SpeechHandle` and
`SpeechCreatedEvent` objects on livekit-agents 1.7.0, and against 1.6.10 source,
in all eight directions — including a bound method saved before recording
started, and a caller inside the `livekit` package itself.

### The attribution caveat now means what it says

`reply_attribution: "inferred"` was published for every reply that followed a
filler. It is now recorded only where the ownership genuinely cannot be read:
a reply your own code asked for with `AgentSession.generate_reply()`, which may
be answering the caller or saying something unrelated; and a realtime reply
placed on the turn whose tool result it is owed to, where a caller talking over
the tool call could have prompted it instead. Builds too old to report these
signals also keep the caveat, rather than being merged on an assumption. A caveat that appears on correct data
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
