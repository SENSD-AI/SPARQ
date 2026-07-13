# Multi-Turn Conversation Support

Design document for adding multi-turn chat support to `State` and the v1 graph. SPARQ is meant to
be a chat application; `data/Q_dataset.json`'s `follow_ups` (conditional questions like "If high
risk -> could you give me a social media blurb?") are one eval slice of that, not the ceiling —
real usage should support arbitrary-depth conversations, not just one follow-up.

**Estimated implementation time: 8-11 days**, for one contributor already familiar with this
codebase. Schema/reducer changes and the per-node `state.query`/`state.answer` indexing updates
are mechanical (~1 day); checkpointer + `thread_id` wiring into `system.py`, including provisioning
an actual Postgres instance (~2-3 days); REPL namespace continuity across turns — determinism,
turn-scoped step keys, seed/merge-back at turn boundaries — is the riskiest and least mechanical
chunk (~2-3 days); the router heuristic rewrite plus wiring prior-turn context into
`router`/`planner`/`aggregator` prompts (~1-2 days); trace/saver restructuring and end-to-end
testing against `Q_dataset.json`'s follow-ups is the remainder (~1-2 days). This revises an earlier
5-8 day estimate upward twice: the first revision (to 7-10 days) hadn't accounted for REPL namespace
continuity or the router heuristic problem below; the second (to 8-11 days) accounts for switching
the checkpointer backend from `AsyncSqliteSaver` to `AsyncPostgresSaver` (see "Cross-turn
persistence" below), which trades a zero-infra file for a real database that needs provisioning,
a connection string/secret, and network reachability. Assumes no major LangGraph API surprises.

---

## Background: the `MessagesState` mixin attempt

The first attempt was `class State(BaseModel, MessagesState)` in `schemas/state.py`, following the
[reducers doc](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers)'s subclassing
pattern. This fails: `MessagesState` is a `TypedDict`, and Python raises a metaclass conflict when
mixing it into a `pydantic.BaseModel` via multiple inheritance.

Fix: don't subclass `MessagesState`. Since `State` is pydantic, add the `messages` field directly
with the same reducer `MessagesState` uses internally — LangGraph reads the reducer from field
metadata regardless of whether the schema is a `TypedDict` or a `BaseModel`. This part of `State` is
already in place today:

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = []
    ...
```

Nothing writes to it yet — every node still builds its prompt from `state.query` (a bare string)
alone. Making `messages` load-bearing is most of what follows.

---

## Conversation history channel: `messages`, not list-accumulating `query`/`answer`

An earlier pass considered turning `query`/`answer` into concatenating-reducer lists —
`Annotated[List[str], operator.add]` — reasoning that a plain `str | None` field (no reducer,
default replace-on-write) only gives you the *previous* turn's answer for free once checkpointing
is in place, not arbitrary-depth history. That's true, but the conclusion doesn't require a second,
bespoke accumulating channel: `state.messages` already does exactly this job, using the reducer
LangChain ships for it.

Settled design: `query` and `answer` stay scalar and represent only the *current* turn — that's all
`router_func`'s conditional edge and the executor's plan-execution loop need. `state.messages`
becomes the single accumulating history channel, and gets exactly one `HumanMessage` (the raw user
query) and one `AIMessage` (the curated final answer — whichever of router's direct answer or
aggregator's narrative actually ran) appended per turn.

This deliberately does **not** store a turn's internal chatter — the executor's per-step ReAct
loops, or the aggregator's own tool calls when it reads generated files. That's consistent with
existing behavior: the executor already discards its internal loop and keeps only the distilled
`StepResult` (via `response["structured_response"]`); the structured record of what happened in a
turn continues to live in `state.results` / `trace.json`, not in `messages`. `messages` is the
curated, conversational channel actually fed to prompts — no filtering logic needed to strip
tool-call noise back out of it later.

Concretely: `router_node` appends `HumanMessage(content=state.query)` to `messages` (via the
existing `add_messages` reducer) alongside its own classification call. `saver_node` is the common
exit node for *both* the `True` and `False` branches of `router_func`, so it's the single choke
point that appends `AIMessage(content=state.answer)` — no duplicated logic across branches.

---

## Cross-turn persistence: checkpointer, not store

None of the above (`messages` accumulation) survives between separate calls to
`Agentic_system.run()` on its own. Today, `run()` builds a fresh `graph_init.compile()` with no
checkpointer, and `state.py`'s reducers only govern how updates merge *within* one
`graph.astream()` call. Once that call returns, the `State` object is gone — the next `run()` call
starts from `{"query": user_query}` with everything else at field defaults. Reducers decide how
history accumulates; a checkpointer decides whether there's anything to accumulate *onto*.

LangGraph has two distinct persistence mechanisms and this is squarely the first one, not the
second:

- **Checkpointer**: snapshots the full graph state after every step, keyed by `thread_id`. This is
  exactly "resume this one conversation where it left off" — what multi-turn chat needs.
- **Store** (`BaseStore`): long-term memory shared *across* threads (e.g. a user's standing
  preferences visible in every conversation they ever have), with production backends
  (`PostgresStore`, `MongoDBStore`, `RedisStore`) and optional semantic search. SPARQ has no
  cross-conversation memory requirement right now — each conversation's history lives entirely in
  that conversation's own `messages` — so `Store` is out of scope for this work. Noted here as a
  documented future extension (e.g. remembering a user's preferred chart style across unrelated
  research threads), not something to design or implement now.

Backend choice: **`AsyncPostgresSaver`** (package `langgraph-checkpoint-postgres`, plus its driver
dependencies — `uv add langgraph-checkpoint-postgres "psycopg[binary,pool]"`), not `InMemorySaver`
and not `AsyncSqliteSaver`. SPARQ is a chat application; an in-memory checkpointer loses every
conversation on process restart, which isn't acceptable even for a v1. `AsyncSqliteSaver` was the
initial pick (file-backed, zero infra), but its own docstring states it "is not recommended for
production workloads due to limitations in SQLite's write performance... consider a more robust
database like PostgreSQL" — direct guidance from the library, not just an inference that Sqlite
tends to age poorly. With Postgres available through the lab's AWS credits (e.g. RDS), there's no
reason to build against the backend the maintainers themselves steer away from.

Reading the actual source (`langgraph/checkpoint/postgres/aio.py`, mirroring the earlier check
against the Sqlite source rather than trusting the reference page alone) surfaces real differences
from Sqlite, not just a class-name swap:

- **`setup()` is the opposite of Sqlite's.** `AsyncSqliteSaver.setup()` is automatic/idempotent and
  "should not be called directly by the user." `AsyncPostgresSaver.setup()` is the reverse — its
  docstring says it "creates the necessary tables ... and runs database migrations. It MUST be
  called directly by the user the first time checkpointer is used." One explicit
  `await checkpointer.setup()` call is needed; safe to run on every process start since it checks a
  `checkpoint_migrations` table before applying anything.
- **Constructor takes a connection *or* a pool.** `AsyncPostgresSaver(conn, pipe=None, serde=None)`
  accepts either a single `psycopg.AsyncConnection` or a `psycopg_pool.AsyncConnectionPool`. This
  matters more here than for Sqlite: `executor_node` already runs plan steps concurrently via
  `asyncio.gather` (parallel worker agents), and `AsyncSqliteSaver` serializes all access behind a
  single `asyncio.Lock` around one connection regardless of backend. A real `AsyncConnectionPool`
  lets those concurrent workers' checkpoint writes actually run in parallel instead of queueing
  behind a lock — use a pool, not a bare connection, when wiring this up.
- **Same `async with ... .from_conn_string(...)` entry point and lifecycle requirement as Sqlite.**
  The connection (or pool) still needs to be opened once and held for the session's lifetime, not
  reopened per `run()` call — the `_get_node_definitions()`/`_build_graph()` restructuring noted
  below applies regardless of which backend this is.
- **`adelete_thread(thread_id)`** is present here too, confirmed — `end_session` doesn't need a
  backend-specific implementation.

This does add real infrastructure Sqlite didn't need: an actual reachable Postgres instance,
a connection string/DSN treated as a secret (belongs in `ENVSettings`, same pattern as the existing
`aws_profile`/api-key fields — not `PathSettings`), and network reachability from wherever SPARQ
runs. That's the reason the top-line estimate moved up by about a day.

`thread_id` needs to be supplied by the caller and held constant across turns of the same
conversation — distinct from today's `run_id`, which is regenerated fresh inside `run()` on every
call (`system.py:77`). Rather than keep both identifiers, `thread_id` takes over `run_id`'s existing
job (scoping `config["configurable"]`) *and* becomes the REPL namespace key (see below) — a single
identifier per conversation. `experiments/00.py` would use one `thread_id` per top-level question in
`Q_dataset.json` so its `follow_ups` continue that thread, and a new `thread_id` per top-level
question so unrelated questions don't share history.

---

## REPL namespace continuity across turns

This is the piece the original pass missed entirely, and it's the riskiest part of the whole change.
Getting the graph's `State` to persist via a checkpointer says nothing about the executor's REPL
namespace — the pickled dict of loaded dataframes/variables that workers execute code against — and
that namespace has its own, separate lifecycle problem.

**Problem 1 — namespace paths aren't durable.** `namespace.py`'s `get_ns_path` allocates a path via
`tempfile.mkstemp` and caches it in a process-local `_ns_paths: dict[str, str]`. Neither survives a
process restart, which defeats the entire reason for choosing a file-backed checkpointer: `State`
would durably remember a conversation, but the dataframes it refers to would be gone. Fix: derive
the path deterministically from `thread_id` (a namespace directory from settings + `f"{thread_id}.pkl"`)
instead of a random tempfile, so a restarted process can find the same file without depending on
`_ns_paths` having survived.

**Problem 2 — step keys collide across turns.** `executor.py` keys per-step namespaces as
`f"{run_id}_step_{id}"`, and `Plan.steps` IDs restart at 1 on every new plan (planner produces a
fresh `Plan` each time `route_func` returns `True`). Reusing `thread_id` directly as the base key
would make turn 3's step 1 silently collide with turn 1's step 1. Fix: introduce a turn index
(derivable once `messages` is wired up — count of `HumanMessage`s — or an explicit `turn_count`
field on `State`) and scope step keys as `f"{thread_id}_turn{n}_step_{id}"`.

**Problem 3 — turns should be able to build on each other's data.** A follow-up like "now filter
that to 2020" should operate on dataframes a previous turn already loaded, not force every new turn
to start from an empty namespace and reload everything. Fix: maintain one canonical thread-level
"committed" namespace per `thread_id`. At the start of a new turn's executor run, seed that turn's
dependency-free steps from the thread-level namespace — the same merge mechanism
`merge_namespaces_of_previous_deps` already uses for within-turn dependencies, applied once more at
the turn boundary. When a turn finishes, merge its completed steps' final namespaces back into the
thread-level namespace so the *next* turn inherits them.

**Lifecycle.** Remove the unconditional `cleanup_run(run_id)` currently in `Agentic_system.run()`'s
`finally` block — as written today it wipes the namespace after every single turn, which is exactly
what multi-turn continuity requires it not to do. Add an explicit `Agentic_system.end_session(thread_id)`
for a caller to invoke when a conversation actually ends (deletes the namespace file, and the
checkpoint thread if the installed LangGraph version exposes thread deletion). Nothing calls this
automatically — that's a serving-layer decision, out of scope here.

---

## Router heuristic collapse under multi-turn

Caught during review, and worth documenting precisely because it's not obvious until you look for
it: giving the router conversation history doesn't just help it resolve "that"/"this" — it can
*break* its own classification if the classification criterion doesn't change too.

The current heuristic (`router_message.txt`): "If it's a question related to salmonella or
socio-economics, ALWAYS defer to planner." In a single-turn setting, topic-membership is a crude but
workable proxy for "does this need data." In an ongoing conversation thread, that proxy stops
correlating with anything useful — once the whole thread *is* a salmonella analysis, every follow-up
is trivially "about salmonella," so `True` becomes the only reachable branch. "What year was that,"
"can you rephrase that," and "what did you mean by GLM" would all trigger a full
plan → parallel-executor cycle, exactly the outcome multi-turn support should avoid.

**Fix**: rewrite the router's decision criterion from "is this in-domain" to "does answering this
require new or updated computation over the data that hasn't already been produced in this
conversation." `True` = needs a new query, filter, model, or chart. `False` = answerable by
reasoning over the conversation history already present — recall, clarification, rephrasing,
summary, or general knowledge. Add a couple of anchoring few-shot examples to `router_message.txt`
distinguishing the two ("user asks to restate a number already given → False"; "user asks to filter
by a new variable → True"). Same boolean output, same graph shape — no new node or edge, just a
corrected prompt.

**Scope decision**: the `False` branch only gets `state.messages` (text), not raw `state.results`.
It cannot correctly answer a request for something that *was* computed in a previous turn but never
stated in the narrative answer (exact figures, "show me that chart again") — it should conservatively
fall back to `True` (recompute) rather than guess, which is correct but occasionally redundant.
Accepted as the v1 tradeoff. A richer extension — exposing the last turn's `state.results` to the
direct-answer path so it can recall precise detail without recomputing — is a documented future
option, not planned now.

---

## Trace/saver semantics across turns

`PathSettings.set_run_dir`'s `model_validator` computes `run_dir` once, at `Settings`/
`Agentic_system` construction time. Every `run()` call on the same instance today writes to the same
`trace.json`/`final_answer.json`, silently overwriting the previous call's output — independently
flagged in `todo.md` for the `experiments/00.py` batch-question loop (same root cause: looping
`run()` calls on one instance collides on `run_dir`).

Fix: scope trace output per `thread_id` and per turn — e.g. `run_dir / thread_id / turn_{n}_trace.json`
— so nothing is overwritten and a full session is reconstructable from its directory. `final_answer.json`
becomes either one file per turn, or a single running `conversation.json` appended to each turn (cheaper,
and arguably more useful for reading back a whole conversation at once). This is a default choice, not
a hard requirement — flagging it as such rather than deciding unilaterally what the on-disk layout
should be.

---

## Task list

Ordered by build dependency, not by doc-section grouping: checkpointer plumbing has to land first,
since nothing about `messages` accumulation is even testable while `State` still resets on every
`run()` call. Per-item estimates sum to ~8.5 days, within the 8-11 day range quoted at the top (the
remaining slack covers the "no major LangGraph API surprises" caveat and integration friction
between items that isn't visible until they're wired together).

- [ ] Provision a Postgres instance (~0.5 day): stand up a reachable database (e.g. AWS RDS via the
  lab's credits), obtain a connection string, confirm network reachability from wherever SPARQ runs.
  Infra work, not code — prerequisite for everything below.
- [ ] `settings.py` / `V1Settings` (~0.5 day): add the Postgres connection string as an `ENVSettings`
  field (secret/credential, same pattern as `aws_profile`/api keys — not `PathSettings`), plus a
  namespace directory path (unrelated to the checkpointer backend, still needed for REPL namespace
  continuity below). Prerequisite for both `system.py` and `namespace.py`.
- [ ] `system.py` (`Agentic_system`) (~1.5 days): add `langgraph-checkpoint-postgres` and
  `psycopg[binary,pool]` dependencies. Restructure so `AsyncPostgresSaver.from_conn_string(...)` —
  backed by an `AsyncConnectionPool`, not a bare connection, given the executor's existing
  `asyncio.gather` parallelism — is opened once and held for the life of the session (an `async with`
  wrapping the session, or an explicit `astart()`/`aclose()` pair) instead of
  `_get_node_definitions()`/`_build_graph()` re-running fresh inside every `run()` call as they do
  today. Call `await checkpointer.setup()` once explicitly — required and idempotent, opposite of
  Sqlite's automatic behavior. Change `run()`'s signature to accept `thread_id` from the caller; pass
  it via `config={"configurable": {"thread_id": thread_id}}`; use it (not a fresh per-call `run_id`)
  as the REPL namespace key. Remove the unconditional `cleanup_run(...)` from `finally`. Add
  `end_session(thread_id)` calling `await checkpointer.adelete_thread(thread_id)` for explicit
  conversation teardown. Once this lands, cross-turn `State` persistence exists and the `messages`
  work below becomes verifiable.
- [ ] `router.py` (~0.5 day): append `HumanMessage(content=state.query)` to `state.messages` (via the
  existing `add_messages` reducer) — the first of the two writes that make `messages` accumulate.
  Also rewrite `router_message.txt`'s decision heuristic (topic-membership → "needs new computation"),
  with anchoring examples, and pass `state.messages` history into the router's LLM call.
- [ ] `saver.py` (~0.5 day): append `AIMessage(content=state.answer)` to `state.messages` — the
  second write; this is the shared exit node for both `router_func` branches, so it's the one place
  it needs to happen. Restructure output paths per "Trace/saver semantics" above
  (`run_dir / thread_id / turn_{n}_trace.json` or equivalent).
- [ ] `planner.py` (~0.25 day): pass `state.messages` history into the planner's LLM call so it
  knows what a prior turn already established, in addition to the existing `data_context`. Depends
  on the router/saver writes above actually populating `messages`.
- [ ] `aggregator.py` (~0.25 day): pass `state.messages` history into the aggregator's LLM call. No
  change needed to its existing `state.results` handling — because that field has no reducer, it
  already naturally survives into a turn where the executor didn't run, which is what the router's
  `False` fallback path relies on implicitly.
- [ ] `namespace.py` (~1.5 days): `get_ns_path` derives a deterministic path from `thread_id` instead
  of `tempfile.mkstemp`. Add turn-scoped step keys (`f"{thread_id}_turn{n}_step_{id}"`). Add
  seed-from-thread-namespace and merge-back-to-thread-namespace helpers for turn boundaries.
- [ ] `executor.py` (~1.5 days): adopt turn-scoped step keys; call the new seed/merge-back helpers at
  the start and end of a turn's execution loop. Verify `completed_plan_steps`/`results` reset
  correctly for a new `True`-routed turn (they already do — `executor_node`'s return replaces both
  fields wholesale each call, no reducer involved) and confirm it's acceptable that a `False`-routed
  turn leaves the *previous* `True` turn's `plan`/`completed_plan_steps`/`results` sitting untouched
  in the checkpoint (harmless: the `False` branch goes straight to `saver`, which doesn't read them).
- [ ] `experiments/00.py` / `todo.md`'s `Q_dataset` items (~0.5 day): use one `thread_id` per
  top-level question in `data/Q_dataset.json`, iterate its `follow_ups` within that same thread so
  they actually test multi-turn behavior instead of running standalone. This also resolves the
  `run_dir` collision `todo.md` flags, once trace output is scoped per `thread_id`/turn as above.
- [ ] Tests (~1 day): namespace path is stable across a simulated restart (fresh process-local
  `_ns_paths`, same file on disk); two sequential `run()` calls with the same `thread_id` show
  `messages` accumulating and turn 2's planner/aggregator prompts containing turn 1's content;
  step-namespace keys don't collide across turns; router prompt returns `False` for
  recall/clarification-style follow-ups and `True` for ones requiring new computation (example-based
  unit tests against the rewritten prompt); end-to-end test against a `Q_dataset.json` entry with
  `follow_ups`.
