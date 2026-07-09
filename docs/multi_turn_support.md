# Multi-Turn Conversation Support

Design document for adding multi-turn chat support to `State` and the v1 graph. SPARQ is meant to
be a chat application; `data/Q_dataset.json`'s `follow_ups` (conditional questions like "If high
risk -> could you give me a social media blurb?") are one eval slice of that, not the ceiling —
real usage should support arbitrary-depth conversations, not just one follow-up.

**Estimated implementation time: 5-8 days**, for one contributor already familiar with this
codebase — schema/reducer changes and the per-node `state.query`/`state.answer` indexing updates
are mechanical (~1-2 days); checkpointer + `thread_id` wiring into `system.py` and verifying
`executor_node`'s step-tracking resets cleanly across turns is the riskier middle chunk
(~2-3 days, most likely to surprise); wiring prior-turn context into `router`/`planner` prompts and
end-to-end testing against `Q_dataset.json`'s follow-ups is the remainder (~2-3 days). Assumes no
major LangGraph API surprises.

---

## Background: the `MessagesState` mixin attempt

The first attempt was `class State(BaseModel, MessagesState)` in `schemas/state.py`, following the
[reducers doc](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers)'s subclassing
pattern. This fails: `MessagesState` is a `TypedDict`, and Python raises a metaclass conflict when
mixing it into a `pydantic.BaseModel` via multiple inheritance.

Fix: don't subclass `MessagesState`. Since `State` is pydantic, add the `messages` field directly
with the same reducer `MessagesState` uses internally — LangGraph reads the reducer from field
metadata regardless of whether the schema is a `TypedDict` or a `BaseModel`:

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = []
    ...
```

---

## What goes in `messages` vs. what goes in `query`/`answer`

`aggregator_node` runs its report-writing step through `create_agent(...)`, which gives it tools
(`filesystemtools` for reading generated charts/CSVs). `agent.invoke(...)` returns that agent's
**entire internal messages list** for the call — tool-call/tool-result messages included — not a
single reply, unlike calling a raw chat model directly (the quickstart's `model.invoke([...])`
pattern, which returns one `AIMessage`).

Two options considered:

1. **Store only the final answer in `messages`** (`response["messages"][-1]`), matching how the
   executor already keeps its own internal ReAct/tool-call loop out of `state.results` (only the
   distilled `StepResult` is kept). Full chatter is discarded.
2. **Store the full turn (including internal chatter) in `messages`**, and rely on a separate,
   lighter channel for what actually gets fed back into future LLM calls.

Rejected (1) once it became clear the internal chatter isn't persisted anywhere else either —
`saver_node` only dumps whatever ends up in `State` after reducers merge each node's returned
update, so discarding it there means it's gone forever, not "logged elsewhere."

Settled on (2): `messages` stores full fidelity per turn (for `trace.json`/debugging), and
`query`/`answer` become the curated, conversational channel actually fed to prompts — no filtering
logic needed to strip tool-call noise back out of `messages` for that purpose.

---

## `query` / `answer` become accumulating lists, not scalars

Reducers govern merging concurrent updates to one key *within* a turn; they don't by themselves
decide whether a value survives into the *next* turn — that's checkpointing. With a plain
`str | None` (no reducer, default replace-on-write), the previous turn's `answer` already survives
into the next turn for free, as long as the node that would overwrite it doesn't run. That's enough
for exactly one level of follow-up (the dataset's case), but not for arbitrary-depth conversations
where every prior turn's query/answer should remain inspectable.

Chosen: a concatenating reducer, the same idiom as `add_messages` but generic list concatenation:

```python
query: Annotated[List[str], operator.add]
answer: Annotated[List[str], operator.add] = []
```

Current turn is `state.query[-1]` / `state.answer[-1]` (once set); prior turns are the slices before
that. `route`, `plan`, `completed_plan_steps`, and `results` stay scalar/replace — each new turn
needs a fresh plan and fresh execution, not a merge with the last turn's.

---

## Cross-turn persistence: checkpointer, not store

None of the above (`messages`, list-accumulating `query`/`answer`) survives between separate calls
to `Agentic_system.run()` on its own. Today, `run()` builds a fresh `graph_init.compile()` with no
checkpointer, and `state.py`'s reducers only govern how updates merge *within* one
`graph.astream()` call. Once that call returns, the `State` object is gone — the next `run()` call
starts from `{"query": [user_query]}` with everything else at field defaults. Reducers decide how
history accumulates; a checkpointer decides whether there's anything to accumulate *onto*.

LangGraph has two distinct persistence mechanisms and this is squarely the first one, not the
second:

- **Checkpointer**: snapshots the full graph state after every step, keyed by `thread_id`. This is
  exactly "resume this one conversation where it left off" — what multi-turn chat needs.
- **Store** (`BaseStore`): long-term memory shared *across* threads (e.g. a user's standing
  preferences visible in every conversation they ever have). SPARQ has no cross-conversation memory
  requirement right now — each conversation's history lives entirely in that conversation's own
  `messages`/`query`/`answer` — so a `Store` isn't needed for this piece of work.

Backend choice: start with `langgraph.checkpoint.memory.InMemorySaver` — sufficient for interactive
CLI use and for the `Q_dataset` batch-eval runner (`experiments/00.py`), since both live for one
process lifetime. It does **not** survive a process restart, so a real deployed chat surface (e.g.
`web_ui_migration.md`) would need a persistent checkpointer (`SqliteSaver` or a Postgres-backed one)
before conversations could outlive a server restart — noted here as a follow-up, not needed for the
CLI/eval use case this doc is scoped to.

`thread_id` also needs a source distinct from the existing `run_id`: `run_id` is regenerated fresh
inside `run()` on every call today (`system.py:77`) and is used for per-run namespace cleanup
(`cleanup_run(run_id)`) — it identifies one *turn's* execution, not the conversation. `thread_id`
needs to be supplied by the caller and held constant across turns of the same conversation (e.g. the
CLI keeps one `thread_id` for the life of a chat session; `experiments/00.py` would use one
`thread_id` per top-level question so its `follow_ups` continue that thread, and a new `thread_id`
per top-level question so unrelated questions don't share history).

---

## Task list

- [ ] `schemas/state.py`: `query`/`answer` → `Annotated[List[str], operator.add]`
- [ ] `router.py:21`: `HumanMessage(content=state.query)` → `state.query[-1]`
- [ ] `planner.py:37`: same change
- [ ] `aggregator.py:60` (`user_query=state.query`): → `state.query[-1]`
- [ ] `aggregator_node`: return `{'answer': [answer], 'messages': response["messages"]}` (list-wrap
      for the reducer; full internal chatter into `messages`)
- [ ] `saver.py`: concise `final_answer.json` pair becomes full lists naturally — arguably nicer,
      shows the whole conversation instead of just the last exchange. Add a second output,
      `trace_summary.json`, via `state.model_dump(exclude={'messages'})`, so there's a readable
      trace without the verbose tool-call chatter alongside the full-fidelity `trace.json`.
- [ ] `system.py` `Agentic_system.run()`: wire up the checkpointer per "Cross-turn persistence"
  above — compile with `InMemorySaver`, accept/track a `thread_id` distinct from `run_id`, pass it
  via `config={"configurable": {"thread_id": ...}}`, and change `input_data` to
  `{"query": [user_query], "messages": [HumanMessage(content=user_query)]}` (lists, since both
  fields use concatenating reducers now)
- [ ] Verify `executor_node` resets step-tracking state (`completed_plan_steps`, in-flight step IDs)
  correctly for a new turn — risk of turn-1 state leaking into turn-2's execution once the graph is
  checkpointed across turns instead of built fresh per call.
- [ ] `router_node`/`planner_node`: use `state.query[:-1]` zipped with `state.answer` to give the
  LLM the prior conversation when interpreting a follow-up (the actual payoff of this whole change —
  today, a follow-up like "If high risk -> could you give me a blurb?" is routed/planned with zero
  context on what "high risk" refers to).
