# Why Not LangGraph Deep Agents (For the Core Pipeline)

A recurring question while designing [parallel execution](parallel_execution.md): LangGraph ships a
`deepagents` module with built-in planning and async subagents — doesn't that solve sparq's
planning and parallelism needs out of the box? This doc lays out why the answer is "no, not for
the Router → Planner → Executor → Aggregator → Saver pipeline" — and where deep-agents-style
infrastructure *does* genuinely fit on the roadmap.

Reference: https://docs.langchain.com/oss/python/deepagents/async-subagents

---

## What deepagents actually provides

| Capability | What it is |
|---|---|
| Planning tool | A generic todo-list tool an agent calls mid-conversation to track its own steps (same pattern as Claude Code's `TodoWrite`) |
| Async subagents | Background task lifecycle: `start_async_task` / `check_async_task` / `update_async_task` / `cancel_async_task` / `list_async_tasks`, served over an Agent Protocol server |
| Coding-agent tools | Generic file-system and shell/code-execution tools, in the spirit of a software-engineering agent |

It's a toolkit for building a *general-purpose autonomous coding/research agent* — closer in
shape to Claude Code itself than to a domain pipeline.

---

## 1. The planning tool is not a structured `Plan`

deepagents' planning tool produces a freeform todo list the LLM manages for itself — useful for
keeping a long agentic session organized, but it carries no schema.

sparq's [`Planner`](../src/sparq/architectures/v1/nodes/planner.py) produces a typed `Plan` —
a list of `Step` objects with `dataset_name`, `rationale`, `task_type` (and, after
[improvement #1](improvements.md), `depends_on`) — derived from `data/data_manifest.json`. This
isn't a convenience for the LLM; it's a contract the **executor programmatically depends on**: it
drives which datasets get loaded, what context gets built per step, how results get structured for
the aggregator, and (with the parallel-execution design) how steps get batched.

Swapping in a generic todo list would mean losing the one artifact that makes the pipeline
auditable, typed, and machine-consumable downstream. You'd be trading a schema for a string.

---

## 2. The REPL is sparq's domain-specific core — deepagents doesn't replace it

The most sophisticated part of sparq is [`tools/python_repl/`](repl_architecture.md): subprocess
isolation via `multiprocessing.spawn`, namespace persistence across steps via pickling, AST-based
last-expression capture, whitelisted package auto-install. This is purpose-built for
epidemiological data analysis workflows — load a dataset, transform it, keep variables alive
across plan steps, surface tracebacks cleanly.

deepagents gives you generic file-system and shell tools tuned for *coding* tasks. Adopting it
would not remove the need to design, build, and maintain a custom execution sandbox — that work
exists either way. The REPL is not a gap deepagents fills; it's sparq's actual contribution.

---

## 3. Async subagents are the wrong shape for intra-plan step parallelism

This is the one that looks the most tempting, because async subagents *do* run things
concurrently. But look at what they're built for versus what the executor needs:

```
                Figure 1.  Two different "parallel" problems
   ══════════════════════════════════════════════════════════════════════

   ASYNC SUBAGENTS (deepagents)              EXECUTOR PARALLEL STEPS (this pipeline)
   ─────────────────────────────             ─────────────────────────────────────
   "go research X"                           Plan = [Step1, Step2, Step3, Step4]
        │                                         (Step2, Step3 share no deps)
        ▼
   start_async_task ──► task_id                       ┌── Step2 ──┐
        │                                   Step1 ──► │           ├──► merge ──► Step4
        │ (user keeps chatting,             Step2 ──► │           │
        │  polls, updates, cancels)                   └── Step3 ──┘
        ▼
   check_async_task(task_id)                 asyncio.gather(step2, step3)
        │                                    — dispatched, awaited, merged,
        ▼                                      all within ONE executor turn —
   result, whenever ready                     no user interaction mid-flight

   Built for: long-running, interruptible,    Built for: fast fan-out/fan-in
   user-checkable background work             inside a single deterministic pass
```

Using async subagents for the executor's batch fan-out would mean polling task IDs and
reconstructing synchronous-feeling join semantics by hand — *more* moving parts than
`asyncio.gather` + per-step namespaces (see [`parallel_execution.md`](parallel_execution.md)),
which reuses primitives (`get_ns_path`, `cleanup_ns`, `load_ns`) that already exist. Forcing a
polling abstraction onto a problem that's naturally a blocking join is friction, not a fit.

---

## Where deep-agents-style infrastructure *does* fit

The user's stated direction — sparq becoming "a conversational system with researching
capabilities" — is a different product surface than the current single-shot pipeline. That surface
*is* the shape async subagents are designed for:

- User: "go investigate Salmonella outbreak patterns in the Midwest while I think about my next
  question" → `start_async_task`, return control immediately
- User keeps chatting, asks something else, comes back later: `check_async_task`
- User changes their mind mid-investigation: `update_async_task` / `cancel_async_task`

This maps directly onto items already on the v2 roadmap in [`improvements.md`](improvements.md) —
the multi-route router's `research_survey` route and the `web_search_node` — which are exactly the
kind of long-running, possibly-multi-turn research work that benefits from a checkable background
task model rather than blocking the conversation.

**Crucially, this is not an alternative to the typed pipeline — it's a thin shell *around* it.**
The pipeline doesn't get bypassed, duplicated, or rebuilt; it becomes the worker that the
conversational layer dispatches to and reports back from:

```
        Figure 2.  The conversational layer wraps the pipeline, it doesn't replace it
   ══════════════════════════════════════════════════════════════════════════════════

   User: "go research Salmonella outbreak patterns in the Midwest, I'll check back later"

        │
        ▼
   start_async_task(query)  ────────────────────┐
        │                                        │   user keeps chatting,
        │  (runs in the background)              │   asks other things,
        ▼                                        │   polls / cancels via
   await Agentic_system().run(query)             │   check_async_task(task_id)
        │                                        │
        │   ════ EXACT SAME PIPELINE, UNCHANGED ═╪═══════════════════════
        │                                        │
        ▼                                        │
   Router → Planner → Executor → Aggregator → Saver
                          │
                          ▼
              asyncio.gather over independent plan
              steps, each its own REPL namespace
              (parallel_execution.md)
        │
        ▼
   result becomes available via check_async_task(task_id) ──► back to user
```

The async-task wrapper changes *how the pipeline is invoked* (background + checkable, instead of
blocking) — it does not change *what the pipeline is* or *how it does the work*. The typed `Plan`,
the executor's batch-parallel steps, and the custom REPL all stay exactly as designed.

---

## Bottom line

| Question | Answer |
|---|---|
| Does deepagents replace the `Planner`? | No — its planning tool is a freeform todo list; the `Plan`/`Step` schema is a typed contract the executor depends on |
| Does deepagents replace the REPL? | No — it provides generic coding-agent tools, not a domain-specific sandboxed execution model |
| Do async subagents replace `asyncio.gather` for parallel plan steps? | No — they solve long-running, user-interruptible background work; intra-plan fan-out is a synchronous blocking join, better served by `asyncio.gather` + per-step namespaces |
| Is there a place for deepagents-style infrastructure on the roadmap? | Yes — as the *conversational layer* that dispatches long-running research tasks into the existing pipeline, once the multi-route router / `research_survey` work lands |

Adopting deepagents wholesale would mean re-architecting the pipeline around its conventions while
still having to build the same domain-specific planning schema and REPL — a larger rewrite for no
reduction in the actual engineering surface. The targeted approach — `asyncio.gather` for
intra-plan parallelism now, async-subagent-style task lifecycle for the conversational layer later
— reuses what already exists and adds infrastructure only where the product shape genuinely
requires it.
