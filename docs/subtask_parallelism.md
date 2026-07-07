# Sub-task Parallelism Within a Worker (`async_subagent_tool`)

Design document for improvement #16: letting a single plan-step worker (`execute_single_step_worker`
in `executor.py`) parallelize independent sub-tasks it identifies mid-execution, where those
sub-tasks need to write variables back into the step's namespace — not just report text.

**Implementation lives in**: `src/sparq/tools/async_subagent/subagent.py`
(`make_async_subagent_tool(ns_path)` → `async_subagent_tool(task: str)`)

---

## Problem

The executor already parallelizes at the **plan-step** level: independent `Step`s in a batch each
get their own worker agent and their own namespace (`docs/parallel_execution.md`). There is no
equivalent one level down — if a single step's worker realizes its own task decomposes into
independent sub-parts (e.g. "compute three unrelated summary statistics over `df1`"), it currently
has no way to fan those out; it does them one tool call at a time in its own agentic loop.

---

## Why not `deepagents.middleware.SubAgentMiddleware` (the generic `task` tool)

`SubAgentMiddleware._get_subagents()` builds each subagent's tools — and thus its REPL tool —
**once, at agent-construction time**, not per `task` call. Every `task` call routed to the same
`subagent_type` reuses the identical REPL tool bound to the identical namespace path
(`make_python_repl_tool(ns_path)` closes over one fixed path).

If the worker's LLM fires multiple `task` calls in parallel in one turn, two concurrent invocations
of the same subagent would both read-modify-write the same pickle file: a lost-update race. This is
the same class of bug `docs/improvements.md` #1 already solved at the step level
(seed/execute/merge/cleanup per-batch namespaces); the generic `task` tool has no equivalent,
because its subagents are static, pre-compiled runnables, not built fresh per call.

**If sub-tasks only needed to report text back** (no persisted variables), the stock `task` tool
would work fine in parallel — give the subagent's REPL tool `ns_path=None` and there's nothing to
race on. That's not the case here: sub-tasks need to leave DataFrames/variables behind for the
parent step to use afterward, so a purpose-built tool with per-call namespace isolation is required.

---

## Why not a single `spawn_subtasks(subtasks: list[str])` tool (rejected alternative)

The first design considered here was one tool that takes a batch of sub-task descriptions and fans
them out internally via `asyncio.gather`. It was rejected for a reason specific to how the worker
already tracks its own task list, not because the batching itself is unsound.

The worker agent runs with `TodoListMiddleware` (`executor.py`, already adopted). Its `Todo` schema
(`langchain/agents/middleware/todo.py`) is:

```python
class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]
```

A flat list, no dependency field, and — critically — **no code wiring it to any other tool's
arguments**. `write_todos` exists purely for the model's own self-tracking. For a
`subtasks: list[str]` tool, the model would have to manually re-derive and re-type sub-task
descriptions into a fresh list argument at the moment of the call, duplicating data that may already
be sitting in its todo list, with nothing keeping the two in sync. Worse, `write_todos`'s own tool
description already documents a *different*, native idiom for "these are independent and
parallelizable":

> "in_progress: Currently working on (you can have multiple tasks in_progress at a time **if they
> are not related to each other and can be run in parallel**)"

A single batching tool asks the model to do something the todo tool never primes it to do — pause,
gather several todo contents, and package them into one unusual list-argument call — cutting against
the middleware's native mark-in_progress → work → mark-completed flow instead of working with it.

---

## Chosen design: one tool, one sub-task, worker-driven parallel dispatch

Instead of a tool that internally fans out over a list, expose a single-task tool and let the
worker's own LLM parallelize by emitting multiple tool calls for it in one turn — the same way it
would call any other tool multiple times per turn.

```
                Figure 1.  Step-level vs sub-task-level parallelism
   ══════════════════════════════════════════════════════════════════════

   STEP-LEVEL (existing, parallel_execution.md)   SUB-TASK-LEVEL (this doc)

   Plan = [Step1, Step2, Step3]                   Step 2's worker marks 3 todos in_progress,
        (Step2, Step3 independent)                then emits 3 async_subagent_tool calls in
                                                   one turn (one per todo)

        ┌── Step2 ──┐                                   ┌── Sub 0 ──┐
   Step1 │           ├──► merge ──► ...        Step2 ──► ├── Sub 1 ──┼──► merge ──► Step2 continues
        └── Step3 ──┘                                   └── Sub 2 ──┘

   dispatched by executor_node,                   dispatched by the worker's own LLM turn;
   asyncio.gather over worker agents               LangGraph's ToolNode gathers concurrent
   each with sub_agent_id = "{run_id}_step_{i}"    tool calls from one AIMessage automatically
```

### This is not a hypothetical capability — it's already how the framework executes tool calls

`create_agent` (used by `execute_single_step_worker`) runs tool calls through
`langgraph.prebuilt.tool_node.ToolNode`. Confirmed directly from the installed source
(`ToolNode._afunc`): when an `AIMessage` contains multiple tool calls, they are dispatched
concurrently —

```python
outputs = await asyncio.gather(*coros)
```

So if the worker's model emits N `async_subagent_tool` calls in one turn, they already run in
parallel with zero custom orchestration. `async_subagent_tool` itself only needs to handle **one**
sub-task per invocation.

```python
# src/sparq/tools/async_subagent/subagent.py
import uuid

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from sparq.schemas.output_schemas import StepResult
from sparq.tools.python_repl.namespace import get_ns_path, load_ns, save_ns, cleanup_ns, get_ns_lock
from sparq.tools.python_repl.python_repl_tool import make_python_repl_tool
from sparq.tools.data_discovery_tools import make_load_dataset_tool

SUBTASK_SYSTEM_PROMPT = """You are handling one independent sub-part of a larger analytical step.
You have access to the variables already computed by the parent step. Use python_repl_tool with
persist_namespace=True so any new variables you create are kept and merged back for the parent
step to use afterward. Report your findings in execution_results."""


def make_async_subagent_tool(ns_path: str, llm: BaseChatModel):
    """Factory for an `async_subagent_tool` scoped to one parent step's namespace.

    Must be constructed fresh per worker (like `make_python_repl_tool`), closing over
    that worker's own `ns_path`, so sub-namespaces are seeded from and merged back
    into the correct parent step.
    """

    @tool
    async def async_subagent_tool(task: str) -> str:
        """Delegate one independent sub-task to a coder agent with access to your current
        variables. Call this multiple times in the same turn for independent sub-tasks —
        they run in parallel. New variables the sub-task creates are merged back into your
        namespace. Do not use this for sub-tasks that depend on each other's output.

        Args:
            task: Natural-language description of one independent sub-task to run.
        """
        sub_id = f"{ns_path}_sub_{uuid.uuid4().hex[:8]}"

        async with get_ns_lock(ns_path):
            base_ns = load_ns(get_ns_path(ns_path))
        save_ns(get_ns_path(sub_id), dict(base_ns))  # seed

        sub_tools = [
            make_load_dataset_tool(get_ns_path(sub_id)),
            make_python_repl_tool(get_ns_path(sub_id)),
            # deliberately no async_subagent_tool here — caps recursion at one level
        ]
        sub_agent = create_agent(
            model=llm,
            tools=sub_tools,
            system_prompt=SUBTASK_SYSTEM_PROMPT,
            response_format=StepResult,
        )
        try:
            response = await sub_agent.ainvoke({"messages": [{"role": "user", "content": task}]})
            result: StepResult = response["structured_response"]
        except Exception as e:
            result = StepResult(id=0, step=task, execution_results="", misc=f"Subtask failed: {e}")

        # Merge back into the shared parent namespace. Locked because this tool call runs
        # concurrently with sibling async_subagent_tool calls from the same worker turn —
        # without the lock, two concurrent read-modify-write cycles race (lost update).
        async with get_ns_lock(ns_path):
            merged = load_ns(get_ns_path(ns_path))
            merged.update(load_ns(get_ns_path(sub_id)))
            save_ns(get_ns_path(ns_path), merged)

        cleanup_ns(sub_id)
        return f"{result.execution_results}\n{result.misc}".strip()

    return async_subagent_tool
```

---

## The merge-back race — and why this design still needs a lock

Design A (the rejected batching tool) avoided any race because all N sub-tasks were gathered inside
*one* tool invocation, with a single sequential merge step after `asyncio.gather` returned. This
design has the opposite shape: N independent tool invocations, each with its own merge step, running
concurrently. Two `async_subagent_tool` calls both doing "read `ns_path` → merge my result → write
`ns_path`" without coordination is a textbook lost-update race — whichever call writes last silently
discards the other's merge.

Each sub-task's *own* namespace (`sub_id`, freshly generated per call) is never shared, so there's no
race in seeding or in running the sub-agent itself — only in the two touchpoints where a call reads
or writes the shared parent `ns_path`. An `asyncio.Lock` scoped to `ns_path` around just those
touchpoints (not around the sub-agent's actual work) serializes the merge without serializing the
computation:

```python
# src/sparq/tools/python_repl/namespace.py — new addition
import asyncio

_ns_locks: dict[str, asyncio.Lock] = {}

def get_ns_lock(ns_id: str) -> asyncio.Lock:
    """Returns a lock scoped to one namespace ID, creating it on first use.
    Used to serialize concurrent read-modify-write merges into a shared namespace
    (see async_subagent_tool) without serializing the underlying computation."""
    if ns_id not in _ns_locks:
        _ns_locks[ns_id] = asyncio.Lock()
    return _ns_locks[ns_id]
```

This mirrors the `_ns_paths` module-level registry pattern already in `namespace.py`. Cleanup:
`_ns_locks` entries for a `run_id` should be swept alongside `cleanup_run(run_id)` to avoid an
unbounded dict over a long-lived process — worth folding into `cleanup_run` rather than adding a
separate sweep.

---

## Namespace lifecycle

```
                Figure 2.  async_subagent_tool namespace lifecycle (one call)
   ══════════════════════════════════════════════════════════════════════

   ns_path  (the calling worker's own namespace — shared across sibling parallel calls)

        [lock] read ns_path ──► seed sub_id (unique per call, no lock needed)
        [unlock]
                 │
           Sub-agent (ainvoke) — runs fully concurrently with sibling calls,
           no shared state touched during this phase
                 │
        [lock] merge: ns.update(load_ns(sub_id)); save_ns(ns_path, ns)
        [unlock]
                 │
           cleanup_ns(sub_id)
```

Only the two bracketed sections are serialized across concurrent sibling calls; sub-agent execution
(the expensive part — LLM calls, REPL subprocess execution) is fully parallel.

**Safety net**: sub-namespace IDs (`f"{ns_path}_sub_{uuid}"`) already start with the same
`f"{run_id}_step_"` prefix `cleanup_run(run_id)` sweeps at the end of a run (verified: `cleanup_run`
matches on `key == run_id or key.startswith(prefix)`), so orphaned sub-namespace files are removed
when the run ends even if a call's own `cleanup_ns` is skipped due to an exception.

---

## Recursion cap

Sub-agents spawned by `async_subagent_tool` get `load_dataset` and `python_repl_tool`, but **not**
`async_subagent_tool` itself — recursion is capped at exactly one level. This avoids unbounded
fan-out/cost and keeps the merge model (flat, one parent + N children) simple. If nested fan-out is
ever needed, revisit this — it is a deliberate simplicity trade-off, not a technical limitation.

---

## Open questions / follow-ups

- **Parallel dispatch is conditional on model behavior, not code-guaranteed.** Unlike the rejected
  batching design (where `asyncio.gather` inside one tool call always ran everything together by
  construction), this design's concurrency only happens if the worker's underlying model actually
  chooses to emit multiple `async_subagent_tool` calls in one turn. Whether Claude-on-Bedrock
  reliably does this for the worker's prompt/tool setup is **untested**. If the model tends to call
  tools one at a time, this design silently degrades to fully sequential sub-task execution — same
  wall-clock cost as no fan-out, no error raised. Needs empirical verification before relying on it;
  if unreliable, the worker's system prompt may need explicit steering (e.g. "when you mark multiple
  todos in_progress because they're independent, call `async_subagent_tool` for all of them in the
  same turn").
- **Concurrency cap**: the rejected batching design had `MAX_SUBTASKS` as an upfront list-length
  check. That doesn't translate directly here, since calls are independent and dispatched by the
  model rather than validated as one batch. If a cap is needed, it would have to be a semaphore
  scoped to `ns_path`, limiting how many `async_subagent_tool` calls run concurrently for one step
  regardless of how many the model fires in one turn.
- **Cost**: each `async_subagent_tool` call is one additional LLM agent loop. No budget/backoff
  exists yet for a worker that calls this heavily.
- **Context bloat after merge**: `execute_single_step_worker` already warns
  (`MAX_NAMESPACE_VARS_WARNING`) when a step inherits too many variables from dependencies; merged
  sub-task variables should be checked against the same threshold.
- **Schema size**: `task: str` is one required field — smaller than the rejected
  `subtasks: list[str]` design, which matters given `docs/improvements.md` #14 (Bedrock
  grammar-size blowup driven by tool schema complexity, not just tool count).
- Cross-reference: `docs/why_not_deepagents.md`'s conclusion that intra-plan fan-out is a blocking
  join, not a background/pollable task, applies identically one level down — no need to reconsider
  `deepagents`' async subagents for this.

---

## Integration points

| File | Change | Why |
|------|--------|-----|
| `src/sparq/tools/async_subagent/subagent.py` | `make_async_subagent_tool(ns_path, llm)` factory (scaffolded, not yet implemented) | Isolated, testable; mirrors existing tool-factory pattern |
| `src/sparq/tools/python_repl/namespace.py` | Add `get_ns_lock(ns_id)`; fold `_ns_locks` cleanup into `cleanup_run` | Serializes the merge-back race described above |
| `src/sparq/architectures/v1/nodes/executor.py` | In `execute_single_step_worker`, add `make_async_subagent_tool(step_ns_id, llm_object)` to `_tools` | Gives the step's worker access to the new tool |
| `src/sparq/architectures/v1/prompts/executor_message.txt` | Add an `async_subagent_tool` bullet to the "Tools at your disposal" list, with guidance tying it to the todo list's in_progress-parallel idiom | Worker needs to know the tool exists, when to use it, and how it composes with `write_todos` |

### Suggested prompt addition

```
- async_subagent_tool(task: str): Runs one independent sub-part of your current task, with its own
  copy of your current variables. New variables it creates are merged back into your namespace
  afterward. Only use this for sub-tasks that are truly independent of each other (no sub-task
  needs another's output). When you mark multiple todos in_progress because they're unrelated and
  can run in parallel, call this tool once per todo in the same turn to actually run them
  concurrently — for sequential/dependent work, just do it directly instead.
```
