# Sub-task Parallelism Within a Worker (`spawn_subtasks`)

Design document for improvement #16: letting a single plan-step worker (`execute_single_step_worker`
in `executor.py`) parallelize independent sub-tasks it identifies mid-execution, where those
sub-tasks need to write variables back into the step's namespace — not just report text.

---

## Problem

The executor already parallelizes at the **plan-step** level: independent `Step`s in a batch each
get their own worker agent and their own namespace (`docs/parallel_execution.md`). There is no
equivalent one level down — if a single step's worker realizes its own task decomposes into
independent sub-parts (e.g. "compute three unrelated summary statistics over `df1`"), it currently
has no way to fan those out; it does them one tool call at a time in its own agentic loop.

### Why not `deepagents.middleware.SubAgentMiddleware` (the generic `task` tool)

This was the first option considered, since `TodoListMiddleware` was already adopted this way (see
`docs/improvements.md` #14). It doesn't fit here, for a reason specific to this codebase rather than
to `SubAgentMiddleware` itself:

1. `make_python_repl_tool(ns_path)` (`tools/python_repl/python_repl_tool.py`) closes over one fixed
   namespace path at tool-construction time.
2. `SubAgentMiddleware._get_subagents()` builds each subagent's tools — and thus its REPL tool —
   **once, at agent-construction time**, not per `task` call. Every `task` call routed to the same
   `subagent_type` reuses the identical REPL tool bound to the identical `ns_path`.
3. `execute_code` does a read-modify-write of the pickle at `ns_path` on every call when persistence
   is on (`tools/python_repl/executor.py`, `_execute_code_in_new_process`: "reading/writing namespace
   via files").

If the worker's LLM fires multiple `task` calls in parallel in one turn — the entire point of
sub-task parallelism — two concurrent invocations of the same subagent would both read-modify-write
the same pickle file: a lost-update race. This is the same class of bug `docs/improvements.md` #1
already solved at the step level (seed/execute/merge/cleanup per-batch namespaces); the generic
`task` tool has no equivalent, because its subagents are static, pre-compiled runnables, not
built fresh per call.

**If sub-tasks only needed to report text back** (no persisted variables), the stock `task` tool
would work fine in parallel — give the subagent's REPL tool `ns_path=None` and there's nothing to
race on. That's not the case here: sub-tasks need to leave DataFrames/variables behind for the
parent step to use afterward, so a purpose-built tool with per-call namespace isolation is required.

---

## Design overview

```
                Figure 1.  Step-level vs sub-task-level parallelism
   ══════════════════════════════════════════════════════════════════════

   STEP-LEVEL (existing, parallel_execution.md)   SUB-TASK-LEVEL (this doc)

   Plan = [Step1, Step2, Step3]                   Step 2's worker, mid-execution, decides its
        (Step2, Step3 independent)                own task splits into 3 independent sub-parts

        ┌── Step2 ──┐                                   ┌── Sub 0 ──┐
   Step1 │           ├──► merge ──► ...        Step2 ──► ├── Sub 1 ──┼──► merge ──► Step2 continues
        └── Step3 ──┘                                   └── Sub 2 ──┘

   dispatched by executor_node,                   dispatched by the step's own worker agent,
   asyncio.gather over worker agents               via a `spawn_subtasks` tool call,
   each with sub_agent_id = "{run_id}_step_{i}"    asyncio.gather over sub-worker agents,
                                                    each with sub_agent_id = "{step_ns_id}_sub_{i}"
```

Same primitives (`get_ns_path`, `load_ns`, `save_ns`, `cleanup_ns` in `namespace.py`), same
seed/execute/merge/cleanup lifecycle, one level deeper. No new namespace machinery is needed.

---

## The `spawn_subtasks` tool

New file: `src/sparq/tools/subtask_tools.py`, following the same factory pattern as
`make_load_dataset_tool` / `make_python_repl_tool` — the tool closes over the *parent step's*
namespace ID so it knows what to seed from and merge back into.

```python
# src/sparq/tools/subtask_tools.py
import asyncio
from pathlib import Path

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from sparq.schemas.output_schemas import StepResult
from sparq.tools.python_repl.namespace import get_ns_path, load_ns, save_ns, cleanup_ns
from sparq.tools.python_repl.python_repl_tool import make_python_repl_tool
from sparq.tools.data_discovery_tools import make_load_dataset_tool

SUBTASK_SYSTEM_PROMPT = """You are handling one independent sub-part of a larger analytical step.
You have access to the variables already computed by the parent step. Use python_repl_tool with
persist_namespace=True so any new variables you create are kept and merged back for the parent
step to use afterward. Report your findings in execution_results."""

MAX_SUBTASKS = 8  # cap fan-out per call — keeps LLM call count and merge risk bounded


def make_spawn_subtasks_tool(step_ns_id: str, llm: BaseChatModel):
    """Factory for a `spawn_subtasks` tool scoped to one parent step's namespace.

    Must be constructed fresh per worker (like `make_python_repl_tool`), closing over
    that worker's own `step_ns_id`, so sub-namespaces are seeded from and merged back
    into the correct parent step.
    """

    @tool
    async def spawn_subtasks(subtasks: list[str]) -> str:
        """Run independent sub-tasks in parallel. Each sub-task gets its own copy of your
        current namespace (variables you've already created) and can create new variables
        of its own. All sub-tasks' variables are merged back into your namespace afterward
        — do not use this for sub-tasks that depend on each other's output.

        Args:
            subtasks: Natural-language descriptions of independent sub-tasks to run in parallel.
        """
        if len(subtasks) > MAX_SUBTASKS:
            return f"Error: {len(subtasks)} subtasks requested, max is {MAX_SUBTASKS}. Batch or reduce."

        base_ns_path = get_ns_path(step_ns_id)
        base_ns = load_ns(base_ns_path)

        sub_ids = [f"{step_ns_id}_sub_{i}" for i in range(len(subtasks))]
        for sub_id in sub_ids:
            save_ns(get_ns_path(sub_id), dict(base_ns))  # seed

        async def run_one(description: str, sub_id: str) -> StepResult:
            sub_ns_path = get_ns_path(sub_id)
            sub_tools = [
                make_load_dataset_tool(sub_ns_path),
                make_python_repl_tool(sub_ns_path),
                # deliberately no spawn_subtasks tool here — caps recursion at one level
            ]
            sub_agent = create_agent(
                model=llm,
                tools=sub_tools,
                system_prompt=SUBTASK_SYSTEM_PROMPT,
                response_format=StepResult,
            )
            try:
                response = await sub_agent.ainvoke(
                    {"messages": [{"role": "user", "content": description}]}
                )
                return response["structured_response"]
            except Exception as e:
                return StepResult(id=0, step=description, execution_results="", misc=f"Subtask failed: {e}")

        results = await asyncio.gather(*(run_one(d, s) for d, s in zip(subtasks, sub_ids)))

        # merge: last-write-wins on key conflicts (same policy as step-level batches)
        merged = dict(base_ns)
        for sub_id in sub_ids:
            merged.update(load_ns(get_ns_path(sub_id)))
        save_ns(base_ns_path, merged)

        for sub_id in sub_ids:
            cleanup_ns(sub_id)

        lines = [f"Sub-task {i} ({subtasks[i]}):\n{r.execution_results}\n{r.misc}".strip()
                 for i, r in enumerate(results)]
        return "\n\n".join(lines)

    return spawn_subtasks
```

Notes on the choices above:

- **Result schema**: reuses `StepResult` rather than a new schema — same shape (`execution_results`,
  `files_generated`, `misc`) already fits a sub-task's output; no schema duplication.
- **Per-subtask error handling**: mirrors `execute_single_step_worker`'s own `try`/`except` around
  `agent.ainvoke` — one failing sub-task returns a `StepResult` with the error in `misc` rather than
  failing the whole `spawn_subtasks` call.
- **`MAX_SUBTASKS` cap**: bounds worst-case LLM call fan-out from a single tool call and keeps the
  merge's blast radius small. Adjust once real usage patterns are known.
- **Tool schema stays minimal**: `subtasks: list[str]` is one required field, no `Optional`/union
  types — consistent with the low-grammar-footprint reasoning from `docs/improvements.md` #14
  (Bedrock grammar-size blowup). This is a smaller schema than deepagents' own `task` tool.

---

## Namespace lifecycle

```
                Figure 2.  spawn_subtasks namespace lifecycle
   ══════════════════════════════════════════════════════════════════════

   step_ns_path  (the calling worker's own namespace)

        ┌──── seed ──────────────────────────────────────────┐
        │  save_ns(sub_0_ns_path, copy of step_ns)            │
        │  save_ns(sub_1_ns_path, copy of step_ns)            │
        │  save_ns(sub_2_ns_path, copy of step_ns)            │
        └──────────────────────────────────────────────────────┘
                 │              │              │
           Sub-agent 0    Sub-agent 1    Sub-agent 2
           (ainvoke)      (ainvoke)      (ainvoke)
                 │              │              │
        ┌──── asyncio.gather ──┴──────────────┘
        │
        ├── merge: step_ns.update(load_ns(sub_0_ns_path))
        │          step_ns.update(load_ns(sub_1_ns_path))
        │          step_ns.update(load_ns(sub_2_ns_path))
        │          save_ns(step_ns_path, step_ns)
        │
        └── cleanup: cleanup_ns(sub_0_id), cleanup_ns(sub_1_id), cleanup_ns(sub_2_id)
```

**Safety net**: sub-namespace IDs (`f"{run_id}_step_{step.id}_sub_{i}"`) already start with the
prefix `f"{run_id}_step_"` that `cleanup_run(run_id)` sweeps at the end of a run — so even if the
tool's own `cleanup_ns` calls are skipped (e.g. an exception before reaching them), orphaned
sub-namespace files are still removed when the run ends. Worth confirming with a test rather than
assuming, since `cleanup_run` was written before sub-namespaces existed.

---

## Recursion cap

Sub-agents spawned by `spawn_subtasks` get `load_dataset` and `python_repl_tool`, but **not**
`spawn_subtasks` itself — recursion is capped at exactly one level. This avoids unbounded
fan-out/cost and keeps the merge model (flat, one parent + N children) simple. If nested fan-out is
ever needed, revisit this — it is a deliberate simplicity trade-off, not a technical limitation.

---

## Integration points

| File | Change | Why |
|------|--------|-----|
| `src/sparq/tools/subtask_tools.py` (new) | `make_spawn_subtasks_tool(step_ns_id, llm)` factory | Isolated, testable; mirrors existing tool-factory pattern |
| `src/sparq/architectures/v1/nodes/executor.py` | In `execute_single_step_worker`, add `make_spawn_subtasks_tool(step_ns_id, llm_object)` to `_tools` | Gives the step's worker access to the new tool |
| `src/sparq/architectures/v1/prompts/executor_message.txt` | Add a `spawn_subtasks` bullet to the "Tools at your disposal" list, with guidance on when to use it (independent sub-parts only, not sequential/dependent work) | Worker needs to know the tool exists and its constraints |
| `src/sparq/tools/python_repl/namespace.py` | No changes | `get_ns_path`, `load_ns`, `save_ns`, `cleanup_ns` reused as-is |

### Suggested prompt addition

```
- spawn_subtasks(subtasks: list[str]): Runs independent sub-parts of your current task in parallel,
  each with its own copy of your current variables. New variables each sub-task creates are merged
  back into your namespace afterward. Only use this when sub-tasks are truly independent of each
  other (no sub-task needs another's output) — for sequential/dependent work, just do it directly.
```

---

## Open questions / follow-ups

- **Cost**: each `spawn_subtasks` call is N additional LLM agent loops. No budget/backoff exists yet
  if a worker calls this repeatedly or with large `subtasks` lists close to `MAX_SUBTASKS`.
- **Context bloat after merge**: `execute_single_step_worker` already warns
  (`MAX_NAMESPACE_VARS_WARNING`) when a step inherits too many variables from dependencies; merged
  sub-task variables should be checked against the same threshold, since a `spawn_subtasks` call
  could itself push a step's namespace over it.
- **`cleanup_run` prefix assumption**: confirm the safety net described above with a test that kills
  a `spawn_subtasks` call mid-flight and checks `_ns_paths` afterward, rather than relying on the
  string-prefix argument alone.
- Cross-reference: `docs/why_not_deepagents.md`'s conclusion that intra-plan fan-out is a blocking
  join, not a background/pollable task, applies identically one level down — no need to reconsider
  `deepagents`' async subagents for this.
