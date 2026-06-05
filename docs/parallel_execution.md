# Parallel Execution Design

Design document for improvement #1: step dependency tracking and concurrent execution of independent plan steps.

---

## Problem

The executor currently runs all plan steps in a sequential loop:

```python
for i, step in enumerate(plan.steps):
    results = process_step(results, step.step_description, i+1)
```

Two issues make concurrency impossible today:

1. **No dependency information.** The `Step` schema has no field expressing which steps depend on which. The executor cannot know that steps 2 and 3 are independent of each other and could run at the same time.

2. **Single shared namespace file.** All steps share one pickle file (`ns_path = get_ns_path(run_id)`). If two subprocesses wrote to it concurrently the last write would win, silently discarding the other step's variables.

3. **Synchronous agent invocation.** `agent.invoke(...)` blocks the thread, so even if the loop were refactored, parallel dispatch would require `agent.ainvoke` and `asyncio.gather`.

---

## Design overview

```
                   Figure 1.  Sequential vs parallel execution
    ═══════════════════════════════════════════════════════════════════

    CURRENT (sequential)            PROPOSED (batched parallel)

    Step 1 ──────────────────►      Batch 1 ──┬── Step 1 ──┐
    Step 2 ──────────────────►                └── Step 2 ──┼──► merge ──► Batch 2 ──► ...
    Step 3 ──────────────────►                    Step 3 ──┘
    Step 4 ──────────────────►
                                    Steps with no unmet deps
    wall time: sum(all steps)       run concurrently per batch.
                                    wall time ≈ longest step per batch
```

The key insight: steps that share no `depends_on` relationship can each get their own private namespace, run fully in parallel, and then merge their results back into the shared namespace before the next batch starts.

---

## Step dependency model

### Schema change

Add one field to `Step` in `src/sparq/schemas/output_schemas.py`:

```python
class Step(BaseModel):
    step_description: str
    datasets: List[str]
    rationale: str
    task_type: List[str]
    depends_on: list[int] = []   # zero-based indices of steps that must finish first
```

An empty list means the step has no prerequisites and can run as soon as its batch is dispatched. The planner is responsible for populating this field.

### Topological batch algorithm

```python
def topological_batches(steps: list[Step]) -> list[list[tuple[int, Step]]]:
    """
    Yield batches of (index, step) pairs that can run concurrently.
    A step enters a batch when all its depends_on indices are in completed.
    """
    completed: set[int] = set()
    remaining = list(enumerate(steps))

    while remaining:
        batch = [(i, s) for i, s in remaining if all(d in completed for d in s.depends_on)]
        if not batch:
            raise ValueError("Cycle detected in step dependencies")
        yield batch
        completed.update(i for i, _ in batch)
        remaining = [(i, s) for i, s in remaining if i not in completed]
```

A batch of size 1 runs the single step with the existing synchronous path. A batch of size > 1 triggers the parallel path.

---

## `sub_agent_id` pattern

Each step in a parallel batch gets its own namespace keyed by:

```python
sub_agent_id = f"{run_id}_step_{step_index}"
sub_ns_path  = get_ns_path(sub_agent_id)
```

`get_ns_path` in `namespace.py` already accepts any string ID and creates a fresh keyed pickle file. No changes to its signature are needed. `cleanup_ns(sub_agent_id)` removes the file after the batch merges.

This gives each parallel step a private file that its REPL subprocesses read from and write to exclusively — no shared mutable state during execution.

---

## Namespace lifecycle for a parallel batch

```
                   Figure 2.  Per-step namespace lifecycle
    ═══════════════════════════════════════════════════════════════════

    base ns_path  (/tmp/<run_id>_ns.pkl)   ← contains vars from prior batches

         ┌──── seed ─────────────────────────────────────────┐
         │  shutil.copy2(base, sub_ns_path_A)                │
         │  shutil.copy2(base, sub_ns_path_B)                │
         └───────────────────────────────────────────────────┘
                    │                    │
              Step A (ainvoke)     Step B (ainvoke)
              writes sub_ns_path_A  writes sub_ns_path_B
                    │                    │
         ┌──── asyncio.gather ───────────┘
         │
         ├── merge: base_ns.update(load_ns(sub_ns_path_A))
         │          base_ns.update(load_ns(sub_ns_path_B))
         │          pickle.dump(base_ns, base_ns_path)
         │
         └── cleanup: cleanup_ns(sub_agent_id_A)
                      cleanup_ns(sub_agent_id_B)
```

**Step 1 — Seed.** Before dispatching, copy the current base pickle into each sub-path. This ensures parallel steps start with all variables that preceding sequential steps produced (e.g. a loaded dataset from batch 1 is visible to both steps in batch 2).

**Step 2 — Execute.** Each step's agent and tools are bound to its own `sub_ns_path`. REPL subprocesses read from and write to only that file. No file is touched by more than one step concurrently.

**Step 3 — Merge.** After `asyncio.gather` returns, load each sub-namespace and merge into the base with `dict.update()`. Last-write-wins on key conflicts. This is acceptable because truly independent steps should not produce variables with the same name — if they do, it is a planner error that the merge makes visible rather than hiding.

**Step 4 — Cleanup.** `cleanup_ns(sub_agent_id)` removes each sub-pickle and pops it from the global `_ns_paths` dict.

---

## Per-step agent construction

Each parallel step needs its own agent with tools closed over its private `sub_ns_path`:

```python
async def process_step_parallel(step, step_index, run_id, base_ns_path, ...):
    sub_agent_id = f"{run_id}_step_{step_index}"

    # Seed namespace from current base state
    sub_ns_path = get_ns_path(sub_agent_id)
    shutil.copy2(base_ns_path, sub_ns_path)

    # Tools bound to this step's private namespace
    tools = [
        make_load_dataset_tool(sub_ns_path),
        make_python_repl_tool(sub_ns_path),
        ...
    ]
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt,
                               response_format=(prompt, ExecutorOutput))

    context = _build_context(prior_results, sub_ns_path)
    user_content = f"{context}\n\nYour current task:\n{step.step_description}"
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config={"recursion_limit": llm_config.recursion_limit}
    )
    return response["structured_response"], sub_agent_id
```

Each agent is independent — different tool closures, different namespace files, different LLM call chains. They share no mutable state.

---

## Merge key-conflict policy

After a parallel batch, sub-namespaces are merged into the base with `dict.update()` in step-index order (lower index wins on conflict). The assumption is that genuinely independent steps do not produce identically named output variables. If they do, the conflict is a signal that the planner's `depends_on` assignment was wrong, not a reason to add complex conflict resolution.

If variable name collisions become a real problem in practice, a simple mitigation is for the executor prompt to instruct the agent to prefix output variable names with the step index (e.g. `step2_result` instead of `result`).

---

## What changes per file

| File | Change | Why |
|------|--------|-----|
| `src/sparq/schemas/output_schemas.py` | Add `depends_on: list[int] = []` to `Step` | Schema must carry dependency info for the executor to batch correctly |
| `src/sparq/architectures/v1/nodes/executor.py` | `executor_node` → `async def`; `process_step` → `async def`; sequential loop → topological batch loop; `agent.invoke` → `agent.ainvoke`; add sub-agent construction, namespace seed/merge/cleanup | Core execution logic |
| `src/sparq/tools/python_repl/namespace.py` | No changes to signatures or logic | `get_ns_path(sub_agent_id)`, `cleanup_ns(sub_agent_id)`, `load_ns(sub_ns_path)` are reused as-is — they already accept arbitrary string IDs |
| `src/sparq/architectures/v1/system.py` | No changes | `run_id` creation and graph wiring are unchanged; LangGraph handles sync and async node functions transparently |

---

## What stays the same

- `run_id` creation in `system.py` — one UUID per `Agentic_system.run()` call, unchanged
- REPL subprocess machinery (`executor.py`, `ast_utils.py`, `namespace.py`) — untouched
- `get_ns_path`, `cleanup_ns`, `load_ns` call signatures — reused as-is
- All non-executor nodes (router, planner, aggregator, saver) — unaffected
- Graph topology in `system.py` — no new edges or nodes
- The `run_id` namespace (base pickle) — still cleaned up by `cleanup_ns(run_id)` in `system.py`'s `finally` block
