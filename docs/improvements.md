# Architectural Improvement Recommendations

## Priority Summary

| Change | Effort | Impact |
|--------|--------|--------|
| Replace router/planner ReAct with structured LLM | Low | Medium (speed, cost) |
| Fix saver bypass on direct-answer path | Low | High (correctness) |
| Fix executor mid-loop state mutation | Low | Medium (correctness) |
| Scope REPL namespace to run ID | Low | High (concurrency safety) |
| Rolling context summarization in executor | Medium | High (context limit avoidance) |
| Step dependency + parallel execution | Medium | High (throughput) |
| State as Pydantic with reducers | Medium | Medium (robustness) |
| Executor → planner feedback loop | High | High (resilience) |

---

## 1. Router and Planner don't need ReAct agents

**Files**: `src/sparq/nodes/router.py`, `src/sparq/nodes/planner.py`

Both nodes use `create_react_agent` with no tools. ReAct adds extra LLM turns for "thinking" that serve no purpose when there are no tools to invoke. Replace with a plain structured LLM call:

```python
# Instead of create_react_agent with no tools:
structured_llm = llm.with_structured_output(Router)
response = structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
```

This is faster, cheaper, and simpler. The ReAct pattern is only justified in the executor, where there are tools to use.

---

## 2. Direct answer path skips the saver

**Files**: `src/sparq/system.py`, `src/sparq/nodes/saver.py`

If `route=False`, the graph edges to `END` without persisting anything:

```python
graph_init.add_conditional_edges("router", router_func, {True: "planner", False: END})
```

Direct answers are never saved. The saver should sit on both exit paths, or the router node should handle its own persistence. As-is, any simple query produces no output files.

---

## 3. Executor mutates state mid-loop

**File**: `src/sparq/nodes/executor.py`

Inside the step loop:

```python
state['executor_results'] = results  # side-effecting mid-loop
```

LangGraph nodes should return a state update dict at the end, not mutate in-place during execution. The intermediate writes during the loop are never observed by any other node — they're dead writes. Move the assignment outside the loop and return it as part of the node's return value.

---

## 4. Global REPL namespace breaks concurrent runs

**File**: `src/sparq/tools/python_repl/executor.py` (or wherever `_PERSISTENT_NS_PATH` is defined)

The persistent namespace path is a module-level global:

```python
_PERSISTENT_NS_PATH = None  # single global per process
```

Two concurrent runs share the same pickle file, causing namespace collisions. The namespace path should be scoped to a run ID — pass it via LangGraph's `RunnableConfig` (e.g., `config["configurable"]["run_id"]`) so each run gets an isolated temp file.

---

## 5. Executor context grows unboundedly

**File**: `src/sparq/nodes/executor.py`

Each step prepends the full accumulated results from all prior steps as context. For a 10-step plan with verbose outputs, step 10 carries everything from steps 1–9. This hits context limits and degrades relevance.

Better approach: maintain a **rolling summary** of prior steps. After each step completes, summarize its output to 2–3 sentences and carry only the summary forward. Keep the full output in `executor_results` for the aggregator.

---

## 6. Plan steps have no dependency information

**File**: `src/sparq/schemas/output_schemas.py`

The `Step` schema has no way to express inter-step dependencies. As a result, all steps always run sequentially even when they are independent. Adding an optional field:

```python
class Step(BaseModel):
    ...
    depends_on: List[int] = []  # indices of steps this step requires
```

would allow the executor to topologically sort steps and run independent ones in parallel via `asyncio.gather`. This could cut wall-clock time significantly for multi-step plans.

---

## 7. State is an unvalidated TypedDict

**File**: `src/sparq/schemas/state.py`

`State` is a plain `TypedDict`, so bad updates silently succeed at runtime. Two improvements:

- Use a Pydantic model for validation on updates.
- Add LangGraph `Annotated` reducers for fields that should merge rather than replace. For example, `executor_results` accumulates across steps and should use `operator.or_` rather than full replacement:

```python
from typing import Annotated
import operator

class State(TypedDict):
    executor_results: Annotated[dict, operator.or_]
```

---

## 8. No replanning on executor failure

**Files**: `src/sparq/nodes/executor.py`, `src/sparq/system.py`

When a step fails, the error is stored and execution continues silently. The aggregator receives a mix of successful results and error strings with no structural distinction. A feedback loop — routing back to the planner when a threshold of steps fail — would make the system more resilient. LangGraph's conditional edges make this straightforward:

```python
graph_init.add_conditional_edges("executor", executor_health_check, {
    "ok": "aggregator",
    "replan": "planner",
    "abort": END,
})
```

---

## 9. Aggregator output isn't structured

**File**: `src/sparq/nodes/aggregator.py`

The router, planner, and executor all use `response_format` for structured output. The aggregator calls `llm.invoke(prompt_str)` and reads `.content` directly. This inconsistency makes it harder to evolve the output schema (e.g., adding citations, confidence scores, or section headers) and bypasses the validation guarantees the rest of the pipeline relies on.
