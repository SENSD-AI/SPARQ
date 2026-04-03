# Architectural Improvement Recommendations

## Priority Summary

| Change | Effort | Impact |
|--------|--------|--------|
| Replace router/planner ReAct with structured LLM | Low | Medium (speed, cost) |
| Fix saver bypass on direct-answer path | Low | High (correctness) |
| Fix executor mid-loop state mutation | Low | Medium (correctness) |
| Scope REPL namespace to run ID | Low | High (concurrency safety) |
| Plot interpretation via agent-controlled tool | Medium | High (output quality) |
| Rolling context summarization in executor | Medium | High (context limit avoidance) |
| Step dependency + parallel execution | Medium | High (throughput) |
| State as Pydantic with reducers | Medium | Medium (robustness) |
| Multi-route router for general use cases | High | High (product scope) |
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

## 9. Executor cannot interpret generated plots

**File**: `src/sparq/nodes/executor.py`, `src/sparq/tools/`

The executor produces plots and documents but has no way to read them back. `files_generated` is just a list of filename strings — the agent cannot inspect the actual content of images, so plots are opaque to subsequent steps.

The executor already has `list_directory` from `FileManagementToolkit` (via `filesystemtools(selected_tools='all')`), so the agent can already discover what files exist in `output_dir`. What's missing is a single additional tool:

**An `interpret_plot(file_path)` tool** — loads a saved image, encodes it, and passes it to a multimodal LLM, returning a text description. The agent calls this selectively on whichever files it judges relevant to the current step.

```python
@tool
def interpret_plot(file_path: str) -> str:
    """Interpret a plot image and return a text description of its findings."""
    path = Path(file_path)
    b64 = base64.b64encode(path.read_bytes()).decode()
    content = [
        {"type": "text", "text": "Describe the key findings in this plot."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]
    response = vision_llm.invoke([HumanMessage(content=content)])
    return response.content
```

**Why tool-based rather than automatic post-processing:** Automatically interpreting every new plot after each step wastes tokens on diagnostic outputs the agent doesn't need. Giving the agent the tool lets it decide which files are worth interpreting for the current task.

**Constraint:** The LLM backing `interpret_plot` must be multimodal. This should be a separate `vision_llm` config entry in `LLMSettings` rather than reusing the executor LLM, so switching the executor to a text-only model doesn't silently break plot interpretation.

---

## 10. Multi-route router for general use cases

**Files**: `src/sparq/nodes/router.py`, `src/sparq/schemas/output_schemas.py`, `src/sparq/system.py`

The current router is binary (`route: bool`) — it only distinguishes "needs data analysis" from "answer directly." This locks the system into a single use case. As a product, the system needs to handle general queries such as image analysis, literature/research surveys, and open-ended conversation.

**Proposed graph:**

```
router (enum)
  → "data_analysis"   → planner → executor → aggregator → saver
  → "image_analysis"  → vision_node          → aggregator → saver
  → "research_survey" → web_search_node      → aggregator → saver
  → "direct_answer"   →                        aggregator → saver
```

**What changes:**

1. **`Router` schema** — replace `route: bool` with `route: Literal["data_analysis", "image_analysis", "research_survey", "direct_answer"]`

2. **Router prompt** — describe each route with clear criteria and examples so the LLM classifies reliably

3. **New nodes:**
   - `vision_node`: multimodal LLM call on user-uploaded images; result stored in state for aggregator
   - `web_search_node`: ReAct agent with a web search tool (e.g. Tavily) for literature surveys and general research questions

4. **Aggregator becomes route-aware** — inputs differ per path (executor results dict vs. vision response vs. search results). Either normalize upstream so aggregator always sees the same shape, or make the aggregator prompt conditional on `state["route"]`

5. **State** — add `vision_input: list | None` for image content blocks; `route` field typed to the new enum

**The planner and executor are unchanged** — they become one branch of the conditional edges rather than the only path.

**Design decision:** An alternative is to not add new nodes but instead give the executor additional tools (`interpret_plot`, web search). The planner would then include image analysis or web search steps in its plan. This is more flexible but harder to prompt reliably and couples general capabilities to the data-analysis planning machinery. Separate branches are cleaner for a product.

---

## 11. Aggregator output isn't structured

**File**: `src/sparq/nodes/aggregator.py`

The router, planner, and executor all use `response_format` for structured output. The aggregator calls `llm.invoke(prompt_str)` and reads `.content` directly. This inconsistency makes it harder to evolve the output schema (e.g., adding citations, confidence scores, or section headers) and bypasses the validation guarantees the rest of the pipeline relies on.
