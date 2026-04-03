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
| Figure numbering tool for executor | Low | Medium (output quality) |
| Aggregator academic referencing of outputs | Medium | High (output quality) |
| Aggregator writes markdown report → PDF | Low | High (output quality) |
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

**Additional bug — double invocation:** Both nodes also stream the agent *and* invoke it separately:

```python
for chunks in agent.stream(agent_input, stream_mode="updates"):
    print(chunks)                        # LLM call 1 — result discarded
response = agent.invoke(agent_input, ...)  # LLM call 2 — this is what's actually used
```

The stream is used only for printing and its result is thrown away. Every router and planner run makes two full LLM calls. Fix: invoke once and print the result, or stream and accumulate.

---

## 2. Direct answer path skips the saver

**Files**: `src/sparq/system.py`, `src/sparq/nodes/saver.py`

If `route=False`, the graph edges to `END` without persisting anything:

```python
graph_init.add_conditional_edges("router", router_func, {True: "planner", False: END})
```

Direct answers are never saved. The saver should sit on both exit paths, or the router node should handle its own persistence. As-is, any simple query produces no output files.

---

## 3. Executor returns full state instead of a partial update

**File**: `src/sparq/nodes/executor.py`

At the end of the step loop:

```python
    state['executor_results'] = results  # assigned inside loop, on every iteration
return state                             # returns the entire state dict
```

Two issues, neither a correctness bug but both wrong by convention:

1. `state['executor_results'] = results` runs on every loop iteration. Only the final assignment matters — all prior ones are overwritten before anything reads them. Move it outside the loop.

2. `return state` returns the full state dict. LangGraph nodes should return only the fields they changed. The correct form is `return {'executor_results': results}`. Returning the whole state silently overwrites every other field with its current value, which happens to be a no-op now but becomes a source of subtle bugs if reducers or concurrent updates are ever introduced (see improvement 7).

Note: this is **not** a data-loss bug. The aggregator correctly receives all step results because the final iteration's write survives in `state` and is included in the return.

---

## 4. Global REPL namespace breaks concurrent runs

**File**: `src/sparq/tools/python_repl/namespace.py`

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
- Add LangGraph `Annotated` reducers for fields that should merge rather than replace. In the current architecture `executor_results` is written once by the executor node, so no reducer is needed today. However, if replanning is introduced (see improvement 8) and the executor runs multiple times, a merge reducer becomes necessary to avoid the second run overwriting the first:

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

## 10. Figure numbering tool for executor

**Files**: `src/sparq/tools/figure_tools.py` (new), `src/sparq/nodes/executor.py`, `src/sparq/prompts/executor_message.txt`

The executor generates plots with no awareness of how many already exist in `output_dir`. This means figure titles like "Correlation Plot" with no numbering, making it impossible for the aggregator (or a reader) to reference them unambiguously.

Add a `get_next_figure_number(directory: str) -> str` tool that counts existing image files in `output_dir` and returns the next label (e.g. `"Figure 3"`). The executor calls this before saving each plot and uses the returned label in the figure title.

```python
@tool
def get_next_figure_number(directory: str) -> str:
    """Return the next sequential figure label based on existing images in the directory."""
    path = Path(directory)
    count = sum(1 for f in path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg'})
    return f"Figure {count + 1}"
```

The executor prompt should be updated to instruct: *"Before saving any plot, call `get_next_figure_number` to obtain the figure label and include it in the plot title."*

---

## 11. Aggregator academic referencing of outputs

**Files**: `src/sparq/nodes/aggregator.py`, `src/sparq/prompts/aggregator_message.txt`, `src/sparq/system.py`

The aggregator currently only sees `executor_results` as a text dict. It has no awareness of files generated by the executor, so it cannot reference plots and documents in its report.

**What to add:**

1. Pass `executor_output_dir` to `aggregator_node`
2. Before invoking the LLM, scan the directory for image files sorted by modification time and build a figure manifest:
   ```
   Figure 1: correlation_matrix.png
   Figure 2: regression_plot.png
   ```
3. Inject this as a `{figures}` template variable into the aggregator prompt
4. Update the aggregator prompt to instruct: *"Write in academic style. Reference figures by their assigned number (e.g. 'as shown in Figure 1'). Do not describe figures without referencing them."*

This mirrors how scientific papers reference figures and ensures the output is self-consistent with what was actually produced.

---

## 12. Aggregator writes markdown report converted to PDF

**Files**: `src/sparq/nodes/aggregator.py`, `src/sparq/nodes/saver.py`, `src/sparq/system.py`

Currently `answer` is a plain string stored in state and dumped into `final_answer.json`. For a product, the output should be a formatted document.

**Changes:**

1. Pass `run_dir` to `aggregator_node`; after the LLM call, write `answer` to `run_dir / "report.md"`
2. In `saver_node`, after writing `trace.json` and `final_answer.json`, attempt PDF conversion:
   ```python
   import subprocess
   md_path = save_dir / "report.md"
   if md_path.exists():
       subprocess.run(["pandoc", str(md_path), "-o", str(save_dir / "report.pdf")], check=False)
   ```
3. PDF conversion is best-effort — if `pandoc` is not installed the markdown is still saved and usable

**Dependency**: `pandoc` must be installed on the system (`apt install pandoc` / `brew install pandoc`). For Docker deployments, add it to the `Dockerfile`.

---

## 13. Multi-route router for general use cases

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

**Design decision on `direct_answer`:** In the proposed graph, `direct_answer` routes through the aggregator. But the router already generates the answer inline — passing it through an extra aggregator LLM call adds latency and cost for no benefit. The `direct_answer` path should either skip the aggregator entirely (going straight to saver) or the aggregator should detect this route and pass the answer through unchanged.

**Design decision on separate nodes vs. extra tools:** An alternative is to not add new nodes but instead give the executor additional tools (`interpret_plot`, web search). The planner would then include image analysis or web search steps in its plan. This is more flexible but harder to prompt reliably and couples general capabilities to the data-analysis planning machinery. Separate branches are cleaner for a product.

---

## 14. Aggregator output isn't structured

**File**: `src/sparq/nodes/aggregator.py`

The router, planner, and executor all use `response_format` for structured output. The aggregator calls `llm.invoke(prompt_str)` and reads `.content` directly. This inconsistency makes it harder to evolve the output schema (e.g., adding citations, confidence scores, or section headers) and bypasses the validation guarantees the rest of the pipeline relies on.
