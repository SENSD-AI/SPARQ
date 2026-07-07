# Architectural Improvement Recommendations

## Priority Summary

| Change | Effort | Impact | Done |
|--------|--------|--------|------|
| Step dependency + parallel execution | Medium | High (throughput) | [ ] |
| Executor → planner feedback loop | High | High (resilience) | [ ] |
| Plot interpretation via agent-controlled tool | Medium | High (output quality) | [ ] |
| Figure numbering tool for executor | Low | Medium (output quality) | [ ] |
| Aggregator academic referencing of outputs | Medium | High (output quality) | [ ] |
| Aggregator writes markdown report → PDF | Low | High (output quality) | [ ] |
| Multi-route router for general use cases | High | High (product scope) | [ ] |
| Aggregator output isn't structured | Low | Medium (robustness) | [ ] |
| LLM model validation at startup | Low | Medium (reliability) | [ ] |
| Human-readable REPL tracebacks | Low | Medium (debuggability) | [x] |
| Data science coding skill for executor | Medium | High (agent code quality) | [ ] |
| **Artifact organizer node** (supersedes #3, #4, #5) | Medium | High (output quality) | [ ] |
| Aggregator token-budget truncation + RAG fallback | Medium | High (robustness) | [ ] |
| Bedrock grammar-size blowup with many bound tools | Medium | High (provider compat) | [ ] |
| User-facing step-completion tracker (resumable) | High | High (product scope) | [ ] |
| Sub-task parallelism within a worker (`spawn_subtasks`) | Medium | Medium (throughput) | [ ] |

---

## 1. Step dependency + parallel execution

**Files**: `src/sparq/schemas/output_schemas.py`, `src/sparq/architectures/v1/nodes/executor.py`

**Design doc**: [`docs/parallel_execution.md`](parallel_execution.md)

### Problem

The `Step` schema has no dependency field, so all steps always run sequentially. A multi-step plan that loads three datasets and then analyses them cannot overlap the loads, even though they are independent. The shared single namespace pickle file (`ns_path = get_ns_path(run_id)`) would also race if two subprocesses wrote to it concurrently.

### Design

**Schema change** — add `depends_on` to `Step`:

```python
class Step(BaseModel):
    ...
    depends_on: list[int] = []  # zero-based indices of steps that must complete first
```

The planner populates this field. Steps with an empty `depends_on` (or whose dependencies are all resolved) form a batch that can run concurrently.

**`sub_agent_id` per parallel step** — rather than sharing the run-level `ns_path`, each step in a parallel batch gets its own isolated namespace keyed by `f"{run_id}_step_{i}"`. The existing `get_ns_path`, `load_ns`, and `cleanup_ns` primitives in `namespace.py` handle this with no changes to their signatures.

**Namespace lifecycle for a parallel batch:**
1. **Seed** — copy the current base pickle into each `sub_ns_path` so parallel steps start with variables from completed sequential steps
2. **Execute** — each step's agent and REPL tools are bound to its own `sub_ns_path`; subprocesses write only to their own file, no races
3. **Merge** — after `asyncio.gather`, load each sub-namespace and `update()` the base; last-write-wins on key conflicts (acceptable because truly independent steps should not create the same variable name)
4. **Cleanup** — `cleanup_ns(sub_agent_id)` removes each sub-pickle

**Executor changes** — `executor_node` and `process_step` become `async`; `agent.invoke` → `agent.ainvoke`. The sequential `for` loop is replaced by a topological batch loop: resolve which steps have no unmet dependencies, dispatch them concurrently, mark complete, repeat.

**`system.py` is unchanged** — `run_id` creation, graph wiring, and the event loop are unaffected. LangGraph is transparent to whether a node function is sync or async.

---

## 2. No replanning on executor failure

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

## 3. Executor cannot interpret generated plots

**Files**: `src/sparq/tools/interpret_image_tool.py` (new), `src/sparq/architectures/v1/nodes/executor.py`, `src/sparq/settings.py`, `config/config.toml`

The executor produces plots but has no way to read them back. `files_generated` is just a list of filename strings — plots are opaque to subsequent steps and to the aggregator.

**Design:** a `make_interpret_image_tool(vision_llm)` factory returns a `@tool` the executor can call immediately after saving any plot. The interpretation is returned as a tool result, visible in the executor's context for subsequent steps and captured in `ExecutorOutput` for the artifact organizer to fold into `artifact_map.json`.

```python
def make_interpret_image_tool(vision_llm):
    @tool
    def interpret_image(file_path: str) -> str:
        """Load a saved plot and return a description of its key findings."""
        path = Path(file_path)
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            path.suffix.lower().lstrip("."), "image/png"
        )
        b64 = base64.b64encode(path.read_bytes()).decode()
        response = vision_llm.invoke([HumanMessage(content=[
            {"type": "text", "text": "Describe the key findings in this figure. Be concise and scientific."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ])])
        return response.content
    return interpret_image
```

The executor prompt is updated to instruct: *"After saving any plot with `plt.savefig`, call `interpret_image` on the saved path and include the interpretation in your findings."*

**Constraint:** `vision_llm` must be a multimodal model. Add a separate `vision_llm` config entry in `LLMSettings` — do not reuse the executor LLM, since that may be text-only. `vision_llm` is also used by the artifact organizer node (improvement #12).

---

## 4. Figure numbering tool for executor

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

## 5. Aggregator academic referencing of outputs

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

## 6. Aggregator writes markdown report converted to PDF

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

## 7. Multi-route router for general use cases

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

4. **Aggregator becomes route-aware** — inputs differ per path (executor results dict vs. vision response vs. search results). Either normalize upstream so aggregator always sees the same shape, or make the aggregator prompt conditional on `state.route`

5. **State** — add `vision_input: list | None` for image content blocks; `route` field typed to the new enum

**The planner and executor are unchanged** — they become one branch of the conditional edges rather than the only path.

**Design decision on `direct_answer`:** In the proposed graph, `direct_answer` routes through the aggregator. But the router already generates the answer inline — passing it through an extra aggregator LLM call adds latency and cost for no benefit. The `direct_answer` path should either skip the aggregator entirely (going straight to saver) or the aggregator should detect this route and pass the answer through unchanged.

**Design decision on separate nodes vs. extra tools:** An alternative is to not add new nodes but instead give the executor additional tools (`interpret_plot`, web search). The planner would then include image analysis or web search steps in its plan. This is more flexible but harder to prompt reliably and couples general capabilities to the data-analysis planning machinery. Separate branches are cleaner for a product.

---

## 8. Aggregator output isn't structured

**File**: `src/sparq/nodes/aggregator.py`

The router, planner, and executor all use `response_format` for structured output. The aggregator calls `llm.invoke(prompt_str)` and reads `.content` directly. This inconsistency makes it harder to evolve the output schema (e.g., adding citations, confidence scores, or section headers) and bypasses the validation guarantees the rest of the pipeline relies on.

---

## 9. LLM model validation at startup

**Files**: `src/sparq/utils/get_llm.py`, `src/sparq/system.py`

After a user sets model IDs in a TOML config there is no verification that those models are actually usable. Errors only surface mid-run when a node first invokes the model, at which point significant context and cost may already have been spent. Three properties should be checked up front:

1. **On-demand throughput** — the model must be available for on-demand inference. A model that only supports provisioned throughput (e.g., a Bedrock model the user hasn't reserved capacity for) will fail at runtime.
2. **Input/output modality** — the pipeline sends text and expects text back. A misconfigured model ID that resolves to an image-only or embedding model will produce cryptic errors deep in the graph.
3. **Active vs. legacy lifecycle** — a legacy/deprecated model ID should surface a warning so users can migrate before it is removed.

**Design:**

Add a `validate_llm(model, provider, node)` function to `get_llm.py` with per-provider implementations:

- **`aws_bedrock`** — uses the `bedrock` management-plane client (distinct from `bedrock-runtime`). For direct model IDs: calls `get_foundation_model` and checks `inferenceTypesSupported` (must contain `ON_DEMAND`), `inputModalities`/`outputModalities` (must contain `TEXT`), and `modelLifecycle.status` (warn if `LEGACY`). For cross-region inference profile IDs (`us.*`, `eu.*`, `ap.*`): calls `get_inference_profile` to verify the profile exists and is `ACTIVE`, then resolves the underlying model ARN to check modalities and lifecycle.
- **`google_genai`** — calls `genai.get_model()` to verify the model exists and that `generateContent` is in `supported_generation_methods`.
- **`openai`** — calls `openai.models.retrieve()` to verify the model ID is accessible.
- **`openrouter`** — no programmatic check available; skipped.

Add a `_validate_models()` method to `Agentic_system` called at the end of `__init__`, iterating over all non-`None` entries in `self.settings.llm_config` and calling `validate_llm` for each. Validation failures raise `ValueError` immediately; legacy lifecycle emits `warnings.warn`.

---

## 10. REPL tracebacks don't show source lines

**Files**: `src/sparq/tools/python_repl/executor.py`, `src/sparq/tools/python_repl/python_repl_tool.py`

When the executor agent's code raises an exception, the traceback currently reads:

```
File ".../executor.py", line 227, in _target
  exec("\n".join(statements), namespace)
File "<string>", line 13, in <module>
TypeError: 'builtin_function_or_method' object is not iterable
```

The internal `executor.py` frame is noise. `File "<string>"` tells the agent nothing about what was on line 13. The agent has to re-read its own previous tool call to infer what went wrong.

The root cause is that `exec(code, namespace)` registers the source as `"<string>"`, so Python's traceback formatter has no source lines to display. The fix uses the same mechanism IPython/Jupyter uses: **inject the source into `linecache`** before execution so Python can retrieve lines by filename when formatting the traceback.

**Changes to `_target` in `executor.py`:**

1. Reconstruct `full_source` from `statements + expr` (reversing the `extract_last_expression` split).
2. Register it in `linecache.cache["<repl>"]` with the format Python expects: `(size, mtime, lines, fullname)`.
3. Replace `exec("\n".join(statements), namespace)` with `compile(..., "<repl>", "exec")` + `exec`.
4. For the `eval(expr, ...)` call: use `ast.parse` + `ast.increment_lineno` to offset the expression to its correct line number in the original source, then compile the shifted AST with `"<repl>"`.

**Changes to `python_repl_tool.py`:**

Include the submitted code block in the error response so the agent sees its full submission alongside the error, not just the failing lines.

**Result** — the traceback the agent sees becomes:

```
File "<repl>", line 13, in <module>
  for item in list.sort():
              ^^^^^^^^^^^
TypeError: 'builtin_function_or_method' object is not iterable
```

This lets the agent immediately identify the offending line without re-reading prior context, reducing unnecessary retry turns.

---

## 12. Artifact organizer node

**Files**: `src/sparq/architectures/v1/nodes/artifact_organizer.py` (new), `src/sparq/schemas/state.py`, `src/sparq/architectures/v1/system.py`

**Design doc**: [`docs/artifact_organizer.md`](artifact_organizer.md)

**Supersedes improvements #4 and #5.** Improvement #3 (plot interpretation) is resolved separately as an executor tool (`interpret_image`) — the artifact organizer receives interpretations pre-computed in `ExecutorOutput` and folds them into the map.

### Problem

After the executor finishes, the output directory is an unorganized mix of CSV files, images, and text reports with no metadata. The aggregator receives raw `ExecutorOutput` dicts and has to infer what was produced without being able to read or reference any of it. Images are not numbered, not organized, and not referenced in a consistent way.

### Design

A new `artifact_organizer_node` is inserted between the executor and aggregator. Internally it runs one subagent and performs file organization:

- **Inspector subagent** (Python REPL + file read tools): scans data files (CSV, parquet) and text reports; distills each executor step's output to a one-line summary
- **File organization**: assigns figure numbers, moves images into `figures/figure_N/` folders, writes `interpretation.md` per figure from the executor's pre-computed interpretations, deletes original image files from the root output directory

### Artifacts produced

For each image found in the output directory:
```
run_dir/
  figures/
    figure_1/
      figure_1.png          ← moved from root output dir
      interpretation.md     ← written by image interpreter subagent
    figure_2/
      figure_2.png
      interpretation.md
```
Original image files in the root output directory are deleted after the organized folders are created.

### Output to state

A JSON file (`artifact_map.json`) is written to `run_dir` and a matching `artifact_map` field is added to `State`. Structure:
```json
{
  "steps": [
    {
      "step": 1,
      "description": "Load dataset and compute incidence rates by region",
      "summary": "Loaded 12,450 records; computed annual incidence rates across 5 regions",
      "figures": ["figure_1"],
      "data_files": ["incidence_by_region.csv"]
    }
  ]
}
```

The aggregator receives this map instead of (or alongside) raw `executor_results`, giving it a compact, pre-interpreted briefing it can reference directly when writing the final report.

---

## 13. Aggregator token-budget truncation + RAG fallback

**Files**: `src/sparq/architectures/v1/nodes/aggregator.py`, `src/sparq/settings.py`

### Problem

`aggregator_node` formats all `StepResult`s into a single prompt string with no check against the target model's context window. A plan with many steps, or steps with large `execution_results`/`misc` text, can silently exceed the model's input limit and fail (or get server-side truncated) at `llm.invoke()`.

### Current state

`count_tokens()` uses a local `tiktoken` `cl100k_base` encoding to estimate token count for the formatted results string before the aggregator LLM call — deliberately not `llm.get_num_tokens()`, since some provider integrations (e.g. `ChatGoogleGenerativeAI`, the default provider here) implement that as a live API call (`self.client.models.count_tokens(...)`), which would add network latency/cost to every aggregator run and would be called repeatedly if used inside a retry loop. `cl100k_base` is an approximation for non-OpenAI models but is local and free.

The budget compared against is `llm_config.max_tokens` (an existing, previously-unused field on `LLMSetting`) falling back to `DEFAULT_CONTEXT_WINDOW = 128_000` if unset. There is no per-model context-window table in the codebase, so this is a coarse guess — see "Follow-up" below.

`truncate_results(results, max_tokens)` is currently a stub (`pass`) — not yet implemented.

### Design for `truncate_results`

Iteratively reduce the formatted size, prioritizing which content survives:

1. **Drop least-important fields first, across all steps**, in order: `misc` → `files_generated` → `step` (description). Re-check token count after each drop; stop as soon as it fits.
2. **Shrink `execution_results` text as a last resort** (this is the actual analytical content and most valuable to the aggregator) — proportionally cut per step, e.g. try keeping 75% / 50% / 25% / 10% of each step's text, re-checking after each pass.
3. **If still over budget after aggressive truncation**, warn (`warnings.warn`) that the result set is too large for single-shot aggregation and recommend a RAG-based retrieval approach instead of blind truncation. Return the maximally-truncated string anyway (best effort) rather than failing the run.

### Follow-up: RAG module (not implemented now)

Truncation is a stopgap. For genuinely large result sets (many steps, large datasets/reports), a retrieval step should let the aggregator pull only the passages relevant to the user's query instead of degrading all step content uniformly. This would be a new module (e.g. `tools/rag/`) that indexes `StepResult` content — and would also serve the "retrieval" track in the v2.2 DAG-based macro-decomposition design and the v2.2 RAG sub-component (see v2 section below). Scoped separately since it's a new subsystem, not a targeted fix.

### Follow-up: per-model context window

`DEFAULT_CONTEXT_WINDOW` is a single guessed constant. A more accurate design would either add a lookup table of known model context windows (with this constant as the fallback for unrecognized models), or a `context_window` field on `LLMSetting` so it's configurable per-node in `config.toml`.

---

## 14. Bedrock grammar-size blowup with many bound tools

**Files**: `src/sparq/architectures/v1/nodes/executor.py`, `config_v1.toml` (local dev config)

### Problem

When the executor worker's local dev config (`config_v1.toml`) points `llm_config.executor` at an `aws_bedrock` model (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`), every worker call fails immediately on the first LLM turn — before any tool is even invoked — with:

```
ValidationException: The model returned the following errors: Compiled grammar size (484.8MB)
exceeds maximum allowed size (300MB). Simplify your JSON schema to reduce grammar complexity.
```

Reproduced 2026-07-02 via `uv run python -m sparq.architectures.v1.nodes.executor` (the fixed `test_executor`/`__main__` block — see item covering the parallel-execution todo list). Confirmed via a LangSmith trace and a direct patch of `ChatBedrockConverse`'s underlying `client.converse` call.

### Root cause (confirmed)

At the time this was reproduced, `execute_single_step_worker` built its agent with `deepagents.create_deep_agent`, which binds 7 built-in tools (`write_todos`, `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `task`) in addition to the 9 tools this codebase defines (`load_dataset`, `get_sheet_names`, `python_repl_tool`, `find_csv_excel_files`, `get_cached_dataset_path`, and the 4 `filesystemtools`) — 16 tools total bound to a single call. `response_format=StepResult` forces the model into a required tool-choice, so Bedrock's Converse API must compile a single constrained-decoding grammar covering all 16 tool schemas plus the structured-output extraction schema at once.

Measured total raw JSON schema text for all 16 tools: ~5KB — far too small to explain a 484.8MB compiled grammar by size alone. The blowup is combinatorial, not size-driven: `anyOf`/nullable fields (e.g. `Optional[str] = None` params such as `python_repl_tool`'s `code` argument) multiply across tools when a single automaton must represent "any of 16 tools, each with its own optional-field branches."

Two unrelated but adjacent tool-schema bugs were found and fixed while investigating (both only surfaced under Bedrock's stricter schema validation — Gemini, the repo's default provider, tolerates them):
- `get_sheet_names(file_path)` had no type annotation, producing an empty `{}` JSON schema Bedrock rejected outright.
- `find_csv_excel_files(root_dir: Path | str)` produced a `"format": "path"` JSON schema field Bedrock doesn't support; narrowed to `root_dir: str` (the function already coerced to `Path` internally).

Fixing those two did not resolve the grammar-size error on its own.

**This is a known, documented limitation of Claude's structured-output/tool-use grammar compiler — not specific to Bedrock, `deepagents`, or this codebase.** See [anthropics/anthropic-sdk-python#1185](https://github.com/anthropics/anthropic-sdk-python/issues/1185): "Structured outputs: 'compiled grammar is too large' error needs better documentation and higher limits for complex schemas." Per Anthropic's own docs, the API compiles the JSON schema of every bound tool (plus any forced-choice structured-output schema) into a single constrained-decoding grammar, and "schema complexity features like optional parameters, union types, nested objects, and number of tools interact in ways that can make the compiled grammar disproportionately large" — confirming the combinatorial (not size-driven) explosion observed here. As of this writing there is no documented workaround besides reducing schema complexity (fewer tools, fewer `Optional`/union fields) — no raised-limit flag or opt-out exists.

### Resolution

Swapping `create_deep_agent` for `langchain.agents.create_agent` (same call shape: `model`, `tools`, `system_prompt`, `response_format=StepResult`) resolved the grammar-size error. `create_agent` does not add `deepagents`' 7 built-in tools, so the worker binds only the 9 tools this codebase defines — enough to bring Bedrock's compiled grammar back under the 300MB limit. Confirmed working end-to-end via `test_executor` against `aws_bedrock` / `us.anthropic.claude-sonnet-4-5-20250929-v1:0` on 2026-07-02 (real analysis output — plots, CSVs, a report — written to the run's output directory).

**Trade-off**: this drops `deepagents`' subagent delegation (`task`) and todo-list tracking (`write_todos`), which were the reason `create_deep_agent` was adopted in the first place. `executor.py` currently imports both `create_agent` (used) and `create_deep_agent` (unused, left in place intentionally as a marker to revisit).

### Not yet done

- Determine whether `deepagents`' subagent delegation / todo tracking can be recovered on Bedrock without hitting the grammar limit — e.g. by trimming which built-in tools `create_deep_agent` binds (if configurable), or by reducing the custom tool set bound alongside it so the combined total stays low enough.
- Confirm whether `response_format`'s forced tool-choice is the primary multiplier — if so, whether there's a way to request `tool_choice="auto"` for structured output on Bedrock specifically, which might allow more tools to coexist.
- Consider whether this limit reappears as more custom tools are added to `execute_single_step_worker` over time, even with plain `create_agent`, since the underlying Anthropic grammar-compiler limit is not itself fixed (see anthropics/anthropic-sdk-python#1185 above) — only worked around here by keeping the tool count low.

---

## 15. User-facing step-completion tracker (resumable)

**Files**: `src/sparq/architectures/v1/nodes/executor.py`, `src/sparq/architectures/v1/system.py`, `src/sparq/schemas/state.py`

**Design doc**: [`docs/parallel_execution.md`](parallel_execution.md), "Future requirement: user-facing step-completion tracker" section

### Problem

There is currently no way for a user-facing UI to show which plan steps have completed while the
executor is mid-run, and no way to hand control back to the user (they disconnect or do something
else) and later reconnect to accurate progress. The executor's current `asyncio.gather`-in-one-node
design (see `parallel_execution.md`) is opaque to LangGraph: the graph only sees `executor_node`
start and finish as a single unit, with no per-step events and no mid-node checkpoint to resume
progress from.

### Design

Two implementation paths, laid out in full in `parallel_execution.md`:

1. **Lightweight, no rearchitecture**: switch `asyncio.gather` to `asyncio.as_completed` so results
   are yielded as each step lands, and call `get_stream_writer()` inside
   `execute_single_step_worker` per completed step; consume via `stream_mode="custom"`. Gives a
   live progress feed while the connection stays open, but the progress state itself isn't
   persisted anywhere the graph's own checkpointer knows about — a reconnecting user would need a
   hand-built side store (e.g. completed step IDs written somewhere, keyed by `run_id`).
2. **Migrate to `Send`-based fan-out**: each worker becomes a distinct graph node invocation
   (dispatched via `langgraph.types.Send`, carrying a `WorkerState` payload), which makes per-step
   completion a native `stream_mode="updates"` event and makes `completed_plan_steps`/`results`
   resumable from a real LangGraph checkpoint — no bespoke side store needed. This is the design
   that was originally scaffolded (`WorkerState` in `state.py`, `Send`/`END` imports in
   `executor.py`) and abandoned in favor of `asyncio.gather` before this requirement existed; see
   `parallel_execution.md`'s "Alternative considered" section for why it lost that comparison and
   what reviving it would require (the dispatcher-loop wiring for dependency batching was never
   built).

### Recommendation

If the actual requirement is "hand control back and reconnect later with accurate progress,"
prefer the `Send`-based migration — resumability is structural there, not bolted on. If a live
progress bar on an open connection turns out to be sufficient in practice, the `get_stream_writer()`
+ `as_completed` change is far less work and ships without touching the executor's control flow.

---

## 16. Sub-task parallelism within a worker (`spawn_subtasks`)

**Files**: `src/sparq/tools/subtask_tools.py` (new), `src/sparq/architectures/v1/nodes/executor.py`, `src/sparq/architectures/v1/prompts/executor_message.txt`

**Design doc**: [`docs/subtask_parallelism.md`](subtask_parallelism.md)

### Problem

The executor parallelizes independent plan `Step`s (`docs/parallel_execution.md`), but a single
step's worker has no way to parallelize independent sub-parts of its own task — e.g. computing
several unrelated statistics over the same loaded dataset happens one tool call at a time.

### Why not `deepagents`' generic `task` tool

Considered and rejected: `SubAgentMiddleware` builds each subagent's tools once at
agent-construction time, so every `task` call to the same subagent type would reuse one REPL tool
bound to one fixed namespace path. Since sub-tasks here need to leave variables behind for the
parent step (not just report text), concurrent `task` calls would race on the same namespace pickle
file — the same class of bug `docs/improvements.md` #1 already solved at the step level. Full
reasoning and a purpose-built `spawn_subtasks` tool (with the same seed/execute/merge/cleanup
namespace lifecycle, one level deeper, and a one-level recursion cap) are in `subtask_parallelism.md`.

### Not yet done

Design only — `spawn_subtasks` has not been implemented or wired into `execute_single_step_worker` yet.

---

# SPARQ v2 Architecture

The following improvements are drawn from the SPARQ v2 system design specification. They are larger in scope than the incremental fixes above — each represents a significant new subsystem rather than a targeted change to existing code. They are listed here for reference and roadmap planning.

## v2 Priority Summary

| Change | Effort | Impact | Done |
|--------|--------|--------|------|
| Data cleaning & validation pre-pipeline | High | High (data quality) | [ ] |
| DAG-based macro-decomposition (Lead Supervisor) | High | High (query complexity) | [ ] |
| Speculative parallel execution tracks (K paths) | High | High (result quality) | [ ] |
| Editorial synthesis gatekeeper | Medium | Medium (output quality) | [ ] |

## v2 Time to Completion

Estimates are for a solo developer. The team column assumes 2–3 people with no coordination overhead on independent phases.

| Component | Solo | Team (2–3) | Notes |
|-----------|------|------------|-------|
| v2.1 Data cleaning & validation loop | 3–4 weeks | 2–3 weeks | Nothing reusable; Cleaner + Validator nodes, feedback loop graph wiring, 4-artifact schema all new |
| v2.2 DAG-based macro-decomposition | 4–6 weeks | 2–4 weeks | DAG schema + dependency-aware dispatcher are new; RAG sub-component alone is ~2–3 weeks |
| v2.3 Speculative parallel execution | 5–7 weeks | 3–5 weeks | Most complex phase; REPL subsystem (~80%) reusable, but K-path fan-out, Docker infra, and self-healing loop are new |
| v2.4 Editorial synthesis gatekeeper | 1–2 weeks | 1 week | Straightforward new node; aggregator structured output (improvement 8) is a prerequisite |
| Integration & end-to-end testing | 2–3 weeks | 1–2 weeks | Cross-phase wiring, failure mode handling, full pipeline tests |
| **Total** | **15–22 weeks** | **9–15 weeks** | |

### Assumptions

- **"Fine-tuned epidemiology model"** in the spec is treated as prompt engineering on a general frontier model, not an actual fine-tuning run. A real fine-tuning effort would add months and is outside scope here.
- **Parallel sandboxing** uses `multiprocessing.spawn` (already the REPL's isolation mechanism) rather than per-path Docker containers. Switching to Docker adds ~2–3 weeks of DevOps work to v2.3.
- **RAG pipeline** (v2.2 Retrieval track) targets locally available documents only. Connecting to live external databases (GenBank, PubMed, etc.) adds 2–4 weeks depending on API availability and ingestion volume.
- Estimates cover implementation and unit tests. Domain validation (i.e., verifying the epidemiological outputs are scientifically correct) is out of scope and handled separately.

### What is reusable from v1

| v1 Component | Reuse in v2 | Reuse % |
|---|---|---|
| `tools/python_repl/` | v2.3 per-script sandbox | ~80% |
| `utils/get_llm.py`, `settings.py` | All new nodes | ~90% |
| `nodes/aggregator.py` | v2.4 draft input | ~30% |
| `nodes/planner.py` | v2.3 script generation | ~20% |
| `system.py` graph wiring | New DAG dispatcher | ~10% |
| `nodes/router.py`, `nodes/executor.py` | Superseded | ~0% |

---

## v2.1. Data Cleaning & Validation Pre-Pipeline

**New files**: `nodes/cleaner.py`, `nodes/data_validator.py`, `schemas/cleaning_artifacts.py`

Currently the system assumes datasets are already clean and ready to use. Before any analytical query is processed, a dedicated pre-pipeline should guarantee structural and semantic integrity of the target dataset.

**Cleaner Agent** (`nodes/cleaner.py`): scans incoming datasets, searches for accompanying data dictionaries or schema documentation. If documentation is absent, autonomously explores file headers, infers column types, and reconstructs the schema. Produces four standardised artifacts:

1. A cleaned dataset
2. A textual parsing and analysis report
3. A structured JSON metadata schema (column names, types, constraints, null tolerances)
4. A data-access loader interface for downstream agents

**Data Validator** (`nodes/data_validator.py`): receives the four artifacts and programmatically generates and executes validation scripts (boundary checks, type assertions, null-value tolerances). If any test fails, it returns the stack trace to the Cleaner Agent for correction. This feedback loop runs up to a configurable maximum (default 10 iterations); if unresolved, the pipeline halts and alerts the user before any analytical node runs.

**Graph change**: the cleaning loop runs as a prerequisite subgraph before the main `router → planner → executor` chain. The JSON metadata schema produced here replaces the static `data_manifest.json` as the planner's data reference.

---

## v2.2. DAG-Based Macro-Decomposition (Lead Supervisor)

**New files**: `nodes/lead_supervisor.py`, `schemas/dag.py`

The current planner produces a flat, ordered list of steps. For complex multi-part queries this is insufficient: independent sub-questions cannot be parallelised and the planner has no way to express dependencies between them.

**Lead Supervisor** (`nodes/lead_supervisor.py`): replaces the planner as the top-level decomposition node. It receives the user query and the JSON metadata schema from the cleaning pipeline, then decomposes the query into a DAG of atomic sub-questions. Each node in the DAG carries:

```python
class DAGNode(BaseModel):
    id: str
    question: str
    track: Literal["clarification", "retrieval", "simple_analysis", "complex_analysis"]
    deps: List[str]  # upstream node IDs that must resolve first
```

**Track routing:**
- `clarification` — missing parameters (geography, strain, date range) are returned directly to the user before execution proceeds
- `retrieval` — routes to a RAG agent querying local scientific literature or genomic databases
- `simple_analysis` — descriptive statistics and basic slicing; routed directly to automated tools
- `complex_analysis` — triggers the speculative parallel execution track (v2.3)

The Lead Supervisor's context window is intentionally restricted to the user query, the metadata schema, and DAG execution state — it never sees raw code traces or executor outputs.

**Graph change**: the existing `planner → executor` edge becomes one branch (`complex_analysis`) of a conditional dispatch from the Lead Supervisor. DAG nodes with no unresolved `deps` are dispatched concurrently via `asyncio.gather`.

---

## v2.3. Speculative Parallel Execution Tracks (Trajectory Supervisor)

**New files**: `nodes/trajectory_supervisor.py`, `nodes/parallel_executor.py`

When the Lead Supervisor dispatches a `complex_analysis` node, a dedicated execution track handles it. This track runs K competing methodological approaches in parallel and synthesises their results rather than committing to a single approach upfront.

**Trajectory Supervisor** (`nodes/trajectory_supervisor.py`): receives the isolated sub-question. Before any code is generated, it defines K execution mandates — explicit, domain-informed descriptions of distinct methodological paths (e.g. a deterministic SEIR baseline, an XGBoost ensemble, a neural representation). It does not generate code; it only specifies strategy.

**Planner** (extended): takes the K mandates and generates one Python script per path. Each script is self-contained and independently executable.

**Parallel Executor** (`nodes/parallel_executor.py`): launches all K scripts simultaneously in isolated sandboxed runtimes (Docker containers or `multiprocessing.spawn` processes). Per-script self-healing: on a standard interpreter exception (`SyntaxError`, `KeyError`, `ModuleNotFoundError`), the error is captured locally and a code-optimised model attempts an automated fix. This inner loop runs up to a configurable maximum (default 10 retries) per script. If a script fails to resolve, it is marked failed; the remaining scripts continue. Failures at this layer never surface to the Trajectory Supervisor's context.

**Synthesis**: once all scripts exit, the Trajectory Supervisor compiles their outputs, metrics, and data summaries into a standardised Fact Package. All successful paths are preserved — no pathway is destructively pruned. The comparative data from alternate methodologies is retained as high-value metadata. The Trajectory Supervisor's local context is then discarded.

**Relationship to existing REPL**: the existing `tools/python_repl/` subsystem handles single-script subprocess isolation and namespace persistence. The parallel executor wraps this per-script, adding the self-healing loop and multi-process fan-out. The REPL internals are largely reusable.

---

## v2.4. Editorial Synthesis Gatekeeper

**New files**: `nodes/synthesis_supervisor.py`

The current aggregator is a single LLM call with no validation of its output. For publication-grade reports, the draft should pass an independent editorial review before being delivered to the user.

**Synthesis Supervisor** (`nodes/synthesis_supervisor.py`): a third independent LLM instance (separate from the Lead Supervisor and Trajectory Supervisor). It receives the original user query, the compiled Fact Packages from the execution phase, and the aggregator's draft report. It checks strictly for:

- Narrative flow and internal logical consistency
- Correct citation of figures by number (cross-checked against the figure manifest from improvement 5)
- Formatting alignment with scientific literature conventions
- Factual consistency against the Fact Packages — it does not re-run code or re-audit arithmetic

If the draft fails review, it is returned to the aggregator with specific editorial notes for rewrite. This loop runs up to a configurable maximum (default 3 iterations). After the ceiling is reached or the report passes, it is locked and delivered.

**Relationship to existing aggregator**: the aggregator node is unchanged; the Synthesis Supervisor sits downstream of it as an additional conditional edge in the graph. The aggregator's structured output schema (improvement 8) is a prerequisite for this node to function reliably.
