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

---

## 1. Plan steps have no dependency information

**File**: `src/sparq/schemas/output_schemas.py`

The `Step` schema has no way to express inter-step dependencies. As a result, all steps always run sequentially even when they are independent. Adding an optional field:

```python
class Step(BaseModel):
    ...
    depends_on: List[int] = []  # indices of steps this step requires
```

would allow the executor to topologically sort steps and run independent ones in parallel via `asyncio.gather`. This could cut wall-clock time significantly for multi-step plans.

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
