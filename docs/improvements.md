# Architectural Improvement Recommendations

## Priority Summary

| Change | Effort | Impact | Done |
|--------|--------|--------|------|
| Scope REPL namespace to run ID | Low | High (concurrency safety) | [ ] |
| Step dependency + parallel execution | Medium | High (throughput) | [ ] |
| State as Pydantic with reducers | Medium | Medium (robustness) | [ ] |
| Executor → planner feedback loop | High | High (resilience) | [ ] |
| Plot interpretation via agent-controlled tool | Medium | High (output quality) | [ ] |
| Figure numbering tool for executor | Low | Medium (output quality) | [ ] |
| Aggregator academic referencing of outputs | Medium | High (output quality) | [ ] |
| Aggregator writes markdown report → PDF | Low | High (output quality) | [ ] |
| Multi-route router for general use cases | High | High (product scope) | [ ] |
| Aggregator output isn't structured | Low | Medium (robustness) | [ ] |
| LLM model validation at startup | Low | Medium (reliability) | [ ] |

---

## 1. Global REPL namespace breaks concurrent runs

**Files**: `src/sparq/tools/python_repl/namespace.py`, `src/sparq/tools/python_repl/executor.py`, `src/sparq/tools/python_repl/python_repl_tool.py`, `src/sparq/architectures/v1/nodes/executor.py`

The persistent namespace path is a module-level global:

```python
_PERSISTENT_NS_PATH = None  # single global per process
```

Two concurrent runs share the same pickle file, causing namespace collisions. Additionally, a per-invocation random ID (e.g. `uuid4()` inside the node) breaks replanning (improvement 4) — the second executor invocation would start with an empty namespace, losing everything the first run computed.

The fix uses `RunnableConfig`: a `run_id` is generated once in `system.py` at the start of each `run()` call and passed explicitly into `graph.astream()` via `config={"configurable": {"run_id": run_id}}`. LangGraph then injects this config into every node invocation, including re-invocations of the same node within one run (e.g. replanning). The namespace temp file is cleaned up in a `finally` block after the graph completes, so files don't accumulate.

**`namespace.py`** — replace the module-level global with a process-level dict keyed by `run_id`, and add a cleanup function:

```python
_ns_paths: dict[str, str] = {}

def get_ns_path_for_run(run_id: str) -> str:
    if run_id not in _ns_paths or not os.path.exists(_ns_paths[run_id]):
        fd, path = tempfile.mkstemp(suffix=f"_{run_id}_persistent_ns.pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump({}, f)
        _ns_paths[run_id] = path
    return _ns_paths[run_id]

def cleanup_ns_for_run(run_id: str):
    path = _ns_paths.pop(run_id, None)
    if path and os.path.exists(path):
        os.unlink(path)
```

Remove `get_persistent_ns_path` and `clear_persistent_namespace`.

**`executor.py` (REPL)** — replace `persist_namespace: bool` with `ns_path: str | None`:

```python
def execute_code(code: str, ns_path: str | None = None, timeout: int = 2*60) -> OutputSchema:
    if ns_path is not None:
        ns_is_temp = False
    else:
        ns_fd, ns_path = tempfile.mkstemp(suffix="_ns.pkl")
        with os.fdopen(ns_fd, "wb") as f:
            pickle.dump({}, f)
        ns_is_temp = True
```

**`python_repl_tool.py`** — change the module-level `@tool` to a factory so the run-scoped path is baked in as a closure. The LLM still controls `persist_namespace`; the path it resolves to is now run-scoped:

```python
def make_python_repl_tool(ns_path: str):
    @tool(args_schema=PythonREPLInput, response_format='content_and_artifact')
    def python_repl_tool(code: str = "", persist_namespace: bool = False):
        execution_result = execute_code(code or "", ns_path=ns_path if persist_namespace else None)
        ...
    return python_repl_tool
```

**`nodes/executor.py`** — accept `RunnableConfig`, resolve the `ns_path`, pass it to the tool factory and to `_build_context`:

```python
from langgraph.types import RunnableConfig

def executor_node(state: State, config: RunnableConfig, llm_config: ..., prompt: ..., output_dir: ...):
    run_id = config.get("configurable", {}).get("run_id", "default")
    ns_path = get_ns_path_for_run(run_id)

    _tools = [..., make_python_repl_tool(ns_path), ...]
    ...
    context = _build_context(results, ns_path)
```

`_build_context` gains a `ns_path: str` parameter replacing the `get_persistent_ns_path()` call on line 38.

**`system.py`** — generate `run_id` explicitly and clean up after the graph finishes:

```python
import uuid
from sparq.tools.python_repl.namespace import cleanup_ns_for_run

async def run(self, user_query: str):
    self._get_node_definitions()
    self._build_graph()
    run_id = str(uuid.uuid4())
    input_data = {"query": user_query}
    try:
        async for chunk in self.graph.astream(
            input=input_data,
            config={"configurable": {"run_id": run_id}},
            stream_mode="updates"
        ):
            print(chunk)
    finally:
        cleanup_ns_for_run(run_id)
```

---

## 2. Plan steps have no dependency information

**File**: `src/sparq/schemas/output_schemas.py`

The `Step` schema has no way to express inter-step dependencies. As a result, all steps always run sequentially even when they are independent. Adding an optional field:

```python
class Step(BaseModel):
    ...
    depends_on: List[int] = []  # indices of steps this step requires
```

would allow the executor to topologically sort steps and run independent ones in parallel via `asyncio.gather`. This could cut wall-clock time significantly for multi-step plans.

---

## 3. State is an unvalidated TypedDict

**File**: `src/sparq/schemas/state.py`

`State` is a plain `TypedDict`, so bad updates silently succeed at runtime. Two improvements:

- Use a Pydantic model for validation on updates.
- Add LangGraph `Annotated` reducers for fields that should merge rather than replace. In the current architecture `executor_results` is written once by the executor node, so no reducer is needed today. However, if replanning is introduced (see improvement 4) and the executor runs multiple times, a merge reducer becomes necessary to avoid the second run overwriting the first:

```python
from typing import Annotated
import operator

class State(TypedDict):
    executor_results: Annotated[dict, operator.or_]
```

---

## 4. No replanning on executor failure

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

## 5. Executor cannot interpret generated plots

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

## 6. Figure numbering tool for executor

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

## 7. Aggregator academic referencing of outputs

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

## 8. Aggregator writes markdown report converted to PDF

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

## 9. Multi-route router for general use cases

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

## 10. Aggregator output isn't structured

**File**: `src/sparq/nodes/aggregator.py`

The router, planner, and executor all use `response_format` for structured output. The aggregator calls `llm.invoke(prompt_str)` and reads `.content` directly. This inconsistency makes it harder to evolve the output schema (e.g., adding citations, confidence scores, or section headers) and bypasses the validation guarantees the rest of the pipeline relies on.

---

## 11. LLM model validation at startup

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

# v1 → v2 Infrastructure Bridge

The following changes are prerequisite scaffolding — neither pure v1 bug fixes nor v2 features. They restructure the codebase to support multiple architectures so that v2 can be built alongside v1 without shared-config collisions or hardcoded paths.

## Bridge Priority Summary

| Change | Effort | Impact | Done |
|--------|--------|--------|------|
| `architectures/` directory structure | Low | High (prerequisite for all below) | [x] |
| `BaseAgenticSettings` refactor in `settings.py` | Low | High (enables per-arch subclassing) | [x] |
| Per-architecture `settings.py` + `default_config.toml` | Low | High (eliminates config discrepancy) | [x] |
| `--architecture` CLI arg in `__main__.py` | Low | High (runtime arch selection) | [x] |
| `setup.py` multi-architecture config copy | Low | Medium (first-run correctness) | [x] |
| Test suite update for new settings structure | Low | Medium (keeps CI green) | [x] |

---

## Bridge.1. `architectures/` Directory Structure

**Status: done.**

`src/sparq/architectures/v1/` (nodes, system, prompts, `__init__.py`) and `src/sparq/architectures/v2/` (stub `__init__.py`) have been created. `__main__.py` imports `Agentic_system` from `architectures/v1/system`.

---

## Bridge.2. `BaseAgenticSettings` Refactor

**File**: `src/sparq/settings.py`

Rename `AgenticSystemSettings` → `BaseAgenticSettings`. Remove `llm_config: LLMSettings` and `model_config` — both are architecture-specific and belong in subclasses. Remove `LLMSettings` class (moves to `v1/settings.py`). Remove module-level constants `INNER_CONFIG_PATH`, `DEV_CONFIG_PATH`, `USER_CONFIG_PATH` — replaced by per-architecture constants.

Keep: `LLMSetting`, `PathSettings`, `ENVSettings`, and `settings_customise_sources` (inherited by subclasses; provides `deep_merge=True` for TOML layering).

---

## Bridge.3. Per-Architecture `settings.py` + `default_config.toml`

**New files**: `src/sparq/architectures/v1/settings.py`, `src/sparq/architectures/v1/default_config.toml`

**`v1/settings.py`** defines:

- Path constants:
  ```python
  V1_INNER_CONFIG = Path(__file__).parent / "default_config.toml"
  V1_DEV_CONFIG   = get_project_root() / "config_v1.toml"
  V1_USER_CONFIG  = get_user_config_dir() / "v1" / "config.toml"
  ```
- `V1LLMSettings(BaseModel)` — router, planner, executor, aggregator fields (same structure as the current `LLMSettings`)
- `V1Settings(BaseAgenticSettings)` — adds `llm_config: V1LLMSettings`, sets `model_config` with `toml_file=[V1_INNER_CONFIG, V1_DEV_CONFIG, V1_USER_CONFIG]`

**`v1/default_config.toml`** is a copy of the root `default_config.toml` with `prompts_dir = "architectures/v1/prompts"`. This resolves the discrepancy where settings reported `src/sparq/prompts` but the active code used `architectures/v1/prompts`.

**`v1/system.py`** is updated to import `V1Settings` instead of `AgenticSystemSettings` and restores `self.prompts_dir = self.settings.paths.prompts_dir`.

---

## Bridge.4. `--architecture` CLI Arg

**File**: `src/sparq/__main__.py`

Add:
```python
parser.add_argument('-a', '--architecture', default='v1', choices=['v1'])
```

Import `Agentic_system` dynamically based on the arg:
```python
if args.architecture == 'v1':
    from sparq.architectures.v1.system import Agentic_system
```

Remove the redundant standalone `AgenticSystemSettings(verbose=True)` instantiation. `Agentic_system` receives a `verbose` param and passes it to `V1Settings` internally, so settings are printed once from the right class.

---

## Bridge.5. `setup.py` Multi-Architecture Config Copy

**File**: `src/sparq/setup.py`

Import `V1_INNER_CONFIG`, `V1_USER_CONFIG` from `sparq.architectures.v1.settings`. Replace the single `INNER_CONFIG_PATH → USER_CONFIG_PATH` copy with per-architecture copies:

```python
V1_USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
if not V1_USER_CONFIG.exists():
    shutil.copy2(V1_INNER_CONFIG, V1_USER_CONFIG)
```

When v2 is added, append its equivalent block here.

---

## Bridge.6. Test Suite Update

**File**: `tests/test_settings.py`

Replace `from sparq.settings import AgenticSystemSettings` with `from sparq.architectures.v1.settings import V1Settings`. Rename `TestAgenticSystemSettings` → `TestV1Settings`. Update the `prompts_dir` assertion to verify the resolved path falls inside `architectures/v1/prompts`.

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
| v2.4 Editorial synthesis gatekeeper | 1–2 weeks | 1 week | Straightforward new node; aggregator structured output (improvement 10) is a prerequisite |
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
- Correct citation of figures by number (cross-checked against the figure manifest from improvement 7)
- Formatting alignment with scientific literature conventions
- Factual consistency against the Fact Packages — it does not re-run code or re-audit arithmetic

If the draft fails review, it is returned to the aggregator with specific editorial notes for rewrite. This loop runs up to a configurable maximum (default 3 iterations). After the ceiling is reached or the report passes, it is locked and delivered.

**Relationship to existing aggregator**: the aggregator node is unchanged; the Synthesis Supervisor sits downstream of it as an additional conditional edge in the graph. The aggregator's structured output schema (improvement 10) is a prerequisite for this node to function reliably.
