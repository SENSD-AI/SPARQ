# 2. Starting, routing, and planning

[Chapter 1](01-architecture.md) introduced the outer graph and its state. Now
follow our county-comparison question into that graph.

## Stage 1: prepare the run

After completing the [repository setup](../../../README.md#setup), start the CLI:

```bash
uv run sparq
```

The CLI loads environment settings and `V1Settings`, displays the effective
settings, and prompts for a question. `uv run sparq -t` takes the configured
`test_query` instead. Both paths invoke the same `Agentic_system.run()` method.

V1 settings combine the packaged defaults, checkout `config_v1.toml`, and user
`v1/config.toml`, with later TOML values overriding earlier ones. The configuration
selects the router, planner, executor, aggregator, and vision models, along with
prompt and output paths. Prompts are loaded when the system instance is created.

Every v1 model-construction call forwards configured `max_tokens`, `temperature`,
`top_p`, and `top_k` as explicit arguments, including for the shared vision model.
Defaults come from the resolved configuration; `get_llm()` does not inject its
own generation defaults and omits any options that are still `None`.
Explicit zero values are preserved. `max_tokens`
limits one generated response, while `recursion_limit` controls agent graph steps.
Sampling support depends on the selected provider and model. The existing `effort`
setting is not forwarded by this generation-options path.

`PathSettings` derives a timestamped `run_dir` when settings are constructed.
Its parent is `output_dir`, whose default is `~/tmp/sparq/output`; an optional
`output_stem` prefixes the timestamp. This matters later: calling `run()` again
on the same instance does not itself allocate another output directory.

On entering `run()`, SPARQ records the question receipt time, prepares node
bindings and the graph, and creates the run directory. It accepts a supplied
run ID or generates a UUID. The ID scopes temporary namespaces and logging; it
is distinct from the timestamped output-directory name. Initial state includes
the question, receipt time, and a human message.

## Stage 2: decide whether analysis is needed

The router sends its system prompt and the question to its configured model,
requesting a structured `Router` response. That response contains a boolean
`route` and an optional direct `answer`.

For our county comparison, an analytical routing decision sends the state to
the planner. For a question the router answers directly, it puts an assistant
message in state and sends it to the saver. Direct answers therefore still
receive JSON and report exports, without creating a plan or running workers.

Routing is a model decision; it is not a guarantee that the resulting analysis
is feasible or correct.

## Stage 3: build a dependency-aware plan

The planner loads the data manifest and short dataset summaries. These describe
the available datasets; loading this context does not yet load their DataFrames
into a Python worker. The planner combines the context, its system prompt, and
the original question to request a structured `Plan`.

A plan contains `Step` records. Each step gives its ID, description, dataset
names, rationale, task types, and a list of dependency IDs. To make dependencies
concrete, imagine this simplified plan for our example:

| Step | Task | Dependencies |
| --- | --- | --- |
| 1 | Load and summarize county isolate counts | None |
| 2 | Load the county population data | None |
| 3 | Join the two datasets and calculate rates | 1, 2 |
| 4 | Plot the rates and describe limitations | 3 |

Step 3 requires the outputs of both earlier steps. Steps 1 and 2 can start
independently. The next chapter follows this exact dependency pattern into the
executor; the actual model-generated plan can differ.

The graph allows up to five planner attempts for Pydantic validation errors.
A schema-valid plan can still contain bad analytical choices or unresolved
dependencies. There is no automatic executor-to-planner feedback edge in the
current outer graph.

## What the executor receives

The planner returns `plan` and `data_context` to the graph state. The question
and routing decision remain available. At this point we have instructions and
dataset metadata; workers still need to locate and load the data and perform
the calculations.

Implementation: [CLI](../../../src/sparq/__main__.py),
[settings](../../../src/sparq/architectures/v1/settings.py),
[path and configuration handling](../../../src/sparq/settings.py),
[router](../../../src/sparq/architectures/v1/nodes/router.py), and
[planner](../../../src/sparq/architectures/v1/nodes/planner.py).

Next: [execute the plan, preserving variables and respecting dependencies](03-execution.md).
