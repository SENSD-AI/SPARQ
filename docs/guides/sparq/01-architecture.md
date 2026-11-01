# 1. Architecture and shared state

A SPARQ run starts with a question and ends with a structured result and saved
files. To understand what happens between those points, begin with three parts:
**the graph decides which stage runs next, agents decide what to do within a
stage, and tools perform concrete operations such as loading data or running Python.**

For our example question about Missouri counties, the graph takes the analytical
route:

```text
START → router → planner → executor → aggregator → saver → END
```

The router can also answer a question directly. In that case, the path is shorter:

```text
START → router → saver → END
```

## The graph gives the run its structure

`Agentic_system` loads prompts and settings, binds each node to its configured
models and paths, and compiles a LangGraph `StateGraph`. A node is a stage in this
graph. The router and planner make structured model calls; the executor creates
worker agents; the aggregator uses an agent to synthesize results. The saver
uses deterministic Python helpers to write files.

The pipeline therefore has a fixed outer order, while the analytical plan and
worker tool calls depend on the question. The planner can generate several
steps, but those steps are all processed inside the executor node. They are not
separate nodes in the outer graph.

## State is the handoff between stages

Suppose the planner has decided which datasets our county comparison needs.
The executor needs that plan without asking the planner to repeat it. SPARQ
passes a `State` object through the graph for this purpose.

| State field | What it contributes |
| --- | --- |
| `query`, `question_asked_at` | The original question and its receipt time |
| `messages` | The human question and assistant messages; the saver reads the last assistant answer |
| `route` | Whether analysis is needed |
| `plan` | The structured analytical steps and their dependencies |
| `data_context` | Dataset metadata used for planning and execution |
| `completed_plan_steps`, `results` | Terminal step IDs and their structured results |
| `artifacts` | Paths returned when generated executor files are organized |

Nodes return updates to this state. Messages use LangGraph's `add_messages`
reducer. Worker agents also have their own local conversations with their tools;
the outer state is not a complete transcript of every worker tool exchange.

## State, Python variables, and files have different jobs

The graph state carries the plan and result summaries. Actual Python objects,
such as loaded DataFrames, are retained in temporary **namespaces**. A namespace
is the mapping from a variable name to its Python value. A worker can refer to
`county_rates` in a later Python call because its namespace preserves that value.
Chapter 3 explains how this works across fresh Python processes.

Plots, exported tables, and reports are files under the run directory. Those
files remain after temporary namespaces are cleaned up. Keeping these three
forms of information in mind makes the later stages easier to follow:

| Information | Lifetime and location |
| --- | --- |
| Plan, messages, and step summaries | Graph state, with snapshots saved to `trace.json` |
| Python variables | Temporary run/step namespace files |
| Generated evidence and final outputs | Run directory on disk |

The current CLI handles one question per invocation. Although the state includes
messages, the graph is compiled without a checkpointer and the CLI has no
conversation loop. Persistent multi-turn chat remains a separate design topic.

## Where this is implemented

- [Graph and run orchestration](../../../src/sparq/architectures/v1/system.py)
- [State contract](../../../src/sparq/schemas/state.py)
- [Routing, planning, and step-result contracts](../../../src/sparq/schemas/planning.py)

Next: [start a run and turn the question into a plan](02-starting-and-planning.md).
