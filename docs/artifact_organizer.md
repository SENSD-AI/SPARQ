# Artifact Organizer Node Design

Design document for improvement #12: a dedicated post-executor node that organizes output
files and produces a structured metadata map for the aggregator.

**Image interpretation is handled by the executor** via a `make_interpret_image_tool`
tool (see improvement #3) — the executor calls it inline after each `plt.savefig`. The
artifact organizer receives those interpretations as part of `ExecutorOutput` and folds
them into the map; it does not call any vision LLM itself.

This design supersedes improvements #4 (figure numbering tool on executor) and #5
(aggregator figure manifest). Improvement #3 (plot interpretation) is resolved separately
as an executor tool.

---

## Problem

After the executor finishes, the output directory is a flat pile of files produced across
all plan steps:

```
run_dir/
  correlation_matrix.png
  regression_plot.png
  time_series.png
  incidence_by_region.csv
  summary_stats.csv
  notes.txt
```

Three things are missing:

1. **No figure organization.** Images have no consistent numbering or folder structure.
   The aggregator cannot reference "Figure 2" because "Figure 2" doesn't exist yet.

2. **No step-level metadata.** The aggregator receives raw `ExecutorOutput` dicts —
   potentially long, verbose, step-specific outputs. There is no compact "what did each
   step actually produce?" summary it can reason from directly.

3. **Wrong node for bookkeeping.** Improvement #4 proposed giving the executor a
   `get_next_figure_number` tool to call before each `plt.savefig`. That conflates
   analysis with file-system bookkeeping and forces the executor to track state it
   shouldn't own.

---

## Pipeline placement

```
           Figure 1.  New node placement in the pipeline
   ══════════════════════════════════════════════════════════════════

   Router → Planner → Executor → Artifact Organizer → Aggregator → Saver
                                         │
                                    (new node)
                                         │
                            reads output dir, spawns subagents,
                            writes figures/ + artifact_map.json
```

The artifact organizer sits between executor and aggregator. It runs once after all
executor steps have completed. Its only inputs are the executor's output directory and
the executor's structured results; its output flows directly into the aggregator.

---

## Subagent architecture

The node uses a single internal subagent. Image interpretations are not produced here —
they arrive pre-computed in the executor's `ExecutorOutput` (written there by the
`interpret_image` tool during execution).

```
           Figure 2.  Internal subagent structure
   ══════════════════════════════════════════════════════════════════

   artifact_organizer_node(state)
        │
        ├─ assign figure numbers from executor_results
        ├─ move images into figures/figure_N/ folders
        ├─ write interpretation.md per figure (from ExecutorOutput)
        │
        ├─ inspector_agent(data_files + text_files)   ─── single subagent
        │      → one-line summary per step
        │
        └─ assemble artifact_map.json
           delete original image files from root dir
```

### Inspector subagent

**Tools**: `make_python_repl_tool(ns_path)` (to inspect data still in the REPL namespace),
file read tool (for text/report files in the output directory)

**Task**: for each executor step, produce a one-line summary of what that step produced —
what data was loaded or computed, what files were written, how many records, what the
key finding was. Can use the REPL to call `.info()` or `.describe()` on a loaded DataFrame
if the summary needs to be grounded in actual values rather than filename inference.

**Output**: list of `{step_index, summary, data_files: []}` entries merged into the JSON map.

---

## Artifact folder structure

After the node completes, the output directory is reorganized:

```
run_dir/
  figures/
    figure_1/
      figure_1.png          ← moved from root (was: correlation_matrix.png)
      interpretation.md     ← written by image interpreter subagent
    figure_2/
      figure_2.png          ← moved from root (was: regression_plot.png)
      interpretation.md     ← written from ExecutorOutput (produced by interpret_image tool)
    figure_3/
      figure_3.png
      interpretation.md
  incidence_by_region.csv   ← data files stay in root (unchanged)
  summary_stats.csv
  notes.txt
  artifact_map.json         ← new: the metadata map (see below)
  trace.json                ← written later by saver
  final_answer.json         ← written later by saver
```

Original image files (e.g. `correlation_matrix.png` in the root) are deleted after the
organized copies are confirmed written. Data files and text reports are not moved.

---

## The artifact map

`artifact_map.json` is the primary output passed to the aggregator. It is a compact,
structured briefing of what the executor produced, keyed by step:

```json
{
  "steps": [
    {
      "step": 1,
      "description": "Load dataset and compute annual incidence rates by region",
      "summary": "Loaded 12,450 records across 5 regions (2015–2023); computed incidence rates per 100k population",
      "figures": [],
      "data_files": ["incidence_by_region.csv"]
    },
    {
      "step": 2,
      "description": "Correlation analysis between temperature and outbreak frequency",
      "summary": "Pearson r = 0.71 (p < 0.001) between mean summer temperature and annual outbreak count across regions",
      "figures": ["figure_1", "figure_2"],
      "data_files": ["correlation_results.csv"]
    },
    {
      "step": 3,
      "description": "Regression model: outbreak count ~ temperature + humidity",
      "summary": "OLS model R² = 0.64; temperature coefficient significant (β = 2.3, p < 0.01), humidity not significant",
      "figures": ["figure_3"],
      "data_files": ["regression_summary.csv"]
    }
  ],
  "figures": {
    "figure_1": {
      "path": "figures/figure_1/figure_1.png",
      "interpretation": "Scatter plot showing positive correlation between mean summer temperature (x-axis) and annual Salmonella outbreak count (y-axis). Points cluster clearly above the trend line at temperatures > 28°C, suggesting a threshold effect."
    },
    "figure_2": {
      "path": "figures/figure_2/figure_2.png",
      "interpretation": "Correlation matrix heatmap across all environmental variables. Temperature shows the strongest correlation with outbreak count (r = 0.71). Humidity and precipitation are weakly correlated with each other but not independently predictive."
    },
    "figure_3": {
      "path": "figures/figure_3/figure_3.png",
      "interpretation": "Regression coefficient plot. Temperature bar extends well beyond the 95% confidence interval boundary; humidity bar overlaps zero, confirming its non-significance."
    }
  }
}
```

The aggregator receives this map (injected as a `{artifact_map}` template variable)
instead of raw executor output dicts. It can reference figures by name, cite summaries
directly, and write a structured report grounded in the actual findings — without needing
to re-run any code or load any files.

---

## Figure numbering logic

Figure numbers are assigned deterministically before any file operations begin:

```python
def assign_figure_numbers(executor_results: list[ExecutorOutput], run_dir: Path) -> list[tuple[Path, int, int]]:
    """
    Returns list of (image_path, figure_number, step_index) sorted by step then mtime.
    """
    assignments = []
    figure_counter = 1
    for step_index, result in enumerate(executor_results):
        images = sorted(
            [Path(run_dir / f) for f in result.files_generated
             if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}],
            key=lambda p: p.stat().st_mtime
        )
        for img_path in images:
            assignments.append((img_path, figure_counter, step_index))
            figure_counter += 1
    return assignments
```

The figure number is used when naming the folder (`figure_N/`) and is also injected into
the executor prompt via a `{next_figure_number}` context variable so the executor can
embed it in the plot title before calling `interpret_image`.

---

## State changes

Add one field to `State` in `src/sparq/schemas/state.py`:

```python
class State(TypedDict):
    ...
    artifact_map: dict | None  # populated by artifact_organizer_node; None until then
```

The aggregator prompt template gains a `{artifact_map}` variable. When the organizer is
present in the graph, this is always populated; when bypassed (e.g. on a `direct_answer`
route), it is `None` and the aggregator falls back to its existing behavior.

---

## What changes per file

| File | Change |
|---|---|
| `src/sparq/architectures/v1/nodes/artifact_organizer.py` | New file — node function, inspector subagent, figure numbering, folder creation, JSON map assembly |
| `src/sparq/schemas/state.py` | Add `artifact_map: dict \| None` field |
| `src/sparq/architectures/v1/system.py` | Add `artifact_organizer_node` to graph; wire `executor → artifact_organizer → aggregator`; instantiate `vision_llm` and pass to executor tool list |
| `src/sparq/settings.py` / `config/config.toml` | Add `vision_llm` config entry under `[llm]` — used by the executor's `interpret_image` tool |
| `src/sparq/architectures/v1/nodes/executor.py` | Add `make_interpret_image_tool(vision_llm)` to tool list; update prompt to call it after each `plt.savefig` |
| `src/sparq/tools/interpret_image_tool.py` | New file — `make_interpret_image_tool(vision_llm)` factory returning the `@tool` |
| `src/sparq/prompts/aggregator_message.txt` | Add `{artifact_map}` template variable; instruct to reference figures by number and cite step summaries |

The REPL subsystem, planner, router, saver, and all schemas except `State` are unchanged.

---

## What this replaces

| Superseded improvement | Original approach | Why this is better |
|---|---|---|
| #4 Figure numbering tool for executor | Give executor a `get_next_figure_number` tool to call before each `plt.savefig` | Numbering is deterministic and consistent when done post-hoc by the organizer; executor cannot miscount by forgetting to call the tool |
| #5 Aggregator academic referencing | Scan output dir and build a figure manifest inside the aggregator node | The organizer produces richer metadata (interpretations from executor + step summaries from inspector) rather than a bare filename list |

Improvement #3 (plot interpretation) is resolved separately as the `interpret_image`
executor tool — see [`docs/improvements.md` #3](improvements.md).
