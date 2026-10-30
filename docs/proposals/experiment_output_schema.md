# Experiment Output Schema

`SystemOutput` captures one SPARQ run against an evaluation question so results can be compared
across ablations/models and fed to a judge later. Its target fields are:

| Field | Type | Source |
|---|---|---|
| `run_id` | `str` | generated in `Agentic_system.run()` and returned as part of `SystemOutput` |
| `query` | `str` | `benchmarks/datasets/questions.json` question text |
| `difficulty` | `int` | `benchmarks/datasets/questions.json`'s `grade` field per question |
| `ablation_config` | `dict` | set by the experiment script itself (which nodes/settings vary per run) |
| `response` | `str` | `State.answer`, written to `final_answer.json` by `saver_node` |
| `token_out` | `int \| None` | SPARQ/LangSmith metadata, see below |
| `models` | `dict` | `V1Settings.llm_config` — per-node provider/model from `config/config.toml`, already available with no extra plumbing |
| `cost` | `float \| None` | Estimated USD cost from SPARQ/LangSmith metadata, see below |
| `time_started` / `time_ended` / `duration` | `datetime` / `datetime` / `float` | measured inside `Agentic_system.run()` |
| `sparq_judge_score` | `dict` | not yet built — no judge exists in this codebase yet |
| `sparq_judge_review` | `str` | not yet built — same as above |

## Token/cost via LangSmith, not manual plumbing

Considered capturing `usage_metadata` off each node's LLM response directly (executor.py:208,
aggregator.py:76 both discard the raw `AIMessage` returned by `agent.invoke()`, which carries
`usage_metadata` from the provider). Rejected in favor of LangSmith, since tracing
(`LANGCHAIN_TRACING_V2`) is already a supported opt-in (`langsmith` is a `pyproject.toml`
dependency) and requires no changes to the graph nodes.

Verified against the LangSmith SDK reference docs (2026-07-16):

- `Client.read_run(run_id)` / `Client.list_runs(...)` return `Run` objects with `total_tokens`,
  `prompt_tokens`, `completion_tokens`, and — computed server-side from LangSmith's pricing map —
  `total_cost`, `prompt_cost`, `completion_cost` directly. No manual summing of prompt+completion
  cost needed.
- Caveat: server-side cost computation only works for models LangSmith has priced. Unclear whether
  this project's non-OpenAI/Anthropic providers (Bedrock, Ollama, OpenRouter, per `CLAUDE.md`) are
  covered — needs a spot check once wired up.
- No SDK helper sums cost across a run tree. A single `agentic_system.run()` call fans out into a
  root run plus per-node child runs (router → planner → executor → aggregator); getting one
  `total_cost`/`total_tokens` per top-level question means calling
  `Client.list_runs(trace_id=root_run_id)` and summing `total_cost` across the children yourself.
- LangSmith's API has a short indexing delay after a run completes before it's queryable — the
  experiment script needs to account for that (poll/retry) rather than querying immediately after
  `run()` returns.

## Batch evaluation output

A batch evaluation should be represented as a first-class experiment with a stable `batch_id` and
one manifest. The proposed `BatchEvalOutput` contains:

| Field | Purpose |
|---|---|
| `batch_id` | Stable UUID, optionally prefixed with a readable timestamp |
| `status` | Overall state: `running`, `completed`, `completed_with_errors`, or `failed` |
| `time_started` / `time_ended` / `duration` | Batch timing metadata |
| `dataset_path` / `dataset_hash` | Identify the exact source dataset used |
| `question_filter` | Record filters such as excluding weather questions or grade `-1` |
| `requested_question_count` | Preserve the CLI request separately from the selected count |
| `selected_question_ids` | Stable dataset IDs for the questions included in the batch |
| `iterations` | Number of repetitions requested per question |
| `max_concurrent_runs` | Concurrency setting used by the experiment |
| `code_version` | Git commit and dirty-worktree signal for reproducibility |
| `configuration` | Shared model, ablation, and evaluation settings |
| `runs` | One `BatchRunEntry` for each `(iteration, question_id)` pair |
| aggregate counts | Planned, completed, failed, cancelled, and interrupted totals |

### Proposed model shape

The exact names can change during implementation, but the intended ownership and optionality are:

```python
RunStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]

BatchStatus = Literal[
    "running",
    "completed",
    "completed_with_errors",
    "failed",
    "interrupted",
]

class BatchRunEntry(BaseModel):
    question_id: int
    question_index: int
    iteration: int
    run_id: str
    result_path: Path
    status: RunStatus = "pending"
    time_started: datetime | None = None
    time_ended: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None

class BatchEvalOutput(BaseModel):
    batch_id: str
    status: BatchStatus = "running"
    time_started: datetime
    time_ended: datetime | None = None
    duration: float | None = None
    dataset_path: Path
    dataset_hash: str
    question_filter: dict
    requested_question_count: int | None
    selected_question_ids: list[int]
    iterations: int
    max_concurrent_runs: int
    code_version: dict
    configuration: dict
    runs: list[BatchRunEntry]
```

`question_index` is display metadata only. `question_id` is the durable identity. Paths stored in
the manifest should be relative to the batch directory so moving or archiving the complete batch
does not break its links.

The manifest may either store aggregate counts or compute them from `runs`. Computing them avoids
two sources of truth; storing them makes external inspection cheaper but requires updating them in
the same locked manifest write as each run status.

The batch manifest should contain indexing and linkage metadata rather than duplicate every
`SystemOutput`. A `BatchRunEntry` should contain:

- `run_id`, generated when the batch plan is created
- `question_id` and, if useful for display, `question_index`
- `iteration`
- `status`
- `result_path`
- optional `error_type` and `error_message`

Because `asyncio.gather(..., return_exceptions=True)` can return exceptions, the in-memory result
type is not strictly `list[SystemOutput]`. Failures must be converted into failed run entries rather
than treated as successful `SystemOutput` values.

## Run status lifecycle

Create all run entries before scheduling their tasks, with `status` defaulting to `pending`.
Derive status from the task lifecycle, not from the response content:

```text
pending → running → completed
                  ↘ failed
                  ↘ cancelled
```

- `pending`: selected and scheduled but not yet holding a semaphore slot.
- `running`: set immediately after the task acquires the semaphore.
- `completed`: the call returned a valid `SystemOutput` and wrote `result.json`.
- `failed`: the call raised an exception; record its type and message.
- `cancelled`: the asyncio task was explicitly cancelled.
- `interrupted`: reconciliation state for an entry left `running` after the batch process exited.

### Creating pending entries

All planned entries should exist before any SPARQ calls begin. Generate the `run_id` at this point
and pass it into `Agentic_system.run()` later. This lets a pending entry already name its eventual
artifact directory and avoids relying on timestamps as identities. For each iteration and selected
question, create an entry before appending its coroutine:

```python
run_entry = BatchRunEntry(
    question_id=question["id"],
    question_index=question_index,
    iteration=iteration,
    run_id=run_id,
    result_path=relative_run_dir / "result.json",
)
batch.runs.append(run_entry)

tasks.append(
    run_bounded(
        semaphore,
        agentic_system,
        question["text"],
        run_entry,
    )
)
```

Because `pending` is the model default, callers do not need to set the string repeatedly. A task
waiting to acquire the semaphore remains `pending`; task creation by itself does not mean that the
SPARQ run has started.

### Owning status transitions

The bounded task wrapper should own run-status transitions because it observes both semaphore
admission and the outcome of `Agentic_system.run()`. Its intended behavior is:

```python
async with semaphore:
    run_entry.status = "running"
    run_entry.time_started = now()
    await persist_manifest()

    try:
        result = await agentic_system.run(question)
    except asyncio.CancelledError:
        run_entry.status = "cancelled"
        run_entry.time_ended = now()
        await persist_manifest()
        raise
    except Exception as error:
        run_entry.status = "failed"
        run_entry.error_type = type(error).__name__
        run_entry.error_message = str(error)
        run_entry.time_ended = now()
        await persist_manifest()
        return None
    else:
        run_entry.status = "completed"
        run_entry.run_id = result.run_id
        run_entry.result_path = relative_result_path
        run_entry.time_ended = now()
        await persist_manifest()
        return result
```

This is illustrative rather than a required function signature. The important rules are that
`running` is set after semaphore acquisition, `CancelledError` is handled separately and re-raised,
and every terminal state is persisted. Returning a structured task outcome instead of `None` is
also reasonable if the caller needs results in memory.

If the wrapper converts ordinary exceptions into failed entries, `asyncio.gather()` no longer
needs `return_exceptions=True`. If exceptions are allowed to escape, keep `return_exceptions=True`
and reconcile each gathered item with its corresponding run entry. Do one or the other; handling
the same exception in both places makes ownership unclear.

For a multi-hour evaluation, write the initial manifest with overall status `running`, then persist
it after each run transition. Protect concurrent manifest writes with an `asyncio.Lock`. Prefer an
atomic write (temporary file followed by rename) so interruption cannot leave a partially written
manifest.

A safe manifest write sequence is:

1. acquire the manifest lock;
2. derive aggregate counts and batch status from the current entries;
3. serialize to a temporary file in the batch directory;
4. atomically replace `batch.json` with that file;
5. release the lock.

The lock prevents in-process task races. Atomic replacement protects readers and process
interruptions. Neither mechanism replaces recovery reconciliation after a machine or process
failure.

### Deriving overall batch status

The batch status should be derived consistently from its run entries:

| Condition | Batch status |
|---|---|
| At least one task is `pending` or `running` while the process is active | `running` |
| Every planned run is `completed` | `completed` |
| At least one run completed and at least one is `failed`, `cancelled`, or `interrupted` | `completed_with_errors` |
| No run completed and all terminal runs failed or were cancelled | `failed` |
| An inactive batch contains stale `running` entries | `interrupted` until reconciliation |

`time_ended` and `duration` should only be finalized after all entries reach terminal states.

When inspecting or resuming an inactive batch, reconcile stored states using its artifacts:

- a valid `result.json` means `completed`;
- a recorded exception means `failed`;
- an entry still marked `running` means `interrupted`;
- an entry never started remains `pending`.

The presence of `log.txt` or `trace.json` alone is not a completion signal because either may be
left behind by a failed run.

Resuming a batch is a separate policy decision. A conservative first implementation should report
and preserve `pending`/`interrupted` entries without automatically rerunning them. A later explicit
resume command can schedule only those entries, preserving the original `batch_id` while assigning
new `run_id` values when execution actually starts.

## Batch result layout

Use the batch as the top-level storage unit and dataset question IDs as directory identifiers:

```text
benchmarks/results/batch_eval/
└── <batch_id>/
    ├── batch.json
    ├── questions.json
    └── runs/
        ├── iteration_001/
        │   ├── question_001/
        │   │   └── <run_id-or-timestamp>/
        │   │       ├── result.json
        │   │       ├── trace.json
        │   │       └── log.txt
        │   └── question_002/
        └── iteration_002/
```

`batch.json` is the authoritative experiment manifest. `questions.json` is an immutable snapshot
of only the selected dataset records. The original `Q_dataset.json` should not be modified to mark
batch membership.

Create the batch directory and both JSON files before scheduling any task. The question snapshot
should include the original question fields (`id`, `text`, `grade`, `weather_related`, and
`follow_ups`) without adding mutable execution state to them. Batch membership and run status
belong in `batch.json`.

## Linking batches, questions, and runs

Keep four identifiers distinct:

- `batch_id` identifies the experiment.
- `question_id` identifies the source dataset question.
- `iteration` identifies a repetition of that question within the batch.
- `run_id` identifies one SPARQ execution.

Each batch run entry points to its `question_id`, `iteration`, `run_id`, and `result_path`. Each
individual `SystemOutput` should carry evaluation context containing `batch_id`, `question_id`, and
`iteration`. This two-way linkage lets a standalone run identify its experiment while allowing the
batch manifest to enumerate all planned and completed runs. Directory positions such as `0`, `1`,
and `2` should not serve as question identities because filtering or reordering the dataset changes
their meaning.

### Example manifest fragment

```json
{
  "batch_id": "2026-08-13T14-05-22-CDT_7b3d0c2a",
  "status": "running",
  "dataset_path": "benchmarks/datasets/questions.json",
  "dataset_hash": "sha256:...",
  "selected_question_ids": [1, 2],
  "iterations": 2,
  "max_concurrent_runs": 3,
  "runs": [
    {
      "question_id": 1,
      "question_index": 0,
      "iteration": 1,
      "status": "completed",
      "run_id": "2e5b...",
      "result_path": "runs/iteration_001/question_001/2e5b.../result.json"
    },
    {
      "question_id": 2,
      "question_index": 1,
      "iteration": 1,
      "status": "running",
      "run_id": "d19a...",
      "result_path": "runs/iteration_001/question_002/d19a.../result.json"
    },
    {
      "question_id": 1,
      "question_index": 0,
      "iteration": 2,
      "status": "pending",
      "run_id": "80cf...",
      "result_path": "runs/iteration_002/question_001/80cf.../result.json"
    }
  ]
}
```

Whether iterations are numbered from zero or one is an implementation choice, but use the same
convention in the manifest, directory names, logs, and CLI output. One-based numbering is easier
for people inspecting results manually.

The readable timestamp in `batch_id` uses the machine's local timezone and includes its timezone
abbreviation, such as `CDT` or `CST`. Manifest timestamps use the same local timezone and retain
their UTC offset (for example, `2026-08-13T14:05:22-05:00`), so the directory and manifest agree
without making experiment timing ambiguous across machines or daylight-saving transitions.

## Recommended implementation order

1. Define `BatchRunEntry` and `BatchEvalOutput`, including literal status types and defaults.
2. Create a stable batch directory, selected-question snapshot, and initial manifest containing all
   `pending` entries.
3. Pass each entry into the bounded wrapper and persist lifecycle transitions under a shared lock.
4. Add `batch_id`, `question_id`, and `iteration` evaluation context to individual run output.
5. Derive final counts and batch status after all tasks settle.
6. Add reconciliation tests for completed, failed, cancelled, and interrupted runs before adding
   automatic resume behavior.

## Not building yet

- `sparq_judge_score` / `sparq_judge_review`: no judge/scoring pipeline exists in this codebase.
  Out of scope until that's designed separately.
- Manual token/cost capture inside the graph nodes: superseded by the LangSmith approach above.
