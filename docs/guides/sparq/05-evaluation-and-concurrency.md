# 5. Batch evaluation and concurrent runs

[Chapter 4](04-reports-and-results.md) completed one question. An evaluation
repeats that same pipeline for selected benchmark questions and records which
run belongs to which question and iteration. No new analytical stage is added
inside the single-run graph.

## Batch evaluation

Run a batch from a repository checkout:

```bash
uv run python -m sparq.evaluation.batch -n 10 -k 1 --evaluation-id baseline-v1
```

The runner selects up to the first `-n` eligible questions (excluding weather
questions and grade `-1`) and runs each for `-k` iterations. Each question run
uses the pipeline from the preceding chapters, with its own UUID, settings, output directory, and
evaluation context. `batch.json` links the planned runs to their question IDs,
iterations, result paths, and execution statuses; `questions.json` preserves the
selected inputs. Separate invocations have separate batch IDs. Reusing an
`evaluation_id` groups them logically without changing which questions are selected.

Generated evaluations belong under `evaluation/results/`. Each invocation owns
one bundle; keep its manifests and run evidence together when copying or archiving.

```text
evaluation/
  datasets/questions.json          # versioned evaluation inputs
  results/                        # generated, ignored by Git
    batch_eval/<id>/
      batch.json                  # configuration, status, question/run linkage
      questions.json              # selected question snapshot
      runs/iteration_001/question_001/<run-id>/
        result.json
        trace.json
        final_answer.json
        log.txt
        executor/                 # index and generated analysis artifacts
        report/                   # Markdown, PDF, optional bibliography
  references/                     # deliberately preserved, versioned evidence
```

Default batch bundle IDs use `YYYY-MM-DDTHH-MM-SSZ_<random suffix>` in UTC.
The default root resolves against the repository checkout, independently of the
working directory. Batch evaluation requires a checkout for its question dataset.
Bedrock model auditing is a [utility with separate cache/artifact locations](../utils/bedrock_utils/README.md).

### Choose a destination

```bash
uv run python -m sparq.evaluation.batch -n 10 -k 1 \
  --evaluation-id baseline-v1 --output evaluation/results/batch_eval/baseline-v1
```

`--output` names the exact bundle directory, not a parent for another generated
timestamp folder. Relative overrides resolve against the working directory and
`~` expands to the home directory. Batch output must not already exist. Bedrock
utility output options are described in its [guide](../utils/bedrock_utils/README.md).

`evaluation_id` groups separate batches logically; it is metadata and does not
alter their directory paths. Batch IDs and run IDs remain independent of an
explicit output directory's name.

### Move and preserve results

`batch.json` run-result paths are relative to the batch bundle root. Resolve
them against the containing bundle, not the shell's working directory. Copy the
entire bundle to preserve these links. Bedrock utility artifacts and their
portability constraints are covered in the [utility guide](../utils/bedrock_utils/README.md).

Historical results from `eval/results/` and `benchmarks/results/` now live under
`evaluation/results/`. Their contents, old timestamp formats, and recorded paths
are preserved. See the [migration record](../../../evaluation/MIGRATION.md) for path
mappings and integrity verification. Inspect provenance before promoting a selected bundle
into `evaluation/references/`; successful execution does not establish scientific
validity.

### Code ownership

- `src/sparq/evaluation/paths.py`: shared generated-results root and bundle naming.
- `src/sparq/evaluation/batch.py`: batch directories, manifest and question linkage.
- `utils/bedrock_utils/`: model discovery and auditing utilities, stored separately
  from evaluation outputs. Historical Bedrock bundles retain their original paths.

Per-run saving follows the [report and result lifecycle](04-reports-and-results.md).

## Follow the batch scheduler

The batch runner first persists its selected questions and a manifest with
planned run entries. Within one iteration it schedules question runs behind a
shared semaphore, currently limited to three concurrent runs. It waits for all
runs in that iteration before starting the next iteration.

Each scheduled run receives a deep copy of baseline settings, its own
`Agentic_system`, UUID, and explicit output directory. Evaluation context adds
the batch ID, optional evaluation ID, question ID, and iteration to its result.
Individual exceptions are recorded on their manifest entries so the remaining
questions can continue. On cancellation or orchestration failure, outstanding
tasks are cancelled and awaited, unfinished entries receive cancellation or
interruption states, and the final manifest is persisted with timing and status.

A run entry marked `completed` means its call returned without an exception. It
does not independently certify every analytical step, numerical claim, or
scientific conclusion. For example, the executor can retain a terminal failed
step and still proceed to synthesis.

## Distinguish the levels of concurrency

We have now encountered several different kinds of overlap. They share resources
in different ways:

| Level | What is isolated | What still needs care |
| --- | --- | --- |
| Independent steps in one run | Per-step namespaces | Shared executor filenames, round barriers, synchronous operations |
| Python calls using one persistent namespace | A fresh child process per call | Concurrent writes to the same pickle are not locked |
| Question runs within a batch | System instance, copied settings, UUID, explicit run directory | Shared environment/provider capacity; installation lock is only process-local |
| Arbitrary calls on the same system instance | No general instance-level isolation guarantee | Mutable node bindings, graph, and reused output directory |

## Why reusing a system instance needs care

`run()` rebuilds node partials and assigns `self.graph`. The output directory is
already held in settings; it is not regenerated on every call. Two sequential
calls on one instance therefore reuse its output location unless the caller
changes it. Different UUIDs isolate their normal namespace keys but do not
prevent saved files from being overwritten or mixed.

Overlapping calls on that same instance also share mutable graph/node bindings.
The compiled graph used by an already-started stream should not be described as
automatically changing when `self.graph` is rebound; the underlying problem is
that the instance and paths are not designed as independently owned per-call
resources. Concurrent reuse is not a supported isolation pattern.

Separate default instances can still choose the same output directory when
settings are constructed in the same second: default path names have
one-second timestamp resolution. Supplying distinct run IDs alone does not fix
that path collision. The batch runner avoids this by assigning explicit
iteration/question/UUID directories and copied settings per run. Callers adding
another concurrent entry point need the same separation of instance, settings,
run ID, and output directory.

## Implementation and further reading

- [Batch scheduling, manifest persistence, and cancellation](../../../src/sparq/evaluation/batch.py)
- [Manifest contracts](../../../src/sparq/schemas/evaluation.py)
- [Run orchestration and mutable instance state](../../../src/sparq/architectures/v1/system.py)
- [Default output-path construction](../../../src/sparq/settings.py)

Return to the [guide contents](README.md), or follow [Bedrock setup](bedrock.md)
for provider configuration. The external [Bedrock auditing utility](../utils/bedrock_utils/README.md)
has its own concurrency and output rules.
