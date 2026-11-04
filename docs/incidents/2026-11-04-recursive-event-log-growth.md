# Recursive event-log growth during a SPARQ run

- Date: 2026-11-04; log times below are America/Chicago.
- Status: run stopped by the user; investigation recorded; no fix implemented as part of this record.
- Run: `04-11-2026_15-18-17`.
- Local evidence directory: `/Users/zayan/tmp/sparq/output/04-11-2026_15-18-17/`.
- Final observed `events.jsonl` size: **24,299,523,616 bytes** (24.30 GB / 22.63 GiB), with a last-modified time displayed as 15:34.
- Scope: SPARQ v1 execution, worker context delivery, tool access boundaries, and event persistence.

## What happened

A worker assigned exploratory analysis searched for a merged dataset from an earlier conceptual step. Its artifact and workspace searches returned no relevant results. It then explored the parent run directory through Python, discovered the activity logs, and read the live `events.jsonl` to look for other workers' results.

The REPL returned its `lines` variable, containing the log contents, in a tool-result namespace artifact. The event writer appended that artifact to the very log that had been read. Later reads included previously embedded log contents, while full state snapshots repeatedly copied the growing tool-message history. Together these produced a logging feedback loop.

The user stopped the run after reporting approximately 24 GB of log growth. This was an interrupted run, not a verified completed analysis.

## Verified worker sequence

The following sequence was recovered from complete events in a bounded prefix of the log. Event numbers are zero-based JSONL line positions in that prefix, not timestamps or stable graph identifiers.

| Event | Observed action or result |
| --- | --- |
| 28, 37–38 | `list_artifacts()` returned `[]`; `list_directory(".")` returned only `index.json`. |
| 69, 72–73 | Searches for `merged` and `missouri` returned no artifact candidates. |
| 124, 127–128 | `list_directory("..")` was denied access outside the working directory; a local CSV search found no files. |
| 182, 185 | Reading `index.json` showed empty artifact categories. |
| 232, 235 | Python `os.walk()` over the parent run directory succeeded and listed `trace.json`, `log.txt`, `events.jsonl`, and `executor/index.json`. |
| 267, 270–271 | Python read `log.txt` and read all current lines of `events.jsonl`. The latter call printed a line count of 270. |
| 305, 316 | A further Python call reread the event log and searched for strings associated with step 4 and merged results. |

The first log-reading code included:

```python
import json
lines = open('/Users/zayan/tmp/sparq/output/04-11-2026_15-18-17/events.jsonl').readlines()
print(len(lines))
for l in lines[-20:]:
    print(l[:300])
```

A subsequent call contained this comment:

```python
# search for step 4 outputs (merged dataset) that are likely saved as a file
```

The messages identify the worker as **Agent 5**, assigned exploratory analysis. Its stream namespace ends in `"4"`; that stream label must not be treated as the plan step ID. In a subsequent search it matched that same label while attempting to find step 4 outputs, illustrating the ambiguity of using raw stream labels to discover step results.

## How it found and accessed the logs

The [worker prompt](../../src/sparq/architectures/v1/prompts/executor_message.txt) receives the executor output directory through `{output_dir}`. The observed `os.walk()` call used its parent run directory, then discovered the log filenames from the directory listing. It did not need advance knowledge that `events.jsonl` existed.

Workers have distinct Python namespace paths derived from run and step IDs, but share the executor output directory. The [executor](../../src/sparq/architectures/v1/nodes/executor.py) passes that shared directory to worker tools.

The [file-management tools](../../src/sparq/tools/filesystemtools.py) are configured with a root directory, and the observed parent-directory request was denied. The [Python executor](../../src/sparq/tools/python_repl/executor.py) sets its current directory using `os.chdir(workspace)` and executes the submitted Python code in a subprocess. That working-directory setting does not enforce the same filesystem boundary: the observed absolute-path traversal and reads succeeded.

The trace establishes that Python was used after the file tool denied access. It does not establish the agent's internal intent or justify attributing malicious intent.

## Why the log expanded

Two implementation behaviors combined:

1. `_namespace_summary()` in the [Python executor](../../src/sparq/tools/python_repl/executor.py) retains any value accepted by `json.dumps()`. Lists and dictionaries can therefore be returned in full, without a size bound. The `lines` list was included under the tool artifact's `namespace` field even though the visible printed output was shortened.
2. `_log_graph_event()` in [system.py](../../src/sparq/architectures/v1/system.py) serializes each event payload without trimming artifacts. The stream subscribes to both `values` and `updates`, with `subgraphs=True`. Full worker-state snapshots repeat prior tool messages and their artifacts.

The cycle was:

```text
read events.jsonl into lines
  -> return lines in a tool artifact
  -> append the artifact to events.jsonl
  -> repeat it in full message-history snapshots
  -> reread the enlarged log, including embedded earlier log contents
```

Direct inspection of the first 32 MiB found embedded event-log strings under `artifact.namespace.lines`. One complete early event occupied 5,908,729 bytes. This is direct evidence of self-embedding; no full-file parse was required to establish it.

## Context handoff: an important qualification

SPARQ's executor is designed to schedule steps according to declared dependencies and provide dependency results and merged Python namespaces to subsequent workers. However, the saved `trace.json` for this run contains **eight plan steps, all with `dependencies: []`**, despite descriptions that refer to earlier steps. The log records all eight workers launching in parallel at 15:18:51.

Consequently, this incident does **not** establish that Agent 5 ignored an available dependency handoff. Its initial recorded task message contains the exploratory-analysis assignment without preceding dependency context. It was looking for a merged dataset while the plan supplied no declared dependency ensuring that dataset was ready and passed along.

The dependency-generation or propagation failure requires investigation separately from the log-growth defect. The saved plan verifies missing dependency declarations, but does not by itself identify where they were lost or omitted.

## Model context and limits of the evidence

- The user reports that this behavior had not occurred with the less capable models used until that day, described as below Sonnet 4.5.
- Sampled worker response metadata identifies `us.anthropic.claude-sonnet-5`.
- This is an observed incident after a model change, not a controlled comparison proving that model capability caused the behavior. Other working-tree or configuration differences have not been isolated.
- Source inspection reflects the working tree during this investigation. Several implementation and configuration files already had uncommitted changes; this record does not attribute the incident to a particular commit.
- Log inspection was bounded: small tail samples and complete events from prefixes up to 32 MiB. The entire roughly 24 GB file was not parsed. No complete tally of duplicated bytes or all subsequent worker behavior is claimed.
- Worker completion messages alone do not validate scientific correctness. This record does not audit the generated analyses or plots.

## Proposed follow-up work

These are investigation and remediation candidates, not implemented fixes or adopted architecture decisions:

- Trace why the saved plan omitted dependencies for conceptually dependent steps; validate scheduling and context delivery with a representative dependent workflow.
- Bound namespace summaries by returning types, lengths, and short previews instead of arbitrarily large JSON-compatible values, including nested containers.
- Limit duplication of tool artifacts and full histories in event persistence while retaining useful provenance.
- Define and consistently enforce the intended filesystem access policy across Python and file tools, including access to internal run logs and required input datasets. Separate working directories alone do not enforce isolation.
- Verify that a log-reading attempt cannot recreate recursive growth after remediation, and that ordinary artifact handoff still works.

## Evidence retained

The local run's `events.jsonl`, `log.txt`, and `trace.json` are the primary evidence. They have not been copied into this repository, truncated, or deleted as part of this documentation task. The incident discussion in the associated chat supplies the user's stop report and prior-model observation. No stable conversation ID was available when this record was written.
