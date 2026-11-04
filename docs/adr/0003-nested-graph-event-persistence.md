# Nested graph event persistence

- Date: 2026-11-03
- Status: Adopted for v1 streaming; live end-to-end verification remains incomplete
- Authority: User selected direct event-file persistence, authorized the streaming
  workaround, and requested this ADR after reviewing the smoke-test blocker.

## Context

The outer SPARQ graph uses the Pydantic `State` schema, which requires `query`.
Worker agents and the agent inside the aggregator have their own message state.
Inspecting only outer state snapshots omits their internal model and tool activity.

In the checked environment, LangGraph 1.2.8 and LangChain 1.3.11 are installed.
The first live smoke test with `version="v2"`, `subgraphs=True`, and both `values`
and `updates` failed when a nested agent emitted a values snapshot containing
messages but no `query`. Inspection of the installed LangGraph
`langgraph/pregel/main.py::_output` showed its v2 values branch applying the outer
output mapper to that nested payload. Validation failed before SPARQ's event
consumer could filter the namespace.

## Decision

Use `graph.astream(..., version="v1", stream_mode=["values", "updates"],
subgraphs=True)` in the v1 runtime. Normalize each returned
`(namespace, mode, data)` tuple into a record with `type`, `ns`, and `data` fields.
This is a compatibility workaround for the inspected version, not a claim that
all v2 streaming is unsuitable.

Append every normalized event directly to `run_dir/events.jsonl`, using
`to_jsonable_python(..., by_alias=False)` before JSON encoding. Each line is one
event; line order is the order received by the single stream consumer. Keep graph
event persistence independent of Loguru's application-message sinks.

Only `values` events with an empty namespace are validated as SPARQ `State`.
Use those snapshots to overwrite `run_dir/trace.json` and retain the final outer
state for response extraction. Nested snapshots remain in the event file and
must not replace the outer state. `self.run_dir` supplies the shared run directory.

Implementation: [v1 system](../../src/sparq/architectures/v1/system.py).
Regression coverage: [message-state tests](../../tests/architectures/v1/nodes/test_message_state.py).

## Verification and limits

The focused message-state, saver, and report suite passed 20 tests. This includes
a real nested graph with a different state schema, without model API calls, and
a synthetic event sequence checking that nested snapshots cannot overwrite the
outer trace or final response.

The live smoke command was:

```bash
PYTHONBREAKPOINT=0 uv --cache-dir /tmp/sparq-uv-cache run --no-sync python scripts/smoke_test.py
```

Evidence is retained under
`/Users/zayan/tmp/sparq/output/03-11-2026_17-36-19/`, especially `events.jsonl` and
`log.txt`. The v1 run passed the earlier streaming failure, saved model/tool
updates from three initial worker invocations, and completed workers 2 and 3.
The retained file contains 662 events, including a subsequent worker retry.

Worker 1 repeatedly called `write_file` without its required `text` argument.
Affected model responses recorded `stopReason="max_tokens"` and 4,096 output
tokens, consistent with truncated tool arguments. It then reached the 200-step
recursion limit and the executor began retrying it. The run was interrupted
(exit 130) to stop the repeated requests. No final `result.json` or report was
produced; live aggregator execution and end-to-end completion were not verified.
The worker/model factory currently does not forward the configured `max_tokens`;
resolving that separate issue is outside this streaming decision.

## Consequences

Outer and nested events remain available after a failure or interruption, while
`trace.json` remains a latest-outer-state snapshot. The event file records emitted
graph data; it is not a checkpoint or a guarantee of replayability or factual
correctness of worker claims.

Full state snapshots repeat accumulated content and can make the event file
large: this interrupted smoke run produced 25,328,829 bytes. Namespaces identify
runtime invocations, including retries, and are not stable plan-step identifiers.
Reconsider the workaround only after checking a candidate LangGraph version with
the different-schema regression and a complete live smoke run.

The official [stream output format documentation](https://docs.langchain.com/oss/python/langgraph/streaming#stream-output-format-v2)
describes the v1/v2 representation difference, and
[subgraph output documentation](https://docs.langchain.com/oss/python/langgraph/streaming#subgraph-outputs)
describes nested streaming. The specific mapper failure above comes from local
source inspection and runtime evidence, rather than a documented API guarantee.
