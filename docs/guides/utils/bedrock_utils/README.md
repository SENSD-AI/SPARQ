# Bedrock model auditing

For AWS credentials and provider configuration, see [Bedrock setup](../../sparq/bedrock.md).

The default mode uses small capability probes rather than a complete SPARQ run:

```bash
uv run python -m utils.bedrock_utils
```

This performs paid inference. It uses SPARQ's configured `AWS_PROFILE` and
`AWS_REGION`, falling back to the AWS default profile and its configured region.
It does not edit project configuration. Four models run in parallel by default.
Run the module command from the repository root. The existing
`scripts/aws/check_models.py` remains a wrapper and can also be invoked by absolute
path from another working directory. This utility is not installed with the SPARQ library.

```bash
uv run python -m utils.bedrock_utils --workers 4 \
  --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

`--model-id` is repeatable and accepts an exact foundation/profile ID or an exact
model/profile name. Selecting a foundation ID that requires profiles expands to
all its discovered routes; an explicit profile ID selects only that route. `--profile`, `--region`, and `--output` override their
defaults. Use a fresh output directory; existing reports are not overwritten.
`--list-only` discovers targets without inference.

## Discovery and capability hints

Discovery lists the regional foundation-model catalog and every inference-profile
page. Targets need verified TEXT input/output and a pay-as-you-go invocation route.
Direct foundation IDs need ON_DEMAND; profiles also qualify when their bases list
INFERENCE_PROFILE. Provisioned-only and non-text targets are skipped. Missing
required modality/throughput metadata is recorded as a skip; incomplete catalog
requests are recorded as discovery errors. Image support is optional.

The catalog API does not provide tool-use or structured-output flags. During a
probe, the installed `langchain-aws` model's `profile` supplies any available
`tool_calling`, `structured_output`, and `image_inputs` hints. Their package version
and source are saved separately from observed results. These are SDK-maintained
hints, not live AWS guarantees or freshly scraped model-card claims. Missing
entries stay null (unknown); neither missing nor negative hints exclude a target
from live testing. List-only results leave these hints unknown.

## Saved inference-profile mapping

Discovery also writes `inference_profiles.json`, including in `--list-only` mode.
It uses the paginated Bedrock API equivalent of:

```bash
aws bedrock list-inference-profiles --region us-east-1 --profile my-profile
```

The snapshot records the AWS profile, region, discovery time, completeness and
errors, with three indexes:

- `by_name`: foundation-model and inference-profile names to lists of profile IDs.
- `by_model_id`: foundation IDs to lists of profile IDs.
- `profiles`: profile ID to its name, ARN, invocation ID, type, status and underlying
  foundation IDs. Application profiles retain their short ID but use the ARN as
  the invocation ID, matching the existing probe behavior.

The mapping is one-to-many: regional, global and application profiles can coexist.
An explicit profile selection is preserved; the script does not guess a preferred
geography. A selected foundation ID/name requiring a profile expands to its listed
routes, deduplicated with any other selected IDs. Direct-only models still use
their direct IDs. Routing skips now list the concrete alternative profile IDs.

The full regional mapping is saved even when `--model-id` narrows the probe targets.
Every audit keeps a snapshot. The latest complete discovery also refreshes the
OS-native user cache, scoped by AWS profile and region. Incomplete discovery leaves
the previous complete cache unchanged; failed cache writes are reported as warnings
without discarding the audit snapshot. Discovery does not prove invocation permission: an
ACTIVE profile can still be denied by account or organization policy.

## Default: capability probes

Each candidate is instantiated through SPARQ's existing `get_llm` factory using
`ChatBedrockConverse`. Four checks run independently:

| Check | What must work |
| --- | --- |
| structured_router | `.with_structured_output(Router)` returns a valid analytical route |
| structured_plan | `.with_structured_output(Plan)` returns a two-step nested plan with the requested dependency |
| tool_round_trip | The model requests a local lookup, receives its result, and uses the returned value |
| tools_with_structure | `create_agent(tools=[lookup], response_format=StepResult)` executes the lookup and returns its value in a valid successful StepResult |

The lookup generates a new opaque value only when executed. The prompt does not
contain the answer. A guessed value, plain answer without a tool call, missing
required fields, or an incorrect result cannot pass. The agent loop is bounded to
eight graph steps. Native JSON-schema output is not independently required: the
same structured-output strategy selected by SPARQ's current interfaces is tested.
No datasets, Python REPLs, image analysis or report generation run in this mode.

Each check has a 120-second deadline (`--check-timeout`). The subprocess also has
a 600-second overall deadline (`--timeout`). Timeouts are inconclusive, not proof
of incompatibility. Increase the deadlines for slower models. AWS documents that
first-time native structured-output schema compilation can take about 90 seconds.

Results distinguish `passed`, `failed` (an observed response did not satisfy the
probe), `unsupported` (an explicit SDK/API limitation), `access_denied`,
`timed_out`, and `error` (other exceptions). Permission denial skips remaining
checks. Other failures do not prevent the remaining capabilities from being
checked. The overall status follows the first unsuccessful check; inspect all
per-check results for partial capabilities. A pass means all four probes passed,
not that every SPARQ workload is reliable. Transient failures still warrant review.

## Optional: full SPARQ integration

```bash
uv run python -m utils.bedrock_utils --mode pipeline --timeout 1200
```

This retains the expanded data-readiness query: two independent cached CSV/Excel
inspection steps feed dependent analysis, a summary table, chart, filesystem
operations, artifact lookup, visual inspection, and a referenced final report.
The candidate is used for router/planner/executor/aggregator; the configured
vision model stays separate. Descriptions, data samples and tool results may be
sent to the configured providers. Cached datasets and the configured vision model
must be available.

The integration checker requires successful calls to all 14 exposed tools,
persistent Python, dependency structure, bounded summary rows, indexed artifacts,
a copied methods document, a saved visual assessment, an embedded PNG, and
JSON/Markdown/PDF/bibliography outputs. It does not independently certify every
numerical claim, every inherited variable, or conditional recovery paths such as
package installation and retries. Historical pipeline results remain unchanged.

## Output and interruption

The reusable mapping is generated cache data; audit bundles are project evidence.
On macOS the default locations are:

```text
~/Library/Caches/sparq/bedrock/<profile>/<region>/inference_profiles.json
utils/bedrock_utils/outputs/<profile>/<region>/
  latest_audit.json                  # relative path to the newest bundle
  audits/<UTC timestamp>_<random suffix>/
```

`platformdirs` selects the appropriate user-cache directory on each OS. `--cache-dir`
overrides the Bedrock cache root; profile/region are appended. Audit paths resolve
beside the utility code, independent of the current working directory. `--output`
still selects an exact bundle directory. A custom output does not update the
project's latest-audit pointer, but it still refreshes the scoped cache. Each bundle
records the cache location and any cache warning. Cache refreshes and report writes
use unique temporary files and atomic replacement.

New Bedrock audits are utility artifacts, separate from scientific evaluation
results. Previous audit bundles were deleted at the user’s request before this move.
The implementation and tests live in `utils/bedrock_utils/` and
`tests/utils/bedrock_utils/`.

- `inference_profiles.json`: the full name/model-ID to profile-ID mapping for the
  selected AWS profile and region.
- `summary.md`: passing models and a table of statuses/reasons; capability mode
  includes a column for each check.
- `results.csv`: target, status, error code, reason, duration, relative artifact
  directory, per-check results, capability hints and their source.
- `report.json`: complete catalog, selected mode, query/description, deadlines,
  discovery errors and results.
- `runs/<model-or-profile-id>/job.json`, `check.json`, `worker.log`: isolated worker inputs,
  checkpoints and diagnostics.
- Capability mode: `runs/<model-or-profile-id>/capability_evidence.json` records the actual
  structured responses, tool requests, executed values and final answers.
- Pipeline mode: `runs/<model-or-profile-id>/run/` holds SPARQ artifacts and
  `tool_calls.json` when the workflow returns or raises.

Artifact-directory paths in `report.json` and `results.csv` are relative to the
audit bundle root. Copy the entire bundle to preserve these links. `job.json`
may retain absolute execution paths; it is historical provenance, not a portable
replay command.

Worker directories use the exact invocation ID when it is already path-safe, for
example `runs/us.openai.gpt-6.1-sol/`. Characters such as colons and slashes are
replaced, and a short hash suffix disambiguates sanitized, case-normalized or
truncated IDs. The full original ID remains in `job.json`. Existing hashed audit
folders are not renamed automatically.

Only the coordinator writes shared reports; each worker has its own process and
folder. Finished capability checks are checkpointed before the next starts. A
hard deadline preserves completed checks and marks the active one timed out;
remaining checks are skipped. On POSIX, timeout or Ctrl-C kills worker process
groups, including REPL descendants. Windows descendant cleanup is unverified.
Ctrl-C preserves finished models, records active workers as interrupted, and
leaves queued models not_tested. Start a fresh sweep to retry selected IDs.

Rich renders progress and final results. Exit 0 means the sweep finished, even if
individual targets failed; 2 means incomplete discovery/invalid arguments; 130
means interrupted. Capability mode has been validated locally with deterministic
models and real LangChain tool/agent execution; no paid capability sweep was run
during implementation.

## Sources

The mapping update used the AWS CLI [list-inference-profiles](https://docs.aws.amazon.com/cli/latest/reference/bedrock/list-inference-profiles.html)
reference, specifically pagination and the output name/ID/ARN/model/status/type
fields. The same paginated SDK response is joined with foundation-model names.


Read for the capability update: LangChain's [model profiles](https://docs.langchain.com/oss/python/langchain/models#model-profiles)
and [agent structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
sections, plus AWS's [foundation-model catalog fields](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_FoundationModelSummary.html),
[tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html), and
[structured output](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html).
The installed model factory, node interfaces and `langchain-aws` profile resolution
were inspected directly. SDK metadata is kept separate from live observations.

Storage follows the [XDG cache/config distinction](https://specifications.freedesktop.org/basedir/latest/)
and [platformdirs user-cache API](https://platformdirs.readthedocs.io/en/latest/api.html#user-cache-directory).
Only the catalogue mapping is a replaceable OS cache; audit evidence remains
project-local and Git-ignored.
