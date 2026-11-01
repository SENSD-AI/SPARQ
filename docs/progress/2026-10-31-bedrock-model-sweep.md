# Bedrock model sweep — 2026-10-31

Implemented the requested script for one basic end-to-end SPARQ run per model in
the configured AWS region. The user explicitly narrowed scope to basic run
success/failure, so there are no separate vision, tool-capability, or CSV-content
probes.

Entry point: `scripts/aws/check_models.py`. See the
[usage guide](../guides/utils/bedrock_utils/README.md) for commands and result formats.

## Behavior

- Discover regional foundation models and all inference-profile pages. Preserve
  non-text models as skipped rows with reasons and record incomplete discovery.
- Use each candidate for router, planner, executor and aggregator; retain the
  existing configured vision role. Run the same small synthetic query through
  the actual `Agentic_system` using copied settings, without rewriting config.
- Check the saved trace for an analytical route, a plan and successful execution
  of every planned step. Require nonempty response and JSON/Markdown/PDF outputs.
- Isolate each target in a subprocess with a per-model deadline. Capture original
  wrapped AWS errors, full worker tracebacks, timeouts and worker crashes.
- Save JSON, CSV and readable model lists after every target, and retain partial
  results on interruption. A failed test is a failed attempt in this environment,
  not proof the model can never work.

## Verification

- 17 focused tests pass: unfiltered/paginated discovery, partial discovery errors,
  non-text exclusions, original error-code retention, continued execution after a
  model fails, interruption/checkpointing, timeouts, worker crashes, default CLI
  execution and list-only mode, trace/output validation, and candidate settings
  injection without mutating baseline settings.
- Full discovery: 130 tests; 127 pass, 2 existing package-manager failures, 1 skip.
  Failures remain `test_install_package` and `test_uninstall_package`.
- New files pass Ruff lint and formatting checks. BasedPyright remains at 71
  baseline errors with no new diagnostics.
- A real child process using an intentionally nonexistent profile returned a
  failed check with `ProfileNotFound`, retaining the SPARQ wrapper and original
  cause. This required no paid inference or external model call.
- CLI help and source/wheel builds pass. The build retains the existing uv/backend
  version-range warning. Installed botocore schemas confirm the catalog/profile
  methods and pagination fields used by the implementation.
- No paid model sweep was started. The pre-existing `config_v1.toml` edit was
  preserved, with its content hash unchanged.

Implementation sources: AWS's ListFoundationModels, ListInferenceProfiles and
Converse documentation (linked in the usage guide), plus current local SPARQ
settings, graph, model factory, executor and saver code. No hard-coded model list
or documentation-based compatibility ranking is used.

The implementation and verification are filed in the registered SPARQ repository
for project `agentic-epidemiology`. External project-log filing remains pending;
this record does not claim delivery to the Obsidian vault.


## Feature-coverage update (2026-10-31)

User instruction: "maybe the query doesnt warrant. make sure the run uses ALL
sparq features". The default query now requests a bounded real-data readiness
study on cached CSV and Excel inputs, with two independent initial steps feeding
later dependent analysis. It requests all 14 exposed tools, persistent Python,
a summary table, chart, filesystem operations, artifact lookup, visual inspection,
and a referenced report. The configured dedicated vision model remains separate
from each text candidate. The default per-model timeout is now 600 seconds.

The worker collects nested tool inputs/outputs/errors locally in `run/tool_calls.json`
and rejects missing successful tool coverage, missing input formats or persistence,
missing parallel-ready/dependent plan structure, failed steps, incomplete indexed
artifacts, invalid summary bounds, a mismatched methods copy, missing visual
assessment, unembedded PNG, or absent report/bibliography outputs. This checks
normal feature paths; retries, package installation, scientific correctness and
actual use of every inherited variable are not independently certified.

Validation: 44 focused sweep/probe/coverage tests pass, including actual local
nested-tool tracing and failed REPL-artifact handling. Scoped BasedPyright, Ruff
lint/format checks and CLI help pass. No paid inference was started for this update;
the expanded task needs a fresh live run. Historical sweep results remain unchanged.
The source inspected for tracing was the installed LangChain implementation;
online API reference requests were unavailable. External project-log filing
remains pending under the existing repository receipt above.


## Small capability probes become the default (2026-10-31)

Accepted user direction: use catalogue/capability information and small tool-use
and structured-output tests instead of requiring a full SPARQ workflow to finish.
The CLI now defaults to `--mode capabilities`; `--mode pipeline` preserves the
expanded integration test. Historical sweep artifacts are unchanged.

The capability worker uses SPARQ's model factory, Router/Plan schemas and
`create_agent(..., response_format=StepResult)` interface. Four independent checks
cover structured routing, a nested dependency plan, a local tool round trip and
tool use combined with structured output. Lookup values are generated only when
the tool executes; the final answer must incorporate the returned value. The
agent is bounded to eight graph steps. No datasets, REPLs, images or reports are
needed. Each check has a 120-second deadline within a 600-second worker deadline.

Installed SDK profile hints are saved with package/source provenance and remain
separate from live observations; missing and negative hints do not exclude
models. This uses documented LangChain metadata, not a scraped or hard-coded AWS
model-card compatibility table. Unknown catalogue tool/schema flags remain unknown.
Reports now expose each check and distinguish unsupported interfaces, access
restrictions, inconclusive timeouts, errors and incorrect responses. Hard timeouts
preserve completed checks. Permission denial skips unattempted checks.

Validation: 52 focused tests pass, including actual local LangChain agent/tool
execution, offline ChatBedrockConverse serialization/parsing with mocked AWS
responses, incomplete-capability results, timeout checkpoints, CLI modes and
existing sweep/pipeline regressions. Scoped BasedPyright has zero diagnostics;
Ruff lint/format checks and CLI help pass. No paid capability sweep was started.
External project-log filing remains pending under the existing repository receipt.


## Saved inference-profile lookup (2026-10-31)

User direction: persist model names to inference-profile IDs using the regional
Bedrock list-inference-profiles response. Discovery now retains each profile's
original ID, ARN, type and status, joins it to foundation IDs/names, and writes
`inference_profiles.json` beside the sweep reports. The mapping is one-to-many,
scoped to the configured AWS profile/region, timestamped, and marked incomplete
when discovery errors occur. It is retained in full even for a targeted sweep.

`--model-id` now accepts exact model/profile names and profile IDs as well as
invocation IDs. Profile-only foundation selections expand to all discovered
routes; explicit profile selections are preserved and deduplicated. Routing-skip
messages now list concrete alternative IDs rather than a generic instruction.
No global/geographic route is silently preferred and discovery is not treated as
proof of invocation permission.

A live metadata-only discovery with AWS profile sensd in us-east-1 produced 99
profiles, 159 name aliases and 67 foundation-ID entries, with no discovery errors.
The Sonnet 4.5 foundation selection resolved to its US and global routes. Snapshot:
`evaluation/results/bedrock_models/2026-11-01T04-52-07Z_9ff830a6/inference_profiles.json`.
No model inference was invoked. All 56 focused tests pass; scoped BasedPyright
and Ruff lint/format checks pass. Historical reports remain unchanged. External
project-log filing remains pending under the existing repository receipt.


## Utility ownership and OS-cache storage (2026-10-31)

User requested utility ownership and more convenient mapping storage, then asked
for best-practice guidance. Bedrock auditing moved from `sparq.evaluation` into
`sparq.utils.bedrock`; tests moved to `tests/utils/bedrock`. The existing script
wrapper remains supported, and `python -m sparq.utils.bedrock` is the module CLI.
The utility no longer imports evaluation path helpers.

The chosen split stores generated reusable mappings in the OS-native cache and
new audit evidence in Git-ignored `artifacts/bedrock/<profile>/<region>/audits/`.
A stable `latest_audit.json` points to the newest default bundle. On this Mac the
mapping is `~/Library/Caches/sparq/bedrock/sensd/us-east-1/inference_profiles.json`.
`--cache-dir` and `--output` provide independent overrides. Incomplete discovery
preserves the prior complete cache; cache-write errors warn rather than discard
per-audit evidence. Atomic writes use unique temporary files for concurrent runs.
`platformdirs` was promoted from a transitive dependency to a direct dependency;
uv resolved the existing lockfile offline without upgrading packages.

A fresh metadata-only discovery verified the stable cache matches its audit
snapshot (99 profiles, no discovery/cache errors). Bundle:
`artifacts/bedrock/sensd/us-east-1/audits/2026-11-01T04-58-12Z_c1fc3835/`.
No model inference ran. All 62 utility tests pass; scoped BasedPyright and Ruff
checks pass. Historical results remain unchanged. The decision and sources are
recorded in ADR 0002. External project-log filing remains pending.


## Readable per-target directories (2026-11-01)

User requested that worker directory names identify the model being checked.
New audits use `runs/<model-or-profile-id>/`, preserving exact path-safe IDs such
as `us.openai.gpt-6.1-sol`. Sanitized/case-normalized/truncated IDs get a readable
prefix plus a short hash suffix to avoid merging distinct targets. Historical
hashed folders and their report paths are unchanged. The original invocation ID
continues to be saved in job.json. All 63 utility tests pass, including distinct
unsafe-ID names and relocated bundle links; scoped type/lint/format checks pass.


## Repository-level utility and local outputs (2026-11-01)

User selected `utils/bedrock_utils/` for the auditor and its `outputs/` subdirectory
for generated evidence. This supersedes the library and artifact locations above.
The module command is now `uv run python -m utils.bedrock_utils`; the script wrapper
remains supported, including from another cwd. Workers receive the checkout root
on PYTHONPATH, preserving the caller cwd. Tests moved to `tests/utils/bedrock_utils/`.
The OS-native inference-profile cache is unchanged. Previous seven audit bundles
and their latest pointer were deleted at the user's explicit request before this
move; no new live inference was launched.

All 64 focused tests pass, including entry-point execution outside the checkout,
worker invocation paths, and the new default output location. Scoped BasedPyright,
Ruff lint and formatting pass. External project-log filing remains pending; this
repository record preserves the accepted decision and verification.
