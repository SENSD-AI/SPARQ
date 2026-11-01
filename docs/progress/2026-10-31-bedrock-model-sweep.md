# Bedrock model sweep — 2026-10-31

Implemented the requested script for one basic end-to-end SPARQ run per model in
the configured AWS region. The user explicitly narrowed scope to basic run
success/failure, so there are no separate vision, tool-capability, or CSV-content
probes.

Entry point: `scripts/aws/check_models.py`. See the
[usage guide](../guides/aws/check_models.md) for commands and result formats.

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
