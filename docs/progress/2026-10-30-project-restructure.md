# Project restructuring verification — 2026-10-30

Branch: `refactor/project-structure`. Baseline: `9541802`.
Implementation commits: `8218fd9` (assets/documentation/launchers) and `18b3f5e`
(package modules and tests). See the [structure record](../proposals/project_structure.md)
for the complete layout, move checklist, and discoveries.

## Results

| Check | Baseline | Restructured checkout |
| --- | --- | --- |
| Full unittest discovery | 108 tests; 5 failures; 1 skip | 109 tests; same 5 failures; 1 skip |
| Focused evaluation/report/node/artifact/vision/settings/utility tests | — | 60 tests pass |
| BasedPyright | 82 errors | Same 82 diagnostics after mapping moved paths; no additions/removals |
| Ruff lint, maintained source/tests plus original batch runner | 94 findings | 91 findings; no new findings after normalizing relocated imports |
| Ruff formatting check over the equivalent file groups | 40 files need formatting | 38 files need formatting; no newly unformatted files |
| Newly split schema/artifact/bundle modules | — | All 5 files pass formatting checks |
| Output contract JSON schemas | 8 models | All 8 match exactly |
| Preserved reference bundle | 24 regular files and external run symlink | All file hashes and the symlink target match |
| Source distribution and wheel build | — | Pass; wheel includes runner, active prompts, configuration, and bundled data |
| Shell syntax, Python compilation, maintained Markdown links | — | Pass |
| Original tracked-file inventory | Captured | Every original file accounted for by retention, relocation, or explicit module split |

Ruff was run as a temporary tool (0.17.3); it was not added as a project dependency.
The build used the installed uv backend and warned that uv 0.11.23 is outside the
existing `uv_build` requirement range. Both artifacts were produced successfully;
this is not a claim that the declared older backend was independently tested.

## Test scenarios

The added repository-layout regression test loads the real relocated question
set, selects an eligible non-weather question, creates its batch, verifies the
benchmark input/output locations, and round-trips the pending manifest through a
temporary file. Existing batch tests exercise mocked worker execution, concurrent
scheduling, failure/cancellation handling, manifest persistence, and exit statuses.

Existing report tests generate synthetic PDFs and inspect their content and images.
Node and vision checks use controlled messages/models; no paid model runs were used.

The full-suite failures match the baseline by test identity:

- `TestExecutor.test_matplotlib_uses_non_interactive_backend`
- `TestExecutor.test_saving_plots`
- `TestExecutor.test_show_after_savefig_does_not_fail`
- `TestPackageInstallation.test_install_package`
- `TestPackageInstallation.test_uninstall_package`

These checks depend on local cache/package-installation conditions. An exploratory
run with writable caches allowed the existing package tests to uninstall NumPy and
pandas, then failed to reinstall them without network access. Both pinned versions
were restored (NumPy 2.2.6, pandas 2.2.3). The final full-suite comparison uses the
original cache conditions, with tokenizer data cached separately. No production
package-management or REPL behavior was changed to mask baseline failures.

## Commands and limits

```bash
uv run python -m sparq.evaluation.batch --help
uv run python -m unittest discover -v
uv run python -m unittest tests.evaluation.test_batch
uv run basedpyright --outputjson
ruff check src tests
ruff format --check src tests
bash -n scripts/run.sh scripts/run_tests.sh scripts/run_docker.sh
uv build --out-dir /tmp/sparq-restructure/dist --offline
```

Commands were executed using the existing `.venv` directly where appropriate and
a writable temporary uv cache. Detailed run logs and comparison inventories are
in `/tmp/sparq-restructure/` for this session; they are not repository artifacts.
The batch help command was also checked from outside the repository working
directory. Benchmark assets intentionally remain checkout-only, outside the wheel.

Docker syntax/launcher paths and its corrected Python entrypoint were inspected;
a Docker image was not built or launched. Sandbox and historical manuscript scripts
were not executed. Local draft/script ignore rules and staged paths were checked;
no secrets, generated results, or local manuscript revisions were committed.

## Project continuity

Owner: `agentic-epidemiology`; source: Codex conversation
`01a18e50-6601-7f73-bec2-76d35c966b14`, explicit structure discussion and implementation
request. The implementation and this receipt are filed in the registered SPARQ
repository. External project-log delivery remains pending until recorded in the
project's Research Log; this local receipt does not claim external delivery.
