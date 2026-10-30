# Project structure

Status: implemented on `refactor/project-structure` following the 2026-10-30
restructuring discussion. This record captures organization and migration
exceptions; it does not adopt the other feature proposals in this directory.

## Layout

```text
SPARQ/
├── README.md, AGENTS.md, CLAUDE.md, CHANGES.md
├── pyproject.toml, uv.lock, Dockerfile, config_v1.toml
├── src/sparq/
│   ├── __main__.py, settings.py, setup.py, logging_config.py
│   ├── settings_old.py, default_config.toml  # legacy sandbox dependencies
│   ├── architectures/
│   │   ├── v1/
│   │   │   ├── system.py, settings.py, default_config.toml
│   │   │   ├── nodes/                       # existing graph nodes
│   │   │   ├── prompts/                     # active v1 prompts
│   │   │   ├── artifacts.py                 # executor artifact organization
│   │   │   └── reporting/                   # bundle, Markdown, PDF
│   │   └── v2/                             # existing placeholder
│   ├── schemas/                            # state, planning, runs,
│   │                                       # evaluation, data context
│   ├── evaluation/                         # maintained batch runner
│   ├── tools/                              # artifact/discovery/filesystem
│   │   ├── image_processing/               # vision tool and prompt
│   │   └── python_repl/                     # executor, namespace, packages
│   ├── data/                               # bundled manifest and summaries
│   └── utils/                              # shared runtime utilities
├── benchmarks/
│   ├── README.md
│   ├── datasets/                           # question set
│   ├── references/                         # preserved evidence bundles
│   └── results/batch_eval/                  # generated; ignored
├── tests/
│   ├── test_settings.py
│   ├── architectures/v1/                   # nodes and reporting
│   ├── evaluation/                         # batch tests
│   ├── utils/                              # helpers and model factory
│   └── tools/                              # tool tests, including python_repl/
├── scripts/                                # run, Docker, test, smoke launchers
│   └── aws/                                # model listing utility
├── docs/
│   ├── README.md
│   ├── guides/                             # data, reports, AWS setup
│   ├── architecture/                       # current REPL/concurrency behavior
│   ├── proposals/                          # roadmap and feature designs
│   ├── audits/                             # dependency/framework assessments
│   ├── adr/                                # adopted decisions
│   ├── progress/                           # historical development summaries
│   └── figures/                            # architecture assets
├── manuscripts/
│   ├── README.md
│   ├── jamia/
│   │   ├── drafts/                         # local revision snapshots; ignored
│   │   └── scripts/                        # historical local scripts; ignored
│   └── thesis/                             # chapter/context/encrypted note
├── sandbox/
│   ├── README.md
│   ├── notebooks/                          # manual checks and investigations
│   ├── scripts/bedrock/                    # SDK and LangChain API trials
│   └── archive/prompts/                    # historical prompt variants
└── eval/results/                           # existing historical local results
```

Generated directories are created on demand. Package/test initializers, local
secrets, Git internals, virtual environments, and caches are omitted from the tree.
Ordinary runtime output remains outside the repository at its configured path.

## Move checklist

| Original group | Destination | Tracking and dependent updates |
| --- | --- | --- |
| `eval/batch_eval.py` | `src/sparq/evaluation/batch.py` | New module command; imports, mocks, input/output paths updated |
| `data/Q_dataset.json` | `benchmarks/datasets/questions.json` | Remains versioned despite the general JSON ignore rule |
| `eval/results/references/` | `benchmarks/references/` | Files and symlink target preserved unchanged; Git exceptions moved |
| Existing generated batches | Original `eval/results/` | Remain ignored; historical paths are not rewritten |
| v1 `helpers/organizer.py` | `artifacts.py` and `reporting/bundle.py` | Separate artifact indexing/moves from final report generation |
| v1 `helpers/report*.py` | `reporting/markdown.py`, `reporting/pdf.py` | Imports and documentation updated |
| `schemas/output_schemas.py` | `planning.py`, `runs.py`, `evaluation.py` | Contract bodies retained; imports updated |
| Root/node/REPL test files | Corresponding test packages | Discovery initializers and mock targets updated |
| `scripts/run_scripts/` | `scripts/` | Docker root calculation and command references updated |
| `eval/smoke_test.py` | `scripts/smoke_test.py` | Maintained repository launcher |
| `utils/bedrock_utils.py` | `scripts/aws/list_models.py` | Operational model listing utility |
| `scripts/try_scripts/` | `sandbox/scripts/bedrock/` | Preserved trial scripts, not maintained examples |
| `notebooks/`, `scratch/` | `sandbox/notebooks/` | Contents preserved, including empty placeholders |
| Archived v1 prompts | `sandbox/archive/prompts/` | Contents preserved; active prompts remain beside the architecture |
| Ignored `work/` files | `manuscripts/jamia/drafts/`, `scripts/` | Remain local and ignored; historical scripts retain their old paths |
| `claude_context/`, architecture chapter, thesis note | `manuscripts/thesis/` | Tracked content preserved; plaintext note remains ignored |
| Engineering documentation | Topic directories under `docs/` | Indexes and relative links updated; original status retained |
| Utility downloader README | `docs/guides/data_setup.md` | Downloader command corrected |

## Contracts and commands

- `planning.py`: `Step`, `Plan`, `Router`, `StepResult`.
- `runs.py`: `RunStatus`, `EvaluationContext`, `SystemOutput`.
- `evaluation.py`: `BatchStatus`, `BatchRunEntry`, `BatchEvalOutput`.
- `uv run python -m sparq.evaluation.batch` replaces `python -m eval.batch_eval`;
  arguments remain unchanged. It requires benchmark assets from a checkout.
- `uv run python scripts/smoke_test.py` and
  `uv run python scripts/aws/list_models.py` are the relocated utility commands.
- Existing `uv run sparq` and configuration precedence are unchanged.
- No compatibility wrappers retain old Python imports or script paths.

## Discoveries and exceptions

- The preserved reference bundle contains an absolute `run_output` symlink to an
  external run. Relocation preserves the link itself; external files are untouched.
- Legacy Bedrock trials still import `settings_old.py`, which uses the package-level
  default configuration. Both remain until a separate modernization effort.
- Manuscript revision scripts refer to historical local paths. They are archived
  unchanged rather than rewritten or executed.
- Two scratch notebooks are empty files. They are preserved, not treated as valid
  runnable notebooks.
- The Docker production entrypoint named a nonexistent module. It now runs
  `python -m sparq`, allowing the existing launcher to pass `-t` as an argument.
- Test relocation exposed a Bedrock mock's order dependence. Its runtime client now
  supplies a string region for control-plane inference; application code is unchanged.

Before additional moves, inspect contents, dependencies, links, and tracking status.
Adjust destinations when evidence supports a better fit, and update this record.
Changes to behavior, deletion, serialized contracts, or publication of local drafts
require a separate scope decision.
