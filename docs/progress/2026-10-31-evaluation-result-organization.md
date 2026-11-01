# Evaluation result organization — 2026-10-31

Scope: the user selected batch/model evaluation folders. Per-run exports and
report generation are unchanged.

Implemented:

- Shared result-root resolution and sortable UTC bundle IDs in
  `sparq/evaluation/paths.py`, used by batch and Bedrock evaluations.
- Batch `--output` selects the exact bundle directory. Default locations remain
  `evaluation/results/batch_eval/` and `evaluation/results/bedrock_models/`.
- Newly saved Bedrock JSON/CSV artifact links are relative to their sweep root,
  including interrupted runs. Batch result links were already relative.
- A [layout and usage guide](../guides/evaluation-results.md), linked from the
  evaluation README and documentation index.

Following the user's explicit consolidation request, `benchmarks/` and legacy
`eval/results/` were moved into `evaluation/`. SHA-256 checks verified all 3,903
moved files, the external run symlink was preserved, and all 315 manifest links
retained their prior existence state. There were 246 already-missing result files.
See the [migration record](../../evaluation/MIGRATION.md). Historical evidence
contents and the user's pre-existing `config_v1.toml` edits were preserved.

Validation:

- `uv run --no-sync python -m unittest discover -s tests/evaluation -v`: 39 pass.
  Synthetic batch and model bundles were renamed and their saved links resolved
  successfully from the new locations. No AWS inference was performed.
- BasedPyright over all six new/modified Python files: zero errors or warnings.
- Ruff lint/format pass for the four shared-path/Bedrock files. The two batch
  files retain seven lint findings and formatting differences also present in
  `HEAD`; lint codes/messages match the baseline exactly.
- Both runner help commands and `git diff --check` pass.
- After consolidation, the 39 evaluation tests and type checks pass again;
  preserved reference `SHA256SUMS` pass, versioned datasets/references remain
  trackable, generated results remain ignored, and both old roots are absent.

Continuity owner: `agentic-epidemiology`, verified through its registered SPARQ
repository. This is the repository delivery record. External project-log filing
remains pending; no Obsidian write is claimed.
