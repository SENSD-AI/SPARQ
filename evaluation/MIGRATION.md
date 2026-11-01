# Evaluation directory consolidation — 2026-10-31

`evaluation/` is the single repository location for evaluation datasets,
preserved references, and generated results. Python implementations remain in
`src/sparq/evaluation/`; tests remain in `tests/evaluation/`.

| Previous location | Current location |
| --- | --- |
| `benchmarks/datasets/` | `evaluation/datasets/` |
| `benchmarks/references/` | `evaluation/references/` |
| `benchmarks/results/` | `evaluation/results/` |
| `eval/results/batch_eval/` | `evaluation/results/batch_eval/` |
| `eval/results/batch_eval_results/` | `evaluation/results/batch_eval_results/` |

The two batch-result collections were merged without name collisions. The legacy
`eval/results/references/` tree contained no files; preserved reference bundles
are under `evaluation/references/`. No compatibility directories or symlinks were
created at the old roots.

## Evidence and historical paths

All 3,903 regular files moved during consolidation were verified with SHA-256
before and after the move. One external run symlink retained its target and
resolution state. All 315 manifest-to-result links retained their pre-move
existence state: 69 result files existed and 246 were already absent. This check
establishes preservation, not successful completion of every historical run.

The local, Git-ignored receipt at
`results/_migration/2026-10-31-consolidation.json` records every moved file's old
and new path, size and hash, plus the manifest-link comparison. It reflects the
move before subsequent source/documentation edits to the evaluation README.

Historical JSON, reports, logs, checksums, and reference documentation were not
rewritten. Recorded `benchmarks/...` or `eval/results/...` paths should be mapped
using the table above. Older manifests may name `data/Q_dataset.json`; the current
question dataset is `evaluation/datasets/questions.json`. Check the recorded
dataset hash before treating it as the same input. Absolute external paths still
require their original data or a separately verified replacement.

The preserved reference bundle's internal README describes an earlier Git ignore
exception. Its current tracked location is `evaluation/references/`; that old
wording is retained because the bundle's checksums cover it.

Obsolete Python bytecode caches and Finder metadata from the legacy `eval/` root
were retained at `/private/tmp/sparq-eval-legacy-cache-2026-10-31/`, outside the
project. All result files, including hidden metadata, were retained in the move.
