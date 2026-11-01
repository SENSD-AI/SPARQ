# Evaluation results

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
Bedrock model auditing is a [utility with separate cache/artifact locations](aws/check_models.md).

## Choose a destination

```bash
uv run python -m sparq.evaluation.batch -n 10 -k 1 \
  --evaluation-id baseline-v1 --output evaluation/results/batch_eval/baseline-v1
```

`--output` names the exact bundle directory, not a parent for another generated
timestamp folder. Relative overrides resolve against the working directory and
`~` expands to the home directory. Batch output must not already exist. Bedrock
utility output options are described in its [guide](aws/check_models.md).

`evaluation_id` groups separate batches logically; it is metadata and does not
alter their directory paths. Batch IDs and run IDs remain independent of an
explicit output directory's name.

## Move and preserve results

`batch.json` run-result paths and newly saved Bedrock `report.json`/`results.csv`
artifact-directory paths are relative to their bundle root. Resolve them against
the containing bundle, not the shell's working directory. Copy the entire bundle
to preserve these links. Bedrock `job.json` is worker input and may retain absolute
execution paths; it is historical provenance, not a portable replay command.

Historical results from `eval/results/` and `benchmarks/results/` now live under
`evaluation/results/`. Their contents, old timestamp formats, and recorded paths
are preserved. See the [migration record](../../evaluation/MIGRATION.md) for path
mappings and integrity verification. Inspect provenance before promoting a selected bundle
into `evaluation/references/`; successful execution does not establish scientific
validity.

## Code ownership

- `sparq/evaluation/paths.py`: shared generated-results root and bundle naming.
- `sparq/evaluation/batch.py`: batch directories, manifest and question linkage.
- `sparq/utils/bedrock/`: model discovery and auditing utilities, stored separately
  from evaluation outputs. Historical Bedrock bundles retain their original paths.

Per-run saving and report generation retain their existing layout.
