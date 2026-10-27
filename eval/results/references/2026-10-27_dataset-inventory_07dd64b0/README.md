# Dataset inventory evaluation reference

This bundle preserves the report produced by the 27 October 2026, 13:07:52 SPARQ
run and a separate assistant factual audit. It is a reference candidate, not a
human-adjudicated gold answer or a formal numerical evaluation score.

## Start here

- [Original generated PDF](run/report/report.pdf)
- [Original generated Markdown](run/report/report.md)
- [Assistant factual audit](audit.md)
- [Structured findings](findings.json)
- [Run details and models](metadata.json)
- [Actual run output directory](run_output/): absolute symlink to the original
  output directory. The `run/` snapshot remains available if the original moves.

## Contents

```text
README.md
metadata.json
findings.json
audit.md
SHA256SUMS
run_output -> /Users/zayan/tmp/sparq/output/27-10-2026_13-07-52
run/
  report/report.pdf
  report/report.md
  report/ref.bib
  executor/                  # three figures, two worker documents, artifact index
  result.json                # original model configurations, timings, response
  trace.json                 # original state, plan, results, supplied summaries
  final_answer.json
  log.txt
evidence/
  verified_facts.json
  run_files_manifest.json
  input_files_manifest.json
  public_sources.json
```

All 13 original run files were copied byte-for-byte, retaining their layout so
relative figure links in the Markdown work. `run_files_manifest.json` records their
SHA-256 hashes. The 30 source dataset/documentation files were fingerprinted, but
raw datasets were not copied. Recomputing facts therefore requires access to those
inputs at the recorded paths or byte-identical copies. Hashes identify the inputs
present during collection; the original run did not record input hashes, so exact
execution-time identity is not independently established.

## Evaluation use

Use `run/report/` as the preserved system output. Use `audit.md`, `findings.json`,
and `evidence/verified_facts.json` as review evidence. Each finding has an ID,
classification, severity, original report line references, a correction or required
qualification, and a provenance limitation.

Keep confirmed errors separate from unsupported quantitative claims and missing
methodological qualifications. Missing saved provenance does not by itself prove
fabrication. Do not count the 12 entries as 12 independent model errors or use their
severities as calibrated weights: related entries overlap and require adjudication.
The report is a negative/reference case, not text to reproduce as a correct answer.

The audit has not been reviewed by a human. It is not an exhaustive data-quality
assessment, and it does not establish which exact model/tool call introduced each
error. Saved trace messages contain only the user and final assistant response;
worker tool-call history is absent. Current checkout metadata reflects collection,
not a verified execution-time code snapshot. Token usage, cost, and original judge
scores are null rather than zero.

This directory is tracked by Git through an exception for
`eval/results/references/`; other evaluation results remain ignored. The symlink
uses an absolute local path and is not portable; the copied run snapshot and audit
remain readable independently. Git stores the symlink target, not the contents of
the original directory it points to.

## Integrity check

From this directory, run:

```bash
shasum -a 256 -c SHA256SUMS
```

`SHA256SUMS` covers all regular files in the bundle except itself. It excludes the
symlink and does not traverse the original run directory.
