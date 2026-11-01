# 4. Synthesis, reports, and saved results

[Chapter 3](03-execution.md) produced structured step results and indexed files.
For our running example, those could include county-rate estimates, a chart,
and notes about incomplete data. The remaining stages turn that evidence into
an answer and preserve it for inspection.

## Stage 5: synthesize the evidence

The aggregator formats all `StepResult` fields into its prompt alongside the
original question. It uses its configured model and can search/list artifacts,
read files, and analyze images through the vision model. It works from the run
root, with the artifact index at `executor/index.json`; it does not receive a
live merged Python namespace or the executor's Python tool.

The agent writes the final assistant message into graph state. Its job is to
explain the findings and limitations coherently. If there are no results, the
node supplies an explicit no-results assistant message instead. A completed
report is still a model-generated synthesis and requires scientific review.

There is a current size-handling limitation: the aggregator counts formatted
result tokens, but its over-budget `truncate_results()` branch is a stub. Long
results can therefore fail at that branch; automatic truncation is not an
implemented safeguard.

## Stage 6: save the answer and render its report

The saver reads the last assistant message and first writes the question and
answer to `final_answer.json`. For a nonempty assistant answer, deterministic
helpers then assemble Markdown, navigation, bibliography, and PDF. The saver
uses the same path for a router's direct answer; its metadata identifies which
model authored that answer.

## Report generation

The v1 saver retains `final_answer.json` and writes `report/report.md`,
`report/report.pdf`, and, when numbered references exist, `report/ref.bib` for
each nonempty final assistant answer.

`organize_report(markdown, run_dir, metadata=None)` assembles this bundle. It is
separate from `organize()`, which moves executor artifacts into category folders. Report
generation leaves those artifacts and their index in place.

### Report header

The saver inserts recorded metadata after the title and before navigation in both
Markdown and PDF:

- The complete user question, displayed as literal text.
- The date and time the question was received by `run()` and report generation
  began, in ISO 8601 format with a UTC offset.
- The report author: aggregator model name and provider.
- The planner and executor model names and providers.

For direct answers that skip analysis, the author is the router model and both
workers are marked `Not used`. Missing historical timestamps or model settings
are marked `Not recorded`; they are not inferred from file dates or report text.
The question receipt timestamp is retained in `trace.json`. Helpers called
without metadata retain their existing behavior.

### Helpers

The helpers live under `src/sparq/architectures/v1/reporting/`:

- `generate_toc`: links to headings below the level-one title, including nested
  headings and distinct anchors for repeated headings. The contents list uses
  links, without page numbers, in both Markdown and PDF.
- `generate_list_of_figures` and `generate_list_of_tables`: collect numbered
  caption paragraphs in appearance order, ignoring code blocks. Empty lists
  are omitted.
- `generate_references`: exports numbered entries under `## References` as
  BibTeX `@misc` records (`ref1`, `ref2`, etc.). The complete supplied entry is
  retained in `note`, with its first Markdown link in `url`. Author, title,
  publication type, and other metadata are not guessed from prose.
- `generate_pdf`: renders the assembled Markdown using ReportLab, with embedded
  fonts and images, wrapping table cells, repeated table headers, and page numbers.

### Markdown contract

The aggregator writes the report body. Helpers insert metadata and navigation
after its title; the aggregator must not also generate contents or figure/table lists.

- Use one `# Title`, then `##` and `###` headings.
- Use ordinary paragraphs, emphasis, links, lists, fenced code, and pipe tables.
- Put each PNG/JPEG figure in a standalone image paragraph. The image destination
  is relative to `report/`: `![Trend](../executor/figures/trend.png)`. For spaces,
  use `![Trend](../executor/figures/trend%20plot.png)`. Filesystem tool calls still
  use paths relative to the run root, such as `executor/figures/trend.png`.
- Use a separate caption paragraph beginning with `Figure 1:` or `Table 1:`.
  Number each kind in appearance order; the helpers preserve the supplied numbers.
- Under `## References`, put each entry in a separate paragraph beginning with
  `[1]`, `[2]`, etc. Use Markdown links for known URLs. Omit the section if no
  sources are available. Bibliography generation does not retrieve sources or
  establish that citations support claims.

Raw HTML, LaTeX mathematics, interactive figures, and remote images are outside
the supported PDF subset. Local image paths must resolve within the run directory.
The embedded Roboto font covers Latin, Greek, and Cyrillic text; it is not a
universal font for every script.

### Failures and regeneration

The saver writes the JSON answer and assembled Markdown before PDF rendering.
A missing figure, invalid image, or layout failure raises an error rather than
claiming the PDF succeeded. Temporary PDFs are removed. An earlier PDF is removed
before regeneration so it cannot be mistaken for a render of the updated Markdown.
An earlier `ref.bib` is removed when the new report has no numbered references.

Direct answers also produce Markdown and PDF, but do not acquire empty navigation
sections. Missing assistant answers produce only the existing JSON export.


## Stage 7: finish the run

After the saver completes, the graph reaches `END`. The outer `run()` loop
cleans up its temporary namespaces, verifies that the final message is an
assistant message, builds `SystemOutput`, saves `result.json`, and returns that
result to the caller. The CLI displays it.

During graph streaming, `trace.json` is overwritten with each latest state.
It is a snapshot rather than a chronological event log. `log.txt` captures
run-scoped logging. If the graph fails, the `finally` block still attempts
namespace cleanup, but the final `result.json` is not written. Previously saved
files can remain as partial evidence.

## Run outputs

A completed analytical run has this layout; executor files and bibliography depend
on what the run produces:

```text
run_dir/
  executor/          # artifact index, generated plots, tables, and other files
  report/
    report.md
    report.pdf
    ref.bib          # only when numbered references exist
  final_answer.json  # question and final assistant answer
  trace.json         # latest streamed graph-state snapshot
  result.json        # structured run result and metadata
  log.txt            # run-scoped logs
```

`SystemOutput` records the run ID, question, difficulty, model configuration,
response, timings, and optional evaluation context. Token/cost and judge fields
are currently left unset by `run()`. Direct answers skip executor processing, so
an executor directory is not required. An exception can leave partial evidence;
the presence of a directory or report alone does not establish run success or
scientific validity. In particular, `result.json` is written only after the graph
finishes successfully.


## Where this is implemented

- [Aggregator and token-size branch](../../../src/sparq/architectures/v1/nodes/aggregator.py)
- [Saver](../../../src/sparq/architectures/v1/nodes/saver.py)
- [Report bundle assembly](../../../src/sparq/architectures/v1/reporting/bundle.py)
- [Markdown helpers and report metadata](../../../src/sparq/architectures/v1/reporting/markdown.py)
- [PDF renderer](../../../src/sparq/architectures/v1/reporting/pdf.py)
- [Run completion and JSON exports](../../../src/sparq/architectures/v1/system.py)

Next: [repeat the run in an evaluation and understand concurrency boundaries](05-evaluation-and-concurrency.md).
