# Report generation

The v1 saver retains `final_answer.json` and writes a report bundle for each
nonempty final assistant answer:

```text
run_dir/
  report/
    report.md
    report.pdf
    ref.bib       # only when numbered references exist
```

`organize_report(markdown, run_dir)` assembles this bundle. It is separate from
`organize()`, which moves executor artifacts into category folders. Report
generation leaves those artifacts and their index in place.

## Helpers

The helpers live under `src/sparq/architectures/v1/helpers/`:

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

## Markdown contract

The aggregator writes the report body. Helpers insert navigation after its title;
the aggregator must not also generate contents or figure/table lists.

- Use one `# Title`, then `##` and `###` headings.
- Use ordinary paragraphs, emphasis, links, lists, fenced code, and pipe tables.
- Put each PNG/JPEG figure in a standalone image paragraph. The image destination
  is relative to `report/`: `![Trend](../executor/figures/trend.png)`. For spaces,
  use `![Trend](<../executor/figures/trend plot.png>)`. Filesystem tool calls still
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

## Failures and regeneration

The saver writes the JSON answer and assembled Markdown before PDF rendering.
A missing figure, invalid image, or layout failure raises an error rather than
claiming the PDF succeeded. Temporary PDFs are removed. An earlier PDF is removed
before regeneration so it cannot be mistaken for a render of the updated Markdown.
An earlier `ref.bib` is removed when the new report has no numbered references.

Direct answers also produce Markdown and PDF, but do not acquire empty navigation
sections. Missing assistant answers produce only the existing JSON export.
