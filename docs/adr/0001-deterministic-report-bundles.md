# Deterministic report bundles

- Date: 2026-10-27
- Status: Adopted for v1
- Authority: User requested report helpers and `run_dir/report/`; renderer and
  input conventions are implementation choices.

## Decision

Keep the aggregator responsible for evidence synthesis and Markdown. Let the saver
call deterministic helpers to derive contents, figure/table lists, BibTeX, and PDF,
then place the final outputs under `run_dir/report/`.

Use markdown-it-py to parse Markdown instead of scanning code fences and headings
with regular expressions alone. Use ReportLab for PDF layout and packaged Roboto
fonts for consistent Latin, Greek, and Cyrillic text. This avoids requiring a
system LaTeX or browser installation.

Retain executor artifacts in their existing locations. Markdown references them
relative to the report directory; the PDF embeds the images. Do not reuse the
artifact organizer's category moves for final reports.

Preserve numbered reference prose in BibTeX `@misc` notes and known URLs. Extracting
author/title/publication fields from arbitrary prose would require inference, so
those fields remain absent. No external references are invented.

## Consequences

The report writer must follow the documented caption, path, and reference
conventions. The renderer supports ordinary Markdown and pipe tables, rather than
arbitrary HTML or LaTeX. Contents use links without page numbers. PDF failures
propagate, retaining the JSON answer and Markdown, without leaving a stale PDF.

See [report generation](../reports.md) for the contract and helper entry points.
