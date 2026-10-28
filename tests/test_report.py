"""Check derived report sections and actual PDF exports without model calls."""

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from sparq.architectures.v1.helpers.organizer import organize_report
from sparq.architectures.v1.helpers.report import (
    ReportMetadata,
    assemble_report,
    generate_list_of_figures,
    generate_list_of_tables,
    generate_references,
    generate_toc,
    inline_text,
    parse_report,
)

REPORT = """# Synthetic research report

## Summary

This synthetic example checks report export. It is not a scientific finding.
The estimated coefficient is β = 0.5 ± 0.1, with μ = 2.

## Results

![Synthetic trend](<../executor/figures/trend plot.png>)

**Figure 1:** A synthetic trend used only to check image embedding.

Table 1: Synthetic observations, with counts in cases.

| Group | Count (cases) | Interpretation |
| --- | ---: | --- |
| A | 12 | Synthetic example |
| B | 18 | Synthetic example |

### Comparison

The example has two groups [1]. **Bold evidence** and *qualified interpretation*.

## References

[1] Example Organization. [Synthetic source](https://example.org/data?a=1&b=2). 2026.
"""


class TestReportSections(unittest.TestCase):
    def test_question_markdown_stays_literal_and_out_of_navigation(self):
        question = "# References\n\n![Figure 9](https://example.org/image.png)\n[1] Fake reference."
        metadata = ReportMetadata(
            question=question,
            question_asked_at=None,
            generated_at=datetime(2026, 10, 28, tzinfo=UTC),
            author_model="author",
            author_role="aggregator",
            planner_model="planner",
            executor_model="executor",
        )
        assembled = assemble_report("# Title\n\n## Results\n\nA result.", metadata)
        tokens = parse_report(assembled)
        children = [child for token in tokens for child in token.children or []]
        self.assertFalse(any(child.type == "image" for child in children))
        headings = [
            inline_text(tokens[index + 1])
            for index, token in enumerate(tokens)
            if token.type == "heading_open"
        ]
        self.assertEqual(headings, ["Title", "Table of contents", "Results"])
        self.assertEqual(generate_references(assembled), "")

    def test_contents_ignore_code_and_have_distinct_duplicate_anchors(self):
        toc = generate_toc("# Title\n\n## Results\n\n```\n## Fake\n```\n\n## Results\n")
        self.assertEqual(
            toc,
            "## Table of contents\n\n- [Results](#results)\n- [Results](#results-1)",
        )

    def test_lists_include_real_caption_paragraphs_and_ignore_code(self):
        markdown = (
            REPORT + "\n```\nFigure 2: Not a figure.\nTable 2: Not a table.\n```\n"
        )
        self.assertEqual(
            generate_list_of_figures(markdown),
            "## List of figures\n\n- Figure 1: A synthetic trend used only to check image embedding.",
        )
        self.assertEqual(
            generate_list_of_tables(markdown),
            "## List of tables\n\n- Table 1: Synthetic observations, with counts in cases.",
        )
        self.assertEqual(generate_list_of_figures("# Plain answer"), "")
        self.assertEqual(generate_list_of_tables("# Plain answer"), "")

    def test_bibliography_preserves_reference_text_without_inventing_fields(self):
        bibliography = generate_references(REPORT)
        self.assertIn("@misc{ref1,", bibliography)
        self.assertIn(
            "note = {Example Organization. Synthetic source. 2026.}", bibliography
        )
        self.assertIn(r"url = {https://example.org/data?a=1\&b=2}", bibliography)
        self.assertNotIn("author =", bibliography)
        self.assertNotIn("title =", bibliography)
        self.assertEqual(
            generate_references("A claim [1] without a reference entry."), ""
        )

    def test_reference_scope_and_bibtex_special_characters(self):
        bibliography = generate_references(
            "## References\n\n[3] A {study} with 20% & x_y.\n\n## Appendix\n\n[4] Not a reference.\n"
        )
        self.assertIn(r"A \{study\} with 20\% \& x\_y.", bibliography)
        self.assertNotIn("ref4", bibliography)
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            generate_references("## References\n\n[1] First.\n\n[1] Second.")


class TestReportBundle(unittest.TestCase):
    def test_bundle_embeds_image_preserves_data_and_has_working_contents_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure = root / "executor/figures/trend plot.png"
            figure.parent.mkdir(parents=True)
            Image.new("RGB", (640, 240), "#245b8a").save(figure)
            original_bytes = figure.read_bytes()

            report_dir = organize_report(REPORT, root)

            markdown = (report_dir / "report.md").read_text()
            self.assertTrue(
                markdown.startswith(
                    "# Synthetic research report\n\n## Table of contents"
                )
            )
            self.assertIn("## List of figures", markdown)
            self.assertIn("## List of tables", markdown)
            self.assertIn("12", markdown)
            self.assertIn("ref1", (report_dir / "ref.bib").read_text())
            self.assertEqual(figure.read_bytes(), original_bytes)
            self.assertFalse((root / "index.json").exists())
            reader = PdfReader(report_dir / "report.pdf")
            text = " ".join(
                " ".join(page.extract_text() for page in reader.pages).split()
            )
            self.assertIn("β = 0.5 ± 0.1", text)
            self.assertIn("μ = 2", text)
            self.assertIn("Count (cases)", text)
            self.assertIn("Example Organization.", text)
            self.assertEqual(sum(len(page.images) for page in reader.pages), 1)
            links = [
                annotation.get_object()
                for page in reader.pages
                for annotation in page.get("/Annots", [])
            ]
            self.assertTrue(any("/Dest" in link for link in links))
            self.assertFalse((report_dir / "report.tmp.pdf").exists())

    def test_long_table_repeats_headers_and_retains_last_row_and_unicode_heading(self):
        with tempfile.TemporaryDirectory() as directory:
            markdown = (
                "# Long report\n\n## β estimates\n\nTable 1: Synthetic cases.\n\n"
            )
            markdown += "| Group | Cases |\n| --- | --- |\n"
            markdown += "\n".join(f"| Synthetic-{row} | {row} |" for row in range(100))
            report_dir = organize_report(markdown, Path(directory))
            pages = PdfReader(report_dir / "report.pdf").pages
            self.assertGreater(len(pages), 2)
            self.assertTrue(all("Cases" in page.extract_text() for page in pages))
            self.assertIn("Synthetic-99", pages[-1].extract_text())

    def test_repeated_export_removes_stale_bibliography_and_empty_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = organize_report(
                "# Title\n\n## References\n\n[1] Supplied source.", root
            )
            self.assertTrue((report_dir / "ref.bib").exists())
            organize_report("A direct answer without references.", root)
            self.assertFalse((report_dir / "ref.bib").exists())
            self.assertEqual(
                (report_dir / "report.md").read_text(),
                "A direct answer without references.\n",
            )
            self.assertEqual(len(PdfReader(report_dir / "report.pdf").pages), 1)

    def test_missing_figure_keeps_markdown_but_does_not_leave_a_stale_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = organize_report("An earlier report.", root)
            with self.assertRaises(FileNotFoundError):
                organize_report(
                    "# Updated report\n\n![Missing](../executor/missing.png)", root
                )
            self.assertIn("Updated report", (report_dir / "report.md").read_text())
            self.assertFalse((report_dir / "report.pdf").exists())
            self.assertFalse((report_dir / "report.tmp.pdf").exists())

    def test_outside_and_remote_figures_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for image in ("../../outside.png", "https://example.org/plot.png"):
                with self.subTest(image=image), self.assertRaises(ValueError):
                    organize_report(f"![Invalid]({image})", root)


if __name__ == "__main__":
    unittest.main()
