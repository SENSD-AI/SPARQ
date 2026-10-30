"""Write deterministic report bundles from final Markdown."""

from pathlib import Path

from sparq.architectures.v1.reporting.markdown import (
    ReportMetadata,
    assemble_report,
    generate_references,
)
from sparq.architectures.v1.reporting.pdf import generate_pdf


def organize_report(
    markdown: str,
    run_dir: Path,
    metadata: ReportMetadata | None = None,
) -> Path:
    """Write the final report bundle without moving executor artifacts.

    Write Markdown first so it remains available if PDF rendering fails.
    A bibliography is emitted only when numbered reference entries exist.
    """
    if not markdown.strip():
        raise ValueError("Cannot organize an empty report.")
    assembled = assemble_report(markdown, metadata)
    bibliography = generate_references(markdown)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "report.md"
    markdown_path.write_text(assembled, encoding="utf-8")
    bibliography_path = report_dir / "ref.bib"
    if bibliography:
        bibliography_path.write_text(bibliography, encoding="utf-8")
    else:
        bibliography_path.unlink(missing_ok=True)
    # Do not leave an old PDF beside an updated Markdown report if rendering fails.
    pdf_path = report_dir / "report.pdf"
    pdf_path.unlink(missing_ok=True)
    temporary_pdf = report_dir / "report.tmp.pdf"
    try:
        generate_pdf(markdown_path, temporary_pdf)
        temporary_pdf.replace(pdf_path)
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return report_dir
