"""Render the report's Markdown subset with embedded local figures."""

from html import escape
from pathlib import Path
from urllib.parse import unquote, urlsplit

import font_roboto
from markdown_it.token import Token
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
)

from sparq.architectures.v1.reporting.markdown import parse_report


def _inline_markup(children: list[Token]) -> str:
    parts = []
    tags = {"strong": "b", "em": "i", "s": "strike"}
    for token in children:
        if token.type in {"text", "code_inline", "html_inline"}:
            parts.append(escape(token.content))
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append("<br/>" if token.type == "hardbreak" else " ")
        elif token.type == "link_open":
            destination = str(token.attrGet("href") or "")
            if destination.startswith("#"):
                destination = unquote(destination)
            href = escape(destination, quote=True)
            parts.append(f'<a href="{href}" color="#245b8a">')
        elif token.type == "link_close":
            parts.append("</a>")
        elif token.type.endswith(("_open", "_close")) and token.tag in tags:
            closing = "/" if token.nesting == -1 else ""
            parts.append(f"<{closing}{tags[token.tag]}>")
    return "".join(parts)


def _local_image(source: str, report_dir: Path) -> Path:
    """Resolve a report-relative image within its run, without remote fetching."""
    url = urlsplit(source)
    if url.scheme or url.netloc:
        raise ValueError(f"Report figures must be local files: {source}")
    path = (report_dir / unquote(url.path)).resolve()
    if not path.is_relative_to(report_dir.parent.resolve()):
        raise ValueError(f"Report figure is outside the run directory: {source}")
    if not path.is_file():
        raise FileNotFoundError(f"Report figure does not exist: {source}")
    return path


def _page_number(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Report", 9)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(A4[0] - document.rightMargin, 30, str(document.page))
    canvas.restoreState()


def generate_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Render headings, paragraphs, lists, code, pipe tables, and PNG/JPEG figures.

    Figures are embedded in the PDF. Missing figures and layout errors propagate
    to the caller; an incomplete PDF must not be reported as a successful export.
    """
    font_dir = Path(font_roboto.font_directory)
    for name, filename in [
        ("Report", "Roboto-Regular.ttf"),
        ("Report-Bold", "Roboto-Bold.ttf"),
        ("Report-Italic", "Roboto-Italic.ttf"),
        ("Report-BoldItalic", "Roboto-BoldItalic.ttf"),
    ]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "Report",
        normal="Report",
        bold="Report-Bold",
        italic="Report-Italic",
        boldItalic="Report-BoldItalic",
    )
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "Report-Bold" if style.name.startswith("Heading") else "Report"
        style.bulletFontName = "Report"
        if style.name.startswith("Heading"):
            style.keepWithNext = True
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 7
    cell_style = ParagraphStyle(
        "ReportCell", parent=styles["BodyText"], fontSize=9, leading=12
    )
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    tokens = parse_report(markdown_path.read_text(encoding="utf-8"))
    flowables: list[Flowable] = []
    list_numbers: list[int | None] = []
    bullet: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open":
            anchor = escape(str(token.attrGet("id") or ""), quote=True)
            markup = _inline_markup(tokens[index + 1].children or [])
            flowables.append(
                Paragraph(
                    f'<a name="{anchor}"/>{markup}',
                    styles[f"Heading{token.tag[1]}"],
                )
            )
            index += 3
            continue
        if token.type in {"bullet_list_open", "ordered_list_open"}:
            start = (
                int(token.attrGet("start") or "1")
                if token.type == "ordered_list_open"
                else None
            )
            list_numbers.append(start)
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            list_numbers.pop()
        elif token.type == "list_item_open":
            number = list_numbers[-1]
            bullet = "\u2022" if number is None else f"{number}."
            if number is not None:
                list_numbers[-1] = number + 1
        elif token.type == "paragraph_open":
            children = tokens[index + 1].children or []
            for child in children:
                if child.type == "image":
                    image_path = _local_image(
                        str(child.attrGet("src") or ""), markdown_path.parent
                    )
                    image = Image(str(image_path))
                    scale = min(
                        document.width / image.imageWidth, 380 / image.imageHeight, 1
                    )
                    image.drawWidth = image.imageWidth * scale
                    image.drawHeight = image.imageHeight * scale
                    image.keepWithNext = True
                    gap = Spacer(1, 8)
                    gap.keepWithNext = True
                    flowables.extend([image, gap])
            markup = _inline_markup(children)
            if markup.strip():
                style = ParagraphStyle(
                    "ReportParagraph",
                    parent=styles["BodyText"],
                    leftIndent=16 * len(list_numbers),
                    bulletIndent=16 * max(0, len(list_numbers) - 1),
                )
                flowables.append(Paragraph(markup, style, bulletText=bullet))
            bullet = None
            index += 3
            continue
        elif token.type == "table_open":
            rows: list[list[Paragraph]] = []
            while tokens[index].type != "table_close":
                if tokens[index].type == "tr_open":
                    rows.append([])
                elif tokens[index].type == "inline":
                    rows[-1].append(
                        Paragraph(
                            _inline_markup(tokens[index].children or []), cell_style
                        )
                    )
                index += 1
            table = Table(
                rows,
                colWidths=[document.width / len(rows[0])] * len(rows[0]),
                repeatRows=1,
                splitInRow=1,
                hAlign="LEFT",
            )
            table.setStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
            flowables.extend([table, Spacer(1, 10)])
        elif token.type in {"fence", "code_block"}:
            flowables.append(
                Preformatted(token.content, styles["Code"], maxLineLength=85)
            )
        index += 1
    document.build(flowables, onFirstPage=_page_number, onLaterPages=_page_number)
