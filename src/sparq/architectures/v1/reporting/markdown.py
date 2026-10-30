"""Derive report navigation and bibliography from the supplied Markdown."""

import re
from datetime import datetime

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel


class ReportMetadata(BaseModel):
    """Recorded run details displayed independently of the generated report body."""

    question: str
    question_asked_at: datetime | None
    generated_at: datetime
    author_model: str
    author_role: str
    planner_model: str
    executor_model: str


def _escape_markdown(text: str) -> str:
    """Keep recorded text literal, including questions containing Markdown images."""
    escaped = re.sub(r"([\\`*_{}\[\]<>()#+\-.!|>~&])", r"\\\1", text)
    return "  \n".join(escaped.splitlines())


def generate_report_header(metadata: ReportMetadata) -> str:
    """Format question, timestamps, and model attribution from run metadata."""
    asked_at = (
        metadata.question_asked_at.isoformat(timespec="seconds")
        if metadata.question_asked_at is not None
        else "Not recorded"
    )
    fields = [
        ("Question asked", asked_at),
        ("Report generated", metadata.generated_at.isoformat(timespec="seconds")),
        ("Author", f"{metadata.author_model} ({metadata.author_role})"),
        ("Planner model", metadata.planner_model),
        ("Executor model", metadata.executor_model),
    ]
    details = "\n".join(
        f"- **{label}:** {_escape_markdown(value)}" for label, value in fields
    )
    return f"**User question**\n\n{_escape_markdown(metadata.question)}\n\n{details}"


def inline_text(token: Token) -> str:
    """Get visible inline text without Markdown formatting."""
    return "".join(
        " " if child.type in {"softbreak", "hardbreak"} else child.content
        for child in token.children or []
        if child.type in {"text", "code_inline", "softbreak", "hardbreak", "image"}
    )


def parse_report(markdown: str) -> list[Token]:
    """Parse Markdown tables and assign consistent, unique heading anchors."""
    tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(markdown)
    used: set[str] = set()
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            title = inline_text(tokens[index + 1]).lower()
            slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-") or "section"
            anchor = slug
            suffix = 1
            while anchor in used:
                anchor = f"{slug}-{suffix}"
                suffix += 1
            used.add(anchor)
            token.attrSet("id", anchor)
    return tokens


def generate_toc(markdown: str) -> str:
    """Generate a linked contents list, excluding the report title."""
    tokens = parse_report(markdown)
    lines = [
        f"{'  ' * (int(token.tag[1]) - 2)}- "
        f"[{inline_text(tokens[index + 1])}](#{token.attrGet('id')})"
        for index, token in enumerate(tokens)
        if token.type == "heading_open" and token.tag != "h1"
    ]
    return "## Table of contents\n\n" + "\n".join(lines) if lines else ""


def _caption_list(markdown: str, kind: str) -> str:
    tokens = parse_report(markdown)
    captions = []
    for index, token in enumerate(tokens):
        if token.type == "paragraph_open" and token.level == 0:
            text = inline_text(tokens[index + 1])
            if re.match(rf"^{kind}\s+\d+[.:]\s+", text):
                captions.append(f"- {text}")
    return f"## List of {kind.lower()}s\n\n" + "\n".join(captions) if captions else ""


def generate_list_of_figures(markdown: str) -> str:
    """List numbered figure captions in report order; omit an empty list."""
    return _caption_list(markdown, "Figure")


def generate_list_of_tables(markdown: str) -> str:
    """List numbered table captions in report order; omit an empty list."""
    return _caption_list(markdown, "Table")


def _bibtex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def generate_references(markdown: str) -> str:
    """Export numbered References entries as BibTeX misc notes, without guessing metadata.

    Entries must be separate paragraphs beginning with [1], [2], etc.
    Preserve the complete visible entry in ``note`` and its first link in ``url``.
    """
    tokens = parse_report(markdown)
    reference_level: int | None = None
    entries: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])
            if reference_level is not None and level <= reference_level:
                break
            if inline_text(tokens[index + 1]).casefold() == "references":
                reference_level = level
        if reference_level is None or token.type != "paragraph_open":
            continue
        inline = tokens[index + 1]
        match = re.match(r"^\[(\d+)\]\s+(.+)$", inline_text(inline))
        if match is None:
            continue
        number, text = match.groups()
        if number in entries:
            raise ValueError(f"Duplicate reference number: {number}")
        urls = [
            str(child.attrGet("href") or "")
            for child in inline.children or []
            if child.type == "link_open"
        ]
        fields = [f"  note = {{{_bibtex_text(text)}}}"]
        if urls:
            fields.append(f"  url = {{{_bibtex_text(urls[0])}}}")
        entries[number] = f"@misc{{ref{number},\n" + ",\n".join(fields) + "\n}"
    return "\n\n".join(entries.values()) + ("\n" if entries else "")


def assemble_report(markdown: str, metadata: ReportMetadata | None = None) -> str:
    """Insert recorded metadata and derived navigation after the report title."""
    tokens = parse_report(markdown)
    title_end = 0
    if tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1":
        title_end = (tokens[0].map or [0, 0])[1]
    lines = markdown.splitlines()
    sections = [
        "\n".join(lines[:title_end]),
        generate_report_header(metadata) if metadata is not None else "",
        generate_toc(markdown),
        generate_list_of_figures(markdown),
        generate_list_of_tables(markdown),
        "\n".join(lines[title_end:]),
    ]
    return (
        "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"
    )
