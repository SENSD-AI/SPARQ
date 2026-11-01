"""Collect local tool evidence for the Bedrock feature-coverage workflow."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tracers.schemas import Run
from pydantic import BaseModel, Field

REQUIRED_TOOLS = {
    "get_cached_dataset_path",
    "find_csv_excel_files",
    "get_sheet_names",
    "load_dataset",
    "python_repl_tool",
    "write_todos",
    "list_directory",
    "file_search",
    "write_file",
    "copy_file",
    "read_file",
    "list_artifacts",
    "search_artifact",
    "analyze_image",
}


class ToolEvidence(BaseModel):
    """A tool's actual input, output and execution error, without LLM traces."""

    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    output: str
    error: str | None = None


def collect_tool_evidence(runs: list[Run]) -> list[ToolEvidence]:
    """Flatten nested local traces and retain failed calls for diagnosis."""
    evidence = []
    for run in runs:
        if run.run_type == "tool":
            output = (run.outputs or {}).get("output", "")
            error = run.error
            if isinstance(output, ToolMessage):
                if (
                    output.status == "error"
                    or getattr(output.artifact, "success", True) is False
                ):
                    error = error or str(output.content)
                output = output.content
            text = (
                output if isinstance(output, str) else json.dumps(output, default=str)
            )
            if text.startswith(("Error:", "Failed to", "PythonError:", "✗")):
                error = error or text
            evidence.append(
                ToolEvidence(name=run.name, inputs=run.inputs, output=text, error=error)
            )
        evidence.extend(collect_tool_evidence(run.child_runs))
    return evidence


def verify_tool_coverage(run_dir: Path) -> None:
    """Require successful calls, persistent Python, both input formats and vision."""
    calls = [
        ToolEvidence.model_validate(item)
        for item in json.loads((run_dir / "tool_calls.json").read_text())
    ]
    successful = [call for call in calls if not call.error]
    missing = REQUIRED_TOOLS - {call.name for call in successful}
    if missing:
        raise ValueError(
            f"SPARQ feature coverage missing successful tools: {sorted(missing)}"
        )
    loads = [call.inputs for call in successful if call.name == "load_dataset"]
    if not any(str(item.get("file_path", "")).endswith(".csv") for item in loads):
        raise ValueError("SPARQ did not load a CSV dataset.")
    if not any(
        str(item.get("file_path", "")).endswith(".xlsx") and item.get("sheet_name")
        for item in loads
    ):
        raise ValueError("SPARQ did not load a named Excel sheet.")
    if not any(
        call.name == "python_repl_tool" and call.inputs.get("persist_namespace") is True
        for call in successful
    ):
        raise ValueError("SPARQ did not use persistent Python execution.")
    # A file-error string from the vision tool is not evidence of image inference.
    for call in successful:
        if call.name == "analyze_image":
            try:
                report = json.loads(call.output)
            except (ValueError, TypeError):
                continue
            if (
                isinstance(report, dict)
                and report.get("summary")
                and report.get("answer")
            ):
                return
    raise ValueError("SPARQ did not return a structured visual assessment.")
