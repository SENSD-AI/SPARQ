"""Run one SPARQ feature-coverage query for the Bedrock model sweep."""

import asyncio
import csv
import json
import math
import sys
import traceback
from pathlib import Path

from langchain_core.tracers.context import collect_runs
from markdown_it import MarkdownIt

from sparq.utils.bedrock.audit import (
    ProbeJob,
    RunResult,
    atomic_json,
    failure_result,
)
from sparq.utils.bedrock.coverage import (
    collect_tool_evidence,
    verify_tool_coverage,
)


def verify_run(run_dir: Path) -> None:
    """Do not accept a returned answer when analytical steps actually failed."""
    trace = json.loads((run_dir / "trace.json").read_text())
    plan = trace.get("plan") or {}
    if trace.get("route") is not True or not plan.get("steps"):
        raise ValueError(
            "The query did not exercise the planner/executor/aggregator path."
        )
    results = trace.get("results", [])
    expected = {step["id"] for step in plan["steps"]}
    completed = {result["id"] for result in results if result.get("success") is True}
    if not results or not expected <= completed:
        failures = [
            {"id": result.get("id"), "reason": result.get("misc", "")}
            for result in results
            if not result.get("success")
        ]
        raise ValueError(f"SPARQ execution steps did not all succeed: {failures}")
    for name in (
        "result.json",
        "final_answer.json",
        "report/report.md",
        "report/report.pdf",
        "report/ref.bib",
    ):
        path = run_dir / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"SPARQ did not save a nonempty {name}.")
    steps = plan["steps"]
    identifiers = {step["id"] for step in steps}
    if (
        len(identifiers) != len(steps)
        or sum(not step.get("dependencies") for step in steps) < 2
        or not any(len(set(step.get("dependencies", []))) >= 2 for step in steps)
        or any(not set(step.get("dependencies", [])) <= identifiers for step in steps)
    ):
        raise ValueError(
            "SPARQ did not exercise independent steps and dependency merging."
        )
    verify_tool_coverage(run_dir)
    index = json.loads((run_dir / "executor/index.json").read_text())
    for category in ("figures", "tables", "docs"):
        paths = list(index.get(category, {}).values())
        if not paths or any(
            not (run_dir / "executor" / path).is_file()
            or not (run_dir / "executor" / path).stat().st_size
            for path in paths
        ):
            raise ValueError(f"SPARQ did not save indexed {category} artifacts.")
    table = run_dir / "executor/tables/feature_summary.csv"
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or any(
        not row.get("dataset")
        or not row.get("numeric_column")
        or not 0 < int(row["non_null_count"]) <= int(row["rows"]) <= 200
        or not math.isfinite(float(row["mean"]))
        for row in rows
    ):
        raise ValueError("SPARQ did not save the bounded two-dataset summary table.")
    docs = run_dir / "executor/docs"
    methods = (docs / "methods.txt").read_text()
    if not methods.strip() or (docs / "methods_copy.txt").read_text() != methods:
        raise ValueError("SPARQ did not preserve the copied methods document.")
    if not (docs / "visual_check.txt").read_text().strip():
        raise ValueError("SPARQ did not save its visual assessment.")
    markdown = (run_dir / "report/report.md").read_text()
    images = []
    for token in MarkdownIt().parse(markdown):
        for child in token.children or []:
            if child.type == "image":
                source = child.attrGet("src")
                if source:
                    images.append(source)
    figures = {
        (run_dir / "executor" / path).resolve() for path in index["figures"].values()
    }
    embedded = {(run_dir / "report" / image).resolve() for image in images}
    if not any(
        path.suffix.lower() == ".png"
        and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for path in figures & embedded
    ):
        raise ValueError("SPARQ did not embed an indexed PNG in its final report.")


async def execute_job(job: ProbeJob) -> RunResult:
    from sparq.architectures.v1.settings import V1Settings
    from sparq.architectures.v1.system import Agentic_system

    settings = V1Settings()  # pyright: ignore[reportCallIssue] -- populated from TOML
    roles = {
        role: getattr(settings.llm_config, role).model_copy(
            update={"provider": "aws_bedrock", "model_name": job.model_id}
        )
        for role in ("router", "planner", "executor", "aggregator")
    }
    run_dir = job.artifact_dir / "run"
    settings = settings.model_copy(
        update={
            "llm_config": settings.llm_config.model_copy(update=roles),
            "paths": settings.paths.model_copy(
                update={"output_dir": job.artifact_dir, "run_dir": run_dir}
            ),
        }
    )
    with collect_runs() as collector:
        try:
            result = await Agentic_system(settings=settings).run(job.query)
        finally:
            run_dir.mkdir(parents=True, exist_ok=True)
            evidence = collect_tool_evidence(collector.traced_runs)
            (run_dir / "tool_calls.json").write_text(
                json.dumps([call.model_dump() for call in evidence], indent=2)
            )
    if not result.response.strip():
        raise ValueError("SPARQ returned an empty final response.")
    verify_run(run_dir)
    return RunResult(
        status="passed",
        reason="SPARQ feature-coverage run completed with verified tool use and saved artifacts.",
    )


async def checked_job(job: ProbeJob) -> RunResult:
    try:
        if job.mode == "capabilities":
            from sparq.utils.bedrock.capabilities import (
                execute_job as execute_capabilities,
            )

            return await execute_capabilities(job)
        return await execute_job(job)
    except Exception as error:  # noqa: BLE001 -- collect per-model failures and continue the sweep
        traceback.print_exc()
        return failure_result(error)


def main() -> int:
    job = ProbeJob.model_validate_json(Path(sys.argv[1]).read_text())
    result = asyncio.run(checked_job(job))
    atomic_json(job.artifact_dir / "check.json", result)
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
