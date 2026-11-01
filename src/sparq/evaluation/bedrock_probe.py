"""Run one basic SPARQ query for the Bedrock model sweep."""

import asyncio
import json
import sys
import traceback
from pathlib import Path

from sparq.evaluation.bedrock_audit import (
    ProbeJob,
    RunResult,
    atomic_json,
    failure_result,
)


def verify_run(run_dir: Path) -> None:
    """Do not accept a returned answer when analytical steps actually failed."""
    trace = json.loads((run_dir / "trace.json").read_text())
    plan = trace.get("plan") or {}
    if trace.get("route") is not True or not plan.get("steps"):
        raise ValueError(
            "The basic query did not exercise the planner/executor/aggregator path."
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
    ):
        path = run_dir / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"SPARQ did not save a nonempty {name}.")


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
    result = await Agentic_system(settings=settings).run(job.query)
    if not result.response.strip():
        raise ValueError("SPARQ returned an empty final response.")
    verify_run(run_dir)
    return RunResult(
        status="passed",
        reason="Basic SPARQ run completed end to end with successful steps and saved reports.",
    )


async def checked_job(job: ProbeJob) -> RunResult:
    try:
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
