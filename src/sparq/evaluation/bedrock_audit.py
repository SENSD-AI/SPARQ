"""Try a basic SPARQ run with each Bedrock model in the configured region."""

import argparse
import csv
import hashlib
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field

QUERY = (
    "For a synthetic Salmonella example, use Python to calculate the sum and mean "
    "of the fictional values 2, 4, 6, then report the results. Make a plan and execute "
    "it; do not load real datasets."
)


class RunResult(BaseModel):
    status: Literal["not_tested", "passed", "failed", "skipped"] = "not_tested"
    reason: str = "Not tested."
    error_code: str | None = None
    error_chain: list[str] = Field(default_factory=list)
    artifact_dir: str | None = None
    seconds: float | None = None


class ModelResult(BaseModel):
    model_id: str
    name: str
    kind: Literal["foundation", "inference_profile"]
    foundation_model_ids: list[str] = Field(default_factory=list)
    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None
    run: RunResult = Field(default_factory=RunResult)


class AuditReport(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    profile: str
    region: str
    query: str = QUERY
    discovery_errors: list[str] = Field(default_factory=list)
    models: list[ModelResult] = Field(default_factory=list)


class ProbeJob(BaseModel):
    model_id: str
    profile: str
    region: str
    query: str
    artifact_dir: Path


def failure_result(error: Exception) -> RunResult:
    """Retain original AWS error codes even when SPARQ wraps them in ValueError."""
    chain: list[str] = []
    codes: list[str] = []
    aws_code: str | None = None
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = type(current).__name__
        if isinstance(current, ClientError):
            code = current.response.get("Error", {}).get("Code", code)
            aws_code = aws_code or code
        codes.append(code)
        notes = " ".join(getattr(current, "__notes__", []))
        chain.append(f"{code}: {current} {notes}".strip())
        current = current.__cause__ or current.__context__
    return RunResult(
        status="failed",
        reason=chain[-1],
        error_code=aws_code or codes[-1],
        error_chain=chain,
    )


def discover_models(client: Any) -> tuple[list[ModelResult], list[str]]:
    """Include the unfiltered foundation catalog and every inference-profile page."""
    models: dict[str, ModelResult] = {}
    errors: list[str] = []
    try:
        for item in client.list_foundation_models()["modelSummaries"]:
            identifier = item["modelId"]
            models[identifier] = ModelResult(
                model_id=identifier,
                name=item.get("modelName", identifier),
                kind="foundation",
                foundation_model_ids=[identifier],
                input_modalities=item.get("inputModalities"),
                output_modalities=item.get("outputModalities"),
            )
    except (BotoCoreError, ClientError) as error:
        errors.append("list_foundation_models: " + failure_result(error).reason)
    try:
        token = None
        while True:
            response = client.list_inference_profiles(
                **({"nextToken": token} if token else {})
            )
            for item in response["inferenceProfileSummaries"]:
                identifier = item["inferenceProfileId"]
                if item.get("type") == "APPLICATION":
                    identifier = item["inferenceProfileArn"]
                bases = sorted(
                    {
                        entry["modelArn"].split("foundation-model/")[-1]
                        for entry in item.get("models", [])
                    }
                )
                models[identifier] = ModelResult(
                    model_id=identifier,
                    name=item.get("inferenceProfileName", identifier),
                    kind="inference_profile",
                    foundation_model_ids=bases,
                )
            token = response.get("nextToken")
            if not token:
                break
    except (BotoCoreError, ClientError) as error:
        errors.append("list_inference_profiles: " + failure_result(error).reason)
    return sorted(models.values(), key=lambda model: model.model_id), errors


def exclusion_reason(model: ModelResult) -> str | None:
    if model.input_modalities is not None and "TEXT" not in model.input_modalities:
        return f"SPARQ needs TEXT input; catalog lists {model.input_modalities}."
    if model.output_modalities is not None and "TEXT" not in model.output_modalities:
        return f"SPARQ needs TEXT output; catalog lists {model.output_modalities}."
    return None


def atomic_json(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def save_report(report: AuditReport, output: Path) -> None:
    """Save the model list, failure reasons, and full evidence after every run."""
    atomic_json(output / "report.json", report)
    usable = [model.model_id for model in report.models if model.run.status == "passed"]
    lines = [
        "# Bedrock / SPARQ run results",
        "",
        f"Region: `{report.region}`; profile: `{report.profile}`.",
        "",
        "## Models that completed the basic run",
        "",
    ]
    lines += [f"- `{identifier}`" for identifier in usable] or ["None verified yet."]
    if report.discovery_errors:
        lines += ["", "## Incomplete discovery", ""] + [
            f"- {error}" for error in report.discovery_errors
        ]
    lines += [
        "",
        "## All models and reasons",
        "",
        "| Target | Status | Reason |",
        "| --- | --- | --- |",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model_id",
                "kind",
                "status",
                "error_code",
                "reason",
                "artifact_dir",
                "seconds",
            ]
        )
        for model in report.models:
            result = model.run
            writer.writerow(
                [
                    model.model_id,
                    model.kind,
                    result.status,
                    result.error_code,
                    result.reason,
                    result.artifact_dir,
                    result.seconds,
                ]
            )
            reason = result.reason.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {model.model_id} | {result.status} | {reason} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stop_worker(process: subprocess.Popen) -> None:
    """Stop a timed-out worker and its REPL children on POSIX."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def run_probe(job: ProbeJob, timeout: int) -> RunResult:
    """Give each real SPARQ run its own process, logs, and hard deadline."""
    job.artifact_dir.mkdir(parents=True, exist_ok=True)
    job_path = job.artifact_dir / "job.json"
    atomic_json(job_path, job)
    started = time.monotonic()
    environment = {
        **os.environ,
        "AWS_PROFILE": job.profile,
        "AWS_REGION": job.region,
        "PYTHONUNBUFFERED": "1",
    }
    with (job.artifact_dir / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "sparq.evaluation.bedrock_probe", str(job_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=os.name == "posix",
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_worker(process)
            result = RunResult(
                status="failed",
                error_code="RunTimeout",
                reason=f"SPARQ did not finish within {timeout}s; worker terminated.",
            )
        except KeyboardInterrupt:
            stop_worker(process)
            raise
        else:
            try:
                result = RunResult.model_validate_json(
                    (job.artifact_dir / "check.json").read_text()
                )
                if process.returncode and result.status == "passed":
                    raise ValueError(
                        "Worker reported success but exited with an error."
                    )
            except (OSError, ValueError) as error:
                result = RunResult(
                    status="failed",
                    error_code="WorkerFailure",
                    reason=f"Worker exited {process.returncode} without a valid result: {error}. See worker.log.",
                )
    result.seconds = round(time.monotonic() - started, 3)
    result.artifact_dir = str(job.artifact_dir)
    return result


def run_models(report: AuditReport, output: Path, timeout: int) -> bool:
    active: ModelResult | None = None
    active_dir: Path | None = None
    try:
        for index, model in enumerate(report.models, 1):
            if model.run.status != "not_tested":
                continue
            print(f"[{index}/{len(report.models)}] {model.model_id}", flush=True)
            digest = hashlib.sha256(model.model_id.encode()).hexdigest()[:12]
            job = ProbeJob(
                model_id=model.model_id,
                profile=report.profile,
                region=report.region,
                query=report.query,
                artifact_dir=output / "runs" / digest,
            )
            active, active_dir = model, job.artifact_dir
            model.run = run_probe(job, timeout)
            save_report(report, output)
            active = None
            print(f"  {model.run.status}: {model.run.reason}", flush=True)
    except KeyboardInterrupt:
        if active is not None and active.run.status == "not_tested":
            active.run = RunResult(
                status="failed",
                error_code="RunInterrupted",
                reason="Interrupted before completion; no successful run was established.",
                artifact_dir=str(active_dir),
            )
        save_report(report, output)
        return False
    return True


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Defaults to AWS_PROFILE or default")
    parser.add_argument("--region", help="Defaults to AWS_REGION or the profile region")
    parser.add_argument("--output", type=Path, help="Results directory")
    parser.add_argument(
        "--model-id", action="append", help="Test only this discovered ID; repeatable"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Discover and report without paid inference",
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=180,
        help="Seconds per model (default: 180)",
    )
    args = parser.parse_args(argv)
    from sparq.settings import ENVSettings

    ENVSettings()
    profile = args.profile or os.getenv("AWS_PROFILE") or "default"
    session = boto3.Session(
        profile_name=profile, region_name=args.region or os.getenv("AWS_REGION")
    )
    region = args.region or os.getenv("AWS_REGION") or session.region_name
    if not region:
        parser.error("Configure AWS_REGION, a profile region, or --region.")
    output = (
        (
            args.output
            or Path("benchmarks/results/bedrock_models")
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        )
        .expanduser()
        .resolve()
    )
    if (output / "report.json").exists():
        parser.error("Report exists; choose a new --output directory.")
    client = session.client(
        "bedrock",
        region_name=region,
        config=Config(
            connect_timeout=10,
            read_timeout=30,
            retries={"mode": "standard", "total_max_attempts": 2},
        ),
    )
    models, errors = discover_models(client)
    if args.model_id:
        unknown = set(args.model_id) - {model.model_id for model in models}
        if unknown and not errors:
            parser.error("Unknown model IDs: " + ", ".join(sorted(unknown)))
        models = [model for model in models if model.model_id in args.model_id]
    report = AuditReport(
        profile=profile, region=region, models=models, discovery_errors=errors
    )
    for model in models:
        reason = exclusion_reason(model)
        if reason:
            model.run = RunResult(status="skipped", reason=reason)
    save_report(report, output)
    print(f"Discovered {len(models)} targets in {region}. Reports: {output}")
    if not args.list_only and not run_models(report, output, args.timeout):
        return 130
    usable = [model.model_id for model in models if model.run.status == "passed"]
    print(f"Models that completed SPARQ ({len(usable)}): " + ", ".join(usable))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
