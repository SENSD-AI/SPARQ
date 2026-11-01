"""Check SPARQ model capabilities on regional, text-capable on-demand Bedrock targets."""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any, Literal

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from sparq.utils.bedrock.paths import (
    audit_context_dir,
    make_output_id,
    model_directory_name,
    profile_cache_path,
)

CONSOLE = Console()
STATUS_STYLES = {
    "passed": "green",
    "failed": "red",
    "skipped": "yellow",
    "not_tested": "dim",
    "timed_out": "yellow",
    "access_denied": "yellow",
    "unsupported": "yellow",
    "error": "red",
}

CAPABILITY_QUERY = "Check structured Router/Plan output, a local tool round trip, and tools combined with structured StepResult output."

QUERY = """Perform a small, reproducible data-readiness study for Salmonella and
socioeconomic research using the configured, already cached datasets. This is a
full SPARQ workflow exercise, not a factual question. Do not download data or
invent observations. Keep computations to at most the first 200 rows per input.

Plan two independent initial steps: inspect one CSV dataset and one Excel (.xlsx)
dataset from the manifest. In each, call get_cached_dataset_path, then
find_csv_excel_files; for Excel also call get_sheet_names and choose an actual
sheet. Load each input using load_dataset. Use python_repl_tool with
persist_namespace=True to retain the loaded frames and computed summaries for
dependent steps. Use different variable names for the two datasets. Report row
counts, missingness, and descriptive statistics of one genuinely numeric column
per dataset; do not treat identifiers as measurements.

In subsequent steps depending on BOTH initial steps, reuse their persisted
variables to create feature_summary.csv with one row per dataset and columns
dataset, rows, numeric_column, non_null_count, mean. Save a labeled PNG chart
comparing missingness fractions (not unrelated measurement scales). Do not merge
unrelated observations or imply causation. Use write_todos to track this work.

Exercise filesystem tools: list_directory, file_search, write_file to save
methods.txt with source paths/sheets, copy_file to create methods_copy.txt, and
read_file to verify it. After the producing steps have finished and artifacts
have been indexed, use list_artifacts and search_artifact to locate the saved
table, methods, and plot. Read the table and methods and call analyze_image on
the saved PNG; verify its visible labels and trend against the computed data.
Save that visual assessment as visual_check.txt. Resolve returned artifact paths
rather than guessing paths after files have been organized.

The aggregator must inspect the saved artifacts and deliver a final report with
the summary table, embedded chart, methods, source provenance, visual assessment,
limitations, and numbered references identifying the actual datasets used.
Finish with SPARQ's normal saved JSON, Markdown, PDF and bibliography outputs.
Keep this a bounded descriptive audit; no large models or exhaustive analyses.
"""


class CheckResult(BaseModel):
    status: Literal[
        "not_tested",
        "passed",
        "failed",
        "skipped",
        "timed_out",
        "access_denied",
        "unsupported",
        "error",
    ] = "not_tested"
    reason: str = "Not tested."
    error_code: str | None = None
    error_chain: list[str] = Field(default_factory=list)
    # Saved reports use paths relative to the report directory.
    artifact_dir: str | None = None
    seconds: float | None = None


class RunResult(CheckResult):
    checks: dict[str, CheckResult] = Field(default_factory=dict)
    capability_hints: dict[str, bool | None] = Field(
        default_factory=lambda: {
            "tool_calling": None,
            "structured_output": None,
            "image_inputs": None,
        }
    )
    hints_source: str | None = None
    active_check: str | None = None


class ModelResult(BaseModel):
    model_id: str
    name: str
    kind: Literal["foundation", "inference_profile"]
    foundation_model_ids: list[str] = Field(default_factory=list)
    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None
    inference_types: list[str] | None = None
    inference_profile_ids: list[str] = Field(default_factory=list)
    inference_profile_id: str | None = None
    inference_profile_arn: str | None = None
    inference_profile_type: str | None = None
    inference_profile_status: str | None = None
    run: RunResult = Field(default_factory=RunResult)


class ProfileRoute(BaseModel):
    name: str
    invocation_id: str
    arn: str | None = None
    type: str | None = None
    status: str | None = None
    foundation_model_ids: list[str]


class InferenceProfileMap(BaseModel):
    source: str = "Bedrock list_inference_profiles joined with list_foundation_models"
    discovered_at: str
    profile: str
    region: str
    complete: bool
    discovery_errors: list[str]
    by_name: dict[str, list[str]] = Field(default_factory=dict)
    by_model_id: dict[str, list[str]] = Field(default_factory=dict)
    profiles: dict[str, ProfileRoute] = Field(default_factory=dict)


class AuditReport(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    profile: str
    region: str
    mode: Literal["capabilities", "pipeline"] = "capabilities"
    check_timeout: int = 120
    timeout: int = 600
    query: str = CAPABILITY_QUERY
    discovery_errors: list[str] = Field(default_factory=list)
    models: list[ModelResult] = Field(default_factory=list)
    inference_profiles: InferenceProfileMap | None = None
    profile_cache_path: Path | None = None
    cache_warning: str | None = None


class LatestAudit(BaseModel):
    audit_dir: str
    created_at: str
    mode: Literal["capabilities", "pipeline"]


class ProbeJob(BaseModel):
    model_id: str
    profile: str
    region: str
    query: str
    artifact_dir: Path
    mode: Literal["capabilities", "pipeline"] = "capabilities"
    check_timeout: int = 120


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
                inference_types=item.get("inferenceTypesSupported"),
            )
    except (BotoCoreError, ClientError) as error:
        errors.append("list_foundation_models: " + failure_result(error).reason)
    foundations = models.copy()
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
                profile = ModelResult(
                    model_id=identifier,
                    name=item.get("inferenceProfileName", identifier),
                    kind="inference_profile",
                    foundation_model_ids=bases,
                    inference_profile_id=item["inferenceProfileId"],
                    inference_profile_arn=item.get("inferenceProfileArn"),
                    inference_profile_type=item.get("type"),
                    inference_profile_status=item.get("status"),
                )
                # A profile is eligible only when every referenced base model qualifies.
                if bases and all(base_id in foundations for base_id in bases):
                    for field in (
                        "input_modalities",
                        "output_modalities",
                        "inference_types",
                    ):
                        values = [
                            getattr(foundations[base_id], field) for base_id in bases
                        ]
                        if all(value is not None for value in values):
                            common = set.intersection(*(set(value) for value in values))
                            if field == "inference_types":
                                pay_as_you_go = {"ON_DEMAND", "INFERENCE_PROFILE"}
                                if not common & pay_as_you_go and all(
                                    set(value) & pay_as_you_go for value in values
                                ):
                                    # Destinations may mix direct and profile-only access.
                                    common.add("INFERENCE_PROFILE")
                            setattr(profile, field, sorted(common))
                models[identifier] = profile
            token = response.get("nextToken")
            if not token:
                break
    except (BotoCoreError, ClientError) as error:
        errors.append("list_inference_profiles: " + failure_result(error).reason)
    for profile in models.values():
        if profile.kind == "inference_profile":
            for base_id in profile.foundation_model_ids:
                if base_id in foundations:
                    foundations[base_id].inference_profile_ids.append(
                        profile.inference_profile_id or profile.model_id
                    )
    for model in foundations.values():
        model.inference_profile_ids = sorted(set(model.inference_profile_ids))
    return sorted(models.values(), key=lambda model: model.model_id), errors


def build_profile_map(
    models: list[ModelResult], profile: str, region: str, errors: list[str]
) -> InferenceProfileMap:
    """Join profile names and foundation names/IDs without discarding alternative routes."""
    mapping = InferenceProfileMap(
        discovered_at=datetime.now(UTC).isoformat(),
        profile=profile,
        region=region,
        complete=not errors,
        discovery_errors=errors,
    )
    foundations = {
        model.model_id: model for model in models if model.kind == "foundation"
    }
    for model in models:
        if model.kind != "inference_profile":
            continue
        identifier = model.inference_profile_id or model.model_id.rsplit("/", 1)[-1]
        mapping.profiles[identifier] = ProfileRoute(
            name=model.name,
            invocation_id=model.model_id,
            arn=model.inference_profile_arn,
            type=model.inference_profile_type,
            status=model.inference_profile_status,
            foundation_model_ids=model.foundation_model_ids,
        )
        mapping.by_name.setdefault(model.name, []).append(identifier)
        for base_id in model.foundation_model_ids:
            mapping.by_model_id.setdefault(base_id, []).append(identifier)
            if base_id in foundations:
                mapping.by_name.setdefault(foundations[base_id].name, []).append(
                    identifier
                )
    for index in (mapping.by_name, mapping.by_model_id):
        for key, values in index.items():
            index[key] = sorted(set(values))
    return mapping


def select_models(
    models: list[ModelResult], selectors: list[str]
) -> tuple[list[ModelResult], list[str]]:
    """Resolve exact IDs/names; expand a profile-only foundation into its discovered routes."""
    by_id = {model.model_id: model for model in models}
    selected: set[str] = set()
    unknown = []
    for selector in selectors:
        matches = (
            [by_id[selector]]
            if selector in by_id
            else [
                model
                for model in models
                if model.name == selector or model.inference_profile_id == selector
            ]
        )
        if not matches:
            unknown.append(selector)
        for match in matches:
            if (
                match.kind == "foundation"
                and match.inference_profile_ids
                and "INFERENCE_PROFILE" in (match.inference_types or [])
                and "ON_DEMAND" not in (match.inference_types or [])
            ):
                selected.update(
                    model.model_id
                    for model in models
                    if model.kind == "inference_profile"
                    and model.inference_profile_id in match.inference_profile_ids
                )
            else:
                selected.add(match.model_id)
    return [model for model in models if model.model_id in selected], sorted(
        set(unknown)
    )


def exclusion_reason(model: ModelResult) -> str | None:
    """Require text and pay-as-you-go access through the target's invocation route."""
    for label, supported in (
        ("TEXT input", model.input_modalities),
        ("TEXT output", model.output_modalities),
    ):
        if supported is None:
            return f"Cannot verify required {label} from catalog metadata."
        if "TEXT" not in supported:
            return f"{label} required; catalog lists {supported}."
    if model.inference_types is None:
        return "Cannot verify required ON_DEMAND throughput from catalog metadata."
    if "ON_DEMAND" in model.inference_types:
        return None
    if "INFERENCE_PROFILE" in model.inference_types:
        if model.kind == "inference_profile":
            return None
        routes = ", ".join(model.inference_profile_ids)
        return (
            f"Routing alias: use inference-profile IDs {routes}; these routes are tested separately."
            if routes
            else "Direct invocation requires an inference profile; none was discovered."
        )
    return f"ON_DEMAND throughput required; catalog lists {model.inference_types}."


def atomic_json(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_report(report: AuditReport, output: Path) -> None:
    """Save the model list, failure reasons, and full evidence after every run."""
    atomic_json(output / "report.json", report)
    if report.inference_profiles is not None:
        atomic_json(output / "inference_profiles.json", report.inference_profiles)
    usable = [model.model_id for model in report.models if model.run.status == "passed"]
    lines = [
        "# Bedrock / SPARQ run results",
        "",
        f"Region: `{report.region}`; profile: `{report.profile}`.",
        "",
        f"Mode: `{report.mode}`. A pass verifies this mode only.",
        "",
        "## Models that passed",
        "",
    ]
    lines += [f"- `{identifier}`" for identifier in usable] or ["None verified yet."]
    if report.inference_profiles is not None:
        lines += [
            "",
            "[Model names and inference-profile IDs](inference_profiles.json). Discovery does not establish invocation permissions.",
        ]
    if report.profile_cache_path is not None:
        lines += ["", f"Reusable profile cache: `{report.profile_cache_path}`."]
    if report.cache_warning:
        lines += ["", f"Cache warning: {report.cache_warning}"]
    if report.discovery_errors:
        lines += ["", "## Incomplete discovery", ""] + [
            f"- {error}" for error in report.discovery_errors
        ]
    check_names = (
        (
            "structured_router",
            "structured_plan",
            "tool_round_trip",
            "tools_with_structure",
        )
        if report.mode == "capabilities"
        else ()
    )
    headings = ["Target", "Status", *check_names, "Reason"]
    lines += [
        "",
        "## All models and reasons",
        "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
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
                "checks",
                "capability_hints",
                "hints_source",
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
                    result.model_dump_json(include={"checks"}),
                    result.model_dump_json(include={"capability_hints"}),
                    result.hints_source,
                ]
            )
            reason = result.reason.replace("|", "\\|").replace("\n", " ")
            checks = [
                result.checks.get(name, CheckResult()).status for name in check_names
            ]
            lines.append(
                "| "
                + " | ".join([model.model_id, result.status, *checks, reason])
                + " |"
            )
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


def wait_for_worker(
    process: subprocess.Popen, timeout: int, cancel: Event | None
) -> bool:
    """Wait within the per-model deadline while allowing sweep-wide cancellation."""
    if cancel is None:
        process.wait(timeout=timeout)
        return True
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            return True
        if cancel.is_set():
            stop_worker(process)
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            process.wait(timeout=min(0.2, remaining))
            return True
        except subprocess.TimeoutExpired:
            continue


def run_probe(job: ProbeJob, timeout: int, cancel: Event | None = None) -> RunResult:
    """Give each target probe its own process, logs, and hard deadline."""
    if cancel is not None and cancel.is_set():
        return RunResult(reason="Interrupted before this model started.")
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
            [sys.executable, "-m", "sparq.utils.bedrock.probe", str(job_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=os.name == "posix",
        )
        try:
            finished = wait_for_worker(process, timeout, cancel)
        except subprocess.TimeoutExpired:
            stop_worker(process)
            try:
                result = RunResult.model_validate_json(
                    (job.artifact_dir / "check.json").read_text()
                )
            except (OSError, ValueError):
                result = RunResult()
            result.status = "timed_out"
            result.error_code = "RunTimeout"
            result.reason = f"Probe did not finish within {timeout}s; usability is inconclusive. Worker terminated."
            for name, check in result.checks.items():
                if check.status == "not_tested":
                    check.status = (
                        "timed_out" if name == result.active_check else "skipped"
                    )
                    check.reason = (
                        "The worker deadline ended; this check did not complete."
                    )
            result.active_check = None
        except KeyboardInterrupt:
            stop_worker(process)
            raise
        else:
            if not finished:
                return RunResult(
                    status="failed",
                    error_code="RunInterrupted",
                    reason="Interrupted before completion; worker terminated.",
                    artifact_dir=str(job.artifact_dir),
                    seconds=round(time.monotonic() - started, 3),
                )
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


def run_models(
    report: AuditReport, output: Path, timeout: int, workers: int = 1
) -> bool:
    """Bound concurrent subprocesses; checkpoint their results only in this thread."""
    if workers < 1:
        raise ValueError("workers must be positive")
    report.timeout = timeout
    eligible = []
    for model in report.models:
        if model.run.status != "not_tested":
            continue
        reason = exclusion_reason(model)
        if reason:
            model.run = RunResult(status="skipped", reason=reason)
        else:
            eligible.append(model)
    save_report(report, output)
    cancel = Event()
    futures: dict[Future[RunResult], tuple[ModelResult, ProbeJob]] = {}
    collected: set[Future[RunResult]] = set()
    interrupted = False
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bedrock-check")
    with Progress(
        SpinnerColumn(),
        TextColumn("SPARQ model checks"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("Checking", total=len(eligible))

        def collect(future: Future[RunResult]) -> bool:
            model, job = futures[future]
            was_interrupted = False
            try:
                result = future.result()
            except KeyboardInterrupt:
                was_interrupted = True
                result = RunResult(
                    status="failed",
                    error_code="RunInterrupted",
                    reason="Interrupted before completion.",
                )
            except Exception as error:  # noqa: BLE001 -- preserve a per-model launch failure
                result = failure_result(error)
            if result.status != "not_tested":
                result.artifact_dir = job.artifact_dir.relative_to(output).as_posix()
                progress.advance(task)
            model.run = result
            collected.add(future)
            save_report(report, output)
            progress.console.print(
                Text(
                    f"{result.status.upper()}  {model.model_id}: {result.reason}",
                    style=STATUS_STYLES[result.status],
                )
            )
            return was_interrupted

        try:
            for model in eligible:
                job = ProbeJob(
                    model_id=model.model_id,
                    profile=report.profile,
                    region=report.region,
                    query=report.query,
                    artifact_dir=output / "runs" / model_directory_name(model.model_id),
                    mode=report.mode,
                    check_timeout=report.check_timeout,
                )
                futures[pool.submit(run_probe, job, timeout, cancel)] = (model, job)
            for future in as_completed(futures):
                if collect(future):
                    interrupted = True
                    break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            cancel.set()
            for future in futures:
                future.cancel()
            # Running probes observe cancel and kill their subprocess groups.
            pool.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if future not in collected and not future.cancelled():
                    collect(future)
            save_report(report, output)
    return not interrupted


def show_summary(report: AuditReport, output: Path) -> None:
    table = Table(title="Bedrock / SPARQ results", header_style="bold cyan")
    for column in ("Model", "Status", "Duration", "Reason"):
        table.add_column(column)
    for model in report.models:
        result = model.run
        duration = f"{result.seconds:.1f}s" if result.seconds is not None else "—"
        table.add_row(
            Text(model.model_id),
            Text(result.status.upper(), style=STATUS_STYLES[result.status]),
            duration,
            Text(result.reason),
        )
    CONSOLE.print(table)
    counts = Counter(model.run.status for model in report.models)
    CONSOLE.print(
        Text(
            " · ".join(
                f"{count} {status.replace('_', ' ')}"
                for status, count in sorted(counts.items())
            ),
            style="bold",
        )
    )
    CONSOLE.print(Text(f"Reports: {output}"))


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
        "--cache-dir",
        type=Path,
        help="Override the OS user-cache Bedrock root; profile/region are appended",
    )
    parser.add_argument(
        "--mode",
        choices=("capabilities", "pipeline"),
        default="capabilities",
        help="Small capability probes (default), or the full SPARQ integration test",
    )
    parser.add_argument(
        "--check-timeout",
        type=positive_int,
        default=120,
        help="Seconds per capability check (default: 120)",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        help="Exact ID or model/profile name; profile-only foundation IDs expand to discovered routes; repeatable",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Discover and report without paid inference",
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=600,
        help="Seconds per model (default: 600)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="Parallel model runs (default: 4; use 1 for serial)",
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
    context_dir = audit_context_dir(profile, region)
    cache_path = profile_cache_path(profile, region, args.cache_dir)
    output = (
        (args.output or context_dir / "audits" / make_output_id(datetime.now(UTC)))
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
    with CONSOLE.status("Discovering regional Bedrock models..."):
        models, errors = discover_models(client)
    profile_map = build_profile_map(models, profile, region, errors)
    if args.model_id:
        models, unknown = select_models(models, args.model_id)
        if unknown and not errors:
            parser.error("Unknown model IDs: " + ", ".join(sorted(unknown)))
    report = AuditReport(
        profile=profile,
        region=region,
        models=models,
        inference_profiles=profile_map,
        profile_cache_path=cache_path,
        discovery_errors=errors,
        mode=args.mode,
        check_timeout=args.check_timeout,
        timeout=args.timeout,
        query=QUERY if args.mode == "pipeline" else CAPABILITY_QUERY,
    )
    for model in models:
        reason = exclusion_reason(model)
        if reason:
            model.run = RunResult(status="skipped", reason=reason)
    if profile_map.complete:
        try:
            atomic_json(cache_path, profile_map)
        except OSError as error:
            report.cache_warning = f"Could not refresh the reusable mapping: {error}. The audit snapshot is retained."
    else:
        report.cache_warning = (
            "Discovery was incomplete; the reusable mapping was left unchanged."
        )
    save_report(report, output)
    if args.output is None:
        atomic_json(
            context_dir / "latest_audit.json",
            LatestAudit(
                audit_dir=str(output.relative_to(context_dir.resolve())),
                created_at=report.created_at,
                mode=report.mode,
            ),
        )
    if report.cache_warning:
        CONSOLE.print(Text(report.cache_warning, style="yellow"))
    else:
        CONSOLE.print(Text(f"Reusable mapping: {cache_path}"))
    eligible = sum(model.run.status == "not_tested" for model in models)
    CONSOLE.print(
        Panel(
            Text(
                f"Region: {region} · Profile: {profile} · Mode: {report.mode}\n{eligible} eligible of {len(models)} discovered targets · {args.workers} workers"
            ),
            title="SPARQ Bedrock sweep",
            border_style="cyan",
        )
    )
    if not args.list_only and not run_models(
        report, output, args.timeout, workers=args.workers
    ):
        show_summary(report, output)
        return 130
    show_summary(report, output)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
