"""Run repeatable, bounded batch evaluations over the question dataset.

The script selects eligible questions from ``data/Q_dataset.json``, executes
each question for the requested number of iterations, and writes a manifest
and per-run artifacts under ``eval/results/batch_eval``. Individual run
failures are recorded in the manifest so the remaining runs can continue;
unexpected orchestration failures cancel outstanding work before the batch is
finalized.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import uuid
from collections import Counter
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import TypedDict, cast

from sparq.architectures.v1.settings import V1Settings
from sparq.architectures.v1.system import Agentic_system
from sparq.schemas.output_schemas import (
    BatchEvalOutput,
    BatchRunEntry,
    EvaluationContext,
)
from sparq.settings import ENVSettings
from sparq.utils.get_package_dir import get_project_root


MAX_CONCURRENT_RUNS = 3
# Resolve paths once so manifests remain meaningful regardless of cwd.
_project_root = get_project_root()
if _project_root is None:
    raise RuntimeError("Could not locate project root")
PROJECT_ROOT: Path = _project_root
FILE_PATH = PROJECT_ROOT / "data" / "Q_dataset.json"
RESULTS_ROOT = Path(__file__).parent / "results" / "batch_eval"


def positive_int(value: str) -> int:
    """Parse a positive integer for an argparse option.

    Args:
        value: Text supplied by the command line.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not at least one.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


class Question(TypedDict):
    id: int
    text: str
    grade: int
    weather_related: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line options for question count and iterations.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run batch evaluation on SPARQ over test questions in data/Q_dataset",
    )
    parser.add_argument("-n", "--n_questions", type=positive_int, default=1)
    parser.add_argument("-k", "--iterations", type=positive_int, default=1)
    parser.add_argument(
        "--evaluation-id",
        help="Shared identifier grouping batches from separate invocations into one evaluation",
    )
    return parser.parse_args()


def load_questions(file_path: Path, limit: int) -> list[Question]:
    """Load, validate, filter, and limit questions from a dataset file.

    Args:
        file_path: JSON dataset containing a top-level ``questions`` list.
        limit: Maximum number of eligible questions to return.

    Returns:
        Valid, non-weather-related questions whose grade is not ``-1``.

    Raises:
        ValueError: If the dataset is malformed, contains duplicate IDs, or
            has no questions matching the filters.
    """
    with open(file_path) as file:
        data = json.load(file)
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Expected a nonempty questions list in {file_path}")

    # Validate the complete source list before applying the selection limit;
    # otherwise malformed or duplicate questions beyond the limit go unnoticed.
    seen_ids: set[int] = set()
    for index, question in enumerate(questions):
        fields = {"id": int, "text": str, "grade": int, "weather_related": bool}
        if not isinstance(question, dict) or any(
            type(question.get(key)) is not expected for key, expected in fields.items()
        ):
            raise ValueError(f"Invalid question at index {index} in {file_path}")
        if question["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {question['id']}")
        seen_ids.add(question["id"])

    selected = [
        question for question in questions
        if not question["weather_related"] and question["grade"] != -1
    ][:limit]
    if not selected:
        raise ValueError("No questions matched the batch filters")
    return cast(list[Question], selected)


def local_now() -> datetime:
    """Return the current local time as a timezone-aware datetime."""
    return datetime.now().astimezone()


def make_batch_id(started_at: datetime) -> str:
    """Create a readable, unique identifier for a batch.

    Args:
        started_at: Batch start time used as the human-readable prefix.

    Returns:
        A timestamped identifier with a random suffix.
    """
    timestamp = started_at.strftime("%Y-%m-%dT%H-%M-%S-%Z")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def hash_file(file_path: Path) -> str:
    """Return a file's SHA-256 digest in ``algorithm:digest`` form.

    Args:
        file_path: File to read.

    Returns:
        The SHA-256 digest prefixed with ``sha256:``.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def get_code_version(project_root: Path) -> dict:
    """Collect Git metadata for reproducibility.

    Args:
        project_root: Git working tree in which to run the commands.

    Returns:
        A mapping containing the ``commit`` hash and ``dirty`` flag. Both
        values are ``None`` when Git metadata cannot be read.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}

    return {"commit": commit, "dirty": dirty}


def update_batch_summary(batch: BatchEvalOutput, active: bool = True) -> None:
    """Recalculate run counts and derive the batch's current status.

    Args:
        batch: Batch manifest to update in place.
        active: Whether unfinished runs represent active work. Set to
            ``False`` when finalizing after interruption.
    """
    counts = Counter(entry.status for entry in batch.runs)
    batch.completed_runs = counts["completed"]
    batch.failed_runs = counts["failed"]
    batch.cancelled_runs = counts["cancelled"]
    batch.interrupted_runs = counts["interrupted"]

    # A cancelled/interrupted run means the batch was stopped before every
    # planned run reached a terminal state, even if no task is running now.
    if not active and any(counts[state] for state in ("pending", "running", "cancelled", "interrupted")):
        batch.status = "interrupted"
    elif counts["pending"] or counts["running"]:
        batch.status = "running"
    elif batch.completed_runs == batch.planned_runs:
        batch.status = "completed"
    elif batch.completed_runs:
        batch.status = "completed_with_errors"
    else:
        batch.status = "failed"


def write_json_atomic(path: Path, value: object) -> None:
    """Write JSON through a temporary file and atomically replace ``path``.

    Args:
        path: Destination file. Its parent directory must already exist.
        value: JSON-serializable value to write.

    Raises:
        OSError: If the temporary file cannot be written or replaced.
        TypeError: If ``value`` is not JSON serializable.
    """
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        # Replacing a completed temporary file prevents readers from seeing a
        # partially written manifest if the process is interrupted mid-write.
        with open(temporary_path, "w") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def persist_manifest(batch: BatchEvalOutput, manifest_path: Path, active: bool = True) -> None:
    """Update summary fields and atomically persist a batch manifest.

    Args:
        batch: Batch manifest to summarize and write.
        manifest_path: Destination path for the manifest JSON file.
        active: Whether unfinished runs should keep the status as ``running``.
    """
    # Synchronous writes cannot interleave between tasks on this event loop.
    update_batch_summary(batch, active=active)
    write_json_atomic(manifest_path, batch.model_dump(mode="json"))


async def execute_run(
    semaphore: asyncio.Semaphore,
    settings: V1Settings,
    question: Question,
    run_entry: BatchRunEntry,
    batch: BatchEvalOutput,
    manifest_path: Path,
) -> None:
    """Run one planned evaluation and record its lifecycle.

    Args:
        semaphore: Concurrency limiter shared by the batch.
        settings: Immutable baseline settings copied for this run.
        question: Dataset question to evaluate.
        run_entry: Manifest entry updated in place.
        batch: Parent batch manifest.
        manifest_path: Path used to persist lifecycle updates.

    Raises:
        asyncio.CancelledError: If the task is cancelled while running.
    """
    try:
        async with semaphore:
            run_entry.status = "running"
            run_entry.time_started = local_now()
            persist_manifest(batch, manifest_path)

            try:
                # Each run gets its own directory and a deep copy of settings,
                # preventing concurrent runs from sharing mutable path state.
                run_dir = manifest_path.parent / run_entry.result_path.parent
                run_dir.mkdir(parents=True, exist_ok=False)
                run_settings = settings.model_copy(deep=True)
                run_settings.paths.run_dir = run_dir
                agentic_system = Agentic_system(settings=run_settings)
                await agentic_system.run(
                    question["text"],
                    run_id=run_entry.run_id,
                    difficulty=question["grade"],
                    evaluation_context=EvaluationContext(
                        batch_id=batch.batch_id,
                        evaluation_id=batch.evaluation_id,
                        question_id=question["id"],
                        iteration=run_entry.iteration,
                    ),
                )
            except Exception as error:
                run_entry.status = "failed"
                run_entry.error_type = type(error).__name__
                run_entry.error_message = str(error)
            else:
                run_entry.status = "completed"
            run_entry.time_ended = local_now()
            persist_manifest(batch, manifest_path)
    except asyncio.CancelledError:
        run_entry.status = "cancelled"
        run_entry.time_ended = local_now()
        # The batch owner persists cancellation after all tasks have stopped.
        raise


async def execute_batch(
    batch: BatchEvalOutput,
    batch_dir: Path,
    questions: list[Question],
    settings: V1Settings,
) -> None:
    """Execute all planned runs while enforcing concurrency and cleanup.

    The batch directory is created first, then the selected questions and an
    initial manifest are persisted. Planned runs are processed in iteration
    order. ``batch.runs`` is grouped by its consecutive ``iteration`` values,
    and one asyncio task is created for each run in the current group. Each
    task uses the shared semaphore, so no more than
    ``batch.max_concurrent_runs`` evaluations execute at once. The function
    waits for every task in an iteration to finish before starting the next
    iteration.

    Individual evaluation errors are handled by ``execute_run`` and recorded
    on their run entries, allowing the remaining runs to continue. If the
    batch itself is cancelled or an orchestration error occurs, outstanding
    tasks are cancelled and awaited before unfinished entries are marked as
    cancelled or interrupted. The final manifest always receives an end time,
    duration, and summary status in the ``finally`` block.

    Args:
        batch: Batch plan and manifest to execute.
        batch_dir: Directory for the batch manifest and run artifacts.
        questions: Validated questions referenced by ``batch.runs``.
        settings: Baseline settings copied for each run.

    Raises:
        BaseException: Propagates cancellation or orchestration errors after
            outstanding tasks have been cancelled and awaited. Errors from an
            individual evaluation do not propagate; they are recorded in the
            corresponding run entry.
    """
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = batch_dir / "batch.json"
    questions_by_id = {question["id"]: question for question in questions}
    semaphore = asyncio.Semaphore(batch.max_concurrent_runs)
    tasks: list[asyncio.Task[None]] = []
    try:
        write_json_atomic(batch_dir / "questions.json", {"questions": questions})
        persist_manifest(batch, manifest_path)
        print(f"Batch {batch.batch_id}: {batch.planned_runs} runs")
        if batch.evaluation_id is not None:
            print(f"Evaluation: {batch.evaluation_id}")
        print(f"Manifest: {manifest_path}")
        for _, entries in groupby(batch.runs, key=lambda entry: entry.iteration):
            # Runs within an iteration may overlap; gather forms the barrier
            # that keeps the next iteration from starting early.
            tasks = []
            for entry in entries:
                tasks.append(asyncio.create_task(execute_run(
                    semaphore, settings, questions_by_id[entry.question_id],
                    entry, batch, manifest_path,
                )))
            await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for entry in batch.runs:
            if entry.status in ("pending", "running"):
                entry.status = "cancelled" if entry.status == "pending" else "interrupted"
                entry.time_ended = local_now()
        raise
    finally:
        batch.time_ended = local_now()
        batch.duration = (batch.time_ended - batch.time_started).total_seconds()
        persist_manifest(batch, manifest_path, active=False)


def create_batch(
    questions: list[Question],
    requested_question_count: int,
    iterations: int,
    configuration: dict,
    evaluation_id: str | None = None,
) -> tuple[BatchEvalOutput, Path]:
    """Build a batch manifest and its planned run entries.

    Args:
        questions: Validated questions selected for evaluation.
        requested_question_count: Count requested by the caller, retained in
            the manifest even when fewer questions are eligible.
        iterations: Number of evaluations to plan per question.
        configuration: Model configuration snapshot stored in the manifest.
        evaluation_id: Shared identifier for batches belonging to one evaluation.

    Returns:
        The initialized batch manifest and its output directory path.
    """
    started_at = local_now()
    batch_id = make_batch_id(started_at)
    batch_dir = RESULTS_ROOT / batch_id
    runs: list[BatchRunEntry] = []

    for iteration in range(1, iterations + 1):
        # Keep this order aligned with execute_batch(), which groups adjacent
        # entries by iteration before scheduling them.
        for question_index, question in enumerate(questions):
            run_id = str(uuid.uuid4())
            relative_run_dir = (
                Path("runs")
                / f"iteration_{iteration:03d}"
                / f"question_{question['id']:03d}"
                / run_id
            )
            runs.append(
                BatchRunEntry(
                    question_id=question["id"],
                    question_index=question_index,
                    iteration=iteration,
                    run_id=run_id,
                    result_path=relative_run_dir / "result.json",
                )
            )

    batch = BatchEvalOutput(
        batch_id=batch_id,
        evaluation_id=evaluation_id,
        time_started=started_at,
        dataset_path=FILE_PATH.relative_to(PROJECT_ROOT),
        dataset_hash=hash_file(FILE_PATH),
        question_filter={"weather_related": False, "excluded_grade": -1},
        requested_question_count=requested_question_count,
        selected_question_ids=[question["id"] for question in questions],
        iterations=iterations,
        max_concurrent_runs=MAX_CONCURRENT_RUNS,
        code_version=get_code_version(PROJECT_ROOT),
        configuration=configuration,
        planned_runs=len(runs),
        runs=runs,
    )
    return batch, batch_dir


async def main() -> int:
    """Parse arguments, execute a batch, and return its process exit status.

    Returns:
        ``0`` for a fully successful batch; ``1`` for a batch with failures,
        cancellations, or interruptions.
    """
    args = parse_args()
    questions = load_questions(FILE_PATH, limit=args.n_questions)
    ENVSettings()
    settings = V1Settings()  # pyright: ignore[reportCallIssue] -- fields come from TOML
    batch, batch_dir = create_batch(
        questions=questions,
        requested_question_count=args.n_questions,
        iterations=args.iterations,
        configuration={"models": settings.llm_config.model_dump(mode="json")},
        evaluation_id=args.evaluation_id,
    )
    await execute_batch(batch, batch_dir, questions, settings)
    print(
        f"Batch {batch.status}: {batch.completed_runs} completed, "
        f"{batch.failed_runs} failed, {batch.cancelled_runs} cancelled"
    )
    return 0 if batch.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
