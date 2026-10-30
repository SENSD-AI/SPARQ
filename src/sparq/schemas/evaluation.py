"""Contracts for batch manifests and run lifecycle entries."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from sparq.schemas.runs import RunStatus

BatchStatus = Literal[
    "running", "completed", "completed_with_errors", "failed", "interrupted"
]


class BatchRunEntry(BaseModel):
    """Lifecycle and artifact location for one question iteration in a batch."""

    question_id: int
    question_index: int
    iteration: int
    run_id: str
    result_path: Path
    status: RunStatus = "pending"
    time_started: datetime | None = None
    time_ended: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None


class BatchEvalOutput(BaseModel):
    """Manifest for a complete batch-evaluation experiment."""

    batch_id: str
    evaluation_id: str | None = None
    status: BatchStatus = "running"
    time_started: datetime
    time_ended: datetime | None = None
    duration: float | None = None
    dataset_path: Path
    dataset_hash: str
    question_filter: dict
    requested_question_count: int
    selected_question_ids: list[int]
    iterations: int
    max_concurrent_runs: int
    code_version: dict
    configuration: dict
    planned_runs: int
    completed_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    interrupted_runs: int = 0
    runs: list[BatchRunEntry]
