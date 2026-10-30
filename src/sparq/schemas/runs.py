"""Contracts for individual runs and their evaluation linkage."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal[
    "pending", "running", "completed", "failed", "cancelled", "interrupted"
]


class EvaluationContext(BaseModel):
    """Links one SPARQ run to its source batch-evaluation question."""

    batch_id: str
    evaluation_id: str | None = None
    question_id: int
    iteration: int


class SystemOutput(BaseModel):
    """Record of a single SPARQ run against an eval dataset question."""

    run_id: str = Field(..., description="Unique identifier for the run")
    query: str = Field(..., description="Question text, from the dataset")
    difficulty: int = Field(
        default_factory=int,
        description="Difficulty grade of the question, from the dataset",
    )
    ablation_config: dict = Field(
        {}, description="Ablation configuration used for this run"
    )
    response: str = Field(..., description="SPARQ's final answer")
    token_out: int | None = Field(
        None, description="Output tokens used by SPARQ, from SPARQ metadata"
    )
    models: dict[str, dict] = Field(
        ...,
        description="Model name and settings used per node, from the LLM config class",
    )
    cost: float | None = Field(
        None, description="Estimated cost of this run in USD, from SPARQ metadata"
    )
    time_started: datetime = Field(
        ..., description="When the run started, from the eval script"
    )
    time_ended: datetime = Field(
        ..., description="When the run ended, from the eval script"
    )
    duration: float = Field(
        ..., description="Run duration in seconds, from the eval script"
    )
    evaluation_context: EvaluationContext | None = Field(
        None, description="Batch-evaluation linkage for this run"
    )
    sparq_judge_score: dict | None = Field(
        None, description="Per-criterion scores from the SPARQ judge"
    )
    sparq_judge_review: str | None = Field(
        None, description="Free-text review from the SPARQ judge"
    )
