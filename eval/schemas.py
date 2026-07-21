from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SystemOutput(BaseModel):
    """Record of a single SPARQ run against an eval dataset question."""

    run_id: str = Field(..., description="Unique identifier for the run")
    query: str = Field(..., description="Question text, from the dataset")
    difficulty: int = Field(..., description="Difficulty grade of the question, from the dataset")
    ablation_config: dict = Field({}, description="Ablation configuration used for this run")
    response: str = Field(..., description="SPARQ's final answer")
    token_out: Optional[int] = Field(None, description="Output tokens used by SPARQ, from SPARQ metadata")
    models: dict[str, str] = Field(..., description="Model name used per node, from the LLM config class")
    cost: Optional[float] = Field(None, description="Estimated cost of this run in USD, from SPARQ metadata")
    time_started: datetime = Field(..., description="When the run started, from the eval script")
    time_ended: datetime = Field(..., description="When the run ended, from the eval script")
    duration: float = Field(..., description="Run duration in seconds, from the eval script")
    sparq_judge_score: Optional[dict] = Field(None, description="Per-criterion scores from the SPARQ judge")
    sparq_judge_review: Optional[str] = Field(None, description="Free-text review from the SPARQ judge")