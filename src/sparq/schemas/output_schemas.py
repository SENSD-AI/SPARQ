from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# Define desired output structure
class Step(BaseModel):
    """Information about a step"""
    id: int = Field(..., description="ID of step (step 1, step 2, ...)")
    step_description: str = Field(..., description="Description of the analytical step")
    datasets: list[str] = Field(..., description="List of dataset names used")
    rationale: str = Field(..., description="Why this step is necessary")
    task_type: list[str] = Field(..., description="The type of computation required e.g. data_retrieval, correlation, visualization")
    dependencies: list[int] = Field([], description="The list of steps this step depends on. If there are no dependencies, this should be empty.") 

class Plan(BaseModel):
    """Information about the the steps in a plan to answer the user query"""
    steps: list[Step]
    wants: str | None = Field(None, description="Further information you need to make a better plan")
    misc: str | None  = Field(None, description="Anything else you want the user to know or just a general scratchpad")

    def pretty_print(self):
        for field_name, value in self.model_dump().items():
            print(f"{field_name}:\n{value}")
        

class Router(BaseModel):
    """Output of the router node"""
    route: bool = Field(..., description="Whether the query requires further planning (True) or can be answered directly (False)")
    answer: str | None = Field(None, description="The answer to the query if it can be answered directly")
    

class StepResult(BaseModel):
    """The result of executing a single step"""
    id: int = Field(..., description="ID of the step (step 1, step 2, ...)")
    step: str = Field(..., description="What you were tasked to do by the user")
    success: bool = Field(..., description="Whether the step was successful or not")
    execution_results: str = Field("", description="Summary of results of running your code.")
    files_generated: list[str] = Field(default_factory=list, description="Files generated during execution")
    misc: str = Field("", description="Anything else you want to note, e.g. caveats, observations, justifications, rationales or next steps")


RunStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]
BatchStatus = Literal["running", "completed", "completed_with_errors", "failed", "interrupted"]


class EvaluationContext(BaseModel):
    """Links one SPARQ run to its source batch-evaluation question."""

    batch_id: str
    evaluation_id: str | None = None
    question_id: int
    iteration: int


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

class SystemOutput(BaseModel):
    """Record of a single SPARQ run against an eval dataset question."""

    run_id: str = Field(..., description="Unique identifier for the run")
    query: str = Field(..., description="Question text, from the dataset")
    difficulty: int = Field(default_factory=int, description="Difficulty grade of the question, from the dataset")
    ablation_config: dict = Field({}, description="Ablation configuration used for this run")
    response: str = Field(..., description="SPARQ's final answer")
    token_out: int | None = Field(None, description="Output tokens used by SPARQ, from SPARQ metadata")
    models: dict[str, dict] = Field(..., description="Model name and settings used per node, from the LLM config class")
    cost: float | None = Field(None, description="Estimated cost of this run in USD, from SPARQ metadata")
    time_started: datetime = Field(..., description="When the run started, from the eval script")
    time_ended: datetime = Field(..., description="When the run ended, from the eval script")
    duration: float = Field(..., description="Run duration in seconds, from the eval script")
    evaluation_context: EvaluationContext | None = Field(None, description="Batch-evaluation linkage for this run")
    sparq_judge_score: dict | None = Field(None, description="Per-criterion scores from the SPARQ judge")
    sparq_judge_review: str | None = Field(None, description="Free-text review from the SPARQ judge")
