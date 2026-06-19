from typing import Any, List, Dict

from pydantic import BaseModel, Field
from sparq.schemas.output_schemas import Plan, Step
from sparq.schemas.data_context import DataContext

id = int  # Type Alias


class State(BaseModel):
    query: str
    route: bool | None = None
    answer: str | None = None
    plan: Plan | None = None
    completed_steps: List[id] = []
    results: Dict[id, Any] = {}

    # data-specific
    data_context: DataContext | None = None

    # TODO: Delete this when parallel execution is implemented
    # executor_results: dict = {}




class WorkerState(BaseModel):
    step: Step = Field(..., description="The current step being processed.")
    context: str | None = None
