from typing import Annotated
from pathlib import Path

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from sparq.schemas.data_context import DataContext
from sparq.schemas.output_schemas import Plan, StepResult

id = int  # Type Alias


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []
    query: str
    route: bool | None = None
    answer: str | None = None
    plan: Plan | None = None
    completed_plan_steps: list[id] = Field([], description="The IDs of the steps of the plan that have been processed already.")
    results: list[StepResult] = []
    artifacts: list[Path] = []

    # data-specific
    data_context: DataContext | None = None
