from datetime import datetime
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from sparq.schemas.data_context import DataContext
from sparq.schemas.planning import Plan, StepResult

id = int  # Type Alias


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    query: str
    question_asked_at: datetime | None = None
    route: bool | None = None
    plan: Plan | None = None
    completed_plan_steps: list[id] = Field([], description="The IDs of the steps of the plan that have been processed already.")
    results: list[StepResult] = []
    artifacts: list[Path] = []

    # data-specific
    data_context: DataContext | None = None
