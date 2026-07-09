from typing import Any, List, Dict, Annotated

from pydantic import BaseModel, Field
from sparq.schemas.output_schemas import Plan, Step, StepResult
from sparq.schemas.data_context import DataContext

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

id = int  # Type Alias


class State(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = []
    query: str
    route: bool | None = None
    answer: str | None = None
    plan: Plan | None = None
    completed_plan_steps: List[id] = Field([], description="The IDs of the steps of the plan that have been processed already.")
    results: List[StepResult] = []

    # data-specific
    data_context: DataContext | None = None
