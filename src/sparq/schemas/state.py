from pydantic import BaseModel
from sparq.schemas.output_schemas import Plan
from sparq.schemas.data_context import DataContext

class State(BaseModel):
    query: str
    route: bool | None = None
    answer: str | None = None
    plan: Plan | None = None

    # data-specific
    data_context: DataContext | None = None

    # executor-specific
    executor_results: dict = {}