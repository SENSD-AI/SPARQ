from pydantic import BaseModel, Field


class Report(BaseModel):
    """New text for a report being written across multiple responses."""

    report: str = Field(
        ..., description="Only the new Markdown written this turn, without repeating earlier text."
    )
    done: bool = Field(
        ..., description="True only when the entire report is finished, including any references."
    )
