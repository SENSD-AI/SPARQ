from pydantic import BaseModel, Field
from typing import Dict, Any

class PythonREPLInput(BaseModel):
    code: str = Field(..., description="The Python code to execute.")
    persist_namespace: bool = Field(False, description="Whether to persist the namespace across executions.")
    
class ExceptionInfo(BaseModel):
    type: str = Field(..., description="The type of the exception.")
    message: str = Field(..., description="The exception message.")
    traceback: str = Field(..., description="The traceback of the exception.")
    extra_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context about the exception.")

class OutputSchema(BaseModel):
    output: str = Field(..., description="The standard output from executing the code.")
    error: ExceptionInfo | None = Field(None, description="Information about any exception that occurred during execution.")
    namespace: Dict[str, Any] = Field(..., description="The namespace after code execution.")
    success: bool = Field(..., description="Indicates whether the code executed successfully.")
